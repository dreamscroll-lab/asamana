"""The actor, on the engine's side: a character's secondary cognition, asked through a channel
to the person performing it (``agent.actor.Actor``).

The character asks its actor at the stages the actor owns, and the step suspends until the
person answers:
- interrupt: whether to break off the action in progress (``evaluate_interrupt``);
- decision: what to do (``decide``);
- arbitration: whether to join an act it is invited into (``consider_invitation``);
- a live conversation: its dialogue turn, or the end of the conversation (``speak``).
The actor answers an open turn through ``answer``. While a DECIDE turn is open their words are
read into a decision to confirm; explicit operations can submit directly (``intent``).
Outside a turn the actor may only ask to break off the act in progress (``break_off``), which
the next interrupt stage takes without asking.

Everything that passes between actor and world is recorded as the actor's thread
(``core.interfaces.actor.ActorEntry``), each entry stamped with its step's world time, told to ``on_entry`` as it is recorded and
handed to the step's snapshot when the step commits (``commit_thread``).

The beats the actor settles without an LLM call (the decision, the interrupt answer, the mind
amendments) also go into the trace as ``BeatTrace``s, so a review reads the character's step
whole; those beats and the intent-reading calls are tagged ``actor``.

The actor also amends the character's mind (``mind``, ``amend``). Amendments land at once while a
DECIDE or INTERRUPT turn is open, since the step waits on the actor and nothing else writes the
character; otherwise, an INVITE or SPEAK turn included, they wait for the start of the next step, where the
character lands them (``land_amendments``).

An idle character decides at the actor's pace (``ActorPace``). It is asked about breaking off an
action whenever anything reached it (``engine.signals``, the same signals that put an AI character
before the world-pressure judgment). Nothing here judges what matters to the actor; every answer
is the actor's.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import TYPE_CHECKING, Any, Callable, Sequence

from agent.personality import ActionStatus
from agent.actor_operations import (
    actor_decision_beat, actor_operations, build_actor_decision_summary, build_actor_operation,
    requires_actor_interpretation,
)
from core.context import (
    annotate_call, get_active_call, note_active_call_adoption, note_call_adoption, observe_actor,
)
from core.interfaces.action import DialogueTurn, Ref
from core.interfaces.trace import SUPERSEDED, BeatKind, Stage, TraceSink, record_beat
from core.prompts import situation_location
from core.interfaces.actor import (
    UNREADABLE,
    ActorAction,
    ActorAmendments,
    ActorAnswer,
    ActorAnswerRejected,
    ActorDecision,
    ActorConfirmation,
    ActorIntent,
    ActorIntentResult,
    ActorOperations,
    ActorDecisionSummary,
    ActorPublicAnswer,
    ActorIdleAnswer,
    ActorEntry,
    ActorInterruptAnswer,
    ActorInvitationAnswer,
    ActorDialogueTurn,
    ActorMind,
    ActorPace,
    ActorTurn,
    AnsweredEntry,
    Invitation,
    AskedEntry,
    BeganEntry,
    BreakOffEntry,
    Conversation,
    DialogueTurnEntry,
    EndedEntry,
    ProgressedEntry,
    SaidEntry,
    TurnKind,
)
from core.logging import get_logger
from core.serialization import dump_json
from engine.actor_channel import ActorChannel, TurnListener
from engine.clock import WorldTime, WorldTimeConfig

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.perception import PerceptionPacket
    from core.interfaces.message import Message
    from core.interfaces.perception import Broadcast, PerceivedSignals

logger = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class _PendingDecision:
    """A decision read from the actor's intent, waiting for the actor to confirm it."""

    summary: ActorDecisionSummary
    decision: ActorDecision
    # The intent-reading call it came from; None for a direct operation.
    reading: Any = None


@dataclasses.dataclass
class _DecideTurn:
    """What the coordinator holds while one DECIDE turn is open: the character and the
    perception intents are read against, every intent-reading call made in the turn, and the
    decision awaiting confirmation. Created when the turn is asked, dropped when it closes."""

    agent: "Agent"
    packet: "PerceptionPacket"
    readings: list[Any] = dataclasses.field(default_factory=list)
    pending: _PendingDecision | None = None
    confirmed: _PendingDecision | None = None

# Told each entry of the actor's thread, as JSON, as it is recorded.
EntryListener = Callable[[dict[str, Any]], None]


def _reason(signals: "PerceivedSignals", *, falling_due: bool = False) -> str:
    """Why the character is asked, for the actor: how many things reached it, and whether an
    agreed time is near. What they were is in the thread's perceptions, on the map and in the
    goals."""
    parts = [f"有 {len(signals)} 件事发生"] if signals else []
    if falling_due:
        parts.append("约定好时间的事项快到了")
    return "，".join(parts)


class ActorCoordinator:
    def __init__(
        self,
        time_config: WorldTimeConfig,
        *,
        on_turn: TurnListener | None = None,
        on_entry: EntryListener | None = None,
        world_id: str = "",
        trace_sink: TraceSink | None = None,
    ) -> None:
        self._time_config = time_config
        self._world_id = world_id
        self._trace_sink = trace_sink
        self._on_turn = on_turn
        # Counts intents across turns, so a reading that returns after a newer intent, or after
        # its turn closed, is dropped as stale.
        self._intent_generation = 0
        self._channel = ActorChannel(self._turn_changed)
        self._on_entry = on_entry
        self._agent: "Agent | None" = None
        self._pace = ActorPace.EVERY_STEP
        self._deciding: _DecideTurn | None = None
        # The thread since the last commit, as JSON.
        self._thread: list[dict[str, Any]] = []
        # The latest step the character perceived: the step an entry outside any stage belongs to.
        self._step = 0
        # The step the character's signals were last weighed in, and the turn's reason from them.
        self._weighed: tuple[int, str] = (0, "")
        # An act of the character's ended and it hasn't decided since: at either pace, it is asked.
        self._freed = False
        # Amendments saved while no turn was open, landing at the start of the next step.
        self._pending_amendments: ActorAmendments | None = None
        # Held while an amendment is decided and lands. It spans awaits (an insight is embedded and
        # stored), and the turn it lands in may be answered meanwhile: the step resumed by that
        # answer waits on this lock, so it never runs alongside the amendment and overwrites it.
        self._amending = asyncio.Lock()

    def _turn_changed(self, turn: ActorTurn | None) -> None:
        self._intent_generation += 1
        if self._deciding is not None:
            self._deciding.pending = None
        if self._on_turn is not None:
            self._on_turn(turn)

    @property
    def channel(self) -> ActorChannel:
        return self._channel

    def perform(self, agent: "Agent", *, pace: ActorPace) -> None:
        """Take ``agent`` over: from now on the character asks this actor."""
        if self._channel.agent_id not in (None, agent.agent_id):
            raise ValueError(f"The actor already performs {self._channel.agent_id!r}")
        self._channel.follow(agent.agent_id)
        agent.actor = self
        self._agent = agent
        self._pace = pace

    def set_pace(self, pace: ActorPace) -> None:
        self._pace = pace

    # ----- the thread --------------------------------------------------------

    def _record(self, entry: ActorEntry) -> None:
        entry = dataclasses.replace(
            entry, time_label=WorldTime.from_step(entry.step, self._time_config).time_label,
        )
        data = json.loads(dump_json(dataclasses.asdict(entry)))
        self._thread.append(data)
        if self._on_entry is None:
            return
        # A listener is an observer: its failure must not fail the step.
        try:
            self._on_entry(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("actor_entry_announce_failed", extra={"kind": entry.kind.value, "error": str(exc)})

    @property
    def live_thread(self) -> list[dict[str, Any]]:
        """The thread recorded since the last commit, which no snapshot holds yet."""
        return list(self._thread)

    def commit_thread(self, count: int) -> None:
        """Drop the first ``count`` entries, which a saved snapshot now holds. Not cleared when the
        snapshot is built: a save that fails leaves them to the next step's snapshot, and entries
        recorded while the save was under way belong to the next one."""
        del self._thread[:count]

    # ----- agent.actor.Actor -------------------------------------------------

    def _reached(self, step: int, signals: "PerceivedSignals", *, falling_due: bool = False) -> bool:
        """Whether anything reached the character this step, or an agreed time came, keeping
        the turn's reason."""
        self._weighed = (step, _reason(signals, falling_due=falling_due))
        return bool(signals) or falling_due

    def _reason_for(self, step: int) -> str:
        weighed_step, reason = self._weighed
        return reason if weighed_step == step else ""

    def perceived(
        self,
        agent: "Agent",
        *,
        step: int,
        inbox: "Sequence[Message]",
        broadcasts: "Sequence[Broadcast]",
    ) -> None:
        self._step = step
        entry = agent.actor_perceived(step, inbox, broadcasts)
        if entry is not None:
            self._record(entry)

    def began(self, *, step: int, description: str, total_steps: int) -> None:
        self._record(BeganEntry(step=step, description=description, total_steps=total_steps))

    def progressed(self, *, step: int, elapsed_steps: int, total_steps: int) -> None:
        self._record(ProgressedEntry(step=step, elapsed_steps=elapsed_steps, total_steps=total_steps))

    def ended(self, *, step: int, fact: str, succeeded: bool, happened: bool) -> None:
        self._freed = True
        self._record(EndedEntry(step=step, fact=fact, succeeded=succeeded, happened=happened))

    def decides_this_step(self, *, step: int, signals: "PerceivedSignals") -> bool:
        # Weighed first, never short-circuited: the turn's reason needs it at either pace.
        reached = self._reached(step, signals)
        return self._pace is ActorPace.EVERY_STEP or self._freed or reached

    def weighs_interrupt(self, *, step: int, signals: "PerceivedSignals", falling_due: bool) -> bool:
        reached = self._reached(step, signals, falling_due=falling_due)
        return self._channel.break_off_requested or reached

    async def evaluate_interrupt(self, agent: "Agent", *, step: int) -> tuple[bool, str]:
        """A break-off already requested needs no turn; otherwise retain the actor's thought.

        A break-off records no trigger even when something reached the character: it was asked
        for before, so it answers nothing that reached it."""
        doing = agent.personality.state.current_action or ""
        self_initiated = self._channel.take_break_off()
        if self_initiated:
            interrupt, thought, trigger = True, "", ""
        else:
            trigger = self._reason_for(step)
            answer = await self._ask(ActorTurn(
                agent_id=agent.agent_id, step=step, kind=TurnKind.INTERRUPT, doing=doing, reason=trigger,
            ))
            assert isinstance(answer, ActorInterruptAnswer)
            interrupt, thought = answer.interrupt, answer.thought
        self._beat(BeatKind.INTERRUPT, Stage.INTERRUPT, agent.agent_id, step, {
            "interrupt": interrupt, "thought": thought, "doing": doing, "trigger": trigger,
            "self_initiated": self_initiated,
        })
        return interrupt, thought

    async def decide(self, agent: "Agent", *, step: int, packet: "PerceptionPacket") -> ActorDecision:
        # The character is idle: a break-off still requested has nothing left to stop.
        self._channel.take_break_off()
        self._freed = False

        def check(answer: ActorAnswer) -> None:
            assert isinstance(answer, ActorDecision)
            if answer.action is not None:
                agent.bind_actor_action(answer.action, packet)

        deciding = self._deciding = _DecideTurn(agent, packet)
        try:
            answer = await self._ask(
                ActorTurn(
                    agent_id=agent.agent_id, step=step, kind=TurnKind.DECIDE,
                    briefing=agent.actor_briefing(packet),
                    # At every step the actor is asked whatever happened; only a decision at the
                    # actor's pace was brought on by something.
                    reason=self._reason_for(step) if self._pace is ActorPace.WHEN_REACHED else "",
                ),
                check=check,
            )
        finally:
            self._deciding = None
        assert isinstance(answer, ActorDecision)
        # Readings the actor didn't confirm never shaped the world.
        chosen = deciding.confirmed.reading if deciding.confirmed is not None else None
        for call in deciding.readings:
            if call is not chosen:
                note_call_adoption(call, False, SUPERSEDED)
        with annotate_call(location=situation_location(packet.spatial)):
            self._beat(BeatKind.DECISION, Stage.DECISION, agent.agent_id, step,
                       actor_decision_beat(answer, packet))
        return answer

    async def consider_invitation(
        self, agent: "Agent", *, step: int, invitation: Invitation, decided: str,
    ) -> bool:
        answer = await self._ask(ActorTurn(
            agent_id=agent.agent_id, step=step, kind=TurnKind.INVITE,
            invitation=invitation, decided=decided,
        ))
        assert isinstance(answer, ActorInvitationAnswer)
        return answer.accept

    async def speak(self, agent: "Agent", *, step: int, conversation: Conversation) -> ActorDialogueTurn:
        def check(answer: ActorAnswer) -> None:
            assert isinstance(answer, ActorDialogueTurn)
            if len(answer.line.strip()) > conversation.max_turn_chars:
                raise ActorAnswerRejected("turn_too_long")

        answer = await self._ask(
            ActorTurn(agent_id=agent.agent_id, step=step, kind=TurnKind.SPEAK, conversation=conversation),
            check=check,
        )
        assert isinstance(answer, ActorDialogueTurn)
        return ActorDialogueTurn(line=answer.line.strip(), end=answer.end)

    def spoke(self, *, step: int, dialogue_turn: DialogueTurn, end: bool) -> None:
        self._record(DialogueTurnEntry(step=step, dialogue_turn=dialogue_turn, end=end))

    async def _ask(self, turn: ActorTurn, *, check: Callable[[ActorAnswer], None] | None = None) -> ActorAnswer:
        # A SPEAK turn's record is the dialogue turn it is answered with (``spoke``).
        if turn.kind is not TurnKind.SPEAK:
            self._record(AskedEntry(step=turn.step, turn=turn))
        answer = await self._channel.request(turn, check=check)
        async with self._amending:   # an amendment landing in this turn finishes first
            pass
        return answer

    # ----- the actor's words and requests -------------------------------------

    def answer(self, answer: ActorPublicAnswer) -> None:
        """Answer the open turn. The answer is recorded once taken, before the turn closes, so
        the thread never shows the turn closed without it; a SPEAK turn's dialogue turn is
        recorded as it is spoken (``spoke``), after the turn closes. Refused answers raise as
        ``ActorChannel.submit`` says."""
        public_answer = answer
        resolved_answer: ActorAnswer
        if isinstance(answer, ActorConfirmation):
            turn = self._channel.pending_turn
            deciding = self._deciding
            pending = deciding.pending if deciding is not None else None
            if turn is None or turn.kind is not TurnKind.DECIDE or pending is None or pending.summary.decision_summary_id != answer.decision_summary_id:
                raise ActorAnswerRejected("decision_summary_not_pending")
            resolved_answer = pending.decision
            deciding.confirmed = pending
        elif isinstance(answer, ActorIdleAnswer):
            resolved_answer = ActorDecision(idle=True)
        elif isinstance(answer, (ActorInterruptAnswer, ActorInvitationAnswer, ActorDialogueTurn)):
            resolved_answer = answer
        else:
            raise ValueError("Expected a public actor answer")

        def taken(turn: ActorTurn) -> None:
            if turn.kind is not TurnKind.SPEAK:
                self._record(AnsweredEntry(step=turn.step, turn_id=turn.turn_id, answer=public_answer))

        self._channel.submit(resolved_answer, on_taken=taken)

    def _performed(self) -> "Agent":
        if self._agent is None:
            raise ValueError("The actor performs no character")
        return self._agent

    def _decision_turn(self, turn_id: str) -> tuple[ActorTurn, _DecideTurn]:
        self._performed()
        turn = self._channel.pending_turn
        deciding = self._deciding
        if turn is None or turn.turn_id != turn_id or turn.kind is not TurnKind.DECIDE or deciding is None:
            raise ValueError("That decision turn is no longer waiting for the actor")
        return turn, deciding

    def operations(self, turn_id: str, target: Ref) -> ActorOperations:
        _turn, deciding = self._decision_turn(turn_id)
        return actor_operations(turn_id, target, deciding.packet)

    async def intent(self, intent: ActorIntent) -> ActorIntentResult:
        """Prepare or submit one intention; only this generation may publish its decision summary."""
        turn, deciding = self._decision_turn(intent.turn_id)
        agent, packet = deciding.agent, deciding.packet
        self._intent_generation += 1
        generation = self._intent_generation
        deciding.pending = None
        words = intent.text if intent.text is not None else intent.description
        base: ActorAction | None = None
        reading = ""
        reading_call: Any = None
        try:
            base = build_actor_operation(intent, packet) if intent.operation is not None else None
            is_direct_operation = not requires_actor_interpretation(intent)
            decision: ActorDecision | None
            if is_direct_operation:
                decision = ActorDecision(action=base)
            else:
                with observe_actor(), annotate_call(actor_words=words, turn_id=turn.turn_id):
                    if base is None:
                        read = await agent.interpret_actor_decision(words, packet)
                    else:
                        read = await agent.interpret_actor_decision(words, packet, intent=intent, base=base)
                    reading_call = get_active_call()
                if reading_call is not None:
                    deciding.readings.append(reading_call)
                decision, reading = read if read is not None else (None, "")
            if decision is not None and decision.action is not None:
                # A selected operation may lack slots until its description is interpreted.
                agent.bind_actor_action(decision.action, packet)
            reason = None if decision is not None else UNREADABLE
        except ActorAnswerRejected as exc:
            note_active_call_adoption(False, exc.reason)
            decision, reason = None, exc.reason
        current = self._channel.pending_turn
        if generation != self._intent_generation or current is None or current.turn_id != turn.turn_id:
            return ActorIntentResult("rejected", "stale_intent")
        decision_summary = build_actor_decision_summary(
            decision, packet, seconds_per_step=self._time_config.seconds_per_step, intent=intent,
        ) if decision is not None else None
        self._record(SaidEntry(
            step=turn.step, turn_id=turn.turn_id, words=words or (base.action_description if base else ""),
            decision_summary=decision_summary, reason=reason, intent=intent,
            reading=reading if reason == UNREADABLE else "",
        ))
        if decision_summary is None or decision is None:
            return ActorIntentResult("rejected", reason)
        deciding.pending = _PendingDecision(decision_summary, decision, reading_call)
        if is_direct_operation:
            self.answer(ActorConfirmation(decision_summary.decision_summary_id))
            return ActorIntentResult("submitted")
        return ActorIntentResult("needs_confirmation", decision_summary=decision_summary)

    def break_off(self) -> None:
        """Break off what the character is doing: now if an INTERRUPT turn is open, otherwise at
        the next interrupt stage. Raises ``ActorAnswerRejected`` when it is doing nothing."""
        agent = self._performed()
        turn = self._channel.pending_turn
        if turn is not None and turn.kind is TurnKind.INTERRUPT:
            self.answer(ActorInterruptAnswer(interrupt=True))
            return
        if agent.personality.state.action_status != ActionStatus.IN_PROGRESS:
            raise ActorAnswerRejected("nothing_to_break_off")
        if self._channel.break_off_requested:
            return
        self._channel.request_break_off()
        self._record(BreakOffEntry(step=self._step))

    # ----- the character's mind ----------------------------------------------

    async def mind(self, *, step: int) -> ActorMind:
        """The character's mind as the actor sees it now."""
        return await self._performed().actor_mind(pending=self._pending_amendments, step=step)

    async def amend(self, amendments: ActorAmendments) -> bool:
        """Save the actor's amendments; whether they landed now (``True``) or wait for the next
        step. Raises ``ActorAnswerRejected`` for one the mind doesn't list."""
        agent = self._performed()
        await agent.check_amendments(amendments)
        async with self._amending:
            # Read under the lock: the turn may be answered while this lands, but its step resumes
            # only once the lock is released (see _ask).
            turn = self._channel.pending_turn
            # Not during an invitation: it is asked after the step's plan is laid, and committing
            # that plan writes the goals it carries over the character's. Not during a conversation:
            # it is spoken while the step's other acts are adjudicated, and its own feedback is yet
            # to land.
            if turn is not None and turn.kind in (TurnKind.DECIDE, TurnKind.INTERRUPT):
                changed = await agent.amend_mind(amendments, step=turn.step)
                stage = Stage.DECISION if turn.kind is TurnKind.DECIDE else Stage.INTERRUPT
                self._beat(BeatKind.MIND, stage, agent.agent_id, turn.step, changed)
                return True
            self._pending_amendments = (
                amendments if self._pending_amendments is None else self._pending_amendments.then(amendments)
            )
            return False

    async def land_amendments(self, agent: "Agent", *, step: int) -> None:
        pending, self._pending_amendments = self._pending_amendments, None
        if pending is not None:
            changed = await agent.amend_mind(pending, step=step)
            # Landed before the character perceives this step.
            self._beat(BeatKind.MIND, Stage.PERCEPTION, agent.agent_id, step, changed)

    def _beat(self, kind: BeatKind, stage: Stage, agent_id: str, step: int, payload: dict[str, Any]) -> None:
        if payload:
            record_beat(self._trace_sink, kind, stage, payload, world_id=self._world_id, step=step,
                        agent_id=agent_id, actor=True)
