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
handed to the step's snapshot when the step commits (``take_thread``).

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

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Callable, Sequence

from agent.personality import ActionStatus
from agent.actor_operations import (
    actor_operations, build_actor_action_summary, build_actor_operation, requires_actor_interpretation,
)
from core.context import annotate_call, note_active_call_adoption
from core.interfaces.action import DialogueTurn, Ref
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
    ActorActionSummary,
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

# Told each entry of the actor's thread, as JSON, as it is recorded.
EntryListener = Callable[[dict[str, Any]], None]


def _reason(signals: "PerceivedSignals") -> str:
    """Why the character is asked, for the actor: how many things reached it. What they were is
    in the thread's perceptions and on the map."""
    return f"有 {len(signals)} 件事发生" if signals else ""


class ActorCoordinator:
    def __init__(
        self,
        time_config: WorldTimeConfig,
        *,
        on_turn: TurnListener | None = None,
        on_entry: EntryListener | None = None,
    ) -> None:
        self._time_config = time_config
        self._on_turn = on_turn
        self._intent_generation = 0
        self._action_summary: tuple[str, ActorActionSummary, ActorDecision] | None = None
        self._channel = ActorChannel(self._turn_changed)
        self._on_entry = on_entry
        self._agent: "Agent | None" = None
        self._pace = ActorPace.EVERY_STEP
        # The character and perception used to interpret intentions during the open DECIDE turn.
        self._asking: "tuple[Agent, PerceptionPacket] | None" = None
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

    def _turn_changed(self, turn: ActorTurn | None) -> None:
        self._intent_generation += 1
        self._action_summary = None
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

    def take_thread(self) -> list[dict[str, Any]]:
        """The thread since the last commit, for the committing step's snapshot."""
        thread, self._thread = self._thread, []
        return thread

    # ----- agent.actor.Actor -------------------------------------------------

    def _reached(self, step: int, signals: "PerceivedSignals") -> bool:
        """Whether anything reached the character this step, keeping the turn's reason."""
        self._weighed = (step, _reason(signals))
        return bool(signals)

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

    def weighs_interrupt(self, *, step: int, signals: "PerceivedSignals") -> bool:
        reached = self._reached(step, signals)
        return self._channel.break_off_requested or reached

    async def evaluate_interrupt(self, agent: "Agent", *, step: int) -> tuple[bool, str]:
        """A break-off already requested needs no turn; otherwise retain the actor's thought."""
        if self._channel.take_break_off():
            return True, ""
        answer = await self._ask(ActorTurn(
            agent_id=agent.agent_id, step=step, kind=TurnKind.INTERRUPT,
            doing=agent.personality.state.current_action or "", reason=self._reason_for(step),
        ))
        assert isinstance(answer, ActorInterruptAnswer)
        return answer.interrupt, answer.thought

    async def decide(self, agent: "Agent", *, step: int, packet: "PerceptionPacket") -> ActorDecision:
        # The character is idle: a break-off still requested has nothing left to stop.
        self._channel.take_break_off()
        self._freed = False

        def check(answer: ActorAnswer) -> None:
            assert isinstance(answer, ActorDecision)
            if answer.action is not None:
                agent.bind_actor_action(answer.action, packet)

        self._asking = (agent, packet)
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
            self._asking = None
        assert isinstance(answer, ActorDecision)
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
        return await self._channel.request(turn, check=check)

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
            pending = self._action_summary
            if turn is None or turn.kind is not TurnKind.DECIDE or pending is None or pending[0] != turn.turn_id or pending[1].action_summary_id != answer.action_summary_id:
                raise ActorAnswerRejected("action_summary_not_pending")
            resolved_answer = pending[2]
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

    def _decision_turn(self, turn_id: str) -> ActorTurn:
        self._performed()
        turn = self._channel.pending_turn
        if turn is None or turn.turn_id != turn_id or turn.kind is not TurnKind.DECIDE or self._asking is None:
            raise ValueError("That decision turn is no longer waiting for the actor")
        return turn

    def operations(self, turn_id: str, target: Ref) -> ActorOperations:
        self._decision_turn(turn_id)
        assert self._asking is not None
        return actor_operations(turn_id, target, self._asking[1])

    async def intent(self, intent: ActorIntent) -> ActorIntentResult:
        """Prepare or submit one intention; only this generation may publish its action summary."""
        turn = self._decision_turn(intent.turn_id)
        assert self._asking is not None
        agent, packet = self._asking
        self._intent_generation += 1
        generation = self._intent_generation
        self._action_summary = None
        words = intent.text if intent.text is not None else intent.description
        base: ActorAction | None = None
        try:
            base = build_actor_operation(intent, packet) if intent.operation is not None else None
            is_direct_operation = not requires_actor_interpretation(intent)
            if is_direct_operation:
                decision = ActorDecision(action=base)
            else:
                with annotate_call(actor_words=words, turn_id=turn.turn_id):
                    if base is None:
                        decision = await agent.interpret_actor_decision(words, packet)
                    else:
                        decision = await agent.interpret_actor_decision(words, packet, intent=intent, base=base)
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
        action_summary = build_actor_action_summary(
            decision.action, packet, seconds_per_step=self._time_config.seconds_per_step, intent=intent,
        ) if decision is not None and decision.action is not None else None
        self._record(SaidEntry(step=turn.step, turn_id=turn.turn_id, words=words or (base.action_description if base else ""), action_summary=action_summary, reason=reason, intent=intent))
        if action_summary is None or decision is None:
            return ActorIntentResult("rejected", reason)
        self._action_summary = (turn.turn_id, action_summary, decision)
        if is_direct_operation:
            self.answer(ActorConfirmation(action_summary.action_summary_id))
            return ActorIntentResult("submitted")
        return ActorIntentResult("needs_confirmation", action_summary=action_summary)

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
        turn = self._channel.pending_turn
        # Not during an invitation: it is asked after the step's plan is laid, and committing
        # that plan writes the goals it carries over the character's. Not during a conversation:
        # it is spoken while the step's other acts are adjudicated, and its own feedback is yet
        # to land.
        if turn is not None and turn.kind in (TurnKind.DECIDE, TurnKind.INTERRUPT):
            await agent.amend_mind(amendments, step=turn.step)
            return True
        self._pending_amendments = (
            amendments if self._pending_amendments is None else self._pending_amendments.then(amendments)
        )
        return False

    async def land_amendments(self, agent: "Agent", *, step: int) -> None:
        pending, self._pending_amendments = self._pending_amendments, None
        if pending is not None:
            await agent.amend_mind(pending, step=step)
