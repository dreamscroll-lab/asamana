"""The actor, on the engine's side: a character's secondary cognition, asked through a channel
to the person performing it (``agent.actor.Actor``).

The character asks its actor at the stages the actor owns, and the step suspends until the
person answers:
- interrupt: whether to break off the action in progress (``evaluate_interrupt``);
- decision: the decision, goals and amendments (``decide``).

What reaches the actor is what the character perceived (``engine.signals``), the same signals
that put an AI character before the world-pressure judgment. Nothing here judges what matters to
the actor; every answer is the actor's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.interfaces.actor import (
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorTurn,
    TriggerKind,
    TurnKind,
    TurnTrigger,
)
from core.logging import get_logger
from engine.actor_channel import ActorAnswer, ActorChannel, TurnListener

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.perception import PerceptionPacket
    from core.interfaces.perception import PerceivedSignals

logger = get_logger(__name__)


class ActorCoordinator:
    def __init__(self, on_turn: TurnListener | None = None) -> None:
        self._channel = ActorChannel(on_turn)
        # While a turn is open: the character, and for a DECIDE turn what it has in front of it,
        # for reading the actor's words against.
        self._asking: "tuple[Agent, PerceptionPacket | None] | None" = None
        # The step's triggers, read once per step: reading consumes the TARGETED note, and the
        # interrupt and decision stages of one step must see the same triggers.
        self._triggers: tuple[int, tuple[TurnTrigger, ...]] | None = None

    @property
    def channel(self) -> ActorChannel:
        return self._channel

    def perform(self, agent: "Agent") -> None:
        """Take ``agent`` over: from now on the character asks this actor."""
        if self._channel.agent_id not in (None, agent.agent_id):
            raise ValueError(f"The actor already performs {self._channel.agent_id!r}")
        self._channel.follow(agent.agent_id)
        agent.actor = self

    def _triggers_for(self, step: int, signals: "PerceivedSignals") -> tuple[TurnTrigger, ...]:
        """Everything that reached the character this step."""
        if self._triggers is not None and self._triggers[0] == step:
            return self._triggers[1]
        triggers: List[TurnTrigger] = [
            *(TurnTrigger(TriggerKind.MESSAGE, m.content) for m in signals.messages),
            *(TurnTrigger(TriggerKind.BROADCAST, b.content) for b in signals.broadcasts),
            *(TurnTrigger(TriggerKind.AMBIENT, e.content) for e in signals.ambient),
        ]
        if self._channel.take_targeted():
            triggers.append(TurnTrigger(TriggerKind.TARGETED))
        self._triggers = (step, tuple(triggers))
        return self._triggers[1]

    # ----- agent.actor.Actor -------------------------------------------------

    def decides_this_step(self, *, step: int, signals: "PerceivedSignals") -> bool:
        return self._channel.wants_decision(step=step, triggers=self._triggers_for(step, signals))

    def weighs_interrupt(self, *, step: int, signals: "PerceivedSignals") -> bool:
        return bool(self._triggers_for(step, signals))

    async def evaluate_interrupt(
        self, agent: "Agent", *, step: int, signals: "PerceivedSignals",
    ) -> tuple[bool, str]:
        """Words about what to do next, said with a break-off, are held for this step's decision
        stage: only there does the character have the lists to read them against."""
        self._asking = (agent, None)
        try:
            answer = await self._channel.request(ActorTurn(
                agent_id=agent.agent_id, step=step, kind=TurnKind.INTERRUPT,
                triggers=self._triggers_for(step, signals),
                doing=agent.personality.state.current_action or "",
            ))
        finally:
            self._asking = None
        assert isinstance(answer, ActorInterruptAnswer)
        if answer.interrupt and answer.then.strip():
            self._channel.hold_next_words(answer.then, step=step)
        return answer.interrupt, answer.thought

    async def decide(
        self, agent: "Agent", *, step: int, packet: "PerceptionPacket", signals: "PerceivedSignals",
    ) -> ActorDecision:
        proposal = await self._proposal(agent, packet, self._channel.take_next_words(step=step), step=step)
        decision = await self._ask_decision(
            agent, packet, step=step, triggers=self._triggers_for(step, signals), proposal=proposal,
        )
        if decision.wait is not None:
            self._channel.start_wait(decision.wait, step=step)
        return decision

    def acted_upon(self) -> None:
        self._channel.note_targeted()

    # -------------------------------------------------------------------------

    async def _ask_decision(
        self,
        agent: "Agent",
        packet: "PerceptionPacket",
        *,
        step: int,
        triggers: tuple[TurnTrigger, ...],
        proposal: ActorDecision | None,
    ) -> ActorDecision:
        since_step, already_listed = self._channel.feedback_window(step=step)
        feedback = await agent.actor_feedback(since_step=since_step, already_listed=already_listed)

        def check(answer: ActorAnswer) -> None:
            assert isinstance(answer, ActorDecision)
            if answer.action is not None:
                agent.bind_actor_action(answer.action, packet)
            if answer.amendments is not None:
                agent.check_amendments(answer.amendments, feedback)

        self._asking = (agent, packet)
        try:
            answer = await self._channel.request(
                ActorTurn(
                    agent_id=agent.agent_id, step=step, kind=TurnKind.DECIDE, triggers=triggers,
                    briefing=agent.actor_briefing(packet), feedback=feedback, proposal=proposal,
                ),
                check=check,
            )
        finally:
            self._asking = None
        assert isinstance(answer, ActorDecision)
        # The next window starts at this step, so only this step's memories could be listed again.
        self._channel.note_decided(
            step=step,
            listed_memory_ids=frozenset(m.memory_id for m in feedback.memories if m.created_step >= step),
        )
        return answer

    async def _proposal(
        self, agent: "Agent", packet: "PerceptionPacket", words: str, *, step: int,
    ) -> ActorDecision | None:
        """What the actor said with this step's break-off, read now that the lists exist."""
        if not words.strip():
            return None
        proposal = await agent.interpret_actor_decision(words, packet)
        if proposal is not None and proposal.action is not None:
            try:
                agent.bind_actor_action(proposal.action, packet)
            except ActorAnswerRejected as exc:
                logger.info(
                    "actor_proposal_rejected",
                    extra={"agent_id": agent.agent_id, "step": step, "reason": exc.reason},
                )
                return None
        return proposal

    async def interpret(self, text: str) -> ActorDecision | ActorInterruptAnswer | None:
        """The actor's words read as an answer to the open turn, for them to confirm.

        Answers nothing and changes nothing. ``None`` when the words couldn't be read; a
        decision whose action the character can't take raises ``ActorAnswerRejected``, so the
        actor hears why before confirming anything.
        """
        turn = self._channel.pending_turn
        if turn is None or self._asking is None:
            raise ValueError("No turn is waiting for the actor")
        agent, packet = self._asking
        if turn.kind is TurnKind.INTERRUPT:
            return await agent.interpret_actor_interrupt(text)
        assert packet is not None
        decision = await agent.interpret_actor_decision(text, packet)
        if decision is not None and decision.action is not None:
            agent.bind_actor_action(decision.action, packet)
        return decision
