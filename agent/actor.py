"""The actor: a character's secondary cognition, driven by a person instead of AI.

``Agent`` is a character's cognition, AI-driven by default. An actor may take the character
over, and then supplies what the AI would otherwise produce: whether it decides this step, its
decision, whether it breaks off an action, whether it joins an act it is invited into (an AI
character always does), its dialogue turns in a conversation, and amendments to its mind.
Perception, adjudication and the computing of feedback stay the character's own, the same for
either; the actor is told, as each happens, what reaches the character (others' acts that landed
on it included) and how its own acts begin, go on and end.

The character asks its actor through this protocol and nobody else does: the runtime calls the
character the same way whoever drives it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.perception import PerceptionPacket
    from core.interfaces.action import DialogueTurn
    from core.interfaces.actor import ActorDecision, ActorDialogueTurn, Conversation, Invitation
    from core.interfaces.message import Message
    from core.interfaces.perception import Broadcast, PerceivedSignals


class Actor(Protocol):
    def perceived(
        self,
        agent: "Agent",
        *,
        step: int,
        inbox: "Sequence[Message]",
        broadcasts: "Sequence[Broadcast]",
    ) -> None:
        """What reaches the character this step, as its perception stage takes it in: the letters
        and broadcasts, and others' acts that landed on it since it last perceived (on ``agent``,
        until its perception clears them)."""
        ...

    def decides_this_step(self, *, step: int, signals: "PerceivedSignals") -> bool:
        """Whether the idle character decides this step. The actor sets its own pace in place
        of the engine's cadence."""
        ...

    async def decide(self, agent: "Agent", *, step: int, packet: "PerceptionPacket") -> "ActorDecision":
        """The decision and goals for this step, already checked against ``packet``. Suspends
        until the actor answers."""
        ...

    def weighs_interrupt(self, *, step: int, signals: "PerceivedSignals") -> bool:
        """Whether to weigh breaking off the action in progress: when the actor asked to, or on
        anything that reached the character, where the AI weighs only what crosses the engine's
        thresholds."""
        ...

    async def evaluate_interrupt(self, agent: "Agent", *, step: int) -> tuple[bool, str]:
        """Break off or carry on, as the actor asked already or answers now. The thought is
        empty: what the actor means to do instead is read at the decision stage."""
        ...

    async def consider_invitation(
        self, agent: "Agent", *, step: int, invitation: "Invitation", decided: str,
    ) -> bool:
        """Whether the free character joins the act it is invited into; ``decided`` is the act it
        decided on this step, which accepting sets aside. Suspends until the actor answers."""
        ...

    async def speak(self, agent: "Agent", *, step: int, conversation: "Conversation") -> "ActorDialogueTurn":
        """The character's dialogue turn in ``conversation``, or the end of it. Suspends until the
        actor answers."""
        ...

    def spoke(self, *, step: int, dialogue_turn: "DialogueTurn", end: bool) -> None:
        """A dialogue turn of the character's live conversation was spoken, by either side;
        ``end``: the conversation ends with it (its line empty when it ended without a word)."""
        ...

    async def land_amendments(self, agent: "Agent", *, step: int) -> None:
        """Apply the amendments the actor saved since the last step, at the start of ``step``."""
        ...

    def began(self, *, step: int, description: str, total_steps: int) -> None:
        """The character set out on an act, its own or one it was drawn into, expected to take
        ``total_steps`` steps from this one."""
        ...

    def progressed(self, *, step: int, elapsed_steps: int, total_steps: int) -> None:
        """The act in progress went on another step."""
        ...

    def ended(self, *, step: int, fact: str, succeeded: bool, happened: bool) -> None:
        """The act in progress ended, as the character remembers it (``fact``; empty when it was
        never adjudicated). ``happened`` is False for an act that never engaged the world."""
        ...
