"""The actor: a character's secondary cognition, driven by a person instead of AI.

``Agent`` is a character's cognition, AI-driven by default. An actor may take the character
over, and then supplies what the AI would otherwise produce: whether it decides this step, its
decision and goals, whether it breaks off an action, and amendments to how it took things.
Perception, adjudication and the computing of feedback stay the character's own, the same for
either.

The character asks its actor through this protocol and nobody else does: the runtime calls the
character the same way whoever drives it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.perception import PerceptionPacket
    from core.interfaces.actor import ActorDecision
    from core.interfaces.perception import PerceivedSignals


class Actor(Protocol):
    def decides_this_step(self, *, step: int, signals: "PerceivedSignals") -> bool:
        """Whether the idle character decides this step. The actor sets its own pace in place
        of the engine's cadence."""
        ...

    async def decide(
        self, agent: "Agent", *, step: int, packet: "PerceptionPacket", signals: "PerceivedSignals",
    ) -> "ActorDecision":
        """The decision, goals and amendments for this step, already checked against
        ``packet``. Suspends until the actor answers."""
        ...

    def weighs_interrupt(self, *, step: int, signals: "PerceivedSignals") -> bool:
        """Whether the actor is asked about breaking off the action in progress: on anything
        that reached the character, where the AI weighs only what crosses the engine's
        thresholds."""
        ...

    async def evaluate_interrupt(
        self, agent: "Agent", *, step: int, signals: "PerceivedSignals",
    ) -> tuple[bool, str]:
        """Break off or carry on, and the character's first-person thought on it."""
        ...

    def acted_upon(self) -> None:
        """Someone else's action, or a world event, landed on the character this step."""
        ...
