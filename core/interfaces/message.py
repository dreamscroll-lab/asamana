"""Message contracts.

MessageSystem charter:

Scope
    MessageSystem carries only cross-location / cross-time information transfer.
    Same-place, same-time information flows through EnvironmentSystem.record_carry_observation
    → SpatialPerception.ambient_events → PerceptionMemoryLayer, not MessageSystem.

Lossless transport
    MessageSystem does no rendering, distortion or subjective interpretation.
    Message.content leaves the pipe byte-identical to how it entered.
    - Distortion arises naturally from memory decay + agents retelling (memory/agent layer)
    - Subjective interpretation is done by the receiving agent (cognition layer), written only
      to EXPERIENTIAL, never altering FACTUAL content

Data-driven, no type enum
    There is no PropagationType. Delivery is fully determined by three fields:
    - recipients: list[str] | None   None = broadcast; list = targeted
    - location_scope: str | None     None = any location; str = target delivery location
    - deliver_step: int              current step = immediate; future step = delayed

    New scenarios come from field combinations, not new types.

No side effects
    MessageSystem doesn't update relations, write memory, trigger interrupts or call the LLM.
    Those are the receiving agent's own decisions.

The receiver decides
    A sender may hint with urgency=Urgency.HIGH etc., but can't force any receiver behavior.
    urgency is a sender→receiver meta-signal, not a command.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.interfaces.urgency import Urgency


@dataclass
class Message:
    """The atomic unit of cross-location / cross-time transfer.

    Field semantics matrix (data-driven, no type enum):

        recipients   | location_scope | meaning
        ─────────────┼────────────────┼────────────────────────────────────────────
        list[str]    | None           | targeted at the listed agents, wherever they are
        None         | str (L)        | area broadcast: every agent at L
        None         | None           | global broadcast: every agent (narrator use case)
        list[str]    | str (L)        | intersection: listed agents who are at L (edge but valid)

    All four cells then subtract ``{sender_id} ∪ actor_ids`` (self-exclusion, below).

    Time is orthogonal: deliver_step > created_step = delayed. ``content`` and ``urgency``
    follow the module charter (lossless transport; the receiver decides).

    ``sender_is_agent``: can the receiver form a relation with this sender? The body kind lives
    on ``EnvironmentSystem``, which ``agent/`` must not use, so the engine declares it. Not
    interchangeable with ``metadata["narrative"]``, which also drops ``sender_name``: a
    non-cognitive body is a real person whose name must stay in memory.

    ``actor_ids`` is this pipe's half of self-exclusion (the other is ``AmbientEvent.actor_ids``):
    the mouth and the author can differ, and an agent who dictated a message an Npc reads aloud
    must not receive their own order and take it for a forgery. Addressing is still the matrix's.
    """

    id: str
    world_id: str
    sender_id: str                         # agent_id / npc_id / "narrator"; never "system"/"world" (those use BroadcastChannel)
    content: str                           # original text, never modified in the pipe
    recipients: list[str] | None           # None = broadcast; list = targeted
    location_scope: str | None             # None = any location; str = target location (unrelated to the sender's location)
    deliver_step: int                      # step at which delivery happens
    created_step: int
    sender_name: str = ""
    intent: str = ""
    urgency: Urgency = Urgency.NORMAL      # 4-level enum — meta-signal, receiver may ignore it
    sender_is_agent: bool = True           # False = sender has no cognition; receivers form no relation with it
    actor_ids: tuple[str, ...] = ()        # all members of the acting body, excluded from delivery; sender always excluded
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageProvider(ABC):
    """Abstract message provider — the underlying queue contract, agnostic of higher-level semantics."""

    @abstractmethod
    async def enqueue(self, message: Message) -> None:
        """Publish a message."""

    @abstractmethod
    async def dequeue_ready(self, world_id: str, current_step: int) -> list[Message]:
        """Return and remove messages whose deliver_step <= current_step."""

    @abstractmethod
    async def peek_pending(self, world_id: str) -> list[Message]:
        """Return queued messages without removing them."""

    @abstractmethod
    async def clear(self, world_id: str | None = None) -> None:
        """Clear queued messages."""
