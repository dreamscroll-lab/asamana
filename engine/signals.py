"""What reached one character this step, read off the perception everyone gets.

One definition shared by every consumer, so they can't drift apart: the world-pressure
evaluator (whether a character has anything to be pressed by), the actor's turns (whether to
wake or ask the actor), and the per-step perception record a character's view is replayed from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from core.interfaces.perception import PerceivedSignals
from engine.broadcast import BroadcastChannel

if TYPE_CHECKING:
    from core.interfaces.message import Message
    from core.interfaces.perception import AmbientEvent, Broadcast, SpatialPerception


def perceived_signals(
    *,
    location_id: str,
    inbox: Sequence["Message"],
    broadcasts: Sequence["Broadcast"],
    ambient: Sequence["AmbientEvent"],
) -> PerceivedSignals:
    return PerceivedSignals(
        messages=tuple(inbox),
        broadcasts=tuple(BroadcastChannel.for_location(list(broadcasts), location_id)),
        ambient=tuple(ambient),
    )


def perception_record(spatial: "SpatialPerception", signals: PerceivedSignals) -> dict[str, Any]:
    """One character's perception this step, as the step snapshot keeps it.

    Ids for who and what was in view; the perceived texts as they reached the character (each
    ambient text is already this observer's view, with its own acts excluded).
    """
    return {
        "location_id": spatial.location_id,
        "agent_ids": list(spatial.visible_agent_ids),
        "entity_ids": [e.entity_id for e in spatial.visible_entities],
        "npc_ids": list(spatial.visible_npc_ids),
        "message_ids": [m.id for m in signals.messages],
        "broadcasts": [b.content for b in signals.broadcasts],
        "ambient": [
            {"text": e.content, "strength": e.strength, "doer_ids": list(e.doer_ids)}
            for e in signals.ambient
        ],
    }
