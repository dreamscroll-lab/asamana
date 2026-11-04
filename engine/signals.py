"""What reached one character this step, read off the perception everyone gets.

One definition shared by every consumer, so they can't drift apart: the world-pressure
evaluator (whether a character has anything to be pressed by, and by what) and the actor's turns
(whether to ask the actor, and the turn's reason).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from core.interfaces.perception import PerceivedSignals
from engine.broadcast import BroadcastChannel

if TYPE_CHECKING:
    from core.interfaces.message import Message
    from core.interfaces.perception import AmbientEvent, Broadcast


def perceived_signals(
    *,
    location_id: str,
    inbox: Sequence["Message"],
    broadcasts: Sequence["Broadcast"],
    ambient: Sequence["AmbientEvent"],
    acted_upon: Sequence[str] = (),
) -> PerceivedSignals:
    return PerceivedSignals(
        messages=tuple(inbox),
        broadcasts=tuple(BroadcastChannel.for_location(list(broadcasts), location_id)),
        ambient=tuple(ambient),
        acted_upon=tuple(acted_upon),
    )
