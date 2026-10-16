"""What reached an actor's character as a step began, to be told in words: the letters and
broadcasts arriving, and others' acts that landed on it since it last perceived. An entry of the
actor's thread, from the same objects its perception stage takes in (``Agent.perceive_step``) and
its signals carry (``PerceivedSignals``); what it saw is the map's to show.

Names come from perception, never from a world table, so the actor knows exactly what the
character knows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from core.interfaces.actor import BriefMessage, PerceivedEntry

if TYPE_CHECKING:
    from core.interfaces.message import Message
    from core.interfaces.perception import Broadcast


def build_perceived_entry(
    step: int,
    inbox: Sequence["Message"],
    broadcasts: Sequence["Broadcast"],
    acted_upon: Sequence[str] = (),
) -> PerceivedEntry | None:
    """``None`` when nothing arrived."""
    if not inbox and not broadcasts and not acted_upon:
        return None
    return PerceivedEntry(
        step=step,
        messages=tuple(
            BriefMessage(sender_name=m.sender_name, content=m.content, urgency=m.urgency.value)
            for m in inbox
        ),
        broadcasts=tuple(b.content for b in broadcasts),
        acted_upon=tuple(acted_upon),
    )
