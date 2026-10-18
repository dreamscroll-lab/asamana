"""What an actor's decision can name.

The briefing is built from the same perception packet an AI character decides on, in the lists
the decision binds against (``agent.decision.candidate_lists``): names come from perception and
relations, never from a world table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.decision import candidate_lists
from core.interfaces.actor import (
    ActorBriefing,
    BriefEntity,
    BriefNpc,
    BriefPerson,
    BriefPlace,
)

if TYPE_CHECKING:
    from agent.perception import PerceptionPacket
    from core.interfaces.perception import SpatialPerception


def build_actor_briefing(packet: "PerceptionPacket") -> ActorBriefing:
    spatial = packet.spatial
    lists = candidate_lists(packet)
    return ActorBriefing(
        people=tuple(
            BriefPerson(
                agent_id=aid,
                name=who.name,
                gender=who.gender,
                present=aid in lists.visible_ids,
            )
            for aid, who in lists.roster
        ),
        npcs=_seen_npcs(spatial),
        places=tuple(
            BriefPlace(
                location_id=rl.location_id,
                name=rl.view.name or "某处",
                travel_seconds=rl.travel_seconds,
            )
            for rl in spatial.reachable_locations
        ),
        entities=_seen_entities(spatial),
        location=BriefPlace(spatial.location_id, (spatial.location_view.name if spatial.location_view else "") or "某处"),
    )


def _seen_npcs(spatial: "SpatialPerception") -> tuple[BriefNpc, ...]:
    return tuple(
        BriefNpc(npc_id=npc_id, name=seen.identity.name or "某人")
        for npc_id, seen in spatial.visible_npcs.items()
    )


def _seen_entities(spatial: "SpatialPerception") -> tuple[BriefEntity, ...]:
    return tuple(
        BriefEntity(entity_id=e.entity_id, name=e.name or "某物", entity_type=e.entity_type,
                    holder_id=e.holder_id, is_takeable=e.is_takeable)
        for e in spatial.visible_entities
    )
