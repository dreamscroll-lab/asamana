"""What an actor's character has in front of it at a decision.

Built from the same perception packet an AI character decides on, so the actor knows exactly
what the character knows: names come from perception and relations, never from a world table,
and the lists are the ones the decision binds against (``agent.decision.candidate_lists``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from agent.decision import candidate_lists
from agent.memory_types import MemoryKind
from core.interfaces.actor import (
    ActorBriefing,
    ActorFeedback,
    BriefMemory,
    BriefMemoryKind,
    BriefPerson,
    BriefPlace,
    BriefRelation,
    BriefEntity,
    BriefNpc,
)
from core.prompts import render_condition

if TYPE_CHECKING:
    from agent.memory_types import Memory
    from core.interfaces.agent_store import AgentRelation
    from agent.perception import PerceptionPacket
    from agent.personality import PersonalityLayer


def build_actor_briefing(
    personality: "PersonalityLayer", packet: "PerceptionPacket", *, seconds_per_step: int,
) -> ActorBriefing:
    spatial = packet.spatial
    context = packet.internal_context
    state = personality.state
    lists = candidate_lists(packet)
    self_id = personality.soul.agent_id
    return ActorBriefing(
        time_label=spatial.world_time_label,
        location_id=spatial.location_id,
        location_name=spatial.location_view.name,
        emotion=context.emotion.primary.value,
        emotion_intensity=context.emotion.intensity,
        emotion_valence=context.emotion.valence,
        vitality=state.vitality,
        condition=render_condition(
            state.condition, now_step=spatial.current_step,
            seconds_per_step=seconds_per_step, with_duration=True,
        ),
        long_term_goals=tuple(state.long_term_goals),
        short_term_goals=tuple(state.short_term_goals),
        people=tuple(
            BriefPerson(
                agent_id=aid,
                name=who.name,
                gender=who.gender,
                present=aid in lists.visible_ids,
                condition=(p.condition if (p := spatial.visible_agents.get(aid)) else ""),
            )
            for aid, who in lists.roster
        ),
        npcs=tuple(
            BriefNpc(
                npc_id=npc_id,
                name=(seen := spatial.visible_npcs[npc_id]).identity.name or "某人",
                description=seen.identity.description,
                busy=seen.busy,
                condition=seen.condition,
            )
            for npc_id in lists.npc_ids
        ),
        places=tuple(
            BriefPlace(
                location_id=rl.location_id,
                name=rl.view.name or "某处",
                description=rl.view.description,
                travel_seconds=rl.travel_seconds,
            )
            for rl in spatial.reachable_locations
        ),
        entities=tuple(
            BriefEntity(
                entity_id=e.entity_id,
                name=e.name or "某物",
                description=e.description,
                content=e.content,
                held_by_me=e.holder_id == self_id,
                holder_name=_holder_name(e.holder_id, self_id=self_id, spatial=spatial),
            )
            for e in spatial.visible_entities
        ),
        memories=(
            _memories(context.factual_memories, BriefMemoryKind.FACTUAL)
            + _memories(context.experiential_memories, BriefMemoryKind.EXPERIENTIAL)
            + _memories(context.insights, BriefMemoryKind.INSIGHT)
            + _memories(context.period_summaries, BriefMemoryKind.PERIOD)
        ),
        relations=tuple(
            BriefRelation(
                agent_id=r.target_agent_id,
                name=r.target_agent_name or "某人",
                gender=r.target_agent_gender,
                trust=r.trust,
                affection=r.affection,
                labels=tuple(r.labels),
                summary=r.history_summary,
                deceased=r.deceased,
            )
            for r in context.relevant_relations
        ),
        foiled_attempts=tuple(context.recent_foiled_attempts),
    )


def build_actor_feedback(
    memories: Iterable["Memory"], relations: Iterable["AgentRelation"],
) -> ActorFeedback:
    return ActorFeedback(
        memories=tuple(
            BriefMemory(memory_id=m.id, kind=_EXPERIENTIAL_KIND[m.kind], text=m.stored_content, created_step=m.created_step)
            for m in memories
        ),
        relations=tuple(
            BriefRelation(
                agent_id=r.to_id,
                name=r.to_name or "某人",
                gender=r.to_gender,
                trust=r.trust_objective,
                affection=r.affection_objective,
                labels=tuple(r.labels),
                summary=r.history_summary,
            )
            for r in relations
        ),
    )


def _holder_name(holder_id: str | None, *, self_id: str, spatial) -> str:
    if not holder_id or holder_id == self_id:
        return ""
    presence = spatial.visible_agents.get(holder_id)
    return (presence.identity.name if presence is not None else "") or "某人"


# How an experiential-stream memory reads to the actor.
_EXPERIENTIAL_KIND: dict[MemoryKind, BriefMemoryKind] = {
    MemoryKind.EVENT: BriefMemoryKind.EXPERIENTIAL,
    MemoryKind.INSIGHT: BriefMemoryKind.INSIGHT,
    MemoryKind.SUMMARY: BriefMemoryKind.PERIOD,
}


def _memories(memories: Iterable["Memory"], kind: BriefMemoryKind) -> tuple[BriefMemory, ...]:
    return tuple(
        BriefMemory(memory_id=m.id, kind=kind, text=m.stored_content, created_step=m.created_step)
        for m in sorted(memories, key=lambda m: m.created_step)
    )
