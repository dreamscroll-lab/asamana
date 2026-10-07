"""Turning a scenario's declarative setup into world and agent state: references, arrivals,
the action scene, and seeded memories."""

from __future__ import annotations

from typing import Any

from agent.memory import _is_emotional_anchor
from agent.memory_types import Memory, MemoryStream
from agent.motivation import ExternalDriveType, ExternalGoal
from agent.need import NeedType
from agent.personality import parse_emotion_type
from core.interfaces.action import ErrandOrder
from core.interfaces.message import Message
from core.interfaces.perception import AmbientEvent, Broadcast, BroadcastType
from core.interfaces.urgency import Urgency, parse_urgency
from core.interfaces.condition import BodyCondition
from engine.presence import attach_presence
from world import World
from world.models import EntityPresence, NpcSeed, WorldEntity, WorldEntityType


def spatials_for(world: World, run_step: int, world_time_label: str) -> dict:
    """Build each active agent's spatial perception with visible-agent names filled."""
    spatials = {}
    for aid, agent in world.agents.items():
        if not agent.is_active:
            continue
        sp = world.environment.spatial_for(agent_id=aid, step=run_step, world_time=world_time_label)
        attach_presence(
            sp, directory=world.directory, agents=world.agents,
            environment=world.environment,
        )
        spatials[aid] = sp
    return spatials


def resolve_ref(ref: str, id_by_name: dict, valid_ids) -> str:
    """Resolve a scenario actor reference (agent name or id) to an agent id."""
    ref = str(ref or "")
    if ref in valid_ids:
        return ref
    return id_by_name.get(ref, ref)


def broadcasts_from_scenario(scenario: dict | None, step: int) -> list[Broadcast]:
    """scenario.broadcasts = [{"content", "severity"?, "location_scope"?, "source"?}]"""
    out: list[Broadcast] = []
    for b in (scenario or {}).get("broadcasts", []):
        content = str(b.get("content", "")).strip()
        if not content:
            continue
        out.append(Broadcast(
            content=content,
            source=str(b.get("source", "system")),
            broadcast_type=BroadcastType.WORLD_EVENT,
            deliver_step=step,
            location_scope=b.get("location_scope"),
            severity=str(b.get("severity", "medium")),
        ))
    return out


def inboxes_from_scenario(
    scenario: dict | None, world_id: str, step: int, agent_ids, name_by_id: dict
) -> dict[str, list[Message]]:
    """scenario.messages = [{"to", "from"?, "content", "urgency"?}] (to/from by name or id)."""
    id_by_name = {v: k for k, v in name_by_id.items()}
    inboxes: dict[str, list[Message]] = {aid: [] for aid in agent_ids}
    for i, m in enumerate((scenario or {}).get("messages", [])):
        content = str(m.get("content", "")).strip()
        to_id = resolve_ref(m.get("to") or m.get("recipient") or "", id_by_name, agent_ids)
        if not content or to_id not in inboxes:
            continue
        from_id = resolve_ref(m.get("from") or m.get("sender") or "narrator", id_by_name, agent_ids)
        inboxes[to_id].append(Message(
            id=f"inj-msg-{step}-{i}",
            world_id=world_id,
            sender_id=from_id,
            content=content,
            recipients=[to_id],
            location_scope=None,
            deliver_step=step,
            created_step=step,
            sender_name=name_by_id.get(from_id, from_id),
            # In a script, from defaults to narrator, which isn't a person one can form a relation with. The
            # sender declares this (see ``Message.sender_is_agent``); otherwise the recipient would start a relation with the narration.
            sender_is_agent=from_id in inboxes,
            urgency=parse_urgency(str(m.get("urgency", "normal"))) or Urgency.NORMAL,
        ))
    return inboxes


def apply_ambient_from_scenario(
    scenario: dict | None, spatials: dict, agent_ids, name_by_id: dict
) -> None:
    """scenario.ambient = [{"content", "strength"?, "location"?, "agent"?, "exclude"?}].

    Appends AmbientEvents into matching agents' spatials in place. No target → global.
    ``exclude`` (agent name/id) drops that agent even if otherwise matched — used to keep
    a location-scoped event off its own actor (e.g. a bystander-reaction test).
    """
    id_by_name = {v: k for k, v in name_by_id.items()}
    for a in (scenario or {}).get("ambient", []):
        content = str(a.get("content", "")).strip()
        if not content:
            continue
        strength = a.get("strength")
        ev = AmbientEvent(content=content, strength=float(strength) if strength is not None else None)
        agent_ref = a.get("agent")
        target_aid = resolve_ref(agent_ref, id_by_name, agent_ids) if agent_ref else None
        loc = a.get("location")
        exclude_ref = a.get("exclude")
        exclude_aid = resolve_ref(exclude_ref, id_by_name, agent_ids) if exclude_ref else None
        for aid, sp in spatials.items():
            if exclude_aid and aid == exclude_aid:
                continue
            if (target_aid and aid == target_aid) or (loc and sp.location_id == loc) or (not agent_ref and not loc):
                sp.ambient_events.append(ev)


def presence_name(sp, agent_id: str) -> str:
    """Display name of someone in the perception packet. Unrecognized → "某人"; never a raw id (that's a layer leak)."""
    presence = sp.visible_agents.get(agent_id)
    return (presence.identity.name if presence else "") or "某人"


def external_goals_from_scenario(
    scenario: dict | None, agent_ids, name_by_id: dict
) -> dict[str, list[ExternalGoal]]:
    """scenario.external_goals = [{"agent"/"to", "text", "urgency"?, "drive_type"?, "source"?, "need"?}].

    ``need`` (NeedType value) → ExternalGoal.related_need — mirrors what WorldPressureEvaluator
    populates in production; only with a related_need + urgency≥HIGH does the goal override the
    dominant need (MotivationBlender.resolve_dominant_need), so scenarios set it to exercise that path.
    """
    id_by_name = {v: k for k, v in name_by_id.items()}
    out: dict[str, list[ExternalGoal]] = {aid: [] for aid in agent_ids}
    for g in (scenario or {}).get("external_goals", []):
        text = str(g.get("text", "")).strip()
        to_id = resolve_ref(g.get("agent") or g.get("to") or "", id_by_name, agent_ids)
        if not text or to_id not in out:
            continue
        try:
            drive = ExternalDriveType(str(g.get("drive_type", "event")))
        except ValueError:
            drive = ExternalDriveType.EVENT
        related_need = None
        if g.get("need"):
            try:
                related_need = NeedType(str(g["need"]))
            except ValueError:
                related_need = None
        src = g.get("source")
        source_id = resolve_ref(src, id_by_name, agent_ids) if src else "world"
        out[to_id].append(ExternalGoal(
            text=text,
            source_id=source_id,
            urgency=parse_urgency(str(g.get("urgency", "normal"))) or Urgency.NORMAL,
            drive_type=drive,
            related_need=related_need,
        ))
    return out


def resolve_location_ref(ref: str, env) -> str | None:
    if not ref:
        return None
    # Resolve against the location roster (id or name); if unrecognized, keep the literal and let feasibility reject it.
    return env.space.resolve(ref) or ref


async def apply_action_scene(scenario: dict, world: World, run_step: int, id_by_name: dict) -> None:
    """Place co-located agents + register items into the ENV (execution needs real
    co-location, not just spatial.visible_agents), and apply persona-state overrides + seed
    memories. Test scaffolding only — no production change."""
    env = world.environment
    valid_ids = list(world.agents.keys())
    actor_ref = scenario.get("actor") or (scenario.get("actions", [{}])[0].get("actor", ""))
    actor_id = resolve_ref(actor_ref, id_by_name, valid_ids)
    base_loc = env.get_body_location(actor_id)
    scene = scenario.get("scene", {}) or {}

    for ref in scene.get("colocate", []):
        tid = resolve_ref(ref, id_by_name, valid_ids)
        if tid in world.agents:
            env.place_agent(agent_id=tid, location_id=base_loc)

    for i, it in enumerate(scene.get("items", [])):
        try:
            etype = WorldEntityType(it.get("type", "item"))
        except ValueError:
            etype = WorldEntityType.ITEM
        # placement uses discriminated coordinates (presence + presence_ref). Don't pass location_id=: it's
        # a read-only property, and any scenario with scene.items would raise TypeError.
        # holder: put this thing in someone's hands. "self" means the actor (which is what delivery scenarios need).
        holder_ref = it.get("holder")
        holder = (
            actor_id if holder_ref in ("self", "me")
            else resolve_ref(str(holder_ref), id_by_name, valid_ids)
        ) if holder_ref else None
        env.register_entity(WorldEntity(
            entity_id=f"act-item-{i}", name=str(it.get("name", "")), entity_type=etype,
            state=str(it.get("state", "intact")),
            description=str(it.get("desc", "")),
            presence=EntityPresence.HELD if holder else EntityPresence.AT_LOCATION,
            presence_ref=holder or base_loc,
            is_takeable=bool(it.get("takeable", False)),
        ))

    # The no-cognition tier: a scenario that sends someone on an errand needs that person to exist first. Name order, so action.bearer can refer to them by name.
    for spec in scene.get("npcs", []) or []:
        env.spawn_npc(
            NpcSeed(
                name=str(spec.get("name", "")),
                gender=str(spec.get("gender", "")),
                age=int(spec.get("age", 30)),
                description=str(spec.get("description", "")),
            ),
            location_id=resolve_location_ref(str(spec.get("at", "")), env) or base_loc,
        )
        if (busy := spec.get("busy_for")) :
            # Pre-set as already running an errand for someone else; the judge has to weigh it (same reason as conditions above).
            npc = env.all_npcs()[-1]
            env.assign_errand(
                ErrandOrder(npc.npc_id, base_loc, ()),
                requester_id=resolve_ref(str(busy), id_by_name, valid_ids),
            )

    for ref, v in (scene.get("vitality") or {}).items():
        aid = resolve_ref(ref, id_by_name, valid_ids)
        if aid in world.agents:
            # personality.state returns a defensive copy — mutate via restore_state so the
            # override lands on the live _state (direct `.state.vitality = v` would no-op).
            p = world.agents[aid].personality
            st = p.state
            st.vitality = max(0.0, min(1.0, float(v)))
            p.restore_state(st)

    for ref, text in (scene.get("conditions") or {}).items():
        aid = resolve_ref(ref, id_by_name, valid_ids)
        if aid in world.agents:
            # The scenario pre-sets someone as restrained so the judge has to weigh it. The field exists
            # because a judge that can't see it rates resistance from a man bound and kneeling as if he were free.
            world.agents[aid].personality.set_condition(
                BodyCondition(description=str(text), since_step=0),
            )

    for ref, emo in (scene.get("emotion") or {}).items():
        aid = resolve_ref(ref, id_by_name, valid_ids)
        if aid in world.agents and isinstance(emo, dict):
            world.agents[aid].personality.update_emotion(
                primary=parse_emotion_type(str(emo.get("type", "neutral"))),
                intensity=float(emo.get("intensity", 0.5)),
                valence=float(emo.get("valence", 0.0)),
                triggered_by="scene",
            )

    # What happened here before: the only intelligence source for covert adjudication. Without it every
    # covert scenario has nothing to find and is ruled not achieved, so the suite can't tell good
    # verdicts from bad.
    happenings = scene.get("happenings") or []
    for i, text in enumerate(happenings):
        env.record_happening(
            location_id=base_loc, outcome=str(text),
            # Laid back in listed order, the last one a beat before this run starts (matching the runtime window).
            step=run_step - len(happenings) + i,
            # Empty member set: these things were done by others in the scenario, and the actor shouldn't count as already holding them.
            actor_ids=(),
        )

    # Best-effort memory seeding (only social TALK retrieves vector memories; relation +
    # persona + scene remain the primary inputs if retrieval comes up empty).
    for e in scene.get("memories", []) or []:
        aid = resolve_ref(e.get("agent") or e.get("to") or "", id_by_name, valid_ids)
        agent = world.agents.get(aid)
        if agent is None:
            continue
        for c in (e.get("factual") or []):
            try:
                await agent.memory_system.seed_factual_memory(current_step=run_step, raw_content=str(c))
            except Exception:  # noqa: BLE001 — seeding is best-effort scaffolding
                pass


def _seed_memory(agent_id: str, world_id: str, content: str, spec: dict, created: int, idx: str) -> Memory:
    """Build one seeded Memory with caller-controlled lifecycle attributes."""
    stream = MemoryStream(str(spec.get("stream", "experiential")))
    kind = str(spec.get("kind", "event"))
    is_fac = stream == MemoryStream.FACTUAL
    return Memory(
        id=f"seed-{kind}-{stream.value}-{idx}-{created}",
        agent_id=agent_id, stream=stream,
        raw_content=content, stored_content=content,
        importance=max(0.0, min(1.0, float(spec.get("importance", 0.5)))),
        created_step=created,
        metadata={"world_id": world_id},
        emotion_valence=0.0 if is_fac else float(spec.get("valence", 0.0)),
        emotion_label="objective" if is_fac else str(spec.get("emotion_label", "neutral")),
        related_agents=list(spec.get("_related_ids", [])),
        decay_score=max(0.0, min(1.0, float(spec.get("decay_score", 1.0)))),
        retrieval_count=int(spec.get("retrieval_count", 0)),
        kind=kind,  # type: ignore[arg-type]
        event_group_id=(f"seed-grp-{idx}-{created}" if kind == "event" else None),
        reflection_depth=int(spec.get("reflection_depth", 0)),
    )


async def seed_memories(agent, world_id: str, scenario: dict, step: int, id_by_name: dict, valid_ids) -> int:
    """Persist scenario.scene.seed_memories into the (isolated) memory store.

    Each spec: {stream, kind?, importance?, decay_score?, valence?, age?, related?, retrieval_count?,
    reflection_depth?} + either ``content`` (×``count``) or ``variants`` (list of distinct contents).
    age → created_step = step - age (lifecycle filters key off current_step - created_step).
    ``age_span: [lo, hi]`` replaces a single age: it spreads this spec's memories' created_step over
    [step-hi, step-lo] (first oldest, last newest), so a cluster really spans a period and a
    compressed summary's time span renders as "start to end" rather than a single moment (matching
    real data where something keeps happening over a stretch).
    """
    seeded = 0
    for si, raw in enumerate((scenario.get("scene", {}) or {}).get("seed_memories", []) or []):
        spec = dict(raw)
        spec["_related_ids"] = [r for r in (resolve_ref(x, id_by_name, valid_ids) for x in (spec.get("related") or [])) if r]
        contents = spec.get("variants") or [spec.get("content", "")] * int(spec.get("count", 1))
        span = spec.get("age_span")
        n = len(contents)
        for ci, content in enumerate(contents):
            if isinstance(span, (list, tuple)) and len(span) == 2:
                lo, hi = float(span[0]), float(span[1])
                frac = ci / (n - 1) if n > 1 else 0.0  # first oldest (hi) → last newest (lo)
                created = max(0, step - int(round(hi - (hi - lo) * frac)))
            else:
                created = max(0, step - int(spec.get("age", 0)))
            mem = _seed_memory(agent.agent_id, world_id, str(content), spec, created, f"{si}-{ci}")
            await agent.memory_system._persist(mem)  # noqa: SLF001 — writes to dry-run shadow store
            seeded += 1
    return seeded


def seed_view(m: Memory) -> dict[str, Any]:
    return {
        "kind": m.kind, "stream": m.stream.value,
        "importance": round(float(m.importance), 3),
        "valence": round(float(m.emotion_valence), 3),
        "decay_score": round(float(m.decay_score), 4),
        "is_anchor": _is_emotional_anchor(m),
        "created_step": m.created_step, "retrieval_count": m.retrieval_count,
    }
