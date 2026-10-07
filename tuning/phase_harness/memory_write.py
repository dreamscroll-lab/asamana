"""Memory write (record_event: factual + experiential + importance) dry-run"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.memory_types import Memory, MemoryStream
from agent.need import NeedType
from agent.personality import parse_emotion_type
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger

from tuning.phase_harness.common import emotion_view, isolate_agent, restore_traced
from tuning.phase_harness.scenario import apply_action_scene, resolve_ref
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


def _written_memory_view(m: Memory) -> dict[str, Any]:
    """JSON-safe view of one freshly written Memory (emotion_label coerced to its value)."""
    lbl = m.emotion_label
    return {
        "stream": m.stream.value,
        "kind": m.kind,
        "content": m.stored_content,
        "importance": round(float(m.importance), 3),
        "emotion_label": lbl.value if hasattr(lbl, "value") else str(lbl),
        "emotion_valence": round(float(m.emotion_valence), 3),
        "event_group_id": m.event_group_id,
        "related_agents": list(m.related_agents),
    }


async def _inject_relations_for(agent, scenario: dict, id_by_name: dict, name_by_id: dict, valid_ids) -> None:
    """Plant a substantive actor→target relation so the write prompts get a relation block.

    scene.relations = [{"from","to","trust"?,"affection"?,"labels"?,"history"?,"interaction_count"?}].
    Baseline relations are often blank defaults (trust 0.5 / no labels / no history / 0 interactions),
    which `_resolve_relation_context` correctly skips (no substantive relation, nothing injected) — so a scenario that wants to
    exercise relation-colored importance/experiential must seed the relationship. Must run AFTER
    isolate_agent: writes land in the dry-run shadow store, baseline ./data untouched. Test scaffolding.
    """
    for r in (scenario.get("scene", {}) or {}).get("relations", []):
        from_id = resolve_ref(str(r.get("from", "")), id_by_name, valid_ids)
        if from_id != agent.agent_id:
            continue
        to_id = resolve_ref(str(r.get("to", "")), id_by_name, valid_ids)
        if not to_id:
            continue
        rel = await agent.relation_system.get_or_create(to_id)
        if "trust" in r:
            rel.trust_objective = max(0.0, min(1.0, float(r["trust"])))
        if "affection" in r:
            rel.affection_objective = max(-1.0, min(1.0, float(r["affection"])))
        if r.get("labels"):
            rel.labels = [str(x) for x in r["labels"]]
        if "history" in r:
            rel.history_summary = str(r["history"])
        rel.interaction_count = max(int(rel.interaction_count), int(r.get("interaction_count", 1)))
        if not rel.to_name:
            rel.to_name = name_by_id.get(to_id, to_id)
        await agent.relation_system._store.save_relation(rel)  # noqa: SLF001 — dry-store shadow


async def run_memory_write(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the memory **write** path (``record_event``) for a restored world.

    Drives the real production write: importance LLM appraisal + asymmetric-write gate
    (``_should_write_experiential``) + factual write (verbatim mirror) + experiential LLM
    rewrite (first-person, personality-colored). Persistence is shadowed by
    _DryRunAgentStore + InMemoryVectorStore so the baseline world's ./data is never written.
    The event + the writer's current emotion/personality are the inputs; the produced
    factual/experiential memories + resolved importance + the write gate are captured.
    Zero production changes (only scene injection and dry-run isolation are scaffolding).
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    _, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = list(world.agents.keys())
    scenario = scenario or {}
    spec = scenario["event"]

    kind = scenario.get("kind", "self")
    actor_refs = scenario.get("actors") if kind == "contrast" else [scenario["actor"]]
    # apply_action_scene placement keys off a single "actor"; give it the first one.
    scenario.setdefault("actor", resolve_ref(actor_refs[0], id_by_name, valid_ids))
    # scene.emotion sets each writer's current emotion — the coloring prism for the
    # experiential rewrite AND the intensity that drives the asymmetric-write gate.
    await apply_action_scene(scenario, world, run_step, id_by_name)

    raw_content = str(spec.get("factual") or spec.get("raw") or "")
    dom = spec.get("dominant_need")
    dominant_need = NeedType(dom) if dom else None
    triggered_by = str(spec.get("triggered_by", "scene"))
    related_refs = spec.get("related") or []
    core_refs = spec.get("core_relations") or []

    writes: list[dict[str, Any]] = []
    if kind == "emotion_contrast":
        # Same agent, same event, different injected emotions — emotion colors the experiential
        # (and is a minor importance input). Tests: experiential shifts with mood while persona +
        # event stay constant; importance stays structurally stable (shouldn't swing with mood).
        actor_id = resolve_ref(scenario["actor"], id_by_name, valid_ids)
        primary_actor_id = actor_id
        agent = world.agents[actor_id]
        isolate_agent(agent, name_by_id)
        await _inject_relations_for(agent, scenario, id_by_name, name_by_id, valid_ids)
        related_ids = [r for r in (resolve_ref(x, id_by_name, valid_ids) for x in related_refs) if r and r != actor_id]
        core_ids = [r for r in (resolve_ref(x, id_by_name, valid_ids) for x in core_refs) if r and r != actor_id]
        for emo in scenario.get("emotions", []) or []:
            agent.personality.update_emotion(
                primary=parse_emotion_type(str(emo.get("type", "neutral"))),
                intensity=float(emo.get("intensity", 0.5)),
                valence=float(emo.get("valence", 0.0)),
                triggered_by="scene")
            set_log_context(world_id=world_id, agent_id=actor_id, step=str(run_step))
            writes.append(await _write_one_event(
                agent, name_by_id=name_by_id, run_step=run_step, raw_content=raw_content,
                experiential_seed=spec.get("experiential"), dominant_need=dominant_need,
                triggered_by=triggered_by, related_ids=related_ids, core_ids=core_ids, world_id=world_id))
        clear_log_context()
    else:
        primary_actor_id = resolve_ref(actor_refs[0], id_by_name, valid_ids)
        for ref in actor_refs:
            actor_id = resolve_ref(ref, id_by_name, valid_ids)
            agent = world.agents[actor_id]
            isolate_agent(agent, name_by_id)
            await _inject_relations_for(agent, scenario, id_by_name, name_by_id, valid_ids)
            related_ids = [r for r in (resolve_ref(x, id_by_name, valid_ids) for x in related_refs) if r and r != actor_id]
            core_ids = [r for r in (resolve_ref(x, id_by_name, valid_ids) for x in core_refs) if r and r != actor_id]
            set_log_context(world_id=world_id, agent_id=actor_id, step=str(run_step))
            writes.append(await _write_one_event(
                agent, name_by_id=name_by_id, run_step=run_step, raw_content=raw_content,
                experiential_seed=spec.get("experiential"), dominant_need=dominant_need,
                triggered_by=triggered_by, related_ids=related_ids, core_ids=core_ids, world_id=world_id))
        clear_log_context()

    shared = {"kind": kind, "event": raw_content,
              "dominant_need": dominant_need.value if dominant_need else None,
              "triggered_by": triggered_by}
    if kind == "contrast":
        inputs = {**shared, "actors": [w["persona"]["actor_name"] for w in writes]}
        outputs = {"kind": "contrast", "event": raw_content, "writes": writes}
    elif kind == "emotion_contrast":
        inputs = {**shared, "actor_name": writes[0]["persona"]["actor_name"] if writes else "",
                  "dominant_need_label": writes[0]["dominant_need_label"] if writes else "",
                  "emotions": [w["persona"]["current_emotion"] for w in writes]}
        outputs = {"kind": "emotion_contrast", "event": raw_content, "writes": writes}
    else:
        w = writes[0]
        inputs = {**shared, **w["persona"], "dominant_need_label": w["dominant_need_label"],
                  "long_term_goals": w["long_term_goals"],
                  "related_people": w["related_people"], "relation_context": w["relation_context"]}
        outputs = {"kind": "write", "importance": w["importance"],
                   "experiential_written": w["experiential"] is not None,
                   "write_gate": w["write_gate"], "factual": w["factual"],
                   "experiential": w["experiential"], "error": w["error"]}
    trace_sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.memory_write", step=run_step, agent_id=primary_actor_id,
        inputs=inputs, outputs=outputs, timestamp=datetime.now().isoformat(),
    ))
    logger.info("tuning_memory_write_dry_run_complete",
                extra={"world_id": world_id, "step": run_step, "kind": kind, "writers": len(writes)})
    return run_step


def _persona_view(agent, name: str) -> dict[str, Any]:
    soul = agent.personality.soul
    return {
        "actor_name": name,
        "is_main_character": agent.is_main_character,
        "core_traits": list(getattr(soul, "core_traits", [])),
        "core_values": list(getattr(soul, "core_values", [])),
        "self_image": (getattr(soul, "self_image", "") or "")[:200],
        "life_goal": getattr(soul, "life_goal", "") or "",
        "current_emotion": emotion_view(agent.personality.state.emotion),
    }


async def _write_one_event(
    agent, *, name_by_id: dict, run_step: int, raw_content: str, experiential_seed,
    dominant_need, triggered_by: str, related_ids: list[str], core_ids: list[str], world_id: str,
) -> dict[str, Any]:
    """Drive record_event for one agent; capture the produced memories + write gate + persona."""
    # Mirror production agent._apply_feedback: resolve {id: name} + relation-context block,
    # and the agent-specific need label fed to the importance prompt (personality.need_label —
    # not the generic Maslow description, so the same NeedType reads differently per persona).
    related_names, relation_ctx = await agent._event_people_context(related_ids)  # noqa: SLF001
    dominant_need_label = (
        f"{dominant_need.value}（{agent.personality.need_label(dominant_need)}）"
        if dominant_need is not None else ""
    )
    gate: dict[str, Any] = {
        "is_main_character": agent.is_main_character,
        "emotion_intensity": round(float(agent.personality.state.emotion.intensity), 3),
        "has_core_relation": any(r in core_ids for r in related_ids),
    }
    before_mem = set(agent.memory_system._entries.keys())  # noqa: SLF001
    try:
        await agent.memory_system.record_event(
            current_step=run_step,
            raw_content=raw_content,
            experiential_content=str(experiential_seed) if experiential_seed else raw_content,
            personality=agent.personality,
            related_agents=related_names,
            related_relations_text=relation_ctx,
            triggered_by=triggered_by,
            dominant_need_label=dominant_need_label,
        )
        error = None
    except Exception as exc:  # noqa: BLE001 — one writer's failure must not abort the run
        logger.warning("memory_write_failed",
                       extra={"world_id": world_id, "actor": agent.agent_id, "error": str(exc)})
        error = str(exc)
    new = [m for mid, m in agent.memory_system._entries.items() if mid not in before_mem]  # noqa: SLF001
    factual = next((m for m in new if m.stream == MemoryStream.FACTUAL), None)
    experiential = next((m for m in new if m.stream == MemoryStream.EXPERIENTIAL), None)
    gate["importance"] = round(float(factual.importance), 3) if factual is not None else None
    return {
        "persona": _persona_view(agent, name_by_id.get(agent.agent_id, agent.agent_id)),
        "dominant_need_label": dominant_need_label,
        "long_term_goals": list(agent.personality.state.long_term_goals),
        "related_people": [related_names[r] for r in related_names],
        "relation_context": relation_ctx,
        "importance": gate["importance"],
        "write_gate": gate,
        "factual": _written_memory_view(factual) if factual is not None else None,
        "experiential": _written_memory_view(experiential) if experiential is not None else None,
        "error": error,
    }
