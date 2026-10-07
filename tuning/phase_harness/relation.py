"""Relation system dry-run

kind=perceive → RelationSystem.perceive (_compute_perceived emotion/memory colouring; pure rules, no LLM)
kind=evolve   → RelationEvolution.evaluate (periodic objective label/summary judgment; LLM)"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.personality import EmotionState, parse_emotion_type
from agent.relation_evolution import LABEL_EVOLUTION_LOOKBACK_STEPS, MAX_RECENT_EVENTS_PER_TARGET
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger

from tuning.phase_harness.common import isolate_agent, restore_traced
from tuning.phase_harness.scenario import resolve_ref, seed_memories, seed_view
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


def _emotion_from_spec(spec: dict) -> EmotionState:
    """Build an EmotionState with explicit intensity/valence so the colouring perceive applies is exactly controlled (for deterministic assertions)."""
    return EmotionState(
        primary=parse_emotion_type(str(spec.get("emotion_label", "neutral"))),
        intensity=float(spec.get("intensity", 0.2)),
        valence=float(spec.get("valence", 0.0)),
    )


def _relation_state_view(rel) -> dict[str, Any]:
    return {
        "labels": list(rel.labels),
        "trust": round(float(rel.trust_objective), 3),
        "affection": round(float(rel.affection_objective), 3),
        "summary": rel.history_summary or "",
        "interaction_count": rel.interaction_count,
    }


async def _apply_relation_setup(
    agent, scenario: dict, id_by_name: dict, valid_ids, name_by_id: dict
) -> dict[str, str]:
    """Set each target's baseline labels/trust/affection/summary/to_name (written to the dry-run shadow store).

    Returns {target_id: must_keep_labels echo} purely for driving convenience; the real
    must_keep/expect values are echoed through outputs.
    relation_setup entries: {target, labels?, trust?, affection?, summary?, to_name?}.
    """
    for raw in scenario.get("relation_setup", []) or []:
        tid = resolve_ref(str(raw.get("target", "")), id_by_name, valid_ids)
        rel = await agent.relation_system.get_or_create(tid)
        if "labels" in raw:
            rel.labels = [str(x).strip() for x in (raw.get("labels") or []) if str(x).strip()]
        if "trust" in raw:
            rel.trust_objective = float(raw["trust"])
        if "affection" in raw:
            rel.affection_objective = float(raw["affection"])
        if raw.get("summary"):
            rel.history_summary = str(raw["summary"])
        rel.to_name = str(raw.get("to_name") or name_by_id.get(tid, tid))
        await agent.relation_system._store.save_relation(rel)  # noqa: SLF001 — dry-run shadow store
    return {}


async def _relation_perceive(agent, scenario: dict, id_by_name, valid_ids, name_by_id) -> dict[str, Any]:
    """Drive the production RelationSystem.perceive; capture objective → perceived (after emotion and memory colouring)."""
    rows: list[dict[str, Any]] = []
    for raw in scenario.get("perceive", []) or []:
        tid = resolve_ref(str(raw.get("target", "")), id_by_name, valid_ids)
        rel = await agent.relation_system.get_or_create(tid)
        obj_trust, obj_aff = float(rel.trust_objective), float(rel.affection_objective)
        emo = _emotion_from_spec(raw)
        bias = float(raw.get("memory_bias", 0.0))
        perceived = await agent.relation_system.perceive(
            tid, emotion=emo, recent_memory_bias=bias, agent_name=name_by_id.get(tid, "")
        )
        rows.append({
            "name": name_by_id.get(tid, tid),
            "emotion": {"label": emo.primary.value, "intensity": round(emo.intensity, 3), "valence": round(emo.valence, 3)},
            "memory_bias": round(bias, 3),
            "labels": list(perceived.labels),
            "objective": {"trust": round(obj_trust, 3), "affection": round(obj_aff, 3)},
            "perceived": {"trust": round(perceived.trust, 3), "affection": round(perceived.affection, 3)},
            "expect_trust_range": raw.get("expect_trust_range"),
            "expect_affection_range": raw.get("expect_affection_range"),
        })
    return {"kind": "perceive", "perceptions": rows, "error": None}


async def _relation_evolve(agent, scenario: dict, step: int, id_by_name, valid_ids, name_by_id) -> dict[str, Any]:
    """Drive the production RelationEvolution.evaluate; capture each candidate target's before/after labels + summary and the recent evidence."""
    if agent.relation_evolution is None:
        return {"kind": "evolve", "targets": [], "applied_count": 0,
                "error": "actor 无 relation_evolution(生产 agent 恒有;此为非生产构造的最小 agent)"}

    # Collect expectations per target name (must_keep_labels / expect_no_change) for the outputs echo and deterministic checks.
    expect_by_name: dict[str, dict] = {}
    for raw in scenario.get("relation_setup", []) or []:
        tid = resolve_ref(str(raw.get("target", "")), id_by_name, valid_ids)
        nm = name_by_id.get(tid, tid)
        expect_by_name[nm] = {
            "must_keep_labels": [str(x) for x in (raw.get("must_keep_labels") or [])],
            "expect_no_change": bool(raw.get("expect_no_change", False)),
        }

    cand_ids = sorted(
        t for t in agent.memory_system.collect_recent_related_agents(
            current_step=step, lookback_steps=LABEL_EVOLUTION_LOOKBACK_STEPS)
        if t != agent.agent_id
    )
    before: dict[str, dict] = {}
    recent_by_id: dict[str, list[str]] = {}
    for tid in cand_ids:
        rel = await agent.relation_system.get_or_create(tid)
        before[tid] = _relation_state_view(rel)
        recent = agent.memory_system.list_recent_memories_mentioning(
            tid, current_step=step, lookback_steps=LABEL_EVOLUTION_LOOKBACK_STEPS)
        recent_by_id[tid] = [
            (m.stored_content or m.raw_content or "").strip()
            for m in recent[-MAX_RECENT_EVENTS_PER_TARGET:]
            if (m.stored_content or m.raw_content or "").strip()
        ]

    try:
        applied = await agent.relation_evolution.evaluate(step)
        error = None
    except Exception as exc:  # noqa: BLE001
        applied, error = [], str(exc)
    applied_by_id = {t: (labels, summary) for (t, labels, summary) in applied}

    targets: list[dict[str, Any]] = []
    for tid in cand_ids:
        rel = await agent.relation_system.get_or_create(tid)
        after = _relation_state_view(rel)
        nm = name_by_id.get(tid) or rel.to_name or tid
        ap = applied_by_id.get(tid)
        exp = expect_by_name.get(nm, {})
        targets.append({
            "name": nm,
            "recent_events": recent_by_id.get(tid, []),
            "before": before[tid],
            "after": after,
            "labels_changed": before[tid]["labels"] != after["labels"],
            "summary_changed": before[tid]["summary"] != after["summary"],
            "applied_labels": list(ap[0]) if ap else [],
            "applied_summary": ap[1] if ap else "",
            "must_keep_labels": exp.get("must_keep_labels", []),
            "expect_no_change": exp.get("expect_no_change", False),
        })
    return {"kind": "evolve", "targets": targets, "applied_count": len(applied), "error": error}


async def run_relation(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the relation subsystem for a restored world.

    kind=perceive drives RelationSystem.perceive (pure-rule _compute_perceived, no LLM); kind=evolve
    drives RelationEvolution.evaluate (LLM objective label/summary judgment,
    captured through the traced router). relation_setup sets each target's baseline
    labels/trust/affection/to_name; scene.seed_memories injects recent evidence (for evolve).
    Persistence is shadowed by _DryRunAgentStore + InMemoryVectorStore; baseline ./data is never
    written. Zero production intrusion.
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    _, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = list(world.agents.keys())
    scenario = scenario or {}
    kind = scenario.get("kind", "evolve")
    step = int(scenario.get("step", 60))

    actor_id = resolve_ref(scenario["actor"], id_by_name, valid_ids)
    agent = world.agents[actor_id]
    isolate_agent(agent, name_by_id)
    await _apply_relation_setup(agent, scenario, id_by_name, valid_ids, name_by_id)
    seeded = await seed_memories(agent, world_id, scenario, step, id_by_name, valid_ids)

    set_log_context(world_id=world_id, agent_id=actor_id, step=str(step))
    if kind == "perceive":
        outputs = await _relation_perceive(agent, scenario, id_by_name, valid_ids, name_by_id)
    else:
        outputs = await _relation_evolve(agent, scenario, step, id_by_name, valid_ids, name_by_id)
    clear_log_context()

    inputs = {
        "kind": kind, "actor_name": name_by_id.get(actor_id, actor_id),
        "is_main_character": agent.is_main_character, "step": step, "seeded": seeded,
        "relation_setup": scenario.get("relation_setup", []),
        "seed_memories": [{**seed_view(m), "content": m.stored_content} for m in agent.memory_system._entries.values()],  # noqa: SLF001
    }
    trace_sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.relation", step=step, agent_id=actor_id,
        inputs=inputs, outputs=outputs, timestamp=datetime.now().isoformat()))
    logger.info("tuning_relation_dry_run_complete",
                extra={"world_id": world_id, "step": step, "kind": kind, "seeded": seeded})
    return step
