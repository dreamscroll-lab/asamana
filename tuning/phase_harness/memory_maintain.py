"""Maintenance (decay / compress / reflect) dry-run"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent.memory import COMPRESSION_TRIGGER_COUNT, _is_compression_candidate, _is_emotional_anchor
from agent.memory_types import MemoryKind, MemoryStream
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger

from tuning.phase_harness.common import isolate_agent, restore_traced
from tuning.phase_harness.scenario import resolve_ref, seed_memories, seed_view
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


async def run_memory_maintain(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the memory **maintenance** mechanisms for a restored world.

    Seeds memories with controlled lifecycle attributes (baseline step-0 has no aged/decayed
    memories), then drives the real production mechanism by scenario ``kind``:
      - decay   → memory_system.apply_decay(step)         (deterministic: anchor protection / rate / no-delete)
      - compress→ memory_system.compress(stream, current_step=step)  (mechanism + LLM summary)
      - reflect → reflection_engine.reflect(step)         (mechanism + LLM insight)
    Persistence shadowed (InMemoryVectorStore + _DryRunAgentStore); baseline ./data untouched.
    Zero production changes (only seed injection and dry-run isolation are scaffolding).
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    # Keep the wrapped container: reflect below needs it. With the original container those calls
    # would go to the production trace channel and miss tuning's own sink, so they'd be absent from llm_calls.
    traced_container, world, run_step, _ = await restore_traced(
        container, config, world_id, trace_sink
    )
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = list(world.agents.keys())
    scenario = scenario or {}
    kind = scenario.get("kind", "decay")
    # The maintenance "current step" is scenario-controlled (seeds carry an `age`; lifecycle
    # filters key off current_step - created_step, which a step-1 restore can't satisfy).
    step = int(scenario.get("step", 60))

    actor_id = resolve_ref(scenario["actor"], id_by_name, valid_ids)
    agent = world.agents[actor_id]
    isolate_agent(agent, name_by_id)
    seeded = await seed_memories(agent, world_id, scenario, step, id_by_name, valid_ids)

    set_log_context(world_id=world_id, agent_id=actor_id, step=str(step))
    inputs = {
        "kind": kind, "actor_name": name_by_id.get(actor_id, actor_id),
        "is_main_character": agent.is_main_character, "step": step, "seeded": seeded,
        "seed_memories": [seed_view(m) for m in agent.memory_system._entries.values()],  # noqa: SLF001
    }
    if kind == "compress":
        outputs = await _maintain_compress(agent, scenario, step, world_id)
    elif kind == "reflect":
        outputs = await _maintain_reflect(agent, traced_container, step, world_id)
    else:
        outputs = await _maintain_decay(agent, step)
    clear_log_context()

    trace_sink.record_phase(PhaseTrace(
        world_id=world_id, phase="stage.memory_maintain", step=step, agent_id=actor_id,
        inputs=inputs, outputs=outputs, timestamp=datetime.now().isoformat(),
    ))
    logger.info("tuning_memory_maintain_dry_run_complete",
                extra={"world_id": world_id, "step": step, "kind": kind, "seeded": seeded})
    return step


async def _maintain_decay(agent, step: int) -> dict[str, Any]:
    """Drive apply_decay; capture per-memory decay_score before→after + counts (no-delete invariant)."""
    ms = agent.memory_system
    before = {mid: (m.decay_score, _is_emotional_anchor(m)) for mid, m in ms._entries.items()}  # noqa: SLF001
    count_before = len(ms._entries)  # noqa: SLF001
    try:
        await ms.apply_decay(step)
        error = None
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    rows: list[dict[str, Any]] = []
    for mid, m in ms._entries.items():  # noqa: SLF001
        db, anchor = before.get(mid, (None, _is_emotional_anchor(m)))
        rows.append({
            "kind": m.kind, "stream": m.stream.value,
            "importance": round(float(m.importance), 3), "valence": round(float(m.emotion_valence), 3),
            "is_anchor": anchor, "retrieval_count": m.retrieval_count,
            "decay_before": round(db, 4) if db is not None else None,
            "decay_after": round(float(m.decay_score), 4),
        })
    return {"kind": "decay", "count_before": count_before, "count_after": len(ms._entries),  # noqa: SLF001
            "memories": rows, "error": error}


async def _maintain_compress(agent, scenario: dict, step: int, world_id: str) -> dict[str, Any]:
    """Drive compress on one stream; capture candidates / produced summaries / deletions / survivors."""
    ms = agent.memory_system
    # compress only acts on factual memory (see MemorySystem.compress); scenarios default to factual.
    stream = MemoryStream(str(scenario.get("stream", "factual")))
    before = dict(ms._entries)  # noqa: SLF001 — snapshot id→Memory for source rendering / deletion diff
    before_ids = set(before.keys())
    before_anchor_ids = {mid for mid, m in before.items() if _is_emotional_anchor(m)}
    # Mirror compress()'s own candidate filter so the trigger logic is checkable deterministically.
    candidate_count = sum(
        1 for m in before.values()
        if m.stream == stream and _is_compression_candidate(m, current_step=step)
    )
    try:
        produced = await ms.compress(stream, current_step=step)
        error = None
    except Exception as exc:  # noqa: BLE001
        produced, error = 0, str(exc)
    after_ids = set(ms._entries.keys())  # noqa: SLF001
    deleted_ids = before_ids - after_ids
    new_summaries = [m for mid, m in ms._entries.items()  # noqa: SLF001
                     if mid not in before_ids and m.kind == MemoryKind.SUMMARY]
    summaries = [{
        "stream": m.stream.value, "content": m.stored_content,
        "importance": round(float(m.importance), 3), "emotion_label": m.emotion_label,
        "source_count": len(m.source_ids),
    } for m in new_summaries]
    # Source cluster the summary stands for = the deleted originals (cap the rendering).
    source_memories = [
        {"kind": before[mid].kind, "stream": before[mid].stream.value, "content": before[mid].stored_content}
        for mid in list(deleted_ids)[:30]
    ]
    return {
        "kind": "compress", "stream": stream.value, "trigger": COMPRESSION_TRIGGER_COUNT,
        "candidates": candidate_count,
        "produced": produced, "summaries": summaries, "source_memories": source_memories,
        "deleted_count": len(deleted_ids),
        "deleted_anchor_count": len(deleted_ids & before_anchor_ids),
        "count_before": len(before_ids), "count_after": len(after_ids), "error": error,
    }


async def _maintain_reflect(agent, container, step: int, world_id: str) -> dict[str, Any]:
    """Drive reflection (mirror runtime: reflection_engine.reflect). Capture insights + grounding/depth."""
    engine = agent.reflection_engine
    if engine is None:
        from agent.reflection import ReflectionEngine
        engine = ReflectionEngine(container.llm_router, agent.memory_system, agent.personality,
                                  agent_id=agent.agent_id)
    ms = agent.memory_system
    before = dict(ms._entries)  # noqa: SLF001
    before_ids = set(before.keys())
    # Candidate pool the reflection drew on (experiential event/insight) — shown to the judge for grounding.
    source_memories = [
        {"kind": m.kind.value, "stream": m.stream.value, "content": m.stored_content}
        for m in before.values()
        if m.stream == MemoryStream.EXPERIENTIAL and m.kind in (MemoryKind.EVENT, MemoryKind.INSIGHT)
    ]
    try:
        result = await engine.reflect(step)
        error = None
    except Exception as exc:  # noqa: BLE001
        result, error = None, str(exc)
    insights = []
    for m in (result.new_insights if result else []):
        src = [ms._entries.get(sid) for sid in m.source_ids]  # noqa: SLF001
        src_kinds = [s.kind for s in src if s is not None]
        insights.append({
            "text": m.stored_content, "importance": round(float(m.importance), 3),
            "depth": m.reflection_depth, "source_count": len(m.source_ids),
            "source_kinds": src_kinds, "grounded": "event" in src_kinds,
            "valence": round(float(m.emotion_valence), 3),
        })
    return {
        "kind": "reflect", "produced": len(insights), "insights": insights,
        "source_memories": source_memories,
        "count_before": len(before_ids), "count_after": len(ms._entries), "error": error,  # noqa: SLF001
    }
