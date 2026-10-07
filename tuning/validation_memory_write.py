"""memory write stage validation suite: scenarios + deterministic checks + LLM judge.

Restores a built world and, per scenario, drives the **real production write path**
(``memory_system.record_event``): importance LLM appraisal + asymmetric-write gate +
factual write (verbatim mirror) + experiential LLM rewrite. Captures the produced
memories + resolved importance + the write gate, then applies structural deterministic
checks + an LLM judge (syntax + semantics of the write).

Persistence isolation: the harness wraps the agent in _DryRunAgentStore (read-through,
write-capture) + InMemoryVectorStore, so writes go to in-memory shadows and the baseline
world's ./data stays clean.

Deterministic checks (always run) are hard tripwires: FACTUAL == the event text verbatim (no LLM
rewrite) with emotion_label=objective and valence=0; importance∈[0,1]; experiential prose has
no agent id and no "第N步/steps"; both streams share event_group_id; experiential emotion is
bound to the mood at write time; the asymmetric-write gate is consistent
(_should_write_experiential is honoured faithfully); no write exceptions. The judge carries the
semantics, using ``llm.judge``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_llm import create_judge
from tuning.judge_memory_write import judge_memory_write
from tuning.phase_harness.memory_write import run_memory_write
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import STEP_LEAK, focus_of, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "memory_write.json"
_CRITERIA = ("fields", "experiential", "importance")


# ---------------------------------------------------------------------------
# Deterministic checks (structural tripwire)
# ---------------------------------------------------------------------------

def _expected_experiential(gate: dict) -> bool | None:
    """Recompute _should_write_experiential from observed inputs (None if unknowable)."""
    inten = gate.get("emotion_intensity")
    imp = gate.get("importance")
    if not isinstance(inten, (int, float)) or not isinstance(imp, (int, float)):
        return None
    if gate.get("is_main_character"):
        return inten >= 0.4 or imp >= 0.4 or bool(gate.get("has_core_relation"))
    return inten >= 0.6


def _check_one_write(event: str, w: dict, cur_emotion: dict, agent_ids: set[str], tag: str = "") -> list[str]:
    """Structural tripwire for one writer's factual + experiential + importance + gate."""
    v: list[str] = []
    pre = f"[{tag}] " if tag else ""
    if w.get("error"):
        v.append(f"{pre}写入异常: {w['error']}")

    fac = w.get("factual")
    if fac is None:
        v.append(f"{pre}未写入 FACTUAL（factual 应无条件写入）")
    else:
        if str(fac.get("content")) != event:
            v.append(f"{pre}FACTUAL 偏离事件原文（factual 不得被 LLM 改写）")
        if fac.get("emotion_label") != "objective":
            v.append(f"{pre}FACTUAL emotion_label 非 objective={fac.get('emotion_label')!r}")
        if abs(float(fac.get("emotion_valence") or 0.0)) > 1e-9:
            v.append(f"{pre}FACTUAL emotion_valence 非 0={fac.get('emotion_valence')!r}")

    imp = w.get("importance")
    if not isinstance(imp, (int, float)) or not (0.0 <= float(imp) <= 1.0):
        v.append(f"{pre}importance 越界/缺失={imp!r}")

    exp = w.get("experiential")
    if exp is not None:
        txt = str(exp.get("content") or "")
        for aid in agent_ids:
            if aid and aid in txt:
                v.append(f"{pre}experiential 泄漏 agent id「{aid}」→「{txt[:30]}…」")
                break
        if STEP_LEAK.search(txt):
            v.append(f"{pre}experiential 出现 step(『第N步/N步/steps』)→「{txt[:30]}…」")
        if fac is not None and exp.get("event_group_id") != fac.get("event_group_id"):
            v.append(f"{pre}双流 event_group_id 未绑定")
        if cur_emotion and exp.get("emotion_label") != cur_emotion.get("primary"):
            v.append(f"{pre}experiential 情绪绑定不符(标签{exp.get('emotion_label')!r} vs 当前{cur_emotion.get('primary')!r})")

    # Asymmetric-write gate consistency: whether experiential was written must match _should_write_experiential's verdict.
    expected = _expected_experiential(w.get("write_gate", {}) or {})
    if expected is not None and (exp is not None) != expected:
        v.append(f"{pre}非对称写入闸门不一致(实际写入={exp is not None}, 应={expected})")
    return v


def _deterministic_checks(phase: dict, agent_ids: set[str]) -> dict[str, Any]:
    inputs = phase.get("inputs", {}) or {}
    outputs = phase.get("outputs", {}) or {}
    event = str(inputs.get("event") or "")
    v: list[str] = []
    if outputs.get("writes") is not None:  # contrast / emotion_contrast — per-writer checks
        for w in outputs.get("writes", []) or []:
            cur = (w.get("persona") or {}).get("current_emotion") or {}
            v += _check_one_write(event, w, cur, agent_ids, tag=(w.get("persona") or {}).get("actor_name", ""))
    else:
        v += _check_one_write(event, outputs, inputs.get("current_emotion") or {}, agent_ids)
    return {"passed": not v, "field_violations": v}


# ---------------------------------------------------------------------------
# Per-scenario run
# ---------------------------------------------------------------------------

async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, agent_ids: set[str], out_dir: Path, judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_memory_write(container, config, world_id, scenario=scenario_meta, sink=sink,
                           trace_dir=str(scenario_dir))

    phase = next((p for p in sink.phases if p.phase == "stage.memory_write"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    phase_dict = {"inputs": inputs, "outputs": outputs}

    det = _deterministic_checks(phase_dict, agent_ids)

    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_memory_write(
        judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
        det_findings=det, judge_scene=judge_scene)

    # write-path cognition calls (importance appraisal + experiential rewrite) — every
    # non-judge call captured by the traced router.
    write_calls = [
        {"scene": c.scene, "agent_id": c.agent_id, "prompt": c.prompt_messages, "response": c.response_content}
        for c in sink.llm_calls if c.scene != judge_scene.value
    ]
    # The quality-judge's own LLM call (prompt + raw verdict) — surfaced so the score is
    # auditable in web, not just the parsed numbers.
    judge_calls = [
        {"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
        for c in sink.llm_calls if c.scene == judge_scene.value
    ]

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "write.json", phase_dict)
    write_json(scenario_dir / "prompt.json",
          {"write_calls": write_calls, "judge_call": judge_calls[-1] if judge_calls else None})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name,
        "criteria_focus": scenario_meta.get("criteria_focus", []),
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_memory_write(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full memory-write validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())

    out_dir = Path(trace_dir) / world_id / "validation" / "memory_write"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(
            container, config, world_id, meta, agent_ids=agent_ids, out_dir=out_dir,
            judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "memory_write", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md,
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="memory_write", avg_label="三项均值",
        columns=[
            ("重点", focus_of),
            ("fields", lambda r: r["scores"]["fields"]),
            ("experiential", lambda r: r["scores"]["experiential"]),
            ("importance", lambda r: r["scores"]["importance"]),
        ],
    )
