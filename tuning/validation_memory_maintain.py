"""memory maintenance stage validation: scenarios + deterministic checks + LLM judge.

Restores a built world and, per scenario, seeds memories with controlled lifecycle attributes,
then drives the **real production maintenance mechanism** by kind:
  - decay    → memory_system.apply_decay   (deterministic ONLY — anchor protection / rate / no-delete)
  - compress → memory_system.compress      (deterministic mechanism + LLM summary semantic judge)
  - reflect  → reflection_engine.reflect    (deterministic mechanism + LLM insight semantic judge)

Persistence isolation: InMemoryVectorStore + _DryRunAgentStore; baseline ./data stays clean.

Deterministic checks (always run) are hard tripwires:
  decay    — anchors (CRITICAL / insight / high-|valence| experiential) keep their decay_score;
             non-anchors decay strictly, at the configured rate; decay∈(0,1]; count unchanged
             (decay never deletes).
  compress — below the candidate threshold: no trigger, no source deletion. When triggered: the
             summary's kind/emotion_label/importance are valid, sources are deleted, anchors are
             never in the deletion set, summary prose has no id/step; non-candidates aren't
             deleted.
  reflect  — insights have kind=insight, every source set has ≥1 event (grounded), depth≤MAX,
             importance∈[base,0.9], text has no id/step.
The judge only carries compress/reflect semantics; decay doesn't call the judge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


from agent.reflection import INSIGHT_BASE_IMPORTANCE, MAX_INSIGHT_IMPORTANCE, MAX_REFLECTION_DEPTH
from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_llm import create_judge
from tuning.judge_memory_maintain import judge_memory_maintain
from tuning.phase_harness.memory_maintain import run_memory_maintain
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import STEP_LEAK, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "memory_maintain.json"
_CRITERIA = ("fidelity", "voice")
# apply_decay base rates (mirror agent.memory.apply_decay for an expected-value cross-check).
_DECAY_BASE = 0.95
_DECAY_EXPERIENTIAL_BONUS = 0.03
_DECAY_CONSOLIDATION_CAP = 0.04


# ---------------------------------------------------------------------------
# Deterministic checks (per kind)
# ---------------------------------------------------------------------------

def _expected_decay_rate(stream: str, retrieval_count: int) -> float:
    rate = _DECAY_BASE + min(retrieval_count * 0.01, _DECAY_CONSOLIDATION_CAP)
    if stream == "experiential":
        rate = min(rate + _DECAY_EXPERIENTIAL_BONUS, 0.99)
    return rate


def _check_decay(out: dict) -> list[str]:
    v: list[str] = []
    if out.get("count_before") != out.get("count_after"):
        v.append(f"decay 删除了记忆(应永不删除): {out.get('count_before')}→{out.get('count_after')}")
    for m in out.get("memories", []) or []:
        db, da = m.get("decay_before"), m.get("decay_after")
        if not isinstance(da, (int, float)) or not (0.0 < da <= 1.0):
            v.append(f"decay_after 越界={da!r}")
            continue
        if m.get("is_anchor"):
            if db is not None and abs(da - db) > 1e-9:
                v.append(f"锚点被衰减({m.get('kind')}/{m.get('stream')}): {db}→{da}")
        else:
            if db is not None:
                if da >= db:  # non-anchor must strictly decay
                    v.append(f"非锚点未衰减({m.get('kind')}/{m.get('stream')}): {db}→{da}")
                exp = round(db * _expected_decay_rate(m.get("stream", ""), int(m.get("retrieval_count", 0))), 4)
                if abs(da - exp) > 0.02:
                    v.append(f"衰减率不符({m.get('stream')}): {db}→{da}, 期望≈{exp}")
    return v


def _check_compress(out: dict) -> list[str]:
    v: list[str] = []
    if out.get("error"):
        v.append(f"compress 异常: {out['error']}")
    cand, trigger, produced = out.get("candidates", 0), out.get("trigger", 30), out.get("produced", 0)
    if cand < trigger:
        # below threshold: must be a no-op (no summaries, no deletions)
        if produced or out.get("deleted_count"):
            v.append(f"候选不足({cand}<{trigger})却触发了压缩(produced={produced}, deleted={out.get('deleted_count')})")
    if out.get("deleted_anchor_count"):
        v.append(f"删除集含锚点 {out['deleted_anchor_count']} 条(锚点绝不可被压缩删除)")
    for s in out.get("summaries", []) or []:
        if s.get("emotion_label") != "summary":
            v.append(f"summary emotion_label 非 summary={s.get('emotion_label')!r}")
        imp = s.get("importance")
        if not isinstance(imp, (int, float)) or not (0.0 <= imp <= 0.95):
            v.append(f"summary importance 越界={imp!r}")
        if s.get("source_count", 0) < 3:
            v.append(f"summary 源数<3(簇下限)={s.get('source_count')}")
        txt = str(s.get("content") or "")
        if STEP_LEAK.search(txt):
            v.append(f"summary 出现 step→「{txt[:30]}…」")
    if produced and out.get("deleted_count", 0) <= 0:
        v.append("产出了 summary 却未删除任何源(删源应在摘要落库后发生)")
    return v


def _check_reflect(out: dict, expect_empty: bool = False) -> list[str]:
    v: list[str] = []
    if out.get("error"):
        v.append(f"reflect 异常: {out['error']}")
    # Empty > fake: a flat scenario should stay empty. Producing an insight breaks "empty when it should be" (a sign of forcing or fabrication).
    if expect_empty and out.get("produced"):
        v.append(f"平淡素材却产出 {out['produced']} 条洞察（该留空而硬挤，疑似虚构；空 > 假）")
    for ins in out.get("insights", []) or []:
        if not ins.get("grounded"):
            v.append("insight 未接地(source 不含任何 event)")
        d = ins.get("depth")
        if not isinstance(d, int) or not (1 <= d <= MAX_REFLECTION_DEPTH):
            v.append(f"insight depth 越界={d!r}(应 1..{MAX_REFLECTION_DEPTH})")
        imp = ins.get("importance")
        if not isinstance(imp, (int, float)) or not (INSIGHT_BASE_IMPORTANCE - 1e-9 <= imp <= MAX_INSIGHT_IMPORTANCE + 1e-9):
            v.append(f"insight importance 越界={imp!r}(应 [{INSIGHT_BASE_IMPORTANCE},{MAX_INSIGHT_IMPORTANCE}])")
        txt = str(ins.get("text") or "")
        if STEP_LEAK.search(txt):
            v.append(f"insight 出现 step→「{txt[:30]}…」")
    return v


def _deterministic_checks(phase: dict, agent_ids: set[str], *, expect_empty: bool = False) -> dict[str, Any]:
    out = phase.get("outputs", {}) or {}
    kind = out.get("kind", "decay")
    if kind == "reflect":
        v = _check_reflect(out, expect_empty)
    else:
        v = {"decay": _check_decay, "compress": _check_compress}.get(kind, _check_decay)(out)
    # narrative-layer id leak guard across any produced prose (summaries / insights).
    for prose in [s.get("content", "") for s in out.get("summaries", []) or []] + \
                 [i.get("text", "") for i in out.get("insights", []) or []]:
        for aid in agent_ids:
            if aid and aid in str(prose):
                v.append(f"产物泄漏 agent id「{aid}」")
                break
    return {"passed": not v, "field_violations": v}


# ---------------------------------------------------------------------------
# Per-scenario run
# ---------------------------------------------------------------------------

async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, agent_ids: set[str], out_dir: Path, judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    kind = scenario_meta.get("kind", "decay")
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_memory_maintain(container, config, world_id, scenario=scenario_meta, sink=sink,
                              trace_dir=str(scenario_dir))

    phase = next((p for p in sink.phases if p.phase == "stage.memory_maintain"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    expect_empty = bool(scenario_meta.get("expect_empty"))
    det = _deterministic_checks({"inputs": inputs, "outputs": outputs}, agent_ids, expect_empty=expect_empty)

    # Judge only when there's a semantic product: decay is mechanism-only; a below-threshold
    # compress (no summaries) or an empty reflect (no insights) is also a mechanism-only no-op.
    has_semantic = (
        (kind == "compress" and outputs.get("summaries"))
        or (kind == "reflect" and outputs.get("insights"))
    )
    if not has_semantic:
        if kind == "reflect" and expect_empty:
            note = "按预期留空（这段平淡，无可洞察——空 > 假，正确）"
        else:
            note = {"decay": "decay 纯机制(确定性),不评语义",
                    "compress": "未产出 summary(候选不足/无可压缩),纯机制验证",
                    "reflect": "未产出 insight,纯机制验证"}.get(kind, "无语义产物")
        judge = {c: {"score": 0, "rationale": note, "issues": []} for c in _CRITERIA}
        judge["overall"] = note
        judge_calls: list = []
    else:
        judge_router = make_judge_router(judge_provider, sink, config)
        judge = await judge_memory_maintain(
            judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
            det_findings=det, judge_scene=judge_scene)
        judge_calls = [
            {"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
            for c in sink.llm_calls if c.scene == judge_scene.value
        ]

    # maintenance LLM calls (summary rewrite / insight generation) captured by the traced router.
    maintain_calls = [
        {"scene": c.scene, "agent_id": c.agent_id, "prompt": c.prompt_messages, "response": c.response_content}
        for c in sink.llm_calls if c.scene != judge_scene.value
    ]

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "maintain.json", {"inputs": inputs, "outputs": outputs})
    write_json(scenario_dir / "prompt.json",
          {"maintain_calls": maintain_calls, "judge_call": judge_calls[-1] if judge_calls else None})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name, "kind": kind,
        # semantic_judged=False ⇒ mechanism-only validation (decay / below-threshold compress / empty
        # reflect); a 0 in scores then means "not applicable", not failure, so reports/UI should show "—".
        "semantic_judged": bool(has_semantic),
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_memory_maintain(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full memory-maintenance validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())
    out_dir = Path(trace_dir) / world_id / "validation" / "memory_maintain"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(container, config, world_id, meta, agent_ids=agent_ids,
                                   out_dir=out_dir, judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "memory_maintain", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md, default_kind="decay",
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="memory_maintain", avg_label="语义均值(compress/reflect)",
        columns=[
            ("kind", lambda r: r.get("kind")),
            ("fidelity", lambda r: r["scores"]["fidelity"] if r.get("semantic_judged") else "—"),
            ("voice", lambda r: r["scores"]["voice"] if r.get("semantic_judged") else "—"),
        ],
    )
