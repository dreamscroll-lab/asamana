"""memory retrieval stage validation: scenarios + deterministic checks + LLM judge.

Restores a built world, rebuilds an isolated memory_system with **real text-embedding-v3 hybrid**
(dense + DashScope learned sparse; the baseline in_memory dim-8 relevance is noise), seeds a
narrative-faithful candidate pool, then drives the real production retrieve_both (per-query
min-max normalized).

Deterministic checks (always run) = structural hard invariants: correct kind classification,
top_k not exceeded, no id/step leak in recalled prose, no near-duplicates (MMR), the contrast
pair of queries recalls different sets (query-sensitivity), optional
must_recall/must_not_recall anchors.
The judge carries semantic recall quality (relevance + coverage).
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_llm import create_judge
from tuning.judge_memory_retrieve import judge_memory_retrieve
from tuning.phase_harness.memory_retrieve import run_memory_retrieve
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import STEP_LEAK, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "memory_retrieve.json"
_CRITERIA = ("relevance", "coverage")
_KIND_OK = {"events": "event", "insights": "insight", "summaries": "summary"}


def _all_recalled(recalled: dict) -> list[dict]:
    return [m for key in _KIND_OK for m in (recalled.get(key) or [])]


def _deterministic_checks(phase: dict, agent_ids: set[str], scenario_meta: dict) -> dict[str, Any]:
    inputs, outputs = phase.get("inputs", {}) or {}, phase.get("outputs", {}) or {}
    v: list[str] = []
    if outputs.get("error"):
        v.append(f"检索异常: {outputs['error']}")
    recalled = outputs.get("recalled", {}) or {}
    top_k = int(inputs.get("top_k", 5))

    # Correct kind classification + counts within limit + no id/step leak.
    for key, expect_kind in _KIND_OK.items():
        items = recalled.get(key) or []
        if len(items) > 2 * top_k + 1:
            v.append(f"{key} 召回超量({len(items)} > 2·top_k)")
        for m in items:
            if m.get("kind") != expect_kind:
                v.append(f"{key} 含错误 kind={m.get('kind')!r}")
    for m in _all_recalled(recalled):
        txt = str(m.get("content") or "")
        for aid in agent_ids:
            if aid and aid in txt:
                v.append(f"召回 prose 泄漏 agent id「{aid}」")
                break
        if STEP_LEAK.search(txt):
            v.append(f"召回 prose 出现 step→「{txt[:24]}…」")

    # No near-duplicates (MMR should already have deduped >0.9).
    ev = [str(m.get("content") or "") for m in (recalled.get("events") or [])]
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            if SequenceMatcher(None, ev[i], ev[j]).ratio() > 0.9:
                v.append(f"召回含近重复:「{ev[i][:18]}…」≈「{ev[j][:18]}…」")
                break

    # query-sensitivity: the two contrast queries shouldn't recall identical event sets.
    if outputs.get("kind") == "contrast":
        a = {m.get("content") for m in (recalled.get("events") or [])}
        b = {m.get("content") for m in ((outputs.get("recalled_b") or {}).get("events") or [])}
        if a and a == b:
            v.append("contrast 两 query 召回完全相同(query 不敏感)")

    # Optional anchors (deterministic assertions for scenarios with a known answer): must_recall must appear, must_not_recall must not (substring match).
    joined = " ｜ ".join(str(m.get("content") or "") for m in _all_recalled(recalled))
    for sub in scenario_meta.get("must_recall", []) or []:
        if sub not in joined:
            v.append(f"应召回却缺失:「{sub}」")
    for sub in scenario_meta.get("must_not_recall", []) or []:
        if sub in joined:
            v.append(f"不应召回却出现:「{sub}」")

    return {"passed": not v, "field_violations": v}


async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, agent_ids: set[str], out_dir: Path, judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_memory_retrieve(container, config, world_id, scenario=scenario_meta, sink=sink,
                              trace_dir=str(scenario_dir))
    phase = next((p for p in sink.phases if p.phase == "stage.memory_retrieve"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    det = _deterministic_checks({"inputs": inputs, "outputs": outputs}, agent_ids, scenario_meta)

    has_recall = bool(_all_recalled(outputs.get("recalled", {}) or {})) and not outputs.get("error")
    if not has_recall:
        judge = {c: {"score": 0, "rationale": "无召回/异常,不评语义", "issues": []} for c in _CRITERIA}
        judge["overall"] = "无召回/异常"
        judge_calls: list = []
    else:
        judge_router = make_judge_router(judge_provider, sink, config)
        judge = await judge_memory_retrieve(
            judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
            det_findings=det, judge_scene=judge_scene)
        judge_calls = [{"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
                       for c in sink.llm_calls if c.scene == judge_scene.value]

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "retrieve.json", {"inputs": inputs, "outputs": outputs})
    write_json(scenario_dir / "prompt.json", {"judge_call": judge_calls[-1] if judge_calls else None})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name, "kind": scenario_meta.get("kind", "single"),
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


async def validate_memory_retrieve(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full memory-retrieval validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())
    out_dir = Path(trace_dir) / world_id / "validation" / "memory_retrieve"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(container, config, world_id, meta, agent_ids=agent_ids,
                                   out_dir=out_dir, judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "memory_retrieve", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md, default_kind="single",
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="memory_retrieve", avg_label="语义均值",
        columns=[
            ("kind", lambda r: r.get("kind")),
            ("relevance", lambda r: r["scores"]["relevance"]),
            ("coverage", lambda r: r["scores"]["coverage"]),
        ],
    )
