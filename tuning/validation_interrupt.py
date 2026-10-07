"""interrupt stage validation: scenarios + deterministic checks + LLM judge.

Validates ``Agent.evaluate_interrupt`` (driven through the real ``runtime._decide_interrupt``):
while a character is mid multi-step action, an external signal arrives and they weigh in first
person whether to interrupt, leaving a first-person inner monologue. Every agent makes this call
through the LLM, thinking before deciding (CLAUDE.md §5: quality first, no rule tier).
Deterministic checks cover the structural invariants (should_interrupt is a bool, no step leak in
thought or progress_hint, the LLM was actually called, thought is non-empty when the decision
succeeds); the LLM judge carries semantic quality (decision_fit + voice).

executor.interrupt (the narrative output after an interrupt) is validated separately and not
repeated here. Zero production intrusion: isolated dry-run (shadow store + InMemory vector
store); baseline ./data is never written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_interrupt import judge_interrupt
from tuning.judge_llm import create_judge
from tuning.phase_harness.interrupt import run_interrupt
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import STEP_LEAK, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "interrupt.json"
_CRITERIA = ("decision_fit", "voice")
_INTERRUPT_SCENE = LLMScene.AGENT_INTERRUPT_DECISION.value


def _deterministic_checks(inputs: dict, outputs: dict, *, llm_call_count: int) -> dict[str, Any]:
    """Structural hard invariants: always run, the tripwire for the main measurement."""
    v: list[str] = []
    if outputs.get("error"):
        v.append(f"运行异常: {outputs['error']}")
        return {"passed": False, "field_violations": v}

    si = outputs.get("should_interrupt")
    thought = str(outputs.get("thought") or "")

    if not isinstance(si, bool):
        v.append(f"should_interrupt 非 bool={si!r}")
    # Step leak (syntax/boundary): neither thought nor progress_hint may contain a bare step.
    if STEP_LEAK.search(thought):
        v.append(f"thought 出现裸 step 泄漏→「{thought[:24]}…」")
    hint = str(inputs.get("progress_hint") or "")
    if STEP_LEAK.search(hint):
        v.append(f"progress_hint 出现裸 step 泄漏→「{hint}」")

    # Every agent's interrupt decision goes through the LLM: it must actually have been called, and
    # thought must be non-empty when the decision succeeds. An empty monologue signals degradation or a parse failure.
    if llm_call_count == 0:
        v.append("中断决策未调用 LLM(全员 LLM 认知路径,不应出现)")
    if si is not None and not thought:
        v.append("thought 为空(疑似退化/解析失败)")

    return {"passed": not v, "field_violations": v}


async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, out_dir: Path, judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_interrupt(container, config, world_id, scenario=scenario_meta, sink=sink,
                        trace_dir=str(scenario_dir))
    phase = next((p for p in sink.phases if p.phase == "stage.interrupt"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}

    # The interrupt decision's LLM calls (captured by the traced router): the count verifies the LLM was really used; the prompt is for the web view.
    interrupt_calls = [
        {"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
        for c in sink.llm_calls if c.scene == _INTERRUPT_SCENE
    ]
    det = _deterministic_checks(inputs, outputs, llm_call_count=len(interrupt_calls))

    # The semantic judge needs an LLM output to grade: skip on a run exception or when no decision came out.
    semantic_judged = (
        not outputs.get("error")
        and isinstance(outputs.get("should_interrupt"), bool)
    )
    if not semantic_judged:
        judge = {c: {"score": 0, "rationale": "无 LLM 产物,不评语义", "issues": []} for c in _CRITERIA}
        judge["overall"] = "无 LLM 产物"
        judge_calls: list = []
    else:
        judge_router = make_judge_router(judge_provider, sink, config)
        judge = await judge_interrupt(
            judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
            det_findings=det, judge_scene=judge_scene)
        judge_calls = [{"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
                       for c in sink.llm_calls if c.scene == judge_scene.value]

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "interrupt.json", {"inputs": inputs, "outputs": outputs})
    write_json(scenario_dir / "prompt.json", {
        "interrupt_call": interrupt_calls[-1] if interrupt_calls else None,
        "judge_call": judge_calls[-1] if judge_calls else None,
    })
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name,
        "criteria_focus": scenario_meta.get("criteria_focus", []),
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "semantic_judged": semantic_judged,
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


async def validate_interrupt(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full interrupt validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    # restore once just to surface a clear error early if the world is missing.
    await WorldInitializer(container).restore(world_id)
    out_dir = Path(trace_dir) / world_id / "validation" / "interrupt"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(container, config, world_id, meta, out_dir=out_dir,
                                   judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "interrupt", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md,
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="interrupt", avg_label="语义均值",
        columns=[
            ("decision_fit", lambda r: r["scores"]["decision_fit"] if r.get("semantic_judged") else "—"),
            ("voice", lambda r: r["scores"]["voice"] if r.get("semantic_judged") else "—"),
        ],
    )
