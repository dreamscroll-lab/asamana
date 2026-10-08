"""long_term_goal stage validation: scenarios + deterministic checks + LLM judge.

Under test is ``NeedEngine.revise_long_term_goals``: the character looks back in first person and
reviews their long-term goals, changing them only when the direction really shifted (achieved /
void / change of heart) and holding otherwise (no change for change's sake). The life goal
(life_goal) is the immovable north star; minor matters and passing moods must not become
long-term goals. Output None = unchanged / list = direction changed.

- Deterministic / rules (always run, tripwire): valid output type; the north star (life_goal)
  is identical before and after the call; on change, the new list is non-empty, actually
  different, written back to the profile, every item non-empty and ≤50 chars, no id / bare step
  leak; on no change, state is untouched.
- LLM judge carries the semantics: revision_fit (changed or held correctly,
  always scored) + goal_quality (quality of the new goals, scored only on change — N/A otherwise
  and excluded from the mean).

Zero production intrusion: isolated dry-run (shadow store + InMemory vector store); baseline
./data is never written.
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
from tuning.judge_long_term_goal import judge_long_term_goal
from tuning.phase_harness.long_term_goal import run_long_term_goal
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import load_scenarios, make_judge_router, run_suite, summary_md, text_leak

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "long_term_goal.json"
_CRITERIA = ("revision_fit", "goal_quality")
_GOAL_SCENE = LLMScene.NEED_GOAL_GENERATION.value
_GOAL_MAX_LEN = 50  # hard cap in production revise_long_term_goals (the prompt suggests ≤25)


def _deterministic_checks(outputs: dict, agent_ids: set[str]) -> dict[str, Any]:
    v: list[str] = []
    if outputs.get("error"):
        return {"passed": False, "field_violations": [f"运行异常: {outputs['error']}"]}

    revised = outputs.get("revised")
    out_goals = outputs.get("output_goals")
    before = list(outputs.get("before_long_term") or [])
    after = list(outputs.get("after_long_term") or [])

    # This stage must never touch the north star (life_goal): the hardest rule invariant.
    if outputs.get("life_goal_before") != outputs.get("life_goal_after"):
        v.append(f"毕生追求(北极星)被改动:「{outputs.get('life_goal_before')}」→「{outputs.get('life_goal_after')}」")

    if revised:
        # On change: output must be a non-empty list, actually different, and written back to the profile (after == output).
        if not isinstance(out_goals, list) or not out_goals:
            v.append(f"标记已改动但 output_goals 非非空列表={out_goals!r}")
        else:
            if after != out_goals:
                v.append(f"输出未同步写回 profile:output={out_goals} 但 after={after}")
            if after == before:
                v.append("标记已改动,但 after 与 before 相同(应确有方向变化)")
            for g in out_goals:
                s = str(g).strip()
                if not s:
                    v.append("产出空目标")
                    continue
                if len(s) > _GOAL_MAX_LEN:
                    v.append(f"目标超长({len(s)}字)「{s[:16]}…」")
                leak = text_leak(s, agent_ids=agent_ids)
                if leak:
                    v.append(f"目标{leak}")
    else:
        # No change: long-term goals must be untouched.
        if after != before:
            v.append(f"标记未改动,但 after≠before:{before}→{after}")

    return {"passed": not v, "field_violations": v}


async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, agent_ids: set[str], out_dir: Path, judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_long_term_goal(container, config, world_id, scenario=scenario_meta, sink=sink,
                             trace_dir=str(scenario_dir))
    phase = next((p for p in sink.phases if p.phase == "stage.long_term_goal"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    det = _deterministic_checks(outputs, agent_ids)

    goal_calls = [
        {"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
        for c in sink.llm_calls if c.scene == _GOAL_SCENE
    ]

    revised = bool(outputs.get("revised"))
    semantic_judged = not outputs.get("error")
    if not semantic_judged:
        judge = {c: {"score": 0, "rationale": "运行异常,不评语义", "issues": []} for c in _CRITERIA}
        judge["overall"] = "运行异常"
        judge_calls: list = []
    else:
        judge_router = make_judge_router(judge_provider, sink, config)
        judge = await judge_long_term_goal(
            judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
            det_findings=det, judge_scene=judge_scene)
        judge_calls = [{"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
                       for c in sink.llm_calls if c.scene == judge_scene.value]
        # goal_quality only applies on change; no change → set to 0 (excluded from the mean) and marked N/A, so it doesn't drag the quality mean.
        if not revised:
            judge["goal_quality"] = {"score": 0, "rationale": "未改动方向,目标成色不适用", "issues": []}

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "long_term_goal.json", {"inputs": inputs, "outputs": outputs})
    write_json(scenario_dir / "prompt.json", {
        "goal_call": goal_calls[-1] if goal_calls else None,
        "judge_call": judge_calls[-1] if judge_calls else None,
    })
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name, "revised": revised,
        "criteria_focus": scenario_meta.get("criteria_focus", []),
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "semantic_judged": semantic_judged,
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


async def validate_long_term_goal(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full long_term_goal validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())
    out_dir = Path(trace_dir) / world_id / "validation" / "long_term_goal"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(container, config, world_id, meta, agent_ids=agent_ids,
                                   out_dir=out_dir, judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "long_term_goal", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md,
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="long_term_goal", avg_label="语义均值(goal_quality 仅计有改动的场景)",
        columns=[
            ("改动?", lambda r: "改" if r.get("revised") else "守"),
            ("revision_fit", lambda r: r["scores"]["revision_fit"]),
            ("goal_quality", lambda r: r["scores"]["goal_quality"] if r.get("revised") else "—"),
        ],
    )
