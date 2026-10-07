"""feedback stage validation suite: scenarios + deterministic checks + LLM judge.

Restores a built world and, per scenario, drives the **real production feedback layer**
(``begin_ongoing_step`` → ``finalize_ongoing_action`` for kind=self / ``agent.apply_target_effect`` for kind=target),
capturing before/after agent state, then applies structural deterministic checks + an LLM
judge (syntax + semantics of the landing).

Persistence isolation: the harness wraps the agent in _DryRunAgentStore (read-through,
write-capture) + InMemoryVectorStore, so every feedback-layer write goes to in-memory shadows
and the baseline world's ./data stays clean.

Deterministic checks (always run) are hard tripwires: written memories have no agent id and no
"第N步/steps"; emotions canonical with intensity/valence in range; appraised gap∈[0,1]; relation
Δ magnitudes bounded; goal statuses valid; vitality∈[0,1] and consistent with death; no landing
exceptions. The judge (``llm.judge``) carries the semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


from agent.personality import EmotionType, parse_emotion_type
from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_feedback import judge_feedback
from tuning.judge_llm import create_judge
from tuning.phase_harness.feedback import run_feedback
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import STEP_LEAK, focus_of, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "feedback.json"
_CRITERIA = ("fields", "appraisal_fidelity", "landing_consistency")
_VALID_GOAL_STATUS = {"active", "completed", "failed", "interrupted"}


# ---------------------------------------------------------------------------
# Deterministic checks (structural tripwire)
# ---------------------------------------------------------------------------

def _deterministic_checks(phase: dict, agent_ids: set[str]) -> dict[str, Any]:
    v: list[str] = []
    outputs = phase.get("outputs", {}) or {}

    if outputs.get("error"):
        v.append(f"落地异常: {outputs['error']}")

    # Written memory prose: no agent id, no step leak.
    for m in outputs.get("new_memories", []) or []:
        txt = str(m.get("content") or "")
        for aid in agent_ids:
            if aid and aid in txt:
                v.append(f"记忆泄漏 agent id「{aid}」→「{txt[:30]}…」")
                break
        if STEP_LEAK.search(txt):
            v.append(f"记忆出现 step(『第N步/N步/steps』)→「{txt[:30]}…」")

    after = outputs.get("after", {}) or {}
    emo = after.get("emotion")
    if emo:
        primary = str(emo.get("primary"))
        if parse_emotion_type(primary) == EmotionType.NEUTRAL and primary.lower() not in ("neutral", "平静", "中性"):
            v.append(f"情绪 primary 非规范={primary!r}")
        for k, lo, hi in (("intensity", 0.0, 1.0), ("valence", -1.0, 1.0)):
            x = emo.get(k)
            if not isinstance(x, (int, float)) or not (lo <= float(x) <= hi):
                v.append(f"情绪 {k} 越界={x!r}")

    gap = outputs.get("appraised_gap")
    if gap is not None and (not isinstance(gap, (int, float)) or not (0.0 <= float(gap) <= 1.0)):
        v.append(f"appraised gap 越界/非法={gap!r}")

    vit = after.get("vitality")
    if vit is not None and (not isinstance(vit, (int, float)) or not (0.0 <= float(vit) <= 1.0)):
        v.append(f"vitality 越界={vit!r}")

    deltas = outputs.get("deltas", {}) or {}
    for n, rd in (deltas.get("relation_deltas") or {}).items():
        for k in ("trust_delta", "affection_delta"):
            d = rd.get(k)
            if not isinstance(d, (int, float)) or not (-1.0 <= float(d) <= 1.0):
                v.append(f"关系 {k}（对{n}）越界={d!r}")

    for g in after.get("goals", []) or []:
        if str(g.get("status")) not in _VALID_GOAL_STATUS:
            v.append(f"目标状态非法={g.get('status')!r}")

    # Death consistency: is_active=False ⇒ vitality must be 0.
    if after.get("is_active") is False and vit not in (0, 0.0):
        v.append(f"死亡不一致: is_active=False 但 vitality={vit!r}")

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

    await run_feedback(container, config, world_id, scenario=scenario_meta, sink=sink,
                       trace_dir=str(scenario_dir))

    phase = next((p for p in sink.phases if p.phase == "stage.feedback"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    phase_dict = {"inputs": inputs, "outputs": outputs}

    det = _deterministic_checks(phase_dict, agent_ids)

    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_feedback(
        judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
        det_findings=det, judge_scene=judge_scene)

    # feedback cognition calls (emotion appraisal / goal eval / experiential rewrite) — every
    # non-judge call captured by the traced router.
    feedback_calls = [
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
    write_json(scenario_dir / "feedback.json", phase_dict)
    write_json(scenario_dir / "prompt.json",
          {"feedback_calls": feedback_calls, "judge_call": judge_calls[-1] if judge_calls else None})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name,
        "kind": scenario_meta.get("kind", "self"),
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

async def validate_feedback(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full feedback validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())

    out_dir = Path(trace_dir) / world_id / "validation" / "feedback"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(
            container, config, world_id, meta, agent_ids=agent_ids, out_dir=out_dir,
            judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "feedback", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md, default_kind="self",
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="feedback", avg_label="三项均值",
        columns=[
            ("类型", lambda r: r.get("kind")),
            ("重点", focus_of),
            ("fields", lambda r: r["scores"]["fields"]),
            ("appraisal", lambda r: r["scores"]["appraisal_fidelity"]),
            ("landing", lambda r: r["scores"]["landing_consistency"]),
        ],
    )
