"""perception_emotion validation suite: core scenarios + deterministic field checks + LLM judge.

The perception-emotion stage produces the pre-decision instinctive emotion. Every agent
takes the LLM path, so its output quality (does the felt
emotion fit the perceived signals, personality and relations — and stay an instinctive
first reaction rather than a decision?) is what matters, and that is judged by an LLM.

Each scenario runs in isolation (own InMemoryTraceSink; ``run_perception_emotion`` restores
a fresh, non-persisted world and calls the production method directly), concurrently via
asyncio.gather. For each scenario it persists input / per-agent / prompt / deterministic
checks / judge results, then writes a summary.{json,md}. Production code is untouched; the
judge takes its own endpoint declaration (``llm.judge``).

No knob sets / A/B: the LLM path has no numeric knobs, and the only lever — the prompt —
is off-limits to test-driven overfitting. Deterministic field checks are the structural
tripwire; the judge carries the semantic / functional verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.need import NeedType
from agent.personality import EmotionType
from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene, extract_json
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_llm import create_judge
from tuning.judge_perception_emotion import judge_perception_emotion
from tuning.phase_harness.perception_emotion import run_perception_emotion
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import focus_of, load_scenarios, make_judge_router, run_suite, summary_md
from tuning.validation import resolve_placeholders

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "perception_emotion.json"
_CRITERIA = ("fields", "semantic", "functional")
_CANON_EMOTIONS = {e.value for e in EmotionType}
_CANON_NEEDS = {nt.value for nt in NeedType}
# The prompt asks for reason "一句话，不超过20字" — leave a little slack and only flag clearly verbose ones.
_REASON_MAX = 24
# The scene this stage's LLM occupies.
_EMOTION_SCENES = {LLMScene.AGENT_DECISION_MAIN.value}
# Negative-polarity emotions: expect valence <= 0 (with a small margin).
_NEGATIVE_EMOTIONS = {"fear", "anger", "sadness", "disgust", "contempt", "shame", "jealousy", "frustration"}


# ---------------------------------------------------------------------------
# Deterministic field checks (syntactic tripwire on the raw LLM output)
# ---------------------------------------------------------------------------

def _deterministic_checks(per_agent: list[dict], raw_by_agent: dict[str, str]) -> dict[str, Any]:
    """Validate the raw LLM emotion output's fields. Lenient re-parse — flags what the
    production parser would silently coerce or clamp, plus polarity self-consistency.

    Operates on raw responses. Agents with no raw response (no signal, or parse fallback →
    None) contribute no field check; their None is recorded as info, not failure.
    """
    eps = 1e-9
    field_issues: list[str] = []
    for aid, raw in raw_by_agent.items():
        try:
            data = extract_json(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            field_issues.append(f"{aid}: 原始输出非合法 JSON")
            continue
        if not isinstance(data, dict):
            field_issues.append(f"{aid}: 输出非 JSON 对象")
            continue
        raw_emo = data.get("emotion")
        emo = str(raw_emo).strip() if raw_emo is not None else ""
        # An omitted / empty emotion is a valid "no new reaction" (the contract allows it; the caller keeps
        # the current emotion). Don't check the emotion fields (intensity/valence/reason) then — omitting them too is correct.
        if not emo:
            continue
        if emo not in _CANON_EMOTIONS:
            field_issues.append(f"{aid}: 非规范 emotion={emo!r}(将经 parse 同义词映射)")
        nums: dict[str, float] = {}
        for fld, (lo, hi) in (("intensity", (0.0, 1.0)), ("valence", (-1.0, 1.0))):
            try:
                fv = float(data.get(fld))
            except (TypeError, ValueError):
                field_issues.append(f"{aid}: {fld} 非数值={data.get(fld)!r}")
                continue
            nums[fld] = fv
            if not (lo - eps <= fv <= hi + eps):
                field_issues.append(f"{aid}: {fld}={fv} 越界[{lo},{hi}](将被裁剪)")
        # polarity self-consistency: a negative-class emotion with positive valence.
        if emo in _NEGATIVE_EMOTIONS and nums.get("valence", 0.0) > eps:
            field_issues.append(f"{aid}: 情绪 {emo!r} 为负向类,但 valence={nums['valence']}>0(极性矛盾)")
        reason = str(data.get("reason", ""))
        if not reason.strip():
            field_issues.append(f"{aid}: 缺少 reason")
        elif len(reason) > _REASON_MAX:
            field_issues.append(f"{aid}: reason 过长({len(reason)}字,期望≤20)")

    # need_activation structural check (on the parsed/serialized harness output): keys must
    # be canonical needs and values in [0,1]. The parser already enforces this, so a hit
    # here flags a harness/contract regression rather than a model mistake.
    activation_issues: list[str] = []
    for a in per_agent:
        act = (a.get("outputs", {}) or {}).get("need_activation") or {}
        for key, value in act.items():
            if key not in _CANON_NEEDS:
                activation_issues.append(f"{a['agent_name']}: 非规范 need={key!r}")
            try:
                fv = float(value)
            except (TypeError, ValueError):
                activation_issues.append(f"{a['agent_name']}: need_activation[{key}] 非数值={value!r}")
                continue
            if not (0.0 - eps <= fv <= 1.0 + eps):
                activation_issues.append(f"{a['agent_name']}: need_activation[{key}]={fv} 越界[0,1]")

    none_agents = [
        a["agent_name"] for a in per_agent
        if (a.get("outputs", {}) or {}).get("emotion") is None
    ]
    return {
        "passed": not (field_issues or activation_issues),
        "field_issues": field_issues,
        "activation_issues": activation_issues,
        "none_output_agents": none_agents,  # info only (no signal / parse fallback)
        "llm_agent_count": len(raw_by_agent),
    }


# ---------------------------------------------------------------------------
# Per-scenario run
# ---------------------------------------------------------------------------

async def _run_scenario(
    container: Container,
    config: Config,
    world_id: str,
    scenario_meta: dict,
    *,
    resolved: dict,
    out_dir: Path,
    judge_provider: LLMProvider,
    judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_perception_emotion(
        container, config, world_id,
        scenario=resolved, sink=sink, trace_dir=str(scenario_dir),
    )

    per_agent = [
        {"agent_name": p.inputs.get("agent_name", p.agent_id), "inputs": p.inputs, "outputs": p.outputs}
        for p in sink.phases
    ]
    emo_calls = [c for c in sink.llm_calls if c.scene in _EMOTION_SCENES]
    raw_by_agent: dict[str, str] = {}
    for c in emo_calls:
        if c.agent_id and c.agent_id not in raw_by_agent:
            raw_by_agent[c.agent_id] = c.response_content

    det = _deterministic_checks(per_agent, raw_by_agent)

    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_perception_emotion(
        judge_router, scenario_meta=scenario_meta, per_agent=per_agent,
        det_findings=det, judge_scene=judge_scene,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", {"meta": scenario_meta, "resolved": resolved})
    write_json(scenario_dir / "agents.json", per_agent)
    write_json(scenario_dir / "prompt.json", {
        "emotion_calls": [
            {"agent_id": c.agent_id, "prompt": c.prompt_messages, "response": c.response_content}
            for c in emo_calls
        ],
    })
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name,
        "criteria_focus": scenario_meta.get("criteria_focus", []),
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_issues", []) + det.get("activation_issues", []),
    }


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_perception_emotion(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenarios_path: str | None = None,
    trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None,
    judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full perception_emotion validation suite. Returns the summary dict.

    The judge uses its own LLM (``llm.judge``, overridable per run with ``judge_model``)
    — decoupled from the production router being evaluated. Tests inject
    ``judge_provider`` to avoid a real call.
    """
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    # One read-only restore to resolve placeholders (agent locations + an empty location).
    world = await WorldInitializer(container).restore(world_id)
    name_by_id = {aid: a.personality.soul.name for aid, a in world.agents.items()}
    id_by_name = {v: k for k, v in name_by_id.items()}
    agent_locations = {aid: world.environment.get_body_location(aid) for aid in world.agents}
    all_locations = list(world.environment.space.all_place_ids())
    occupied = {loc for loc in agent_locations.values() if loc}
    empty_location = next((loc for loc in all_locations if loc not in occupied), "nowhere_void")

    out_dir = Path(trace_dir) / world_id / "validation" / "perception_emotion"

    async def _one(meta: dict) -> dict:
        resolved = resolve_placeholders(
            meta["scenario"], agent_locations=agent_locations,
            id_by_name=id_by_name, empty_location=empty_location,
        )
        return await _run_scenario(
            container, config, world_id, meta,
            resolved=resolved, out_dir=out_dir, judge_provider=judge_llm, judge_scene=judge_scene,
        )

    return await run_suite(
        "perception_emotion", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md,
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="perception_emotion", avg_label="三项均值",
        columns=[
            ("重点", focus_of),
            ("fields", lambda r: r["scores"]["fields"]),
            ("semantic", lambda r: r["scores"]["semantic"]),
            ("functional", lambda r: r["scores"]["functional"]),
        ],
    )
