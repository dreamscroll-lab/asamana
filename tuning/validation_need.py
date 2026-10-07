"""need_engine (motivation) validation suite: scenarios + deterministic checks + LLM judge.

NeedEngine.run combines need intensity, situational relevance, runtime adjustments and the
external-pressure override into dominant_need + the remaining active_needs ranked by scores + the
LLM-generated short_term_goals. Each scenario restores independently, feeds inputs from the real
appraisal (perception → need), runs ``run_need_engine``, then applies deterministic structural
checks + an LLM judge (mainly semantic).

Deterministic checks (syntax/structure, always run) are the main measurement's hard tripwires:
dominant in the canonical set, canonical scores, 1 to _SHORT_TERM_GOAL_CAP goals, and the consistency invariant
dominant_need == argmax(scores) always holds (external-pressure boosts are already in scores, so
there's no override exception). The judge carries semantic judgment. Zero production changes;
the judge uses its own ``llm.judge`` declaration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.goals import _SHORT_TERM_GOAL_CAP
from agent.need import NeedType
from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_llm import create_judge
from tuning.judge_need import judge_need
from tuning.phase_harness.need import run_need_engine
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import focus_of, load_scenarios, make_judge_router, run_suite, summary_md
from tuning.validation import resolve_placeholders

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "need.json"
_CRITERIA = ("fields", "ranking", "goals")
_CANON_NEEDS = {nt.value for nt in NeedType}
_GOAL_MAX = 30  # the prompt asks for ≤25 chars; leave slack
_GOAL_GEN_SCENE = LLMScene.NEED_GOAL_GENERATION.value


# ---------------------------------------------------------------------------
# Deterministic checks (structural + override invariant)
# ---------------------------------------------------------------------------

#: Literal traces in goal text that "this is set for some time". For measurement only; never use
#: it to extract a date and compute due: the arithmetic goes wrong and calendars differ. It only
#: answers the boolean "is a time mentioned". This table belongs to tuning's metric, not production code.
_TIME_MENTION = re.compile(
    r"(卯时|辰时|巳时|午时|未时|申时|酉时|戌时|亥时|子时|丑时|寅时"
    r"|天亮前|天明前|入夜|今夜|今晚|明日|明晨|明早|次日|后日|清早|傍晚|正午|日落前"
    r"|初[一二三四五六七八九十]|廿[一二三四五六七八九]|三十日"
    r"|\d+月\d+日|\d+点|\d+时前)"
)


def _mentions_a_time(text: str) -> bool:
    return bool(_TIME_MENTION.search(text))


def _deterministic_checks(per_agent: list[dict]) -> dict[str, Any]:
    field_v: list[str] = []
    override_v: list[str] = []
    due_v: list[str] = []
    for a in per_agent:
        name = a.get("agent_name", "?")
        out = a.get("outputs", {}) or {}
        dominant = out.get("dominant_need")
        scores = out.get("scores") or {}
        goals = out.get("short_term_goals") or []
        ents = out.get("goal_entities") or []

        if dominant is not None and dominant not in _CANON_NEEDS:
            field_v.append(f"{name}: dominant_need 非规范={dominant!r}")
        # scores only include needs present in the agent's innate_needs (possibly a subset of the 5),
        # so don't require all of them; only check non-empty, canonical keys, non-negative values.
        if not scores:
            field_v.append(f"{name}: scores 为空")
        if set(scores) - _CANON_NEEDS:
            field_v.append(f"{name}: scores 含非规范需求 {sorted(set(scores) - _CANON_NEEDS)}")
        if any((not isinstance(v, (int, float))) or v < 0 for v in scores.values()):
            field_v.append(f"{name}: scores 含非法值")
        # Short-term goals are a capped FIFO queue (unfinished ones stay, new ones are appended); the live queue is always non-empty and ≤ cap.
        if not (1 <= len(goals) <= _SHORT_TERM_GOAL_CAP):
            field_v.append(f"{name}: short_term_goals 数量={len(goals)}(应 1-{_SHORT_TERM_GOAL_CAP})")
        for g in goals:
            if not str(g).strip():
                field_v.append(f"{name}: 存在空目标")
            elif len(str(g)) > _GOAL_MAX:
                field_v.append(f"{name}: 目标过长({len(str(g))}字)「{str(g)[:16]}…」")
        if len(set(goals)) != len(goals):
            field_v.append(f"{name}: 目标重复")
        # Only check that a goal's related_need is canonical (None = wildcard, valid). Don't require it to
        # equal dominant: goals in the FIFO queue persist across dominant changes, so older goals can carry an earlier need and deferred intents carry None.
        for e in ents:
            rn = e.get("related_need")
            if rn is not None and rn not in _CANON_NEEDS:
                field_v.append(f"{name}: goal related_need 非规范={rn!r}")
            # Consistency: if the text names a time but no due was set, the commitment has no anchor to
            # compare against and never falls due. The reverse isn't checked: a due without a written time usually means "right away", which is harmless.
            if _mentions_a_time(str(e.get("text") or "")) and not e.get("has_due"):
                due_v.append(f"{name}: 目标写了时点却没定下时候「{str(e.get('text'))[:20]}…」")

        # Coherence invariant — dominant_need must always equal argmax(scores). External pressure
        # is folded additively into scores (MotivationBlender.external_pressure_boost), so there is
        # no override exception: dominant can never contradict the reported scores.
        argmax = max(scores, key=lambda k: scores[k]) if scores else None
        if argmax is not None and dominant != argmax:
            override_v.append(f"{name}: dominant 应为评分最高 {argmax}(外部加成已计入 scores),却是 {dominant}")

    return {
        "passed": not (field_v or override_v or due_v),
        "field_violations": field_v,
        "override_violations": override_v,
        "due_violations": due_v,
        "dominant_by_agent": {a.get("agent_name"): (a.get("outputs", {}) or {}).get("dominant_need") for a in per_agent},
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

    await run_need_engine(
        container, config, world_id,
        scenario=resolved, sink=sink, trace_dir=str(scenario_dir),
    )

    per_agent = [
        {"agent_name": p.inputs.get("agent_name", p.agent_id), "inputs": p.inputs, "outputs": p.outputs}
        for p in sink.phases
    ]
    goal_calls = [c for c in sink.llm_calls if c.scene == _GOAL_GEN_SCENE]

    det = _deterministic_checks(per_agent)

    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_need(
        judge_router, scenario_meta=scenario_meta, per_agent=per_agent,
        det_findings=det, judge_scene=judge_scene,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", {"meta": scenario_meta, "resolved": resolved})
    write_json(scenario_dir / "agents.json", per_agent)
    write_json(scenario_dir / "prompt.json", {
        "goal_calls": [
            {"agent_id": c.agent_id, "prompt": c.prompt_messages, "response": c.response_content}
            for c in goal_calls
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
        + det.get("field_violations", [])
        + det.get("override_violations", [])
        + det.get("due_violations", []),
    }


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_need(
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
    """Run the full need_engine validation suite. Returns the summary dict."""
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

    out_dir = Path(trace_dir) / world_id / "validation" / "need"

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
        "need", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md,
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="need_engine", avg_label="三项均值",
        columns=[
            ("重点", focus_of),
            ("fields", lambda r: r["scores"]["fields"]),
            ("ranking", lambda r: r["scores"]["ranking"]),
            ("goals", lambda r: r["scores"]["goals"]),
        ],
    )
