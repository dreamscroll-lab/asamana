"""decision (decide) validation suite: scenarios + deterministic checks + LLM judge.

The decide stage is the last cognition link (intent → world-visible action). This suite restores a
built world, injects per-scenario signals (incl. **memories**), runs the real production path
(``_build_internal_context`` → ``decision_engine.decide``) via ``run_decision``, then applies
structural deterministic checks + an LLM judge (semantic, per what decide is for: judgment lens ×
direction × reality → one concrete, feasible action).

Deterministic checks (syntax/structure, always run) are the main measurement's hard tripwires:
action_type in the canonical set, valid target binding (MOVE destination adjacent), description
non-empty and not too long, estimated_steps ≥ 1, inner monologue present. The judge carries the
semantics. Zero production intrusion (only memory injection is test scaffolding); the judge uses
its own ``llm.judge`` declaration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.models import Config
from core.container import Container
from core.interfaces.action import KIND_AGENT, KIND_NPC, KIND_OBJECT, ActionType
from world.models import WorldEntityType
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_decision import judge_decision
from tuning.judge_llm import create_judge
from tuning.phase_harness.decision import run_decision
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import focus_of, load_scenarios, make_judge_router, run_suite, summary_md
from tuning.validation import resolve_placeholders

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "decision.json"
_CRITERIA = ("fields", "action_fit", "coherence")
_CANON_ACTIONS = {at.value for at in ActionType}
# decision.py derives the target kind in code from the channel (person → agent, cognition-less body
# → npc, listed entity → its entity_type, off-list self-described target → object); the LLM never
# fills it in. A value outside this range means a binding branch is wrong, not the model.
# Derived from the two authorities, not copied: a hand copy missing a kind (npc, say) would flag
# legal decisions.
_CANON_ITEM_TYPES = {KIND_AGENT, KIND_NPC, KIND_OBJECT} | {t.value for t in WorldEntityType}
_DESC_MAX = 80  # the prompt asks for 15–40 chars; leave slack
_DECIDE_SCENE = LLMScene.AGENT_DECISION_MAIN.value


# ---------------------------------------------------------------------------
# Deterministic checks (structural tripwire)
# ---------------------------------------------------------------------------

def _deterministic_checks(per_agent: list[dict]) -> dict[str, Any]:
    field_v: list[str] = []
    for a in per_agent:
        name = a.get("agent_name", "?")
        out = a.get("outputs", {}) or {}
        inp = a.get("inputs", {}) or {}
        atype = out.get("action_type")
        desc = str(out.get("action_description") or "")
        steps = out.get("estimated_steps")
        target = out.get("target") or {}

        if atype not in _CANON_ACTIONS:
            field_v.append(f"{name}: action_type 非规范={atype!r}")
        if not desc.strip():
            field_v.append(f"{name}: action_description 为空")
        elif len(desc) > _DESC_MAX:
            field_v.append(f"{name}: action_description 过长({len(desc)}字)「{desc[:16]}…」")
        if not isinstance(steps, int) or steps < 1:
            field_v.append(f"{name}: estimated_steps 非法={steps!r}(应≥1)")
        if not str(out.get("inner_monologue") or "").strip():
            field_v.append(f"{name}: inner_monologue 为空")
        # Target validity (the checkable part: a MOVE destination must be among the agent's reachable locations)
        acts_on = target.get("acts_on") or []
        if atype == ActionType.MOVE.value:
            dest = next((r.get("id") for r in acts_on if r.get("kind") == "location"), None)
            reachable = set(inp.get("reachable_locations") or [])
            if not dest or dest not in reachable:
                field_v.append(f"{name}: MOVE 目的地={dest!r} 不在可达地点 {sorted(reachable)} 内")
        # PHYSICAL must bind a target whose kind falls in the derived range. The binding layer already
        # hard-rejects the step when it can't bind, so both should always hold here. They stay as a
        # regression net: if either fires, the guard or a binding branch is broken.
        if atype == ActionType.PHYSICAL.value:
            if not acts_on:
                field_v.append(f"{name}: PHYSICAL 未绑定目标")
            for ref in acts_on:
                kind = ref.get("kind")
                if kind not in _CANON_ITEM_TYPES:
                    field_v.append(
                        f"{name}: PHYSICAL 目标类别缺失/非法={kind!r}（须∈{sorted(_CANON_ITEM_TYPES)}）"
                    )

    return {"passed": not field_v, "field_violations": field_v}


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

    await run_decision(
        container, config, world_id,
        scenario=resolved, sink=sink, trace_dir=str(scenario_dir),
    )

    per_agent = [
        {"agent_name": p.inputs.get("agent_name", p.agent_id), "inputs": p.inputs, "outputs": p.outputs}
        for p in sink.phases
    ]
    # The AGENT_DECISION_MAIN scene carries both main-character appraisal and everyone's decide; only
    # decide has the action-menu annotation, so use it to filter out appraisal calls. Otherwise the decision prompt gets mixed with appraisal prompts.
    decide_calls = [
        c for c in sink.llm_calls
        if c.scene == _DECIDE_SCENE and "action_menu" in c.extra
    ]

    det = _deterministic_checks(per_agent)

    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_decision(
        judge_router, scenario_meta=scenario_meta, per_agent=per_agent,
        det_findings=det, judge_scene=judge_scene,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", {"meta": scenario_meta, "resolved": resolved})
    write_json(scenario_dir / "agents.json", per_agent)
    write_json(scenario_dir / "prompt.json", {
        "decide_calls": [
            {"agent_id": c.agent_id, "prompt": c.prompt_messages, "response": c.response_content}
            for c in decide_calls
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
        + det.get("field_violations", []),
    }


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_decision(
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
    """Run the full decision validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    name_by_id = {aid: a.personality.soul.name for aid, a in world.agents.items()}
    id_by_name = {v: k for k, v in name_by_id.items()}
    agent_locations = {aid: world.environment.get_body_location(aid) for aid in world.agents}
    all_locations = list(world.environment.space.all_place_ids())
    occupied = {loc for loc in agent_locations.values() if loc}
    empty_location = next((loc for loc in all_locations if loc not in occupied), "nowhere_void")

    out_dir = Path(trace_dir) / world_id / "validation" / "decision"

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
        "decision", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md,
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="decision", avg_label="三项均值",
        columns=[
            ("重点", focus_of),
            ("fields", lambda r: r["scores"]["fields"]),
            ("action_fit", lambda r: r["scores"]["action_fit"]),
            ("coherence", lambda r: r["scores"]["coherence"]),
        ],
    )
