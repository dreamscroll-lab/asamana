"""world_pressure validation suite: controlled scenarios + LLM judge + deterministic checks.

Runs each scenario in isolation (own InMemoryTraceSink + output dir; run_world_pressure
restores a fresh, non-persisted world), concurrently via asyncio.gather. For each scenario
it persists input / per-agent / world-prompt / deterministic-checks / judge results, then
writes a summary.{json,md}. Production code is untouched; the judge reuses an existing
scene's provider.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene, extract_json
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge import judge_world_pressure
from tuning.judge_llm import create_judge
from tuning.phase_harness.world_pressure import run_world_pressure
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import focus_of, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "world_pressure.json"
_VALID_DRIVE = {"authority", "threat", "obligation", "event"}
_VALID_URGENCY = {"low", "normal", "high", "critical"}
_CRITERIA = ("boundary", "reasonableness", "fields")
_EMPTY_SENTINEL = "@empty_location"


# ---------------------------------------------------------------------------
# Placeholder resolution
# ---------------------------------------------------------------------------

def resolve_placeholders(
    scenario: dict, *, agent_locations: dict, id_by_name: dict, empty_location: str
) -> dict:
    """Resolve @loc_of:<name> / @empty_location placeholders in a scenario dict."""
    resolved = copy.deepcopy(scenario)

    def resolve_loc(value: Any) -> Any:
        if value == _EMPTY_SENTINEL:
            return empty_location
        if isinstance(value, str) and value.startswith("@loc_of:"):
            name = value.split(":", 1)[1]
            aid = id_by_name.get(name, name)
            return agent_locations.get(aid, value)
        return value

    for b in resolved.get("broadcasts", []):
        if "location_scope" in b:
            b["location_scope"] = resolve_loc(b["location_scope"])
    for a in resolved.get("ambient", []):
        if "location" in a:
            a["location"] = resolve_loc(a["location"])
    return resolved


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------

def _ngrams(s: str, n: int = 2) -> set[str]:
    """Chinese n-grams of a string (for heuristic content-overlap detection)."""
    chars = [c for c in s if "一" <= c <= "鿿"]
    return {"".join(chars[i:i + n]) for i in range(len(chars) - n + 1)}


def _deterministic_checks(
    scenario_meta: dict, resolved: dict, per_agent: list[dict], raw_responses: list[str]
) -> dict[str, Any]:
    findings: dict[str, Any] = {"passed": True, "notes": []}

    # 1. Field validity — lenient re-parse of each per-agent world_pressure response
    #    (per-agent schema: {"goals": [...]}).
    bad_fields: list[str] = []
    for raw in raw_responses:
        try:
            data = extract_json(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            bad_fields.append("raw response 非合法 JSON")
            continue
        for g in (data.get("goals", []) if isinstance(data, dict) else []):
            if not isinstance(g, dict):
                continue
            if not str(g.get("text", "")).strip():
                bad_fields.append("缺少 text 的目标(会被丢弃)")
            dt = str(g.get("drive_type", ""))
            if dt and dt not in _VALID_DRIVE:
                bad_fields.append(f"非规范 drive_type={dt!r}(将被静默映射为 event)")
            ur = str(g.get("urgency", ""))
            if ur and ur not in _VALID_URGENCY:
                bad_fields.append(f"非规范 urgency={ur!r}(将回退 normal)")
    findings["field_issues"] = bad_fields

    # 1b. Semantic leak (heuristic tripwire) — does an agent's goal text overlap with
    #     content it could NOT perceive (a message not addressed to it, or a broadcast
    #     scoped to another location)? Expected empty.
    # Public identity terms (everyone's names + roles) are common knowledge, not private
    # content — sharing them (e.g. a role title like "crown prince") is never a leak. Derived from
    # data, theme-neutral.
    public_terms: list[str] = []
    for a in per_agent:
        public_terms.append(a.get("agent_name", ""))
        for c in (a.get("inputs", {}) or {}).get("co_located", []) or []:
            public_terms += [str(c.get("name", "")), str(c.get("role", ""))]
    public_grams = set().union(*(_ngrams(t) for t in public_terms)) if public_terms else set()

    semantic_leaks: list[str] = []
    for a in per_agent:
        aname = a["agent_name"]
        inp = a.get("inputs", {}) or {}
        loc = inp.get("location_id")
        # Publicly perceivable text for this agent — co-located names/roles, its own
        # messages, perceivable broadcasts, ambient. n-grams shared with these are NOT leaks.
        perceivable: list[str] = [aname]
        for c in inp.get("co_located", []) or []:
            perceivable += [str(c.get("name", "")), str(c.get("role", ""))]
        for m in inp.get("injected_messages", []) or []:
            perceivable.append(str(m.get("content", "")))
        for b in inp.get("injected_broadcasts", []) or []:
            perceivable.append(str(b.get("content", "")))
        for e in inp.get("injected_ambient", []) or []:
            perceivable.append(str(e.get("content", "")))
        perceivable_grams = set().union(*(_ngrams(t) for t in perceivable)) if perceivable else set()

        forbidden: list[str] = []
        for m in resolved.get("messages", []):
            if str(m.get("to", "")) != aname:
                forbidden.append(str(m.get("content", "")))
        for b in resolved.get("broadcasts", []):
            scope = b.get("location_scope")
            if scope and "@" not in str(scope) and scope != loc:
                forbidden.append(str(b.get("content", "")))
        goal_text = " ".join(
            g.get("text", "") for g in (a.get("outputs", {}) or {}).get("external_goals", [])
        )
        goal_grams = _ngrams(goal_text)
        for content in forbidden:
            shared = (_ngrams(content) & goal_grams) - perceivable_grams - public_grams
            if shared:
                semantic_leaks.append(
                    f"{aname} 的目标疑似引用了不可感知内容(共享词:{sorted(shared)[:3]}) ← 「{content[:24]}…」"
                )
    if semantic_leaks:
        findings["semantic_leaks"] = semantic_leaks
        findings["passed"] = False

    # NOTE: relational reasonableness (e.g. treating a high-trust ally as a threat) is
    # left to the LLM judge — it has the co-located relations and the sharpened rubric.
    # Don't add a deterministic "threat goal mentions an ally" check: it's too noisy, since it
    # can't tell "order an ally to stand guard" from "watch an ally for defection".

    # 2. Location hard boundary — for a broadcast scoped to the empty location,
    #    no agent is there, so every agent's goals must be empty.
    # Compare scope (a location_id) against agents' location_id (not the display name).
    agent_loc_ids = {a.get("inputs", {}).get("location_id") for a in per_agent}
    empty_scoped = [b for b in resolved.get("broadcasts", [])
                    if b.get("location_scope") and "@" not in str(b.get("location_scope"))
                    and b["location_scope"] not in agent_loc_ids]
    if empty_scoped:
        offenders = [a["agent_name"] for a in per_agent
                     if (a.get("outputs", {}) or {}).get("external_goals")]
        findings["location_boundary_empty"] = {
            "scoped_to": [b["location_scope"] for b in empty_scoped],
            "expected": "all goals empty",
            "offending_agents": offenders,
            "passed": not offenders,
        }
        if offenders:
            findings["passed"] = False

    # 3. Message recipient boundary — each injected message must appear only in its
    #    recipient's perceivable inputs (guards the harness routing).
    leaks: list[str] = []
    for m in resolved.get("messages", []):
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        carriers = [a["agent_name"] for a in per_agent
                    if any(im.get("content") == content
                           for im in (a.get("inputs", {}) or {}).get("injected_messages", []))]
        if len(carriers) > 1:
            leaks.append(f"消息「{content[:20]}…」出现在多个角色: {carriers}")
    if leaks:
        findings["recipient_leaks"] = leaks
        findings["passed"] = False

    if bad_fields:
        findings["passed"] = False
    return findings


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

    await run_world_pressure(
        container, config, world_id,
        scenario=resolved, sink=sink, trace_dir=str(scenario_dir),
    )

    per_agent = [{"agent_name": p.outputs.get("agent_name", p.agent_id),
                  "inputs": p.inputs, "outputs": p.outputs} for p in sink.phases]
    # Per-agent world_pressure → one LLM call per signal-bearing agent.
    world_calls = [c for c in sink.llm_calls if c.scene == LLMScene.WORLD_PRESSURE.value]
    raw_responses = [c.response_content for c in world_calls]

    det = _deterministic_checks(scenario_meta, resolved, per_agent, raw_responses)

    # Dedicated judge router: all scenes map to the judge provider, independent of the
    # production router; wrapped so the judge call is traced.
    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_world_pressure(
        judge_router, scenario_meta=scenario_meta, per_agent=per_agent,
        det_findings=det, judge_scene=judge_scene,
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", {"meta": scenario_meta, "resolved": resolved})
    write_json(scenario_dir / "agents.json", per_agent)
    write_json(scenario_dir / "prompt.json", {
        "world_pressure_calls": [
            {"prompt": c.prompt_messages, "response": c.response_content} for c in world_calls
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
        + det.get("field_issues", []) + det.get("recipient_leaks", []) + det.get("semantic_leaks", [])
        + (det.get("location_boundary_empty", {}).get("offending_agents", []) and
           [f"location 越界: {det['location_boundary_empty']['offending_agents']}"] or []),
    }


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_world_pressure(
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
    """Run the full world_pressure validation suite. Returns the summary dict.

    The judge uses its own LLM (``llm.judge``, overridable per run with ``judge_model``)
    — decoupled from the production router being evaluated. Tests inject
    ``judge_provider`` directly to avoid a real model call.
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

    out_dir = Path(trace_dir) / world_id / "validation" / "world_pressure"

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
        "world_pressure", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md, summary_extra={"empty_location_used": empty_location},
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="world_pressure", avg_label="三项均值",
        columns=[
            ("重点", focus_of),
            ("boundary", lambda r: r["scores"]["boundary"]),
            ("reason.", lambda r: r["scores"]["reasonableness"]),
            ("fields", lambda r: r["scores"]["fields"]),
        ],
        header_extra="  ·  空地点占位: `" + summary["empty_location_used"] + "`",
    )
