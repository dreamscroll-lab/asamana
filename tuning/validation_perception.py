"""perception validation suite: boundary scenarios × knob sets + deterministic invariant checks + A/B diff.

perception is a pure rule subsystem (no LLM generation). Scenarios target threshold / cap
boundaries. This suite:
- runs every scenario once under each knob set (knob_sets);
- deterministic invariant checks (the main measurement): thresholds, per-source caps, dedup;
- A/B diff: baseline knob set vs the others, listing per-agent differences in the selected
  set (change a threshold/cap and see how the boundary moves).

One read-only restore is reused for the whole suite (perception doesn't mutate world state);
zero production code changes.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.perception_layer import _CAPS_BG, _CAPS_MAIN, PerceptionTuning
from config.models import Config
from core.container import Container
from core.logging import get_logger
from engine.clock import GlobalClock
from world.initializer import WorldInitializer

from tuning.phase_harness.perception import perceive_world
from tuning.trace import write_json
from tuning.validation import resolve_placeholders

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "perception.json"
_SCALAR_KNOBS = (
    "threshold_main", "threshold_bg", "ambient_default_strength",
    "importance_high_cutoff", "importance_medium_cutoff",
)


# ---------------------------------------------------------------------------
# Knob set → PerceptionTuning
# ---------------------------------------------------------------------------

def _build_tuning(overrides: dict | None) -> PerceptionTuning:
    """Merge a knob-set override dict (scalar knobs only) onto the production defaults."""
    base = PerceptionTuning()
    if not overrides:
        return base
    kwargs = {f: float(overrides[f]) for f in _SCALAR_KNOBS if f in overrides}
    return dataclasses.replace(base, **kwargs)


def _tuning_view(t: PerceptionTuning) -> dict[str, Any]:
    """JSON-safe view of a tuning's scalar knobs."""
    return {f: getattr(t, f) for f in _SCALAR_KNOBS}


# ---------------------------------------------------------------------------
# Deterministic invariant checks (rule reasonableness)
# ---------------------------------------------------------------------------

def _deterministic_checks(agents: list[dict], tuning: PerceptionTuning) -> dict[str, Any]:
    """Verify the selection obeys the selection rules under this tuning.

    Universal invariants (not scenario-specific): every selected item clears the active
    threshold; per-source counts respect caps; no duplicate (source, content).
    """
    eps = 1e-9
    threshold_v: list[str] = []
    cap_v: list[str] = []
    dedup_v: list[str] = []
    for a in agents:
        is_main = a["is_main_character"]
        threshold = tuning.threshold_main if is_main else tuning.threshold_bg
        caps = _CAPS_MAIN if is_main else _CAPS_BG
        name = a["agent_name"]
        sel = a["selected"]
        by_source: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        for i in sel:
            if i["signal_strength"] < threshold - eps:
                threshold_v.append(f"{name}: {i['source']}「{i['content'][:16]}」强度{i['signal_strength']}<阈值{threshold}")
            by_source[i["source"]] = by_source.get(i["source"], 0) + 1
            key = (i["source"], i["content"])
            if key in seen:
                dedup_v.append(f"{name}: 重复 {i['source']}「{i['content'][:16]}」")
            seen.add(key)
        for src, cnt in by_source.items():
            cap = caps.get(src, 0)
            if cnt > cap:
                cap_v.append(f"{name}: {src} 选中 {cnt} > cap {cap}")

    return {
        "passed": not (threshold_v or cap_v or dedup_v),
        "threshold_violations": threshold_v,
        "cap_violations": cap_v,
        "dedup_violations": dedup_v,
        # info (not pass/fail): for human reading and comparison
        "selected_counts": {a["agent_name"]: len(a["selected"]) for a in agents},
        "importance_dist": _importance_dist(agents),
    }


def _importance_dist(agents: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for a in agents:
        for i in a["selected"]:
            dist[i["importance"]] = dist.get(i["importance"], 0) + 1
    return dist


# ---------------------------------------------------------------------------
# A/B diff (baseline vs variant)
# ---------------------------------------------------------------------------

def _diff_selections(base_agents: list[dict], var_agents: list[dict]) -> dict[str, Any]:
    base_by = {a["agent_id"]: a for a in base_agents}
    out: dict[str, Any] = {}
    for va in var_agents:
        ba = base_by.get(va["agent_id"])
        if ba is None:
            continue
        base_keys = {(i["source"], i["content"]): i for i in ba["selected"]}
        var_keys = {(i["source"], i["content"]): i for i in va["selected"]}
        added = [f"{k[0]}「{k[1][:20]}」" for k in var_keys if k not in base_keys]
        removed = [f"{k[0]}「{k[1][:20]}」" for k in base_keys if k not in var_keys]
        imp_changed = [
            f"{k[0]}「{k[1][:16]}」{base_keys[k]['importance']}→{var_keys[k]['importance']}"
            for k in var_keys if k in base_keys and base_keys[k]["importance"] != var_keys[k]["importance"]
        ]
        if added or removed or imp_changed:
            out[va["agent_name"]] = {"added": added, "removed": removed, "importance_changed": imp_changed}
    return out


# ---------------------------------------------------------------------------
# Suite entry point
# ---------------------------------------------------------------------------

async def validate_perception(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenarios_path: str | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> dict[str, Any]:
    """Run the perception validation suite (boundary scenarios × knob_sets). Returns summary.

    Pure deterministic: invariant checks + A/B diff between knob sets. No LLM.
    """
    path = Path(scenarios_path) if scenarios_path else _DEFAULT_SCENARIOS
    spec = json.loads(path.read_text(encoding="utf-8"))
    scenarios = spec["scenarios"]
    knob_sets = spec.get("knob_sets") or {"baseline": {}}
    tunings = {label: _build_tuning(ov) for label, ov in knob_sets.items()}
    baseline_label = "baseline" if "baseline" in tunings else next(iter(tunings))

    # One read-only restore for the whole suite — perception never mutates world state.
    world = await WorldInitializer(container).restore(world_id)
    name_by_id = {aid: a.personality.soul.name for aid, a in world.agents.items()}
    id_by_name = {v: k for k, v in name_by_id.items()}
    agent_locations = {aid: world.environment.get_body_location(aid) for aid in world.agents}
    all_locations = list(world.environment.space.all_place_ids())
    occupied = {loc for loc in agent_locations.values() if loc}
    empty_location = next((loc for loc in all_locations if loc not in occupied), "nowhere_void")
    run_step = world.current_step + 1
    world_time_label = GlobalClock(world.clock_config, start_step=run_step).current.time_label

    out_dir = Path(trace_dir) / world_id / "validation" / "perception"
    rows: list[dict] = []

    for meta in scenarios:
        resolved = resolve_placeholders(
            meta["scenario"], agent_locations=agent_locations,
            id_by_name=id_by_name, empty_location=empty_location,
        )
        per_knob: dict[str, dict] = {}
        for label, tuning in tunings.items():
            result = await perceive_world(
                world, world_id, run_step, world_time_label, scenario=resolved, tuning=tuning
            )
            per_knob[label] = {
                "agents": result["agents"],
                "checks": _deterministic_checks(result["agents"], tuning),
                "tuning": _tuning_view(tuning),
            }
        diffs = {
            label: _diff_selections(per_knob[baseline_label]["agents"], per_knob[label]["agents"])
            for label in tunings if label != baseline_label
        }

        scen_dir = out_dir / meta["name"]
        write_json(scen_dir / "input.json", {"meta": meta, "resolved": resolved})
        write_json(scen_dir / "selection.json", {label: per_knob[label]["agents"] for label in tunings})
        write_json(scen_dir / "checks.json", {label: per_knob[label]["checks"] for label in tunings})
        write_json(scen_dir / "diff.json", diffs)
        write_json(scen_dir / "tuning.json", {label: per_knob[label]["tuning"] for label in tunings})

        rows.append({
            "name": meta["name"],
            "criteria_focus": meta.get("criteria_focus", []),
            "deterministic_passed": all(per_knob[l]["checks"]["passed"] for l in tunings),
            "knob_passed": {l: per_knob[l]["checks"]["passed"] for l in tunings},
            "diff_counts": {
                l: sum(len(d["added"]) + len(d["removed"]) + len(d["importance_changed"]) for d in diffs[l].values())
                for l in diffs
            },
            "issues": [
                i for l in tunings
                for k in ("threshold_violations", "cap_violations", "dedup_violations")
                for i in per_knob[l]["checks"].get(k, [])
            ],
        })

    summary = {
        "world_id": world_id,
        "generated_at": datetime.now().isoformat(),
        "scenario_count": len(rows),
        "knob_sets": list(tunings),
        "baseline": baseline_label,
        "all_deterministic_passed": all(r["deterministic_passed"] for r in rows),
        "scenarios": rows,
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    logger.info("perception_validation_complete",
                extra={"world_id": world_id, "scenarios": len(rows), "knob_sets": list(tunings)})
    return summary


def _summary_md(summary: dict) -> str:
    lines = [
        "# perception 验证汇总",
        f"- world_id: `{summary['world_id']}`",
        f"- 生成时间: {summary['generated_at']}",
        f"- 场景数: {summary['scenario_count']}  ·  旋钮组: {', '.join(summary['knob_sets'])}  ·  基准: `{summary['baseline']}`",
        f"- 确定性全过: {'✅' if summary['all_deterministic_passed'] else '❌'}",
        "",
        "| 场景 | 重点 | 确定性 | A/B diff |",
        "|---|---|---|---|",
    ]
    for r in summary["scenarios"]:
        det = "✅" if r["deterministic_passed"] else "❌"
        focus = "/".join(r.get("criteria_focus", []))
        diff = ", ".join(f"{l}:{n}" for l, n in (r.get("diff_counts") or {}).items()) or "—"
        lines.append(f"| {r['name']} | {focus} | {det} | {diff} |")
    lines.append("\n## Flagged issues")
    any_issue = False
    for r in summary["scenarios"]:
        if r.get("issues"):
            any_issue = True
            lines.append(f"\n**{r['name']}**")
            lines.extend(f"- {i}" for i in r["issues"])
    if not any_issue:
        lines.append("\n（无）")
    return "\n".join(lines) + "\n"
