"""action stage validation suite: scenarios + deterministic checks + LLM judge.

Restores a built world and, per scenario, drives the real production executors /
``runtime._arbitrate_execution`` through the four phases (before / start / ongoing / completion)
or the arbitration verdict, then applies structural deterministic checks + an LLM judge
(syntax + semantics of each phase's output).

Deterministic checks (always run) are hard tripwires: no agent id and no "第N步/steps" in any
narrative/outcome/memory; every failure carries a failure_reason; relation_updates target valid
agents with bounded magnitudes; emotions canonical; new entity states non-empty; arbitration has
no double occupancy and no ids in rejection text. The judge carries the semantics. Zero
production intrusion (only scene/memory injection is scaffolding); the judge comes from
``llm.judge``.
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

from tuning.judge_action import judge_action
from tuning.judge_llm import create_judge
from tuning.phase_harness.action import run_action
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import STEP_LEAK, focus_of, load_scenarios, make_judge_router, run_suite, summary_md

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "action.json"
_CRITERIA = ("fields", "outcome_fidelity", "consequence_realism")
_NARRATION_SCENE = LLMScene.AGENT_ACTION_NARRATION.value


# ---------------------------------------------------------------------------
# Deterministic checks (structural tripwire)
# ---------------------------------------------------------------------------

def _errand_texts(order: dict) -> list[str]:
    """The narrative-layer text an errand carries.

    Filter by exclusion rather than listing keys: ``*_id`` fields are code-layer coordinates and
    ``bearer``/``recipient`` are already translated to names upstream; every other string gets
    scanned. If an errand gains another text field it's scanned automatically. A safety net with a
    hole is worse than none, because the missed field is the one nobody looks at twice.
    """
    return [
        str(v) for k, v in order.items()
        if isinstance(v, str) and not k.endswith("_id")
    ]


def _texts_of_result(r: dict) -> list[str]:
    # observation (the onlooker's downgraded text) is a narrative field written into onlookers' memory and embedded, so it's covered by the id/step leak scan too.
    out: list[str] = [str(r.get("outcome") or ""), str(r.get("observation") or ""),
                      str(r.get("factual_memory") or ""), str(r.get("failure_reason") or "")]
    out += [str(d.get("line") or "") for d in (r.get("dialogue") or [])]
    out += [str(e.get("factual_memory") or "") for e in (r.get("target_effects") or [])]
    # A condition persists and keeps re-entering prompts, so it needs guarding against ids / steps even more than a one-off outcome.
    out += [str(e.get("condition") or "") for e in (r.get("target_effects") or [])]
    out += [str(c.get("perception") or "") for c in (r.get("entity_state_changes") or [])]
    # Made things stay in the world, enter other agents' reachable lists and get embedded, so they
    # need guarding against ids / steps even more than a one-off outcome.
    for s in r.get("entity_spawns") or []:
        out += [str(s.get("name") or ""), str(s.get("description") or ""),
                str(s.get("perception") or "")]
    # Text sent out with an errand is delivered verbatim and written into someone else's memory, so it's guarded against ids / steps as well.
    for o in r.get("errand_orders") or []:
        out += _errand_texts(o)
    out += [str(e.get("condition") or "") for e in (r.get("npc_effects") or [])]
    return [t for t in out if t]


def _deterministic_checks(phase: dict, agent_ids: set[str]) -> dict[str, Any]:
    v: list[str] = []
    outputs = phase.get("outputs", {}) or {}

    def _scan_text(label: str, text: str) -> None:
        for aid in agent_ids:
            if aid and aid in text:
                v.append(f"{label}: 文本泄漏 agent id「{aid}」→「{text[:30]}…」")
                break
        if STEP_LEAK.search(text):
            v.append(f"{label}: 叙事层出现 step(『第N步/N步/steps』)→「{text[:30]}…」")

    if outputs.get("kind") == "arbitration":
        verdicts = outputs.get("verdicts", [])
        actors = [vd.get("actor") for vd in verdicts]
        if len(actors) != len(set(actors)):
            v.append("仲裁: 同一 actor 出现多条裁决(疑似双占)")
        for vd in verdicts:
            _scan_text(f"仲裁/{vd.get('actor')}", str(vd.get("outcome") or ""))
            if vd.get("kind") == "被拒" and vd.get("succeeded"):
                v.append(f"仲裁/{vd.get('actor')}: 被拒却 succeeded=True")
        return {"passed": not v, "field_violations": v}

    # lifecycle
    phases = outputs.get("phases", {}) or {}
    start = phases.get("start") or {}
    results: list[dict] = []
    if start.get("kind") == "immediate" and start.get("result"):
        results.append(start["result"])
    for t in phases.get("tick", []) or []:
        for n in t.get("narratives", []):
            _scan_text("行动中", str(n.get("narrative") or ""))
    for r in (phases.get("complete", {}) or {}).get("results", []) or []:
        results.append(r)

    for r in results:
        for txt in _texts_of_result(r):
            _scan_text("产出", txt)
        if not r.get("succeeded", True) and not r.get("adjudication_failed") and not r.get("failure_reason"):
            v.append("判为失败却没有 failure_reason(渲染层将无从说明为何没成)")
        if not str(r.get("factual_memory") or "").strip():
            v.append("factual_memory 为空")
        for u in r.get("relation_updates") or []:
            if u.get("target") not in agent_ids:
                v.append(f"relation_updates 目标非法={u.get('target')!r}")
            for k in ("trust_delta", "affection_delta"):
                d = u.get(k)
                if not isinstance(d, (int, float)) or not (-1.0 <= float(d) <= 1.0):
                    v.append(f"relation_updates {k} 越界={d!r}")
        for e in r.get("target_effects") or []:
            et = e.get("emotion_type")
            if et is not None and parse_emotion_type(str(et)) == EmotionType.NEUTRAL and str(et).lower() not in ("neutral", "平静", "中性"):
                v.append(f"target_effect emotion_type 非规范={et!r}")
            vd = e.get("vitality_damage")
            if not isinstance(vd, (int, float)) or not (0.0 <= float(vd) <= 1.0):
                v.append(f"target_effect vitality_damage 越界={vd!r}")
            # The three states are mutually exclusive: leaving a condition and clearing it at once is a self-contradictory verdict.
            if e.get("condition") and e.get("condition_cleared"):
                v.append("target_effect 同时施加与解除处境")
        for c in r.get("entity_state_changes") or []:
            if not str(c.get("new_state") or "").strip():
                v.append("entity_state_change new_state 为空")
        # Accepted-or-not must line up exactly with whether the errand exists in the world. Ruling "he set
        # off" without placing the errand forks the requester's memory from the world on the spot; ruling
        # "declined" while placing it sends someone on an errand nobody gave.
        orders = r.get("errand_orders") or []
        if str(r.get("action_type") or "") == "errand" and not r.get("adjudication_failed"):
            if r.get("succeeded") and not orders:
                v.append("判为接下却没有 errand_order(世界里不会有这趟差事)")
            if not r.get("succeeded") and orders:
                v.append("判为不接却下了 errand_order(没人托的人上了路)")
        # What was entrusted is privileged: onlookers can see he went over and said a few words, not what
        # was said. Taken regardless of variant, as above: a new variant's text is still the entrusted content and must not leak to onlookers either.
        observation = str(r.get("observation") or "")
        for line in (t for o in orders for t in _errand_texts(o) if t.strip()):
            if line in observation:
                v.append("observation 漏出了交代的内容(privileged)")
        for s in r.get("entity_spawns") or []:
            # A nameless thing can't be made (spawn_entity rejects it outright); declaring one that can't be made is a wasted declaration.
            if not str(s.get("name") or "").strip():
                v.append("entity_spawn name 为空")
            # Don't check for "failed but left something behind": what's left is orthogonal to whether the
            # intent was achieved (see _WorkVerdict). A half-made thing is exactly the trace a failed beat should leave.

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

    await run_action(container, config, world_id, scenario=scenario_meta, sink=sink,
                     trace_dir=str(scenario_dir))

    phase = next((p for p in sink.phases if p.phase == "stage.action"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    phase_dict = {"inputs": inputs, "outputs": outputs}

    det = _deterministic_checks(phase_dict, agent_ids)

    judge_router = make_judge_router(judge_provider, sink, config)
    judge = await judge_action(
        judge_router, scenario_meta=scenario_meta, inputs=inputs, outputs=outputs,
        det_findings=det, judge_scene=judge_scene)

    narration_calls = [
        {"agent_id": c.agent_id, "prompt": c.prompt_messages, "response": c.response_content}
        for c in sink.llm_calls if c.scene == _NARRATION_SCENE
    ]

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "phases.json", phase_dict)
    write_json(scenario_dir / "prompt.json", {"narration_calls": narration_calls})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name,
        "kind": scenario_meta.get("kind", "lifecycle"),
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

async def validate_action(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full action validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    # Don't pass enable_thinking here: thinking is an endpoint property set in ``llm.judge.params`` (see create_judge).
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())

    out_dir = Path(trace_dir) / world_id / "validation" / "action"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(
            container, config, world_id, meta, agent_ids=agent_ids, out_dir=out_dir,
            judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "action", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md, default_kind="lifecycle",
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="action", avg_label="三项均值",
        columns=[
            ("类型", lambda r: r.get("kind")),
            ("重点", focus_of),
            ("fields", lambda r: r["scores"]["fields"]),
            ("fidelity", lambda r: r["scores"]["outcome_fidelity"]),
            ("realism", lambda r: r["scores"]["consequence_realism"]),
        ],
    )
