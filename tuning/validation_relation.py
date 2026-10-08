"""relation stage validation: scenarios + deterministic checks + LLM judge.

Validates the two relation paths (agent/relation.py and agent/relation_evolution.py):

- kind=perceive — `RelationSystem.perceive` (internally `_compute_perceived`): the objective
  trust/affection baseline is colored by current emotion + recent-memory bias. Pure rules, no
  LLM, so deterministic checks only (rest identity, bounded bend-not-flip, correct sign,
  optional numeric anchor bands).
- kind=evolve — `RelationEvolution.evaluate`: the agent's objective third-person
  judgment of label + summary. Deterministic checks cover the structural invariants (valid labels,
  blood / structural bonds kept via must_keep, no id/step leak, no changes where none belong);
  the LLM judge carries semantic quality (fidelity + objectivity).

Zero production intrusion: isolated dry-run (shadow store + InMemory vector store); baseline
./data is never written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


from agent.relation import K_EMOTION_PULL, K_MEMORY_PULL
from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_llm import create_judge
from tuning.judge_relation import judge_relation
from tuning.phase_harness.relation import run_relation
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import load_scenarios, make_judge_router, run_suite, summary_md, text_leak

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "relation.json"
_CRITERIA = ("fidelity", "objectivity")
_EPS = 1e-6
# Labels take the form "type:role" or just "type". The type part must never be one of the prompt's
# category meta-words ("血亲" / "结构性绑定" / "叙事性定性"): those name the categories used to explain
# the evolution rules, not actual relation types. An LLM emitting a meta-category as the type
# prefix (e.g. "叙事性定性:<a deceased rival>") is a format violation.
_TAXONOMY_META = ("血亲", "结构性绑定", "叙事性定性", "关系类型")


def _check_perceive(outputs: dict, scenario_meta: dict) -> list[str]:
    v: list[str] = []
    for row in outputs.get("perceptions", []) or []:
        nm = row.get("name", "?")
        obj, per = row.get("objective", {}), row.get("perceived", {})
        emo = row.get("emotion", {})
        bias = float(row.get("memory_bias", 0.0))
        intensity, valence = float(emo.get("intensity", 0.0)), float(emo.get("valence", 0.0))
        emotion_pull = intensity * valence * K_EMOTION_PULL
        memory_pull = bias * K_MEMORY_PULL
        max_disp = abs(emotion_pull) + abs(memory_pull)
        for dim in ("trust", "affection"):
            o, p = float(obj.get(dim, 0.0)), float(per.get(dim, 0.0))
            delta = p - o
            # rest identity: no emotional shift and no memory bias → perceived always equals objective.
            if abs(emotion_pull) < _EPS and abs(memory_pull) < _EPS and abs(delta) > _EPS:
                v.append(f"{nm}.{dim} rest-identity 失败:中性无偏置却 {o}→{p}")
            # bend-not-flip: the shift can't exceed |emotion_pull| + |memory_pull| (clamping only makes it smaller).
            if abs(delta) > max_disp + 1e-4:
                v.append(f"{nm}.{dim} 位移越界:|{p}-{o}|={abs(delta):.3f} > 上界{max_disp:.3f}")
            # Correct sign: positive net pull → perceived ≥ objective, negative → ≤ (clamping to the edge allowed).
            pull = emotion_pull + memory_pull
            if pull > _EPS and p < o - 1e-4:
                v.append(f"{nm}.{dim} 净拉力为正却下降:{o}→{p}")
            if pull < -_EPS and p > o + 1e-4:
                v.append(f"{nm}.{dim} 净拉力为负却上升:{o}→{p}")
        # Optional numeric anchor bands (answers known from objective math, not telegraphed).
        for dim, key in (("trust", "expect_trust_range"), ("affection", "expect_affection_range")):
            rng = row.get(key)
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                p = float(per.get(dim, 0.0))
                if not (float(rng[0]) - 1e-4 <= p <= float(rng[1]) + 1e-4):
                    v.append(f"{nm}.{dim} 越出预期带:{p} ∉ [{rng[0]}, {rng[1]}]")
    return v


def _check_evolve(outputs: dict, agent_ids: set[str]) -> list[str]:
    v: list[str] = []
    for t in outputs.get("targets", []) or []:
        nm = t.get("name", "?")
        after = t.get("after", {}) or {}
        after_labels = list(after.get("labels", []) or [])
        # Valid label: non-empty string, no id/step leak, type part isn't a category meta-word.
        for lab in t.get("applied_labels", []) or []:
            s = str(lab).strip()
            if not s:
                v.append(f"{nm} 产出空 label")
                continue
            leak = text_leak(lab, agent_ids=agent_ids)
            if leak:
                v.append(f"{nm} label {leak}")
            kind_seg = s.split(":", 1)[0].strip()
            if kind_seg in _TAXONOMY_META:
                v.append(f"{nm} label 非法:类型段误用分类元词「{kind_seg}」(应为具体关系类型)→「{s}」")
        # Blood / structural bonds in must_keep must still be in after.labels.
        for keep in t.get("must_keep_labels", []) or []:
            if keep not in after_labels:
                v.append(f"{nm} 不可删 label「{keep}」被移除(现为 {after_labels})")
        # labels changed where nothing should have.
        if t.get("expect_no_change") and t.get("labels_changed"):
            v.append(f"{nm} 预期无变动却改写 labels:{t.get('before',{}).get('labels')}→{after_labels}")
        # summary has no id/step leak.
        leak = text_leak(after.get("summary", ""), agent_ids=agent_ids)
        if leak:
            v.append(f"{nm} summary {leak}")
    return v


def _deterministic_checks(outputs: dict, agent_ids: set[str], scenario_meta: dict) -> dict[str, Any]:
    v: list[str] = []
    if outputs.get("error"):
        v.append(f"运行异常: {outputs['error']}")
    if outputs.get("kind") == "perceive":
        v += _check_perceive(outputs, scenario_meta)
    else:
        v += _check_evolve(outputs, agent_ids)
    return {"passed": not v, "field_violations": v}


async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, agent_ids: set[str], out_dir: Path, judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    kind = scenario_meta.get("kind", "evolve")
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_relation(container, config, world_id, scenario=scenario_meta, sink=sink,
                       trace_dir=str(scenario_dir))
    phase = next((p for p in sink.phases if p.phase == "stage.relation"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}
    det = _deterministic_checks(outputs, agent_ids, scenario_meta)

    # The semantic judge only scores evolve (perceive is pure rules, so semantic review doesn't apply, like memory_maintain's decay).
    semantic_judged = kind == "evolve" and not outputs.get("error") and bool(outputs.get("targets"))
    if not semantic_judged:
        judge = {c: {"score": 0, "rationale": "纯规则/无候选,不评语义", "issues": []} for c in _CRITERIA}
        judge["overall"] = "纯规则/无候选"
        judge_calls: list = []
    else:
        judge_router = make_judge_router(judge_provider, sink, config)
        judge = await judge_relation(
            judge_router, scenario_meta=scenario_meta, outputs=outputs,
            det_findings=det, judge_scene=judge_scene)
        judge_calls = [{"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
                       for c in sink.llm_calls if c.scene == judge_scene.value]
    # evolve's evaluate LLM call (captured by the traced router), for viewing the raw judge prompt/output on the web.
    evolve_calls = [{"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
                    for c in sink.llm_calls if c.scene == LLMScene.RELATION_LABEL_EVOLUTION.value]

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "relation.json", {"inputs": inputs, "outputs": outputs})
    write_json(scenario_dir / "prompt.json", {"evolve_call": evolve_calls[-1] if evolve_calls else None,
                                         "judge_call": judge_calls[-1] if judge_calls else None})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name, "kind": kind,
        "scores": {c: judge.get(c, {}).get("score", 0) for c in _CRITERIA},
        "overall": judge.get("overall", ""),
        "semantic_judged": semantic_judged,
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in _CRITERIA for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


async def validate_relation(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full relation validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())
    out_dir = Path(trace_dir) / world_id / "validation" / "relation"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(container, config, world_id, meta, agent_ids=agent_ids,
                                   out_dir=out_dir, judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "relation", world_id, scenarios, _one, out_dir=out_dir, criteria=_CRITERIA,
        render_md=_summary_md, default_kind="evolve",
    )


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="relation", avg_label="语义均值(仅 evolve)",
        columns=[
            ("kind", lambda r: r.get("kind")),
            ("fidelity", lambda r: r["scores"]["fidelity"] if r.get("semantic_judged") else "—"),
            ("objectivity", lambda r: r["scores"]["objectivity"] if r.get("semantic_judged") else "—"),
        ],
    )
