"""event stage validation: scenarios + deterministic checks + LLM judge.

Validates the two author-layer paths (the two EVENT_TIMING LLM calls in engine/event.py):

- kind=gate — `_passes_llm_check` (the pacing gate). Deterministic checks: the raw verdict is
  valid JSON, reason is non-empty (reason-first), inject is a bool, and the briefing/prompt has
  no step or id leak. The LLM judge scores pacing judgment quality.
- kind=plan — `_generate_event_plan` (event generation). Deterministic checks: a valid event
  comes out (at least one channel), reason/narrative_desc are non-empty, severity/urgency/
  is_positive are valid enums, recipients/location resolve through IndexedRef, and the
  briefing/prompt/output have no step or id leak. The LLM judge scores
  groundedness/craft/discipline.

Zero production intrusion: a standalone EventSystem + InMemory snapshot; baseline ./data is
never written. The semantic judge isn't fed answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.models import Config
from core.container import Container
from core.interfaces.llm import LLMProvider, LLMScene, extract_json
from core.interfaces.severity import Severity
from core.interfaces.urgency import Urgency
from core.logging import get_logger
from world.initializer import WorldInitializer

from tuning.judge_event import judge_event, criteria_for
from tuning.judge_llm import create_judge
from tuning.phase_harness.event import run_event
from tuning.trace import InMemoryTraceSink, write_json, dump_llm_calls
from tuning.validation_suite import load_scenarios, make_judge_router, run_suite, summary_md, text_leak

logger = get_logger(__name__)

_DEFAULT_SCENARIOS = Path(__file__).parent / "scenarios" / "event.json"
_VALID_SEVERITY = {s.value for s in Severity}
_VALID_URGENCY = {u.value for u in Urgency}


def _check_gate(outputs: dict, raw: dict | None) -> list[str]:
    v: list[str] = []
    if not isinstance(raw, dict):
        v.append("门原始输出非合法 JSON(extract_json 失败)")
        return v
    if "inject" not in raw or not isinstance(raw.get("inject"), bool):
        v.append(f"门缺少合法 inject 布尔字段(得到 {raw.get('inject')!r})")
    reason = str(raw.get("reason", "")).strip()
    if not reason:
        v.append("门缺少 reason(reason-first 契约)")
    if outputs.get("inject") is None:
        v.append("门未产出 inject 解析结果")
    return v


def _enum_value(raw: Any) -> str:
    """Return the enum's canonical value.

    Production parsing already turns the LLM's "medium" into a ``Severity`` / ``Urgency`` enum. Both
    are ``(str, Enum)``, so ``str()`` yields ``'Severity.MEDIUM'``, not ``'medium'``; calling
    ``str()`` directly would flag valid output as invalid (a false-positive tripwire).
    """
    return str(getattr(raw, "value", raw))


def _check_plan(outputs: dict, raw: dict | None) -> list[str]:
    v: list[str] = []
    plan = outputs.get("plan")
    if not plan:
        v.append("未产出有效事件(plan 为空 —— 通道皆缺或解析失败)")
        return v
    if not str(plan.get("narrative_desc", "")).strip():
        v.append("narrative_desc 为空")
    if isinstance(raw, dict) and not str(raw.get("reason", "")).strip():
        v.append("缺少 reason(reason-first lite-CoT 前置字段)")
    bc, msg = plan.get("broadcast"), plan.get("message")
    if not (bc or msg or plan.get("spawn") or plan.get("alter") or plan.get("destroy")):
        v.append("事件无任何通道(spawn/alter/destroy/broadcast/message 皆 null)")
    if bc:
        sev = _enum_value(bc.get("severity", ""))
        if sev not in _VALID_SEVERITY:
            v.append(f"broadcast.severity 非法:{sev!r}")
    if msg:
        urg = _enum_value(msg.get("urgency", ""))
        if urg not in _VALID_URGENCY:
            v.append(f"message.urgency 非法:{urg!r}")
        if not (msg.get("recipient_ids") or []):
            v.append("message 无有效 recipients(IndexedRef resolve 后为空)")
    ip = plan.get("is_positive")
    if ip not in (True, False, None):
        v.append(f"is_positive 非法:{ip!r}")
    return v


def _check_leaks(outputs: dict, prompt_text: str, agent_ids: set[str], location_ids: set[str]) -> list[str]:
    """Record a leak when a step or an agent/location id appears anywhere in the briefing, the actual LLM prompt, or the output."""
    v: list[str] = []
    # Briefing (text after downward translation)
    leak = text_leak(outputs.get("brief", ""), agent_ids=agent_ids, location_ids=location_ids)
    if leak:
        v.append(f"简报{leak}")
    # The prompt actually sent to the LLM
    leak = text_leak(prompt_text, agent_ids=agent_ids, location_ids=location_ids)
    if leak:
        v.append(f"LLM prompt {leak}")
    # Narrative content from the plan (entities / broadcast / message / narrative_desc)
    plan = outputs.get("plan") or {}
    texts = [plan.get("narrative_desc", "")]
    if plan.get("broadcast"):
        texts.append(plan["broadcast"].get("content", ""))
    if plan.get("message"):
        texts.append(plan["message"].get("content", ""))
    for slot in ("spawn", "alter", "destroy"):
        thing = plan.get(slot) or {}
        texts.extend(str(thing.get(k, "")) for k in ("name", "description", "content", "observation"))
    for t in texts:
        leak = text_leak(t, agent_ids=agent_ids, location_ids=location_ids)
        if leak:
            v.append(f"产出内容{leak}")
    return v


def _deterministic_checks(
    outputs: dict, raw: dict | None, prompt_text: str,
    agent_ids: set[str], location_ids: set[str],
) -> dict[str, Any]:
    v: list[str] = []
    if outputs.get("error"):
        v.append(f"运行异常: {outputs['error']}")
    kind = outputs.get("kind", "plan")
    if kind == "gate":
        v += _check_gate(outputs, raw)
    else:
        v += _check_plan(outputs, raw)
    v += _check_leaks(outputs, prompt_text, agent_ids, location_ids)
    return {"passed": not v, "field_violations": v}


async def _run_scenario(
    container: Container, config: Config, world_id: str, scenario_meta: dict,
    *, agent_ids: set[str], location_ids: set[str], out_dir: Path,
    judge_provider: LLMProvider, judge_scene: LLMScene,
) -> dict[str, Any]:
    name = scenario_meta["name"]
    kind = scenario_meta.get("kind", "plan")
    crit = criteria_for(kind)
    scenario_dir = out_dir / name
    sink = InMemoryTraceSink()

    await run_event(container, config, world_id, scenario=scenario_meta, sink=sink,
                    trace_dir=str(scenario_dir))
    phase = next((p for p in sink.phases if p.phase == "stage.event"), None)
    inputs = phase.inputs if phase else {}
    outputs = phase.outputs if phase else {}

    # Pick this stage's EVENT_TIMING call (one each for gate or plan) for the raw JSON, leak check and web display.
    event_calls = [c for c in sink.llm_calls if c.scene == LLMScene.EVENT_TIMING.value]
    last = event_calls[-1] if event_calls else None
    prompt_text = ""
    raw: dict | None = None
    if last is not None:
        prompt_text = "\n".join(
            m.get("content", "") if isinstance(m, dict) else str(m) for m in (last.prompt_messages or [])
        )
        try:
            raw = extract_json(last.response_content or "")
        except (json.JSONDecodeError, ValueError):
            raw = None

    det = _deterministic_checks(outputs, raw, prompt_text, agent_ids, location_ids)

    semantic_judged = not outputs.get("error")
    if not semantic_judged:
        judge = {c: {"score": 0, "rationale": "运行异常,不评语义", "issues": []} for c in crit}
        judge["overall"] = "运行异常"
        judge_calls: list = []
    else:
        judge_router = make_judge_router(judge_provider, sink, config)
        judge = await judge_event(
            judge_router, scenario_meta=scenario_meta, outputs=outputs,
            det_findings=det, judge_scene=judge_scene)
        judge_calls = [{"scene": c.scene, "prompt": c.prompt_messages, "response": c.response_content}
                       for c in sink.llm_calls if c.scene == judge_scene.value]

    event_call_view = (
        {"scene": last.scene, "prompt": last.prompt_messages, "response": last.response_content}
        if last is not None else None
    )

    scenario_dir.mkdir(parents=True, exist_ok=True)
    # Every LLM call from this run (upstream cognition and the judge included), shown one by one in the developer tools.
    dump_llm_calls(sink, scenario_dir / "llm_calls.json")
    write_json(scenario_dir / "input.json", scenario_meta)
    write_json(scenario_dir / "event.json", {"inputs": inputs, "outputs": outputs})
    write_json(scenario_dir / "prompt.json", {"event_call": event_call_view,
                                         "judge_call": judge_calls[-1] if judge_calls else None})
    write_json(scenario_dir / "checks.json", det)
    write_json(scenario_dir / "judge.json", judge)

    return {
        "name": name, "kind": kind,
        "scores": {c: judge.get(c, {}).get("score", 0) for c in crit},
        "overall": judge.get("overall", ""),
        "semantic_judged": semantic_judged,
        "deterministic_passed": det.get("passed", True),
        "issues": [i for c in crit for i in judge.get(c, {}).get("issues", [])]
        + det.get("field_violations", []),
    }


async def validate_event(
    container: Container, config: Config, world_id: str,
    *, scenarios_path: str | None = None, trace_dir: str = "./data/tuning_traces",
    judge_model: str | None = None, judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run the full EventSystem validation suite. Returns the summary dict."""
    scenarios = load_scenarios(scenarios_path, _DEFAULT_SCENARIOS)
    judge_llm = judge_provider or create_judge(config, judge_model)

    world = await WorldInitializer(container).restore(world_id)
    agent_ids = set(world.agents.keys())
    location_ids = set(world.environment.snapshot_state().get("location_names", {}).keys())
    out_dir = Path(trace_dir) / world_id / "validation" / "event"

    async def _one(meta: dict) -> dict:
        return await _run_scenario(container, config, world_id, meta, agent_ids=agent_ids,
                                   location_ids=location_ids, out_dir=out_dir,
                                   judge_provider=judge_llm, judge_scene=judge_scene)

    return await run_suite(
        "event", world_id, scenarios, _one, out_dir=out_dir, criteria=_ALL_CRITERIA,
        render_md=_summary_md, default_kind="plan",
    )


_ALL_CRITERIA = ("pacing", "groundedness", "concreteness", "craft", "discipline")


def _summary_md(summary: dict) -> str:
    return summary_md(
        summary, title="event", avg_label="语义均值",
        columns=[
            ("kind", lambda r: r.get("kind")),
            ("语义分", lambda r: (
                "、".join(f"{c}={v}" for c, v in r["scores"].items())
                if r.get("semantic_judged") else "—"
            )),
        ],
    )
