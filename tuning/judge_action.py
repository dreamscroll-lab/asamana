"""LLM-as-judge for the action stage (executor lifecycle + arbitration) validation.

Scores one scenario's executor/arbitration outputs across the four phases
(before / start / ongoing / completion) on three criteria:

1. fields (syntax/structure) — fields present, types canonical, no id/step leak, dialogue/JSON
   well-formed.
2. outcome_fidelity (semantics, narrative fidelity) — outcome/dialogue/verdict text is credible
   and fits persona + emotion + scene; no clichés, no parroting expected, no drifting off-scene.
   Dialogue may contain misunderstanding and conflict (don't harmonize it).
3. consequence_realism — success/damage/detection/relation_dir/gap are proportionate to the inputs.

Deduction-based (full marks by default, only penalize citable problems), reusing the decide
judge's negative-constraint / don't-bind-the-answer safeguards.
Arbitration scenarios score fields/consequence on whether the verdicts are consistent (no double
occupancy, sound rejection reasons, passive joins attributed correctly).
The judge uses a dedicated LLMRouter (``llm.judge``), and the judge call itself is traced.
"""

from __future__ import annotations

import json
from typing import Any

from tuning.plan_view import render_target

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgement

logger = get_logger(__name__)

_CRITERIA = ("fields", "outcome_fidelity", "consequence_realism")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「行动(action)」阶段:一个已选定的行动如何被**执行**——"
    "行动前(可行性/裁决)、行动开始(start)、行动中(tick 进度)、行动完成(complete/中断)四个阶段的产出,"
    "以及多 agent 并发时的**裁决**结果。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调;"
    "**不要因为它没产出你设想的那个结果就扣分**——只要产出在该处境下站得住、不违反下列约束,就视为合理。"
    "对白/裁决允许双方误解、冲突、不欢而散,**不要因为'不够和谐/没达成'而扣分**。只输出 JSON。"
)


def _phase_block(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines: list[str] = []
    kind = outputs.get("kind")
    if kind == "arbitration":
        lines.append("### 类型:并发裁决")
        lines.append("同步行动:")
        for a in inputs.get("actions", []):
            lines.append(f"  - {a.get('actor')}: {a.get('action_type')} → {a.get('target')}（{a.get('estimated_steps')}步）")
        lines.append("裁决结果:")
        for v in outputs.get("verdicts", []):
            lines.append(f"  - {v.get('actor')}: 【{v.get('kind')}】succeeded={v.get('succeeded')} "
                         f"passive={v.get('is_passive_join')} ongoing={v.get('ongoing')} ｜ {v.get('outcome')}")
        return "\n".join(lines)

    # lifecycle
    tier = "主角" if inputs.get("is_main_character") else "背景"
    traits = "、".join(inputs.get("core_traits") or []) or "—"
    lines.append(f"### 行动者:{inputs.get('actor_name')}（{tier}，位于 {inputs.get('location')}；性格：{traits}）")
    if inputs.get("core_values"):
        lines.append("  价值观：" + "、".join(inputs["core_values"]))
    lines.append(f"  体力:{inputs.get('vitality')}")
    emo = inputs.get("emotion")
    if emo:
        lines.append(f"  此刻情绪：{emo.get('primary')}（强度{emo.get('intensity')}，效价{emo.get('valence')}）")
    lines.append(
        f"  行动:{inputs.get('action_type')}"
        f"（{render_target(inputs.get('target'))}，{inputs.get('estimated_steps')}步）"
    )
    lines.append(f"  行动描述:{inputs.get('action_description')}")
    if inputs.get("expected_outcome"):
        lines.append(f"  期望结果:{inputs.get('expected_outcome')}")
    if inputs.get("co_present"):
        lines.append("  同处在场者:" + "、".join(inputs["co_present"]))
    if inputs.get("seeded_memories"):
        lines.append("  相关记忆:" + "；".join(inputs["seeded_memories"]))

    phases = outputs.get("phases", {}) or {}
    pre = phases.get("pre") or {}
    if pre.get("checked"):
        lines.append(f"  【行动前·可行性】ok={pre.get('ok')} ｜ {pre.get('reason') or '通过'}")
    start = phases.get("start") or {}
    if start.get("kind") == "immediate":
        lines.append("  【行动开始/完成·即时】" + _result_line(start.get("result", {})))
    elif start.get("kind") == "multi_step":
        s = start.get("state", {})
        lines.append(f"  【行动开始·多步】参与者={s.get('participant_ids')} 预计={s.get('estimated_steps')}步 目的={s.get('purpose')}")
    for t in phases.get("tick", []) or []:
        for n in t.get("narratives", []):
            lines.append(f"  【行动中】{n.get('agent')}：{n.get('narrative')}")
    comp = phases.get("complete") or {}
    if comp.get("kind") and comp.get("kind") != "immediate(同 start)":
        lines.append(f"  【行动完成·{comp.get('kind')}】")
        for r in comp.get("results", []):
            lines.append("    " + _result_line(r))
    return "\n".join(lines)


def _result_line(r: dict[str, Any]) -> str:
    who = f"[{r.get('actor')}视角] " if r.get("actor") else ""
    parts = [f"{who}succeeded={r.get('succeeded')}"]
    if r.get("failure_reason"):
        parts.append(f"why={r['failure_reason']}")
    out = (r.get("outcome") or "").replace("\n", " ")
    parts.append(f"outcome「{out}」")
    if r.get("dialogue"):
        dlg = " ｜ ".join(f"{d.get('speaker')}:{d.get('line')}" for d in r["dialogue"])
        parts.append(f"对话[{dlg}]")
    if r.get("relation_updates"):
        parts.append("关系Δ=" + "、".join(
            f"{u.get('target')}(t{u.get('trust_delta')},a{u.get('affection_delta')})" for u in r["relation_updates"]))
    for e in r.get("target_effects") or []:
        parts.append(f"对方[{e.get('agent_id')}]情绪={e.get('emotion_type')}({e.get('emotion_intensity')}) "
                     f"伤={e.get('vitality_damage')} 记忆「{e.get('factual_memory')}」")
    for c in r.get("entity_state_changes") or []:
        parts.append(f"物→{c.get('entity_id')}={c.get('new_state')}(owner={c.get('owner_id')})")
    for s in r.get("entity_spawns") or []:
        parts.append(f"新增物「{s.get('name')}」({s.get('description')})")
    if r.get("detected"):
        parts.append("detected=True")
    return " ｜ ".join(parts)


def _build_prompt(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景:{scenario_meta.get('name')}\n"
        f"说明:{scenario_meta.get('description', '')}\n"
        f"预期:{scenario_meta.get('expect', '')}\n"
        f"重点验证项:{focus}\n"
        "(注意:上面的「说明/预期」是设计意图,只说『何为合理』,**不喂答案**。只依据下面实际产出评判;"
        "不要因为产出与你设想的不同而扣分,只罚可指证的具体问题。**对话/多人行动的『完成』会为每个"
        "参与者各产一条结果(各自第一人称判定成败/关系),两条结果 succeeded/gap 不同是信息不对称的"
        "正常表现,绝非自相矛盾,不要因此扣分。**)\n\n"
        "## 实际执行产出\n"
        f"{_phase_block(inputs, outputs)}\n\n"
        "## 确定性预检结果(供参考,可佐证 fields)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调)\n"
        "1. fields(语法/结构)——出现以下即扣:任何 outcome/对话/记忆里出现 agent id 或『第N步/steps』(叙事层禁);"
        "字段缺失(成败/失败缘由);对话/JSON 格式损坏;emotion 取非规范情绪;判失败却无缘由、或判成功却给了缘由;"
        "relation_updates 指向不存在的对象或幅度离谱。\n"
        "2. outcome_fidelity(语义·叙事保真)——出现以下即扣:outcome/对话/裁决文本套话化、复读 expected_outcome、"
        "脱离人设/情绪/现场、书面腔、把心事直白说尽;多步进度叙事与行动不符;对话被强行和谐化(明明该有冲突却一团和气)。"
        "**注意:对白允许误解/争执/不欢而散——这不是缺陷,不扣。**\n"
        "3. consequence_realism(后果真实)——出现以下即扣:成败/伤害/暴露/关系走向/gap 与输入明显不相称"
        "(例:力竭之人毫不费力地完成高难任务;当众动手却判无人察觉;被攻击者反而好感上升;密谋败露 gap 却为 0);"
        "PHYSICAL 无视对手实力/伤势;裁决出现双占(一具身体一步被消费两次)或否决理由不成立。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容:\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "outcome_fidelity": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "consequence_realism": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_action(
    router: LLMRouter,
    *,
    scenario_meta: dict,
    inputs: dict,
    outputs: dict,
    det_findings: dict,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Return per-criterion scores + rationale + issues + overall. Never raises."""
    prompt = _build_prompt(scenario_meta, inputs, outputs, det_findings)
    try:
        response = await router.complete(
            judge_scene,
            [LLMMessage(role="system", content=_SYSTEM), LLMMessage(role="user", content=prompt)],
            temperature=0.2,
            # Per dimension = rationale ≤80 chars (120) + score (3) + issues ≤3 × (≤40 chars 60 + structure 5) (195)
            # + structure (15) ≈ 333 tok; 3 dimensions + overall (≤60 chars, 90) ≈ 1099.
            max_tokens=output_budget(1099),
        )
    except Exception as exc:  # noqa: BLE001 — a judge failure must not abort the suite
        logger.warning("judge_call_failed", extra={"scenario": scenario_meta.get("name"), "error": str(exc)})
        return empty_judgement(_CRITERIA, f"judge 调用失败: {exc}")
    try:
        data = extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        return empty_judgement(_CRITERIA, "judge 输出非合法 JSON")
    if not isinstance(data, dict):
        return empty_judgement(_CRITERIA, "judge 输出非对象")
    result: dict[str, Any] = {}
    for c in _CRITERIA:
        entry = data.get(c) if isinstance(data.get(c), dict) else {}
        try:
            score = int(entry.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        result[c] = {
            "score": max(0, min(5, score)),
            "rationale": str(entry.get("rationale", "")),
            "issues": [str(x) for x in entry.get("issues", []) if str(x).strip()],
        }
    result["overall"] = str(data.get("overall", ""))
    return result
