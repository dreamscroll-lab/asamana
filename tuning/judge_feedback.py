"""LLM-as-judge for the feedback stage (completion-cell landing + target effects) validation.

Scores one scenario's feedback landing across three criteria:

1. fields (syntax/structure) — canonical emotions, numbers in range, written memories free of
   id/step leaks, relation/goal fields valid.
2. appraisal_fidelity (emotion/memory, subjective fidelity) — do the appraised emotion and the
   experiential memory fit the persona and situation, in proportion, in first person, without
   clichés, templating or leaks? A creative task: negative constraints first, no positive
   example content.
3. landing_consistency — the landed pieces (emotion direction, need shock, relation Δ direction,
   goal verdict, vitality, target reaction) are mutually consistent and proportionate to the
   action result (success/failure, gap, what was done to whom).

Deduction-based (full marks by default, only penalize citable problems), reusing the action/decide
judges' negative-constraint / don't-bind-the-answer safeguards.
The judge uses a dedicated LLMRouter (``llm.judge``), and the judge call itself is traced.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgment

logger = get_logger(__name__)

_CRITERIA = ("fields", "appraisal_fidelity", "landing_consistency")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「反馈(feedback)」阶段:一个行动**已经执行完**,其结果如何"
    "**落地**到 agent 的内部状态——情绪(由人格主观评价生成)、需求强度、与他人的关系、写入的记忆"
    "(客观事实 + 第一人称体验)、体力/存亡、短期目标的推进;承受方(被作用的人)也会据此更新情绪/关系/记忆。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调;"
    "**不要因为它没产出你设想的那个情绪/数值就扣分**——只要落地在该处境下站得住、不违反下列约束,就视为合理。"
    "情绪允许复杂、克制、甚至与表面结果相反(若人格如此)。只输出 JSON。"
)


def _emotion_str(e: dict | None) -> str:
    if not e:
        return "（无）"
    return f"{e.get('primary')}（强度{e.get('intensity')}，效价{e.get('valence')}）"


def _block(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines: list[str] = []
    kind = inputs.get("kind", "self")
    tier = "主角" if inputs.get("is_main_character") else "背景"
    traits = "、".join(inputs.get("core_traits") or []) or "—"
    lines.append(f"### {('行动者' if kind == 'self' else '承受方')}:{inputs.get('actor_name')}"
                 f"（{tier}；性格：{traits}）")
    if inputs.get("core_values"):
        lines.append("  价值观：" + "、".join(inputs["core_values"]))

    spec = inputs.get("spec", {}) or {}
    if kind == "self":
        lines.append(f"  本次行动:{spec.get('type')} ｜ 描述「{spec.get('description', '')}」")
        lines.append(f"  结果:{'成功' if spec.get('succeeded', True) else '失败'}｜缘由={spec.get('failure_reason', '') or '—'}"
                     f" ｜ 实际「{spec.get('outcome', '')}」")
        if spec.get("expected_outcome"):
            lines.append(f"  原期望:{spec.get('expected_outcome')}")
        lines.append(f"  当时最迫切需求:{inputs.get('dominant_need') or '无'}")
        if spec.get("relation_updates"):
            lines.append("  执行器声明的关系Δ:" + "、".join(
                f"{u.get('target')}(t{u.get('trust_delta')},a{u.get('affection_delta')})" for u in spec["relation_updates"]))
        if spec.get("vitality_damage"):
            lines.append(f"  自身体力损耗:{spec.get('vitality_damage')}")
    else:
        lines.append(f"  施加者:{inputs.get('from_name')}")
        lines.append(f"  发生在我身上的事(客观事实):「{spec.get('factual_memory', '')}」")
        lines.append(f"  执行器判定我应有的情绪:{spec.get('emotion_type')}"
                     f"（强度{spec.get('emotion_intensity')}，效价{spec.get('emotion_valence')}）")
        if spec.get("vitality_damage"):
            lines.append(f"  对我的伤害:{spec.get('vitality_damage')}")

    before, after = outputs.get("before", {}) or {}, outputs.get("after", {}) or {}
    deltas = outputs.get("deltas", {}) or {}
    lines.append("\n  —— 落地结果(before → after)——")
    lines.append(f"  情绪:{_emotion_str(before.get('emotion'))} → {_emotion_str(after.get('emotion'))}")
    if "need_shift" in deltas:
        lines.append("  需求强度变化:" + "、".join(f"{k}{v:+}" for k, v in deltas["need_shift"].items()))
    if "vitality" in deltas:
        lines.append(f"  体力:{deltas['vitality']['from']} → {deltas['vitality']['to']}")
    if "is_active" in deltas:
        lines.append(f"  存亡:{deltas['is_active']['from']} → {deltas['is_active']['to']}（触发死亡）")
    if "relation_deltas" in deltas:
        lines.append("  关系变化:" + "、".join(
            f"对{n} 信任{v['trust_delta']:+} 好感{v['affection_delta']:+}" for n, v in deltas["relation_deltas"].items()))
    if "goal_transitions" in deltas:
        lines.append("  目标推进:" + "、".join(
            f"「{t['goal']}」{t['from']}→{t['to']}" for t in deltas["goal_transitions"]))
    new_mem = outputs.get("new_memories", []) or []
    if new_mem:
        lines.append("  新写入记忆:")
        for m in new_mem:
            lines.append(f"    [{m.get('stream')}] {m.get('content')}")
    if outputs.get("error"):
        lines.append(f"  ⚠️ 落地异常:{outputs['error']}")
    return "\n".join(lines)


def _build_prompt(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景:{scenario_meta.get('name')}\n"
        f"说明:{scenario_meta.get('description', '')}\n"
        f"预期:{scenario_meta.get('expect', '')}\n"
        f"重点验证项:{focus}\n"
        "(注意:上面的「说明/预期」是设计意图,只说『何为合理』,**不喂答案**。只依据下面实际落地评判;"
        "不要因为落地与你设想的不同而扣分,只罚可指证的具体问题。情绪/记忆是主观产物,允许有个人色彩。)\n\n"
        "## 实际反馈落地\n"
        f"{_block(inputs, outputs)}\n\n"
        "## 确定性预检结果(供参考,可佐证 fields)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调)\n"
        "1. fields(语法/结构)——出现以下即扣:写入记忆里出现 agent id 或『第N步/N步/steps』(叙事层禁);"
        "情绪取非规范类别;情绪强度∉[0,1] 或效价∉[-1,1];关系 Δ 指向不存在对象或幅度离谱;"
        "目标状态非法。\n"
        "2. appraisal_fidelity(情绪/记忆·主观保真)——出现以下即扣:appraised 情绪与人格/处境明显不符或"
        "套模板(『成功必喜、失败必沮』);experiential 记忆套话化、书面腔、跳出第一人称、把心事直白说尽、"
        "或与实际经历不符;记忆/情绪复读 outcome 原文而无主观加工。**情绪克制、复杂、甚至苦乐参半不扣。**\n"
        "3. landing_consistency(落地一致)——出现以下即扣:情绪效价方向与成败/gap 明显矛盾(大败却高兴正效价、"
        "顺遂却剧烈挫败);need shock 方向反了(成功反而加压);关系 Δ 方向与互动不符(被攻击者反而更信任施害者);"
        "目标未真正达成却判 COMPLETED(或已达成却仍 active);承受方反应与所受对待不相称;致命伤害却未触发死亡。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容:\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "appraisal_fidelity": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "landing_consistency": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_feedback(
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
        return empty_judgment(_CRITERIA, f"judge 调用失败: {exc}")
    try:
        data = extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        return empty_judgment(_CRITERIA, "judge 输出非合法 JSON")
    if not isinstance(data, dict):
        return empty_judgment(_CRITERIA, "judge 输出非对象")
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
