"""LLM-as-judge for the relation **evolution** stage — SEMANTIC quality of label/summary judgment.

Under test is `RelationEvolution.evaluate`: every few steps an agent's recent memories
about someone are read and the relation's current label + summary are judged objectively from a
third-person view, then written back. Structural correctness (well-formed labels / blood and
structural bonds not illegally removed or altered / no id or step leak / nothing that should stay
put was changed) is covered by deterministic checks; the judge scores only semantic judgment
quality:

1. fidelity (judgment follows the evidence) — is the change (or non-change) in label/summary
   objectively derived from the recent events? Penalize: changes without grounds (evidence can't
   support them), ignoring strong evidence, a summary that distorts the trajectory. Most
   relations should stay unchanged; keeping them unchanged on that basis is correct and must not
   be penalized for "not changing the way you imagined".
2. objectivity (third-person, objective) — do summary/label stay third-person and objective,
   without first-person emotion or lyricism? Label wording stays consistent rather than drifting
   between synonyms. Penalize: first-person or lyrical summaries, subjective labels, wording drift.

Functional third-party verdict, reason first (cite the events relied on, then score).
Deduction-based (full marks by default, only penalize citable problems). Don't feed it
answers: give only design intent and the general evolution rules, never tell the judge what the
right answer for this relation is. The judge model comes from ``llm.judge``.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgement

logger = get_logger(__name__)

_CRITERIA = ("fidelity", "objectivity")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是**关系演化**:一个角色每隔一段时间,以**第三视角客观评判器**的身份"
    "回看自己近期关于某人的记忆,判断这段关系当前的 label(关系标签)与 summary(客观概述)应当是什么。\n"
    "通用演化规则(据此判断改动是否站得住,**不是某关系的标准答案**):\n"
    "· 血亲(父子/兄弟等生物亲缘)是身份事实,永不可删改,只能在其上叠加;\n"
    "· 其他结构性绑定(婚姻/师承/契约/隶属)只在出现明确的'绑定改变事件'(和离/断绝/革职等)时才改,"
    "不因情感漂移而动;\n"
    "· 叙事性定性(朋友/敌人/盟友/对手)可随证据自由演化;\n"
    "· 多数关系在多数时间应保持不变——稳健优先,不为改而改。\n"
    "评分以**扣分制**为主:满分(5)默认,只在发现**可指证的具体问题**时下调。**不要因为它没改成你设想的样子就扣分**"
    "——只要改动(或保持不变)由近期事件撑得住、summary 客观不抒情、label 合法且措辞一致,就视为合理。只输出 JSON。"
)


def _target_block(t: dict[str, Any]) -> str:
    before, after = t.get("before", {}) or {}, t.get("after", {}) or {}
    evs = t.get("recent_events", []) or []
    ev_text = "\n".join(f"      · {e}" for e in evs) or "      · （无明显近期事件）"
    chg = []
    if t.get("labels_changed"):
        chg.append("labels 改了")
    if t.get("summary_changed"):
        chg.append("summary 改了")
    chg_text = "、".join(chg) if chg else "未改动"
    return (
        f"  - {t.get('name', '?')}（本次：{chg_text}）\n"
        f"    近期事件证据：\n{ev_text}\n"
        f"    改前 labels：{before.get('labels')} ｜ 信任{before.get('trust')} 好感{before.get('affection')}"
        f" ｜ 概述：{before.get('summary') or '（空）'}\n"
        f"    改后 labels：{after.get('labels')} ｜ 概述：{after.get('summary') or '（空）'}"
    )


def _prompt(meta: dict, outputs: dict, det: dict) -> str:
    targets = outputs.get("targets", []) or []
    blocks = "\n".join(_target_block(t) for t in targets) or "  （无候选关系）"
    return (
        f"## 场景:{meta.get('name')}\n"
        f"说明:{meta.get('description', '')}\n预期:{meta.get('expect', '')}\n"
        "(「说明/预期」是设计意图,只说何为合理,**不喂答案**(不告诉你正确 label/summary);"
        "只依据下面每段关系的近期事件 + 改前/改后判断改动是否站得住。)\n\n"
        f"## 各候选关系(改前→改后)\n{blocks}\n\n"
        "## 确定性预检(供参考)\n"
        f"{json.dumps(det, ensure_ascii=False)}\n\n"
        "## 扣分项(各 1-5;默认 5,命中才下调)\n"
        "1. fidelity(评判忠于证据)——出现以下即扣:label/summary 改了但近期事件**撑不起**这个改动(改而无据);"
        "近期事件呈现**强烈且明确**的转折却视而不改;summary **歪曲/夸大**了关系走向。"
        "**该不变而保持不变、证据平淡而维持原状,都是对的,不扣**。\n"
        "2. objectivity(第三视角客观)——出现以下即扣:summary 写成**第一人称/抒情/情绪宣泄**(应是冷静的第三视角概述);"
        "label **主观情绪化**或在近义词间**漂移**(同一类型换着措辞写)。\n\n"
        "## 严格输出 JSON(reason 在前):\n"
        '{"fidelity": {"rationale": "先点出依据的具体事件,再下结论,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "objectivity": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_relation(
    router: LLMRouter, *, scenario_meta: dict, outputs: dict,
    det_findings: dict, judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Score the semantic judgment quality of relation evolution. Never raises (falls back to empty_judgement on failure)."""
    name = scenario_meta.get("name", "")
    prompt = _prompt(scenario_meta, outputs, det_findings)
    try:
        response = await router.complete(
            judge_scene, [LLMMessage(role="system", content=_SYSTEM), LLMMessage(role="user", content=prompt)],
            # Per dimension = rationale ≤80 chars (120) + score (3) + issues ≤3 × (≤40 chars 60 + structure 5) (195)
            # + structure (15) ≈ 333 tok; 2 dimensions + overall (≤60 chars, 90) ≈ 766.
            temperature=0.2, max_tokens=output_budget(766))
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge_call_failed", extra={"scenario": name, "error": str(exc)})
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
        result[c] = {"score": max(0, min(5, score)), "rationale": str(entry.get("rationale", "")),
                     "issues": [str(x) for x in entry.get("issues", []) if str(x).strip()]}
    result["overall"] = str(data.get("overall", ""))
    return result
