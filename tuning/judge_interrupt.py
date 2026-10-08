"""LLM-as-judge for the interrupt-decision (Agent.evaluate_interrupt) stage validation.

The stage under test is the interrupt decision: while a character is mid multi-step action, an
external signal (message / broadcast / external threat) arrives; in first person they weigh
"is this weighty enough to drop what I'm doing right now", decide whether to interrupt, and leave
a first-person inner monologue (thought). Two criteria:

1. decision_fit (semantics: does the decision hold up) — given the character's persona, values,
   emotion and the weight of the task at hand, plus how urgent the signal really is, is the
   interrupt / don't-interrupt decision reasonable? Either can be reasonable — penalize only
   citable distortions (jumpy overreaction / numb to something that truly matters / contradicts
   persona and values / clearly out of step with the weight of the task at hand).
2. voice (first-person monologue quality) — is thought a genuine first-person weighing that
   leads to the decision? No emotion labels, no breaking first person, no contradicting the final
   decision, no code-layer concepts like step.

Functional third-party judge: deduction-based (full marks by default), led by negative
constraints, not bound to an answer set, reason before score (analyze first, then judge).
Uses a dedicated LLMRouter (``llm.judge``), and the judge call itself is traced.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgment

logger = get_logger(__name__)

_CRITERIA = ("decision_fit", "voice")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「中断决策(interrupt)」阶段:角色正进行某个多步行动时,突然出现一个"
    "外部信号(消息/广播/外部威胁),角色以**第一人称**在内心权衡——这事够不够分量让『我』立刻放下手头的事——"
    "然后决定是否中断(should_interrupt),并留下一段第一人称内心独白(thought)。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调。\n"
    "**关键:中断或不中断本身都可能是合理的——绝不要因为角色没做出你设想的那个选择就扣分。** 只看两点:"
    "① 这个决定在该角色的人设/价值观/情绪、手头之事的轻重与进度、以及信号的真实紧要程度下站不站得住;"
    "② 那段独白是不是真的第一人称、由权衡推出决定。\n"
    "先写 rationale(简短分析)再给 score——先析后判。只输出 JSON。"
)


def _scene_block(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    traits = "、".join(inputs.get("core_traits") or []) or "—"
    values = "、".join(inputs.get("core_values") or []) or "—"
    emo = inputs.get("emotion") or {}
    emo_str = (f"{emo.get('primary')}（强度{emo.get('intensity')}，效价{emo.get('valence')}）"
               if emo else "—")
    goals = "；".join(inputs.get("short_term_goals") or []) or "—"
    si = outputs.get("should_interrupt")
    si_str = "中断（放下手头的事）" if si is True else ("不中断（继续手头的事）" if si is False else "—")
    return (
        f"### 角色：{inputs.get('agent_name')}（{inputs.get('role') or '—'}）\n"
        f"  性格：{traits}\n"
        f"  价值观：{values}\n"
        f"  此刻情绪：{emo_str}\n"
        f"  短期目标：{goals}\n"
        f"  正在做的事：「{inputs.get('ongoing_purpose')}」（{inputs.get('ongoing_action_type')}，进度：{inputs.get('progress_hint')}）\n"
        f"  突然到来的信号〔来源：{inputs.get('source') or '—'}〕：{inputs.get('reason')}\n"
        f"  → 角色的决定：{si_str}\n"
        f"  → 第一人称内心独白：{outputs.get('thought') or '（空）'}"
    )


def _build_prompt(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    return (
        f"## 场景：{scenario_meta.get('name')}\n"
        f"说明：{scenario_meta.get('description', '')}\n"
        f"预期（仅说明何为失真,不喂答案——中断或不中断都可能合理）：{scenario_meta.get('expect', '')}\n\n"
        "## 角色处境与所作的中断决定\n"
        f"{_scene_block(inputs, outputs)}\n\n"
        "## 确定性预检结果(供参考,可佐证 voice 的 step 泄漏等结构问题)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调,问题越严重/越多扣得越狠)\n"
        "1. decision_fit(决定是否站得住)——**只依据该角色自己的处境与这一个信号**判断,出现以下即扣:"
        "为一桩对『我』其实无关紧要 / 可从容稍后处理的小事,慌忙丢下要紧的手头之事(一惊一乍);"
        "对一个于『我』真正性命攸关 / 迫在眉睫的信号麻木无视、毫无警觉地继续手头之事;"
        "决定违背角色的人设 / 价值观 / 底线;决定与手头之事的轻重和进度明显脱节(如手头事就快收尾且至关重要,"
        "却为小事中断)。**只要决定在该角色处境下站得住,不论中断与否都不扣。**\n"
        "2. voice(第一人称独白)——出现以下即扣:独白不是第一人称(旁观者口吻、『作为AI/请评估』、被命名成一个"
        "情绪标签如『震惊』『愤怒』而没写出在权衡什么);独白空洞、看不出具体在掂量什么;独白与最终的中断决定"
        "自相矛盾;独白泄漏 step 等代码层概念(如『第N步』『还剩X步』)。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容(每项 rationale 在前、score 在后):\n"
        '{"decision_fit": {"rationale": "一句话分析,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "voice": {"rationale": "一句话分析,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_interrupt(
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
            # + structure (15) ≈ 333 tok; 2 dimensions + overall (≤60 chars, 90) ≈ 766.
            max_tokens=output_budget(766),
        )
    except Exception as exc:  # noqa: BLE001 — a judge failure must not abort the suite
        logger.warning("judge_call_failed",
                       extra={"scenario": scenario_meta.get("name"), "error": str(exc)})
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
