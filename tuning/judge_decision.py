"""LLM-as-judge for the decision (decide) stage validation.

Scores one scenario's chosen action (per agent) against what decide is for: through the
judgment lens (persona + emotion), toward the direction (dominant_need + short-term
goals), within reality (bindable options + bottom lines), pick one concrete, feasible
action. Three criteria:

1. fields (syntax/structure) — canonical action_type, valid target binding, concrete first-person
   description, step count and fields present.
2. action_fit (semantics: was the right thing chosen) — through the lens, does the chosen action
   reasonably serve the direction while staying within reality?
3. coherence — inner monologue / expectation consistent with the chosen action, first person,
   no foreknowledge of the future.

Deduction-based (full marks by default, only penalize citable problems), reusing the
perception_emotion / need judges' negative constraints and their "per character, no borrowing
signals across characters" safeguard.
Uses a dedicated LLMRouter (``llm.judge``), and the judge call itself is traced.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from tuning.plan_view import render_target
from core.logging import get_logger
from tuning.judge_result import empty_judgement

logger = get_logger(__name__)

_CRITERIA = ("fields", "action_fit", "coherence")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「决策(decide)」阶段:角色在**判断透镜**(自身人设/价值观/底线 + 此刻情绪)"
    "下、朝**方向**(最迫切的需求 + 短期目标)、在**现实**(此刻可绑定的人/地点/物 + 不可逾越的底线)里,"
    "**只选了一个**具体可行的行动(action)。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调;"
    "**不要因为它没选你设想的那个 action 就扣分**——只要所选 action 不违反下列约束、在该角色处境下站得住,就视为合理。"
    "只输出 JSON。"
)


def _agent_block(per_agent: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for a in per_agent:
        inp = a.get("inputs", {}) or {}
        out = a.get("outputs", {}) or {}
        tier = "主角" if inp.get("is_main_character") else "背景"
        traits = "、".join(inp.get("core_traits") or []) or "—"
        lines.append(f"### {a.get('agent_name')}（{tier}，位于 {inp.get('location', '?')}；性格：{traits}）")
        # Judgment lens
        if inp.get("core_values"):
            lines.append("  价值观：" + "、".join(inp["core_values"]))
        if inp.get("hard_constraints"):
            lines.append("  底线(不可逾越)：" + "；".join(inp["hard_constraints"]))
        emo = inp.get("emotion")
        if emo:
            lines.append(f"  此刻情绪：{emo.get('primary')}（强度{emo.get('intensity')}，效价{emo.get('valence')}）")
        # Direction
        lines.append(f"  最迫切的需求：{inp.get('dominant_need') or '—'}")
        if inp.get("short_term_goals"):
            lines.append("  短期目标：" + "；".join(inp["short_term_goals"]))
        # Soft reference
        if inp.get("relation_context") and inp["relation_context"] not in ("无", "none"):
            lines.append("  关系：" + str(inp["relation_context"]).replace("\n", " ｜ "))
        for label, key in (("近期经历(客观)", "factual_memories"), ("近期经历(亲历)", "experiential_memories"),
                           ("已形成的判断", "insights")):
            if inp.get(key):
                lines.append(f"  {label}：" + "；".join(str(x) for x in inp[key]))
        msgs = inp.get("injected_messages") or []
        if msgs:
            lines.append("  收到消息：" + "；".join(f"{m.get('from')}「{m.get('content')}」({m.get('urgency')})" for m in msgs))
        bcs = inp.get("injected_broadcasts") or []
        if bcs:
            lines.append("  广播：" + "；".join(str(b.get("content")) for b in bcs))
        amb = inp.get("injected_ambient") or []
        if amb:
            lines.append("  环境观察：" + "；".join(str(e.get("content")) for e in amb))
        egs = inp.get("external_goals") or []
        if egs:
            lines.append("  外部压力：" + "；".join(
                f"[{g.get('drive_type')}/{g.get('urgency')}] {g.get('text')}" for g in egs))
        # Reality (bindable options)
        lines.append("  在场(可对其行动)：" + ("、".join(inp.get("visible_agents") or []) or "无人"))
        lines.append("  可前往(可达)：" + ("、".join(inp.get("reachable_locations") or []) or "无"))
        if inp.get("visible_entities"):
            lines.append("  可见之物：" + "、".join(inp["visible_entities"]))
        # Output: the chosen action
        tgt_str = render_target(out.get("target"))
        lines.append(f"  → 所选 action：{out.get('action_type')}（{tgt_str}，{out.get('estimated_steps')}步）")
        lines.append(f"  → 行动描述：{out.get('action_description')}")
        lines.append(f"  → 内心独白：{out.get('inner_monologue') or '（无）'}")
        lines.append(f"  → 期望结果：{out.get('expected_outcome') or '（无）'}")
    return "\n".join(lines)


def _build_prompt(scenario_meta: dict, per_agent: list[dict], det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景：{scenario_meta.get('name')}\n"
        f"说明：{scenario_meta.get('description', '')}\n"
        f"预期：{scenario_meta.get('expect', '')}\n"
        f"重点验证项：{focus}\n"
        "（注意：上面的「说明/预期」是设计意图,**不代表每个角色都感知到了其中的信号**——同一场景里不同角色收到的"
        "信号(外部压力/私信/记忆)往往**只定向发给某一个角色**。务必以下面每个角色**各自**列出的信号为准;评判某角色时"
        "**只能引用它名下列出的信号**,绝不得借用发给别人的定向信号。**名下没有某信号的角色,绝不能因『没响应该信号』"
        "扣分——那信号不是发给它的。** 若你判某角色「该如此行动却没有」,必须援引**它自己名下**的信号原文为据。"
        "预期里某个 action type 只是「该场景下最自然的一种选择」,**角色选了别的合理 action 同样不扣**。）\n\n"
        "## 各角色的判断透镜 / 方向 / 现实 与所选 action\n"
        f"{_agent_block(per_agent)}\n\n"
        "## 确定性预检结果(供参考,可佐证 fields)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调,问题越严重/越多扣得越狠)\n"
        "1. fields(语法/结构)——出现以下即扣:action_type 不在 7 类规范内;target 绑定非法(指向不在场的人 / "
        "去不相邻的地点 / PHYSICAL 没绑定目标);action_description 空泛不可上手、跳出第一人称、明显超长;"
        "内心独白 / 期望结果缺失;estimated_steps < 1。\n"
        "2. action_fit(语义·选得对不对)——**逐角色、只依据它自己的信号 + 透镜/方向/现实**判断,出现以下即扣:"
        "所选 action 无视该角色此刻最迫切的需求与短期目标、漫无目的;违背它的人设 / 价值观 / 底线;"
        "把 action 绑到**不在场 / 够不着**的目标;脱离它已感知到的信号、臆造没感知到的人事;"
        "**仅因某人在场(并无指向它的动作)就对其采取激烈 action(对抗 / 加害 / 严防到行动)**;"
        "在没有任何指向性触发时跳到极端 action。\n"
        "3. coherence(连贯)——出现以下即扣:内心独白 / 期望结果与所选 action 互相矛盾;独白写成了旁观者口吻或"
        "事后命名、而非此刻被激起的理由;预设了它此刻无从得知的未来 / 信息。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容:\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "action_fit": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "coherence": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_decision(
    router: LLMRouter,
    *,
    scenario_meta: dict,
    per_agent: list[dict],
    det_findings: dict,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Return per-criterion scores + rationale + issues + overall. Never raises."""
    prompt = _build_prompt(scenario_meta, per_agent, det_findings)
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
        logger.warning("judge_call_failed",
                       extra={"scenario": scenario_meta.get("name"), "error": str(exc)})
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
