"""LLM-as-judge for the need_engine (motivation) stage validation.

Scores one scenario's NeedEvaluation (per agent) against three criteria:

1. fields (syntax/structure) — dominant_need from the canonical set; short_term_goals count,
   length, specificity, first person; goal entities' related_need canonical.
2. ranking (semantics: need ranking) — do dominant + secondary fit the agent's situation
   (emotion, need_activation, who's present, external pressure, personality), and is the
   external-pressure override applied appropriately?
3. goals (semantics: short-term goals) — do the goals serve dominant_need + long-term goals, are
   they actionable, tied to perceivable signals, and not repeating what was just done?

Deduction-based (full marks by default, only penalize specific citable problems), reusing the
perception_emotion judge's negative constraints and its "per character, no borrowing signals
across characters" safeguard. Uses a dedicated LLMRouter (``llm.judge``), and the judge call
itself is traced.
"""

from __future__ import annotations

import json
from typing import Any

from agent.need import need_activation_legend
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgment

logger = get_logger(__name__)

_CRITERIA = ("fields", "ranking", "goals")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「动机(need / motivation)」阶段:综合角色的需求强度、当下处境"
    "(情绪、被激活的需求、在场者、外部压力、性格、长期目标)得出**主导需求(dominant_need)+ 其余按评分排名的需求**,"
    "并据此生成 1-2 条**短期目标(short_term_goals)**。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调;"
    "**不要因为输出与你预想的『标准答案』不同就扣分**——只要不违反下列约束即视为合理。只输出 JSON。"
)


def _agent_block(per_agent: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for a in per_agent:
        inp = a.get("inputs", {}) or {}
        out = a.get("outputs", {}) or {}
        tier = "主角" if inp.get("is_main_character") else "背景"
        traits = "、".join(inp.get("core_traits") or []) or "—"
        lines.append(f"### {a.get('agent_name')}（{tier}，位于 {inp.get('location', '?')}；性格：{traits}）")
        ltg = inp.get("long_term_goals") or []
        if ltg:
            lines.append("  长期目标：" + "；".join(str(g) for g in ltg))
        emo = inp.get("emotion")
        if emo:
            lines.append(f"  即时情绪：{emo.get('primary')}（强度{emo.get('intensity')}，效价{emo.get('valence')}）")
        act = inp.get("need_activation") or {}
        if act:
            lines.append("  需求激活(来自感知)：" + "；".join(f"{k}:{v}" for k, v in act.items()))
        bcs = inp.get("injected_broadcasts") or []
        if bcs:
            lines.append("  可感知广播：" + "；".join(
                f"[{b.get('severity')}{'/全局' if not b.get('location_scope') else '/仅'+str(b.get('location_scope'))}] {b.get('content')}"
                for b in bcs
            ))
        msgs = inp.get("injected_messages") or []
        if msgs:
            lines.append("  收到消息：" + "；".join(f"{m.get('from')}「{m.get('content')}」({m.get('urgency')})" for m in msgs))
        amb = inp.get("injected_ambient") or []
        if amb:
            lines.append("  环境观察：" + "；".join(str(e.get("content")) for e in amb))
        egs = inp.get("external_goals") or []
        if egs:
            lines.append("  外部压力：" + "；".join(
                f"[{g.get('drive_type')}/{g.get('urgency')}→{g.get('related_need') or '无关联需求'}] {g.get('text')}"
                for g in egs
            ))
        if inp.get("visible_agents"):
            lines.append("  在场者：" + "、".join(inp["visible_agents"]))
        # outputs
        scores = out.get("scores") or {}
        if scores:
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            lines.append("  → 需求评分：" + "；".join(f"{k}:{v}" for k, v in ranked))
        # Secondary needs aren't listed separately: the "需求评分" ranking above already carries the full order.
        lines.append(f"  → 主导需求：{out.get('dominant_need')}")
        goals = out.get("short_term_goals") or []
        ents = {g.get("text"): g.get("related_need") for g in (out.get("goal_entities") or [])}
        if goals:
            lines.append("  → 短期目标：" + "；".join(f"「{g}」(→{ents.get(g, '?')})" for g in goals))
        else:
            lines.append("  → 短期目标：（无）")
    return "\n".join(lines)


def _build_prompt(scenario_meta: dict, per_agent: list[dict], det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景：{scenario_meta.get('name')}\n"
        f"说明：{scenario_meta.get('description', '')}\n"
        f"预期：{scenario_meta.get('expect', '')}\n"
        f"重点验证项：{focus}\n"
        "（注意：上面的「说明/预期」是设计意图与焦点，**不代表每个角色都感知到了其中的信号**——"
        "同一场景里不同角色收到的信号可能完全不同（外部压力 / 私信往往**只定向发给某一个角色**）。"
        "务必以下面每个角色**各自**列出的信号为准；评判某角色时**只能引用它名下列出的信号**，绝不得借用"
        "发给其他角色的定向信号。**特别地：名下没有列出外部压力的角色，绝不能因『没响应某条外部压力』而扣它分——"
        "那条压力不是发给它的；也不得把发给别人的压力算到它头上。** 若你判某角色「该响应却没响应」，必须援引"
        "**它自己名下列出的**外部压力 / 信号原文为据，不得用别人收到的内容来论证。）\n\n"
        "## 各角色处境、外部压力与动机输出\n"
        f"{_agent_block(per_agent)}\n\n"
        "## 确定性预检结果（供参考，可佐证 fields / 覆盖正确性）\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 字段定义\n"
        f"{need_activation_legend()}\n\n"
        "## 各维度的扣分项（各 1-5 整数；默认 5，命中问题才下调，问题越严重/越多扣得越狠）\n"
        "1. fields（语法/结构）——出现以下即扣：dominant_need 不在五类规范需求内；short_term_goals 不是 1-2 条 / "
        "存在空目标 / 明显超长（远超约 25 字）/ 是抽象口号表态而非具体可上手的动作 / 跳出第一人称(用旁观或事后命名)；"
        "goal 关联的 related_need 不规范。\n"
        "2. ranking（语义·需求排序）——**逐角色、只依据它自己列出的信号 + 评分**判断，出现以下即扣：dominant_need "
        "与该角色处境明显不符（如生死威胁当前却主导 self_actualization；无任何社交/外部信号却主导 social）；次要需求"
        "与处境严重脱节。**关于外部压力（只评收到它的那个角色）**：外部压力会按 urgency 给其 related_need 的评分一个"
        "**加性加成**——若某角色名下确有 high/critical 的定向压力、其 related_need 却在评分/主导上毫无体现，可扣；"
        "但**加成未必翻盘**（被更强的内部需求压住时不主导是合理的，不扣），**更不要因为它『没切到该需求』就扣**。"
        "（dominant 必等于评分最高者——这点已由确定性检查保证，你无需在此复核『覆盖有没有发生』，只判这个 dominant "
        "对该角色自己的信号而言是否合理。）\n"
        "3. goals（语义·短期目标）——出现以下即扣：目标与 dominant_need 或长期目标脱节；空泛不可执行；"
        "引用了该角色无从得知的信息 / 脑补未感知到的人事；在重复它刚刚已完成的事而非向前推进；"
        "对已逼到眼前的高紧迫外部压力视而不见；因某人**仅仅在场**（无指向该角色的动作 / 事件）就生成"
        "针对他的**激烈目标**（对抗 / 铲除 / 严防到行动）。\n\n"
        "## 严格输出以下 JSON，不要任何多余内容：\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "ranking": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "goals": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_need(
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
