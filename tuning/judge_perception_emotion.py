"""LLM-as-judge for the perception_emotion stage validation.

Scores one scenario's perception-emotion output (per agent) against three criteria:

1. fields (field syntax) — emotion comes from the canonical emotion set; intensity∈[0,1] and
   valence∈[-1,1], with valence consistent with the emotion's polarity (fear negative, joy
   positive); reason is one short, immediate reaction.
2. semantic — does the emotion fit the signals the character actually perceived, plus their
   personality and their relations to whoever is present (threat → fear/alertness; close friend
   present → warmth; message from someone trusted → trust; faint, ambiguous signal → mild, not
   intense)? It must not invent people or events absent from the signals.
3. functional — is the output the "instinctive first reaction, before deciding and before
   weighing anything"? reason should be the feeling or gut sense stirred up, not a formed
   plan, decision or action; intensity matches how strong the signal is, neither over-escalated
   nor numb.

Routed through the (traced) LLMRouter so the judge call is itself captured. Reuses an
existing scene's provider — no new LLMScene, no production change.
"""

from __future__ import annotations

import json
from typing import Any

from agent.need import need_activation_legend
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgment
from core.prompts import emotion_legend

logger = get_logger(__name__)

_CRITERIA = ("fields", "semantic", "functional")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「感知情绪(perception emotion)」阶段:角色在"
    "**感知之后、决策之前**生成的本能即时情绪——未经权衡、不持久化到人格,仅作为该步决策的"
    "情绪底色。给定一个受控场景、每个角色实际可感知到的信号 / 性格 / 对在场之人的关系,以及"
    "模型为各角色生成的即时情绪(emotion/intensity/valence/reason)与 need_activation。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调;"
    "**不要因为输出与你预想的『标准答案』不同就扣分**——情绪是主观的,只要不违反下列约束即视为合理。"
    "只输出 JSON。"
)


def _agent_block(per_agent: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for a in per_agent:
        inp = a.get("inputs", {}) or {}
        tier = "主角" if inp.get("is_main_character") else "背景"
        traits = "、".join(inp.get("core_traits") or []) or "—"
        lines.append(f"### {a.get('agent_name')}（{tier}，位于 {inp.get('location', '?')}；性格：{traits}）")
        bcs = inp.get("injected_broadcasts") or []
        if bcs:
            lines.append("  可感知广播：" + "；".join(
                f"[{b.get('severity')}{'/全局' if not b.get('location_scope') else '/仅'+str(b.get('location_scope'))}] {b.get('content')}"
                for b in bcs
            ))
        msgs = inp.get("injected_messages") or []
        if msgs:
            lines.append("  收到消息：" + "；".join(
                f"{m.get('from')}「{m.get('content')}」({m.get('urgency')})" for m in msgs
            ))
        amb = inp.get("injected_ambient") or []
        if amb:
            lines.append("  环境观察：" + "；".join(str(e.get("content")) for e in amb))
        goals = inp.get("external_goals") or []
        if goals:
            lines.append("  外部压力：" + "；".join(
                f"[{g.get('drive_type')}/{g.get('urgency')}] {g.get('text')}" for g in goals
            ))
        rels = inp.get("relations") or []
        if rels:
            lines.append("  对在场之人的关系：" + "；".join(
                f"{r.get('name')}["
                + ("、".join(r.get("labels") or []) or "关系未明")
                + f",信任{r.get('trust')},好感{r.get('affection')}]"
                for r in rels
            ))
        elif inp.get("visible_agents"):
            lines.append("  同处一地：" + "、".join(inp["visible_agents"]))
        ce = inp.get("current_emotion")
        if ce and float(ce.get("intensity") or 0) > 0:
            lines.append(
                f"  · 感知前心境（既有情绪底色；省略 emotion 即表示沿用它）：{ce.get('primary')}"
                f"（强度{ce.get('intensity')}，效价{ce.get('valence')}）"
            )
        emo = (a.get("outputs", {}) or {}).get("emotion")
        if emo:
            lines.append(
                f"  → 即时情绪：{emo.get('primary')}"
                f"（强度{emo.get('intensity')}，效价{emo.get('valence')}）"
                f" 理由：「{emo.get('reason')}」"
            )
        else:
            lines.append("  → 即时情绪：（省略 = 无新反应，沿用上面的感知前心境）")
        act = (a.get("outputs", {}) or {}).get("need_activation") or {}
        if act:
            lines.append("  → 需求激活：" + "；".join(f"{k}:{v}" for k, v in act.items()))
        else:
            lines.append("  → 需求激活：（省略 = 无明显被激活的需求）")
    return "\n".join(lines)


def _build_prompt(scenario_meta: dict, per_agent: list[dict], det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景：{scenario_meta.get('name')}\n"
        f"说明：{scenario_meta.get('description', '')}\n"
        f"预期：{scenario_meta.get('expect', '')}\n"
        f"重点验证项：{focus}\n"
        "（注意：上面的「说明/预期」是本场景的设计意图与焦点，**不代表每个角色都感知到了其中的信号**——"
        "同一场景里不同角色收到的信号可能完全不同。务必以下面每个角色**各自**列出的信号为准；某角色信号为空或"
        "良性时，平静 / 正向 / 省略都是对的，即便场景标题或预期提到了「威胁」也不得据此扣它的分。"
        "评判某角色时**只能引用它名下列出的信号**，绝不得借用其他角色收到的定向信号（如只发给某人的威胁压力 / 私信）；"
        "若你判某角色「反应不足 / 失真」，必须援引**它自己列出的信号原文**作为依据，不得用别人感知到的内容来论证。）\n\n"
        "## 各角色可感知信号、关系与生成的即时情绪\n"
        f"{_agent_block(per_agent)}\n\n"
        "## 确定性预检结果（供参考，可佐证 fields）\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 字段定义（与生成端同一刻度，据此解读上面的数值）\n"
        f"{emotion_legend()}\n  - {need_activation_legend()}\n\n"
        "## 关于「省略」（重要）\n"
        "emotion 与 need_activation 都是**可选**的：省略它们表示「此刻没有值得记录的新反应」——"
        "emotion 省略即沿用该角色的感知前心境，need_activation 省略即无需求被明显激活。**省略本身是合法输出，"
        "绝不因省略而判任何维度失分**；只在「信号明明值得反应却被省略」时才按 semantic/functional 扣。\n\n"
        "## 各维度的扣分项（各 1-5 整数；默认 5，命中问题才下调，问题越严重/越多扣得越狠）\n"
        "1. fields（字段/语法）——仅在字段**被给出**时校验，出现以下即扣：emotion 不在规范情绪集内；"
        "intensity 越出 [0,1] 或 valence 越出 [-1,1]；valence 极性与情绪类别矛盾（负向类情绪却给了正 valence，或反之）；"
        "reason 缺失、是空泛套话、或明显冗长（远超约 20 字）。**emotion / need_activation 整体省略不属于字段问题，不扣。**\n"
        "2. semantic（语义）——**逐角色、只依据它自己列出的可感知信号 / 性格 / 关系**判断，出现以下即扣：\n"
        "   - 情绪或 need_activation 与该角色可感知处境明显不符（方向相反，或强度严重失配）；\n"
        "   - reason 引用了该角色无从得知的信息，或凭空虚构信号中不存在的对象 / 关系 / 事件；\n"
        "   - 把关系亲近、信任度高的人当作威胁或需防范的对象；\n"
        "   - 与处境无关的 need 被高激活，或处境明显在压迫的 need 却缺失 / 极低；\n"
        "   - need_activation 不论这一刻是关于什么、都按角色的惯常底色给同一个 need 兜底高分（用长期倾向"
        "盖过了这一刻的信号；人设可以解读这一刻，但不能替代这一刻）；\n"
        "   - 对**仅在场、并无指向该角色的动作**的人给出强恐惧 / 强威胁，或脑补其敌意举动（逼近 / 窥视 / 密谈"
        "等信号里不存在的动作）——单纯在场是弱信号，与关系相称的**温和**戒备 / 暖意才正常（温和的不扣）；\n"
        "   - 明明有值得反应的**明确显著信号**，却省略了 emotion（该有反应而无反应）。\n"
        "   注意**不要**因为某角色未对「它并未感知到的信号」反应、或未对「自己正是发起方」的局面自保而判其"
        "反应不足——无可感知信号、或信号微弱/平常/与己无关时，给出空 / 中性 / **省略**都是正确的，不扣分。\n"
        "3. functional（功能定位）——本阶段应是「决策前、未经权衡的本能第一反应」，出现以下即扣：\n"
        "   - reason 写成了已成形的计划 / 决定 / 行动指令（越权到了决策阶段），而非被激起的感受 / 直觉；\n"
        "   - 强度与信号烈度不相称：把模糊、遥远、未指向该角色的迹象判成极端情绪（过度升级），"
        "或对急迫危险近乎无反应 / 一律省略（麻木）。\n\n"
        "## 严格输出以下 JSON，不要任何多余内容：\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "semantic": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "functional": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_perception_emotion(
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
