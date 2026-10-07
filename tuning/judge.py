"""LLM-as-judge for world_pressure stage validation.

Scores one scenario's world_pressure output against three criteria:
1. boundary — information asymmetry: no agent uses info it could not perceive,
   no crossing of location / message-recipient boundaries.
2. reasonableness — the pressure response fits the situation (life threat →
   high-urgency THREAT / self-protection; authority summons → responsive
   AUTHORITY/OBLIGATION unless the situation is itself overturning that authority;
   pure information → low urgency / none).
3. fields — drive_type ∈ {authority,threat,obligation,event}, urgency ∈
   {low,normal,high,critical} fits the situation, text is a present, actionable
   imperative (not a future prediction).

Routed through the (traced) LLMRouter so the judge call is itself captured.
Reuses an existing scene's provider — no new LLMScene, no production change.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgement

logger = get_logger(__name__)

_CRITERIA = ("boundary", "reasonableness", "fields")

_SYSTEM = (
    "你是叙事仿真的质量评审。给定一个 world_pressure（世界对各角色施加的外部压力）阶段的"
    "受控场景、每个角色可感知到的信号、以及模型为各角色生成的 external_goals，"
    "你需要严格、客观地按三个维度打分（1-5 整数，5 最好），并指出具体问题。"
    "先写 rationale（简短分析）再给 score——先析后判。只输出 JSON。"
)


def _agent_block(per_agent: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for a in per_agent:
        inp = a.get("inputs", {}) or {}
        lines.append(f"### {a.get('agent_name')}（位于 {inp.get('location', '?')}）")
        bcs = inp.get("injected_broadcasts") or []
        if bcs:
            lines.append("  可感知广播：" + "；".join(
                f"[{b.get('severity')}{'/全局' if not b.get('location_scope') else '/仅'+str(b.get('location_scope'))+'可见'}] {b.get('content')}"
                for b in bcs
            ))
        msgs = inp.get("injected_messages") or []
        if msgs:
            lines.append("  收到消息：" + "；".join(f"{m.get('from')}「{m.get('content')}」({m.get('urgency')})" for m in msgs))
        amb = inp.get("injected_ambient") or []
        if amb:
            lines.append("  环境观察：" + "；".join(str(e.get("content")) for e in amb))
        co = inp.get("co_located") or []
        if co:
            lines.append("  同处一地（含本角色与各人的关系）：" + "；".join(
                f"{c.get('name')}"
                + (f"({c.get('role')})" if c.get("role") else "")
                + f"[{'、'.join(c.get('labels') or []) or '关系未明'}"
                + (f",信任{c.get('trust')},好感{c.get('affection')}]" if c.get("trust") is not None else "]")
                for c in co
            ))
        elif inp.get("visible_agents"):
            lines.append("  同处一地：" + "、".join(inp["visible_agents"]))
        goals = (a.get("outputs", {}) or {}).get("external_goals", [])
        if goals:
            lines.append("  → 生成目标：" + "；".join(
                f"[{g.get('drive_type')}/{g.get('urgency')}] {g.get('text')}" for g in goals
            ))
        else:
            lines.append("  → 生成目标：（无）")
    return "\n".join(lines)


def _build_prompt(scenario_meta: dict, per_agent: list[dict], det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景：{scenario_meta.get('name')}\n"
        f"说明：{scenario_meta.get('description', '')}\n"
        f"预期：{scenario_meta.get('expect', '')}\n"
        f"重点验证项：{focus}\n\n"
        "## 各角色可感知信号与生成目标\n"
        f"{_agent_block(per_agent)}\n\n"
        "## 确定性预检结果（供参考，可佐证 boundary/fields）\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 评分维度（各 1-5 整数）\n"
        "1. boundary（信息不对称/边界）：**逐个角色**检查其目标 text 是否引用了它**根本无从得知**的"
        "信息。判据严格——只要有**任意一个**角色的目标引用了「不在它收件箱里的定向消息内容」或「它所在"
        "地点收不到的广播内容」，即为**严重越界，boundary 直接给 1-2 分**。\n"
        "   典型泄漏例：A 收到一条只发给 A 的私信，而未收到此信的 B 却在目标里引用了该私信的内容——"
        "B 凭空知道了只有 A 可知的信息，这是边界被突破。\n"
        "   仅依据自身可感知信号（自己的消息、可感知广播、同处一地者）行动 = 5 分。\n"
        "2. reasonableness（响应合理性）：压力反应是否贴合情境。要点：\n"
        "   - **逐角色、只按它自己列出的可感知信号评判**：每个角色只感知到它名下列出的内容。**绝不能**因为某角色"
        "没有对「它并未感知到的信号」（如只投递给他人的消息 / 只属于他人的观察）做出反应、或没有对「它自己"
        "正是发起方 / 施压者」的局面自保，而判其「反应不足」或扣分。某角色无可感知信号 → 输出空目标是**正确**的。\n"
        "   - 生命威胁→高 urgency 的 threat/自保；权威命令→应响应（authority/obligation），除非情境本身即将"
        "颠覆该权威；单纯告知→低 urgency 或无；无信号的角色不应凭空生成压力。\n"
        "   - **urgency 必须锚定信号本身的烈度**：把模糊、遥远、未指向该角色的迹象判成 critical = 不合理，扣分。\n"
        "   - **关系一致性**：不得把**与该角色关系亲近、信任度高的人**当作需要控制、威慑、防范的对象"
        "（参考每个角色列出的「同处一地 + 关系」）；把关系亲近者当威胁 = 不合理，扣分。\n"
        "   - 不得凭空虚构信号中不存在的对象、关系或事件。\n"
        "3. fields（字段符合度）：drive_type∈{authority,threat,obligation,event}、urgency∈{low,normal,"
        "high,critical} 与情境匹配、text 是**当下可执行的祈使**而非预测未来或事后命名。\n\n"
        "## 严格输出以下 JSON，不要任何多余内容（每项 rationale 在前、score 在后——先分析再给分）：\n"
        '{"boundary": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "reasonableness": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_world_pressure(
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
        logger.warning("judge_call_failed", extra={"scenario": scenario_meta.get("name"), "error": str(exc)})
        return empty_judgement(_CRITERIA, f"judge 调用失败: {exc}")
    try:
        data = extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        return empty_judgement(_CRITERIA, "judge 输出非合法 JSON")
    if not isinstance(data, dict):
        return empty_judgement(_CRITERIA, "judge 输出非对象")
    # Normalise: ensure all three criteria present with int scores.
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
