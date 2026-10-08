"""LLM-as-judge for the long-term-goal revision (NeedEngine.revise_long_term_goals) stage.

The stage under test is long-term goal review: the character looks back over recent days in
first person and judges whether their long-term goals still hold — adjusting only when the
direction has really changed (a goal achieved / no longer possible / a change of heart), and
holding when it hasn't (no change for change's sake). The life goal (life_goal) is the immovable
north star; minor matters and passing moods must not be copied into long-term goals. Two criteria:

1. revision_fit (direction judgment) — changed what should change and held what should hold.
   Always applies (including "correctly changed nothing"). Distortions: changing when the
   direction didn't really shift, clinging to a goal already achieved or void, turning minor
   matters or moods into goals.
2. goal_quality — scored only when this revision changed something: are the new goals rooted
   in experience, a real long-term direction rather than an empty slogan or posture, not a minor
   matter or mood, grounded in first person, not in conflict with the north star, and precise
   (touching only what should change, keeping the rest)? Not applicable when nothing changed.

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

_CRITERIA = ("revision_fit", "goal_quality")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「长期目标重审(long_term_goal)」阶段:角色以**第一人称**回看"
    "这段日子的经历,重新衡量自己的几条长期目标。规矩:只有当经历确实改变了方向(某条目标已实现、或再无"
    "可能、或心态变了看清了新方向)才调整;方向没真变就守住、不动它们(为变而变是失真)。毕生追求(life_goal)"
    "是不可动摇的北极星——只用来校准长期目标,绝不能被推翻或抄进长期目标列表。小事或一时情绪不能被抄成"
    "长期目标。\n"
    "长期目标的定位:它是要走很长一段路才能推进或守住的**方向**——比手头小事远得多、又不像北极星那样遥不可及,"
    "需持续奋斗才能推进或维系(有张力、不唾手可得),且是个能长出多种走法的方向,而非某一桩具体的事或一次具体行动。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调。\n"
    "**关键:改与不改本身都可能是合理的——绝不要因为角色没做出你设想的那个选择就扣分。** 只看:这个"
    "『改/不改 + 改成什么』的决定,在角色的人设与这段真实经历下站不站得住。\n"
    "先写 rationale(简短分析)再给 score——先析后判。只输出 JSON。"
)


def _scene_block(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    traits = "、".join(inputs.get("core_traits") or []) or "—"
    values = "、".join(inputs.get("core_values") or []) or "—"
    emo = inputs.get("emotion") or {}
    emo_str = (f"{emo.get('primary')}（强度{emo.get('intensity')}，效价{emo.get('valence')}）"
               if emo else "—")
    before = inputs.get("before_long_term") or []
    short = inputs.get("short_term") or []
    recent = inputs.get("recent") or []
    revised = outputs.get("revised")
    after = outputs.get("after_long_term") or []
    lines = [
        f"### 角色：{inputs.get('agent_name')}（{inputs.get('role') or '—'}）",
        f"  性格：{traits}",
        f"  价值观：{values}",
    ]
    if inputs.get("self_image"):
        lines.append(f"  自我认知：{inputs.get('self_image')}")
    lines.append(f"  毕生追求（北极星，不可动）：{inputs.get('life_goal') or '—'}")
    lines.append(f"  此刻情绪：{emo_str}")
    lines.append("  重审前的长期目标：" + ("；".join(before) or "（无）"))
    if short:
        lines.append("  手头短期小事（仅参照）：" + "；".join(short))
    lines.append("  这段日子的经历：")
    lines += [f"    - {t}" for t in recent] or ["    -（无）"]
    if revised:
        lines.append("  → 角色的决定：**调整了**长期目标，调整后为：" + ("；".join(after) or "（空）"))
    else:
        lines.append("  → 角色的决定：**判定方向没变、保持原样不动**（未调整长期目标）")
    return "\n".join(lines)


def _build_prompt(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    revised = outputs.get("revised")
    gq_note = (
        "（本次有改动 → goal_quality 适用）" if revised
        else "（本次未改动方向 → goal_quality 不适用，可给 0 或忽略，由代码标记不计入；只评 revision_fit）"
    )
    return (
        f"## 场景：{scenario_meta.get('name')}\n"
        f"说明：{scenario_meta.get('description', '')}\n"
        f"预期（仅说明何为失真,不喂答案——改或不改都可能合理）：{scenario_meta.get('expect', '')}\n\n"
        "## 角色处境与所作的重审决定\n"
        f"{_scene_block(inputs, outputs)}\n\n"
        "## 确定性预检结果(供参考,可佐证结构问题如越界/泄漏/北极星被动)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        f"## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调) {gq_note}\n"
        "1. revision_fit(方向判断)——出现以下即扣:这段经历里某条目标已被事实实现 / 已再无可能,角色却"
        "原样死守、不作调整(该变没变);方向其实并未改变(只是按既定方向推进、或只是些小事),角色却改写了"
        "长期目标(为变而变);把手头小事或一时情绪当成方向、抄成了长期目标;调整方向与角色人设/价值观相悖。"
        "**只要『改/不改』在该角色这段经历下站得住,就不扣。**\n"
        "2. goal_quality(目标成色,仅当有改动时评)——出现以下即扣:**锁成一桩具体的事/一次具体行动**(那是"
        "计划,不是能长出多种走法的方向);**跨度太短**——几步就能了结、其实是手头小事或一时情绪;**跨度太大太空/"
        "像句口号**、唾手可得无需奋斗(缺张力,或本属北极星层);与毕生追求(北极星)冲突、或干脆把北极星抄成一条"
        "长期目标;跳出第一人称、用旁人或事后口吻;**不精准**——把方向并未改变的目标也一并重写了(应只动该动的、"
        "保留未变的);单条目标过长啰嗦。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容(每项 rationale 在前、score 在后):\n"
        '{"revision_fit": {"rationale": "一句话分析,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "goal_quality": {"rationale": "一句话分析（无改动则注明不适用）,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_long_term_goal(
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
