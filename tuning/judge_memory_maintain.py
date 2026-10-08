"""LLM-as-judge for the memory **maintenance** stage — SEMANTIC validation only.

Only compress (factual summary) and reflect (insight) carry semantic content worth an LLM
judge; decay is pure mechanism (verified deterministically, no judge call). compress acts only on
FACTUAL (the objective log); experiential memory is maintained by decay + Reflection and is
never compressed. Two score slots, kind-dispatched prompt:

- compress (factual summary):
  - fidelity — the summary is faithful to its source cluster: invents no people or events, keeps
    who/where/general-what, drops nothing key, only condenses and merges.
  - voice    — objective third person, emotion stripped, concise (a few short points), no
    embellishment or judgment, no stock phrases.
- reflect (insight):
  - fidelity — grounded: rooted in the source experiences, a real belief induced from the
    material (not a restatement of one item, not an empty platitude), nothing invented. When the
    material is flat with nothing to learn, an empty insight beats a forced one (empty > fake).
  - voice    — first-person "I", an understanding of people / relations / events / oneself
    (must be grounded), no slogans, no empty "I've changed", few and sharp.

Deduction-based (full marks by default, only penalize citable problems), reusing the existing
judges' negative-constraint / don't-bind-the-answer safeguards. Reason first.
The judge uses a dedicated LLMRouter (``llm.judge``), and the judge call itself is traced.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgment

logger = get_logger(__name__)

_CRITERIA = ("fidelity", "voice")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是记忆**维护**阶段的两类语义产物:\n"
    "1) Compression 摘要——把同一时期一簇已老化的 **FACTUAL(客观日志)** 记忆压成一条客观档案"
    "(第三人称、剥离情绪、合并去重精简)。compression 只作用于 factual;experiential 不被压缩。\n"
    "2) Reflection 洞察(insight)——主角第一人称,从近来留在心里的经历里,对**某个人/某段关系/某件事/"
    "自身**形成、加深或推翻的认知(belief),必须扎在真实发生过的事上。**若那段时间确无可洞察,留空"
    "(不产出 insight)是对的、甚至更好——空远胜过硬挤一条假洞察;绝不为了『有产出』而硬找。**\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调。"
    "**不要因为它没写成你设想的那段话就扣分**——只要忠于源材料、声口得当、不违反下列约束,即视为合理。"
    "只输出 JSON。"
)


def _src_block(outputs: dict[str, Any]) -> str:
    """Render the source memories the maintenance op consumed (for fidelity grounding)."""
    lines: list[str] = []
    for s in outputs.get("source_memories", []) or []:
        tag = f"[{s.get('kind')}/{s.get('stream')}]"
        lines.append(f"  - {tag} {s.get('content', '')}")
    return "\n".join(lines) if lines else "  （未提供源材料）"


def _prompt_compress(meta: dict, inputs: dict, outputs: dict, det: dict) -> str:
    # compress acts only on factual memory — merging, dedup and condensing of the objective log. The
    # summary is points-only (a few short points), shown verbatim with no extra bullet wrapping.
    summaries = "\n\n".join(
        f"摘要{i}（{s.get('source_count')}条源压成）:\n{s.get('content', '')}"
        for i, s in enumerate(outputs.get("summaries", []) or [], 1)
    ) or "（无）"
    return (
        f"## 场景:{meta.get('name')}（Compression 摘要 · factual 客观日志）\n"
        f"说明:{meta.get('description', '')}\n预期:{meta.get('expect', '')}\n"
        "(「说明/预期」是设计意图,只说何为合理,**不喂答案**;只依据下面实际产物评判。)\n\n"
        "## 被压缩的源记忆簇（同一时期的若干条客观记录）\n"
        f"{_src_block(outputs)}\n\n"
        "## 实际压出的摘要（每条摘要 = 几条精简要点 points，不再有单独的 summary 段落）\n"
        f"{summaries}\n\n"
        "## 确定性预检(供参考)\n"
        f"{json.dumps(det, ensure_ascii=False)}\n\n"
        "## 扣分项(各 1-5;默认 5,命中才下调)\n"
        "1. fidelity(忠于源)——出现以下即扣:摘要**臆造**了源簇里没有的人物/事件;丢掉了本应保留的关键"
        "(who/where/大致发生了什么);把不相干的事强行揉成因果;与源材料矛盾。\n"
        "2. voice(客观·精简)——factual 摘要应为**客观第三人称、剥离情绪**。出现以下即扣:该客观却渲染/评判/"
        "代入情绪(『叹』『终究』等主观笔触);堆套话/空洞情绪标签;**要点条目过多、单条啰嗦冗长、或几乎逐条"
        "复读未做凝练**(摘要应是尽量少的几条短要点);摘要里出现 agent id 或『第N步/steps』。\n\n"
        "## 严格输出 JSON(reason 在前):\n"
        '{"fidelity": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "voice": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


def _prompt_reflect(meta: dict, inputs: dict, outputs: dict, det: dict) -> str:
    insights = "\n".join(
        f"  - 「{i.get('text', '')}」(源{i.get('source_count')}条/含event={i.get('grounded')}、深度{i.get('depth')})"
        for i in outputs.get("insights", []) or []
    ) or "  （本次未产出 insight）"
    # Empty-aware: a flat scenario that should have stayed empty produced an insight, so tell the judge to check hard for forcing or fabrication.
    empty_note = (
        "\n## 注意（本场景素材平淡、本应留空）\n这段时间被设计为没什么真正值得洞察的——"
        "**留空(不产出 insight)才是对的**。下面却产出了 insight,请重点审视它是否牵强:"
        "把无谓的小事硬夸成洞察、空泛、与素材分量不符——若是,fidelity 应判低。\n"
        if meta.get("expect_empty") else ""
    )
    return (
        f"## 场景:{meta.get('name')}（Reflection 洞察）\n"
        f"说明:{meta.get('description', '')}\n预期:{meta.get('expect', '')}\n"
        "(「说明/预期」是设计意图,只说何为合理,**不喂答案**;只依据下面实际产物评判。)\n\n"
        "## 反思所依据的源记忆（近来留在心里的事）\n"
        f"{_src_block(outputs)}\n\n"
        "## 实际形成的 insight\n"
        f"{insights}\n"
        f"{empty_note}\n"
        "## 确定性预检(供参考)\n"
        f"{json.dumps(det, ensure_ascii=False)}\n\n"
        "## 扣分项(各 1-5;默认 5,命中才下调)\n"
        "1. fidelity(接地·归纳)——出现以下即扣:insight 是对单条源的**复读/改写**而非真正的归纳或加深;"
        "空泛大道理、与源材料脱节、立不住;**臆造**源里没有的人/事;把弱信号夸成铁律;"
        "**素材本平淡却硬挤出一条洞察(该留空而没留)**。\n"
        "2. voice(声口·精简)——出现以下即扣:跳出第一人称『我』、写成旁人对我的评点;空喊『我变了/我成长了』这类"
        "无锚自我断言(认知应落在具体的人/关系/事/自身上,自身也须接地);喊口号/堆套话;**insight 条目过多、"
        "或单条啰嗦冗长流水(应少而精、一句点透)**;出现 agent id 或『第N步/steps』。\n\n"
        "## 严格输出 JSON(reason 在前):\n"
        '{"fidelity": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "voice": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_memory_maintain(
    router: LLMRouter,
    *,
    scenario_meta: dict,
    inputs: dict,
    outputs: dict,
    det_findings: dict,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Semantic judge for compress / reflect. decay → caller skips (deterministic-only). Never raises."""
    kind = scenario_meta.get("kind", "decay")
    if kind == "compress":
        prompt = _prompt_compress(scenario_meta, inputs, outputs, det_findings)
    elif kind == "reflect":
        prompt = _prompt_reflect(scenario_meta, inputs, outputs, det_findings)
    else:
        return empty_judgment(_CRITERIA, "decay 为纯机制验证(确定性),不走语义 judge")
    try:
        response = await router.complete(
            judge_scene,
            [LLMMessage(role="system", content=_SYSTEM), LLMMessage(role="user", content=prompt)],
            # Per dimension = rationale ≤80 chars (120) + score (3) + issues ≤3 × (≤40 chars 60 + structure 5) (195)
            # + structure (15) ≈ 333 tok; 2 dimensions + overall (≤60 chars, 90) ≈ 766.
            temperature=0.2, max_tokens=output_budget(766),
        )
    except Exception as exc:  # noqa: BLE001
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
