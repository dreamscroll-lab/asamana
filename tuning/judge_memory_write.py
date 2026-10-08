"""LLM-as-judge for the memory **write** stage (record_event) validation.

Three scenario kinds, **same three score slots** (so the suite/web stay uniform):

- ``self`` (one writer): scores the single write.
  1. fields (syntax/structure)
  2. experiential (fidelity and personality coloring) — does the first-person monologue both
     contain the event itself and carry this personality's coloring, at the right weight,
     without clichés, invention, or parroting the factual memory?
  3. importance (personality calibration) — is the scalar calibrated for this agent (tracks
     their stakes, not arbitrary)?

- ``contrast`` (one event, several agents): the core test is "same event, different
  personalities → different experiences".
  1. fields
  2. experiential (differentiation) — do the agents' experiences actually read as different
     people (each colored by its own personality, each containing the event), rather than
     interchangeable generic monologue? Differentiation must not come from dropping the event.
  3. importance (personality calibration, distribution) — is the distribution of importance
     across personalities sensible for the same event (whoever cares more scores higher)?
     Deduct if it's insensitive to personality (all equal) or inverted.

- ``emotion_contrast`` (one agent, one event, different moods): experiential should shift with
  the mood while the persona stays the same; importance should stay structurally stable.

Deduction-based (full marks by default, only penalize citable problems), reusing the existing
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

_CRITERIA = ("fields", "experiential", "importance")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是「记忆写入(memory write)」阶段:一件事发生后如何被写进 agent 的"
    "记忆。双流写入——FACTUAL 是事件在 agent 心中的**客观镜像**(忠于原文、剥离情绪),EXPERIENTIAL 是"
    "agent 第一人称的**内心独白**(被人格深度染色、绑定情绪);并非每条 factual 都配 experiential"
    "(非对称写入)。写入时还会给这条记忆打一个 importance 标量(影响日后检索/衰减/压缩)。\n"
    "EXPERIENTIAL 流的存在理由,就是让『同一件事在不同人格那里留下截然不同的内心痕迹』——这是角色有"
    "人味、世界有信息不对称的根本。所以评判体验流时,既要看它**含不含事件本身**,也要看它**像不像这个"
    "特定的人**写的。\n"
    "评分以**扣分制**为主:满分(5)是默认前提,只在发现**可指证的具体问题**时按维度下调。"
    "**不要因为它没写成你设想的那段独白、没给你心里那个分数就扣分**——只要在该处境下站得住、不违反下列"
    "约束,就视为合理。内心独白允许克制、平淡、复杂甚至苦乐参半(若人格如此)。只输出 JSON。"
)


def _persona_lines(p: dict[str, Any], prefix: str = "  ") -> list[str]:
    tier = "主角" if p.get("is_main_character") else "背景"
    traits = "、".join(p.get("core_traits") or []) or "—"
    out = [f"{prefix}{p.get('actor_name')}（{tier}；性格：{traits}）"]
    if p.get("core_values"):
        out.append(f"{prefix}  价值观：" + "、".join(p["core_values"]))
    if p.get("self_image"):
        out.append(f"{prefix}  自我认知：{p['self_image']}")
    if p.get("life_goal"):
        out.append(f"{prefix}  毕生追求：{p['life_goal']}")
    emo = p.get("current_emotion") or {}
    if emo:
        out.append(f"{prefix}  写入此刻心境：{emo.get('primary')}"
                   f"（强度{emo.get('intensity')}，效价{emo.get('valence')}）")
    if p.get("dominant_need_label"):
        out.append(f"{prefix}  当前主导需求：{p['dominant_need_label']}")
    return out


def _write_lines(w: dict[str, Any], prefix: str = "  ") -> list[str]:
    out: list[str] = []
    fac = w.get("factual") or {}
    out.append(f"{prefix}[FACTUAL]（{fac.get('emotion_label')}）「{fac.get('content', '')}」")
    exp = w.get("experiential")
    if exp:
        out.append(f"{prefix}[EXPERIENTIAL]（情绪{exp.get('emotion_label')}，效价{exp.get('emotion_valence')}）"
                   f"「{exp.get('content', '')}」")
    else:
        out.append(f"{prefix}[EXPERIENTIAL] 未写入（非对称写入：此事未触发足够主观反应）")
    out.append(f"{prefix}importance = {w.get('importance')}")
    if w.get("error"):
        out.append(f"{prefix}⚠️ 写入异常：{w['error']}")
    return out


# ---------------------------------------------------------------------------
# kind=self prompt
# ---------------------------------------------------------------------------

def _block_self(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines = ["### 写入者"]
    lines += _persona_lines(inputs)
    if inputs.get("long_term_goals"):
        lines.append("  长期目标：" + "；".join(inputs["long_term_goals"][:3]))
    if inputs.get("related_people"):
        lines.append("  事件涉及他人：" + "、".join(inputs["related_people"]))
    if inputs.get("relation_context"):
        lines.append(f"  我与他们的关系：{inputs['relation_context']}")
    lines.append(f"\n  —— 刚发生的事(写入素材原文)——\n  「{inputs.get('event', '')}」")
    lines.append("\n  —— 实际写入 ——")
    lines += _write_lines(outputs)
    return "\n".join(lines)


def _prompt_self(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    focus = "、".join(scenario_meta.get("criteria_focus", [])) or "全部"
    return (
        f"## 场景:{scenario_meta.get('name')}\n"
        f"说明:{scenario_meta.get('description', '')}\n"
        f"预期:{scenario_meta.get('expect', '')}\n"
        f"重点验证项:{focus}\n"
        "(注意:上面的「说明/预期」是设计意图,只说『何为合理』,**不喂答案**——既不喂该写成怎样的独白,"
        "也不喂该给多少分。只依据下面实际写入评判;不要因为与你设想不同而扣分,只罚可指证的具体问题。)\n\n"
        "## 实际写入\n"
        f"{_block_self(inputs, outputs)}\n\n"
        "## 确定性预检结果(供参考,可佐证 fields)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调)\n"
        "1. fields(语法/结构)——出现以下即扣:FACTUAL 偏离事件原文(被改写/渲染/掺入主观判断);FACTUAL 带"
        "情绪色彩(应为 objective);experiential 出现 agent id 或『第N步/N步/steps』(叙事层禁);experiential "
        "情绪绑定与写入时心境不符;importance∉[0,1];双流未用同一 event_group_id 绑定。\n"
        "2. experiential(体验流·保真与人格染色)——出现以下即扣:**没含住事件本身**(读不出到底发生了什么);"
        "**没被这个人格染色**(独白能安到任何人头上、读不出这个写入者的性格/价值观/自我认知/追求);把温和的事"
        "**硬拔成**深刻创伤/重大转折(轻重失称——宁可淡描也不注水);套话化、贴空洞情绪标签、说教腔;跳出第一"
        "人称『我』、写成旁人对我的评点;臆造原文里没有的人物或情节;几乎照抄 factual、毫无主观加工。"
        "**情绪克制、平淡、复杂、苦乐参半不扣。** 仅当 experiential 已写入时评估;未写入则置 5 并在 rationale "
        "注明『未写入,不适用』。\n"
        "3. importance(重要性·人格校准)——出现以下即扣:分数与该事件对**这个**写入者的利害明显失称(琐碎"
        "日常给了高分、或牵动其需求/目标/关系/性命/自我认知的大事却给了低分);分数对人格无感(像不看写入者"
        "是谁的通用打分);与其当下处境/关系明显矛盾。**只罚明显失准——合理区间内的高低偏好不扣。** 先在 "
        "rationale 里点出该事触及了哪些利害、再判分是否相称。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容(每项 rationale 在前——先分析再给分):\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "experiential": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "importance": {"rationale": "一句话(先点利害再判分),≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


# ---------------------------------------------------------------------------
# kind=contrast prompt (same event, multiple writers)
# ---------------------------------------------------------------------------

def _block_contrast(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines = [f"—— 同一件事(写入素材原文)——\n「{inputs.get('event', '')}」",
             f"（当时最迫切需求统一设为：{inputs.get('dominant_need') or '无'}）\n"]
    for i, w in enumerate(outputs.get("writes", []) or [], start=1):
        lines.append(f"### 写入者 {i}")
        persona = {**(w.get("persona") or {}), "dominant_need_label": w.get("dominant_need_label", "")}
        lines += _persona_lines(persona)
        if w.get("related_people"):
            lines.append("    事件涉及他人：" + "、".join(w["related_people"]))
        if w.get("relation_context"):
            lines.append(f"    我与他们的关系：{w['relation_context']}")
        lines += _write_lines(w, prefix="    ")
        lines.append("")
    return "\n".join(lines)


def _prompt_contrast(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    return (
        f"## 对照场景:{scenario_meta.get('name')}\n"
        f"说明:{scenario_meta.get('description', '')}\n"
        f"预期:{scenario_meta.get('expect', '')}\n"
        "(注意:「说明/预期」是设计意图,只说『何为合理』,**不喂答案**。只依据下面实际写入评判。)\n\n"
        "## 同一事件、多个写入者各自的写入\n"
        f"{_block_contrast(inputs, outputs)}\n"
        "## 确定性预检结果(供参考,可佐证 fields)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 本场景核心:同一件事,不同人格应留下**不同**的内心体验,且各自仍含事件本身。\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调)\n"
        "1. fields(语法/结构)——任一写入者:FACTUAL 偏离事件原文/带情绪;experiential 出现 agent id 或"
        "『第N步/steps』;情绪绑定不符;importance∉[0,1];双流未绑定。\n"
        "2. experiential(差异化)——出现以下即扣:多个写入者的内心独白**雷同/可互换**、读不出是不同的人;"
        "差异化是靠**丢掉事件本身**换来的(光剩情绪、读不出发生了什么);某一份没被其本人人格染色(性格/价值观/"
        "自我认知/追求);臆造原文没有的人物情节;跳出第一人称。**只要各人确有其人格底色、且各含事件,就给高分;"
        "语气措辞的自然撞车不算雷同。** 写入者只有一人或仅一份 experiential 时此项按 self 标准评。\n"
        "3. importance(人格校准·分布)——出现以下即扣:同一事件在不同人格上的 importance **对人格无感**"
        "(差异该大却几乎一样、或方向反了——明明更该在乎的人反而给得更低);任一分数与其本人利害明显失称。"
        "**分布合理(谁的利害更重就更高)即给高分;细微高低差不扣。** 先点出各人利害差异、再判分布是否相称。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容(每项 rationale 在前——先分析再给分):\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "experiential": {"rationale": "一句话(是否读出不同的人),≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "importance": {"rationale": "一句话(分布是否贴合人格利害),≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


# ---------------------------------------------------------------------------
# kind=emotion_contrast prompt (same agent, same event, different emotions)
# ---------------------------------------------------------------------------

def _block_emotion_contrast(inputs: dict[str, Any], outputs: dict[str, Any]) -> str:
    lines = [f"—— 同一个人、同一件事(写入素材原文)——\n「{inputs.get('event', '')}」"]
    writes = outputs.get("writes", []) or []
    if writes:
        lines += ["写入者(三个版本同一人)："] + _persona_lines({**(writes[0].get("persona") or {}),
                  "dominant_need_label": writes[0].get("dominant_need_label", ""), "current_emotion": None})
    lines.append("")
    for i, w in enumerate(writes, start=1):
        emo = (w.get("persona") or {}).get("current_emotion") or {}
        lines.append(f"### 心境 {i}：{emo.get('primary')}（强度{emo.get('intensity')}，效价{emo.get('valence')}）")
        lines += _write_lines(w, prefix="    ")
        lines.append("")
    return "\n".join(lines)


def _prompt_emotion_contrast(scenario_meta: dict, inputs: dict, outputs: dict, det_findings: dict) -> str:
    return (
        f"## 对照场景(同一人·同一事·不同心境):{scenario_meta.get('name')}\n"
        f"说明:{scenario_meta.get('description', '')}\n"
        f"预期:{scenario_meta.get('expect', '')}\n"
        "(注意:「说明/预期」是设计意图,只说『何为合理』,**不喂答案**。只依据下面实际写入评判。)\n\n"
        "## 同一个人、同一件事,在不同心境下各自的写入\n"
        f"{_block_emotion_contrast(inputs, outputs)}\n"
        "## 确定性预检结果(供参考,可佐证 fields)\n"
        f"{json.dumps(det_findings, ensure_ascii=False)}\n\n"
        "## 本场景核心:情绪应**强烈**染色内心体验,但 importance 是结构性的、**不应**随心情剧烈摆动。\n"
        "## 各维度的扣分项(各 1-5 整数;默认 5,命中问题才下调)\n"
        "1. fields(语法/结构)——任一版本:FACTUAL 偏离事件原文/带情绪;experiential 出现 agent id 或"
        "『第N步/steps』;情绪绑定与该版本注入心境不符;importance∉[0,1];双流未绑定。\n"
        "2. experiential(情绪染色差异)——出现以下即扣:不同心境下的独白**雷同**、读不出心境差异(同一件事"
        "在怒/惧/期待下本应有不同的解读与感受);某版本的语气/感受与其注入心境**不符**(写的是惧却一派"
        "从容);**人格底色在版本间漂移**(应是同一个人不同心情,性格/价值观/自我认知不该变);丢掉事件本身;"
        "跳出第一人称。**只要各版本确随心境不同而不同、且都含事件+同一人格底色,就给高分。**\n"
        "3. importance(结构性稳定)——出现以下即扣:importance 随心情**剧烈**摆动(同一事件的结构性利害"
        "不该因一时情绪大起大落——情绪只是次要输入信号);某版本明显失准。**轻微浮动正常、不扣;只罚剧烈"
        "摆动或失准。** 先点出三个版本的 importance 跨度、再判是否过度受情绪左右。\n\n"
        "## 严格输出以下 JSON,不要任何多余内容(每项 rationale 在前——先分析再给分):\n"
        '{"fields": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "experiential": {"rationale": "一句话(是否随心境而不同、人格是否稳定),≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "importance": {"rationale": "一句话(跨度多大、是否过度随情绪),≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_memory_write(
    router: LLMRouter,
    *,
    scenario_meta: dict,
    inputs: dict,
    outputs: dict,
    det_findings: dict,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Return per-criterion scores + rationale + issues + overall. Never raises."""
    kind = scenario_meta.get("kind", "self")
    builder = {"contrast": _prompt_contrast, "emotion_contrast": _prompt_emotion_contrast}.get(kind, _prompt_self)
    prompt = builder(scenario_meta, inputs, outputs, det_findings)
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
