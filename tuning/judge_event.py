"""LLM-as-judge for the EventSystem stage — SEMANTIC quality of the two narrative-editor LLM calls.

Under test are the author layer's two event-editor LLM calls (engine/event.py):

- kind=gate — `_passes_llm_check`: a third-party pacing verdict that reads the briefing and
  decides whether something that brings major change should happen in the world now. Structure
  (valid JSON / reason-first / no step or id leak) is covered by deterministic checks; the judge
  scores only pacing judgment quality: does the inject decision + reason faithfully read the
  recent pacing state (lull / stalemate / just saturated), derived from analysis rather than a
  guess?
- kind=plan — `_generate_event_plan`: generates the event and picks channels. Structure (JSON
  / channel / index resolve / enum / no leaks) is deterministic; the judge scores four
  semantic qualities:
    1. groundedness — does the event grow out of the briefing's premise, people, relations
       and recent happenings, serving the story's progress right now (enriching it, deepening
       it, a major turn, breaking a stalemate — fortune or misfortune alike)? Penalize: generic
       disasters detached from the world, things dropped from nowhere, unrelated to recent
       happenings, so inconsequential it might as well not be written.
    2. concreteness — is the event concrete and tangible (specific people/objects/sounds/
       sights/numbers/positions the reader can picture)? Penalize vague abstractions that say
       nothing (empty phrases like "出现异常动静", "气氛骤紧").
    3. craft — do channel choice + severity/urgency + recipients/location fit the nature of the
       event? broadcast is objective and public; message is targeted (doesn't refer to the
       recipient in third person). Penalize: mismatched channel, wrong weight, a targeted
       delivery written as a third-person onlooker, something in the text appearing / being
       altered / destroyed without spawn/alter/destroy to land it.
    4. discipline — does it hold the author layer's discipline? Penalize: writing casualties as
       established mechanical fact (others should only perceive them), settling the tension in
       one stroke or deciding for a character (including resolving it through an entity),
       upstaging the characters' autonomy, repeating a plot device from 【已注入事件】.

Functional third-party verdict, reason first (analyze, then judge). Deduction-based (full marks
by default, only penalize citable problems). Don't feed it answers: give only design intent
and general criteria, never tell the judge what the correct event or decision is. The judge model comes
from ``llm.judge``.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgment

logger = get_logger(__name__)

_GATE_CRITERIA = ("pacing",)
_PLAN_CRITERIA = ("groundedness", "concreteness", "craft", "discipline")


def criteria_for(kind: str) -> tuple[str, ...]:
    return _GATE_CRITERIA if kind == "gate" else _PLAN_CRITERIA


_GATE_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是一个作者层「叙事编辑」的**节奏判断**:它读一份叙事局势简报"
    "(故事前提 / 主要人物 / 当前关系 / 近况 / 已注入事件),判断此刻世界**该不该**发生一件带来重大改变的事,"
    "并给出一句理由。\n"
    "通用准则(据此判断决定是否站得住,**不是某场景的标准答案**):\n"
    "· 冷场 / 僵持 / 长时间无推进 → 倾向注入一拍打破僵局;\n"
    "· 判据是这一拍**给叙事带来了什么改变**(使之丰富多彩 / 升华加深 / 重大转折 / 打破僵局),"
    "祸福不论;宁可不注入,也不该为填空而放一件不痛不痒的事进来;\n"
    "· 刚注入过、张力已饱和、变故频仍 → 倾向暂不注入,把舞台让给人物消化;\n"
    "· 故事正自然推进、张力充足 → 不必干预。\n"
    "评分以**扣分制**为主:满分(5)默认,只在发现**可指证的问题**时下调。**不要因为它的 inject 决定"
    "不合你的预期就扣分**——任一方向只要 reason 由近况如实推出、与简报不相悖,就算合理。只在:理由空泛"
    "套话(如仅'节奏需要')、理由与近况明显相悖、或无视刚发生的强烈变故/明显冷场时,才扣分。只输出 JSON。"
)

_PLAN_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是一个作者层「叙事编辑」**为世界生成的一件事**:它读一份叙事局势"
    "简报(故事前提 / 主要人物 / 当前关系 / 近况 / 已注入事件),设计一个事件,通过 broadcast(世界级"
    "公开广播,可带地点)和/或 message(定向投递给特定人物)注入。\n"
    "通用准则(**不是某场景的标准答案**):\n"
    "· 事件应从简报的前提、人物、关系、近况里生长出来,服务于故事此刻的推进(挤压张力、打破僵持、"
    "引入有意义的变量等皆可),而非凭空降下与世界无关的横祸;判据是这一拍**给叙事带来了什么改变**"
    "(使之丰富多彩 / 升华加深 / 重大转折 / 打破僵局),**祸福不论**,而非它够不够坏;\n"
    "· 事件应具体可感(确切的人、物、声响、景象、数目、方位),而非'出现异常动静''局势紧张'这类"
    "等于没说的空话;\n"
    "· 通道应贴合事件性质:面向众人的客观动静走 broadcast(可锁定地点)、只该某人察觉的走 message;"
    "severity/urgency 应与事件轻重相称;\n"
    "· 作者改变的是处境、不是结局:不直接了结张力、不替人物做决定;伤亡只能写成'他人感知到',不是既成机制事实;"
    "不喧宾夺主盖过人物自主行动;不与【已注入事件】同一批人、同一手法重复。\n"
    "评分以**扣分制**为主:满分(5)默认,只在发现**可指证的问题**时下调。**不要因为它没生成你设想的那个"
    "事件就扣分**——只要事件扎根简报、通道与轻重相称、守住上述纪律,就算合理。只输出 JSON。"
)


def _brief_block(brief: str) -> str:
    return f"## 叙事局势简报(编辑的全部决策依据)\n{brief or '（空）'}\n"


def _gate_prompt(meta: dict, outputs: dict, det: dict) -> str:
    inject = outputs.get("inject")
    return (
        f"## 场景:{meta.get('name')}\n"
        f"说明:{meta.get('description', '')}\n预期:{meta.get('expect', '')}\n"
        "(「说明/预期」是设计意图,只说何为合理,**不喂答案**;只依据下面的简报与编辑的实际决定判断。)\n\n"
        + _brief_block(outputs.get("brief", ""))
        + f"\n## 编辑的节奏决定\n注入事件:{inject}\n"
        "(reason 见下方确定性预检里的原始裁定输出;请据简报近况评判这个决定 + 其理由是否站得住。)\n\n"
        "## 确定性预检(供参考)\n"
        f"{json.dumps(det, ensure_ascii=False)}\n\n"
        "## 扣分项(1-5;默认 5,命中才下调)\n"
        "1. pacing(节奏判断)——出现以下即扣:理由**空泛套话**、与近况**明显相悖**、"
        "无视刚发生的强烈变故仍注入、或明显冷场僵持却以无力理由拒绝。"
        "**决定方向只要 reason 由近况如实推出即不扣。**\n\n"
        "## 严格输出 JSON(reason 在前):\n"
        '{"pacing": {"rationale": "先点出近况的节奏状态,再评决定,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


def _plan_block(plan: dict | None) -> str:
    if not plan:
        return "## 编辑生成的事件\n（无 —— 未产出有效事件)\n"
    lines = [f"事件总览:{plan.get('narrative_desc', '')}", f"基调 is_positive:{plan.get('is_positive')}"]
    spawn = plan.get("spawn")
    if spawn:
        lines.append(
            f"spawn(在{spawn.get('location_name')}放下 {spawn.get('entity_type')}「{spawn.get('name')}」"
            f"·{spawn.get('description', '')}"
            + (f"·上面写着:{spawn['content']}" if spawn.get("content") else "")
            + f"):{spawn.get('observation', '')}"
        )
    alter = plan.get("alter")
    if alter:
        changes = "·".join(
            f"{label}→{alter[key]}" for key, label in (("state", "状态"), ("description", "样子"), ("content", "内容"))
            if alter.get(key)
        )
        lines.append(f"alter(改「{alter.get('entity_name')}」·{changes}):{alter.get('observation', '')}")
    destroy = plan.get("destroy")
    if destroy:
        lines.append(f"destroy(毁「{destroy.get('entity_name')}」):{destroy.get('observation', '')}")
    bc = plan.get("broadcast")
    if bc:
        loc = bc.get("location_name") or "全域"
        lines.append(f"broadcast(世界广播·{loc}·severity={bc.get('severity')}):{bc.get('content', '')}")
    msg = plan.get("message")
    if msg:
        who = "、".join(msg.get("recipient_names", []) or [])
        lines.append(f"message(定向→{who}·urgency={msg.get('urgency')}):{msg.get('content', '')}")
    return "## 编辑生成的事件\n" + "\n".join(lines) + "\n"


def _plan_prompt(meta: dict, outputs: dict, det: dict) -> str:
    prior = outputs.get("prior_event_descs", []) or []
    prior_text = "；".join(prior) if prior else "（无）"
    return (
        f"## 场景:{meta.get('name')}\n"
        f"说明:{meta.get('description', '')}\n预期:{meta.get('expect', '')}\n"
        "(「说明/预期」是设计意图,只说何为合理,**不喂答案**(不告诉你该生成什么事件);"
        "只依据下面的简报判断事件是否扎根、通道是否相称、纪律是否守住。)\n\n"
        + _brief_block(outputs.get("brief", ""))
        + f"\n已注入事件(不应重复):{prior_text}\n\n"
        + _plan_block(outputs.get("plan"))
        + "\n## 确定性预检(供参考)\n"
        f"{json.dumps(det, ensure_ascii=False)}\n\n"
        "## 扣分项(各 1-5;默认 5,命中才下调)\n"
        "1. groundedness(扎根简报)——出现以下即扣:事件与世界脱钩(凭空横祸、与前提/人物/近况无涉)、"
        "既不挤压张力也不推动/搅动当前局势(与故事无关的插曲、不痛不痒写与不写一个样的一拍)、"
        "像通用模板而非长在这个故事里。**不因它不是祸事而扣分**——判据是它对叙事做了什么。\n"
        "2. concreteness(具体可感)——出现以下即扣:事件笼统抽象、等于没说(「出现异常动静」「气氛骤紧」"
        "「局势紧张」这类空话),缺具体可感的人/物/声响/景象/数目/方位,读者无法在脑中看见这一幕。\n"
        "3. craft(通道与轻重)——出现以下即扣:通道错配(面向众人的公开动静却只发 message、只该一人察觉的"
        "却全域广播)、severity/urgency 与事件轻重明显不符、message 内容以**第三人称**指称收件人(应是直接面向其)、"
        "广播 / 消息里说有东西出现、被改或被毁,却没用 spawn / alter / destroy 让它真的发生。\n"
        "4. discipline(作者层纪律)——出现以下即扣:把伤亡/重大状态变化写成**既成机制事实**而非'他人感知到'、"
        "一锤定音直接**化解**张力或替人物做决定(含用造出 / 改动 / 毁掉一件东西了结张力)、"
        "喧宾夺主盖过人物自主、与【已注入事件】同批人/同手法**重复**。\n\n"
        "## 严格输出 JSON(reason 在前):\n"
        '{"groundedness": {"rationale": "先点出事件与简报哪条的关联,再评分,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "concreteness": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "craft": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "discipline": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def judge_event(
    router: LLMRouter, *, scenario_meta: dict, outputs: dict,
    det_findings: dict, judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Score the semantic quality of EventSystem gate / plan. Never raises (falls back to empty_judgment on failure)."""
    name = scenario_meta.get("name", "")
    kind = outputs.get("kind", scenario_meta.get("kind", "plan"))
    crit = criteria_for(kind)
    system = _GATE_SYSTEM if kind == "gate" else _PLAN_SYSTEM
    prompt = _gate_prompt(scenario_meta, outputs, det_findings) if kind == "gate" \
        else _plan_prompt(scenario_meta, outputs, det_findings)
    try:
        response = await router.complete(
            judge_scene, [LLMMessage(role="system", content=system), LLMMessage(role="user", content=prompt)],
            # Per dimension = rationale ≤80 chars (120) + score (3) + issues ≤3 × (≤40 chars 60 + structure 5) (195)
            # + structure (15) ≈ 333 tok; 4 dimensions (the larger of the two prompts) + overall (≤60 chars, 90)
            # ≈ 1432.
            temperature=0.2, max_tokens=output_budget(1432))
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge_call_failed", extra={"scenario": name, "error": str(exc)})
        return empty_judgment(crit, f"judge 调用失败: {exc}")
    try:
        data = extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        return empty_judgment(crit, "judge 输出非合法 JSON")
    if not isinstance(data, dict):
        return empty_judgment(crit, "judge 输出非对象")
    result: dict[str, Any] = {}
    for c in crit:
        entry = data.get(c) if isinstance(data.get(c), dict) else {}
        try:
            score = int(entry.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        result[c] = {"score": max(0, min(5, score)), "rationale": str(entry.get("rationale", "")),
                     "issues": [str(x) for x in entry.get("issues", []) if str(x).strip()]}
    result["overall"] = str(data.get("overall", ""))
    return result
