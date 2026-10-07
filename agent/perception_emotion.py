"""The pre-decision appraisal: the instinctive emotion and need activation a perceived moment stirs.

Builds the in-character prompt from what an agent perceives this step and parses the reply;
``Agent`` decides when to ask and which model tier answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from agent.motivation import ExternalGoal
from agent.need import NeedType, need_activation_legend
from agent.personality import (
    EmotionState,
    EmotionType,
    PersonalityLayer,
    parse_emotion_type,
)
from agent.relation import PerceivedRelation, render_relation_lines
from core.interfaces.llm import extract_json
from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, Situation, SpatialPerception
from core.interfaces.urgency import Urgency
from core.numeric import clamp
from typing import NamedTuple

from core.context import GivenFacts
from core.prompts import (
    EMOTION_INTENSITY_DEFINITION,
    EMOTION_VALENCE_DEFINITION,
    URGENCY_SCALE_DESCRIPTION,
    SituationVoice,
    condition_line,
    relation_legend,
    render_perceived_signals,
    render_situation_header,
    urgency_label,
)


@dataclass(frozen=True)
class PerceptionAppraisal:
    """One pre-decision appraisal: an instinctive emotion plus per-need activation.

    ``emotion`` is None when no signal warranted a reaction (the caller keeps last-step emotion).
    ``need_activation`` is empty when the LLM produced nothing; need scoring then falls back to
    disposition (I×W) plus structured runtime adjustments, so it never depends on the LLM.
    """

    emotion: EmotionState | None
    need_activation: Dict[NeedType, float] = field(default_factory=dict)


class PerceptionPrompt(NamedTuple):
    """What this perception call puts in front of the model.

    ``signals`` feeds a structured trace field (the cross-step view); ``facts`` is the same
    material in declared form (sass reads it as "what it was given"). Both come from one pass.
    """

    system: str
    user: str
    signals: list[str]
    facts: GivenFacts


def build_perception_emotion_prompt(
    *,
    personality: PersonalityLayer,
    spatial: SpatialPerception,
    inbox: List[Message],
    broadcasts: List[Broadcast],
    external_goals: List[ExternalGoal],
    perceived_relations: "List[PerceivedRelation] | None" = None,
    recent_memory_texts: "list[str] | None" = None,
    seconds_per_step: int = 3600,
) -> PerceptionPrompt | None:
    """Build the perception emotion prompt; None when no channel has a signal (skip the call).

    ``signals`` and ``facts`` are handed out from here, not re-rendered by the caller: only this
    function knows what was actually laid out (relation lines are capped, the header may be
    omitted), and a recomputation would let the audit cite things the model never saw.
    Field definitions vary with has_pressure, so they go in the user prompt, not system.
    """
    # External pressure is agent-layer (needs urgency_label), so it's appended here, not in core.
    signals: list[str] = render_perceived_signals(spatial=spatial, inbox=inbox, broadcasts=broadcasts)
    has_pressure = False
    for goal in external_goals[:5]:
        if goal.urgency >= Urgency.NORMAL:
            has_pressure = True
            signals.append(
                f"外部压力（{goal.drive_type.value}，紧迫度{urgency_label(goal.urgency)}）：{goal.text}"
            )
    if not signals:
        return None

    signals_block = "\n".join(f"- {s}" for s in signals)

    # Relations to those present: an old friend brings warmth, a rival wariness.
    relation_lines = render_relation_lines(perceived_relations or [])
    relation_block = (
        "\n我对其他人的关系认知（也属于我此刻的感知）：\n"
        + f"关系刻度：{relation_legend()}\n"
        + "\n".join(f"- {l}" for l in relation_lines)
        if relation_lines
        else ""
    )

    # List only the legends relevant to this call's signals; unrelated ones dilute the prompt.
    field_defs = [
        f"  - {EMOTION_INTENSITY_DEFINITION}",
        f"  - {EMOTION_VALENCE_DEFINITION}",
    ]
    if has_pressure:
        field_defs.append(f"  - {URGENCY_SCALE_DESCRIPTION}")
    field_defs.append(f"  - {need_activation_legend()}")
    field_defs_block = "\n".join(field_defs)

    emotion_types = EmotionType.prompt_list()
    # Persona (include_emotion=False: mood has its own block), prior mood (a backdrop, not the
    # answer) and perceived signals (the only input). Don't let the old emotion be copied, or
    # persona/backstory be read as happening now.
    cur_emotion = personality.state.emotion
    mood_block = (
        f"\n\n【我当前的心境】（这是我在感知下列信息之前就有的既有情绪，属于我当前处境的一部分，不是要我照搬的答案）\n我此前感到{cur_emotion.summary()}。"
        if cur_emotion.intensity > 0.3 else ""
    )
    # Recent past excludes this step (it would duplicate the signals) and is background only
    # (system rule 5). Goals are not injected: short-term goals are plans with preset moves and
    # would echo into reason as if perceived.
    recent_block = (
        "\n\n【我近来的经历】（只作背景，帮我掂量此刻感知到的信息意味着什么、是不是某件事的延续或升级；不是要我对旧事重新生情）\n"
        + "\n".join(f"- {t}" for t in (recent_memory_texts or []))
        if recent_memory_texts else ""
    )
    # The location is backdrop (a fortress at midnight vs one's own courtyard) and lives only in
    # this header; the signal block must not repeat it. Rule 1 keeps it from sparking strong emotion.
    situation_header = render_situation_header(
        Situation.from_spatial(spatial), voice=SituationVoice.FIRST,
    )
    # An ongoing condition is backdrop and sits with the header. Don't move it into the signal
    # block: a bound person would re-trigger emotion about being bound every step, though that
    # was an event only on the step they were tied up.
    condition_block = condition_line(
        personality.state.condition, voice=SituationVoice.FIRST,
        now_step=spatial.current_step, seconds_per_step=seconds_per_step,
    )
    system_prompt = f"""\
你此刻完全代入一个角色，以第一人称「我」把此刻感知到的信息化作情绪与需求波动。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【我要做的】
以我这个人的第一视角，仅就我此刻感知到的信息，给出此刻未经权衡的即时情绪反应——这是我决策之前的本能第一反应，不是计划或决定；强度与信号本身的烈度相称。
再结合**这一刻感知到的信息**和我是谁，判断这一刻触动了我哪些需求（对照字段定义里各需求的描述）：只给这一刻真正触动的打分，其余省略。

【我要守的】
 1.不为填满输出而臆造。reason 只能落在我所感知到的信息上，绝不脑补我没感知到的人和事；情绪与需求的烈度由信号本身决定，而非由"此刻该有反应"这种预期决定。**仅仅是某人在场**（没有指向我的动作或事件）是**弱信号**，至多引起与关系相称的**温和**反应（轻度警觉或轻度暖意）；**强烈情绪要有指向我的动作、事件或消息来支撑**——绝不把"某人在场"脑补成他正在做什么（逼近、窥视、密谈等信号里并不存在的举动）。
 2.既有心境只是底色，不是答案。我当前的心境（若有给出）只是这股本能反应的起点——新信息可能强化它、扭转它、或不触动它；我不照搬既有情绪当输出，只给这些新信息真正激起的那一下。
 3.need_activation 由**这一刻发生的事**和**我是谁**共同决定：用我的人设去解读这一刻对我意味着什么——但不论这一刻是关于什么、都按惯常底色把同一个需求拉高，那是用底色盖过了当下；也不漏掉这一刻真正在触动的那个需求（该激活的没激活，比把无关需求抬高更糟）。
 4.我可以没有反应。如果这些信息确实不足以激起任何新的情绪波动，就**整体省略 emotion 字段**（系统会沿用我原有的情绪）；同样，没有哪个需求被这一刻真正触动时，**省略或留空 need_activation**。宁可不输出，也不硬凑戏剧性的情绪或需求。
 5.【我近来的经历】（若给出）只是背景——帮我掂量此刻感知到的信息是不是某件事的延续或升级、分量该有多重；但它本身不是此刻的感知信息。我绝不因为"之前发生过某事"就凭空生出情绪、或把某个需求硬抬高；情绪与 need_activation 仍只能由我这一刻真正感知到的信息激起。
 6.严格输出以下 JSON（emotion / need_activation 无反应时可省略），不写任何多余内容。先在 reason 里点出"我是被哪条感知信息触动的"，再据此给这一下的情绪——情绪由这个触点推出，而不是先挑个情绪再补理由；若没有任何信息触动我，就连同 reason 一起整体省略 emotion：
{{"reason": "第一人称，我这股反应是被哪条感知信息激起的，一句话不超过20字（无 emotion 时一并省略）", "emotion": "有反应时从以下选一，否则省略：{emotion_types}", "intensity": 0.0~1.0, "valence": -1.0~1.0, "need_activation": {{"<需求名>": 0.0~1.0}}}}"""
    user_prompt = f"""\
{situation_header}{condition_block}

【我是谁】（我固有的身份、性格、价值观；这是我这个人，不是当下正在发生的事）
{personality.to_prompt_context(include_goals=False, include_emotion=False)}{mood_block}{recent_block}

【我此刻感知到的全部信息】（这是我当前唯一的输入；除此之外我一无所知，不臆测未列出的人、事或动向）
{signals_block}{relation_block}

【字段定义】
{field_defs_block}

我依上面说定的规矩与 JSON 格式说出此刻的即时情绪与需求波动，只输出 JSON、不写任何多余内容。"""
    facts = (
        GivenFacts()
        .add("此刻何时何地", situation_header)
        .add("此刻感知到", signals)
        .add("关系", relation_lines)
    )
    return PerceptionPrompt(system_prompt, user_prompt, signals, facts)


def parse_perception_emotion_response(content: str) -> PerceptionAppraisal:
    """Parse an appraisal response. A missing or empty emotion means "no new reaction"
    (emotion=None), not a default neutral; so do invalid JSON and wrong field types."""
    empty = PerceptionAppraisal(emotion=None, need_activation={})
    try:
        data = extract_json(content)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    need_activation = _parse_need_activation(data)
    try:
        emotion = emotion_from_payload(data, triggered_by=str(data.get("reason", "")))
    except (TypeError, ValueError):
        emotion = None
    return PerceptionAppraisal(emotion=emotion, need_activation=need_activation)


def emotion_from_payload(data: dict, *, triggered_by: str) -> EmotionState | None:
    """An emotion JSON payload (``emotion`` / ``intensity`` / ``valence``) → EmotionState, shared by
    the perception and feedback appraisals. None when ``emotion`` is missing or empty; a field of
    the wrong type raises TypeError/ValueError for the caller's fallback."""
    raw_emotion = data.get("emotion")
    if raw_emotion is None or str(raw_emotion).strip() == "":
        return None
    return EmotionState(
        primary=parse_emotion_type(str(raw_emotion)),
        intensity=clamp(float(data.get("intensity", 0.3)), 0.0, 1.0),
        valence=clamp(float(data.get("valence", 0.0)), -1.0, 1.0),
        triggered_by=triggered_by,
    )


def _parse_need_activation(data: dict) -> Dict[NeedType, float]:
    """Extract a per-need activation map from an appraisal payload (lenient)."""
    raw = data.get("need_activation")
    if not isinstance(raw, dict):
        return {}
    out: Dict[NeedType, float] = {}
    for key, value in raw.items():
        try:
            need = NeedType(str(key))
            activation = float(value)
        except (TypeError, ValueError):
            continue
        out[need] = max(0.0, min(1.0, activation))
    return out
