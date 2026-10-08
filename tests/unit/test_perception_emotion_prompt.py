"""Unit tests for `build_perception_emotion_prompt` and `parse_perception_emotion_response`.

Covers what's unit-testable: prompt structure, signal caps, response parsing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.perception_emotion import (
    PerceptionAppraisal, build_perception_emotion_prompt as _raw_build_perception_emotion_prompt,
    parse_perception_emotion_response,
)


def build_perception_emotion_prompt(**kwargs) -> str | None:
    """Join the (system, user) prefix-cache split into one string, so substring/order assertions hold."""
    built = _raw_build_perception_emotion_prompt(**kwargs)
    if built is None:
        return None
    system, user, _signals, _facts = built
    return system + "\n" + user
from agent.need import NeedType
from agent.motivation import ExternalDriveType, ExternalGoal
from agent.relation import PerceivedRelation
from core.interfaces.urgency import Urgency
from core.prompts import (
    EMOTION_INTENSITY_DEFINITION,
    EMOTION_VALENCE_DEFINITION,
    RELATION_SCALE_LEGEND,
    SIGNAL_CAPS,
    URGENCY_SCALE_DESCRIPTION,
    urgency_label,
)
from agent.personality import (
    EmotionState,
    EmotionType,
)
from core.interfaces.perception import (
    PerceivedIdentity,
    PerceivedPresence,
    AmbientEvent,
    Broadcast,
    BroadcastType,
    LocationView,
    SpatialPerception,
)


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


def _personality_stub(
    *,
    name: str = "李世民",
    short_term_goals: list[str] | None = None,
    emotion: EmotionState | None = None,
    condition=None,
) -> SimpleNamespace:
    """Minimal personality stub.

    `to_prompt_context(include_goals=...)` behavior:
    - include_goals=True returns text with goals (injects `state.short_term_goals`)
    - include_goals=False doesn't inject them

    `emotion` defaults to None → EmotionState() (intensity 0.2, below the 0.3 mood-injection
    threshold → not injected).
    """
    state_goals = list(short_term_goals or [])
    cur_emotion = emotion if emotion is not None else EmotionState()

    def to_prompt_context(*, include_goals: bool = True, include_emotion: bool = True) -> str:
        base = f"你是{name}。你的核心性格：果断。"
        if include_goals and state_goals:
            return base + f"\n你眼下最想做的是：{'；'.join(state_goals)}。"
        return base

    return SimpleNamespace(
        soul=SimpleNamespace(name=name),
        # condition matches the real StateLayer: the field always exists, defaulting to None. A stub
        # missing a field forces production code into getattr guards, bending product code to fit a
        # test double.
        state=SimpleNamespace(
            short_term_goals=state_goals, emotion=cur_emotion, condition=condition,
        ),
        to_prompt_context=to_prompt_context,
    )


def _spatial(
    *,
    visible: dict[str, str] | None = None,
    ambient: list[str] | None = None,
) -> SpatialPerception:
    visible = visible or {}
    return SpatialPerception(
        location_id="loc_a",
        location_view=LocationView(name="loc_a", description=""),
        world_time_label="辰时",
        current_step=1,
        visible_agents={k: PerceivedPresence(PerceivedIdentity(name=v)) for k, v in sorted(visible.items())},
        ambient_events=[AmbientEvent(content=text) for text in (ambient or [])],
    )


def _msg(
    sender_id: str, sender_name: str, content: str,
    urgency: Urgency = Urgency.NORMAL, *, is_agent: bool = True,
):
    return SimpleNamespace(
        sender_id=sender_id,
        sender_name=sender_name,
        content=content,
        urgency=urgency,
        sender_is_agent=is_agent,
    )


def test_prompt_includes_current_location_as_backdrop() -> None:
    """The current location (name + description, narrative layer) is background carried only by
    situation_header; there's no duplicate "我此刻身处" signal line, and location_id never leaks."""
    spatial = SpatialPerception(
        location_id="xuanwu_gate",
        location_view=LocationView(name="玄武门", description="皇城北门"),
        world_time_label="卯时", current_step=1,
        visible_agents={"a1": PerceivedPresence(PerceivedIdentity(name="尉迟恭"))},  # one real signal
        ambient_events=[],
    )
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(), spatial=spatial,
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    assert "我此刻身处：" not in prompt                  # no duplicate location signal line
    assert "玄武门" in prompt and "皇城北门" in prompt   # location still there, via the header
    assert "xuanwu_gate" not in prompt                  # location_id doesn't leak


def test_prompt_includes_situation_header_with_time_and_place() -> None:
    """Situation anchoring: the prompt opens with a first-person "我此刻在...,时间为..." header, giving
    both place and time (calendar labels). time_label comes from SpatialPerception.world_time_label;
    code-layer coordinates like steps never appear."""
    spatial = SpatialPerception(
        location_id="xuanwu_gate",
        location_view=LocationView(name="玄武门", description="皇城北门"),
        world_time_label="卯时三刻", current_step=1,
        visible_agents={"a1": PerceivedPresence(PerceivedIdentity(name="尉迟恭"))},
        ambient_events=[],
    )
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(), spatial=spatial,
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    # First-person header with both place and time
    assert "我此刻在玄武门" in prompt and "时间为卯时三刻" in prompt
    # The header comes first (lost-in-the-middle), before the persona/signal blocks
    assert prompt.index("我此刻在玄武门") < prompt.index("【我是谁】")
    assert prompt.index("我此刻在玄武门") < prompt.index("【我此刻感知到的全部信息】")
    # No code-layer coordinates leak
    assert "第1步" not in prompt and "step=" not in prompt


def test_lone_location_with_no_signals_returns_none() -> None:
    """Location only, no perception signals → still None (location is added after the guard, so no
    prompt is fabricated)."""
    spatial = SpatialPerception(
        location_id="x", location_view=LocationView(name="某处", description="d"),
        world_time_label="卯时", current_step=1,
        visible_agents={},
        ambient_events=[],
    )
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(), spatial=spatial,
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is None


def _bc(content: str, severity: str = "low") -> Broadcast:
    return Broadcast(
        content=content,
        source="system",
        broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1,
        severity=severity,
    )


def _goal(text: str, urgency: Urgency, drive: ExternalDriveType = ExternalDriveType.THREAT) -> ExternalGoal:
    return ExternalGoal(
        text=text,
        source_id="world",
        urgency=urgency,
        drive_type=drive,
        related_need=None,
    )


def _rel(
    target_id: str,
    name: str,
    *,
    trust: float = 0.5,
    affection: float = 0.0,
    labels: list[str] | None = None,
    history: str = "",
) -> PerceivedRelation:
    return PerceivedRelation(
        trust=trust,
        affection=affection,
        target_agent_id=target_id,
        target_agent_name=name,
        labels=labels or [],
        history_summary=history,
    )


# ---------------------------------------------------------------------------
# Signal gating
# ---------------------------------------------------------------------------


def test_no_signals_returns_none() -> None:
    """All channels empty → None, no prompt built."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[],
        broadcasts=[],
        external_goals=[],
    )
    assert prompt is None


def test_external_goal_below_threshold_excluded() -> None:
    """external_goal urgency < 0.4 → kept out of the prompt (hard 0.4 threshold)."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[],
        broadcasts=[],
        external_goals=[_goal("低紧迫", Urgency.LOW)],
    )
    # The only signal is filtered by the 0.4 threshold, leaving nothing → None
    assert prompt is None


def test_external_goal_at_threshold_included() -> None:
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[],
        broadcasts=[],
        external_goals=[_goal("临界紧迫", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert "临界紧迫" in prompt


# ---------------------------------------------------------------------------
# Per-channel caps
# ---------------------------------------------------------------------------


def test_ambient_is_capped() -> None:
    """Ask ``SIGNAL_CAPS`` for the cap; don't copy a number here."""
    cap = SIGNAL_CAPS["ambient"]
    spatial = _spatial(ambient=[f"事件{i:02d}" for i in range(cap + 2)])
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=spatial,
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    for i in range(cap):
        assert f"环境观察：事件{i:02d}" in prompt
    for i in range(cap, cap + 2):
        assert f"事件{i:02d}" not in prompt


def test_inbox_cap_20() -> None:
    inbox = [_msg(f"u{i}", f"用户{i}", f"消息{i}") for i in range(22)]
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=inbox, broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    for i in range(20):
        assert f"消息{i}" in prompt
    assert "消息20" not in prompt
    assert "消息21" not in prompt


def test_broadcast_cap_5() -> None:
    broadcasts = [_bc(f"广播{i}") for i in range(7)]
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=broadcasts, external_goals=[],
    )
    assert prompt is not None
    for i in range(5):
        assert f"广播{i}" in prompt
    assert "广播5" not in prompt
    assert "广播6" not in prompt


def test_external_goal_cap_5() -> None:
    """external_goal is sliced to [:5] before urgency filtering; goals among the first 5 with
    urgency>=NORMAL go in."""
    goals = [_goal(f"目标{i}", Urgency.HIGH) for i in range(7)]
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[], external_goals=goals,
    )
    assert prompt is not None
    for i in range(5):
        assert f"目标{i}" in prompt
    assert "目标5" not in prompt
    assert "目标6" not in prompt


def test_visible_agents_are_capped() -> None:
    """The number of co-present people shown on one line comes from ``SIGNAL_CAPS``; don't copy a
    number here."""
    cap = SIGNAL_CAPS["visible"]
    # Zero-padded ids: ``visible_agent_ids`` sorts lexically, so ``a10`` would come before ``a2``.
    visible = {f"a{i:02d}": f"角色{i:02d}" for i in range(cap + 2)}
    spatial = _spatial(visible=visible)
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=spatial,
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    for i in range(cap):
        assert f"角色{i:02d}" in prompt
    for i in range(cap, cap + 2):
        assert f"角色{i:02d}" not in prompt


# ---------------------------------------------------------------------------
# Prompt structure (constants + include_goals=False)
# ---------------------------------------------------------------------------


def test_prompt_includes_intensity_definition() -> None:
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert EMOTION_INTENSITY_DEFINITION in prompt


def test_prompt_includes_valence_definition() -> None:
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert EMOTION_VALENCE_DEFINITION in prompt


def test_need_activation_combines_situation_and_persona() -> None:
    """need_activation must depend on both what happens this moment and the persona: the persona
    interprets the moment, but must not raise the same need by habit regardless of events, nor miss
    the need this moment actually touches. Locked against accidental deletion; asserts only abstract
    key phrases, with no scene/event examples and no named need (e.g. safety)."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert "用我的人设去解读这一刻" in prompt              # persona takes part in interpretation
    assert "都按惯常底色把同一个需求拉高" in prompt        # counter-constraint: habit mustn't override the moment
    assert "漏掉这一刻真正在触动的那个需求" in prompt      # don't miss what should be activated
    assert "尤其 safety" not in prompt                     # names no specific need (anchoring)


def test_mere_presence_is_a_weak_signal() -> None:
    """Mere presence = weak signal, mild reaction; strong reactions need a directed event. Locked
    against deletion (abstract, names no need or scene)."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert "仅仅是某人在场" in prompt
    assert "强烈情绪要有指向我的动作、事件或消息来支撑" in prompt


def test_prompt_does_not_inject_short_term_goals() -> None:
    """Perception emotion must not inject short-term goals. Those are first-person action plans
    with built-in actions (e.g. "sternly confront someone for barging into the palace at night");
    injected, the model echoes the planned action into reason as a perceived fact (fabrication) and
    writes reason as an action plan (overstepping into decision). Recent factual background is
    different and is kept; see the test below."""
    goals_text = "登基称帝"
    personality = _personality_stub(short_term_goals=[goals_text])
    prompt = build_perception_emotion_prompt(
        personality=personality,
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert goals_text not in prompt
    assert "我手头的短期目标" not in prompt
    assert "你眼下最想做的是" not in prompt


def test_prompt_injects_recent_memory_as_background() -> None:
    """Recent factual memory, when supplied, enters as a BACKGROUND block (【我近来的经历】), framed as
    context for reading the current perception — not new events to react to. (Goals stay
    excluded — see test above; only the objective recent-factual backdrop is injected.)"""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
        recent_memory_texts=["约半日前，我与王翦在帐中密谈"],
    )
    assert prompt is not None
    assert "王翦在帐中密谈" in prompt                 # background content (block-specific)
    assert "不是要我对旧事重新生情" in prompt         # background framing (block-specific, against drift)


def test_prompt_omits_recent_memory_block_when_empty() -> None:
    """With no recent background (caller passes empty/None), the whole block is omitted. Asserts the
    block-specific framing string is absent; the section name "我近来的经历" is always in the system
    rules, so it can't signal the block."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert "不是要我对旧事重新生情" not in prompt


def test_prompt_uses_emotion_type_prompt_list() -> None:
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert EmotionType.prompt_list() in prompt


# ---------------------------------------------------------------------------
# Urgency description + perceived relation injection
# ---------------------------------------------------------------------------


def test_goal_uses_urgency_label_not_raw_value() -> None:
    """External pressure lines use the Chinese urgency_label, not the bare enum value."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("叛军压境", Urgency.HIGH)],
    )
    assert prompt is not None
    assert urgency_label(Urgency.HIGH) in prompt
    assert "紧迫度【high】" not in prompt


def test_pressure_injects_urgency_scale_legend() -> None:
    """With an external pressure signal → inject URGENCY_SCALE_DESCRIPTION so the LLM understands
    the scale."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("叛军压境", Urgency.HIGH)],
    )
    assert prompt is not None
    assert URGENCY_SCALE_DESCRIPTION in prompt


def test_no_pressure_omits_urgency_scale_legend() -> None:
    """No external pressure → no urgency legend (avoids irrelevant explanation)."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(visible={"a1": "魏徵"}),
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    assert URGENCY_SCALE_DESCRIPTION not in prompt


def test_perceived_relations_inject_labels_numbers_and_legend() -> None:
    """Perceived relation injection: labels (relation type) + trust/affection numbers (current
    affect) + history + relation_legend() (numeric scale + label format)."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(visible={"a1": "长孙无忌"}),
        inbox=[], broadcasts=[], external_goals=[],
        perceived_relations=[
            _rel("a1", "长孙无忌", trust=0.9, affection=0.8, labels=["挚友"],
                 history="玄武门并肩"),
        ],
    )
    assert prompt is not None
    assert "我对其他人的关系认知" in prompt
    assert "长孙无忌" in prompt and "挚友" in prompt and "玄武门并肩" in prompt
    # Both the current affect numbers and the numeric scale legend should be present
    assert "信任度0.90" in prompt and "好感度0.80" in prompt
    assert RELATION_SCALE_LEGEND in prompt


def test_perceived_relation_without_labels_uses_placeholder() -> None:
    """A relation with no labels → placeholder description, no error."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(visible={"a1": "陌生人"}),
        inbox=[], broadcasts=[], external_goals=[],
        perceived_relations=[_rel("a1", "陌生人")],
    )
    assert prompt is not None
    assert "陌生人：[未明确]" in prompt


def test_perceived_relations_cap_5() -> None:
    rels = [_rel(f"a{i}", f"臣{i}", labels=[f"标签{i}"]) for i in range(7)]
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(visible={f"a{i}": f"臣{i}" for i in range(7)}),
        inbox=[], broadcasts=[], external_goals=[],
        perceived_relations=rels,
    )
    assert prompt is not None
    for i in range(5):
        assert f"标签{i}" in prompt
    assert "标签5" not in prompt
    assert "标签6" not in prompt


def test_no_relations_omits_relation_section() -> None:
    """No perceived relations → no relation block."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(),
        inbox=[], broadcasts=[],
        external_goals=[_goal("foo", Urgency.NORMAL)],
    )
    assert prompt is not None
    assert "我对在场之人的关系认知" not in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_valid_response() -> None:
    raw = json.dumps({"emotion": "fear", "intensity": 0.7, "valence": -0.6, "reason": "宫变将起"})
    appraisal = parse_perception_emotion_response(raw)
    assert isinstance(appraisal, PerceptionAppraisal)
    em = appraisal.emotion
    assert em is not None
    assert em.primary == EmotionType.FEAR
    assert em.intensity == 0.7
    assert em.valence == -0.6
    assert em.triggered_by == "宫变将起"


def test_parse_unknown_emotion_fallback() -> None:
    """Unknown emotion field → parse_emotion_type fallback (usually NEUTRAL or a synonym)."""
    raw = json.dumps({"emotion": "totally_unknown_xyz", "intensity": 0.5, "valence": 0.0, "reason": ""})
    em = parse_perception_emotion_response(raw).emotion
    assert em is not None
    # parse_emotion_type just mustn't raise; NEUTRAL isn't required, since _synonyms may grow
    assert isinstance(em.primary, EmotionType)


def test_parse_intensity_clamp_high() -> None:
    raw = json.dumps({"emotion": "fear", "intensity": 1.8, "valence": 0.0, "reason": ""})
    em = parse_perception_emotion_response(raw).emotion
    assert em is not None
    assert em.intensity == 1.0


def test_parse_intensity_clamp_low() -> None:
    raw = json.dumps({"emotion": "fear", "intensity": -0.5, "valence": 0.0, "reason": ""})
    em = parse_perception_emotion_response(raw).emotion
    assert em is not None
    assert em.intensity == 0.0


def test_parse_valence_clamp_high() -> None:
    raw = json.dumps({"emotion": "joy", "intensity": 0.5, "valence": 1.7, "reason": ""})
    em = parse_perception_emotion_response(raw).emotion
    assert em is not None
    assert em.valence == 1.0


def test_parse_valence_clamp_low() -> None:
    raw = json.dumps({"emotion": "fear", "intensity": 0.5, "valence": -1.5, "reason": ""})
    em = parse_perception_emotion_response(raw).emotion
    assert em is not None
    assert em.valence == -1.0


def test_parse_markdown_fenced_json() -> None:
    """The LLM may use a ```json fence; extract_json strips it."""
    raw = "```json\n" + json.dumps({"emotion": "trust", "intensity": 0.4, "valence": 0.3, "reason": ""}) + "\n```"
    em = parse_perception_emotion_response(raw).emotion
    assert em is not None
    assert em.primary == EmotionType.TRUST


def test_parse_invalid_json_returns_empty_appraisal() -> None:
    """Non-JSON content → emotion=None, need_activation empty; no exception and no default
    fallback."""
    appraisal = parse_perception_emotion_response("not json at all")
    assert appraisal.emotion is None
    assert appraisal.need_activation == {}


def test_parse_non_dict_returns_empty_appraisal() -> None:
    """The LLM returns an array instead of an object → empty appraisal."""
    appraisal = parse_perception_emotion_response("[1, 2, 3]")
    assert appraisal.emotion is None
    assert appraisal.need_activation == {}


# ---------------------------------------------------------------------------
# _perceive_relevant_relations: in view + message senders (non-agent sources filtered, deduplicated)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perceive_relevant_relations_includes_senders_filters_system() -> None:
    from agent.agent import Agent

    captured: dict = {}

    async def fake_perceive_many(ids, *, emotion, memory_biases, identities):
        captured["ids"] = list(ids)
        captured["identities"] = dict(identities)
        return []

    async def fake_significant_relations(*, limit):
        return []  # no established relations → test only the in-view + sender paths

    fake_self = SimpleNamespace(
        agent_id="self-id",
        memory_system=SimpleNamespace(
            related_memory_bias=lambda *, target_agent_id, current_step: 0.0
        ),
        relation_system=SimpleNamespace(
            perceive_many=fake_perceive_many,
            significant_relations=fake_significant_relations,
        ),
        personality=SimpleNamespace(state=SimpleNamespace(emotion=EmotionState())),
        _dead_agent_ids=frozenset(),  # the dead set cached by perceive_step (none here)
        _known_agents={},             # sender gender only from meeting in person
    )
    spatial = _spatial(visible={"v1": "在场者"})
    inbox = [
        _msg("remote", "远方人", "你在何处"),    # absent sender → included
        # Narration also declares itself non-human from the sending side (set when the engine
        # publishes): the receiver trusts this field, not ids. Matching by id misses senders like
        # director, who then get treated as people.
        _msg("narrator", "", "旁白", is_agent=False),
        _msg("v1", "在场者", "我也发了"),         # already in view → deduplicated
        # An NPC reporting back is also a Message. Relations are two-way and there's nobody on its
        # side; letting it in would render "我对他的信任 0.42" and put it on the contact list, earning an
        # undeliverable letter.
        _msg("npc_runner", "太极宫老宦", "我去了皇城一趟，回来了。", is_agent=False),
    ]
    await Agent._perceive_relevant_relations(fake_self, spatial, inbox, current_step=1)

    assert captured["ids"] == ["v1", "remote"]   # non-human filtered, v1 deduped
    assert captured["identities"]["remote"].name == "远方人"
    assert "npc_runner" not in captured["identities"]


@pytest.mark.asyncio
async def test_resolve_relation_context_marks_deceased() -> None:
    """Relation context for memory writes also marks the dead "（已死亡）", consistent with
    decision/emotion/goals. deceased reads self._dead_agent_ids cached by perceive_step (decided by
    the caller)."""
    from agent.agent import Agent
    from core.interfaces.agent_store import AgentRelation

    names = {"live": "活人", "dead": "故人"}

    async def fake_load_existing(aid):
        return AgentRelation(
            world_id="w", from_id="me", to_id=aid,
            trust_objective=0.6, affection_objective=0.3, updated_step=1,
            history_summary="", interaction_count=2, to_name=names[aid], labels=["友"],
        )

    fake_self = SimpleNamespace(
        _known_agents={aid: PerceivedIdentity(name=nm) for aid, nm in names.items()},
        _dead_agent_ids=frozenset({"dead"}),
        relation_system=SimpleNamespace(load_existing=fake_load_existing),
    )
    text = await Agent._resolve_relation_context(fake_self, ["live", "dead"])
    assert "故人（已死亡）" in text                       # the dead are marked
    assert "活人（已死亡）" not in text and "活人" in text  # the living aren't


@pytest.mark.asyncio
async def test_resolve_relation_context_carries_gender_beside_deceased() -> None:
    """Relation lines for memory writes match the decision side: gender and deceased share one
    bracket; with no gender in the perception cache, fall back to the relation's to_gender."""
    from agent.agent import Agent
    from core.interfaces.agent_store import AgentRelation

    genders = {"live": "女", "dead": "男"}

    async def fake_load_existing(aid):
        return AgentRelation(
            world_id="w", from_id="me", to_id=aid,
            trust_objective=0.6, affection_objective=0.3, updated_step=1,
            history_summary="", interaction_count=2, to_name=aid, labels=["友"],
            to_gender=genders[aid],
        )

    fake_self = SimpleNamespace(
        _known_agents={"live": PerceivedIdentity(name="活人"), "dead": PerceivedIdentity(name="故人", gender="男")},
        _dead_agent_ids=frozenset({"dead"}),
        relation_system=SimpleNamespace(load_existing=fake_load_existing),
    )
    text = await Agent._resolve_relation_context(fake_self, ["live", "dead"])
    assert "故人（男，已死亡）" in text
    assert "活人（女）" in text


# ---------------------------------------------------------------------------
# need_activation parsing
# ---------------------------------------------------------------------------


def test_parse_need_activation_valid() -> None:
    raw = json.dumps({"emotion": "fear", "intensity": 0.7, "valence": -0.6, "reason": "危",
                      "need_activation": {"safety": 0.9, "social": 0.3}})
    act = parse_perception_emotion_response(raw).need_activation
    assert act == {NeedType.SAFETY: 0.9, NeedType.SOCIAL: 0.3}


def test_parse_need_activation_clamp_and_drop_unknown() -> None:
    raw = json.dumps({"emotion": "fear", "intensity": 0.5, "valence": -0.2, "reason": "",
                      "need_activation": {"safety": 1.8, "bogus_need": 0.5, "esteem": -0.4}})
    act = parse_perception_emotion_response(raw).need_activation
    assert act == {NeedType.SAFETY: 1.0, NeedType.ESTEEM: 0.0}  # unknown dropped, values clamped


def test_parse_omitted_emotion_returns_none() -> None:
    """No emotion field = no new reaction → emotion None (the caller keeps the current emotion), not
    forced neutral; need_activation is still parsed."""
    raw = json.dumps({"need_activation": {"safety": 0.5}})
    appraisal = parse_perception_emotion_response(raw)
    assert appraisal.emotion is None
    assert appraisal.need_activation == {NeedType.SAFETY: 0.5}


def test_parse_empty_emotion_string_returns_none() -> None:
    raw = json.dumps({"emotion": "", "intensity": 0.2, "valence": 0.0, "reason": ""})
    assert parse_perception_emotion_response(raw).emotion is None


def test_parse_null_emotion_returns_none() -> None:
    assert parse_perception_emotion_response(json.dumps({"emotion": None})).emotion is None


def test_parse_need_activation_missing_defaults_empty() -> None:
    raw = json.dumps({"emotion": "joy", "intensity": 0.4, "valence": 0.4, "reason": ""})
    assert parse_perception_emotion_response(raw).need_activation == {}


def test_parse_need_activation_non_dict_defaults_empty() -> None:
    raw = json.dumps({"emotion": "joy", "intensity": 0.4, "valence": 0.4, "reason": "",
                      "need_activation": "not a dict"})
    assert parse_perception_emotion_response(raw).need_activation == {}


def test_prompt_injects_current_mood_when_intense() -> None:
    """Existing emotion intensity > 0.3 → injected as the current-mood backdrop (explicitly not the
    answer), with a counter-constraint."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(emotion=EmotionState(primary=EmotionType.ANGER, intensity=0.8, valence=-0.7)),
        spatial=_spatial(ambient=["有人当众顶撞你"]),
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    assert "【我当前的心境】" in prompt
    assert "愤怒" in prompt
    assert "不是要我照搬的答案" in prompt


def test_prompt_omits_current_mood_when_calm() -> None:
    """Existing emotion intensity ≤ 0.3 (default 0.2) → no mood block, to avoid noise."""
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),  # default EmotionState() intensity 0.2
        spatial=_spatial(ambient=["远处有动静"]),
        inbox=[], broadcasts=[], external_goals=[],
    )
    assert prompt is not None
    assert "【我当前的心境】" not in prompt


def test_prompt_includes_need_activation_legend_and_field() -> None:
    prompt = build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(ambient=["远处有动静"]),
        inbox=[],
        broadcasts=[],
        external_goals=[],
    )
    assert prompt is not None
    assert "need_activation" in prompt
    # legend lists the canonical needs from NeedType
    assert "safety" in prompt and "self_actualization" in prompt


# ---------------------------------------------------------------------------
# Ongoing personal condition: background, not a signal
# ---------------------------------------------------------------------------


def test_standing_condition_sits_with_the_anchor_not_in_the_signal_block() -> None:
    """Condition, like location, is background and must sit outside 【我此刻感知到的全部信息】.

    The signal block holds what happens this moment. Mixed in, a condition such as being tied up
    would re-trigger the same emotion every step, each written to memory and embedded. As
    background it colors the mood without re-triggering.
    """
    from core.interfaces.condition import BodyCondition

    built = _raw_build_perception_emotion_prompt(
        personality=_personality_stub(condition=BodyCondition("双手被反绑", since_step=10)),
        spatial=_spatial(ambient=["宫门方向骤起兵刃相击"]),
        inbox=[], broadcasts=[], external_goals=[], seconds_per_step=3600,
    )
    assert built is not None
    _system, user, _signals, _facts = built

    assert "我此刻的处境：双手被反绑" in user
    # After the time/place anchor, before the persona; not in the perception block.
    assert user.index("我此刻的处境") < user.index("【我是谁】")
    signal_block = user.split("【我此刻感知到的全部信息】", 1)[1]
    assert "双手被反绑" not in signal_block


def test_condition_line_carries_duration() -> None:
    """Tied up for three days versus just now weigh differently; duration is half of this
    background.

    The fixture's current_step=1, so since_step is 0 and one step is an hour → exactly "约1小时".
    """
    from core.interfaces.condition import BodyCondition

    built = _raw_build_perception_emotion_prompt(
        personality=_personality_stub(condition=BodyCondition("双手被反绑", since_step=0)),
        spatial=_spatial(ambient=["有人走近"]),
        inbox=[], broadcasts=[], external_goals=[], seconds_per_step=3600,
    )
    assert built is not None
    assert "我此刻的处境：双手被反绑（已持续约1小时）" in built[1]
    # Narrative layer: duration is natural language, never a step count.
    assert "第1步" not in built[1] and "step" not in built[1]


def test_no_condition_line_when_unencumbered() -> None:
    built = _raw_build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(ambient=["有人走近"]),
        inbox=[], broadcasts=[], external_goals=[], seconds_per_step=3600,
    )
    assert built is not None
    assert "我此刻的处境" not in built[1]


def test_the_relations_declared_are_the_ones_the_prompt_actually_listed() -> None:
    """The returned relation_lines must be exactly the lines in the prompt, since relation lines
    are capped.

    Re-rendering them outside would bypass that cap: an agent with 6 relations would see only 5 in
    the prompt while the declaration lists 6, so audit cites a relation the model never saw.
    """
    built = _raw_build_perception_emotion_prompt(
        personality=_personality_stub(),
        spatial=_spatial(visible={"a1": "甲"}),
        inbox=[], broadcasts=[], external_goals=[],
        perceived_relations=[_rel(f"a{i}", f"人{i}", trust=0.5) for i in range(1, 8)],
    )
    assert built is not None
    _system, user, _signals, facts = built

    declared = [f.split("：", 1)[1] for f in facts if f.startswith("关系：")]
    assert declared, "用例没造出关系行，这条守卫会空转"
    listed = [ln[2:] for ln in user.splitlines() if ln.startswith("- ") and "信任度" in ln]
    assert declared == listed, "申报的与 prompt 里列出的必须逐条相同（含条数上限）"

    # Every declared fact must be found in the prompt; this holds for the situation header and
    # perception signals, not just relations.
    for fact in facts:
        body = fact.split("：", 1)[1]
        assert body in user, f"申报了 prompt 没铺出去的东西：{fact!r}"


def test_emotion_payload_parsing_is_shared_and_strict_about_presence() -> None:
    """Both appraisals read the payload through one parser: the same defaults and clamps."""
    from agent.perception_emotion import emotion_from_payload
    from agent.personality import EmotionType

    assert emotion_from_payload({"emotion": " "}, triggered_by="x") is None
    assert emotion_from_payload({}, triggered_by="x") is None
    emotion = emotion_from_payload({"emotion": "fear", "valence": -3}, triggered_by="风声")
    assert emotion is not None
    assert (emotion.primary, emotion.intensity, emotion.valence) == (EmotionType.FEAR, 0.3, -1.0)
    assert emotion.triggered_by == "风声"
