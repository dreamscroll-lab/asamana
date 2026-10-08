"""Gap-filler: _build_decision_prompt + helpers + IndexedRef index resolution.

Covers code paths test_decision.py doesn't:
- _build_decision_prompt 9-section structure
- _step_duration_hint's 4 time branches
- _extract_json_object with markdown fences + nested objects
- _expected_outcome's 5 action_type branches
- decide()/_llm_select failure → None (no decision, Rule 1 fallback tier-1)
- person_index / destination_index resolution
"""

from __future__ import annotations

import json
from typing import List

import pytest

from agent.decision import (
    _ACTION_SPACE,
    ActionCandidate,
    DecisionEngine,
    DecisionStatus,
)
from core.interfaces.llm import extract_json, extract_json_object
from agent.memory_types import Memory, MemoryStream
from agent.need import NeedEvaluation, NeedState, NeedType
from agent.relation import PerceivedRelation
from agent.motivation import ExternalDriveType, ExternalGoal
from agent.perception import InternalContext, PerceptionPacket
from agent.personality import (
    EmotionState,
    EmotionType,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from core.interfaces.action import ActionType
from core.interfaces.message import Message
from core.interfaces.perception import (
    PerceivedIdentity,
    PerceivedPresence,
    Broadcast, BroadcastType, LocationView, ReachableLocation, SpatialPerception, VisibleEntity,
)
from core.interfaces.urgency import Urgency


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _decision_prompt(engine, *args, **kwargs) -> str:
    """Join the (system, user) prefix-cache split back into one string for assertions."""
    system, user, _facts = engine._build_decision_prompt(*args, **kwargs)  # noqa: SLF001
    return system + "\n" + user


def _make_personality(
    *, name: str = "李世民", agent_id: str = "agent-shimin",
    hard_constraints: list[str] | None = None,
    short_term_goals: list[str] | None = None,
    long_term_goals: list[str] | None = None,
) -> PersonalityLayer:
    soul = SoulLayer(
        name=name, agent_id=agent_id, role="秦王",
        background="唐朝开国功臣",
        core_traits=("果决", "胆识过人"),
        core_values=("建功立业",),
        self_image="我是开疆拓土的功臣",
        life_goal="登基为帝",
        hard_constraints=tuple(hard_constraints or ()),
    )
    state = StateLayer(
        agent_id=agent_id, step=1,
        emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
        current_location="qinwang_fu",
        short_term_goals=list(short_term_goals or []),
        long_term_goals=list(long_term_goals or []),
    )
    return PersonalityLayer(soul=soul, state=state)


def _make_need_eval(
    dominant: NeedType | None = NeedType.SAFETY,
    *, prompt_context: str = "你最迫切的需求:维持安全",
    short_term_goals: list[str] | None = None,
    external_goals: list[ExternalGoal] | None = None,
) -> NeedEvaluation:
    active = [NeedState(type=dominant, label="安全", intensity=0.8, weight=0.5)] if dominant else []
    return NeedEvaluation(
        dominant_need=dominant,
        scores={dominant: 0.8} if dominant else {},
        active_needs=active,
        short_term_goals=list(short_term_goals or []),
        long_term_goals=[],
        prompt_context=prompt_context,
        external_goals=list(external_goals or []),
    )


def _mem(content: str, stream: MemoryStream, mid: str = "m1") -> Memory:
    return Memory(
        id=mid, agent_id="agent-shimin", stream=stream,
        raw_content=content, stored_content=content,
        importance=0.5, created_step=1, metadata={},
    )


def _identity(value) -> PerceivedIdentity:
    """A roster entry can be a PerceivedIdentity or just a name string; missing means an
    unrecognized stranger."""
    if isinstance(value, PerceivedIdentity):
        return value
    return PerceivedIdentity(name=value) if value else PerceivedIdentity()


def _make_packet(
    *, agent_id: str = "agent-shimin",
    visible_agent_ids: list[str] | None = None,
    visible_agents: dict[str, str] | None = None,
    reachable_location_ids: list[str] | None = None,
    location_views: dict[str, LocationView] | None = None,
    inbox: list[Message] | None = None,
    broadcasts: list[Broadcast] | None = None,
    need_eval: NeedEvaluation | None = None,
    factual_memories: list[Memory] | None = None,
    experiential_memories: list[Memory] | None = None,
    insights: list[Memory] | None = None,
    period_summaries: list[Memory] | None = None,
    relevant_relations: list | None = None,
) -> PerceptionPacket:
    spatial = SpatialPerception(
        location_id="qinwang_fu",
        location_view=LocationView(name="秦王府", description=""),
        world_time_label="辰时",
        current_step=1,
        visible_agents={
            aid: PerceivedPresence(_identity((visible_agents or {}).get(aid)))
            for aid in list(visible_agent_ids or [])
        },
        reachable_locations=[
            ReachableLocation(
                location_id=lid,
                view=(location_views or {}).get(lid, LocationView(name=lid)),
                travel_seconds=3600,
            )
            for lid in (reachable_location_ids or [])
        ],
    )
    need_evaluation = need_eval or _make_need_eval()
    internal_context = InternalContext(
        emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
        dominant_need=need_evaluation.dominant_need,
        active_needs=list(need_evaluation.active_needs),
        short_term_goals=list(need_evaluation.short_term_goals),
        long_term_goals=[],
        factual_memories=list(factual_memories or []),
        experiential_memories=list(experiential_memories or []),
        relevant_relations=list(relevant_relations or []),
        need_evaluation=need_evaluation,
        insights=list(insights or []),
        period_summaries=list(period_summaries or []),
    )
    return PerceptionPacket(
        agent_id=agent_id, step=1,
        spatial=spatial,
        inbox=list(inbox or []),
        broadcasts=list(broadcasts or []),
        internal_context=internal_context,
    )


@pytest.fixture
def main_engine(container):
    return DecisionEngine(container.llm_router)


@pytest.fixture
def bg_engine(container):
    return DecisionEngine(container.llm_router)


# ---------------------------------------------------------------------------
# _build_decision_prompt 9-section structure
# ---------------------------------------------------------------------------


def test_main_prompt_contains_role_section(main_engine) -> None:
    personality = _make_personality(name="李世民")
    packet = _make_packet()
    prompt = _decision_prompt(main_engine, personality, packet, candidates=[])
    assert "【我是谁】" in prompt
    assert "李世民" in prompt
    assert "核心性格：" in prompt  # the persona block has no person; the 【我是谁】 heading carries it


def test_main_prompt_contains_motivation_section(main_engine) -> None:
    need_eval = _make_need_eval(prompt_context="你最迫切的需求:维持安全防御")
    packet = _make_packet(need_eval=need_eval)
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "【我要往哪儿】" in prompt
    assert "我最迫切的需求与目标" in prompt
    assert "维持安全防御" in prompt


def test_main_prompt_external_goals_labeled_by_urgency(main_engine) -> None:
    """external_goals fall into three urgency bands: 【紧急】≥0.75, 【一般】≥0.45, 【留意】<0.45."""
    egs = [
        ExternalGoal(text="紧急任务", source_id="world", urgency=Urgency.HIGH,
                     drive_type=ExternalDriveType.THREAT),
        ExternalGoal(text="重要任务", source_id="world", urgency=Urgency.NORMAL,
                     drive_type=ExternalDriveType.OBLIGATION),
        ExternalGoal(text="留意任务", source_id="world", urgency=Urgency.LOW,
                     drive_type=ExternalDriveType.EVENT),
    ]
    need_eval = _make_need_eval(external_goals=egs)
    packet = _make_packet(need_eval=need_eval)
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "外部驱动" in prompt
    assert "【紧急】 紧急任务" in prompt
    assert "【一般】 重要任务" in prompt
    assert "【留意】 留意任务" in prompt


def test_main_prompt_external_goals_section_omitted_when_empty(main_engine) -> None:
    packet = _make_packet(need_eval=_make_need_eval(external_goals=[]))
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "外部驱动" not in prompt


def test_main_prompt_contains_relation_section(main_engine) -> None:
    # Relations are rendered from relevant_relations (structured) via describe_relations(), not a
    # pre-rendered string.
    rel = PerceivedRelation(trust=0.85, affection=0.4, target_agent_id="agent-weichi",
                            target_agent_name="尉迟恭", labels=["心腹"])
    packet = _make_packet(relevant_relations=[rel])
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "【供我参考的】" in prompt
    assert "尉迟恭" in prompt and "心腹" in prompt
    assert "0.85" in prompt


def test_main_prompt_contains_factual_and_experiential_memories(main_engine) -> None:
    fac = [_mem("府内增设宿卫", MemoryStream.FACTUAL, "f1")]
    exp = [_mem("我感到压力倍增", MemoryStream.EXPERIENTIAL, "e1")]
    packet = _make_packet(factual_memories=fac, experiential_memories=exp)
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "我近来的经历（客观发生" in prompt
    assert "按发生先后排列" in prompt  # ordering rule is given to the LLM (MEMORY_ORDER_HINT)
    assert "府内增设宿卫" in prompt
    assert "我近来的经历（我当前的主观理解与感受" in prompt
    assert "我感到压力倍增" in prompt


def test_main_prompt_insights_section_when_present(main_engine) -> None:
    insights = [_mem("东宫已不可信", MemoryStream.FACTUAL, "i1")]
    packet = _make_packet(insights=insights)
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "我已经形成的判断" in prompt
    assert "东宫已不可信" in prompt


def test_main_prompt_visible_agents_use_indexed_format(main_engine) -> None:
    """The '此处有' section must list visible agents numbered #N, not as 'name(id)'."""
    packet = _make_packet(
        visible_agent_ids=["agent-weichi", "agent-cheng"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭"), "agent-cheng": PerceivedIdentity(name="程知节")},
    )
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "#1 尉迟恭" in prompt
    assert "#2 程知节" in prompt
    # Opaque ids stay out of the prompt (they invite hallucination)
    assert "agent-weichi" not in prompt
    assert "agent-cheng" not in prompt


def test_main_prompt_visible_agents_empty_says_无人(main_engine) -> None:
    packet = _make_packet(visible_agent_ids=[])
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "我能触及的人：无" in prompt


def test_main_prompt_visible_entities_empty_says_无(main_engine) -> None:
    """When any of the three lists is empty, the prompt must say so explicitly. The item list is
    the one most easily omitted silently.

    If the prompt omits the item list while the schema asks for an index from "我看得见的", the
    model fills physical_entity_index=1 against a list never printed, physical_without_target
    rejects it, and the beat is wasted, often several steps in a row.
    """
    packet = _make_packet()          # no visible_entities
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "我够得着的东西：无" in prompt
    # All three lists behave the same: any empty one is reported as empty
    assert "我能触及的人：无" in prompt
    assert "我可前往：无处" in prompt


def test_main_prompt_reachable_locations_use_indexed_format(main_engine) -> None:
    """The '可前往' section is numbered #N and the LLM should output destination_index. Locations
    use narrative names and don't leak ids."""
    packet = _make_packet(
        reachable_location_ids=["xuanwu_gate", "donggong"],
        location_views={
            "xuanwu_gate": LocationView(name="玄武门", description=""),
            "donggong": LocationView(name="东宫", description=""),
        },
    )
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "#1 玄武门" in prompt
    assert "#2 东宫" in prompt
    assert "xuanwu_gate" not in prompt and "donggong" not in prompt


def test_main_prompt_hard_constraints_section_when_present(main_engine) -> None:
    personality = _make_personality(hard_constraints=["不可弑亲", "不可叛国"])
    packet = _make_packet()
    prompt = _decision_prompt(main_engine, personality, packet, candidates=[])
    assert "我绝不逾越的底线" in prompt
    assert "不可弑亲" in prompt
    assert "不可叛国" in prompt


def test_main_prompt_hard_constraints_section_omitted_when_empty(main_engine) -> None:
    personality = _make_personality(hard_constraints=[])
    packet = _make_packet()
    prompt = _decision_prompt(main_engine, personality, packet, candidates=[])
    assert "我绝不逾越的底线" not in prompt


def test_main_prompt_uses_post_perception_emotion_not_stale_state(main_engine) -> None:
    """The decision mood must come from internal_context.emotion (after perception), not
    personality.state.emotion (left over from the last step). The two differ here: state=JOY
    (would leak if to_prompt_context rendered it), internal=ANGER (post-perception, should
    render)."""
    personality = _make_personality()
    personality.state.emotion = EmotionState(primary=EmotionType.JOY, intensity=0.9, valence=0.8)
    packet = _make_packet()
    packet.internal_context.emotion = EmotionState(primary=EmotionType.ANGER, intensity=0.9, valence=-0.8)
    prompt = _decision_prompt(main_engine, personality, packet, candidates=[])
    assert packet.internal_context.emotion.summary() in prompt   # post-perception emotion is rendered
    assert personality.state.emotion.summary() not in prompt      # stale state emotion doesn't leak (include_emotion=False)


def test_main_prompt_is_first_person_in_character(main_engine) -> None:
    """The main-character decision prompt is first-person role-play (sections + negative
    constraints + an explicit task), not a functional "请选择…"."""
    prompt = _decision_prompt(main_engine, _make_personality(), _make_packet(), candidates=[])
    assert "【我要往哪儿】" in prompt
    assert "【我要做的】" in prompt          # explicit task statement (Task block)
    assert "【我被约束的】" in prompt
    assert "请选择最符合" not in prompt


def test_main_prompt_injects_ambient_channel(main_engine) -> None:
    """The decision prompt must include ambient_events, the same perception channel used by
    perception and motivation."""
    from types import SimpleNamespace
    packet = _make_packet()
    packet.spatial.ambient_events = [SimpleNamespace(content="远处火光冲天")]
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "远处火光冲天" in prompt      # ambient (location-level public event)


def test_main_prompt_action_space_listed_with_indices(main_engine) -> None:
    candidates = [
        ActionCandidate(ActionType.TALK, "对话"),
        ActionCandidate(ActionType.WORK, "工作"),
    ]
    prompt = _decision_prompt(main_engine, _make_personality(), _make_packet(), candidates)
    assert "[0] TALK" in prompt or "0." in prompt or "TALK" in prompt
    assert "[1] WORK" in prompt or "1." in prompt or "WORK" in prompt


def test_main_prompt_json_schema_uses_index_fields(main_engine) -> None:
    """The JSON schema must ask for person_index / destination_index."""
    prompt = _decision_prompt(main_engine, _make_personality(), _make_packet(), candidates=[])
    assert '"person_indices"' in prompt
    assert '"destination_index"' in prompt
    # Id-valued field names must not appear in the schema (selection is by index)
    assert '"target_agent_id"' not in prompt
    assert '"location_id"' not in prompt


def test_main_prompt_inbox_messages_rendered(main_engine) -> None:
    from types import SimpleNamespace
    inbox = [SimpleNamespace(
        sender_id="agent-yuan", sender_name="李渊", sender_is_agent=True,
        content="即刻觐见", urgency=Urgency.HIGH, metadata={},
    )]
    packet = _make_packet(inbox=inbox)
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    # A message from a real agent goes under "有人捎话给我" with the sender's name.
    assert "有人捎话给我：" in prompt
    assert "李渊" in prompt
    assert "即刻觐见" in prompt


def test_main_prompt_narrator_message_labels_unknown_source(main_engine) -> None:
    """The narrator (metadata.narrative) is treated like a message of unknown origin: it goes under
    "有人捎话给我" headed "不知来源". System concepts (the narrative engine) never enter an
    in-character prompt."""
    from types import SimpleNamespace
    inbox = [SimpleNamespace(
        sender_id="narrator", sender_name="不知来源", sender_is_agent=True,
        content="李世民心头突然涌起一阵莫名的不安", urgency=Urgency.HIGH,
        metadata={"narrative": True},
    )]
    packet = _make_packet(inbox=inbox)
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "有人捎话给我：" in prompt
    assert "不知来源：李世民心头突然涌起一阵莫名的不安" in prompt
    assert "叙事引擎" not in prompt and "narrator" not in prompt


def test_main_prompt_broadcasts_rendered(main_engine) -> None:
    bc = Broadcast(
        content="宫廷政变迹象", source="system",
        broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1, severity="high",
    )
    packet = _make_packet(broadcasts=[bc])
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "宫廷政变迹象" in prompt


# ---------------------------------------------------------------------------
# _step_duration_hint's 4 time branches
# ---------------------------------------------------------------------------


def test_step_duration_hint_seconds(container) -> None:
    engine = DecisionEngine(container.llm_router, seconds_per_step=30)
    assert engine._step_duration_hint() == "约30秒"


def test_step_duration_hint_minutes(container) -> None:
    engine = DecisionEngine(container.llm_router, seconds_per_step=300)
    assert engine._step_duration_hint() == "约5分钟"


def test_step_duration_hint_hours(container) -> None:
    engine = DecisionEngine(container.llm_router, seconds_per_step=7200)
    assert engine._step_duration_hint() == "约2小时"


def test_step_duration_hint_days(container) -> None:
    engine = DecisionEngine(container.llm_router, seconds_per_step=172800)
    assert engine._step_duration_hint() == "约2天"


# ---------------------------------------------------------------------------
# coerce_bool: a boolean the LLM writes as a string must not be silently read backwards
# ---------------------------------------------------------------------------


def test_coerce_bool_stringified_false_is_false() -> None:
    """Plain bool("false") is True, so an honestly judged failure would silently become a success.

    If a judge schema describes success as a string with an explanation, the model copies the
    quotes and outputs "success": "false". Every failed ruling then flips to success with no
    signal downstream.
    """
    from core.interfaces.llm import coerce_bool
    assert bool("false") is True                       # this is exactly why plain bool can't be used
    for token in ("false", "False", " FALSE ", "no", "0", "否", "假", "null", "none"):
        assert coerce_bool(token, True) is False, token
    # Empty string = the model gave nothing, same as missing → use the call site's default, not
    # false
    assert coerce_bool("", True) is True
    assert coerce_bool("   ", False) is False


def test_coerce_bool_passes_through_real_booleans_and_defaults() -> None:
    from core.interfaces.llm import coerce_bool
    assert coerce_bool(True, False) is True
    assert coerce_bool(False, True) is False
    assert coerce_bool("true", False) is True
    assert coerce_bool("是", False) is True
    assert coerce_bool(1, False) is True
    assert coerce_bool(0, True) is False
    assert coerce_bool(None, True) is True             # missing → each call site's own conservative default
    assert coerce_bool(None, False) is False
    assert coerce_bool({"unexpected": 1}, False) is False


# ---------------------------------------------------------------------------
# _extract_json_object: markdown fence + nested + edge cases
# ---------------------------------------------------------------------------


def test_extract_json_object_plain() -> None:
    obj = extract_json_object('{"a": 1, "b": "x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_json_object_markdown_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    obj = extract_json_object(raw)
    assert obj == {"a": 1}


def test_extract_json_object_nested() -> None:
    raw = '{"outer": {"inner": [1, 2, 3]}}'
    obj = extract_json_object(raw)
    assert obj == {"outer": {"inner": [1, 2, 3]}}


def test_extract_json_object_returns_none_on_invalid() -> None:
    assert extract_json_object("not json at all") is None


def test_extract_json_object_returns_none_on_empty() -> None:
    assert extract_json_object("") is None


def test_braces_inside_strings_do_not_end_the_object() -> None:
    raw = 'note {"a": "say \\"}\\" ok"} trailing }'
    assert extract_json_object(raw) == {"a": 'say "}" ok'}
    assert extract_json(raw) == {"a": 'say "}" ok'}


def test_extract_json_object_skips_a_span_that_does_not_parse() -> None:
    assert extract_json_object('{not json} then {"a": 1}') == {"a": 1}


# ---------------------------------------------------------------------------
# _expected_outcome's 5 action_type branches
# ---------------------------------------------------------------------------


def test_expected_outcome_talk(container) -> None:
    engine = DecisionEngine(container.llm_router)
    out = engine._expected_outcome(ActionType.TALK, _make_need_eval())
    assert "交谈" in out


def test_expected_outcome_rest(container) -> None:
    engine = DecisionEngine(container.llm_router)
    out = engine._expected_outcome(ActionType.REST, _make_need_eval())
    assert "恢复" in out or "平静" in out


def test_expected_outcome_physical(container) -> None:
    engine = DecisionEngine(container.llm_router)
    out = engine._expected_outcome(ActionType.PHYSICAL, _make_need_eval())
    assert "物理" in out or "影响" in out


def test_expected_outcome_covert(container) -> None:
    """COVERT's expected outcome needs both parts: acting unnoticed, and learning something not
    known before.

    "Achieve the goal without drawing attention" alone describes covert action as quietly getting
    something done, but it gets nothing done. It doesn't change the world, only the actor's model
    of it. Without the "what was learned" part, the gap would measure whether it was done quietly
    instead of what was gained.
    """
    engine = DecisionEngine(container.llm_router)
    out = engine._expected_outcome(ActionType.COVERT, _make_need_eval())
    assert any(w in out for w in ("不被察觉", "不引人注意", "暗中")), "缺「不被察觉」这一维"
    assert any(w in out for w in ("探到", "得知", "知道")), "缺「探到什么」这一维"


def test_expected_outcome_work_default_uses_dominant_need(container) -> None:
    engine = DecisionEngine(container.llm_router)
    need_eval = _make_need_eval(dominant=NeedType.ESTEEM)
    out = engine._expected_outcome(ActionType.WORK, need_eval)
    assert "推进" in out


def test_expected_outcome_work_no_dominant_need(container) -> None:
    engine = DecisionEngine(container.llm_router)
    need_eval = _make_need_eval(dominant=None)
    out = engine._expected_outcome(ActionType.WORK, need_eval)
    assert "稳定" in out


# ---------------------------------------------------------------------------
# Three-valued decision result (DecisionStatus, Rule 1 fallback tier-1):
# - LLM failure / unparseable output → FAILED (no fallback WORK action is invented)
# - act:false → NO_ACTION (a deliberate choice not to act: a successful decision that just
#   does nothing this beat)
# Neither FAILED nor NO_ACTION produces an action or changes state; they differ only in status
# and logging (code layer).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_returns_failed_on_llm_failure(container) -> None:
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet()

    async def _raising(*args, **kwargs):
        raise RuntimeError("simulated outage")

    engine._llm_router.complete_with_retry = _raising  # type: ignore[method-assign]
    result = await engine.decide(personality=_make_personality(), packet=packet)
    assert result.status is DecisionStatus.FAILED
    assert result.action is None


@pytest.mark.asyncio
async def test_decide_returns_failed_on_unparseable_output(container) -> None:
    from core.interfaces.llm import LLMScene

    engine = DecisionEngine(container.llm_router)
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = "not json at all"
    result = await engine.decide(personality=_make_personality(), packet=_make_packet())
    assert result.status is DecisionStatus.FAILED
    assert result.action is None


@pytest.mark.parametrize("act_value", [False, "false", "False", " false "])
@pytest.mark.asyncio
async def test_decide_returns_no_action_when_act_false(container, act_value) -> None:
    """Explicit act=false (a JSON boolean, or a string the LLM occasionally emits) → NO_ACTION, a
    deliberate non-action rather than a failure. No action is invented."""
    from core.interfaces.llm import LLMScene

    engine = DecisionEngine(container.llm_router)
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = json.dumps(
        {"inner_monologue": "此刻没有该出手的事，我按兵不动。", "act": act_value},
        ensure_ascii=False,
    )
    result = await engine.decide(personality=_make_personality(), packet=_make_packet())
    assert result.status is DecisionStatus.NO_ACTION
    assert result.action is None


# ---------------------------------------------------------------------------
# Semantic retry: an HTTP-200 JSON reply missing fields shouldn't burn the beat.
# complete_with_retry only retries transport failures, so the decision path's "retry once before
# falling back" covers the semantic level too. The retry resends the same request (no prompt
# changes, no filling fields in for the LLM); only when both attempts are unusable does it return
# FAILED (tier-1 no-op).
# ---------------------------------------------------------------------------


def _stub_responses(engine, payloads: List[str]) -> List[int]:
    """Make the decision LLM return ``payloads`` in order. Returns a 1-elem call counter."""
    from core.interfaces.llm import LLMResponse

    calls = [0]

    async def _sequenced(*args, **kwargs):
        content = payloads[min(calls[0], len(payloads) - 1)]
        calls[0] += 1
        return LLMResponse(content=content, input_tokens=1, output_tokens=1, model="stub")

    engine._llm_router.complete_with_retry = _sequenced  # type: ignore[method-assign]
    return calls


# A production-shaped payload: the LLM picks REST and leaves out action_description. REST has
# no target to name, and a prompt that describes action_description as applying to the named
# target leads the model to drop it too.
_OBJECTLESS_REST_MISSING_DESCRIPTION = json.dumps(
    {
        "inner_monologue": "连番奔波心神俱疲，须先寻处休整恢复体力。",
        "act": True,
        "selected_index": 6,
        "expected_outcome": "恢复体力与心神",
        "estimated_steps": 3,
    },
    ensure_ascii=False,
)

_OBJECTLESS_REST_COMPLETE = json.dumps(
    {
        "inner_monologue": "连番奔波心神俱疲，须先寻处休整恢复体力。",
        "act": True,
        "selected_index": 6,
        "action_description": "寻一僻静处暂歇，闭目养神以缓解连日奔波的疲惫",
        "expected_outcome": "恢复体力与心神",
        "estimated_steps": 3,
    },
    ensure_ascii=False,
)


@pytest.mark.asyncio
async def test_decide_retries_once_and_recovers_the_beat(container) -> None:
    """First attempt incomplete (missing action_description) → one retry returns complete output
    → the beat survives."""
    engine = DecisionEngine(container.llm_router)
    calls = _stub_responses(
        engine, [_OBJECTLESS_REST_MISSING_DESCRIPTION, _OBJECTLESS_REST_COMPLETE]
    )

    result = await engine.decide(personality=_make_personality(), packet=_make_packet())

    assert calls[0] == 2, "残缺输出必须触发一次重投"
    assert result.status is DecisionStatus.ACTED
    assert result.action is not None
    assert result.action.action_type is ActionType.REST
    assert result.action.estimated_steps == 3
    assert "僻静" in result.action.action_description


@pytest.mark.asyncio
async def test_decide_failed_when_retry_also_unusable(container) -> None:
    """Both attempts incomplete → FAILED. No filling fields in for the LLM, no invented action
    (Rule 1 fallback tier-1)."""
    engine = DecisionEngine(container.llm_router)
    calls = _stub_responses(engine, [_OBJECTLESS_REST_MISSING_DESCRIPTION])

    result = await engine.decide(personality=_make_personality(), packet=_make_packet())

    assert calls[0] == 2, "只重投一次,不无限重试"
    assert result.status is DecisionStatus.FAILED
    assert result.action is None


@pytest.mark.asyncio
async def test_decide_does_not_retry_a_usable_output(container) -> None:
    """A usable first attempt shouldn't cost a second call; the retry only exists to rescue
    incomplete output."""
    engine = DecisionEngine(container.llm_router)
    calls = _stub_responses(engine, [_OBJECTLESS_REST_COMPLETE])

    result = await engine.decide(personality=_make_personality(), packet=_make_packet())

    assert calls[0] == 1
    assert result.status is DecisionStatus.ACTED


class _CallBox:
    """Stand-in for the active LLMCallTrace (same three fields the router fills)."""

    def __init__(self) -> None:
        self.parse_ok = None
        self.adopted = None
        self.reject_reason = ""


@pytest.mark.asyncio
async def test_discarded_decision_records_its_reason_on_the_trace(container) -> None:
    """When a beat is burned, the trace must record why. Otherwise the inspector just shows a
    normal-looking call."""
    from core.context import set_active_call

    engine = DecisionEngine(container.llm_router)
    _stub_responses(engine, [_OBJECTLESS_REST_MISSING_DESCRIPTION])
    box = _CallBox()
    set_active_call(box)
    try:
        result = await engine.decide(personality=_make_personality(), packet=_make_packet())
    finally:
        set_active_call(None)

    assert result.status is DecisionStatus.FAILED
    assert box.adopted is False
    assert box.reject_reason == "missing_action_description"


@pytest.mark.asyncio
async def test_deliberate_no_action_is_adopted_not_discarded(container) -> None:
    """act:false is accepted output: the LLM said not to act this beat and the engine did that.

    Deliberate restraint is the opposite of discarded output. Recording it as discarded would pad
    the discard rate with healthy behavior and bury the real burned beats in noise.
    """
    from core.context import set_active_call

    engine = DecisionEngine(container.llm_router)
    calls = _stub_responses(
        engine, [json.dumps({"inner_monologue": "此刻无事可做。", "act": False}, ensure_ascii=False)]
    )
    box = _CallBox()
    set_active_call(box)
    try:
        result = await engine.decide(personality=_make_personality(), packet=_make_packet())
    finally:
        set_active_call(None)

    assert result.status is DecisionStatus.NO_ACTION
    assert box.adopted is True
    assert box.reject_reason == ""
    assert calls[0] == 1, "act:false 是成功的决策,不该触发重投"


@pytest.mark.asyncio
async def test_acted_decision_is_adopted(container) -> None:
    from core.context import set_active_call

    engine = DecisionEngine(container.llm_router)
    _stub_responses(engine, [_OBJECTLESS_REST_COMPLETE])
    box = _CallBox()
    set_active_call(box)
    try:
        result = await engine.decide(personality=_make_personality(), packet=_make_packet())
    finally:
        set_active_call(None)

    assert result.status is DecisionStatus.ACTED
    assert box.adopted is True


def test_prompt_demands_description_even_without_an_object(container) -> None:
    """Prompt contract: action_description is required for target-less actions too.

    The parser always requires it. If the prompt describes it as something that only applies to
    the target just named, target-less actions (resting, working alone) learn to omit it and the
    beat is lost. Both ends of the contract must agree.
    """
    engine = DecisionEngine(container.llm_router)
    system, _, _facts = engine._build_decision_prompt(_make_personality(), _make_packet(), _ACTION_SPACE)
    assert "必填" in system and "不可省" in system
    assert "只关乎我自己" in system


# ---------------------------------------------------------------------------
# IndexedRef index resolution: _parse_llm_selection and the schema
# ---------------------------------------------------------------------------


def test_parse_person_index_resolves_to_agent_id(container) -> None:
    """LLM outputs person_index=2 → resolves to visible_agent_ids[1]."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-weichi", "agent-cheng"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭"), "agent-cheng": PerceivedIdentity(name="程知节")},
    )
    candidates = [ActionCandidate(ActionType.TALK, "对话")]
    raw = json.dumps({
        "selected_index": 0,
        "person_indices": [2],  # → agent-cheng
        "action_description": "与程知节商议",
        "inner_monologue": "",
        "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_agents == ["agent-cheng"]


def test_parse_destination_index_resolves_to_location_id(container) -> None:
    """LLM outputs destination_index=1 → resolves to reachable_location_ids[0]."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(reachable_location_ids=["xuanwu_gate", "donggong"])
    candidates = [ActionCandidate(ActionType.MOVE, "移动")]
    raw = json.dumps({
        "selected_index": 0,
        "destination_index": 1,
        "action_description": "前往玄武门",
        "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_place == "xuanwu_gate"


def test_parse_move_without_resolved_destination_rejected(container) -> None:
    """The LLM picked MOVE without a valid destination_index (missing, out of range, or
    hallucinated) → an incomplete decision; return None and skip this step. Same as a
    hallucinated TALK/SEND target: a MOVE with no destination must never reach the executor and
    produce a fake journey."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(reachable_location_ids=["xuanwu_gate", "donggong"])
    candidates = [ActionCandidate(ActionType.MOVE, "移动")]
    # destination_index omitted
    omitted = json.dumps({
        "selected_index": 0, "action_description": "动身离开", "estimated_steps": 1,
    })
    assert engine._parse_llm_selection(omitted, candidates, packet) is None
    # destination_index out of range (only 2 neighboring locations, picks 9)
    out_of_range = json.dumps({
        "selected_index": 0, "destination_index": 9,
        "action_description": "动身离开", "estimated_steps": 1,
    })
    assert engine._parse_llm_selection(out_of_range, candidates, packet) is None


def _absent_sender_msg() -> Message:
    return Message(
        id="m1", world_id="w", sender_id="agent-weichi", content="甲士已备，请速回示",
        recipients=["agent-shimin"], location_scope=None, deliver_step=1, created_step=1,
        sender_name="尉迟恭",
    )


def test_parse_person_index_resolves_absent_recipient(container) -> None:
    """SEND_MESSAGE person_index → the messageable list (including absent people who sent
    messages) → binds the absent Yuchi Gong.

    SEND_MESSAGE must be able to bind a target who isn't present.
    """
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(visible_agent_ids=[], inbox=[_absent_sender_msg()])
    candidates = [ActionCandidate(ActionType.SEND_MESSAGE, "传讯")]
    raw = json.dumps({
        "selected_index": 0, "person_indices": [1],
        "action_description": "回信尉迟恭：按计划即刻行事", "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_agents == ["agent-weichi"]  # binds even when not present


def test_parse_send_message_stray_item_id_not_bound(container) -> None:
    """An announcement that also fills item_id by mistake → item_id is not treated as a
    recipient."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(visible_agent_ids=[])
    candidates = [ActionCandidate(ActionType.SEND_MESSAGE, "传讯")]
    raw = json.dumps({
        "selected_index": 0, "item_id": "禁军统领", "message_announce": True,
        "action_description": "通告全军戒备", "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acts_on == []


def test_parse_physical_entity_index_resolves_visible_entity(container) -> None:
    """PHYSICAL physical_entity_index → a visible item (by index), binding entity_id plus the
    environment's own entity_type.

    The category never comes from the LLM. It's an execution routing key (the same-location check
    and effects on people both branch on it), and a wrong guess would turn a blow on a person into
    one on a thing, so only the environment's value counts."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet()
    packet.spatial.visible_entities.append(
        VisibleEntity(entity_id="ent-token", name="调兵符节", entity_type="item", is_takeable=True))
    candidates = [ActionCandidate(ActionType.PHYSICAL, "物理干预")]
    raw = json.dumps({
        "selected_index": 0, "physical_entity_index": 1,
        "action_description": "我一把抓起调兵符节藏入怀中", "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_ids == ["ent-token"]
    assert sel.target.acted_on_kind == "item"


def test_parse_physical_person_index_on_person(container) -> None:
    """PHYSICAL physical_person_index → someone present from "我能触及的人"; kind is set to agent
    by code."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-jc", "agent-yj"],
        visible_agents={"agent-jc": PerceivedIdentity(name="李建成"), "agent-yj": PerceivedIdentity(name="李元吉")},
    )
    candidates = [ActionCandidate(ActionType.PHYSICAL, "物理干预")]
    raw = json.dumps({
        "selected_index": 0, "physical_person_index": 1,
        "action_description": "我夺刀制住李建成", "inner_monologue": "先发制人", "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_ids == ["agent-jc"]   # physical_person_index=1 → list[0]
    assert sel.target.acted_on_kind == "agent"


def test_parse_physical_channels_never_cross_namespaces(container) -> None:
    """An out-of-range person index never falls back to the item list. Grabbing something from
    another namespace is worse than coming back empty.

    The channel is decided by which field was filled; an index that can't resolve there doesn't
    resolve, and the guard rejects the beat. A "category" field choosing the list would look an
    out-of-range person index up in the item table, turning "pin down Li Jiancheng" into "grab the
    troop tally" while the narration still names the person."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-jc"], visible_agents={"agent-jc": PerceivedIdentity(name="李建成")})
    packet.spatial.visible_entities.append(
        VisibleEntity(entity_id="ent-token", name="调兵符节", entity_type="item"))
    candidates = [ActionCandidate(ActionType.PHYSICAL, "物理干预")]
    raw = json.dumps({
        "selected_index": 0, "physical_person_index": 9,   # out-of-range person index
        "action_description": "我按住李建成", "estimated_steps": 1,
    })
    assert engine._parse_llm_selection(raw, candidates, packet) is None


def test_parse_physical_without_target_rejected(container) -> None:
    """PHYSICAL with no bindable target → return None and skip the step, unconditionally, not
    dependent on a category the LLM reports.

    Gated on a reported "person", an omitted category would bypass the guard: "grabbing his
    sleeve" would execute with an empty target, the judge would invent the target from the
    description, and the renderer would draw a punch into thin air. All three incomplete forms
    must be blocked."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-jc"],
        visible_agents={"agent-jc": PerceivedIdentity(name="李建成")},
    )
    candidates = [ActionCandidate(ActionType.PHYSICAL, "物理干预")]
    # ① No target field filled (the most common production form: the model only put the person in
    # person_indices)
    unbound = json.dumps({
        "selected_index": 0, "person_indices": [1],
        "action_description": "扑向李建成，死死拽住他衣袖", "estimated_steps": 1,
    })
    assert engine._parse_llm_selection(unbound, candidates, packet) is None
    # ② A self-filled item_id goes through the abstract-target channel, and its kind is always
    # object. People can only be reached by index, so a self-filled value can never produce an
    # agent target and can't be used to bypass the presence filter.
    #
    # Known limit: if the self-filled name is actually a person ("禁军统领"), it's still
    # adjudicated as an untracked thing. Catching that would require guessing from the string
    # whether a name looks like a person, which is the keyword matching Rule 7 forbids.
    freeform = json.dumps({
        "selected_index": 0, "item_id": "宫门",
        "action_description": "我用肩撞开紧闭的宫门", "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(freeform, candidates, packet)
    assert sel is not None and sel.target.acted_on_kind == "object"
    # ③ The person index points to someone on the list who isn't present
    not_present = json.dumps({
        "selected_index": 0, "physical_person_index": 2,
        "action_description": "我冲上去制住尉迟恭", "estimated_steps": 1,
    })
    assert engine._parse_llm_selection(not_present, candidates, packet) is None


def test_main_prompt_message_roster_and_entity_index(main_engine) -> None:
    """Decision prompt: includes the "我能触及的人" list (tagged present/absent), entities as #N,
    and the index fields in the schema."""
    packet = _make_packet(visible_agent_ids=[], inbox=[_absent_sender_msg()])
    packet.spatial.visible_entities.append(
        VisibleEntity(entity_id="e1", name="符节", entity_type="item"))
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "我能触及的人" in prompt
    assert "尉迟恭（不在场）" in prompt
    assert "#1 符节" in prompt
    assert '"person_indices"' in prompt
    # PHYSICAL's person and item channels are separate fields: the index namespace is in the field
    # name, not decided by another field.
    assert '"physical_person_index"' in prompt
    assert '"physical_entity_index"' in prompt
    assert '"physical_recipient_index"' in prompt


def test_recipient_index_precedes_action_description_in_schema(main_engine) -> None:
    """The recipient index is a precondition for generation: without choosing the recipient first,
    "hand the dagger to Yuchi Gong" can't be written.

    If the words come first and the index after, the model just picks someone from the list to
    fit. So by §5b dependency order it must come before the content it constrains. Parsing reads by
    key and ignores order, so this ordering is free.
    """
    prompt = _decision_prompt(main_engine, _make_personality(), _make_packet(), candidates=[])
    assert prompt.index('"physical_recipient_index"') < prompt.index('"action_description"')


def test_item_list_marks_who_holds_what(main_engine) -> None:
    """One list, one set of indices. Holding is an attribute of an entry, not a second namespace.

    A separate list would mean a second index space, and index spaces bleeding into each other is
    how a physical target gets bound to the wrong thing. Names come only from the perception
    packet (the holder is always at my location), and no id appears in the prompt.
    """
    packet = _make_packet(
        visible_agent_ids=["agent-weichi"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭")},
    )
    packet.spatial.visible_entities.extend([
        VisibleEntity(entity_id="e1", name="灯笼", entity_type="item"),
        VisibleEntity(entity_id="e2", name="短刀", entity_type="item", holder_id="agent-shimin",
                      is_takeable=True),
        VisibleEntity(entity_id="e3", name="虎符", entity_type="item", holder_id="agent-weichi",
                      is_takeable=True),
        VisibleEntity(entity_id="e4", name="密函", entity_type="item", holder_id="agent-ghost"),
    ])
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])

    assert "我够得着的东西：" in prompt
    assert "#1 灯笼" in prompt and "持有" not in prompt.split("#1 灯笼")[1].split("\n")[0]
    assert "#2 短刀（由我持有；可交出）" in prompt     # in my hands: what I can do is hand it over
    assert "#3 虎符（由尉迟恭持有）" in prompt       # in someone else's hands: not marked "可取"
    assert "#4 密函（由某人持有" in prompt      # an unrecognized holder falls back to a descriptive referent
    for leaked in ("agent-shimin", "agent-weichi", "agent-ghost", "e1", "e2"):
        assert leaked not in prompt


def test_parse_binds_a_recipient_alongside_the_entity(container) -> None:
    """A hand-over is "item + optional recipient". What's acted on is still the item; the
    recipient only modifies it and doesn't affect routing.

    The recipient goes into reaches, not acts_on: a hand-over is done to the item, and the person
    in acts_on would report a gift as an attack and drop the item from the aim.
    """
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-weichi"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭")},
    )
    packet.spatial.visible_entities.append(
        VisibleEntity(entity_id="e1", name="虎符", entity_type="item", holder_id="agent-shimin"))
    candidates = [ActionCandidate(ActionType.PHYSICAL, "动手")]

    sel = engine._parse_llm_selection(json.dumps({
        "selected_index": 0, "physical_entity_index": 1, "physical_recipient_index": 1,
        "action_description": "我把虎符双手奉给尉迟恭", "estimated_steps": 1,
    }), candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_ids == ["e1"] and sel.target.acted_on_kind == "item"
    assert sel.target.reached_agents == ["agent-weichi"]
    assert sel.target.acted_on_agents == []

    # Not handed to anyone, just put down → empty recipient, still a valid entity binding.
    dropped = engine._parse_llm_selection(json.dumps({
        "selected_index": 0, "physical_entity_index": 1,
        "action_description": "我把虎符搁在案上", "estimated_steps": 1,
    }), candidates, packet)
    assert dropped is not None and dropped.target.reached_agents == []


def test_a_bad_recipient_burns_the_beat(container) -> None:
    """Filled but resolves to no one, or nothing to hand over: both are rejected, not silently
    dropped.

    action_description is already written as a hand-over. Executing it as a blow on a person or
    with no target would produce output whose narration contradicts its structure, which is the
    kind of incomplete decision physical_without_target exists to block.
    """
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-weichi"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭")},
    )
    packet.spatial.visible_entities.append(
        VisibleEntity(entity_id="e1", name="虎符", entity_type="item", holder_id="agent-shimin"))
    candidates = [ActionCandidate(ActionType.PHYSICAL, "动手")]

    # ① Recipient index out of range
    assert engine._parse_llm_selection(json.dumps({
        "selected_index": 0, "physical_entity_index": 1, "physical_recipient_index": 9,
        "action_description": "我把虎符交给他", "estimated_steps": 1,
    }), candidates, packet) is None
    # ② Recipient filled but the person channel was used, so there's nothing to hand over
    assert engine._parse_llm_selection(json.dumps({
        "selected_index": 0, "physical_person_index": 1, "physical_recipient_index": 1,
        "action_description": "我把虎符交给他", "estimated_steps": 1,
    }), candidates, packet) is None


def test_message_roster_excludes_deceased() -> None:
    """The dead aren't on the SEND_MESSAGE candidate list (no messaging corpses); the living are."""
    from agent.decision import _message_roster
    from agent.relation import PerceivedRelation
    packet = _make_packet(
        visible_agent_ids=[],
        relevant_relations=[
            PerceivedRelation(trust=0.6, affection=0.4, target_agent_id="alive", target_agent_name="活着"),
            PerceivedRelation(trust=0.3, affection=-0.1, target_agent_id="dead", target_agent_name="已死", deceased=True),
        ],
    )
    roster_ids = [aid for aid, _who in _message_roster(packet)]
    assert "alive" in roster_ids
    assert "dead" not in roster_ids


def test_parse_out_of_range_person_index_falls_through(container) -> None:
    """person_index out of range → filtered by IndexedRef, continuing down the fallback path
    (empty target)."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-weichi"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭")},
    )
    candidates = [ActionCandidate(ActionType.TALK, "对话")]
    raw = json.dumps({
        "selected_index": 0,
        "person_indices": [99],  # out of range
        "action_description": "与某人对话",
        "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    # Out of range falls back to an empty target. TALK with no target then fails the spatial check
    # → None, or returns an empty target; the key assertion is that no wrong agent id gets in
    if sel is not None:
        assert sel.target.acted_on_agents != ["agent-weichi"]  # no wrong binding


def test_parse_non_integer_person_index_falls_through(container) -> None:
    """Non-integer person_index → filtered by IndexedRef."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-weichi"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭")},
    )
    candidates = [ActionCandidate(ActionType.TALK, "对话")]
    raw = json.dumps({
        "selected_index": 0,
        "person_indices": ["abc"],  # string → filtered by IndexedRef
        "action_description": "对话",
        "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    # The index is invalid at parse time → no agent bound; the later spatial check (TALK needs a
    # visible target) rejects the whole selection
    assert sel is None or sel.target.acted_on_agents != ["agent-weichi"]


def test_parse_index_takes_precedence_over_string_field(container) -> None:
    """When both person_index and target_agent_id are given, the index wins."""
    engine = DecisionEngine(container.llm_router)
    packet = _make_packet(
        visible_agent_ids=["agent-weichi", "agent-cheng"],
        visible_agents={"agent-weichi": PerceivedIdentity(name="尉迟恭"), "agent-cheng": PerceivedIdentity(name="程知节")},
    )
    candidates = [ActionCandidate(ActionType.TALK, "对话")]
    raw = json.dumps({
        "selected_index": 0,
        "person_indices": [1],        # → Yuchi Gong
        "target_agent_id": "agent-cheng",  # distractor field
        "action_description": "对话",
        "estimated_steps": 1,
    })
    sel = engine._parse_llm_selection(raw, candidates, packet)
    assert sel is not None
    assert sel.target.acted_on_agents == ["agent-weichi"]


# ---------------------------------------------------------------------------
# Unified decision prompt: background agents (is_main_character=False) go through the same
# _build_decision_prompt with the same index format
# ---------------------------------------------------------------------------


def test_background_prompt_visible_agents_use_indexed_format(bg_engine) -> None:
    packet = _make_packet(
        visible_agent_ids=["agent-1", "agent-2"],
        visible_agents={"agent-1": PerceivedIdentity(name="甲"), "agent-2": PerceivedIdentity(name="乙")},
    )
    prompt = _decision_prompt(bg_engine, _make_personality(), packet, candidates=[])
    assert "#1 甲" in prompt
    assert "#2 乙" in prompt
    # Opaque ids must not appear
    assert "agent-1" not in prompt


def test_background_prompt_reachable_locations_listed_when_present(bg_engine) -> None:
    """The background prompt must list reachable_locations too, not only the main one. Locations
    use narrative names, not ids."""
    packet = _make_packet(
        reachable_location_ids=["xuanwu_gate", "donggong"],
        location_views={
            "xuanwu_gate": LocationView(name="玄武门", description=""),
            "donggong": LocationView(name="东宫", description=""),
        },
    )
    prompt = _decision_prompt(bg_engine, _make_personality(), packet, candidates=[])
    assert "我可前往" in prompt
    assert "#1 玄武门（脚程约" in prompt
    assert "#2 东宫（脚程约" in prompt


def test_background_prompt_schema_uses_index_fields(bg_engine) -> None:
    prompt = _decision_prompt(bg_engine, _make_personality(), _make_packet(), candidates=[])
    assert '"person_indices"' in prompt
    assert '"destination_index"' in prompt
    assert '"target_agent_id"' not in prompt
    assert '"location_id"' not in prompt


def test_roster_line_carries_gender_beside_presence(main_engine) -> None:
    """The reachable list carries gender. Whoever I speak to or act on this beat, the words need
    forms of address and pronouns, and without gender the model has to guess.

    Gender shares the parentheses with "在场"; two separate groups would render
    "长孙无垢（女）（在场）".
    """
    packet = _make_packet(
        visible_agent_ids=["agent-wude"],
        visible_agents={"agent-wude": PerceivedIdentity(name="长孙无垢", gender="女")},
        relevant_relations=[PerceivedRelation(
            trust=0.6, affection=0.4, target_agent_id="agent-jc",
            target_agent_name="李建成", target_agent_gender="男",
        )],
    )
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "#1 长孙无垢（女，在场）" in prompt
    assert "#2 李建成（男，不在场）" in prompt


def test_roster_omits_gender_when_unknown(main_engine) -> None:
    """Unknown gender means name only. Never make one up for the LLM (same rule as a missing name
    falling back to "某人")."""
    packet = _make_packet(
        visible_agent_ids=["agent-x"],
        visible_agents={"agent-x": PerceivedIdentity(name="无名氏")},
    )
    prompt = _decision_prompt(main_engine, _make_personality(), packet, candidates=[])
    assert "#1 无名氏（在场）" in prompt
