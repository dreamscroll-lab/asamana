"""Unit tests for the agent decision subsystem.

* ``_ACTION_SPACE`` is a static module-level catalog.  Tests assert that
  every expected action type is present and that descriptions are populated.

* ``decide()`` defers to the LLM for the actual choice and narrative.
  Tests for it assert the resulting ``AgentAction`` is well-formed and
  that an LLM failure yields FAILED with no action.

* ``_parse_llm_selection`` binds the ``ActionTarget`` from the LLM's
  response (indices such as ``person_indices`` / ``destination_index``) and
  rejects structurally illegal target bindings (e.g. TALK to a non-visible
  agent, MOVE to a non-reachable location).
"""

from __future__ import annotations

import json

import pytest

from agent.decision import (
    ActionCandidate,
    DecisionEngine,
    DecisionStatus,
    _insights_with_sources,
)
from agent.need import (
    NeedEvaluation,
    NeedState,
    NeedType,
)
from agent.perception import InternalContext, PerceptionPacket
from agent.personality import activity_status_for
from agent.personality import (
    EmotionState,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from core.interfaces.action import ActionType, AgentAction
from core.interfaces.message import Message
from core.interfaces.perception import (
    Broadcast,
    LocationView,
    NpcIdentity,
    PerceivedNpc,
    PerceivedPresence,
    ReachableLocation,
    SpatialPerception,
    VisibleEntity,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_personality(
    *,
    name: str = "Test Agent",
    agent_id: str = "agent-1",
    hard_constraints: list[str] | None = None,
) -> PersonalityLayer:
    soul = SoulLayer(
        name=name,
        agent_id=agent_id,
        role="advisor",
        core_traits=("pragmatic",),
        core_values=("loyalty",),
        hard_constraints=tuple(hard_constraints or ()),
    )
    state = StateLayer(
        agent_id=agent_id,
        step=1,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        current_location="palace",
    )
    return PersonalityLayer(soul=soul, state=state)


def _make_need_evaluation(
    dominant: NeedType | None = NeedType.SOCIAL,
    *,
    short_term_goals: list[str] | None = None,
) -> NeedEvaluation:
    """Build a minimal NeedEvaluation suitable for tests."""
    active = [
        NeedState(type=dominant, label="dominant", intensity=0.8, weight=1.0)
    ] if dominant is not None else []
    return NeedEvaluation(
        dominant_need=dominant,
        scores={dominant: 0.8} if dominant is not None else {},
        active_needs=active,
        short_term_goals=list(short_term_goals or []),
        long_term_goals=[],
        prompt_context="",
    )


def _make_packet(
    *,
    agent_id: str = "agent-1",
    step: int = 1,
    location_id: str = "palace",
    visible_agent_ids: list[str] | None = None,
    reachable_location_ids: list[str] | None = None,
    visible_entities: list | None = None,
    visible_npcs: dict | None = None,
    inbox: list[Message] | None = None,
    broadcasts: list[Broadcast] | None = None,
    dominant_need: NeedType | None = NeedType.SOCIAL,
    short_term_goals: list[str] | None = None,
) -> PerceptionPacket:
    spatial = SpatialPerception(
        location_id=location_id,
        location_view=LocationView(name=location_id, description=""),
        world_time_label="辰时",
        current_step=step,
        visible_agents={aid: PerceivedPresence() for aid in visible_agent_ids or []},
        reachable_locations=[
            ReachableLocation(location_id=lid, view=LocationView(name=lid), travel_seconds=3600)
            for lid in (reachable_location_ids or [])
        ],
        visible_entities=list(visible_entities or []),
        visible_npcs=dict(visible_npcs or {}),
    )
    need_eval = _make_need_evaluation(dominant_need, short_term_goals=short_term_goals)
    internal_context = InternalContext(
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        dominant_need=dominant_need,
        active_needs=list(need_eval.active_needs),
        short_term_goals=list(short_term_goals or []),
        long_term_goals=[],
        factual_memories=[],
        experiential_memories=[],
        relevant_relations=[],
        need_evaluation=need_eval,
    )
    return PerceptionPacket(
        agent_id=agent_id,
        step=step,
        spatial=spatial,
        inbox=inbox or [],
        broadcasts=broadcasts or [],
        internal_context=internal_context,
    )


def _make_engine(container: object) -> DecisionEngine:
    return DecisionEngine(container.llm_router)  # type: ignore[attr-defined]


def _decision_prompt(engine: DecisionEngine, *args, **kwargs) -> str:
    """Join the (system, user) prefix-cache split back into one string for assertions."""
    system, user, _facts = engine._build_decision_prompt(*args, **kwargs)  # noqa: SLF001
    return system + "\n" + user


# ---------------------------------------------------------------------------
# AgentAction post-init coercion
# ---------------------------------------------------------------------------


def test_agent_action_post_init_populates_content_from_description() -> None:
    action = AgentAction(action_type=ActionType.REST, action_description="Take a rest.")

    assert action.content == "Take a rest."


def test_agent_action_post_init_populates_description_from_content() -> None:
    action = AgentAction(action_type=ActionType.REST, content="Take a break.")

    assert action.action_description == "Take a break."


def test_agent_action_action_type_string_coerced_to_enum() -> None:
    action = AgentAction(action_type="rest", action_description="Take a break.")

    assert action.action_type == ActionType.REST


# ---------------------------------------------------------------------------
# Activity status mapping
# ---------------------------------------------------------------------------


def test_activity_status_mapping() -> None:
    assert activity_status_for(ActionType.TALK).value == "talking"
    assert activity_status_for(ActionType.REST).value == "resting"
    assert activity_status_for(ActionType.MOVE).value == "moving"
    assert activity_status_for(ActionType.COVERT).value == "covert"
    assert activity_status_for(ActionType.WORK).value == "working"
    assert activity_status_for(ActionType.PHYSICAL).value == "working"


# ---------------------------------------------------------------------------
# Static action space catalog
# ---------------------------------------------------------------------------


def test_action_space_contains_all_expected_types() -> None:
    from agent.decision import _ACTION_SPACE

    types = {c.action_type for c in _ACTION_SPACE}
    assert ActionType.TALK in types
    assert ActionType.SEND_MESSAGE in types
    assert ActionType.MOVE in types
    assert ActionType.WORK in types
    assert ActionType.PHYSICAL in types
    assert ActionType.COVERT in types
    assert ActionType.REST in types


def test_action_space_descriptions_are_nonempty() -> None:
    from agent.decision import _ACTION_SPACE

    for candidate in _ACTION_SPACE:
        assert candidate.description, f"{candidate.action_type} has empty description"


def test_action_space_rest_description_mentions_multiple_steps() -> None:
    from agent.decision import _ACTION_SPACE

    rest = next(c for c in _ACTION_SPACE if c.action_type == ActionType.REST)
    assert "多个步骤" in rest.description


def test_an_index_outside_the_roster_is_recorded_not_just_dropped() -> None:
    """``IndexedRef.resolve`` silently drops out-of-range indices. Nothing is left in the world,
    so unless the engine records it, it can never be found.

    It records the slot, the index given, and the list length, not the list contents, so review
    doesn't need to know any slot names. This rarely fires in real worlds, so this test is the
    only guard.
    """
    from agent.decision import _Slots

    slots = _Slots(payload={"person_indices": [1, 9], "destination_index": 7},
                   packet=object(), visible_ids=[], roster_ids=["a1", "a2"],
                   reachable_ids=["loc1"], entity_ids=[], entity_types={}, npc_ids=[],
                   own_item_ids=[])
    assert slots.many("person_indices", slots.roster_ids) == ["a1"]     # #9 is outside the list
    assert slots.one("destination_index", slots.reachable_ids) is None  # #7, same
    assert len(slots.dropped) == 2
    assert "person_indices" in slots.dropped[0] and "9" in slots.dropped[0]
    assert "destination_index" in slots.dropped[1] and "7" in slots.dropped[1]

    # Record nothing when everything is in range; a silent channel shouldn't make noise.
    clean = _Slots(payload={"person_indices": [1]}, packet=object(), visible_ids=[],
                   roster_ids=["a1"], reachable_ids=[], entity_ids=[], entity_types={},
                   npc_ids=[], own_item_ids=[])
    assert clean.many("person_indices", clean.roster_ids) == ["a1"] and clean.dropped == []


def test_every_declared_fact_is_one_the_prompt_actually_showed() -> None:
    """Every declared fact, minus its channel prefix, must appear verbatim in this call's own
    prompt.

    This is the foundation of the ``given_facts`` contract: review treats the list as everything
    the model saw, so an entry the prompt never gave makes review cite something unseen and invert
    the verdict. The opposite half (a channel the list omits) can't be tested here; only building
    both in one function prevents it.

    Declare bare values only: a prompt fragment (like ``expected_part``) drags the prompt's section
    headings into the facts.
    """
    from agent.memory_types import Memory, MemoryKind, MemoryStream

    def _mem(i: int, stream: MemoryStream, kind: MemoryKind, text: str) -> Memory:
        return Memory(id=f"m{i}", stream=stream, agent_id="a1", raw_content=text,
                      stored_content=text, importance=0.8, created_step=i, kind=kind)

    packet = _make_packet()
    aw = packet.internal_context
    aw.factual_memories = [_mem(1, MemoryStream.FACTUAL, MemoryKind.EVENT, "我在东宫见了太子")]
    aw.experiential_memories = [_mem(2, MemoryStream.EXPERIENTIAL, MemoryKind.EVENT, "我觉得他在试探我")]
    aw.insights = [_mem(3, MemoryStream.EXPERIENTIAL, MemoryKind.INSIGHT, "他等的不是准信")]
    aw.period_summaries = [_mem(4, MemoryStream.EXPERIENTIAL, MemoryKind.SUMMARY, "那几日人心浮动")]
    aw.recent_foiled_attempts = ["遣人递信，未得回音（已试过2次）"]

    from agent.decision import _ACTION_SPACE

    _system, user, facts = DecisionEngine(None)._build_decision_prompt(  # noqa: SLF001
        _make_personality(), packet, _ACTION_SPACE)
    assert facts, "用例没造出任何事实，这条守卫会空转"
    for fact in facts:
        channel, _, body = fact.partition("：")
        assert body, f"申报少了通道前缀：{fact!r}"
        assert body in user, f"申报了 prompt 没给的东西：[{channel}] {body!r}"
        assert "\n【" not in fact, f"prompt 片段被整块塞进了通道：{fact!r}"


def test_insights_with_sources_renders_insight_and_evidence() -> None:
    """Decision prompt contract: an insight gets its own line in the prompt, with its sources.

    This is where Reflection output is shown to the LLM as "已形成的判断（依据：…）". The sources
    make a belief traceable, so new experience can confirm or shake it.
    """
    from agent.memory_types import Memory, MemoryStream

    source_a = Memory(
        id="src-a",
        stream=MemoryStream.EXPERIENTIAL,
        stored_content="X turned away when I greeted him in the courtyard.",
    )
    source_b = Memory(
        id="src-b",
        stream=MemoryStream.EXPERIENTIAL,
        stored_content="X left the banquet without saying goodbye to me.",
    )
    insight = Memory(
        id="ins-1",
        stream=MemoryStream.EXPERIENTIAL,
        stored_content="X is avoiding me — trust between us may already be gone.",
        kind="insight",
        source_ids=["src-a", "src-b"],
    )

    rendered = _insights_with_sources(
        [insight],
        {"ins-1": [source_a, source_b]},
        now_step=10, seconds_per_step=3600,
    )

    assert "trust between us may already be gone" in rendered
    # The sources must follow immediately
    assert "依据：" in rendered
    assert "turned away" in rendered
    assert "left the banquet" in rendered


def test_insights_with_sources_empty_returns_placeholder() -> None:
    assert _insights_with_sources([], {}, now_step=10, seconds_per_step=3600) == "无"


def test_insights_with_sources_missing_sources_omits_evidence() -> None:
    """With no sources (edge cases like compression removed them and fixup didn't match), the
    insight is still shown."""
    from agent.memory_types import Memory, MemoryStream

    insight = Memory(
        id="ins-2",
        stream=MemoryStream.EXPERIENTIAL,
        stored_content="A free-floating belief.",
        kind="insight",
        source_ids=[],
    )
    rendered = _insights_with_sources([insight], {}, now_step=10, seconds_per_step=3600)
    assert "A free-floating belief" in rendered
    assert "依据" not in rendered


# ---------------------------------------------------------------------------
# decide() — happy path and fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decide_returns_valid_agent_action(container: object) -> None:
    """A usable LLM selection → a well-formed AgentAction with a known type."""
    from core.interfaces.llm import LLMScene

    engine = _make_engine(container)
    personality = _make_personality()
    packet = _make_packet(visible_agent_ids=["agent-2"])
    # selected_index=3 → WORK (no target binding required).
    container.llm_router.get(LLMScene.AGENT_DECISION_MAIN).fixed_response = json.dumps(
        {"selected_index": 3, "action_description": "整理书卷", "inner_monologue": "", "estimated_steps": 1},
        ensure_ascii=False,
    )

    result = await engine.decide(personality=personality, packet=packet)

    assert result.status is DecisionStatus.ACTED
    action = result.action
    assert isinstance(action, AgentAction)
    assert isinstance(action.action_type, ActionType)
    assert action.action_description
    assert action.agent_id == personality.soul.agent_id
    assert action.step == packet.spatial.current_step


@pytest.mark.asyncio
async def test_decide_returns_failed_when_llm_fails(
    container: object,
    monkeypatch,
) -> None:
    """When the decision LLM raises (even after retry), decide() returns FAILED — no
    fabricated action. The runtime skips the agent's step (Rule 1 fallback tier-1)."""
    engine = _make_engine(container)
    personality = _make_personality()
    packet = _make_packet()

    async def raising(*args, **kwargs):
        raise RuntimeError("simulated LLM outage")

    # Patch the router seam: any LLM call raises, exercising the engine's
    # internal try/except → FAILED (no decision).
    monkeypatch.setattr(engine._llm_router, "complete", raising)  # noqa: SLF001
    monkeypatch.setattr(engine._llm_router, "complete_with_retry", raising)  # noqa: SLF001

    result = await engine.decide(personality=personality, packet=packet)

    assert result.status is DecisionStatus.FAILED
    assert result.action is None


def test_parse_llm_selection_picks_indexed_candidate(container: object) -> None:
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.WORK, description="推进事务"),
        ActionCandidate(
            action_type=ActionType.MOVE,
            description="移动到相邻地点，可选目的地：garden",
        ),
        ActionCandidate(
            action_type=ActionType.REST,
            description="休息恢复精力",
        ),
    ]
    packet = _make_packet(reachable_location_ids=["garden"])
    payload = json.dumps(
        {
            "selected_index": 1,
            "destination_index": 1,
            "action_description": "我前往庭院散心。",
            "inner_monologue": "心绪杂乱，需要透透气。",
            "estimated_steps": 2,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001

    assert selection is not None
    assert selection.action_type == ActionType.MOVE
    assert selection.target.acted_on_place == "garden"
    assert selection.action_description == "我前往庭院散心。"
    assert selection.inner_monologue == "心绪杂乱，需要透透气。"
    assert selection.estimated_steps == 2


def test_send_message_falls_back_to_the_spoken_words(container: object) -> None:
    """If SEND_MESSAGE omits action_description, fall back to message_content. The two fields sit
    right next to each other, and a field readable from its neighbor isn't worth losing the whole
    decision over."""
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.SEND_MESSAGE, description="传讯")]
    payload = json.dumps(
        {
            "selected_index": 0,
            "message_content": "速回东宫，不得延误。",
            "inner_monologue": "得让他立刻回来。",
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates)  # noqa: SLF001

    assert selection is not None
    assert selection.action_description == "速回东宫，不得延误。"
    assert selection.message_content == "速回东宫，不得延误。"


def test_only_send_message_may_borrow_the_spoken_words(container: object) -> None:
    """Other actions have no such neighbor to read. A missing description means it really didn't
    say what to do, so the whole decision is still dropped."""
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.WORK, description="推进事务")]
    payload = json.dumps(
        {"selected_index": 0, "message_content": "речь", "inner_monologue": "x"},
        ensure_ascii=False,
    )

    assert engine._parse_llm_selection(payload, candidates) is None  # noqa: SLF001


def test_a_slot_binds_only_for_the_type_that_declares_it(container: object) -> None:
    """``destination_index`` belongs to MOVE and ``item_id`` to PHYSICAL. Other actions that fill
    them don't get them bound.

    Otherwise you get "resting at A while the record is bound to B and the description says going
    back to B", and the action can't move him anyway.
    """
    engine = _make_engine(container)
    packet = _make_packet(reachable_location_ids=["garden"])
    payload = json.dumps(
        {"selected_index": 0, "destination_index": 1, "item_id": "门",
         "action_description": "我就地歇一会儿。", "inner_monologue": "累了。"},
        ensure_ascii=False,
    )

    for kind in (ActionType.REST, ActionType.WORK):
        sel = engine._parse_llm_selection(  # noqa: SLF001
            payload, [ActionCandidate(action_type=kind, description="x")], packet)
        assert sel is not None and sel.target.acts_on == [], kind

    move = engine._parse_llm_selection(  # noqa: SLF001
        payload, [ActionCandidate(action_type=ActionType.MOVE, description="移动")], packet)
    assert move is not None and move.target.acted_on_place == "garden"

    physical = engine._parse_llm_selection(  # noqa: SLF001
        payload, [ActionCandidate(action_type=ActionType.PHYSICAL, description="动手")], packet)
    assert physical is not None and physical.target.acted_on_ids == ["门"]


def test_parse_llm_selection_rejects_out_of_range_index(container: object) -> None:
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.WORK, description="推进事务")]
    payload = json.dumps(
        {
            "selected_index": 5,
            "action_description": "ignored",
            "inner_monologue": "ignored",
        }
    )

    assert engine._parse_llm_selection(payload, candidates) is None  # noqa: SLF001


def test_parse_llm_selection_rejects_non_json(container: object) -> None:
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.WORK, description="推进事务")]

    assert engine._parse_llm_selection("not json at all", candidates) is None  # noqa: SLF001


def test_parse_llm_selection_binds_person_via_person_index(container: object) -> None:
    """TALK / SEND_MESSAGE bind people via person_indices (index → people list). COVERT doesn't
    bind; see below."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(
            action_type=ActionType.TALK,
            description="与可见角色面对面交谈",
        )
    ]
    packet = _make_packet(visible_agent_ids=["agent-2"])  # roster = visible = [agent-2]
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1],
            "action_description": "我去和他说话。",
            "inner_monologue": "需要了解他的意图。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001

    assert selection is not None
    assert selection.action_type == ActionType.TALK
    assert selection.target.single_acted_on_agent == "agent-2"
    # It's done to a person, so it's neither a location nor an item; acts_on holds one kind at a
    # time.
    assert selection.target.acted_on_kind == "agent"


def test_send_message_parses_message_content_and_routes_to_content(container: object) -> None:
    """SEND_MESSAGE: message_content (the exact words to the recipient) is parsed and lands on
    AgentAction.content via _build_action; action_description (the actor's own account) stays
    separate."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.SEND_MESSAGE, description="向特定角色发送异步消息")
    ]
    packet = _make_packet(visible_agent_ids=["agent-2"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1],
            "action_description": "我向父皇传讯，揭发太子谋反",
            "message_content": "父皇！太子欲谋逆，请明察！",
            "inner_monologue": "必须先发制人。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001
    assert selection is not None
    assert selection.message_content == "父皇！太子欲谋逆，请明察！"

    from agent.personality import PersonalityLayer, SoulLayer, StateLayer

    soul = SoulLayer(name="李世民", role="秦王", agent_id="agent-1")
    personality = PersonalityLayer(soul=soul, state=StateLayer(agent_id="agent-1", step=0))
    action = engine._build_action(selection, personality, packet)  # noqa: SLF001
    # content carries the exact words; action_description stays the actor's own account
    assert action.content == "父皇！太子欲谋逆，请明察！"
    assert action.action_description == "我向父皇传讯，揭发太子谋反"
    # Async action: decide pins expected_outcome to "the message is delivered" (a fixed value),
    # regardless of what the LLM expects
    from agent.decision import _SEND_MESSAGE_EXPECTED_OUTCOME
    assert action.expected_outcome == _SEND_MESSAGE_EXPECTED_OUTCOME


def test_talk_folds_message_content_into_action_description(container: object) -> None:
    """TALK: a message_content the model filled anyway (the substance of the action) is merged into
    action_description rather than dropped.

    Otherwise the substance (names, places, deadlines) disappears with the field, dialogue
    generation gets only an empty topic, and the conversation comes out wrong.
    """
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.TALK, description="与在场角色交谈")]
    packet = _make_packet(visible_agent_ids=["agent-2"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1],
            "action_description": "我压低声音，对张浩说那条匿名短信的事。",
            "message_content": "我收到匿名短信，说李华在南郊待建区，明天三点前不去就找不到了。",
            "inner_monologue": "得让他帮我判断真假。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001
    assert selection is not None
    assert selection.action_type == ActionType.TALK
    assert "我压低声音，对张浩说那条匿名短信的事。" in selection.action_description
    assert "南郊待建区" in selection.action_description

    from agent.personality import PersonalityLayer, SoulLayer, StateLayer

    soul = SoulLayer(name="林语晴", role="学生", agent_id="agent-1")
    personality = PersonalityLayer(soul=soul, state=StateLayer(agent_id="agent-1", step=0))
    action = engine._build_action(selection, personality, packet)  # noqa: SLF001
    # TALK has no delivery channel: content still mirrors action_description (with the content
    # merged in) and carries no separate message
    assert action.action_description == selection.action_description
    assert action.content == selection.action_description


def test_talk_without_message_content_keeps_description_unchanged(container: object) -> None:
    """When TALK leaves message_content empty, action_description is unchanged (nothing
    appended)."""
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.TALK, description="与在场角色交谈")]
    packet = _make_packet(visible_agent_ids=["agent-2"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1],
            "action_description": "我去和他说话。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )
    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001
    assert selection is not None
    assert selection.action_description == "我去和他说话。"


def test_covert_does_not_bind_agent_target(container: object) -> None:
    """COVERT doesn't bind a structured agent target (the executor reads only action_description
    and the scene and never uses target). Even if the LLM fills person_indices, COVERT's target
    stays empty."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.COVERT, description="暗中行动或观察")
    ]
    packet = _make_packet(visible_agent_ids=["agent-2"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1],  # filled, but COVERT still doesn't bind
            "action_description": "暗中观察李建成的动向。",
            "inner_monologue": "得摸清他的部署。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )
    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001
    assert selection is not None
    assert selection.action_type == ActionType.COVERT
    assert selection.target.acted_on_agents == []


def test_send_message_binds_multiple_recipients(container: object) -> None:
    """SEND_MESSAGE supports multiple recipients: person_indices=[1,2] → one action aimed at two
    people (broadcast to a group)."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.SEND_MESSAGE, description="向特定角色发送异步消息")
    ]
    packet = _make_packet(visible_agent_ids=["agent-2", "agent-3"])  # roster = [agent-2, agent-3]
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1, 2],
            "action_description": "我向两位重臣同时传讯",
            "message_content": "速来议事。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )
    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001
    assert selection is not None
    assert set(selection.target.acted_on_agents) == {"agent-2", "agent-3"}


def test_talk_with_multiple_indices_binds_only_first(container: object) -> None:
    """TALK is strictly one-on-one: even with several person_indices, only the first is bound
    (only SEND supports multiple recipients)."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.TALK, description="与可见角色面对面交谈")
    ]
    packet = _make_packet(visible_agent_ids=["agent-2", "agent-3"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1, 2],
            "action_description": "我去交谈。",
            "inner_monologue": "先谈一个。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )
    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001
    assert selection is not None
    assert len(selection.target.acted_on_agents) == 1  # TALK takes only the first
    assert selection.target.acted_on_agents[0] in {"agent-2", "agent-3"}


def test_send_message_directed_with_unresolvable_index_is_rejected(container: object) -> None:
    """A targeted SEND (person_index given) with an empty list or an out-of-range index is a
    hallucinated recipient. Parsing rejects it (returns None) and never quietly turns it into
    "broadcast to an empty room". Covers the model picking SEND with person_index=1 even when the
    roster is empty."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.SEND_MESSAGE, description="向特定角色发送异步消息")
    ]
    packet = _make_packet(visible_agent_ids=[])  # nobody present → empty roster
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1],  # points into an empty list → no recipient resolves
            "action_description": "向常何发送密信",
            "message_content": "速速控制门禁。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )
    assert engine._parse_llm_selection(payload, candidates, packet) is None  # noqa: SLF001


def test_an_announcement_must_be_declared_not_inferred_from_an_empty_roster(
    container: object,
) -> None:
    """An announcement must be declared explicitly. Undeclared and with no recipients means a
    message aimed at nobody, so the whole step is rejected.

    Both look the same structurally (empty acts_on). Treating empty as an announcement would turn
    "the person isn't on the list" into "I'll shout to everyone". Announcing with nobody present is
    still allowed; whether anyone hears it is a delivery-time question.
    """
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.SEND_MESSAGE, description="向场景内所有人广播")
    ]
    packet = _make_packet(visible_agent_ids=[])
    base = {
        "selected_index": 0,
        "action_description": "向全城宣告戒严",
        "message_content": "即刻起全城戒严。",
        "estimated_steps": 1,
    }
    declared = engine._parse_llm_selection(  # noqa: SLF001
        json.dumps({**base, "message_announce": True}, ensure_ascii=False), candidates, packet
    )
    assert declared is not None
    assert declared.action_type == ActionType.SEND_MESSAGE
    assert declared.target.acted_on_agents == []  # announcement, no targeted recipient

    assert engine._parse_llm_selection(  # noqa: SLF001
        json.dumps(base, ensure_ascii=False), candidates, packet
    ) is None


def test_parse_llm_selection_extracts_location_id(container: object) -> None:
    """LLM responses bind MOVE targets via destination_index."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(
            action_type=ActionType.MOVE,
            description="移动到相邻地点，可选目的地：garden",
        )
    ]
    packet = _make_packet(reachable_location_ids=["garden"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "destination_index": 1,
            "action_description": "前往花园。",
            "inner_monologue": "换个地方观察。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001

    assert selection is not None
    assert selection.target.acted_on_place == "garden"
    assert selection.target.acted_on_kind == "location"


def test_parse_llm_selection_extracts_item_id(container: object) -> None:
    """LLM responses bind PHYSICAL targets via item_id."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(
            action_type=ActionType.PHYSICAL,
            description="与场景中的物品互动，可选物品：sword",
        )
    ]
    payload = json.dumps(
        {
            "selected_index": 0,
            "item_id": "sword",
            "action_description": "拿起剑。",
            "inner_monologue": "需要自保。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates)  # noqa: SLF001

    assert selection is not None
    assert selection.target.acted_on_ids == ["sword"]
    assert selection.target.acted_on_agents == []
    assert selection.target.acted_on_place is None


def test_parse_llm_selection_no_target_when_not_needed(container: object) -> None:
    """For action types with no target (REST/WORK), ActionTarget is empty."""
    engine = _make_engine(container)
    candidates = [
        ActionCandidate(action_type=ActionType.REST, description="休息恢复精力")
    ]
    payload = json.dumps(
        {
            "selected_index": 0,
            "action_description": "仔细思考。",
            "inner_monologue": "需要冷静。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates)  # noqa: SLF001

    assert selection is not None
    assert selection.target.acts_on == []


# ---------------------------------------------------------------------------
# Target structural validation
# ---------------------------------------------------------------------------


def test_parse_llm_selection_rejects_talk_to_absent_person(container: object) -> None:
    """TALK whose person_index picks someone on the list who isn't present → rejected (talking face
    to face requires presence)."""
    from agent.decision import _ACTION_SPACE
    from core.interfaces.message import Message

    engine = _make_engine(container)
    # agent-99, not present, is on the reachable list because it sent a message; person_index=2
    # picks it.
    msg = Message(id="m", world_id="w", sender_id="agent-99", content="hi",
                  recipients=["agent-1"], location_scope=None, deliver_step=1, created_step=1,
                  sender_name="远人")
    packet = _make_packet(visible_agent_ids=["agent-2"], inbox=[msg])  # roster=[agent-2 (present), agent-99 (absent)]
    content = json.dumps(
        {"selected_index": 0, "person_indices": [2],
         "action_description": "找人说话。", "inner_monologue": "需要交流。", "estimated_steps": 1},
        ensure_ascii=False,
    )

    assert engine._parse_llm_selection(content, _ACTION_SPACE, packet) is None  # noqa: SLF001


def test_decision_prompt_carries_my_own_vitality_when_it_is_failing(container: object) -> None:
    """I was told about every hit when it happened, but nothing else says how far it has added up.
    The line is omitted when vitality is high."""
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    candidates = _ACTION_SPACE

    hurt = _make_personality()
    hurt.apply_vitality_damage(0.8)
    assert "我此刻的体力：将竭" in _decision_prompt(engine, hurt, packet, candidates)

    assert "我此刻的体力" not in _decision_prompt(engine, _make_personality(), packet, candidates)


def test_parse_llm_selection_rejects_talk_without_any_target(container: object) -> None:
    """TALK that names nobody → rejected.

    The prompt already says under 【此刻够不着的行动】 that TALK isn't available with nobody
    present, yet the model still picks TALK with empty person_indices. Letting it through produces
    an embedded first-person memory ("想要做:…,但是结果是:想要对话,但是不知道对话目标。") that is
    a non-event and puts the parser's diagnostic into narrative memory.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=[])          # nobody present
    content = json.dumps(
        {"selected_index": 0, "person_indices": [],
         "action_description": "我端坐御座，等他们入殿后先以家常话相问。",
         "inner_monologue": "该先想好怎么开场。", "estimated_steps": 1},
        ensure_ascii=False,
    )

    assert engine._parse_llm_selection(content, _ACTION_SPACE, packet) is None  # noqa: SLF001


def test_parse_llm_selection_rejects_talk_without_target_even_with_people_present(
    container: object,
) -> None:
    """Someone is present but nobody was named: still rejected. The guard is unconditional, like
    PHYSICAL / MOVE. TALK is one-on-one and face to face by definition; without a counterpart it
    isn't a degraded conversation, it isn't a conversation at all."""
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    content = json.dumps(
        {"selected_index": 0, "person_indices": [],
         "action_description": "我开口说话。", "inner_monologue": "想说点什么。", "estimated_steps": 1},
        ensure_ascii=False,
    )

    assert engine._parse_llm_selection(content, _ACTION_SPACE, packet) is None  # noqa: SLF001


def test_parse_llm_selection_accepts_talk_to_visible_agent(container: object) -> None:
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])  # roster=[agent-2 (present)]
    content = json.dumps(
        {"selected_index": 0, "person_indices": [1],
         "action_description": "找他说话。", "inner_monologue": "需要了解。", "estimated_steps": 1},
        ensure_ascii=False,
    )

    result = engine._parse_llm_selection(content, _ACTION_SPACE, packet)  # noqa: SLF001

    assert result is not None
    assert result.target.single_acted_on_agent == "agent-2"


def test_parse_llm_selection_accepts_move_to_reachable_location(container: object) -> None:
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(reachable_location_ids=["garden"])
    content = json.dumps(
        {"selected_index": 2, "destination_index": 1,
         "action_description": "前往花园。", "inner_monologue": "换个环境。", "estimated_steps": 1},
        ensure_ascii=False,
    )

    result = engine._parse_llm_selection(content, _ACTION_SPACE, packet)  # noqa: SLF001

    assert result is not None
    assert result.target.acted_on_place == "garden"


# ---------------------------------------------------------------------------
# Background prompt includes hard_constraints; estimated_steps capped at 10
# ---------------------------------------------------------------------------


def test_decision_prompt_surfaces_broadcasts(container: object) -> None:
    """The unified decision prompt must show world broadcasts."""
    from core.interfaces.perception import BroadcastType

    engine = _make_engine(container)
    personality = _make_personality()
    packet = _make_packet(
        broadcasts=[
            Broadcast(
                content="大雨倾盆，道路泥泞难行",
                source="system",
                broadcast_type=BroadcastType.WORLD_EVENT,
                deliver_step=1,
                severity="high",
            ),
        ],
    )
    from agent.decision import _ACTION_SPACE

    prompt = _decision_prompt(engine, personality, packet, _ACTION_SPACE)  # noqa: SLF001

    assert "世界传来" in prompt
    assert "大雨倾盆，道路泥泞难行" in prompt
    # Code-layer enums must not appear in first-person text (on Python 3.11+ an f-string renders a
    # str-Enum as "BroadcastType.WORLD_EVENT")
    assert "BroadcastType" not in prompt
    assert "world_event" not in prompt


def test_decision_prompt_anchors_time_and_place_via_situation_header(container: object) -> None:
    """The decision prompt's current time and place come only from the shared situation_header
    (first person, first line). There is no separate "我在：..." line, and location_id never
    leaks."""
    engine = _make_engine(container)
    personality = _make_personality()
    packet = _make_packet(location_id="palace")
    from agent.decision import _ACTION_SPACE

    prompt = _decision_prompt(engine, personality, packet, _ACTION_SPACE)  # noqa: SLF001

    # The header gives both place and time (first-person voice)
    assert "我此刻在palace，当前时间为辰时。" in prompt
    # No duplicate location line
    assert "我在：" not in prompt
    # The header is the first line (before 【我是谁】)
    assert prompt.index("我此刻在palace") < prompt.index("【我是谁】")


def test_decision_prompt_includes_long_term_goals_as_list(container: object) -> None:
    """Long-term goals appear in the decision prompt under 【我要往哪儿】 as a list, one per line,
    not joined into one line."""
    engine = _make_engine(container)
    personality = _make_personality()
    packet = _make_packet()
    packet.internal_context.need_evaluation.long_term_goals = ["重整朝纲", "为父复仇"]
    from agent.decision import _ACTION_SPACE

    prompt = _decision_prompt(engine, personality, packet, _ACTION_SPACE)  # noqa: SLF001

    assert "我的长期目标" in prompt
    # List form: each goal on its own bulleted line, not joined like "重整朝纲；为父复仇".
    assert "  - 重整朝纲" in prompt and "  - 为父复仇" in prompt
    assert "重整朝纲；为父复仇" not in prompt
    assert prompt.index("【我要往哪儿】") < prompt.index("我的长期目标") < prompt.index("【供我参考的】")


def test_decision_prompt_omits_long_term_line_when_empty(container: object) -> None:
    """With no long-term goals the line isn't rendered (avoids empty noise)."""
    engine = _make_engine(container)
    packet = _make_packet()  # _make_need_evaluation defaults long_term_goals=[]
    from agent.decision import _ACTION_SPACE

    prompt = _decision_prompt(engine, _make_personality(), packet, _ACTION_SPACE)  # noqa: SLF001

    assert "我的长期目标" not in prompt


def test_parse_llm_selection_preserves_large_estimated_steps(container: object) -> None:
    """LLM-returned estimated_steps is preserved verbatim — no upper cap.

    An engine-wide cap would be an arbitrary speed limiter. World-physics constraints (e.g. MOVE
    distance) live in WorldConfig; semantic constraints (PHYSICAL/SEND_MESSAGE immediacy) live in
    their executors. LLM-judged "how long do I plan to spend" is otherwise free.
    """
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.WORK, description="专注工作")]
    payload = json.dumps(
        {
            "selected_index": 0,
            "action_description": "着手处理积压事务。",
            "inner_monologue": "需要很长时间。",
            "estimated_steps": 50,
        },
        ensure_ascii=False,
    )

    result = engine._parse_llm_selection(payload, candidates)  # noqa: SLF001

    assert result is not None
    assert result.estimated_steps == 50


def test_parse_llm_selection_accepts_steps_within_bounds(container: object) -> None:
    """Reasonable estimated_steps values pass through unchanged."""
    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.REST, description="休息")]
    payload = json.dumps(
        {
            "selected_index": 0,
            "action_description": "稍作休息。",
            "inner_monologue": "需要恢复精力。",
            "estimated_steps": 3,
        },
        ensure_ascii=False,
    )

    result = engine._parse_llm_selection(payload, candidates)  # noqa: SLF001

    assert result is not None
    assert result.estimated_steps == 3


# ---------------------------------------------------------------------------
# _expected_outcome returns Chinese
# ---------------------------------------------------------------------------


def test_expected_outcome_returns_chinese(container: object) -> None:
    """_expected_outcome must return Chinese for all action types."""
    engine = _make_engine(container)
    need_eval = _make_need_evaluation(NeedType.SOCIAL)

    for action_type in ActionType:
        outcome = engine._expected_outcome(action_type, need_eval)  # noqa: SLF001
        # Basic check: no pure-ASCII sentences (English). Allow mixed content.
        assert outcome, f"{action_type} returned empty outcome"
        # Must contain at least one CJK character
        assert any("一" <= ch <= "鿿" for ch in outcome), (
            f"{action_type} outcome is not Chinese: {outcome!r}"
        )


# ---------------------------------------------------------------------------
# MOVE forcibly takes people present along (separate field move_carry_indices)
# ---------------------------------------------------------------------------

def _carry_move(indices: list[int] | None) -> str:
    payload = {"selected_index": 2, "destination_index": 1,
               "action_description": "拽着他去花园。", "inner_monologue": "带走。",
               "estimated_steps": 1}
    if indices is not None:
        payload["move_carry_indices"] = indices
    return json.dumps(payload, ensure_ascii=False)


def test_move_carries_a_present_body(container: object) -> None:
    """Carrying someone off doesn't ask whether they're willing or what condition they're in. If
    they're present, they can be taken."""
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"], reachable_location_ids=["garden"])

    result = engine._parse_llm_selection(_carry_move([1]), _ACTION_SPACE, packet)  # noqa: SLF001

    assert result is not None
    assert result.action_type == ActionType.MOVE
    assert result.target.acted_on_place == "garden"
    # Carrying takes up their turn, but it isn't done to them. A move is done to a location.
    assert result.target.claimed_agents == ["agent-2"]
    assert result.target.acted_on_agents == []


def test_carry_index_pointing_at_an_absent_body_is_ignored_not_fatal(container: object) -> None:
    """Indices that can't be carried (absent, out of range, hallucinated) are ignored one by one;
    the move itself still stands.

    This is deliberately the opposite of physical_recipient, which rejects the whole step when it
    can't resolve. The recipient of a hand-over is the substance of the action (without one, "give
    it to him" means nothing). Carrying is an extra on top of a move that is complete and valid by
    itself, and dropping the whole decision over one bad extra costs more than it saves.
    """
    from agent.decision import _ACTION_SPACE
    from core.interfaces.message import Message

    engine = _make_engine(container)
    # agent-99, not present, is on the reachable list because it sent a message, and is first.
    msg = Message(id="m", world_id="w", sender_id="agent-99", content="hi",
                  recipients=["agent-1"], location_scope=None, deliver_step=1, created_step=1,
                  sender_name="远人")
    packet = _make_packet(inbox=[msg], reachable_location_ids=["garden"])

    result = engine._parse_llm_selection(_carry_move([1, 77]), _ACTION_SPACE, packet)  # noqa: SLF001

    assert result is not None
    assert result.target.acted_on_place == "garden"
    assert result.target.acted_on_agents == []


def test_move_carry_does_not_share_the_person_indices_channel(container: object) -> None:
    """Which list an index refers to, and what relation it means, is in the field name.
    person_indices can't carry anyone off.

    With a shared field, the model would first have to correctly answer an implicit question (are
    these person_indices recipients, the person I'm talking to, or people I'm dragging along?)
    before it could fill the right index, and nothing in the structure catches a wrong answer.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"], reachable_location_ids=["garden"])
    content = json.dumps(
        {"selected_index": 2, "destination_index": 1, "person_indices": [1],
         "action_description": "去花园。", "inner_monologue": "换个环境。", "estimated_steps": 1},
        ensure_ascii=False,
    )

    result = engine._parse_llm_selection(content, _ACTION_SPACE, packet)  # noqa: SLF001

    assert result is not None
    assert result.target.acted_on_agents == []


def test_plain_move_still_carries_nobody(container: object) -> None:
    """Without move_carry_indices it's an ordinary walk. Carrying only adds; it doesn't change the
    normal path."""
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"], reachable_location_ids=["garden"])

    result = engine._parse_llm_selection(_carry_move(None), _ACTION_SPACE, packet)  # noqa: SLF001

    assert result is not None
    assert result.target.acted_on_place == "garden"
    assert result.target.acted_on_agents == []


# ---------------------------------------------------------------------------
# Characterization tests for the parse guards. Each one guards a failure the model really
# produces; a refactor must keep them as they are.
# ---------------------------------------------------------------------------


def _sel(engine, payload: dict, action_type: ActionType, packet=None, desc="做一件事"):
    body = {"action_description": desc, "inner_monologue": "…", "selected_index": 0, **payload}
    return engine._parse_llm_selection(  # noqa: SLF001
        json.dumps(body, ensure_ascii=False),
        [ActionCandidate(action_type=action_type, description="x")],
        packet,
    )


def test_envelope_guards_reject_before_anything_else(container: object) -> None:
    """No chosen candidate, no decision: non-JSON, a missing index, or an out-of-range index all
    reject the whole step."""
    engine = _make_engine(container)
    cands = [ActionCandidate(action_type=ActionType.REST, description="休息")]
    assert engine._parse_llm_selection("不是 JSON", cands) is None            # noqa: SLF001
    assert engine._parse_llm_selection('{"action_description":"x"}', cands) is None  # noqa: SLF001
    # A bool is not an index: in Python True == 1, so without this guard it would count as picking
    # item 1.
    assert engine._parse_llm_selection('{"selected_index":true,"action_description":"x"}', cands) is None  # noqa: SLF001
    assert engine._parse_llm_selection('{"selected_index":9,"action_description":"x"}', cands) is None  # noqa: SLF001


def test_talk_needs_a_present_partner(container: object) -> None:
    """TALK is one-on-one and face to face by definition. With no counterpart, or one who isn't
    present, it isn't a degraded conversation; it doesn't happen."""
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    assert _sel(engine, {}, ActionType.TALK, packet) is None                       # empty list
    assert _sel(engine, {"person_indices": [99]}, ActionType.TALK, packet) is None  # hallucinated index


def test_a_talk_may_be_overheard_by_the_extra_indices(container: object) -> None:
    """TALK talks to one person. Extra indices are listeners and go into reaches; their turns stay
    free."""
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2", "agent-3"])
    sel = _sel(engine, {"person_indices": [1, 2]}, ActionType.TALK, packet)
    assert sel is not None
    assert sel.target.acted_on_agents == ["agent-2"]
    assert [r.id for r in sel.target.claims] == ["agent-2"]
    assert [r.id for r in sel.target.reaches] == ["agent-3"]


def test_a_directed_message_to_nobody_resolvable_is_rejected(container: object) -> None:
    """Indices given but all out of range = hallucinated recipients, so the whole step is rejected.
    It does not quietly become a broadcast to an empty room."""
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    assert _sel(engine, {"person_indices": [99]}, ActionType.SEND_MESSAGE, packet) is None
    # An explicitly declared announcement needs no recipient index and is structurally valid.
    assert _sel(engine, {"message_announce": True}, ActionType.SEND_MESSAGE, packet) is not None


def test_naming_someone_outranks_the_announce_flag(container: object) -> None:
    """Both channels given means the model contradicted itself. The targeted channel wins and the
    announcement flag is ignored, without rejecting the step: the indices resolved to real people,
    so this beat really did land on someone."""
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    sel = _sel(
        engine,
        {"person_indices": [1], "message_announce": True},
        ActionType.SEND_MESSAGE,
        packet,
    )
    assert sel is not None
    assert sel.target.acted_on_agents == ["agent-2"]


def test_physical_channels_never_fall_back_across_namespaces(container: object) -> None:
    """The field that was filled decides the channel, and an out-of-range index fails only within
    that channel.

    Falling back across namespaces would turn "pin someone down" into "grab something": wrong
    target and wrong kind, while the description still names the person.
    """
    engine = _make_engine(container)
    packet = _make_packet(
        visible_agent_ids=["agent-2"],
        visible_entities=[VisibleEntity(entity_id="seal", name="印玺", entity_type="item")],
    )
    # Person index out of range → fails within its channel → no target → whole step rejected. Never
    # fall back to the item list
    assert _sel(engine, {"physical_person_index": 99}, ActionType.PHYSICAL, packet) is None
    on_person = _sel(engine, {"physical_person_index": 1}, ActionType.PHYSICAL, packet)
    assert on_person is not None and on_person.target.acted_on_agents == ["agent-2"]
    on_thing = _sel(engine, {"physical_entity_index": 1}, ActionType.PHYSICAL, packet)
    assert on_thing is not None and on_thing.target.acted_on_ids == ["seal"]


def test_a_handover_needs_both_the_thing_and_the_hand(container: object) -> None:
    """The recipient of a hand-over is the substance of the action. A missing item or an
    unresolvable recipient rejects the whole step; neither side is silently dropped."""
    engine = _make_engine(container)
    packet = _make_packet(
        visible_agent_ids=["agent-2"],
        visible_entities=[VisibleEntity(entity_id="seal", name="印玺", entity_type="item")],
    )
    # Recipient out of range → rejected
    assert _sel(engine, {"physical_entity_index": 1, "physical_recipient_index": 99},
                ActionType.PHYSICAL, packet) is None
    # Recipient filled but no entity channel (nothing to hand over) → rejected
    assert _sel(engine, {"physical_person_index": 1, "physical_recipient_index": 1},
                ActionType.PHYSICAL, packet) is None
    # Both present: the item goes into acts_on, the person into reaches (receiving doesn't cost
    # their turn)
    ok = _sel(engine, {"physical_entity_index": 1, "physical_recipient_index": 1},
              ActionType.PHYSICAL, packet)
    assert ok is not None
    assert ok.target.acted_on_ids == ["seal"] and [r.id for r in ok.target.reaches] == ["agent-2"]
    assert ok.target.claims == []


def test_a_carried_body_must_be_here_and_never_costs_the_whole_beat(container: object) -> None:
    """Carrying is an extra on top of a move: absent or out-of-range indices are ignored one by
    one, and the move itself still stands."""
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"], reachable_location_ids=["garden"])
    sel = _sel(engine, {"destination_index": 1, "move_carry_indices": [1, 99]},
               ActionType.MOVE, packet)
    assert sel is not None
    assert sel.target.acted_on_place == "garden"
    assert [r.id for r in sel.target.claims] == ["agent-2"]


def test_scalar_fields_take_the_conservative_reading(container: object) -> None:
    """Scalar fields quietly take a conservative value; they're never worth dropping the whole
    decision."""
    engine = _make_engine(container)
    sel = _sel(engine, {"estimated_steps": 0, "inner_monologue": 5, "expected_outcome": None},
               ActionType.REST)
    assert sel is not None
    assert sel.estimated_steps == 1 and sel.inner_monologue == "" and sel.expected_outcome == ""
    # Accept a scalar index too: the model writing one integer instead of an array shouldn't void
    # the whole decision.
    packet = _make_packet(visible_agent_ids=["agent-2"])
    talk = _sel(engine, {"person_indices": 1}, ActionType.TALK, packet)
    assert talk is not None and talk.target.acted_on_agents == ["agent-2"]


def test_a_talks_spoken_words_are_folded_into_its_description(container: object) -> None:
    """TALK has no delivery channel, so message_content would otherwise be dropped entirely, and
    the substance with it."""
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    sel = _sel(engine, {"person_indices": [1], "message_content": "明日辰时东门见"},
               ActionType.TALK, packet, desc="我把约定告诉他")
    assert sel is not None and "明日辰时东门见" in sel.action_description


def _runner(name: str, *, busy: bool = False, condition: str = "") -> PerceivedNpc:
    return PerceivedNpc(
        identity=NpcIdentity(name=name, gender="男", age=40), busy=busy, condition=condition,
    )


def test_a_runner_already_out_is_excluded_not_merely_labeled(container: object) -> None:
    """Someone out on an errand can't be sent on another. This has to appear under
    【此刻够不着的行动】.

    The "busy" tag on the list doesn't stop the person giving orders. With a single errand-runner
    at a location, every errand from that location fails while he's out, and each failure only
    returns "一时抽不开身", so the same order is retried unchanged next beat.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    personality = _make_personality()

    idle = _make_packet(visible_npcs={"npc_a": _runner("老宦", busy=False)})
    prompt = _decision_prompt(engine, personality, idle, _ACTION_SPACE)
    assert "不可选 ERRAND" not in prompt
    assert "老宦" in prompt

    out = _make_packet(visible_npcs={"npc_a": _runner("老宦", busy=True)})
    prompt = _decision_prompt(engine, personality, out, _ACTION_SPACE)
    assert "这一拍谁也派不出去：不可选 ERRAND" in prompt
    assert "老宦" in prompt          # he's really standing here; he just can't be sent

    empty = _make_packet()
    assert "此处没有听人吩咐做事的：不可选 ERRAND" in _decision_prompt(
        engine, personality, empty, _ACTION_SPACE
    )


def test_a_runner_who_cannot_move_is_excluded_on_the_same_footing(container: object) -> None:
    """Someone who can't move can't be sent either. ``ErrandExecutor`` rejects on both busyness and
    condition; if the menu checked only one, the orderer would keep commanding someone who is bound
    to refuse. The reason belongs on the list line; the exclusion sentence doesn't repeat it.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    personality = _make_personality()

    held = _make_packet(visible_npcs={"npc_a": _runner("老宦", condition="双手被反绑")})
    prompt = _decision_prompt(engine, personality, held, _ACTION_SPACE)
    assert "这一拍谁也派不出去：不可选 ERRAND" in prompt
    assert "老宦" in prompt and "双手被反绑" in prompt      # he's present and his condition is visible

    # One restrained and one idle → errands are still possible, so don't exclude the whole block.
    one_free = _make_packet(visible_npcs={
        "npc_a": _runner("老宦", condition="双手被反绑"),
        "npc_b": _runner("小黄门"),
    })
    assert "不可选 ERRAND" not in _decision_prompt(
        engine, personality, one_free, _ACTION_SPACE
    )


def test_the_line_a_runner_carries_is_framed_as_his_own_words(container: object) -> None:
    """"Carry a message" must say on the menu line itself who speaks to whom. The warning in the
    slot sits forty lines further down.

    In practice the model often writes its instructions to the runner into this field, and the
    runner then reads them aloud verbatim in public, including things like "quietly find out,
    don't alert anyone".
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    prompt = _decision_prompt(
        engine, _make_personality(),
        _make_packet(visible_npcs={"npc_a": _runner("老宦", busy=False)}),
        _ACTION_SPACE,
    )
    menu = next(line for line in prompt.splitlines() if "[ERRAND]" in line)
    assert "原样说出去" in menu            # the words are his to say, not my instructions to him
    assert "当众" in menu                  # no addressee means a public reading
    assert "不会打听" in menu              # he can't bring information back; don't send him to ask


def test_the_runner_list_shown_and_the_list_bound_are_one_list(container: object) -> None:
    """The list printed and the list used for binding at parse time must come from the same source
    in the same order, and neither filters by busyness.

    The busy person stays on the list for two reasons, the second of which fails silently. First,
    "stop someone on an errand" only makes sense while he's busy (``physical_npc_index`` binds to
    this list). Second, removing people only on the display side would make printed #1 a different
    person; the model writes 1 and the code binds the first entry, a wrong binding with no error
    anywhere. Errands are blocked by 【此刻够不着的行动】, not by trimming the list.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_npcs={
        "npc_out": _runner("老宦", busy=True),
        "npc_idle": _runner("小黄门", busy=False),
    })
    prompt = _decision_prompt(engine, _make_personality(), packet, _ACTION_SPACE)

    shown = [line for line in prompt.splitlines() if line.strip().startswith("#")]
    listed = [line for line in shown if "老宦" in line or "小黄门" in line]
    assert [l.strip()[:2] for l in listed] == ["#1", "#2"]        # both present, original order
    assert "老宦" in listed[0] and "正在忙" in listed[0]          # the busy one is printed with a tag
    assert list(packet.spatial.visible_npc_ids) == ["npc_out", "npc_idle"]  # same list, same order, used for binding
    assert "不可选 ERRAND" not in prompt                          # someone is still idle → errands stay open


def test_the_cacheable_half_of_the_decision_prompt_never_varies(container: object) -> None:
    """The ``system`` part must be byte-identical across calls; it's the part the provider caches
    as a prefix.

    Decisions are the highest-traffic call in the system, and this part holds the role rules,
    action menu, and output schema. Mixing in anything per-agent or per-step breaks the cache on
    every call, with no error at all; the bill just goes up.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    plain, _, _ = engine._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )
    crowded, _, _ = engine._build_decision_prompt(  # noqa: SLF001
        _make_personality(name="另一个人"),
        _make_packet(
            visible_agent_ids=["a2"],
            reachable_location_ids=["gate"],
            visible_npcs={"npc_a": _runner("老宦", busy=True)},
        ),
        _ACTION_SPACE,
    )
    assert plain == crowded


def test_a_busy_runner_is_not_answered_by_pointing_at_a_letter(container: object) -> None:
    """Not being able to send him on an errand does not mean you can message him. He isn't on the
    "people I can reach" list at all.

    An exclusion line suggesting "use SEND_MESSAGE to pass word instead" gets followed, and since
    he isn't on that list the model picks #1, so a private order meant for a servant reaches
    someone else and is written into memory as such. Pointing to an alternative that can't reach
    the target is worse than pointing to nothing.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    packet = _make_packet(visible_npcs={"npc_a": _runner("老宦", busy=True)})
    prompt = _decision_prompt(engine, _make_personality(), packet, _ACTION_SPACE)

    block = prompt.split("【此刻够不着的行动】")[1].split("【")[0]
    assert "不可选 ERRAND" in block
    assert "改用 SEND_MESSAGE" not in block      # don't present messaging as an alternative way to reach him
    assert "传讯到不了他" in block
    # The list itself must be clear too, so nobody takes it for a second list of message
    # recipients
    assert "不在上面「我能触及的人」名单里" in prompt


def test_the_runner_list_is_called_one_name_everywhere(container: object) -> None:
    """When a slot asks the model for an index from some named list, that name must be exactly the
    header the prompt prints.

    If the two sides use different names, the model first has to realize they're the same list
    before it can fill the slot correctly, and that's a question it shouldn't have to answer.
    """
    import re
    from pathlib import Path

    src = Path("agent/decision.py").read_text()
    header = re.search(r'lines\.append\(\s*f?"(此处[^"（{]*)', src).group(1)
    assert set(re.findall(r'填"(此处[^"]*)"', src)) == {header}


def test_a_room_holding_only_a_runner_can_still_lay_hands_on_him(container: object) -> None:
    """Can't talk is not the same as can't touch. The tier without cognition can't hold a
    conversation, but can be stopped, pinned down, or have what he's holding taken.

    "Is there anyone to talk to" and "is there anyone to act on" are different questions with
    different answers. Judging both by one "people present" check would exclude PHYSICAL entirely
    when the only person in the room is an errand-runner, and that's exactly when someone wants to
    stop him: only whoever runs into him while he's out can do it.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    only_runner = _make_packet(visible_npcs={"npc_a": _runner("传诏宦官", busy=True)})
    prompt = _decision_prompt(engine, _make_personality(), only_runner, _ACTION_SPACE)
    assert "不可选 TALK" in prompt                      # talking still isn't possible
    assert "不可选 PHYSICAL" not in prompt              # but acting on him stays open
    assert "physical_npc_index" in prompt               # and there's a slot to put it in
    assert "这一拍谁也派不出去" in prompt                # no one to send is still excluded

    # Only a truly empty room (no people and no things) excludes acting physically too.
    empty = _make_packet()
    assert "不可选 PHYSICAL" in _decision_prompt(
        engine, _make_personality(), empty, _ACTION_SPACE
    )



def test_a_hand_on_a_runner_is_not_a_way_to_send_him(container: object) -> None:
    """The line after "nobody can be sent" must not read as "so use force instead".

    With "ERRAND not available" directly followed by "you can still act on them physically", the
    model has the emperor grab a guard's shoulder and order him to go scouting, using force to issue
    the order that couldn't be given. Acting physically means stopping, pinning, or seizing; it
    isn't sending someone on an errand.
    """
    from agent.decision import _ACTION_SPACE

    engine = _make_engine(container)
    prompt = _decision_prompt(
        engine, _make_personality(),
        _make_packet(visible_npcs={"npc_a": _runner("老宦", busy=True)}),
        _ACTION_SPACE,
    )
    block = prompt.split("【此刻够不着的行动】")[1].split("【")[0]
    assert "physical_npc_index" in block      # the path to stop him is still there
    assert "不是差遣" in block                 # but it isn't a substitute for an errand
