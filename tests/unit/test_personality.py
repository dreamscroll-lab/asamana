"""Unit tests for agent personality subsystem."""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from agent.goals import GoalEntity, GoalStatus
from agent.personality import (
    ActionStatus,
    AgentActivityStatus,
    EmotionState,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)


def _make_soul(**kwargs: object) -> SoulLayer:
    defaults: dict = dict(
        name="Test Agent",
        role="scholar",
        agent_id="agent-1",
        core_traits=["curious"],
        core_values=["truth"],
        self_image="a learned person",
        life_goal="Uncover hidden knowledge.",
        hard_constraints=[],
    )
    defaults.update(kwargs)
    return SoulLayer(**defaults)  # type: ignore[arg-type]


def test_soul_layer_construction_preserves_all_fields() -> None:
    soul = _make_soul(name="Li Shimin", age=28, gender="male")

    assert soul.name == "Li Shimin"
    assert soul.age == 28
    assert soul.gender == "male"
    assert soul.core_traits == ("curious",)
    assert soul.life_goal == "Uncover hidden knowledge."


def test_soul_layer_is_frozen() -> None:
    soul = _make_soul()

    with pytest.raises(FrozenInstanceError):
        soul.name = "Changed"  # type: ignore[misc]


def test_soul_layer_sequence_fields_are_immutable() -> None:
    soul = _make_soul(hard_constraints=["Never insult the king."])

    assert soul.hard_constraints == ("Never insult the king.",)
    with pytest.raises(AttributeError):
        soul.core_traits.append("reckless")  # type: ignore[attr-defined]


def test_state_layer_defaults_are_sensible() -> None:
    state = StateLayer()

    assert state.step == 0
    assert state.emotion.primary == "neutral"
    assert state.activity_status == AgentActivityStatus.IDLE
    assert state.action_status == ActionStatus.IDLE
    assert state.current_location == "unknown"
    assert state.dominant_need is None
    assert state.short_term_goals == []


def test_personality_update_location_updates_state() -> None:
    personality = PersonalityLayer(soul=_make_soul())

    personality.update_location(step=5, location="library")

    assert personality.state.current_location == "library"
    assert personality.state.step == 5


def test_personality_state_property_is_defensive_snapshot() -> None:
    personality = PersonalityLayer(soul=_make_soul())
    state = personality.state

    state.current_location = "forged"
    state.emotion.primary = "forged"

    assert personality.state.current_location == "unknown"
    assert personality.state.emotion.primary == "neutral"


def test_complete_action_with_none_emotion_keeps_current_mood() -> None:
    """complete_action(emotion=None) keeps the current mood, but the result still lands in
    action_history.

    emotion=None means the appraisal LLM failed (fallback tier 1: write nothing). Resetting the
    mood to neutral would fabricate an emotion change. The action itself is an objective fact and
    is still recorded.
    """
    personality = PersonalityLayer(soul=_make_soul())
    personality.update_emotion(primary="anger", intensity=0.8, valence=-0.6)

    personality.begin_action(
        step=3,
        description="Search the archives.",
        activity_status=AgentActivityStatus.WORKING,
        target="archive",
        estimated_steps=1,
    )
    personality.complete_action(
        step=4,
        description="Search the archives.",
        result_summary="Found nothing.",
        succeeded=False,
        emotion=None,
    )

    # Mood unchanged (neither overwritten nor reset to the default)
    assert personality.state.emotion.primary == "anger"
    assert personality.state.emotion.intensity == pytest.approx(0.8)
    assert personality.state.emotion.valence == pytest.approx(-0.6)
    # The action result still lands (a failed action → FAILED, a real world event)
    assert personality.state.action_status == ActionStatus.FAILED
    assert personality.state.last_action == "Search the archives."
    assert personality.state.last_action_succeeded is False


def test_personality_begin_and_complete_action_lifecycle() -> None:
    personality = PersonalityLayer(soul=_make_soul())
    emotion_after = EmotionState(primary="joy", intensity=0.6, valence=0.4)

    personality.begin_action(
        step=3,
        description="Search the archives.",
        activity_status=AgentActivityStatus.WORKING,
        target="archive",
        estimated_steps=2,
    )

    assert personality.state.action_status == ActionStatus.IN_PROGRESS
    assert personality.state.current_action == "Search the archives."
    assert personality.state.activity_status == AgentActivityStatus.WORKING
    assert personality.state.action_remaining_steps == 1

    personality.complete_action(
        step=5,
        description="Search the archives.",
        result_summary="Found the document.",
        succeeded=True,
        emotion=emotion_after,
    )

    assert personality.state.action_status == ActionStatus.COMPLETED
    assert personality.state.current_action is None
    assert personality.state.last_action == "Search the archives."
    assert personality.state.last_action_succeeded is True
    assert personality.state.activity_status == AgentActivityStatus.IDLE
    assert personality.state.emotion.primary == "joy"


def test_personality_update_emotion_clamps_values() -> None:
    personality = PersonalityLayer(soul=_make_soul())

    personality.update_emotion(primary="angry", intensity=2.5, valence=-3.0, triggered_by="insult")

    assert personality.state.emotion.intensity == 1.0
    assert personality.state.emotion.valence == -1.0
    assert personality.state.emotion.triggered_by == "insult"


def test_personality_apply_need_state_writes_all_fields() -> None:
    personality = PersonalityLayer(soul=_make_soul())

    personality.apply_need_state(
        step=7,
        active_needs=["social", "esteem"],
        dominant_need="social",
        long_term_goals=["Restore the dynasty."],
        short_term_goal_entities=[
            GoalEntity(id="stg-7", text="Talk to the minister.", goal_type="short_term"),
            GoalEntity(
                id="stg-6", text="Already done.", goal_type="short_term",
                status=GoalStatus.COMPLETED,
            ),
        ],
    )

    assert personality.state.step == 7
    assert personality.state.active_needs == ["social", "esteem"]
    assert personality.state.dominant_need == "social"
    assert personality.state.long_term_goals == ["Restore the dynasty."]
    assert personality.state.short_term_goals == ["Talk to the minister."]


def test_personality_to_prompt_context_includes_name_and_emotion() -> None:
    personality = PersonalityLayer(
        soul=_make_soul(
            name="Wei Zheng",
            background="A court official known for direct counsel.",
            core_values=["truth", "duty"],
        )
    )
    personality.update_emotion(primary="anticipation", intensity=0.5, valence=0.3)

    context = personality.to_prompt_context()

    assert "Wei Zheng" in context
    assert "期待" in context
    assert "court official" in context
    assert "truth" in context
    assert "intensity=" not in context
    assert "valence=" not in context


def test_identity_text_is_the_single_identity_head() -> None:
    """The one identity-header renderer: name, N years old, gender; empty fields are omitted, with
    no empty brackets or dangling commas."""
    assert _make_soul(name="李世民", age=28, gender="男").identity_text() == "李世民，28岁，男"
    assert _make_soul(name="李世民", age=0, gender="男").identity_text() == "李世民，男"
    assert _make_soul(name="李世民", age=28, gender="").identity_text() == "李世民，28岁"
    assert _make_soul(name="", age=0, gender="").identity_text() == "某人"


def test_prompt_context_identity_head_carries_gender() -> None:
    """The persona block's first line is the identity header; in-character prompts rely on it to
    know their own gender."""
    soul = _make_soul(name="长孙无垢", age=24, gender="女")
    text = PersonalityLayer(soul=soul, state=StateLayer()).to_prompt_context()
    assert text.startswith("长孙无垢，24岁，女。")
