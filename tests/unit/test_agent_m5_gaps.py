"""Gap-filler: apply_target_effect + the _apply_feedback vitality/death path.

Covers code paths test_agent.py doesn't:
- the full apply_target_effect contract (7 cases)
- vitality_damage / death trigger in _apply_feedback (4 cases)
"""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine, NeedType
from agent.personality import (
    EmotionState,
    EmotionType,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from agent.relation import RelationSystem
from core.interfaces.action import (
    Ref,
    ActionResult,
    ActionTarget,
    ActionType,
    AgentAction,
    TargetAgentEffect,
)
from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider


def _neutral_emotion() -> EmotionState:
    """Pre-appraised emotion for direct _apply_feedback (pure application) tests."""
    return EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0)


def _make_personality(*, agent_id: str = "agent-target", initial_vitality: float = 1.0) -> PersonalityLayer:
    soul = SoulLayer(
        name="李建成", agent_id=agent_id, role="太子",
        core_traits=("谨慎",),
        core_values=("家族",),
    )
    state = StateLayer(
        agent_id=agent_id, step=1,
        emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
        current_location="donggong",
        vitality=initial_vitality,
    )
    return PersonalityLayer(soul=soul, state=state)


def _make_agent(container, *, agent_id: str = "agent-target", vitality: float = 1.0) -> Agent:
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    return Agent(
        world_id="world-1",
        agent_id=agent_id,
        personality=_make_personality(agent_id=agent_id, initial_vitality=vitality),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,
            world_id="world-1", agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id="world-1", agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=False,
    )


# ---------------------------------------------------------------------------
# Full apply_target_effect contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_target_effect_records_factual_memory(container) -> None:
    agent = _make_agent(container)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="李世民一箭射中我胸膛",
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    # B's memory is written via record_event (dual-stream); the factual stream keeps the
    # raw event content. (_make_agent uses the InMemory provider)
    contents = [m.stored_content for m in agent.memory_system._entries.values()]
    assert any("一箭射中" in c for c in contents)


@pytest.mark.asyncio
async def test_apply_target_effect_with_no_fact_records_no_memory(container) -> None:
    """A producer that failed sends an empty fact; the harm still lands, but no memory does, or
    the experiential rewrite would make one up from nothing."""
    agent = _make_agent(container)
    before = agent.personality.state.vitality
    effect = TargetAgentEffect(agent_id="agent-target", factual_memory="", vitality_damage=0.3)
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    assert agent.memory_system._entries == {}
    assert agent.personality.state.vitality < before


@pytest.mark.asyncio
async def test_apply_target_effect_updates_emotion_when_type_present(container) -> None:
    agent = _make_agent(container)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="被射中",
        emotion_type=EmotionType.FEAR.value,
        emotion_intensity=0.9, emotion_valence=-0.9,
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    emotion = agent.personality.state.emotion
    assert emotion.primary == EmotionType.FEAR
    assert abs(emotion.intensity - 0.9) < 1e-6
    assert abs(emotion.valence - (-0.9)) < 1e-6


@pytest.mark.asyncio
async def test_apply_target_effect_emotion_type_none_skips_emotion_update(container) -> None:
    """emotion_type=None → skip the emotion update and keep the original emotion."""
    agent = _make_agent(container)
    original = agent.personality.state.emotion
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="无情绪标注的事件",
        emotion_type=None,
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    assert agent.personality.state.emotion.primary == original.primary
    assert agent.personality.state.emotion.intensity == original.intensity


@pytest.mark.asyncio
async def test_apply_target_effect_updates_relation_when_present(container) -> None:
    agent = _make_agent(container)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="被攻击",
        relation_toward_actor=("agent-shimin", -0.3, -0.6),
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    relation = await agent.relation_system.get_or_create("agent-shimin")
    # Negative trust is tripled → -0.9, clamped to 0 (initial 0.5 + -0.9 → -0.4 → clamp to 0)
    assert relation.trust_objective < 0.5
    # Negative affection adds directly (initial 0, plus -0.6 → -0.6)
    assert relation.affection_objective < 0.0


@pytest.mark.asyncio
async def test_apply_target_effect_relation_none_skips_relation_update(container) -> None:
    agent = _make_agent(container)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="无关系标注",
        relation_toward_actor=None,
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    # The relation wasn't triggered by get_or_create, so it's not in the store
    relation_loaded = await container.agent_store.load_relation(
        "world-1", "agent-target", "agent-shimin"
    )
    # This effect didn't trigger a relation update, and seed_factual_memory won't either;
    # the relation should not exist
    assert relation_loaded is None


@pytest.mark.asyncio
async def test_apply_target_effect_vitality_damage_below_lethal(container) -> None:
    """vitality_damage > 0 but not lethal: vitality drops, is_alive stays True."""
    agent = _make_agent(container, vitality=1.0)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="轻伤",
        vitality_damage=0.3,
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    assert agent.personality.state.vitality == pytest.approx(0.7)
    assert agent.personality.is_alive is True
    assert agent.is_active is True


@pytest.mark.asyncio
async def test_apply_target_effect_lethal_vitality_triggers_death(container) -> None:
    """vitality_damage drives vitality to zero → set_active=False, death_cause recorded, and NO
    death memory written for the dead."""
    agent = _make_agent(container, vitality=0.2)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="致命一击",
        vitality_damage=0.5,  # 0.2 - 0.5 = -0.3 → clamped to 0
        death_cause="死于刀兵",
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    assert agent.personality.state.vitality == 0.0
    assert agent.personality.is_alive is False
    assert agent.is_active is False
    assert agent.death_cause == "死于刀兵"
    # After death there's no experiencing subject: never write an "I died" memory for the dead.
    contents = [m.stored_content for m in agent.memory_system._entries.values()]
    assert not any("殒没" in c for c in contents)


@pytest.mark.asyncio
async def test_apply_target_effect_vitality_zero_damage_skips_vitality_call(container) -> None:
    """vitality_damage=0 (default) → the vitality path isn't triggered."""
    agent = _make_agent(container, vitality=1.0)
    effect = TargetAgentEffect(
        agent_id="agent-target",
        factual_memory="无伤事件",
        vitality_damage=0.0,
    )
    await agent.apply_target_effect(effect, from_agent_id="agent-shimin", step=5)
    assert agent.personality.state.vitality == 1.0
    assert agent.personality.is_alive is True


# ---------------------------------------------------------------------------
# _apply_feedback vitality / death path
# ---------------------------------------------------------------------------


def _action_result_with_damage(damage: float, *, target_agent_id: str = "agent-2") -> ActionResult:
    action = AgentAction(
        action_type=ActionType.PHYSICAL,
        action_description="冒险行动",
        agent_id="agent-target",
        target=ActionTarget(acts_on=[Ref.agent(target_agent_id)]),
        step=5,
    )
    return ActionResult(
        action=action,
        expected_outcome="完成",
        outcome="勉强完成",
        succeeded=True,
        factual_memory="冒险后受伤",
        relation_updates=[],
        vitality_damage=damage,
    )


@pytest.mark.asyncio
async def test_apply_feedback_vitality_damage_below_lethal(container) -> None:
    """vitality_damage > 0 but not lethal in _apply_feedback: drops, but is_alive stays True."""
    agent = _make_agent(container, vitality=1.0)
    result = _action_result_with_damage(0.3)
    await agent._apply_feedback(result=result, dominant_need=NeedType.SAFETY, emotion=_neutral_emotion())
    assert agent.personality.state.vitality == pytest.approx(0.7)
    assert agent.personality.is_alive is True
    assert agent.is_active is True


@pytest.mark.asyncio
async def test_apply_feedback_lethal_vitality_triggers_death(container) -> None:
    """Lethal vitality_damage in _apply_feedback → set_active=False, death_cause recorded, and NO
    death memory written for the dead."""
    agent = _make_agent(container, vitality=0.1)
    result = _action_result_with_damage(0.5)
    await agent._apply_feedback(result=result, dominant_need=NeedType.SAFETY, emotion=_neutral_emotion())
    assert agent.personality.state.vitality == 0.0
    assert agent.personality.is_alive is False
    assert agent.is_active is False
    assert agent.death_cause == "生命耗尽。"
    contents = [m.stored_content for m in agent.memory_system._entries.values()]
    assert not any("殒没" in c for c in contents)


@pytest.mark.asyncio
async def test_apply_feedback_zero_vitality_damage_unchanged(container) -> None:
    """_apply_feedback with vitality_damage=0 → vitality unchanged."""
    agent = _make_agent(container, vitality=0.8)
    result = _action_result_with_damage(0.0)
    await agent._apply_feedback(result=result, dominant_need=NeedType.SAFETY, emotion=_neutral_emotion())
    assert agent.personality.state.vitality == pytest.approx(0.8)
    assert agent.is_active is True
