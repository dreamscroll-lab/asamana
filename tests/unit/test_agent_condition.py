"""BodyCondition — contract lock for the persistent "a person's ongoing condition" slot.

The field exists because everything a restrain ruling produces (ambient outcome, factual_memory,
emotion/relation delta) decays, while "his hands are tied behind his back right now" must stay
true and readable on every later step. So this file doesn't lock "can it be written", but that it
survives every path that silently drops fields: the hand-written field list in _copy_state, the
two independent persistence field lists, the per-step derivation into the perception packet, and
the three-state landing.
"""

from __future__ import annotations

import json

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import PersonalityLayer, SoulLayer
from agent.relation import RelationSystem
from core.interfaces.action import (
    ActionResult,
    ActionType,
    AgentAction,
    TargetAgentEffect,
)
from core.interfaces.condition import (
    BodyCondition,
    condition_from_dict,
    condition_to_dict,
)
from core.interfaces.llm import LLMRouter, LLMScene
from core.prompts import person_referent, render_condition
from providers.llm.mock import MockLLMProvider

BOUND = BodyCondition(description="双手被反绑", source_agent_id="a1", since_step=10)


def _personality(agent_id: str = "a2") -> PersonalityLayer:
    return PersonalityLayer(soul=SoulLayer(name="李元吉", gender="男", agent_id=agent_id))


def _make_agent(container, agent_id: str = "a2") -> Agent:
    """Built like test_agent_m5_gaps — feedback landing needs a real Agent, not a stand-in."""
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    return Agent(
        world_id="w",
        agent_id=agent_id,
        personality=_personality(agent_id),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store, world_id="w", agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id="w", agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=False,
    )


# ---------------------------------------------------------------------------
# State layer: where the field is most easily lost silently
# ---------------------------------------------------------------------------

def test_condition_survives_a_state_round_trip() -> None:
    """``_copy_state`` rebuilds StateLayer field by field by hand, and both ``.state`` and
    ``restore_state`` go through it.

    Miss that line and the field quietly becomes None on every state read — written but
    unreadable, no error, and every other test stays green. Same reason as
    test_last_decision_step_survives_a_state_round_trip.
    """
    p = _personality()
    p.set_condition(BOUND)
    assert p.state.condition == BOUND                    # read path

    p.restore_state(p.state)
    assert p.state.condition == BOUND                    # restore path


def test_clear_condition_frees_the_slot() -> None:
    p = _personality()
    p.set_condition(BOUND)
    p.clear_condition()
    assert p.state.condition is None


def test_a_new_condition_replaces_rather_than_stacks() -> None:
    """Single slot: tying someone up and then locking him in the dungeon is a replacement.
    Stacked conditions aren't modeled; description and location carry them separately.
    (The one exception is in the "single-slot replacement rule" section below.)"""
    p = _personality()
    p.set_condition(BOUND)
    p.set_condition(BodyCondition("被囚于地牢", "a1", 30))
    assert p.state.condition.description == "被囚于地牢"


# ---------------------------------------------------------------------------
# Serialization: both providers must produce the same thing
# ---------------------------------------------------------------------------

def test_dict_round_trip_is_lossless() -> None:
    assert condition_from_dict(condition_to_dict(BOUND)) == BOUND


def test_serialized_form_is_json_native() -> None:
    """The storage DTO and the snapshot payload only take JSON-native values — existing saves and
    snapshots both go through a json round-trip."""
    revived = json.loads(json.dumps(condition_to_dict(BOUND), ensure_ascii=False))
    assert condition_from_dict(revived) == BOUND


@pytest.mark.parametrize("bad", [None, {}, {"description": "   "}, "双手被反绑", 42, []])
def test_unreadable_input_reads_as_no_condition(bad: object) -> None:
    """No readable condition with a description = no condition. This is the normal path, not
    error tolerance: most people have no condition at any time, and existing saves don't have the
    key at all."""
    assert condition_from_dict(bad) is None


# ---------------------------------------------------------------------------
# Rendering: the two forms of the single renderer
# ---------------------------------------------------------------------------

def test_roster_form_is_bare_and_profile_form_carries_duration() -> None:
    assert render_condition(BOUND) == "双手被反绑"
    assert render_condition(
        BOUND, now_step=34, seconds_per_step=3600, with_duration=True,
    ) == "双手被反绑（已持续约1天）"


def test_duration_never_renders_a_step_count() -> None:
    """step is a code-layer coordinate; the narrative layer only has time points and durations
    (CLAUDE.md narrative/code layer boundary)."""
    text = render_condition(BOUND, now_step=34, seconds_per_step=3600, with_duration=True)
    assert "10" not in text and "34" not in text and "步" not in text


def test_no_condition_renders_nothing_at_all() -> None:
    """An empty string lets the caller omit the whole line — "处境：无" would add a line of noise
    for everyone on every step."""
    assert render_condition(None) == ""
    assert render_condition(BodyCondition(description="  ")) == ""


def test_roster_mark_shares_one_bracket_with_gender_and_presence() -> None:
    """Condition goes in the same parentheses as gender/presence, not a separate group (otherwise
    it renders a doubled "乙（男）（在场）")."""
    assert person_referent("李元吉", "男", "在场", render_condition(BOUND)) == "李元吉（男，在场，双手被反绑）"


# ---------------------------------------------------------------------------
# Feedback landing: three states
# ---------------------------------------------------------------------------

def _effect(**kw) -> TargetAgentEffect:
    return TargetAgentEffect(agent_id="a2", factual_memory="", **kw)


@pytest.mark.asyncio
async def test_target_effect_sets_a_condition(container) -> None:
    agent = _make_agent(container)
    await agent.apply_target_effect(_effect(condition_set=BOUND), from_agent_id="a1", step=10)
    assert agent.personality.state.condition == BOUND


@pytest.mark.asyncio
async def test_target_effect_clears_a_condition(container) -> None:
    agent = _make_agent(container)
    agent.personality.set_condition(BOUND)
    await agent.apply_target_effect(_effect(condition_cleared=True), from_agent_id="a1", step=12)
    assert agent.personality.state.condition is None


@pytest.mark.asyncio
async def test_an_unrelated_blow_leaves_a_standing_condition_alone(container) -> None:
    """The default "unchanged" is the fail-safe state: a punch landing on a bound man shouldn't
    untie his ropes as a side effect.

    It's the only silent state of the three, and so the easiest to write as "not given means
    clear".
    """
    agent = _make_agent(container)
    agent.personality.set_condition(BOUND)
    await agent.apply_target_effect(_effect(vitality_damage=0.2), from_agent_id="a1", step=12)
    assert agent.personality.state.condition == BOUND


# ---------------------------------------------------------------------------
# Feedback landing: the actor's own axis
# ---------------------------------------------------------------------------

def _own_result(**kw) -> ActionResult:
    """The actor's own PHYSICAL result — the condition lands on himself, not via target_effects."""
    return ActionResult(
        action=AgentAction(
            agent_id="a2", step=12, action_type=ActionType.PHYSICAL,
            action_description="奋力挣脱", estimated_steps=1,
        ),
        expected_outcome="", outcome="乙奋力挣扎。", **kw,
    )


@pytest.mark.asyncio
async def test_a_condition_can_land_on_the_actor_himself(container) -> None:
    """Others usually do the pinning, yet the one restrained is the actor — this consequence
    doesn't go through a target; only the actor's own axis can carry it."""
    agent = _make_agent(container)
    await agent._apply_feedback(
        result=_own_result(actor_condition_set=BOUND), dominant_need=None, emotion=None,
    )
    assert agent.personality.state.condition == BOUND


@pytest.mark.asyncio
async def test_the_actor_can_free_himself(container) -> None:
    """A bound man breaking free is the escape valve's legitimate ending; without this axis no one
    could ever undo his ropes."""
    agent = _make_agent(container)
    agent.personality.set_condition(BOUND)
    await agent._apply_feedback(
        result=_own_result(actor_condition_cleared=True), dominant_need=None, emotion=None,
    )
    assert agent.personality.state.condition is None


@pytest.mark.asyncio
async def test_an_ordinary_action_leaves_the_actors_condition_alone(container) -> None:
    """Default "unchanged" — same rule as the target axis; the silent state is the easiest to
    write as "not given means clear"."""
    agent = _make_agent(container)
    agent.personality.set_condition(BOUND)
    await agent._apply_feedback(result=_own_result(), dominant_need=None, emotion=None)
    assert agent.personality.state.condition == BOUND


# ---------------------------------------------------------------------------
# Single-slot replacement rule
# ---------------------------------------------------------------------------

def test_a_self_limiting_condition_cannot_displace_a_standing_one() -> None:
    """A scrape picked up during a failed struggle must not displace the restraint — it expires on
    its own, so failing to struggle free would end up freeing him. When the single slot can't hold
    both, keep the one that can't be undone."""
    p = _personality()
    p.set_condition(BOUND)
    p.set_condition(BodyCondition("手腕被扣处泛红微痛", "a1", since_step=36, until_step=37))
    assert p.state.condition == BOUND


def test_a_standing_condition_still_displaces_a_self_limiting_one() -> None:
    """No limit the other way: tied up while still drugged, the binding is the one that can't be
    undone, so it stays."""
    p = _personality()
    p.set_condition(BodyCondition("中了软筋散", "a1", since_step=5, until_step=9))
    p.set_condition(BOUND)
    assert p.state.condition == BOUND


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_self_limiting_condition_lifts_on_time(container) -> None:
    agent = _make_agent(container)
    agent.personality.set_condition(BodyCondition("中了软筋散", "a1", since_step=5, until_step=9))
    assert agent.expire_condition(8) is False
    assert agent.personality.state.condition is not None
    assert agent.expire_condition(9) is True
    assert agent.personality.state.condition is None


@pytest.mark.asyncio
async def test_a_rope_never_unties_itself(container) -> None:
    """``until_step is None`` = needs outside help to lift. An automatic timeout would just replay
    the "state is unseen" problem in another form: a prisoner nobody attends to suddenly goes free."""
    agent = _make_agent(container)
    agent.personality.set_condition(BOUND)
    assert agent.expire_condition(10_000) is False
    assert agent.personality.state.condition == BOUND
