"""Unit tests for Agent's _known_agents cache and resolver.

Prerequisite for record_event's Mapping form — Agent accumulates perceived agent names into
the cache, and later record_event calls build an {id: identity} map via _resolve_agent_identities.
"""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine, NeedType
from agent.personality import EmotionState, PersonalityLayer, SoulLayer, StateLayer
from agent.relation import RelationSystem
from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.perception import PerceivedIdentity
from providers.llm.mock import MockLLMProvider


def _make_agent(container, *, agent_id: str = "agent-1") -> Agent:
    soul = SoulLayer(name="Test", agent_id=agent_id, role="x", core_traits=[], core_values=[])
    state = StateLayer(
        step=1,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        current_location="loc",
        dominant_need=NeedType.SOCIAL.value,
    )
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    return Agent(
        world_id="world-1",
        agent_id=agent_id,
        personality=PersonalityLayer(soul=soul, state=state),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,
            world_id="world-1", agent_id=agent_id,
        ),
        need_engine=NeedEngine(router),
        relation_system=RelationSystem(world_id="world-1", agent_id=agent_id),
        is_main_character=False,
    )


def test_remember_agent_stores_mapping(container) -> None:
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民"))
    assert agent._known_agents == {"a2": PerceivedIdentity(name="李世民")}


def test_remember_agent_skips_empty(container) -> None:
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name=""))
    assert "a2" not in agent._known_agents


def test_remember_agent_overwrites(container) -> None:
    """A more accurate name provided later overwrites the old value."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民"))
    agent.remember_agent("a2", PerceivedIdentity(name="秦王李世民"))
    assert agent._known_agents["a2"].name == "秦王李世民"


@pytest.mark.asyncio
async def test_resolve_agent_identities_uses_cache(container) -> None:
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民"))
    agent.remember_agent("a3", PerceivedIdentity(name="李建成"))
    result = await agent._resolve_agent_identities(["a2", "a3"])
    assert result == {"a2": PerceivedIdentity(name="李世民"), "a3": PerceivedIdentity(name="李建成")}


@pytest.mark.asyncio
async def test_resolve_agent_identities_leaves_a_stranger_unnamed(container) -> None:
    """Unknown id → no name (the renderer describes them), never a bare id (the value goes into
    embedded memory prose)."""
    agent = _make_agent(container)
    result = await agent._resolve_agent_identities(["unknown_agent"])
    assert result == {"unknown_agent": PerceivedIdentity(name="")}


@pytest.mark.asyncio
async def test_resolve_agent_identities_empty_input(container) -> None:
    agent = _make_agent(container)
    assert await agent._resolve_agent_identities([]) == {}


@pytest.mark.asyncio
async def test_resolve_agent_identities_mixed(container) -> None:
    """Partly known + partly unknown → known ones keep their name, unknown ones have none (no id
    leak)."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民"))
    result = await agent._resolve_agent_identities(["a2", "a99"])
    assert result == {"a2": PerceivedIdentity(name="李世民"), "a99": PerceivedIdentity(name="")}


def test_remember_agent_keeps_gender_when_a_later_sighting_omits_it(container) -> None:
    """Gender is only known from meeting face to face; later name-only messages must not erase it
    (see Agent.remember_agent)."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民", gender="男"))
    agent.remember_agent("a2", PerceivedIdentity(name="秦王李世民"))
    assert agent._known_agents["a2"] == PerceivedIdentity(name="秦王李世民", gender="男")


@pytest.mark.asyncio
async def test_resolve_agent_identities_keeps_gender(container) -> None:
    """The memory monologue says 他/她 of the people it lists; without gender it guesses, and a
    wrong guess is embedded for good."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民", gender="男"))
    assert await agent._resolve_agent_identities(["a2"]) == {"a2": PerceivedIdentity(name="李世民", gender="男")}


@pytest.mark.asyncio
async def test_resolve_agent_identities_falls_back_to_the_relation_s_name(container) -> None:
    """Someone I know but haven't perceived this run (away, or after a restore) keeps the name my
    relation holds; only a stranger stays unnamed."""
    agent = _make_agent(container)
    relation = await agent.relation_system.get_or_create("a2")
    relation.to_name = "李建成"
    assert await agent._resolve_agent_identities(["a2", "a99"]) == {
        "a2": PerceivedIdentity(name="李建成"), "a99": PerceivedIdentity(name=""),
    }


@pytest.mark.asyncio
async def test_relation_context_names_people_by_the_same_rule(container) -> None:
    """The relations block names people as memory prose does: a perceived name, its missing gender
    filled from my relation, and the relation's name for someone not perceived this run."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李建成"))
    for aid, name, gender in (("a2", "", "男"), ("a3", "李元吉", "男")):
        relation = await agent.relation_system.get_or_create(aid)
        relation.to_name, relation.to_gender = name, gender
        relation.interaction_count = 1
    block = await agent._resolve_relation_context(["a2", "a3"])
    assert "李建成（男）" in block
    assert "李元吉（男）" in block


@pytest.mark.asyncio
async def test_an_absent_relation_keeps_the_name_i_know_them_by(container) -> None:
    """A relation made mid-action has no ``to_name`` yet. When the person comes back to mind
    absent, the name I perceived must be cached, never the "某人" fallback: cached, it would
    stand in for the real name everywhere afterwards, a restore included."""
    from core.interfaces.perception import LocationView, SpatialPerception

    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民", gender="男"))
    await agent.relation_system.apply_interaction("a2", trust_delta=-0.3, affection_delta=-0.3, step=1)
    await agent.relation_system.apply_interaction("a3", trust_delta=-0.3, affection_delta=-0.3, step=1)

    alone = SpatialPerception(
        location_id="loc", location_view=LocationView(name="某处", description=""),
        world_time_label="卯时", current_step=2,
    )
    await agent._perceive_relevant_relations(alone, [], current_step=2)

    assert (await agent.relation_system.load_existing("a2")).to_name == "李世民"
    assert (await agent.relation_system.load_existing("a2")).to_gender == "男"
    assert (await agent.relation_system.load_existing("a3")).to_name == ""
