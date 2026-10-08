"""Unit tests for Agent's _known_agents cache and resolver.

Prerequisite for record_event's Mapping form — Agent accumulates perceived agent names into
the cache, and later record_event calls build a dict[str, str] via _resolve_agent_names.
"""

from __future__ import annotations


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
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id="world-1", agent_id=agent_id),
        agent_store=container.agent_store,
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


def test_resolve_agent_names_uses_cache(container) -> None:
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民"))
    agent.remember_agent("a3", PerceivedIdentity(name="李建成"))
    result = agent._resolve_agent_names(["a2", "a3"])
    assert result == {"a2": "李世民", "a3": "李建成"}


def test_resolve_agent_names_falls_back_to_descriptive(container) -> None:
    """Unknown id → descriptive referent "某人", never a bare id (the value goes into
    embedded memory prose)."""
    agent = _make_agent(container)
    result = agent._resolve_agent_names(["unknown_agent"])
    assert result == {"unknown_agent": "某人"}


def test_resolve_agent_names_empty_input(container) -> None:
    agent = _make_agent(container)
    assert agent._resolve_agent_names([]) == {}


def test_resolve_agent_names_mixed(container) -> None:
    """Partly known + partly unknown → known ones use name, unknown fall back to the descriptive
    "某人" (no id leak)."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民"))
    result = agent._resolve_agent_names(["a2", "a99"])
    assert result == {"a2": "李世民", "a99": "某人"}


def test_remember_agent_keeps_gender_when_a_later_sighting_omits_it(container) -> None:
    """Gender is only known from meeting face to face; later name-only messages must not erase it
    (see Agent.remember_agent)."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民", gender="男"))
    agent.remember_agent("a2", PerceivedIdentity(name="秦王李世民"))
    assert agent._known_agents["a2"] == PerceivedIdentity(name="秦王李世民", gender="男")


def test_resolve_agent_names_gives_bare_names_not_referents(container) -> None:
    """Memory prose is natural narration: only names here; gender tags belong only to lists
    (person_referent)."""
    agent = _make_agent(container)
    agent.remember_agent("a2", PerceivedIdentity(name="李世民", gender="男"))
    assert agent._resolve_agent_names(["a2"]) == {"a2": "李世民"}
