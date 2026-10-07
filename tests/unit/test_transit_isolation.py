"""IN_TRANSIT pseudo-location isolation and mid-journey snapshot restore (movement guards).

IN_TRANSIT is not "a room all travelers share":
- travelers can't see, talk to, or act physically on each other, and don't hear each other's
  location ambient (global signals still reach them);
- perception and feasibility text never contain the code-layer token "__in_transit__"; the location
  is referred to in narrative terms;
- in-transit records such as MOVE ticks are not carried as ambient (guarded by
  runtime._carry_step_observations);
- on mid-journey snapshot restore, travelers "turn back" to the origin registered by the
  environment (transit_origins), or to their initial location if none is registered.
"""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import EmotionState, PersonalityLayer, SoulLayer
from agent.relation import RelationSystem
from engine.clock import WorldTime, WorldTimeConfig
from engine.environment import IN_TRANSIT, EnvironmentSystem
from world.initializer import WorldInitializer
from world.models import AgentDefinition, AgentTier
from core.interfaces.place import Place


def _world_time(step: int = 1) -> WorldTime:
    return WorldTime.from_step(step, WorldTimeConfig(start_hour=6, seconds_per_step=3600))


def _make_env() -> EnvironmentSystem:
    env = EnvironmentSystem()
    for entity_id, name, connections in (
        ("hall", "大殿", {"garden": 3}),
        ("garden", "后花园", {"hall": 3}),
    ):
        env.space.register_place(Place(
            place_id=entity_id, name=name, description="",
            connections=connections, is_public=True, capacity=50,
        ))
    env.place_agent(agent_id="agent-a", location_id="hall")
    env.place_agent(agent_id="agent-b", location_id="hall")
    return env


def _make_agent(container, agent_id: str, name: str) -> Agent:
    return Agent(
        world_id="w",
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(name=name, role="official", agent_id=agent_id),
        ),
        decision_engine=DecisionEngine(container.llm_router),
        llm_router=container.llm_router,
        memory_system=MemorySystem(
            container.llm_router, container.embedding, container.vector_store,
            world_id="w", agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id="w", agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Isolation between travelers
# ─────────────────────────────────────────────────────────────────────────────


def test_in_transit_agents_invisible_to_each_other() -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
    env.move_body(body_id="agent-b", location_id=IN_TRANSIT)
    env.begin_step(step=1, world_time=_world_time())

    spatial = env.spatial_for(agent_id="agent-a")
    assert spatial.visible_agent_ids == []
    assert spatial.reachable_locations == []
    assert IN_TRANSIT not in spatial.location_view.name
    assert spatial.location_view.name  # a narrative location name, not an empty string


def test_in_transit_location_ambient_isolated() -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
    env.move_body(body_id="agent-b", location_id=IN_TRANSIT)
    # Simulate a producer writing location ambient into the pseudo-location (a backstop beyond the
    # guard)
    env.record_carry_observation(
        location_id=IN_TRANSIT, observation="甲正赶往秘密据点", actor_ids=("agent-a",),
    )
    env.begin_step(step=1, world_time=_world_time())

    spatial_b = env.spatial_for(agent_id="agent-b")
    assert not any("秘密据点" in ev.content for ev in spatial_b.ambient_events), (
        "在途者读到了其他旅人的位置 ambient——IN_TRANSIT 被当成了共享房间"
    )


def test_talk_infeasible_when_either_side_in_transit() -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
    env.move_body(body_id="agent-b", location_id=IN_TRANSIT)
    # Two travelers' location strings are equal, but they must not count as being in the same place
    result = env.check_talk_feasibility("agent-a", ["agent-b"])
    assert not result.ok

    env.move_body(body_id="agent-a", location_id="hall")
    result = env.check_talk_feasibility("agent-a", ["agent-b"])
    assert not result.ok


def test_physical_infeasible_when_both_in_transit() -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
    env.move_body(body_id="agent-b", location_id=IN_TRANSIT)
    result = env.check_physical_feasibility("agent-a", "agent-b", "agent")
    assert not result.ok


def test_move_feasibility_reason_has_no_pseudo_token() -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
    result = env.check_move_feasibility("agent-a", "garden")
    assert not result.ok
    assert IN_TRANSIT not in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# transit_origins registration and snapshot round-trip
# ─────────────────────────────────────────────────────────────────────────────


def test_transit_origin_lifecycle_and_snapshot_roundtrip() -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
    assert env.transit_origin("agent-a") == "hall"
    assert env.snapshot_state()["transit_origins"] == {"agent-a": "hall"}

    # Registration is cleared on arrival
    env.move_body(body_id="agent-a", location_id="garden")
    assert env.transit_origin("agent-a") is None
    assert env.snapshot_state()["transit_origins"] == {}

    # restore round-trip
    fresh = _make_env()
    fresh.restore_state({"transit_origins": {"agent-b": "garden"}, "entity_states": {}})
    assert fresh.transit_origin("agent-b") == "garden"


# ─────────────────────────────────────────────────────────────────────────────
# Mid-journey snapshot restore: send travelers back to their origin
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repatriation_lands_at_transit_origin(container) -> None:
    env = _make_env()
    env.move_body(body_id="agent-a", location_id=IN_TRANSIT)  # origin registered as hall
    agent = _make_agent(container, "agent-a", "甲")
    agent.personality.update_location(location=IN_TRANSIT)

    initializer = WorldInitializer(container)
    initializer._repatriate_transit_agents({"agent-a": agent}, [], env)  # noqa: SLF001

    assert env.get_body_location("agent-a") == "hall"
    assert agent.personality.state.current_location == "hall"


@pytest.mark.asyncio
async def test_repatriation_falls_back_to_initial_location(container) -> None:
    """A snapshot with no transit_origins entry → fall back to
    AgentDefinition.initial_location."""
    env = _make_env()
    env.place_agent(agent_id="agent-a", location_id=IN_TRANSIT)  # place doesn't register an origin
    assert env.transit_origin("agent-a") is None
    agent = _make_agent(container, "agent-a", "甲")
    agent.personality.update_location(location=IN_TRANSIT)

    definition = AgentDefinition(
        agent_id="agent-a", name="甲", tier=AgentTier.BACKGROUND,
        soul=SoulLayer(name="甲", role="", agent_id="agent-a"),
        initial_location="garden",
        initial_emotion=EmotionState(),
    )
    initializer = WorldInitializer(container)
    initializer._repatriate_transit_agents({"agent-a": agent}, [definition], env)  # noqa: SLF001

    assert env.get_body_location("agent-a") == "garden"
    assert agent.personality.state.current_location == "garden"
