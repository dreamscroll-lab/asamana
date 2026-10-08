"""Unit tests for the engine environment subsystem."""

from __future__ import annotations

import dataclasses
import pathlib


from core.interfaces.action import EntitySpawn, EntityStateChange
from core.prompts import SituationVoice, render_location
from core.interfaces.perception import SpatialPerception, VisibleEntity
from engine.clock import WorldTime, WorldTimeConfig
from engine.environment import IN_TRANSIT, EnvironmentSystem
from world.models import EntityPresence, WorldEntity, WorldEntitySeed, WorldEntityType
from core.interfaces.place import Place
from worlds.tiled import TiledWorldConfig


def _world_time(step: int = 1) -> WorldTime:
    return WorldTime.from_step(step, WorldTimeConfig(start_hour=6, seconds_per_step=60))


def _make_location(entity_id: str, name: str, connections: dict | None = None, capacity: int = 50) -> Place:
    return Place(
        place_id=entity_id,
        name=name,
        connections=connections or {},
        capacity=capacity,
        is_public=True,
    )


def _make_item(entity_id: str, name: str, location_id: str | None = None, owner_id: str | None = None,
               is_takeable: bool = True, state: str = "intact") -> WorldEntity:
    if owner_id is not None:
        presence = EntityPresence.HELD
        presence_ref: str | None = owner_id
    else:
        presence = EntityPresence.AT_LOCATION
        presence_ref = location_id
    return WorldEntity(
        entity_id=entity_id,
        name=name,
        entity_type=WorldEntityType.ITEM,
        state=state,
        presence=presence,
        presence_ref=presence_ref,
        is_takeable=is_takeable,
        is_public=True,
    )


def _make_connected_env(seconds: int = 3600) -> EnvironmentSystem:
    """Environment with two named locations connected by a walk of ``seconds``."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "hall", connections={"garden": seconds}))
    env.space.register_place(_make_location("garden", "garden", connections={"hall": seconds}))
    env.place_agent(agent_id="agent-1", location_id="hall")
    env.place_agent(agent_id="agent-2", location_id="hall")
    return env


# ---------------------------------------------------------------------------
# spatial_for() — core SpatialPerception surface
# ---------------------------------------------------------------------------


def test_spatial_for_returns_spatial_perception_with_basic_fields() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="loc1")

    spatial = env.spatial_for(agent_id="a1", step=1, world_time="辰时")

    assert isinstance(spatial, SpatialPerception)
    assert spatial.location_id == "loc1"
    assert spatial.current_step == 1
    assert spatial.world_time_label == "辰时"


def test_spatial_for_returns_visible_agents_only_at_same_location() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="loc1")
    env.place_agent(agent_id="a2", location_id="loc1")
    env.place_agent(agent_id="a3", location_id="loc2")

    spatial = env.spatial_for(agent_id="a1", step=1, world_time="辰时")

    assert "a2" in spatial.visible_agent_ids
    assert "a3" not in spatial.visible_agent_ids
    assert "a1" not in spatial.visible_agent_ids


def test_spatial_for_unknown_agent_yields_unknown_location() -> None:
    env = EnvironmentSystem()

    spatial = env.spatial_for(agent_id="ghost")

    assert spatial.location_id == "unknown"
    assert spatial.visible_agent_ids == []


def test_remove_agent_clears_world_state() -> None:
    """Death: remove_body clears position and lists, so the dead drop out of others' view."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "hall"))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    env.begin_step(step=1, world_time=_world_time(1))

    env.remove_body("a2")

    assert env.get_body_location("a2") == "unknown"
    assert "a2" not in env.bodies_at("hall")
    spatial = env.spatial_for(agent_id="a1", step=1, world_time="辰时")
    assert "a2" not in spatial.visible_agent_ids


def test_spatial_for_uses_current_step_and_time_when_omitted() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="loc1")
    env.begin_step(step=7, world_time=_world_time(7))

    spatial = env.spatial_for(agent_id="a1")

    assert spatial.current_step == 7
    assert spatial.world_time_label == _world_time(7).time_label


# ---------------------------------------------------------------------------
# begin_step() — clears step-scoped annotations
# ---------------------------------------------------------------------------


def test_carry_observation_flipped_then_cleared_by_begin_step() -> None:
    """carry → step_annotations on next begin_step, then cleared on the step after."""
    env = EnvironmentSystem()
    env.place_agent(agent_id="agent-1", location_id="hall")
    env.record_carry_observation(location_id="hall", observation="A candle flickered.")

    env.begin_step(step=2, world_time=_world_time(2))
    spatial = env.spatial_for(agent_id="agent-1")
    assert any("candle" in ev.content for ev in spatial.ambient_events)

    env.begin_step(step=3, world_time=_world_time(3))
    spatial = env.spatial_for(agent_id="agent-1")
    ambient = " ".join(ev.content for ev in spatial.ambient_events)
    assert "candle" not in ambient


# ---------------------------------------------------------------------------
# carry observation visibility rules
# ---------------------------------------------------------------------------


def test_carry_observation_appears_in_next_step_ambient_events() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="agent-1", location_id="library")

    env.record_carry_observation(
        location_id="library", observation="Someone left a note on the table."
    )
    env.begin_step(step=2, world_time=_world_time(2))
    spatial = env.spatial_for(agent_id="agent-1")

    assert any("note" in ev.content for ev in spatial.ambient_events)


# ---------------------------------------------------------------------------
# visible_entities in spatial_for()
# ---------------------------------------------------------------------------


def test_visible_entities_filter_by_co_presence() -> None:
    """Being present is enough to be perceived. The test is co-location, not ownership.

    ``ring`` is invisible because a2, who holds it, isn't placed anywhere in the world (see the
    control below), not because someone holds it. Pinning both cases apart keeps "held" from being
    read as "not here".
    """
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")

    on_floor = _make_item("lantern", "lantern", location_id="hall", is_takeable=True, state="ready")
    held = _make_item("ring", "ring", owner_id="a2", is_takeable=True, state="ready")
    elsewhere = _make_item("scroll", "scroll", location_id="garden", is_takeable=True, state="ready")
    env.register_entity(on_floor)
    env.register_entity(held)
    env.register_entity(elsewhere)

    visible_names = {e.name for e in env.spatial_for(agent_id="a1").visible_entities}
    assert "lantern" in visible_names
    assert "ring" not in visible_names       # a2 isn't anywhere
    assert "scroll" not in visible_names

    # Control: put a2 in the same place and the same item becomes visible. What changed is where
    # he stands, not who owns it.
    env.place_agent(agent_id="a2", location_id="hall")
    assert "ring" in {e.name for e in env.spatial_for(agent_id="a1").visible_entities}


def test_visible_entities_in_spatial_perception() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")

    sword = _make_item("sword", "铁剑", location_id="hall", is_takeable=True, state="intact")
    flag = WorldEntity(
        entity_id="flag", name="旌旗", entity_type=WorldEntityType.LANDMARK,
        state="intact", presence_ref="hall", is_public=True,
    )
    env.register_entity(sword)
    env.register_entity(flag)

    spatial = env.spatial_for(agent_id="a1")

    entity_names = {e.name for e in spatial.visible_entities}
    assert "铁剑" in entity_names
    assert "旌旗" in entity_names
    assert all(isinstance(e, VisibleEntity) for e in spatial.visible_entities)


def test_visible_entities_includes_items_held_by_a_co_present_agent() -> None:
    """Something in another person's hand is still in this room; holder_id carries who holds it.

    This line must not filter out held items: an item would vanish the moment someone picks it up,
    even for its holder, while his goals still depend on it.
    """
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    env.register_entity(_make_item("pouch", "钱袋", owner_id="a2", is_takeable=True))

    seen = {e.name: e for e in env.spatial_for(agent_id="a1").visible_entities}
    assert "钱袋" in seen
    assert seen["钱袋"].holder_id == "a2"

    # The holder walks away, the item goes with him and is no longer co-located.
    env.place_agent(agent_id="a2", location_id="garden")
    assert "钱袋" not in {e.name for e in env.spatial_for(agent_id="a1").visible_entities}


def test_my_own_held_items_are_visible_to_me() -> None:
    """An agent always knows what he is holding; the hidden flag doesn't hide it from him.

    Hidden items in someone else's hand stay invisible, or this would be X-ray vision.
    """
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    mine = _make_item("letter", "密信", owner_id="a1")
    mine.is_public = False
    theirs = _make_item("token", "暗记", owner_id="a2")
    theirs.is_public = False
    env.register_entity(mine)
    env.register_entity(theirs)

    seen = {e.name: e for e in env.spatial_for(agent_id="a1").visible_entities}
    assert "密信" in seen and seen["密信"].holder_id == "a1"
    assert "暗记" not in seen


def test_a_traveler_still_carries_his_own_things() -> None:
    """An agent in transit is co-located only with himself, so "things here" are what he carries.

    Leaving it empty would make carried items vanish mid-trip and reappear on arrival, and would
    accept an item the perception layer never listed (a prompt/parse mismatch).
    """
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "hall", connections={"garden": 3}))
    env.space.register_place(_make_location("garden", "garden", connections={"hall": 3}))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    env.register_entity(_make_item("blade", "短刀", owner_id="a1"))
    env.register_entity(_make_item("pouch", "钱袋", owner_id="a2"))
    env.register_entity(_make_item("lamp", "灯", location_id="hall"))
    env.place_agent(agent_id="a1", location_id=IN_TRANSIT)

    seen = {e.name: e for e in env.spatial_for(agent_id="a1").visible_entities}
    assert seen.keys() == {"短刀"}
    assert seen["短刀"].holder_id == "a1"


def test_items_present_at_is_ground_plus_co_present_hands() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="garden")
    env.register_entity(_make_item("lamp", "灯", location_id="hall"))
    env.register_entity(_make_item("blade", "短刀", owner_id="a1"))
    env.register_entity(_make_item("pouch", "钱袋", owner_id="a2"))
    env.register_entity(_make_item("scroll", "卷轴", location_id="garden"))
    doomed = _make_item("vase", "花瓶", location_id="hall")
    env.register_entity(doomed)
    env.change_entity_state(EntityStateChange(entity_id="vase", new_state="destroyed", destroyed=True))

    assert {e.name for e in env.items_present_at("hall")} == {"灯", "短刀"}
    assert {e.name for e in env.items_present_at("garden")} == {"钱袋", "卷轴"}


def test_find_item_by_name_does_not_reach_across_the_map() -> None:
    """Items with the same name must not answer across places, or a sword in someone's hand on the
    far side of the map would stand in for the one right here."""
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="garden")
    env.register_entity(_make_item("sword-far", "剑", owner_id="a2"))

    assert env.find_item("剑", "hall") is None
    assert env.find_item("剑", "garden") is not None


def test_feasibility_lets_the_judge_decide_a_contested_grab() -> None:
    """"Are we in the same place" is objective, for rules; "can I take it" is the judge's call (§5).

    The rule layer must not reject "belongs to someone else": taking an item back could then only
    target the person, recorded as a scuffle with no item involved.
    """
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "hall"))
    env.space.register_place(_make_location("garden", "garden"))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    env.register_entity(_make_item("tally", "虎符", owner_id="a2"))

    ok = env.check_physical_feasibility("a1", "tally", "item")
    assert ok.ok and ok.resolved_id == "tally"

    # The holder walks away, so the item is out of reach. The message says only where I am, not
    # where he went (that would be free scouting).
    env.place_agent(agent_id="a2", location_id="garden")
    denied = env.check_physical_feasibility("a1", "tally", "item")
    assert not denied.ok
    assert "hall" in denied.reason and "garden" not in denied.reason

    # What I hold is always within reach.
    env.register_entity(_make_item("blade", "短刀", owner_id="a1"))
    assert env.check_physical_feasibility("a1", "blade", "item").ok


# ---------------------------------------------------------------------------
# change_entity_state — single mutation path
# ---------------------------------------------------------------------------


def test_change_entity_state_updates_state_and_triggers_observation() -> None:
    """Entity state change observations carry to next step's ambient (not current step).

    The acting agent must not read his own change again through ambient (he already knows it via
    ActionResult/_apply_feedback); only onlookers (other agents in the same place) should.
    """
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")  # bystander
    sword = _make_item("sword", "铁剑", location_id="hall")
    env.register_entity(sword)

    env.change_entity_state(
        EntityStateChange(entity_id="sword", new_state="damaged", perception="铁剑被损坏了。"),
        acting_agent_id="a1",
    )
    assert env._entities["sword"].state == "damaged"

    env.begin_step(step=1, world_time=_world_time(1))
    spatial_a2 = env.spatial_for(agent_id="a2")
    assert any("损坏" in ev.content for ev in spatial_a2.ambient_events)
    # the actor doesn't read his own change again through ambient
    spatial_a1 = env.spatial_for(agent_id="a1")
    assert not any("损坏" in ev.content for ev in spatial_a1.ambient_events), (
        "F-fixture-1:acting agent 不应通过 ambient 读到自己的实体变更观察。"
    )


def test_change_entity_state_with_owner_id_moves_to_carried() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    item = _make_item("key", "钥匙", location_id="hall")
    env.register_entity(item)

    env.change_entity_state(
        EntityStateChange(entity_id="key", new_state="held", owner_id="a1"),
    )

    entity = env._entities["key"]
    assert entity.owner_id == "a1"
    assert entity.location_id is None


def test_change_entity_state_with_location_id_drops_item() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    item = _make_item("key", "钥匙", owner_id="a1")
    env.register_entity(item)

    env.change_entity_state(
        EntityStateChange(entity_id="key", new_state="intact", location_id="hall"),
    )

    entity = env._entities["key"]
    assert entity.location_id == "hall"
    assert entity.owner_id is None


def test_a_concealed_item_put_down_becomes_visible_to_everyone_here() -> None:
    """Hiding only makes sense for something carried. An item dropped while not public meets
    neither visibility condition ("in my hand" or "public"), so even the person who dropped it
    can't see it, and nobody can ever pick it up again."""
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    letter = dataclasses.replace(_make_item("letter", "密信", owner_id="a1"), is_public=False)
    env.register_entity(letter)

    def sees(viewer: str) -> bool:
        return any(e.entity_id == "letter" for e in env.spatial_for(agent_id=viewer).visible_entities)

    assert sees("a1") and not sees("a2")          # carried: the holder knows, others can't see it

    env.change_entity_state(EntityStateChange(entity_id="letter", location_id="hall"))

    assert env._entities["letter"].is_public is True
    assert sees("a1") and sees("a2")


def test_a_concealed_item_handed_over_stays_concealed() -> None:
    """Handing an item to someone keeps it carried: the new holder sees it, others still don't.
    Making dropped items public applies only to dropping."""
    env = EnvironmentSystem()
    for aid in ("a1", "a2", "a3"):
        env.place_agent(agent_id=aid, location_id="hall")
    letter = dataclasses.replace(_make_item("letter", "密信", owner_id="a1"), is_public=False)
    env.register_entity(letter)

    env.change_entity_state(EntityStateChange(entity_id="letter", owner_id="a2"))

    assert env._entities["letter"].is_public is False


def test_change_entity_state_returns_false_for_unknown_entity() -> None:
    env = EnvironmentSystem()

    result = env.change_entity_state(EntityStateChange(entity_id="nonexistent", new_state="broken"))

    assert result is False


# ---------------------------------------------------------------------------
# snapshot_state includes entity_states
# ---------------------------------------------------------------------------


def test_snapshot_state_includes_entity_states() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    item = _make_item("torch", "火炬", location_id="hall", state="lit")
    env.register_entity(item)

    state = env.snapshot_state()

    assert "entity_states" in state
    assert "torch" in state["entity_states"]
    assert state["entity_states"]["torch"]["name"] == "火炬"
    assert state["entity_states"]["torch"]["state"] == "lit"


def test_carried_observations_survive_a_snapshot_round_trip() -> None:
    """What onlookers are due to perceive next step is saved with the step, or a restore would
    drop that beat."""
    from core.serialization import dump_json
    import json

    env = EnvironmentSystem()
    env.record_carry_observation(
        location_id="hall", observation="有人在殿上拔刀相向。", strength=0.6, actor_ids=("a1",),
    )

    restored = EnvironmentSystem()
    restored.restore_state(json.loads(dump_json(env.snapshot_state())))
    restored.begin_step(step=2, world_time=_world_time(2))
    env.begin_step(step=2, world_time=_world_time(2))

    assert restored._step_annotations == env._step_annotations


def test_snapshot_state_entity_states_excludes_locations() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="a1", location_id="hall")

    state = env.snapshot_state()

    assert "hall" not in state.get("entity_states", {})


# ---------------------------------------------------------------------------
# Movement and basic state plumbing
# ---------------------------------------------------------------------------


def test_place_agent_records_location_in_spatial() -> None:
    env = EnvironmentSystem()

    env.place_agent(agent_id="agent-1", location_id="garden")
    spatial = env.spatial_for(agent_id="agent-1")

    assert spatial.location_id == "garden"


def test_move_agent_updates_location_in_spatial() -> None:
    env = EnvironmentSystem()

    env.place_agent(agent_id="agent-1", location_id="garden")
    env.move_body(body_id="agent-1", location_id="palace")
    spatial = env.spatial_for(agent_id="agent-1")

    assert spatial.location_id == "palace"


def test_snapshot_state_carries_where_each_body_stands() -> None:
    """Occupancy is stored once. There is no inverted index (who is where): nothing reads it, and it
    would drift from the primary copy."""
    env = EnvironmentSystem()
    env.place_agent(agent_id="agent-1", location_id="hall")
    env.place_agent(agent_id="agent-2", location_id="garden")

    state = env.snapshot_state()

    assert state["body_locations"] == {"agent-1": "hall", "agent-2": "garden"}
    assert "location_occupancy" not in state


def test_environment_loads_world_config_locations() -> None:
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))

    assert env.space.get("donggong").name == "东宫"
    # zhuque_gate -> mingde_gate is about an hour's walk along the main avenue, a direct edge.
    assert env.space.shortest_path("zhuque_gate", "mingde_gate") == ["zhuque_gate", "mingde_gate"]
    assert 45 * 60 <= env.space.reachable_from("zhuque_gate")["mingde_gate"] <= 75 * 60


def test_tiled_travel_cost_is_walking_time_not_hops() -> None:
    """A route costs the sum of its walks; no hop is rounded up to a clock tick on the map."""
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    path = env.space.shortest_path("donggong", "taiji_palace")
    # Not directly connected: the gate lies between, so it stays on the route.
    assert path == ["donggong", "xuanwu_gate", "taiji_palace"]
    total = env.space.reachable_from("donggong")["taiji_palace"]
    assert total == sum(env.space.edge_seconds(a, b) for a, b in zip(path, path[1:]))
    # Two hops of palace grounds take minutes, not two steps of any clock.
    assert total < 2 * 3600


# ---------------------------------------------------------------------------
# spatial_for() — reachable_locations (multi-hop shortest-path horizon)
# ---------------------------------------------------------------------------


def test_spatial_for_populates_reachable_locations_from_connections() -> None:
    env = _make_connected_env()

    spatial = env.spatial_for(agent_id="agent-1", step=1, world_time="辰时")

    assert [rl.location_id for rl in spatial.reachable_locations] == ["garden"]
    assert spatial.reachable_locations[0].travel_seconds == 3600
    assert spatial.reachable_locations[0].view.name == "garden"


def test_spatial_for_reachable_locations_empty_when_location_unknown() -> None:
    env = EnvironmentSystem()

    spatial = env.spatial_for(agent_id="ghost")

    assert spatial.reachable_locations == []


def test_spatial_for_reachable_spans_beyond_direct_neighbors() -> None:
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-1", location_id="zhuque_gate")

    spatial = env.spatial_for(agent_id="agent-1", step=1, world_time="step=0001")

    reachable_ids = {rl.location_id for rl in spatial.reachable_locations}
    # Direct neighbors are still reachable at their edge weight...
    direct = {"huangcheng", "west_market", "jianfu_temple", "mingde_gate"}
    assert direct <= reachable_ids
    # ...but reachability now extends to the whole connected component — strictly
    # more than the 4 immediate neighbors, and never the agent's own location.
    assert len(spatial.reachable_locations) > len(direct)
    assert "zhuque_gate" not in reachable_ids
    # The list is ordered by ascending travel time.
    ordered = [rl.travel_seconds for rl in spatial.reachable_locations]
    assert ordered == sorted(ordered)


# ---------------------------------------------------------------------------
# SpaceManager.shortest_path / reachable_from caching
# ---------------------------------------------------------------------------


def test_shortest_path_returns_waypoint_sequence() -> None:
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    path = env.space.shortest_path("donggong", "zhuque_gate")

    assert path is not None
    assert path[0] == "donggong" and path[-1] == "zhuque_gate"
    # consecutive nodes are real edges; summed edge weights == total travel time.
    total = sum(env.space.edge_seconds(a, b) for a, b in zip(path, path[1:]))
    assert total == env.space.reachable_from("donggong")["zhuque_gate"]


def test_shortest_path_none_when_unreachable() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("island_a", "甲", connections={}))
    env.space.register_place(_make_location("island_b", "乙", connections={}))

    assert env.space.shortest_path("island_a", "island_b") is None


def test_reachable_from_is_cached_until_topology_changes() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("a", "A", connections={"b": 1}))
    env.space.register_place(_make_location("b", "B", connections={"a": 1}))

    env.space.reachable_from("a")
    assert "a" in env.space._path_cache            # memoized after first query
    assert env.space.reachable_from("a") is not None

    # Registering / re-registering locations invalidates the memo (topology changed).
    env.space.register_place(_make_location("c", "C", connections={"b": 1}))
    env.space.register_place(_make_location("b", "B", connections={"a": 1, "c": 1}))  # add b→c back-edge
    assert env.space._path_cache == {}
    assert env.space.reachable_from("a")["c"] == 2  # new node now reachable via b


# ---------------------------------------------------------------------------
# check_move_feasibility
# ---------------------------------------------------------------------------


def test_check_move_feasibility_connected_locations() -> None:
    env = _make_connected_env(seconds=3600)

    result = env.check_move_feasibility("agent-1", "garden")

    assert result.ok is True
    assert result.travel_seconds == 3600
    assert result.resolved_id == "garden"


def test_check_move_feasibility_by_location_name() -> None:
    env = _make_connected_env(seconds=3600)

    result = env.check_move_feasibility("agent-1", "garden")

    assert result.ok is True
    assert result.resolved_id == "garden"


def test_check_move_feasibility_unknown_destination() -> None:
    env = _make_connected_env()

    result = env.check_move_feasibility("agent-1", "throne_room")

    assert result.ok is False
    assert "找不到地点" in result.reason
    assert "throne_room" not in result.reason   # the reason reaches memory prose: no ids


def test_check_move_feasibility_multi_hop_is_reachable() -> None:
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-1", location_id="donggong")

    # donggong does not directly border zhuque_gate, but a path exists across the map, so a
    # multi-hop move is feasible: feasibility is path-based, not one-hop.
    result = env.check_move_feasibility("agent-1", "朱雀门")

    assert result.ok is True
    # Summed edge weights along the shortest path, longer than any single edge out of donggong.
    assert result.travel_seconds > max(env.space.get("donggong").connections.values())


def test_check_move_feasibility_no_path_between_islands() -> None:
    # Two disconnected locations (no connections either way) → genuinely unreachable.
    env = EnvironmentSystem()
    env.space.register_place(_make_location("island_a", "孤岛甲", connections={}))
    env.space.register_place(_make_location("island_b", "孤岛乙", connections={}))
    env.place_agent(agent_id="agent-1", location_id="island_a")

    result = env.check_move_feasibility("agent-1", "island_b")

    assert result.ok is False
    assert "没有通路" in result.reason


def test_check_move_feasibility_capacity_limit() -> None:
    env = _make_connected_env()
    # Capacity is fixed at world build. Place is frozen and can't change at runtime; that is the
    # contract.
    env.space.register_place(_make_location("garden", "garden", connections={"hall": 1}, capacity=1))
    env.place_agent(agent_id="occupant", location_id="garden")

    result = env.check_move_feasibility("agent-1", "garden")

    assert result.ok is False
    assert "人满为患" in result.reason


def test_check_move_feasibility_returns_world_config_steps() -> None:
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-1", location_id="zhuque_gate")

    result = env.check_move_feasibility("agent-1", "太极宫")

    assert result.ok is True
    assert result.path == ("zhuque_gate", "huangcheng", "chengtian_gate", "taiji_palace")
    assert result.travel_seconds == sum(
        env.space.edge_seconds(a, b) for a, b in zip(result.path, result.path[1:])
    )


# ---------------------------------------------------------------------------
# check_talk_feasibility
# ---------------------------------------------------------------------------


def test_check_talk_feasibility_colocated() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="agent-1", location_id="hall")
    env.place_agent(agent_id="agent-2", location_id="hall")

    result = env.check_talk_feasibility("agent-1", ["agent-2"])

    assert result.ok is True
    assert result.resolved_id == "hall"


def test_check_talk_feasibility_different_locations() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.space.register_place(_make_location("garden", "后花园"))
    env.place_agent(agent_id="agent-1", location_id="hall")
    env.place_agent(agent_id="agent-2", location_id="garden")

    result = env.check_talk_feasibility("agent-1", ["agent-2"], target_label="李二")

    assert result.ok is False
    # The reason names the target and the actor's own location; it doesn't reveal where the target
    # actually is and contains no ids.
    assert "李二" in result.reason and "大殿" in result.reason
    assert "后花园" not in result.reason
    assert "agent-2" not in result.reason


def test_check_talk_feasibility_no_target() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="agent-1", location_id="hall")

    result = env.check_talk_feasibility("agent-1", [])

    assert result.ok is False
    assert "不知道对话目标" in result.reason


def test_check_talk_feasibility_succeeds_for_co_located_agents() -> None:
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-1", location_id="donggong")
    env.place_agent(agent_id="agent-2", location_id="donggong")

    result = env.check_talk_feasibility("agent-1", ["agent-2"])
    assert result.ok is True


# ---------------------------------------------------------------------------
# MOVE distance via check_move_feasibility
#
# MOVE is the only action whose duration is world-physics-determined; that
# duration lives in Location.connections and surfaces via FeasibilityResult.steps.
# All other action durations are LLM-driven (action.estimated_steps) and never
# pass through environment.
# ---------------------------------------------------------------------------


def test_check_move_feasibility_returns_connections_distance() -> None:
    """MOVE distance is read from Location.connections via SpaceManager."""
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-1", location_id="zhuque_gate")

    # Neighbor -> that edge's walking time.
    result = env.check_move_feasibility("agent-1", "west_market")
    assert result.ok is True
    assert result.travel_seconds == env.space.edge_seconds("zhuque_gate", "west_market")
    assert result.resolved_id == "west_market"

    # Three places away -> cost adds up per hop, past the first one.
    env.place_agent(agent_id="agent-2", location_id="zhuque_gate")
    far = env.check_move_feasibility("agent-2", "太极宫")
    assert far.ok is True
    assert far.travel_seconds > env.space.edge_seconds("zhuque_gate", far.path[1])


# ---------------------------------------------------------------------------
# ambient_events — no structural tags
# ---------------------------------------------------------------------------


def test_spatial_for_ambient_events_contain_no_structural_tags() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    spatial = env.spatial_for(agent_id="a1", step=1, world_time="辰时")

    for event in spatial.ambient_events:
        assert not event.content.startswith("location:"), f"structural tag leaked: {event!r}"
        assert not event.content.startswith("occupancy:"), f"structural tag leaked: {event!r}"
        assert not event.content.startswith("description:"), f"structural tag leaked: {event!r}"


def test_spatial_for_ambient_events_empty_without_observations() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")

    spatial = env.spatial_for(agent_id="a1", step=1, world_time="辰时")

    assert spatial.ambient_events == []


def test_spatial_for_ambient_events_contain_carried_observations() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    env.record_carry_observation(
        location_id="hall", observation="A fire started near the gate."
    )
    env.begin_step(step=1, world_time=_world_time(1))

    spatial = env.spatial_for(agent_id="a1", step=1, world_time="辰时")

    assert any("fire" in e.content for e in spatial.ambient_events)


def test_change_entity_state_ignores_owner_id_for_non_takeable_entity() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    landmark = WorldEntity(
        entity_id="gate_sign",
        name="城门告示",
        entity_type=WorldEntityType.LANDMARK,
        presence_ref="hall",
        is_takeable=False,
        state="intact",
    )
    env.register_entity(landmark)

    env.change_entity_state(
        EntityStateChange(entity_id="gate_sign", new_state="defaced", owner_id="a1"),
        acting_agent_id="a1",
    )

    entity = env.find_item("gate_sign")
    assert entity is not None
    assert entity.state == "defaced"
    assert entity.owner_id is None          # is_takeable=False → owner assignment ignored
    assert entity.location_id == "hall"     # location unchanged


def test_check_physical_feasibility_landmark_returns_infeasible_when_not_at_location() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("garden", "后花园"))
    env.place_agent(agent_id="a1", location_id="garden")
    landmark = WorldEntity(
        entity_id="notice_board",
        name="告示牌",
        entity_type=WorldEntityType.LANDMARK,
        presence_ref="hall",
        is_takeable=False,
        state="active",
    )
    env.register_entity(landmark)

    result = env.check_physical_feasibility("a1", "notice_board", "landmark")

    assert not result.ok
    # The message names the actor's location; the landmark's actual location isn't revealed.
    assert "告示牌不在后花园" in result.reason
    assert "hall" not in result.reason


def test_check_physical_feasibility_agent_reason_names_own_location() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.space.register_place(_make_location("garden", "后花园"))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="b1", location_id="garden")

    result = env.check_physical_feasibility("a1", "b1", "agent")

    assert not result.ok
    assert "大殿" in result.reason        # the actor's own location appears
    assert "后花园" not in result.reason  # the target's actual location doesn't
    assert "b1" not in result.reason      # no ids


def test_check_physical_feasibility_landmark_succeeds_when_at_location() -> None:
    env = EnvironmentSystem()
    env.place_agent(agent_id="a1", location_id="hall")
    landmark = WorldEntity(
        entity_id="notice_board",
        name="告示牌",
        entity_type=WorldEntityType.LANDMARK,
        presence_ref="hall",
        is_takeable=False,
        state="active",
    )
    env.register_entity(landmark)

    result = env.check_physical_feasibility("a1", "notice_board", "landmark")

    assert result.ok
    assert result.resolved_id == "notice_board"


# ---------------------------------------------------------------------------
# restore_state() — inverse of snapshot_state() for mutable entity state
# ---------------------------------------------------------------------------

def test_restore_state_reapplies_item_state_location_and_owner() -> None:
    """A freshly seeded environment recovers item drift from a captured snapshot."""
    # Live environment: an item gets picked up and damaged during the run.
    live = EnvironmentSystem()
    live.space.register_place(_make_location("hall", "hall"))
    live.register_entity(_make_item("sword", "sword", location_id="hall"))
    live.place_agent(agent_id="a1", location_id="hall")
    live.change_entity_state(
        EntityStateChange(entity_id="sword", new_state="bloodied", owner_id="a1"),
    )
    captured = live.snapshot_state()

    # Fresh environment rebuilt from step-0 seeds: sword is intact, on the floor.
    fresh = EnvironmentSystem()
    fresh.space.register_place(_make_location("hall", "hall"))
    fresh.register_entity(_make_item("sword", "sword", location_id="hall"))

    fresh.restore_state(captured)

    restored = fresh.find_item("sword")
    assert restored is not None
    assert restored.state == "bloodied"
    assert restored.owner_id == "a1"
    assert restored.location_id is None
    # Carried item no longer sits at its seed location.
    assert all(e.entity_id != "sword" for e in fresh.get_items_at("hall"))
    assert [e.entity_id for e in fresh.get_items_of("a1")] == ["sword"]


def test_restore_state_reapplies_a_seeded_items_privacy() -> None:
    """A runtime change to a seed item's is_public must be restored too. Otherwise a hidden private
    item becomes visible to everyone after restore (the same reason snapshot_state writes this
    field)."""
    live = EnvironmentSystem()
    live.space.register_place(_make_location("hall", "hall"))
    live.register_entity(_make_item("letter", "letter", location_id="hall"))
    live.find_item("letter").is_public = False
    captured = live.snapshot_state()

    fresh = EnvironmentSystem()
    fresh.space.register_place(_make_location("hall", "hall"))
    fresh.register_entity(_make_item("letter", "letter", location_id="hall"))
    assert fresh.find_item("letter").is_public is True   # seeds start public

    fresh.restore_state(captured)

    assert fresh.find_item("letter").is_public is False


def test_restore_state_reregisters_entity_created_mid_run() -> None:
    """An item that did not exist among step-0 seeds is re-registered on restore."""
    fresh = EnvironmentSystem()
    fresh.space.register_place(_make_location("hall", "hall"))

    fresh.restore_state(
        {
            "entity_states": {
                "letter": {
                    "name": "letter",
                    "entity_type": "item",
                    "state": "sealed",
                    "presence": "at_location",
                    "presence_ref": "hall",
                }
            }
        }
    )

    letter = fresh.find_item("letter")
    assert letter is not None
    assert letter.name == "letter"
    assert letter.state == "sealed"
    assert [e.entity_id for e in fresh.get_items_at("hall")] == ["letter"]


def test_restore_state_ignores_missing_or_malformed_payload() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "hall"))
    # Must not raise on absent or wrongly-typed entity_states.
    env.restore_state({})
    env.restore_state({"entity_states": "not-a-mapping"})


# --------------------------------------------------------------------------- #
# A place that is not open to all comers


def test_a_gated_place_says_so_in_what_a_person_perceives() -> None:
    """Access restriction is a visible fact, so it reaches perception along with the place
    description. It is not a gate that stops anyone.

    Map authors mark restricted places with `is_public: false` in the template, and this checks
    that the flag reaches the agent.
    """
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="palace", name="太极宫", description="宫城正殿",
        connections={"market": 1}, is_public=False,
    ))
    env.space.register_place(_make_location("market", "西市", connections={"palace": 1}))
    env.place_agent(agent_id="a1", location_id="market")

    # Assert that the fact is stated, not the exact wording: the wording may change, and this test
    # guards whether the fact gets through.
    assert render_location(env.location_view("palace")).startswith("太极宫——宫城正殿")
    assert "不对外开放" in render_location(env.location_view("palace"))
    assert "不对外开放" not in render_location(env.location_view("market"))

    # It reaches the decision-maker: every reachable-place entry carries the flag.
    reachable = {r.location_id: r.view for r in env.spatial_for(agent_id="a1").reachable_locations}
    assert reachable["palace"].is_public is False


def test_a_gated_place_is_not_a_locked_door() -> None:
    """Whether someone may enter depends on who they are and why. That is a judgment, not a
    threshold, so the engine doesn't block it.

    Blocking it would lock the emperor out of his own palace.
    """
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="palace", name="太极宫", connections={"market": 1}, is_public=False,
    ))
    env.space.register_place(_make_location("market", "西市", connections={"palace": 1}))
    env.place_agent(agent_id="a1", location_id="market")

    assert env.check_move_feasibility(agent_id="a1", target_location="palace").ok is True


# --------------------------------------------------------------------------- #
# Reachable items: ordering and limit (the binding menu)


def _stocked_room() -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    env.place_agent(agent_id="other", location_id="hall")
    # Registered in the reverse of the expected order: ground item first, mine last.
    env.register_entity(_make_item("ground", "地上那件", location_id="hall"))
    env.register_entity(_make_item("theirs", "他手上那件", owner_id="other"))
    env.register_entity(_make_item("mine", "我手上那件", owner_id="me"))
    return env


def test_reachable_things_are_ordered_by_what_it_costs_to_use_them() -> None:
    """Mine -> lying here -> in someone else's hand. Registration order differs, so the list really
    is sorted."""
    env = _stocked_room()

    assert [e.name for e in env.spatial_for(agent_id="me").visible_entities] == [
        "我手上那件", "地上那件", "他手上那件",
    ]


def test_the_same_things_keep_the_same_order() -> None:
    """Indices must be stable. If an item is #1 one step and #3 the next, the agent has to relearn
    the world."""
    env = _stocked_room()

    first = [e.entity_id for e in env.spatial_for(agent_id="me").visible_entities]
    second = [e.entity_id for e in env.spatial_for(agent_id="me").visible_entities]
    assert first == second


def test_the_menu_has_a_ceiling() -> None:
    """The list is a binding menu. Making it too long offers a fake action space, since items near
    the end are never chosen."""
    from engine.environment import MAX_VISIBLE_ENTITIES

    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    for i in range(MAX_VISIBLE_ENTITIES + 5):
        env.register_entity(_make_item(f"g{i}", f"第{i}件", location_id="hall"))

    assert len(env.spatial_for(agent_id="me").visible_entities) == MAX_VISIBLE_ENTITIES


def test_a_crowded_room_never_hides_what_i_am_holding() -> None:
    """Truncation drops items in others' hands first and never what I hold: that is the only thing
    I can use without anyone's consent."""
    from engine.environment import MAX_VISIBLE_ENTITIES

    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    for i in range(MAX_VISIBLE_ENTITIES + 5):
        env.register_entity(_make_item(f"g{i}", f"第{i}件", location_id="hall"))
    env.register_entity(_make_item("mine", "我手上那件", owner_id="me"))

    names = [e.name for e in env.spatial_for(agent_id="me").visible_entities]
    assert names[0] == "我手上那件"


def test_a_thing_just_made_outranks_the_ones_that_were_always_there() -> None:
    """Without this tier the order falls back to registration order, and starting seeds always
    outrank items made during the run."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    env.register_entity(_make_item("seed", "开局就有的", location_id="hall"))   # created_step=0
    env.begin_step(step=9, world_time=_world_time(9))
    env.spawn_entity(EntitySpawn(name="刚做出来的"), ground=env.get_body_location("me"), actor_id="me")

    assert [e.name for e in env.spatial_for(agent_id="me").visible_entities] == [
        "刚做出来的", "开局就有的",
    ]


def test_someone_elses_new_thing_still_ranks_below_my_old_one() -> None:
    """Cost first, then age: the ordering answers "what can I do with it", not "what stands out
    most"."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    env.place_agent(agent_id="other", location_id="hall")
    env.register_entity(_make_item("mine", "我的旧物", owner_id="me"))         # created_step=0
    env.begin_step(step=9, world_time=_world_time(9))
    env.spawn_entity(EntitySpawn(name="他刚拿到的", holder_id="other"), ground=env.get_body_location("other"), actor_id="other")

    assert [e.name for e in env.spatial_for(agent_id="me").visible_entities] == [
        "我的旧物", "他刚拿到的",
    ]


def test_when_it_appeared_survives_a_restore() -> None:
    """The creation step is a recorded fact, not a derived value. Losing it on restore changes the
    list order."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    env.begin_step(step=9, world_time=_world_time(9))
    env.spawn_entity(EntitySpawn(name="刚做出来的"), ground=env.get_body_location("me"), actor_id="me")

    fresh = EnvironmentSystem()
    fresh.space.register_place(_make_location("hall", "大殿"))
    fresh.restore_state(env.snapshot_state())

    assert fresh.all_live_entities()[0].created_step == 9


def test_the_judge_reads_the_same_scene_the_actor_saw() -> None:
    """If two places ordered the list separately, the judge could rule on something the actor
    never saw. The order is defined in one place only."""
    from engine.directory import LiveWorldDirectory
    from engine.scene import SceneVisibility, assemble_scene_context

    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    env.register_entity(_make_item("seed", "开局就有的", location_id="hall"))
    env.begin_step(step=9, world_time=_world_time(9))
    env.spawn_entity(EntitySpawn(name="刚做出来的", holder_id="me"), ground=env.get_body_location("me"), actor_id="me")

    seen = [e.name for e in env.spatial_for(agent_id="me").visible_entities]
    scene = assemble_scene_context(
        "me", environment=env, directory=LiveWorldDirectory.from_agents({}, env),
        include_header=False, voice=SituationVoice.FIRST,
        visibility=SceneVisibility.OWN_EYES,
    ).text
    # The list order matches the order in the scene text.
    assert seen == ["刚做出来的", "开局就有的"]
    assert scene.index("刚做出来的") < scene.index("开局就有的")


def test_empty_hands_are_stated_not_omitted() -> None:
    """When listed separately, empty hands must be stated: what is missing is a key fact for
    adjudication. When not listed separately, that line doesn't exist at all."""
    from engine.directory import LiveWorldDirectory
    from core.prompts import SituationVoice
    from engine.scene import SceneVisibility, assemble_scene_context

    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")

    def scene(split):
        return assemble_scene_context(
            "me", environment=env, directory=LiveWorldDirectory.from_agents({}, env),
            include_header=False, voice=SituationVoice.FIRST,
            visibility=SceneVisibility.OWN_EYES, split_own_entities=split,
        ).text

    assert "- 我手上的东西：无" in scene(True)
    assert "我手上的东西" not in scene(False)


def test_first_person_scene_marks_his_own_things_as_his_to_hand_over() -> None:
    """First-person scene: items in his own hand are marked as handable ("可交出"), whether they
    appear in the item lines or in a separate line for what he holds. The god's-eye view (the
    third-party judge) has no "self", so it marks only the holder, never "可取"."""
    from engine.directory import LiveWorldDirectory
    from core.prompts import SituationVoice
    from engine.scene import SceneVisibility, assemble_scene_context

    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="me", location_id="hall")
    env.register_entity(_make_item("seal", "印信", owner_id="me"))

    def scene(**kw):
        return assemble_scene_context(
            "me", environment=env, directory=LiveWorldDirectory.from_agents({}, env),
            include_header=False, **kw,
        ).text

    first = dict(voice=SituationVoice.FIRST, visibility=SceneVisibility.OWN_EYES)
    for text in (scene(**first), scene(**first, split_own_entities=True)):
        assert "印信（可交出）" in text or "印信（由我持有；可交出）" in text
        assert "可取" not in text
    god = scene(voice=SituationVoice.THIRD, visibility=SceneVisibility.GOD)
    assert "印信（由某人持有）" in god and "可取" not in god


def test_the_judge_may_read_more_than_the_menu_offers() -> None:
    """Truncation costs are asymmetric: an actor missing an item just loses an option, but a judge
    missing one rules on an incomplete scene."""
    from engine.environment import MAX_SCENE_ENTITIES, MAX_VISIBLE_ENTITIES

    assert MAX_SCENE_ENTITIES > MAX_VISIBLE_ENTITIES

def test_who_is_standing_here_has_exactly_one_answer() -> None:
    """Occupancy has one source of truth, so every path asking "who is here" gives the same answer.

    The place object must not keep its own presence list: two registries drift, and then
    ``spatial_for`` and ``agents_at`` disagree, so TALK admission, message delivery and overhearing
    find no one.
    """
    env = EnvironmentSystem()                      # no places registered at all
    env.place_agent(agent_id="a1", location_id="nowhere")
    env.place_agent(agent_id="a2", location_id="nowhere")

    assert env.bodies_at("nowhere") == ["a1", "a2"]
    assert env.agents_at("nowhere") == ["a1", "a2"]
    assert set(env.spatial_for(agent_id="a1").visible_agents) == {"a2"}


def test_a_place_does_not_remember_who_is_standing_on_it() -> None:
    """Occupancy is per-step runtime state and must not live on ``WorldEntity``, an identity object.

    That would be a second source of truth, and ``dataclasses.asdict`` would bake it into the
    persisted world config assets, producing ghost occupants on restore.
    """
    names = {f.name for f in dataclasses.fields(WorldEntity)}
    assert "body_ids" not in names, "在场名单不属于地点,问 EnvironmentSystem.bodies_at"
    assert "current_events" not in names


# --------------------------------------------------------------------------- #
# Places and things are separate types; keep them apart


def test_a_place_and_a_thing_share_nothing_but_a_name() -> None:
    """They share only the identity header. Merging them into one type would give each side fields
    that mean nothing to it, and those fields are exactly how you'd end up "destroying a palace"
    or "creating a place with no connections".
    """
    place_fields = {f.name for f in dataclasses.fields(Place)}
    entity_fields = {f.name for f in dataclasses.fields(WorldEntity)}

    assert place_fields & entity_fields == {"name", "description", "is_public"}
    for placement in ("presence", "presence_ref", "is_takeable", "created_step"):
        assert placement not in place_fields, f"地点没有 {placement}"
    for spatial in ("connections", "capacity"):
        assert spatial not in entity_fields, f"物没有 {spatial}"


def test_a_thing_can_never_be_a_place() -> None:
    """The enum has no ``LOCATION`` value. The guard lives in the type, not at call sites."""
    assert {t.value for t in WorldEntityType} == {"item", "landmark"}


def test_making_a_thing_cannot_make_a_place() -> None:
    """``spawn_entity`` can't create a place: a declared kind of "location" is rejected rather than
    quietly creating an unreachable island."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.spawn_entity(
        EntitySpawn(name="密室", entity_type="location"), ground=env.get_body_location("a1"), actor_id="a1"
    ) is False
    assert env.space.all_place_ids() == ["hall"]
    assert env.all_live_entities() == []


def test_a_place_cannot_be_destroyed_through_the_thing_channel() -> None:
    """A half-destroyed place can't exist: places aren't in the item registry, so they can't be
    changed through it."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("palace", "太极宫", connections={"market": 1}))
    env.space.register_place(_make_location("market", "西市", connections={"palace": 1}))
    env.place_agent(agent_id="a1", location_id="market")

    assert env.change_entity_state(
        EntityStateChange(entity_id="palace", destroyed=True, new_name="废墟")
    ) is False
    assert env.space.name_of("palace") == "太极宫"
    assert env.check_move_feasibility("a1", "太极宫").ok is True


def test_a_place_says_its_name_once() -> None:
    """The description is just the description, without the name prepended.

    The renderer already produces "name——description". Including the name again here would have
    agents read "玄武门——玄武门 - 宫城北门" every step, in the highest-traffic prompt in the system.
    """
    env = EnvironmentSystem()
    env.space.register_place(Place(place_id="gate", name="玄武门", description="宫城北门"))

    view = env.location_view("gate")

    assert view.description == "宫城北门"
    assert not view.description.startswith(view.name)
    assert render_location(view) == "玄武门——宫城北门"


def test_the_world_takes_its_space_once_and_never_again() -> None:
    """Space is fixed at world build: in production code ``register_place`` is called only during
    world building.

    Adding places at runtime would invalidate the cached shortest paths and shift the indices in
    the reachable-places list, which is the list ``destination_index`` binds to.
    """
    import ast
    import subprocess

    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    sources = subprocess.run(
        ["git", "ls-files", "agent/*.py", "core/*.py", "engine/*.py", "world/*.py",
         "worlds/*.py", "interaction/*.py", "providers/*.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    sources = [rel for rel in sources if (pathlib.Path(repo) / rel).exists()]

    callers: list[str] = []
    for rel in sources:
        tree = ast.parse((pathlib.Path(repo) / rel).read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "register_place":
                continue
            enclosing = next(
                (n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.lineno <= node.lineno <= (n.end_lineno or n.lineno)),
                "<module>",
            )
            callers.append(f"{rel}:{enclosing}")

    assert callers == ["engine/environment.py:_load_places"], callers


def test_reach_does_not_turn_on_what_kind_of_thing_it_is() -> None:
    """Whether something is within reach depends only on whether it is here, for every kind,
    including kinds that don't exist yet.

    This guards one thing: there must be no unconditional pass for unrecognized kinds. With one,
    adding a new kind would silently put things on the other side of the world within reach.
    """
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.space.register_place(_make_location("far", "远处"))
    env.place_agent(agent_id="a1", location_id="hall")
    env.register_entity(WorldEntity(
        entity_id="car", name="马车", entity_type=WorldEntityType.ITEM, presence_ref="far",
    ))

    for kind in ("item", "landmark", "object", "a-kind-that-does-not-exist-yet", None):
        result = env.check_physical_feasibility("a1", "car", kind)
        assert result.ok is False, kind
        assert "马车不在大殿" == result.reason, kind


def test_a_thing_in_the_hand_of_someone_standing_here_is_within_reach() -> None:
    """In the hand of someone here = within reach. Whether it can be taken is for the judge to
    decide, not the rules."""
    env = EnvironmentSystem()
    env.space.register_place(_make_location("hall", "大殿"))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    env.register_entity(_make_item("blade", "短刀", owner_id="a2"))

    for kind in ("item", "object", "a-kind-that-does-not-exist-yet"):
        result = env.check_physical_feasibility("a1", "blade", kind)
        assert (result.ok, result.resolved_id) == (True, "blade"), kind


def test_whether_a_kind_can_be_carried_has_exactly_one_answer() -> None:
    """Only the enum answers whether a kind of thing can be taken.

    Don't hand-write this check at call sites (seeding, spawning, restore defaults): when a new
    carryable kind is added, missing one copy silently makes it untakeable, and PHYSICAL's judge
    prompt stops offering ``seize``.
    """
    import ast
    import subprocess

    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    sources = subprocess.run(
        ["git", "ls-files", "agent/*.py", "core/*.py", "engine/*.py", "world/*.py",
         "worlds/*.py", "interaction/*.py", "providers/*.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    sources = [rel for rel in sources if (pathlib.Path(repo) / rel).exists()]

    offenders: list[str] = []
    for rel in sources:
        for node in ast.walk(ast.parse((pathlib.Path(repo) / rel).read_text())):
            if not isinstance(node, ast.Compare):
                continue
            rendered = ast.unparse(node)
            if "WorldEntityType.ITEM" in rendered and "entity_type" in rendered:
                offenders.append(f"{rel}: {rendered}")
    assert offenders == [], offenders

    # And the enum answers correctly, including the conservative default for unknown kinds.
    assert WorldEntityType.ITEM.is_takeable is True
    assert WorldEntityType.LANDMARK.is_takeable is False
    assert WorldEntitySeed(name="车", entity_type="a-kind-that-does-not-exist-yet").is_takeable is False
