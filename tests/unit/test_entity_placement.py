"""Discriminated entity placement: at-location / held / destroyed lifecycle.

Placement is a single ``presence`` axis, not two orthogonal ``location_id`` /
``owner_id`` coordinates, and destruction is a modelled terminal:

- pick-up binds an item to its holder (location cleared), drop reverses it;
- destroy removes an item from the live world (unperceivable, unfindable,
  unpickable) while keeping a tombstone for name resolution + snapshot;
- the snapshot tombstone stops restore from resurrecting a destroyed seed;
- ``EntityStateChange`` destroy wins over pick-up/drop and propagates perception;
- nothing outside ``EnvironmentSystem`` writes an entity's fields directly.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from core.interfaces.action import EntitySpawn, EntityStateChange
from engine.clock import WorldTime, WorldTimeConfig
from engine.environment import EnvironmentSystem
from world.models import EntityPresence, WorldEntity, WorldEntityType
from core.interfaces.place import Place


def _location(entity_id: str, name: str) -> Place:
    return Place(
        place_id=entity_id, name=name, )


def _item(entity_id: str, name: str, location_id: str, *, takeable: bool = True) -> WorldEntity:
    return WorldEntity(
        entity_id=entity_id, name=name, entity_type=WorldEntityType.ITEM,
        presence_ref=location_id, is_takeable=takeable,
    )


def _env_with_hall() -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(_location("hall", "大殿"))
    env.begin_step(step=1, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    return env


# --------------------------------------------------------------------------- #
# Model — derived placement views


def test_default_presence_is_at_location() -> None:
    item = _item("sword", "剑", "hall")
    assert item.presence is EntityPresence.AT_LOCATION
    assert item.location_id == "hall"
    assert item.owner_id is None
    assert item.is_destroyed is False


# --------------------------------------------------------------------------- #
# Pick-up / drop


def test_pickup_binds_to_holder_and_clears_location() -> None:
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall"))
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.change_entity_state(EntityStateChange(entity_id="sword", new_state="intact", owner_id="a1"))
    sword = env.get_entity("sword")
    assert sword.presence is EntityPresence.HELD
    assert sword.owner_id == "a1"
    assert sword.location_id is None
    assert [e.entity_id for e in env.get_items_of("a1")] == ["sword"]
    # A held item is not "at" the location any more.
    assert env.get_items_at("hall") == []


def test_drop_returns_item_to_location() -> None:
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall"))
    env.change_entity_state(EntityStateChange(entity_id="sword", new_state="intact", owner_id="a1"))

    env.change_entity_state(EntityStateChange(entity_id="sword", new_state="intact", location_id="hall"))
    sword = env.get_entity("sword")
    assert sword.presence is EntityPresence.AT_LOCATION
    assert sword.owner_id is None and sword.location_id == "hall"
    assert [e.entity_id for e in env.get_items_at("hall")] == ["sword"]


# --------------------------------------------------------------------------- #
# Destruction terminal


def test_destroy_removes_from_live_world_but_keeps_tombstone() -> None:
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall", takeable=False))
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.change_entity_state(
        EntityStateChange(entity_id="sword", new_state="destroyed", destroyed=True)
    )
    sword = env.get_entity("sword")
    assert sword.is_destroyed and sword.presence is EntityPresence.DESTROYED

    # Gone from every live-world query channel: perception, location listing, find.
    env.begin_step(step=2, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    spatial = env.spatial_for(agent_id="a1")
    assert all(v.entity_id != "sword" for v in spatial.visible_entities)
    assert env.get_items_at("hall") == []
    assert env.find_item("sword") is None
    assert env.find_item("剑", "hall") is None
    # Tombstone stays for name resolution / snapshot.
    assert env.get_entity("sword") is not None


def test_destroy_propagates_perception_to_bystanders() -> None:
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall", takeable=False))
    env.place_agent(agent_id="actor", location_id="hall")
    env.place_agent(agent_id="witness", location_id="hall")

    env.change_entity_state(
        EntityStateChange(entity_id="sword", new_state="destroyed", destroyed=True, perception="剑被砸碎了"),
        acting_agent_id="actor",
    )
    # Carry flips into next step's ambient; the witness (not the actor) perceives it.
    env.begin_step(step=2, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    witnessed = [ev.content for ev in env.spatial_for(agent_id="witness").ambient_events]
    assert any("剑被砸碎了" in c for c in witnessed)
    # The actor excludes their own action (self-exclusion).
    assert all("剑被砸碎了" not in ev.content for ev in env.spatial_for(agent_id="actor").ambient_events)


def test_destroy_wins_over_pickup_and_location() -> None:
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall"))
    # Destroy flag set alongside owner/location — destruction is terminal, wins.
    env.change_entity_state(
        EntityStateChange(
            entity_id="sword", new_state="x", destroyed=True, owner_id="a1", location_id="hall",
        )
    )
    assert env.get_entity("sword").is_destroyed
    assert "sword" not in env._items


# --------------------------------------------------------------------------- #
# Snapshot / restore — the resurrection trap


def test_snapshot_carries_destroyed_tombstone() -> None:
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall"))
    env.change_entity_state(EntityStateChange(entity_id="sword", new_state="destroyed", destroyed=True))

    states = env.snapshot_state()["entity_states"]
    assert states["sword"]["presence"] == "destroyed"
    assert states["sword"]["presence_ref"] is None


def test_restore_rekills_reseeded_destroyed_entity() -> None:
    # Snapshot after destruction.
    env = _env_with_hall()
    env.register_entity(_item("sword", "剑", "hall"))
    env.change_entity_state(EntityStateChange(entity_id="sword", new_state="destroyed", destroyed=True))
    snapshot = env.snapshot_state()

    # A fresh environment re-instantiates the seed ALIVE (as restore does), then
    # the tombstone overlay must re-kill it rather than leave it resurrected.
    fresh = _env_with_hall()
    fresh.register_entity(_item("sword", "剑", "hall"))
    assert "sword" in fresh._items  # alive before overlay
    fresh.restore_state(snapshot)
    assert fresh.get_entity("sword").is_destroyed
    assert "sword" not in fresh._items
    assert fresh.get_items_at("hall") == []


def test_restore_reregisters_midrun_destroyed_entity_as_tombstone() -> None:
    # An entity created mid-run and destroyed: not in step-0 seeds, so restore
    # re-registers it — and must keep it out of the live view.
    env = _env_with_hall()
    snapshot = {
        "entity_states": {
            "shard": {
                "name": "碎片", "entity_type": "item", "state": "destroyed",
                "presence": "destroyed", "presence_ref": None,
            }
        }
    }
    env.restore_state(snapshot)
    assert env.get_entity("shard").is_destroyed
    assert "shard" not in env._items


# --------------------------------------------------------------------------- #
# Spawn — the one way a thing enters the world after step 0


def test_spawn_lands_in_the_makers_hands() -> None:
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.spawn_entity(
        EntitySpawn(name="换防部署令", description="圈定亲信的名单", holder_id="a1"),
        ground=env.get_body_location("a1"), actor_id="a1",
    )

    made = env.all_live_entities()[0]
    assert made.name == "换防部署令"
    assert made.presence is EntityPresence.HELD
    assert made.owner_id == "a1"
    assert made.description == "圈定亲信的名单"


def test_spawn_without_a_holder_falls_to_the_makers_feet() -> None:
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")

    env.spawn_entity(EntitySpawn(name="一面木牌"), ground=env.get_body_location("a1"), actor_id="a1")

    made = env.all_live_entities()[0]
    assert made.presence is EntityPresence.AT_LOCATION
    assert made.location_id == "hall"


def test_a_private_product_is_visible_only_to_its_maker() -> None:
    """is_public enforces the privileged contract for work output: others see him writing at his desk
    but not what he writes."""
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    env.spawn_entity(
        EntitySpawn(name="换防部署令", holder_id="a1", is_public=False), ground=env.get_body_location("a1"), actor_id="a1",
    )

    assert [e.name for e in env.spatial_for(agent_id="a1").visible_entities] == ["换防部署令"]
    assert env.spatial_for(agent_id="a2").visible_entities == []


def test_spawn_ids_are_deterministic_and_never_collide() -> None:
    """Ids go into snapshots and target refs, so they must be reproducible. Two things with the same
    name are told apart by a sequence number, not a uuid."""
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")

    env.spawn_entity(EntitySpawn(name="密信"), ground=env.get_body_location("a1"), actor_id="a1")
    env.spawn_entity(EntitySpawn(name="密信"), ground=env.get_body_location("a1"), actor_id="a1")

    ids = sorted(e.entity_id for e in env.all_live_entities())
    assert len(ids) == 2
    assert ids[1] == f"{ids[0]}-2"


def test_a_nameless_spawn_creates_nothing() -> None:
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.spawn_entity(EntitySpawn(name="   "), ground=env.get_body_location("a1"), actor_id="a1") is False
    assert env.all_live_entities() == []


def test_one_place_fills_up() -> None:
    env = EnvironmentSystem(max_entities_per_place=1)
    env.space.register_place(_location("hall", "大殿"))
    env.begin_step(step=1, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.spawn_entity(EntitySpawn(name="第一件"), ground=env.get_body_location("a1"), actor_id="a1") is True
    assert env.spawn_entity(EntitySpawn(name="第二件"), ground=env.get_body_location("a1"), actor_id="a1") is False
    assert [e.name for e in env.all_live_entities()] == ["第一件"]


def test_a_crowded_room_does_not_stop_the_next_room() -> None:
    """The cap is per place. The menu lists items at the current place, so one full room must not stop
    new items appearing everywhere else."""
    env = EnvironmentSystem(max_entities_per_place=1)
    env.space.register_place(_location("hall", "大殿"))
    env.space.register_place(_location("garden", "后园"))
    env.begin_step(step=1, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="garden")

    assert env.spawn_entity(EntitySpawn(name="第一件"), ground=env.get_body_location("a1"), actor_id="a1") is True
    assert env.spawn_entity(EntitySpawn(name="第二件"), ground=env.get_body_location("a1"), actor_id="a1") is False   # hall is full
    assert env.spawn_entity(EntitySpawn(name="第三件"), ground=env.get_body_location("a2"), actor_id="a2") is True    # garden is unaffected


def test_a_full_floor_does_not_fill_a_pair_of_hands() -> None:
    """Ground and hands are counted separately: a full floor doesn't stop someone holding a new item."""
    env = EnvironmentSystem(max_entities_per_place=1)
    env.space.register_place(_location("hall", "大殿"))
    env.begin_step(step=1, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    env.place_agent(agent_id="a1", location_id="hall")

    assert env.spawn_entity(EntitySpawn(name="地上那件"), ground=env.get_body_location("a1"), actor_id="a1") is True
    assert env.spawn_entity(EntitySpawn(name="再一件"), ground=env.get_body_location("a1"), actor_id="a1") is False
    assert env.spawn_entity(
        EntitySpawn(name="手上那件", holder_id="a1"), ground=env.get_body_location("a1"), actor_id="a1",
    ) is True


def test_a_destroyed_thing_gives_its_place_back() -> None:
    """Tombstones don't count toward the cap. They are no longer in anyone's reachable list, and
    counting them would let a place fill up with its own history."""
    env = EnvironmentSystem(max_entities_per_place=1)
    env.space.register_place(_location("hall", "大殿"))
    env.begin_step(step=1, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    env.place_agent(agent_id="a1", location_id="hall")
    env.spawn_entity(EntitySpawn(name="第一件"), ground=env.get_body_location("a1"), actor_id="a1")
    doomed = env.all_live_entities()[0].entity_id

    env.change_entity_state(EntityStateChange(entity_id=doomed, new_state="destroyed", destroyed=True))

    assert env.spawn_entity(EntitySpawn(name="第二件"), ground=env.get_body_location("a1"), actor_id="a1") is True


def test_a_spawn_carries_its_observation_to_bystanders() -> None:
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    env.spawn_entity(
        EntitySpawn(name="一面木牌", perception="在大殿，常何立起一面木牌。"), ground=env.get_body_location("a1"), actor_id="a1",
    )
    env.begin_step(step=2, world_time=WorldTime.from_step(2, WorldTimeConfig()))

    seen = [ev.content for ev in env.spatial_for(agent_id="a2").ambient_events]
    assert "在大殿，常何立起一面木牌。" in seen
    # Self-exclusion: the creator doesn't read their own output back as a third-party observation.
    assert env.spatial_for(agent_id="a1").ambient_events == []


def test_a_runtime_made_thing_survives_restore_whole() -> None:
    """A runtime-created thing has no seed entry, so the snapshot payload is its only record. A field
    not saved is lost for good."""
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")
    env.spawn_entity(
        EntitySpawn(name="换防部署令", description="圈定亲信的名单", holder_id="a1", is_public=False),
        ground=env.get_body_location("a1"), actor_id="a1",
    )
    made_id = env.all_live_entities()[0].entity_id

    fresh = _env_with_hall()
    fresh.place_agent(agent_id="a1", location_id="hall")
    fresh.place_agent(agent_id="a2", location_id="hall")
    fresh.restore_state(env.snapshot_state())

    restored = fresh.get_entity(made_id)
    assert restored is not None
    assert restored.description == "圈定亲信的名单"
    assert restored.is_public is False
    assert restored.is_takeable is True
    assert restored.owner_id == "a1"
    # Visibility still holds after restore: a hidden item doesn't become public because it was saved.
    assert fresh.spatial_for(agent_id="a2").visible_entities == []


def test_a_placed_thing_needs_real_ground() -> None:
    """While travelling or unregistered, get_body_location returns a placeholder location. Creating
    nothing is better than attaching an item to a place that doesn't exist."""
    env = _env_with_hall()   # the actor was never placed with place_agent

    assert env.spawn_entity(EntitySpawn(name="一道栅栏"), ground=env.get_body_location("nobody"), actor_id="nobody") is False
    assert env.all_live_entities() == []


def test_a_held_thing_needs_no_ground() -> None:
    """A held item ignores where the holder stands; it is bound to the person, not the place."""
    env = _env_with_hall()

    assert env.spawn_entity(
        EntitySpawn(name="一封密信", holder_id="wanderer"), ground=env.get_body_location("wanderer"), actor_id="wanderer",
    ) is True
    assert env.all_live_entities()[0].owner_id == "wanderer"


def test_a_placed_thing_announces_itself_where_it_lands() -> None:
    """A placed item reaches onlookers' ambient events on the next step via carry, mirroring picking
    one up."""
    env = _env_with_hall()
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    env.spawn_entity(
        EntitySpawn(name="一道栅栏", perception="在大殿，甲把一道栅栏留在了此处。"), ground=env.get_body_location("a1"), actor_id="a1",
    )
    env.begin_step(step=2, world_time=WorldTime.from_step(2, WorldTimeConfig()))

    assert "一道栅栏" in " ".join(
        ev.content for ev in env.spatial_for(agent_id="a2").ambient_events
    )
    # The creator doesn't read their own output back as a third-party observation (self-exclusion).
    assert env.spatial_for(agent_id="a1").ambient_events == []


# --------------------------------------------------------------------------- #
# Only EnvironmentSystem writes entities
# --------------------------------------------------------------------------- #

_ENTITY_FIELDS = frozenset(f.name for f in dataclasses.fields(WorldEntity))
_WRITE_OWNER = Path("engine/environment.py")
# Same field name but not an entity: ``NeedType.__new__`` sets description on enum members.
_NOT_ENTITIES = {(Path("agent/need.py"), "obj.description")}


def _entity_field_writes(source: str) -> list[str]:
    """Assignments shaped like ``x.<WorldEntity field> = ...``. ``self.`` / ``cls.`` are other classes
    writing their own fields and don't count."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for attr in (target.elts if isinstance(target, ast.Tuple) else [target]):
                if (
                    isinstance(attr, ast.Attribute) and attr.attr in _ENTITY_FIELDS
                    and not (isinstance(attr.value, ast.Name) and attr.value.id in ("self", "cls"))
                ):
                    hits.append(ast.unparse(attr))
    return hits


def test_the_write_scan_catches_a_direct_field_write() -> None:
    assert _entity_field_writes("item = env.get_entity('x')\nitem.presence_ref = 'hall'\n") == [
        "item.presence_ref",
    ]
    assert _entity_field_writes("self.state = 1\nscores[d.name] = 2\n") == []


def test_entities_are_written_only_inside_environment_system() -> None:
    """``get_entity`` / ``find_item`` / ``all_live_entities`` return live objects that any holder can
    mutate. Entity changes must go through ``change_entity_state`` / ``spawn_entity``; bypassing them
    skips placement invariants, tombstones, perception delivery and the self-filter.
    """
    found = [
        f"{path}: {hit}"
        for package in ("engine", "agent", "world", "interaction", "core", "providers")
        for path in sorted(Path(package).rglob("*.py"))
        if path != _WRITE_OWNER
        for hit in _entity_field_writes(path.read_text(encoding="utf-8"))
        if (path, hit) not in _NOT_ENTITIES
    ]
    assert not found, f"EnvironmentSystem 之外直接写了物件字段: {found}"
