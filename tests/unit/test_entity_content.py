"""The content a thing carries: who can read it, how it travels with the thing, how it persists.

Readable = held in hand, or fixed to a place (a stele, a notice). An item on the ground or in
someone else's hand can be seen but not read. This is a separate axis from ``is_public``, which
decides whether it can be seen at all.
"""

from __future__ import annotations

from core.interfaces.action import EntitySpawn, EntityStateChange
from core.interfaces.llm import LLMRouter, LLMScene
from core.prompts import render_entity
from engine.clock import WorldTime, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.scene import SceneVisibility, SituationVoice, assemble_scene_context
from providers.llm.mock import MockLLMProvider
from world.models import EntityPresence, WorldEntity, WorldEntitySeed, WorldEntityType
from core.interfaces.place import Place

SECRET = "明晨卯时，玄武门换防"
CARVED = "敢有擅入者斩"


def _letter(**over) -> WorldEntity:
    kw = dict(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM,
        presence_ref="hall", is_takeable=True, content=SECRET,
    )
    kw.update(over)
    return WorldEntity(**kw)


def _stele() -> WorldEntity:
    return WorldEntity(
        entity_id="stele", name="石碑", entity_type=WorldEntityType.LANDMARK,
        presence_ref="hall", is_takeable=False, content=CARVED,
    )


def _env() -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(Place(place_id="hall", name="大殿"))
    env.begin_step(step=1, world_time=WorldTime.from_step(1, WorldTimeConfig()))
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    return env


def _content_seen_by(env: EnvironmentSystem, agent_id: str) -> dict[str, str]:
    return {e.name: e.content for e in env.spatial_for(agent_id=agent_id).visible_entities}


# --------------------------------------------------------------------------- #
# readable_by


def test_holder_reads_what_he_holds_and_nobody_else_does() -> None:
    held = _letter(presence=EntityPresence.HELD, presence_ref="a1")
    assert held.readable_by("a1")
    assert not held.readable_by("a2")


def test_an_item_on_the_ground_is_read_by_nobody_until_picked_up() -> None:
    assert not _letter().readable_by("a1")


def test_a_landmark_is_read_by_anyone_there() -> None:
    assert _stele().readable_by("a1")
    assert _stele().readable_by("a2")


def test_a_destroyed_thing_is_read_by_nobody() -> None:
    gone = _letter(presence=EntityPresence.DESTROYED, presence_ref=None)
    assert not gone.readable_by("a1")


# --------------------------------------------------------------------------- #
# perception


def test_perception_hands_content_only_to_who_can_read_it() -> None:
    env = _env()
    # Held openly by a1: a2 can see the letter but can't read what it says.
    env.register_entity(_letter(presence=EntityPresence.HELD, presence_ref="a1", is_public=True))
    env.register_entity(_stele())
    assert _content_seen_by(env, "a1") == {"密信": SECRET, "石碑": CARVED}
    assert _content_seen_by(env, "a2") == {"密信": "", "石碑": CARVED}


def test_picking_it_up_is_what_makes_it_readable() -> None:
    env = _env()
    env.register_entity(_letter())
    assert _content_seen_by(env, "a1") == {"密信": ""}
    env.change_entity_state(EntityStateChange(entity_id="letter", owner_id="a1"))
    assert _content_seen_by(env, "a1") == {"密信": SECRET}


def test_seizing_it_moves_the_reading_with_the_thing() -> None:
    env = _env()
    env.register_entity(_letter(presence=EntityPresence.HELD, presence_ref="a1", is_public=True))
    env.change_entity_state(EntityStateChange(entity_id="letter", owner_id="a2"))
    assert _content_seen_by(env, "a2") == {"密信": SECRET}
    assert _content_seen_by(env, "a1") == {"密信": ""}


# --------------------------------------------------------------------------- #
# scene: own eyes vs the functional judge's god view


def _scene(
    env: EnvironmentSystem, agent_id: str, visibility: SceneVisibility, *, reveal: bool = False,
) -> str:
    return assemble_scene_context(
        agent_id, environment=env, directory=LiveWorldDirectory.from_agents({}, env),
        voice=SituationVoice.FIRST, visibility=visibility, reveal_all_content=reveal,
    ).text


def test_own_eyes_scene_reads_only_what_the_viewer_can_read() -> None:
    env = _env()
    env.register_entity(_letter(presence=EntityPresence.HELD, presence_ref="a1", is_public=True))
    env.register_entity(_stele())
    mine = _scene(env, "a1", SceneVisibility.OWN_EYES)
    theirs = _scene(env, "a2", SceneVisibility.OWN_EYES)
    assert SECRET in mine and CARVED in mine
    assert "密信" in theirs and SECRET not in theirs and CARVED in theirs


def test_the_god_view_lists_a_hidden_letter_but_does_not_read_it() -> None:
    """GOD decides which things are listed, not what can be read. The TALK writer and PHYSICAL's
    public outcome both use this scene."""
    env = _env()
    env.register_entity(_letter(presence=EntityPresence.HELD, presence_ref="a1", is_public=False))
    scene = _scene(env, "a2", SceneVisibility.GOD)
    assert "密信" in scene and SECRET not in scene


def test_only_a_judge_granted_it_reads_everything() -> None:
    env = _env()
    env.register_entity(_letter(presence=EntityPresence.HELD, presence_ref="a1", is_public=False))
    assert SECRET in _scene(env, "a2", SceneVisibility.GOD, reveal=True)


def test_the_physical_target_block_carries_no_content() -> None:
    from engine.executors.physical import PhysicalExecutor

    env = _env()
    letter = _letter(presence=EntityPresence.HELD, presence_ref="a1")
    env.register_entity(letter)
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    executor = PhysicalExecutor(router, LiveWorldDirectory.from_agents({}, env))
    assert SECRET not in executor._entity_target_block(letter, actor_id="a2")


# --------------------------------------------------------------------------- #
# render / spawn / snapshot


def test_render_entity_prints_content_only_when_handed_it() -> None:
    letter = _letter(description="火漆封口")
    assert render_entity(letter) == "密信（可取）：火漆封口"
    assert render_entity(letter, content=SECRET) == f"密信（可取）：火漆封口，内容：「{SECRET}」"


def test_spawned_content_lands_on_the_thing() -> None:
    env = _env()
    spawn = EntitySpawn(name="密信", content=SECRET, holder_id="a1", is_public=False)
    assert env.spawn_entity(spawn, ground=env.get_body_location("a1"), actor_id="a1")
    assert env.get_entity(spawn.entity_id).content == SECRET


def test_content_survives_restore_for_made_and_seeded_things_alike() -> None:
    env = _env()
    env.register_entity(_letter(content=""))
    env.change_entity_state(EntityStateChange(entity_id="letter", new_content=SECRET))
    made = EntitySpawn(name="账册", content="欠粮三百石", holder_id="a1")
    env.spawn_entity(made, ground=env.get_body_location("a1"), actor_id="a1")
    snapshot = env.snapshot_state()

    fresh = _env()
    fresh.register_entity(_letter(content=""))   # seeds are rebuilt as they were at creation
    fresh.restore_state(snapshot)
    assert fresh.get_entity("letter").content == SECRET
    assert fresh.get_entity(made.entity_id).content == "欠粮三百石"


def test_empty_new_content_leaves_it_alone() -> None:
    env = _env()
    env.register_entity(_letter())
    env.change_entity_state(EntityStateChange(entity_id="letter", new_state="已拆封"))
    assert env.get_entity("letter").content == SECRET


def test_seed_content_round_trips() -> None:
    seed = WorldEntitySeed(name="石碑", entity_type="landmark", content=CARVED)
    assert seed.as_dict()["content"] == CARVED
    assert "properties" not in seed.as_dict()
