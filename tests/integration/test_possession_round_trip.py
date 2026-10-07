"""End-to-end: a thing changes hands and comes back to the ground.

A held item must stay visible to its holder and to people nearby, and it must be able to
change hands or go back to the ground without the holder dying. Narrative hand-overs only
count if the world state can express them.

Each hop below asserts both halves: the world state moved, and the people standing there can
see that it moved.
"""

from __future__ import annotations

import pytest

from core.interfaces.action import Deed, EntityStateChange
from engine.environment import EnvironmentSystem
from world.models import EntityPresence, WorldEntity, WorldEntityType
from core.interfaces.place import Place


def _seen_by(env: EnvironmentSystem, agent_id: str) -> dict[str, str | None]:
    """name → holder_id, as that agent perceives it."""
    return {e.name: e.holder_id for e in env.spatial_for(agent_id=agent_id).visible_entities}


@pytest.fixture()
def env() -> EnvironmentSystem:
    system = EnvironmentSystem()
    for loc_id, name in (("hall", "大殿"), ("garden", "庭院")):
        system.space.register_place(Place(
            place_id=loc_id, name=name, connections={}, capacity=50, is_public=True,
        ))
    system.place_agent(agent_id="agent-a", location_id="hall")
    system.place_agent(agent_id="agent-b", location_id="hall")
    system.register_entity(WorldEntity(
        entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.AT_LOCATION,
        presence_ref="hall", is_takeable=True, state="intact",
    ))
    return system


def test_a_thing_can_be_taken_handed_over_and_set_down(env: EnvironmentSystem) -> None:
    # Start: lying on the ground, visible to both, held by no one.
    assert _seen_by(env, "agent-a") == {"虎符": None}
    assert _seen_by(env, "agent-b") == {"虎符": None}

    # (1) A picks it up (SEIZE). It must not disappear from view at this point, for others or
    # for A.
    env.change_entity_state(
        EntityStateChange(entity_id="tally", new_state="intact", owner_id="agent-a"),
        acting_agent_id="agent-a",
    )
    assert env.get_entity("tally").owner_id == "agent-a"
    assert _seen_by(env, "agent-a") == {"虎符": "agent-a"}   # I can see what I'm holding
    assert _seen_by(env, "agent-b") == {"虎符": "agent-a"}   # others can see who holds it

    # B can now reach it. Whether B can take it is up to the judge; the rule only checks
    # co-location.
    assert env.check_physical_feasibility("agent-b", "tally", "item").ok

    # (2) A hands it to B (RELINQUISH + recipient). HELD -> HELD, a direct hand-to-hand transfer.
    env.change_entity_state(
        EntityStateChange(entity_id="tally", new_state="intact", owner_id="agent-b"),
        acting_agent_id="agent-a",
    )
    assert env.get_entity("tally").owner_id == "agent-b"
    assert _seen_by(env, "agent-b") == {"虎符": "agent-b"}

    # (3) B puts it down (RELINQUISH, no recipient). HELD -> AT_LOCATION without waiting for
    # the holder to die.
    env.change_entity_state(
        EntityStateChange(entity_id="tally", new_state="intact", location_id="hall"),
        acting_agent_id="agent-b",
    )
    entity = env.get_entity("tally")
    assert entity.presence is EntityPresence.AT_LOCATION and entity.location_id == "hall"
    assert _seen_by(env, "agent-a") == {"虎符": None}


def test_a_carried_thing_travels_with_its_holder(env: EnvironmentSystem) -> None:
    """A held item moves with its holder."""
    env.change_entity_state(
        EntityStateChange(entity_id="tally", new_state="intact", owner_id="agent-a"),
        acting_agent_id="agent-a",
    )
    env.place_agent(agent_id="agent-a", location_id="garden")

    assert _seen_by(env, "agent-a") == {"虎符": "agent-a"}
    assert _seen_by(env, "agent-b") == {}                       # whoever stayed in the hall can no longer see it
    assert not env.check_physical_feasibility("agent-b", "tally", "item").ok


def test_relinquish_is_a_physical_deed_the_contract_can_carry() -> None:
    """Letting go is PHYSICAL's seventh verb, not a parenthetical inside some person-directed deed.

    Don't fold handing-over into restrain: that deed mutates nothing, so every hand-over in the
    narrative layer would be a no-op in world state.
    """
    from core.interfaces.action import PHYSICAL_DEEDS
    assert Deed.RELINQUISH in PHYSICAL_DEEDS
