"""The observer's read model gives every consumer an id to look things up by — names are for printing."""

from __future__ import annotations

from core.interfaces.snapshot import WorldSnapshot
from engine.environment import IN_TRANSIT
from engine.executors.social import SocialExecutor
from interaction.models import DialogueTurn, EntityView, StepEvent


def _snapshot(**kwargs: object) -> WorldSnapshot:
    return WorldSnapshot(world_id="w", step=3, timestamp="2026-09-30 00:00:00", **kwargs)


def test_agent_location_carries_its_id_beside_the_name() -> None:
    transit = {"from_location_id": "loc-gate", "to_location_id": "loc-hall",
               "path": ["loc-gate", "loc-yard", "loc-hall"], "arrivals": [0, 1, 3],
               "elapsed_steps": 1, "total_steps": 3}
    # A mover's own record keeps his departure room until he arrives; only the environment
    # knows where the body is.
    mover = {"location_id": "loc-gate", "location_name": "玄武门", "transit": transit}
    event = StepEvent.from_snapshot(_snapshot(
        agent_states={
            "a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-gate", "location_name": "玄武门"},
            "b": {"agent_id": "b", "agent_name": "乙", **mover},
            "c": {"agent_id": "c", "agent_name": "丙", **mover},
        },
        metadata={"environment": {
            "body_locations": {"a": "loc-gate", "b": "loc-yard", "c": IN_TRANSIT},
            "location_names": {"loc-gate": "玄武门", "loc-yard": "内院", "loc-hall": "大殿"},
        }},
    ))

    assert (event.agent_states["a"].location, event.agent_states["a"].location_id) == ("玄武门", "loc-gate")
    # On a waypoint he is in that room; between two rooms he is in none.
    assert (event.agent_states["b"].location, event.agent_states["b"].location_id) == ("内院", "loc-yard")
    assert (event.agent_states["c"].location, event.agent_states["c"].location_id) == ("途中", "")
    assert event.agent_states["b"].transit == transit


def test_an_arrival_reads_as_standing_in_the_destination_with_its_route() -> None:
    arrival = {"from_location_id": "loc-gate", "to_location_id": "loc-hall",
               "path": ["loc-gate", "loc-yard", "loc-hall"], "arrivals": [0, 1, 1],
               "elapsed_steps": 1, "total_steps": 1}
    event = StepEvent.from_snapshot(_snapshot(
        agent_states={"a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-hall",
                            "location_name": "大殿", "arrival": arrival}},
        metadata={"environment": {"body_locations": {"a": "loc-hall"}}},
    ))

    state = event.agent_states["a"]
    assert (state.location, state.location_id) == ("大殿", "loc-hall")
    assert state.transit is None
    assert state.arrival == arrival


def test_npc_location_carries_its_id_beside_the_name() -> None:
    event = StepEvent.from_snapshot(_snapshot(metadata={"environment": {
        "npc_states": {"n1": {"name": "小宦官"}, "n2": {"name": "信使"}},
        "body_locations": {"n1": "loc-gate", "n2": IN_TRANSIT},
        "location_names": {"loc-gate": "玄武门"},
    }}))

    placed, moving = event.npcs
    assert (placed.location, placed.location_id) == ("玄武门", "loc-gate")
    assert (moving.location, moving.location_id) == ("途中", "")


def test_entities_are_a_projection_not_the_restore_payload() -> None:
    event = StepEvent.from_snapshot(_snapshot(metadata={"environment": {
        "transit_origins": {"a": "loc-gate"},
        "entity_states": {"e1": {
            "name": "密信", "entity_type": "item", "state": "sealed", "presence": "held",
            "presence_ref": "a", "is_public": False, "is_takeable": True, "created_step": 2,
            "content": "明日举事",
        }},
    }}))

    # created_step is included, is_takeable is not: the former is a world fact (the step the thing
    # appeared) only the backend knows; the latter is seed-rebuild authority, meaningless to
    # observers.
    assert event.entities == {"e1": EntityView(
        name="密信", entity_type="item", state="sealed", presence="held",
        presence_ref="a", is_public=False, content="明日举事", created_step=2,
    )}
    assert not hasattr(event.entities["e1"], "is_takeable")
    assert not hasattr(event, "raw_environment")


def test_dialogue_turns_name_their_speaker_by_id() -> None:
    event = StepEvent.from_snapshot(_snapshot(actions_this_step=[{
        "agent_id": "a", "agent_name": "甲",
        "dialogue": [{"speaker_id": "b", "speaker": "乙", "line": "来了"}],
    }]))

    assert event.actions[0].dialogue == [DialogueTurn(speaker_id="b", speaker="乙", line="来了")]


def test_talk_transcript_is_tagged_with_speaker_ids() -> None:
    turns = SocialExecutor._parse_dialogue(
        {"dialogue": [{"speaker": 1, "line": "来了"}, {"speaker": 2, "line": "嗯"}]},
        ("a", "甲"),
        ("b", "乙"),
    )

    assert turns == [
        {"speaker_id": "a", "speaker": "甲", "line": "来了"},
        {"speaker_id": "b", "speaker": "乙", "line": "嗯"},
    ]


def test_step_carries_its_relations_on_both_paths() -> None:
    relations = {
        "a->b": {"from_id": "a", "to_id": "b", "trust_objective": 0.8, "affection_objective": 0.3,
                 "labels": ["盟友"], "interaction_count": 2, "history_summary": "同谋"},
        # A baseline record from mere co-occurrence is not a relationship.
        "b->a": {"from_id": "b", "to_id": "a", "labels": [], "interaction_count": 0,
                 "history_summary": ""},
    }
    replay = StepEvent.from_snapshot(_snapshot(agent_relations=relations))
    live = StepEvent.from_runtime_payload(
        {"world_id": "w", "step": 3, "agent_relations": relations}
    )

    assert [(e.from_id, e.to_id, e.labels) for e in replay.relations] == [("a", "b", ["盟友"])]
    assert live.relations == replay.relations
