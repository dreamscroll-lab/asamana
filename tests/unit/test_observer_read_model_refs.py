"""The observer's read model places every body, thing and event in the scope it is in, and gives
every consumer an id to look things up by — names are for printing."""

from __future__ import annotations

from engine.environment import IN_TRANSIT
from engine.executors.social import SocialExecutor
from core.interfaces.action import DialogueTurn
from interaction.models import EntityPrivateView, EntityView, StepView
from tests.unit.step_view import TRANSIT, agent, agents, events, npcs, snapshot_of

PLACES = {"loc-gate": "玄武门", "loc-yard": "内院", "loc-hall": "大殿"}


def test_an_agent_stands_in_the_scope_his_body_is_in() -> None:
    transit = {"from_location_id": "loc-gate", "to_location_id": "loc-hall",
               "path": ["loc-gate", "loc-yard", "loc-hall"], "arrivals": [0, 1, 3],
               "elapsed_steps": 1, "total_steps": 3}
    # The record says where the body is: on a waypoint, or between two (IN_TRANSIT).
    view = StepView.from_snapshot(snapshot_of(
        places=PLACES,
        agents={
            "a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-gate"},
            "b": {"agent_id": "b", "agent_name": "乙", "location_id": "loc-yard", "transit": transit},
            "c": {"agent_id": "c", "agent_name": "丙", "location_id": IN_TRANSIT, "transit": transit},
        },
    ))

    placed = {agent_id: where for agent_id, (where, _) in agents(view).items()}
    # On a waypoint he is in that room; between two rooms he is in none.
    assert placed == {"a": "loc-gate", "b": "loc-yard", "c": TRANSIT}
    assert view.locations["loc-yard"].name == "内院"
    # A passer-by is seen; his route is his own.
    assert agent(view, "b").private.transit == transit
    # Every place is listed, empty or not.
    assert set(view.locations) == {"loc-gate", "loc-yard", "loc-hall"}


def test_an_arrival_stands_in_the_destination_and_its_move_carries_the_route() -> None:
    arrival = {"from_location_id": "loc-gate", "to_location_id": "loc-hall",
               "path": ["loc-gate", "loc-yard", "loc-hall"], "arrivals": [0, 1, 1],
               "elapsed_steps": 1, "total_steps": 1}
    view = StepView.from_snapshot(snapshot_of(
        places=PLACES,
        agents={"a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-hall"}},
        records=[{"agent_id": "a", "initiator_id": "a", "execution_id": "x",
                  "action_type": "move", "phase": "settled", "location_id": "loc-hall"}],
        routes={"a": arrival},
    ))

    assert agents(view)["a"][0] == "loc-hall"
    assert agent(view, "a").private.transit is None
    [(_, move)] = events(view, "action")
    assert move.private.route == arrival


def test_an_npc_stands_in_the_scope_its_body_is_in() -> None:
    view = StepView.from_snapshot(snapshot_of(
        places=PLACES,
        npcs={
            "n1": {"name": "小宦官", "location_id": "loc-gate"},
            "n2": {"name": "信使", "location_id": IN_TRANSIT},
        },
    ))

    assert {npc_id: where for npc_id, (where, _) in npcs(view).items()} == {
        "n1": "loc-gate", "n2": TRANSIT,
    }


def test_entities_are_a_projection_not_the_restore_payload() -> None:
    view = StepView.from_snapshot(snapshot_of(
        agents={"a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-gate"}},
        entities={"e1": {
            "name": "密信", "entity_type": "item", "state": "sealed", "presence": "held",
            "presence_ref": "a", "is_public": False, "is_takeable": True, "created_step": 2,
            "content": "明日举事",
        }},
    ))

    # created_step is included, is_takeable is not: the former is a world fact (the step the thing
    # appeared) only the backend knows; the latter is seed-rebuild authority, meaningless to
    # observers.
    held = agent(view, "a").private.carrying
    assert held == {"e1": EntityView(
        name="密信", entity_type="item", state="sealed", is_public=False, created_step=2,
        # Seeing a thing is not reading it: the words are the holder's.
        private=EntityPrivateView(content="明日举事"),
    )}
    assert not hasattr(held["e1"], "is_takeable")


def test_an_entity_is_listed_where_it_is() -> None:
    def thing(presence: str, ref: str | None) -> dict[str, object]:
        return {"name": "物", "entity_type": "item", "state": "", "presence": presence,
                "presence_ref": ref, "is_public": True}

    view = StepView.from_snapshot(snapshot_of(
        places={"loc-gate": "玄武门"},
        agents={"a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-gate"}},
        entities={
            "ground": thing("at_location", "loc-gate"),
            "hand": thing("held", "a"),
            "gone": thing("destroyed", None),
        },
    ))

    assert set(view.locations["loc-gate"].entities) == {"ground"}
    assert set(agent(view, "a").public.carrying) == {"hand"}
    # A destroyed thing is no longer in the world: listed nowhere.
    listed = {
        eid
        for scope in view.locations.values()
        for eid in [*scope.entities, *(e for a in scope.agents
                                       for e in (*a.public.carrying, *a.private.carrying))]
    }
    assert listed == {"ground", "hand"}


def test_dialogue_turns_name_their_speaker_by_id() -> None:
    view = StepView.from_snapshot(snapshot_of(records=[{
        "agent_id": "a", "location_id": "loc-gate",
        "dialogue": [{"speaker_id": "b", "speaker": "乙", "line": "来了"}],
    }]))

    [(_, act)] = events(view, "action")
    assert act.private.dialogue == [DialogueTurn(speaker_id="b", speaker="乙", line="来了")]


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


def test_relations_ride_their_holder_on_both_paths() -> None:
    relations = {
        "a->b": {"from_id": "a", "to_id": "b", "trust_objective": 0.8, "affection_objective": 0.3,
                 "labels": ["盟友"], "interaction_count": 2, "history_summary": "同谋"},
        # A baseline record from mere co-occurrence is not a relationship.
        "b->a": {"from_id": "b", "to_id": "a", "labels": [], "interaction_count": 0,
                 "history_summary": ""},
    }
    snapshot = snapshot_of(
        agents={
            "a": {"agent_id": "a", "agent_name": "甲", "location_id": "loc-gate"},
            "b": {"agent_id": "b", "agent_name": "乙", "location_id": "loc-gate"},
        },
        relations=relations,
    )
    replay = StepView.from_snapshot(snapshot)
    # Live observers get the very snapshot the step saved.
    live = StepView.from_runtime_payload({"type": "step", "snapshot": snapshot})

    edges = agent(replay, "a").private.relations
    assert [(e.from_id, e.to_id, e.labels) for e in edges] == [("a", "b", ["盟友"])]
    assert agent(replay, "b").private.relations == []
    assert live == replay
