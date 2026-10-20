"""The world's setup: the build's conversion, the rules every setup keeps, the director's
revisions, and where the setup lives from build to confirmation."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from agent.memory import memory_collection_name
from agent.memory_types import MemoryStream
from agent.need import NeedState, NeedType
from agent.personality import EmotionState, EmotionType
from engine.application import NarrativeApplication
from world import WorldCatalog
from world.models import HistoricalEventSeed, NpcSeed, RelationSeed, ThemeAnalysis, WorldEntitySeed
from world.setup import CharacterRevision, SetupHistoricalEvent, SetupRelation, WorldSetup

from interaction.api import create_app
from tests.unit.world_setups import setup_of


PLACES = frozenset({"palace", "garden"})


def _setup() -> WorldSetup:
    return setup_of({
        "a": {"name": "甲", "gender": "男", "is_main_character": True},
        "b": {"name": "乙", "gender": "女"},
        "c": {"name": "丙"},
    })


def _relation(source: str, target: str, **values) -> SetupRelation:
    return SetupRelation(**{
        "source_id": source, "target_id": target, "trust": 0.5, "affection": 0.0, "labels": (),
        "reverse_trust": 0.5, "reverse_affection": 0.0, "reverse_labels": (), **values,
    })


def _event(*figure_ids: str, **values) -> SetupHistoricalEvent:
    return SetupHistoricalEvent(**{
        "event": "事", "hours_before_start": 2, "figure_ids": figure_ids, "importance": 0.5, **values,
    })


def _revise(setup: WorldSetup, agent_id: str, revision: CharacterRevision) -> WorldSetup:
    return setup.with_character(agent_id, revision, places=PLACES)


# ----- the build's conversion -----------------------------------------------


def test_the_build_speaks_names_and_the_setup_ids() -> None:
    characters = _setup().characters
    analysis = ThemeAnalysis(
        theme_input="t", world_name="w", era_description="", core_tension="", narrative_theme="",
        world_time_config={"seconds_per_step": 3600},
        initial_relations=[
            RelationSeed(
                "甲", "乙", trust=0.7, labels=["父子：父亲", " ", "父子:", "父子:父亲"],
                reverse_labels=["父子 : 儿子"],
            ),
            RelationSeed("乙", "甲"),  # a second relation for the pair: the first stands
            RelationSeed("甲", "丁"),  # nobody called 丁
        ],
        historical_events=[
            HistoricalEventSeed("甲乙同游", hours_before_start=5, related_figures=["甲", "乙", "甲"]),
            HistoricalEventSeed("丁独行", related_figures=["丁"]),  # nobody left to remember it
        ],
    )
    setup = WorldSetup.from_build("t", analysis, characters)
    [relation] = setup.relations
    assert (relation.source_id, relation.target_id, relation.trust) == ("a", "b", 0.7)
    assert relation.labels == ("父子:父亲",) and relation.reverse_labels == ("父子:儿子",)
    assert relation.reverse_trust == 0.7  # an unset reverse mirrors the forward value
    assert [(e.event, e.figure_ids) for e in setup.historical_events] == [("甲乙同游", ("a", "b"))]
    setup.validate(PLACES)


def test_the_build_gives_every_npc_and_thing_its_own_id() -> None:
    """Ids come from names, and two names can share a slug: the setup still gives each its own."""
    analysis = ThemeAnalysis(
        theme_input="t", world_name="w", era_description="", core_tension="", narrative_theme="",
        world_time_config={"seconds_per_step": 3600},
        npc_seeds=[NpcSeed(name="脚夫", location="palace"), NpcSeed(name="更夫", location="palace")],
        world_entity_seeds=[
            WorldEntitySeed(name="Red Seal", entity_type="item", location="palace"),
            WorldEntitySeed(name="red-seal", entity_type="item", location="palace"),
        ],
    )
    setup = WorldSetup.from_build("t", analysis, _setup().characters)
    npc_ids = [n.npc_id for n in setup.npc_seeds]
    entity_ids = [e.entity_id for e in setup.entity_seeds]
    assert all(i.startswith("npc_") for i in npc_ids) and len(set(npc_ids)) == 2
    assert entity_ids[0].startswith("seed_") and len(set(entity_ids)) == 2
    setup.validate(PLACES)


def test_validate_refuses_a_shared_or_missing_id() -> None:
    shared = replace(_with_seeds(npcs=("脚夫",)), npc_seeds=(NpcSeed(name="脚夫", npc_id="a", location="palace"),))
    with pytest.raises(ValueError, match="share an id"):
        shared.validate(PLACES)
    missing = replace(_with_seeds(), npc_seeds=(NpcSeed(name="脚夫", location="palace"),))
    with pytest.raises(ValueError, match="id"):
        missing.validate(PLACES)


def test_a_setup_round_trips_through_its_document() -> None:
    setup = _setup().with_relations([_relation("a", "b")], places=PLACES).with_history(
        [_event("a", "b")], places=PLACES,
    )
    assert WorldSetup.from_dict(setup.as_dict()) == setup


# ----- the rules ------------------------------------------------------------


@pytest.mark.parametrize("broken", [
    lambda s: replace(s, characters=()),
    lambda s: replace(s, characters=tuple(replace(d, tier=d.tier.BACKGROUND) for d in s.characters)),
    lambda s: replace(s, characters=(*s.characters, s.characters[0])),  # two share an id
    lambda s: replace(s, characters=tuple(
        replace(d, soul=replace(d.soul, name="甲")) if d.agent_id == "b" else d for d in s.characters
    )),
    lambda s: replace(s, characters=tuple(replace(d, initial_location="moon") for d in s.characters)),
    lambda s: replace(s, lore=replace(s.lore, world_time_config={"seconds_per_step": 99 * 3600})),
])
def test_validate_refuses_a_setup_no_world_can_start_from(broken) -> None:
    with pytest.raises(ValueError):
        broken(_setup()).validate(PLACES)


def _with_seeds(npcs: tuple[str, ...] = (), things: tuple[str, ...] = ()) -> WorldSetup:
    return replace(
        _setup(),
        npc_seeds=tuple(NpcSeed(name=n, npc_id=f"npc_{i}", location="palace") for i, n in enumerate(npcs)),
        entity_seeds=tuple(
            WorldEntitySeed(name=t, entity_id=f"seed_{i}", entity_type="item", location="palace")
            for i, t in enumerate(things)
        ),
    )


@pytest.mark.parametrize("npcs,things", [
    (("甲",), ()),             # an NPC named like a character
    ((), ("乙",)),             # a thing named like a character
    (("守卫",), ("守卫",)),     # an NPC named like a thing
    ((), ("Letter", "letter")),  # one name to an entity id
    (("守卫", " 守卫 "), ()),
])
def test_no_two_characters_npcs_or_things_share_a_name(npcs, things) -> None:
    with pytest.raises(ValueError, match="used twice"):
        _with_seeds(npcs, things).validate(PLACES)


def test_npcs_and_things_stand_on_the_map() -> None:
    lost = replace(_with_seeds(things=("玉玺",)), entity_seeds=(
        WorldEntitySeed(name="玉玺", entity_id="seed_0", entity_type="item", location="moon"),
    ))
    with pytest.raises(ValueError, match="location of"):
        lost.validate(PLACES)


def test_distinct_names_pass() -> None:
    _with_seeds(("守卫",), ("玉玺", "密信")).validate(PLACES)


def test_a_new_character_cannot_take_a_things_name() -> None:
    with pytest.raises(ValueError, match="used twice"):
        _with_seeds((), ("玉玺",)).with_character_added("玉玺", _NEWCOMER, places=PLACES, cast_limit=15)


# ----- revisions -------------------------------------------------------------


def test_character_revision_sets_only_given_fields() -> None:
    setup = _revise(_setup(), "a", CharacterRevision(gender="女", core_traits=("沉稳",)))
    soul = setup.character("a").soul
    assert soul.gender == "女" and soul.core_traits == ("沉稳",) and soul.name == "甲"
    assert setup.character("b").soul.gender == "女"


def test_character_revision_reaches_the_definition() -> None:
    setup = _revise(_setup(), "b", CharacterRevision(
        is_main_character=True,
        initial_location="garden",
        initial_emotion=EmotionState(primary=EmotionType.FEAR, intensity=0.7, valence=-0.5),
        long_term_goals=("守住家业",),
        hard_constraints=("绝不背叛友人",),
    ))
    definition = setup.character("b")
    assert definition.is_main_character and definition.initial_location == "garden"
    assert definition.initial_emotion.primary is EmotionType.FEAR
    assert definition.long_term_goals == ["守住家业"]
    assert definition.soul.hard_constraints == ("绝不背叛友人",)


def test_needs_left_out_get_the_baseline() -> None:
    need = NeedState(type=NeedType.ESTEEM, label="想被人看得起", intensity=0.8, weight=1.8)
    needs = _revise(_setup(), "a", CharacterRevision(needs=(need,))).character("a").soul.innate_needs
    assert {n.type for n in needs} == set(NeedType)
    assert [n for n in needs if n.type is NeedType.ESTEEM] == [need]


@pytest.mark.parametrize("revision", [
    CharacterRevision(age=-1),
    CharacterRevision(gender="未知"),
    CharacterRevision(gender=""),
    CharacterRevision(initial_location="moon"),
    CharacterRevision(initial_emotion=EmotionState(intensity=1.5)),
    CharacterRevision(needs=(
        NeedState(type=NeedType.SAFETY, label="安稳", intensity=0.5),
        NeedState(type=NeedType.SAFETY, label="安稳", intensity=0.5, is_hidden=True),
    )),
    CharacterRevision(needs=(NeedState(type=NeedType.SAFETY, label="安稳", intensity=0.5, weight=3.0),)),
    CharacterRevision(is_main_character=False),  # the only main character
])
def test_character_revision_refuses_bad_values(revision: CharacterRevision) -> None:
    with pytest.raises(ValueError):
        _revise(_setup(), "a", revision)


def test_character_revision_refuses_an_unknown_character() -> None:
    with pytest.raises(ValueError):
        _revise(_setup(), "nobody", CharacterRevision(role="将军"))


_NEWCOMER = CharacterRevision(
    role="商人", age=40, gender="女", is_main_character=False, initial_location="garden",
)


def test_added_character_gets_an_id_a_free_color_and_the_baseline_needs() -> None:
    base = _setup()
    setup, agent_id = base.with_character_added("丁", _NEWCOMER, places=PLACES, cast_limit=15)
    definition = setup.character(agent_id)
    assert definition.soul.name == "丁" and definition.initial_location == "garden"
    assert definition.soul.color not in {d.soul.color for d in base.characters}
    assert {n.type for n in definition.soul.innate_needs} == set(NeedType)
    assert setup.characters[:3] == base.characters


@pytest.mark.parametrize("name,revision,limit", [
    ("甲", _NEWCOMER, 15),  # taken
    ("", _NEWCOMER, 15),
    ("丁", replace(_NEWCOMER, initial_location=None), 15),
    ("丁", replace(_NEWCOMER, role=""), 15),
    ("丁", _NEWCOMER, 3),  # cast full
])
def test_added_character_is_refused(name: str, revision: CharacterRevision, limit: int) -> None:
    with pytest.raises(ValueError):
        _setup().with_character_added(name, revision, places=PLACES, cast_limit=limit)


def test_removed_character_takes_its_relations_and_leaves_events_it_shared() -> None:
    setup = _setup().with_relations(
        [_relation("a", "b"), _relation("a", "c")], places=PLACES,
    ).with_history([_event("a", "b", event="甲乙同游"), _event("b", event="乙独行")], places=PLACES)
    removed = setup.without_character("b", places=PLACES)
    assert [d.agent_id for d in removed.characters] == ["a", "c"]
    assert [(r.source_id, r.target_id) for r in removed.relations] == [("a", "c")]
    assert [(e.event, e.figure_ids) for e in removed.historical_events] == [("甲乙同游", ("a",))]


def test_the_last_main_character_cannot_be_removed() -> None:
    with pytest.raises(ValueError):
        _setup().without_character("a", places=PLACES)


@pytest.mark.parametrize("event", [
    _event("a", event=""),
    _event("a", hours_before_start=0),
    _event(),
    _event("a", "a"),
    _event("d"),
    _event("a", importance=1.5),
])
def test_history_refuses_bad_events(event: SetupHistoricalEvent) -> None:
    with pytest.raises(ValueError):
        _setup().with_history([event], places=PLACES)


@pytest.mark.parametrize("relations", [
    [_relation("a", "a")],
    [_relation("a", "b"), _relation("b", "a")],
    [_relation("a", "d")],
    [_relation("a", "b", trust=1.2)],
    [_relation("a", "b", reverse_affection=-2.0)],
    [_relation("a", "b", labels=(" ",))],
    [_relation("a", "b", labels=("父子:",))],
    [_relation("a", "b", reverse_labels=(":父亲",))],
    [_relation("a", "b", labels=("父子:父亲:长子",))],
    [_relation("a", "b", labels=("父子：父亲",))],
    [_relation("a", "b", reverse_labels=("相识", "相识"))],
])
def test_relations_refuse_bad_seeds(relations: list[SetupRelation]) -> None:
    with pytest.raises(ValueError):
        _setup().with_relations(relations, places=PLACES)


def test_step_length_must_be_one_the_clock_takes() -> None:
    setup = _setup().with_seconds_per_step(6 * 3600, places=PLACES)
    assert setup.lore.seconds_per_step == 6 * 3600
    for seconds in (0, 25 * 3600):
        with pytest.raises(ValueError):
            _setup().with_seconds_per_step(seconds, places=PLACES)


# ----- through the API ------------------------------------------------------
#
# A revision writes the setup alone; what it derives exists only once the world is confirmed,
# so the stores are checked after confirming.


async def _built_world(container, test_config, catalog: WorldCatalog) -> tuple[str, dict[str, str]]:
    """A world built and not yet confirmed, and its cast's ids by name."""
    application = NarrativeApplication(container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    setup = await application.world_setup(world_id)
    return world_id, {d.soul.name: d.agent_id for d in setup.characters}


def _client(container, test_config, catalog: WorldCatalog) -> TestClient:
    return TestClient(create_app(container, test_config, catalog=catalog))


async def _seeds(container, world_id: str, agent_id: str) -> list[dict]:
    results = await container.vector_store.list_all(
        memory_collection_name(world_id, agent_id, MemoryStream.FACTUAL)
    )
    return [r.payload for r in results]


@pytest.mark.asyncio
async def test_setup_lists_events_and_relations_by_id(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        setup = client.get(f"/api/worlds/{world_id}/setup").json()
    [event] = setup["historical_events"]
    assert sorted(event["figure_ids"]) == sorted([ids["Li Shimin"], ids["Li Jiancheng"]])
    assert event["hours_before_start"] == 48
    [relation] = setup["relations"]
    assert (relation["source_id"], relation["target_id"]) == (ids["Li Shimin"], ids["Li Jiancheng"])
    assert {c["agent_id"] for c in setup["characters"]} == set(ids.values())


@pytest.mark.asyncio
async def test_a_revision_writes_the_setup_and_nothing_else(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        response = client.put(f"/api/worlds/{world_id}/setup/history", json={"historical_events": [
            {"event": "高祖召见", "hours_before_start": 3, "figure_ids": [ids["Li Shimin"]], "importance": 0.8},
        ]})
    assert response.status_code == 200
    assert await mock_build_container.snapshot.list_steps(world_id) == []
    assert await mock_build_container.agent_store.list_agent_ids(world_id) == []
    assert await _seeds(mock_build_container, world_id, ids["Li Shimin"]) == []


@pytest.mark.asyncio
async def test_revised_character_reaches_profile_and_the_confirmed_world(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    target = ids["Li Jiancheng"]
    with _client(mock_build_container, test_config, catalog) as client:
        client.get(f"/api/worlds/{world_id}/agents/{target}/profile")  # warm the profile cache
        response = client.patch(
            f"/api/worlds/{world_id}/setup/characters/{target}",
            json={"gender": "女", "secret": "暗藏兵符"},
        )
        assert response.status_code == 200
        profile = client.get(f"/api/worlds/{world_id}/agents/{target}/profile").json()
        cast = {c["agent_id"]: c for c in client.get(f"/api/worlds/{world_id}/agents").json()}
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
    assert profile["gender"] == "女" and profile["secret"] == "暗藏兵符"
    assert cast[target]["gender"] == "女"
    relation = await mock_build_container.agent_store.load_relation(world_id, ids["Li Shimin"], target)
    assert relation is not None and relation.to_gender == "女"


@pytest.mark.asyncio
async def test_name_is_not_revisable(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        response = client.patch(
            f"/api/worlds/{world_id}/setup/characters/{ids['Li Shimin']}", json={"name": "秦王"},
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_revised_history_becomes_the_backstory_memories(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    shimin, jiancheng, gaozu = ids["Li Shimin"], ids["Li Jiancheng"], ids["Emperor Gaozu"]
    with _client(mock_build_container, test_config, catalog) as client:
        response = client.put(f"/api/worlds/{world_id}/setup/history", json={"historical_events": [
            {"event": "高祖于太极宫召见秦王", "hours_before_start": 3, "figure_ids": [shimin, gaozu], "importance": 0.8},
        ]})
        assert response.status_code == 200
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
    assert [m["raw_content"] for m in await _seeds(mock_build_container, world_id, shimin)] == ["高祖于太极宫召见秦王"]
    assert [m["raw_content"] for m in await _seeds(mock_build_container, world_id, gaozu)] == ["高祖于太极宫召见秦王"]
    assert await _seeds(mock_build_container, world_id, jiancheng) == []


@pytest.mark.asyncio
async def test_revised_step_length_prices_the_backstory(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        url = f"/api/worlds/{world_id}/setup/seconds-per-step"
        assert client.put(url, json={"seconds_per_step": 12 * 3600}).status_code == 200
        assert client.put(url, json={"seconds_per_step": 30 * 3600}).status_code == 400
        meta = client.get(f"/api/worlds/{world_id}").json()
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
    assert meta["seconds_per_step"] == 12 * 3600
    # 48 hours before the start, at 12 hours a step.
    assert [m["created_step"] for m in await _seeds(mock_build_container, world_id, ids["Li Shimin"])] == [-4]


@pytest.mark.asyncio
async def test_revised_relations_survive_a_reset(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    shimin, jiancheng, gaozu = ids["Li Shimin"], ids["Li Jiancheng"], ids["Emperor Gaozu"]
    with _client(mock_build_container, test_config, catalog) as client:
        response = client.put(f"/api/worlds/{world_id}/setup/relations", json={"relations": [{
            "source_id": shimin, "target_id": gaozu, "trust": 0.6, "affection": 0.3,
            "labels": ["父子：父亲"], "reverse_trust": 0.5, "reverse_affection": 0.2,
            "reverse_labels": ["父子:儿子"],
        }]})
        assert response.status_code == 200
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
        assert client.post(f"/api/worlds/{world_id}/reset").status_code == 200
        edges = {(e["from_id"], e["to_id"]): e for e in client.get(f"/api/worlds/{world_id}/graph").json()["edges"]}
    assert set(edges) == {(shimin, gaozu), (gaozu, shimin)}
    assert edges[(shimin, gaozu)]["labels"] == ["父子:父亲"]
    assert await mock_build_container.agent_store.load_relation(world_id, shimin, jiancheng) is None


@pytest.mark.asyncio
async def test_a_label_out_of_form_is_refused(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        response = client.put(f"/api/worlds/{world_id}/setup/relations", json={"relations": [{
            "source_id": ids["Li Shimin"], "target_id": ids["Emperor Gaozu"], "trust": 0.5,
            "affection": 0.0, "labels": ["父子:"], "reverse_trust": 0.5, "reverse_affection": 0.0,
            "reverse_labels": [],
        }]})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_a_refused_revision_changes_nothing(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        before = client.get(f"/api/worlds/{world_id}/setup").json()
        response = client.put(f"/api/worlds/{world_id}/setup/relations", json={"relations": [{
            "source_id": ids["Li Shimin"], "target_id": "nobody", "trust": 0.5, "affection": 0.0,
            "labels": [], "reverse_trust": 0.5, "reverse_affection": 0.0, "reverse_labels": [],
        }]})
        after = client.get(f"/api/worlds/{world_id}/setup").json()
    assert response.status_code == 400
    assert after == before


@pytest.mark.asyncio
async def test_setup_is_open_until_the_world_is_confirmed(mock_build_container, test_config) -> None:
    """Whoever reviews the world, and whether anyone does: the setup only locks on confirming."""
    catalog = WorldCatalog(None)
    application = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    agent_id = (await application.world_setup(world_id)).characters[0].agent_id
    edit = {"role": "将军"}
    with _client(mock_build_container, test_config, catalog) as client:
        url = f"/api/worlds/{world_id}/setup/characters/{agent_id}"
        assert client.patch(url, json=edit).status_code == 200
        client.post(f"/api/worlds/{world_id}/confirm")
        assert client.patch(url, json=edit).status_code == 409


@pytest.mark.asyncio
async def test_setup_is_not_the_actors(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    application = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    character_id = (await application.world_setup(world_id)).characters[0].agent_id
    with _client(mock_build_container, test_config, catalog) as client:
        client.post(f"/api/worlds/{world_id}/confirm", json={"character_id": character_id})
        assert client.get(f"/api/worlds/{world_id}/setup").status_code == 403


@pytest.mark.asyncio
async def test_added_character_joins_the_confirmed_world(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config, catalog) as client:
        place = client.get(f"/api/worlds/{world_id}/places").json()[0]["location_id"]
        response = client.post(f"/api/worlds/{world_id}/setup/characters", json={
            "name": "尉迟恭", "role": "将军", "age": 40, "gender": "男", "is_main_character": False,
            "initial_location": place, "long_term_goals": ["护主周全"],
        })
        assert response.status_code == 200
        agent_id = response.json()["agent_id"]
        cast = {c["agent_id"] for c in client.get(f"/api/worlds/{world_id}/agents").json()}
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
        assert client.post(f"/api/worlds/{world_id}/reset").status_code == 200
        nodes = {n["id"] for n in client.get(f"/api/worlds/{world_id}/graph").json()["nodes"]}
    assert agent_id in cast and agent_id in nodes
    state = await mock_build_container.agent_store.load_initial_agent_state(world_id, agent_id)
    assert state is not None and state.current_location == place and state.long_term_goals == ["护主周全"]


@pytest.mark.asyncio
async def test_removed_character_is_not_in_the_confirmed_world(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    jiancheng = ids["Li Jiancheng"]
    with _client(mock_build_container, test_config, catalog) as client:
        assert client.delete(f"/api/worlds/{world_id}/setup/characters/{jiancheng}").status_code == 200
        setup = client.get(f"/api/worlds/{world_id}/setup").json()
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
        nodes = {n["id"] for n in client.get(f"/api/worlds/{world_id}/graph").json()["nodes"]}
    assert jiancheng not in nodes
    assert jiancheng not in await mock_build_container.agent_store.list_agent_ids(world_id)
    assert setup["relations"] == []
    [event] = setup["historical_events"]
    assert event["figure_ids"] == [ids["Li Shimin"]]


@pytest.mark.asyncio
async def test_a_failed_confirmation_leaves_the_world_unconfirmed_and_retryable(
    mock_build_container, test_config, monkeypatch,
) -> None:
    catalog = WorldCatalog(None)
    application = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")
    original = mock_build_container.agent_store.save_initial_agent_state

    async def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mock_build_container.agent_store, "save_initial_agent_state", fail)
    with pytest.raises(OSError):
        await application.confirm_world(world_id)
    assert not catalog.get(world_id).get("confirmed") and not application.has_session(world_id)

    monkeypatch.setattr(mock_build_container.agent_store, "save_initial_agent_state", original)
    world = await application.confirm_world(world_id)
    # The retry starts clean: one seed per participant, not the failed attempt's on top.
    shimin = next(a.agent_id for a in world.agents.values() if a.personality.soul.name == "Li Shimin")
    assert len(await _seeds(mock_build_container, world_id, shimin)) == 1
    assert catalog.get(world_id)["confirmed"]


@pytest.mark.asyncio
async def test_a_confirmation_failing_after_step_zero_leaves_the_setup_file_authoritative(
    mock_build_container, test_config, monkeypatch,
) -> None:
    catalog = WorldCatalog(None)
    application = NarrativeApplication(mock_build_container, test_config, catalog=catalog)
    world_id = await application.build_world("玄武门之变", template="changan_iso")

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(catalog, "set_confirmed", fail)
        with pytest.raises(OSError):
            await application.confirm_world(world_id)
    assert await mock_build_container.snapshot.load(world_id, 0) is not None

    agent_id = (await application.world_setup(world_id)).characters[0].agent_id
    await application.revise_setup(
        world_id,
        lambda setup, places: setup.with_character(agent_id, CharacterRevision(role="将军"), places=places),
    )

    assert (await application.world_setup(world_id)).character(agent_id).soul.role == "将军"


@pytest.mark.asyncio
async def test_confirming_moves_the_setup_from_its_file_into_step_zero(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id, ids = await _built_world(mock_build_container, test_config, catalog)
    snapshots = mock_build_container.snapshot
    with _client(mock_build_container, test_config, catalog) as client:
        client.patch(f"/api/worlds/{world_id}/setup/characters/{ids['Li Shimin']}", json={"role": "天策上将"})
        stored = await snapshots.load_world_setup(world_id)
        assert client.post(f"/api/worlds/{world_id}/confirm").status_code == 200
        # The frozen copy still answers the read-only review.
        roles = {c["agent_id"]: c["role"] for c in client.get(f"/api/worlds/{world_id}/agents").json()}
        assert client.get(f"/api/worlds/{world_id}/setup").status_code == 200
    step_zero = await snapshots.load(world_id, 0)
    assert await snapshots.load_world_setup(world_id) is None
    assert step_zero is not None and step_zero.setup == stored
    assert roles[ids["Li Shimin"]] == "天策上将"
