"""The Chang'an Tiled template is a well-formed, engine-ready world substrate.

These validate the .tmj on its OWN terms (locations present, connection graph
integral, metadata complete) rather than against a hand-coded oracle — the tmj
is the source of truth and must be free to evolve. They also lock the
requirements WorldInitializer places on a template (deepcopy + serialize) and the
multi-template seam (pick a template by name).
"""

from __future__ import annotations

import copy

import pytest

from core.interfaces.place import Place
from world.stored_config import StoredWorldConfig, serialize_world_config
from worlds.tiled import TiledWorldConfig, read_template_asset, template_asset_path

_CHANGAN_LOCATIONS = {
    # Faithful Tang Chang'an — important landmarks only (palaces, markets, major
    # temples, key wards + gates), positioned per the standard map of Tang Chang'an.
    "xuanwu_gate", "yeting_palace", "taiji_palace", "donggong", "daming_palace",
    "chengtian_gate", "huangcheng", "buzheng_fang", "chongren_fang", "zhuque_gate",
    "pingkang_fang", "xingqing_palace", "west_market", "east_market", "jinguang_gate",
    "chunming_gate", "daxingshan_temple", "daci_en_temple", "qujiang_pool", "mingde_gate",
}

# The ward layout follows the ART, not the schematic: palaces the art has no
# building for are absent, and residential wards it does show are named. The cast of locations is a per-template fact;
# everything structural below is not.
_CHANGAN_ISO_LOCATIONS = (
    _CHANGAN_LOCATIONS - {"daming_palace", "pingkang_fang", "daxingshan_temple"}
) | {
    "yining_fang", "shiliuwang_zhai", "west_grove", "huaide_fang",
    "yongyang_fang", "zhaoxing_fang", "quchi_fang",
    "jianfu_temple",  # west of the axis: the SMALL pagoda (Jianfu, not Daxingshan)
}
_TEMPLATE_LOCATIONS = {"changan_iso": _CHANGAN_ISO_LOCATIONS}
_CHANGAN_TEMPLATES = tuple(_TEMPLATE_LOCATIONS)


def _all_templates() -> list[str]:
    """Every template on disk, discovered rather than listed.

    The structural guarantees below are what makes a map a DROP-IN: satisfy them
    and the engine and renderer need no code for your setting. Listing template
    names here would mean a new one silently opts out of the very checks that
    promise are true of it — so the suite finds them instead."""
    from worlds.tiled import list_templates

    return list_templates(include_examples=True)


ALL_TEMPLATES = _all_templates()


def test_the_repo_ships_a_setting_unlike_the_others() -> None:
    """The invariants above only prove PORTABILITY if something unlike Chang'an is
    among the things they are applied to; otherwise a theme assumption could creep into the
    shared path, true of every sample.

    `metro` is the counterweight: a modern city on a modern calendar, its own layer
    set and its own cast, sharing one code path with a Tang capital.

    Not guarded here: projection is a fixed contract (see the orientation test below);
    tile size is free in the code, but every shipped map uses 128×64, so that freedom is
    unproven until a map at another grid size exists."""
    others = [t for t in ALL_TEMPLATES if "changan" not in t]
    assert others, "every template is the same setting — portability is untested"

    identities = set()
    for name in ALL_TEMPLATES:
        config = TiledWorldConfig(template=name)
        context = config.to_runtime_context()
        identities.add((context["era_name"], context["calendar"]))
    assert len(identities) > 1, "every template is the same era on the same calendar"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_any_template_is_engine_ready(template: str) -> None:
    """The whole contract a setting must meet, applied to every setting present."""
    config = TiledWorldConfig(template=template)
    entities = config.get_places()
    assert entities, f"{template}: no locations"

    ids = set(entities)
    for lid, entity in entities.items():
        assert isinstance(entity, Place), lid
        assert entity.name and entity.description, lid
        assert entity.capacity > 0, lid
        assert entity.connections, f"{template}: {lid} has no connections"
        for neighbour, weight in entity.connections.items():
            assert neighbour in ids, f"{template}: {lid} → unknown {neighbour}"
            assert isinstance(weight, int) and weight > 0

    assert config.get_world_description()
    context = config.to_runtime_context()
    assert context["world_name"] and context["template"] == template
    # Every location is reachable — an isolated one strands whoever starts there.
    start = next(iter(entities))
    seen, queue = {start}, [start]
    while queue:
        for neighbour in entities[queue.pop()].connections:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    assert seen == ids, f"{template}: unreachable {ids - seen}"


@pytest.mark.parametrize("template", _CHANGAN_TEMPLATES)
def test_changan_template_locations_and_connection_graph_are_wellformed(template: str) -> None:
    entities = TiledWorldConfig(template=template).get_places()
    assert set(entities) == _TEMPLATE_LOCATIONS[template]

    ids = set(entities)
    for lid, e in entities.items():
        assert isinstance(e, Place)
        assert e.name and e.description, lid
        assert e.capacity > 0, lid
        assert e.connections, f"{lid} has no connections"
        for nbr, weight in e.connections.items():
            assert nbr in ids, f"{lid} → unknown location {nbr}"
            assert isinstance(weight, int) and weight > 0, f"{lid}->{nbr} weight"


@pytest.mark.parametrize("template", _CHANGAN_TEMPLATES)
def test_changan_template_metadata_present(template: str) -> None:
    cfg = TiledWorldConfig(template=template)
    assert cfg.get_world_description()
    assert cfg.get_location_aliases()  # non-empty alias table
    ctx = cfg.to_runtime_context()
    # A map states its name in plain language — it is what a user picks from and
    # what the theme analyzer takes as a naming cue, not a code-layer handle.
    assert ctx["world_name"] == "长安"
    assert ctx["era_name"] == "大唐"  # neutral era (specific reign comes from theme)
    assert ctx["template"] == template


def _authored_connections(template: str) -> dict[str, list[str]]:
    """The `connections` strings exactly as authored in the .tmj — NOT the parsed
    graph. The parser drops an unknown neighbour silently, so anything read back
    off `WorldEntity.connections` is dangling-free by construction and can never
    witness the dangling edges the next test looks for."""
    doc = TiledWorldConfig(template=template).render_map()
    assert doc is not None
    out: dict[str, list[str]] = {}

    def walk(layers: list) -> None:
        for layer in layers:
            walk(layer.get("layers", []))
            for obj in layer.get("objects", []):
                if obj.get("type") != "location":
                    continue
                props = {p["name"]: p["value"] for p in obj.get("properties", [])}
                out[props["location_id"]] = [
                    n.strip() for n in str(props.get("connections", "")).split(",") if n.strip()
                ]

    walk(doc["layers"])
    return out


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_authored_connections_are_neither_dangling_nor_one_way(template: str) -> None:
    """Deleting a location in Tiled leaves its NEIGHBOURS pointing at a ghost, and
    `get_places` answers by quietly dropping those edges — so the map loses
    connectivity with nothing anywhere reporting it. Likewise a neighbour listed on
    one side only yields a one-way street the movement graph never intended. Both
    are only visible against the authored text."""
    authored = _authored_connections(template)
    assert set(authored) == set(TiledWorldConfig(template=template).get_places())

    dangling = {lid: [n for n in nbrs if n not in authored] for lid, nbrs in authored.items()}
    assert not any(dangling.values()), f"connections name deleted locations: {dangling}"

    one_way = [(lid, n) for lid, nbrs in authored.items() for n in nbrs if lid not in authored[n]]
    assert not one_way, f"edges listed on one side only: {one_way}"


@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_no_alias_points_at_a_location_that_no_longer_exists(template: str) -> None:
    """An alias for a deleted location fails WORSE than a dangling connection: it
    silently falls through to `None`, so the name a character would naturally use
    resolves to "no such place" rather than to anywhere. Nothing logs it, and the
    parsed entity graph can't show it — the alias table has to be checked directly."""
    cfg = TiledWorldConfig(template=template)
    known = set(cfg.get_places())
    aliases = cfg.get_location_aliases()
    dangling = {alias: target for alias, target in aliases.items() if target not in known}
    assert not dangling, f"aliases point at deleted locations: {dangling}"
    # …and each one really does resolve, rather than merely naming a live id.
    assert all(cfg.resolve_location_id(alias) == target for alias, target in aliases.items())


@pytest.mark.parametrize("template", _CHANGAN_TEMPLATES)
def test_every_location_is_reachable(template: str) -> None:
    """One unreachable ward is a place the simulation can strand an agent in."""
    entities = TiledWorldConfig(template=template).get_places()
    start = next(iter(entities))
    seen, queue = {start}, [start]
    while queue:
        for nbr in entities[queue.pop()].connections:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    assert seen == set(entities), f"unreachable: {set(entities) - seen}"


def test_isometric_template_nests_its_locations_in_a_group_layer() -> None:
    """The isometric map files its object layer inside a Tiled GROUP, so the layer
    walk has to recurse. A non-recursive one finds zero locations on a map that is
    perfectly valid — this pins the recursion, not just the count."""
    doc = TiledWorldConfig(template="changan_iso").render_map()
    assert doc is not None
    top_level = [layer for layer in doc["layers"] if layer.get("type") == "objectgroup"]
    assert not top_level, "no object layer should sit at the top level, or this test is vacuous"
    found = TiledWorldConfig(template="changan_iso").get_places()
    assert set(found) == _CHANGAN_ISO_LOCATIONS  # a non-recursive walk finds none of them


def test_isometric_object_coordinates_are_scaled_by_tile_height() -> None:
    """Tiled projects an isometric map's object coordinates into a space whose unit
    is the TILE HEIGHT on both axes, not the obvious width.

    Reading an isometric map with the width (128 here vs a 64 height) halves every
    derived distance, and so every edge's walking time."""
    doc = TiledWorldConfig(template="changan_iso").render_map()
    assert doc is not None
    assert doc["orientation"] == "isometric"
    assert doc["tilewidth"] != doc["tileheight"]  # the two divisors must disagree

    places = TiledWorldConfig(template="changan_iso").get_places()
    # Zhuque Gate → Mingde Gate is ~26 tiles × 144 s ≈ 62 min; reading in tilewidth halves it.
    assert places["zhuque_gate"].connections["mingde_gate"] >= 50 * 60, (
        "distances halved — object coords were read in tile WIDTHS"
    )


def test_each_map_opens_at_its_own_moment() -> None:
    """The opening time of day is declared by each map; there is no deployment-level override.

    One config can't know when every map should open, and a global override would force a modern
    city into a Tang dawn. This asserts the maps' times aren't all equal; an override would
    collapse them to one value.
    """
    clocks = {
        name: tuple(
            TiledWorldConfig(template=name).to_runtime_context()[key]
            for key in ("start_month", "start_day", "start_hour")
        )
        for name in ALL_TEMPLATES
    }
    assert len(set(clocks.values())) > 1, f"every map opens at the same moment: {clocks}"


@pytest.mark.parametrize("template", _CHANGAN_TEMPLATES)
def test_resolve_location_id_via_generic_contract(template: str) -> None:
    cfg = TiledWorldConfig(template=template)
    assert cfg.resolve_location_id("玄武门") == "xuanwu_gate"  # by name
    assert cfg.resolve_location_id("donggong") == "donggong"  # by id
    assert cfg.resolve_location_id("大雁塔") == "daci_en_temple"  # by alias
    assert cfg.resolve_location_id("不存在之地") is None


@pytest.mark.parametrize("template", _CHANGAN_TEMPLATES)
def test_deepcopy_and_serialize_roundtrip(template: str) -> None:
    """Initializer deep-copies the template then persists the copy; restore must
    rebuild the identical substrate from the serialized data alone."""
    cfg = TiledWorldConfig(template=template)
    clone = copy.deepcopy(cfg)  # must not raise
    assert set(clone.get_places()) == set(cfg.get_places())

    restored = StoredWorldConfig(serialize_world_config(cfg))
    original = cfg.get_places()
    rebuilt = restored.get_places()
    assert set(rebuilt) == set(original)
    for lid, entity in original.items():
        assert rebuilt[lid].connections == entity.connections
        assert rebuilt[lid].name == entity.name



@pytest.mark.parametrize("template", ALL_TEMPLATES)
def test_a_template_is_one_self_contained_directory(template: str) -> None:
    """Map and art live together under `templates/<name>/`, mirroring the Tiled
    project that authored them. Adding a setting is copying one directory in;
    there is no second place where half of a map lives."""
    from worlds.tiled import MAP_FILENAME, template_dir

    home = template_dir(template)
    assert (home / MAP_FILENAME).is_file()
    # Every image the map names resolves INSIDE that directory — that is what lets
    # the asset route serve art by the path written in the .tmj.
    doc = TiledWorldConfig(template=template).render_map()
    assert doc is not None
    images = [str(ts["image"]) for ts in doc["tilesets"] if ts.get("image")]
    assert images
    for image in images:
        assert (home / image).is_file(), f"{template}: missing art {image}"


def test_template_name_cannot_escape_the_templates_directory() -> None:
    """The name arrives from a world's persisted config and is pasted into a path.
    A traversing value must land on a missing template, not on another directory."""
    from worlds.tiled import TEMPLATES_DIR, template_dir

    for hostile in ("../worlds", "../../etc", "changan/../../worlds", "/etc"):
        resolved = template_dir(hostile)
        assert not resolved.exists() or resolved.parent == TEMPLATES_DIR.resolve()
        with pytest.raises((FileNotFoundError, IsADirectoryError, PermissionError, OSError)):
            TiledWorldConfig(template=hostile).get_places()


def test_template_selectable_by_name_via_factory() -> None:
    """Multi-template seam: a template is chosen by name through config/factory
    (`config: tiled, params: {template: <name>}`). Adding a setting = drop a .tmj
    + set the param, no new code. An unknown template surfaces at parse time."""
    import worlds  # noqa: F401 — triggers registration
    from core.factory import ComponentKind, ProviderFactory

    cfg = ProviderFactory.create("tiled", kind=ComponentKind.WORLD, template="changan_iso")
    assert cfg.to_runtime_context()["template"] == "changan_iso"
    assert len(cfg.get_places()) == 25

    missing = ProviderFactory.create("tiled", kind=ComponentKind.WORLD, template="atlantis")
    with pytest.raises(FileNotFoundError):
        missing.get_places()


def test_template_asset_stays_inside_its_template(tmp_path) -> None:
    root = tmp_path / "tpl"
    (root / "tiles").mkdir(parents=True)
    (root / "tiles" / "a.png").write_bytes(b"png")
    (tmp_path / "secret.txt").write_text("no")

    assert read_template_asset(root, "tiles/a.png") == b"png"
    assert template_asset_path(root, "../secret.txt") is None
    assert template_asset_path(root, str(tmp_path / "secret.txt")) is None
    assert template_asset_path(root, "tiles/missing.png") is None
