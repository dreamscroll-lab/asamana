"""Every world template must satisfy the template contract.

Maps are authored outside this codebase and dropped in as directories, so the
contract has no author to enforce it at write time — this suite is where a map
that breaks it gets caught, instead of the simulation. Parametrized over
whatever templates are present: adding a map automatically adds its coverage,
and nothing here knows any particular map by name.
"""

from __future__ import annotations

import json
import shutil

import pytest
from PIL import Image

from worlds.template_check import (
    AGE_BRACKETS,
    GENDERS,
    REQUIRED_ORIENTATION,
    REQUIRED_POSES,
    _atlas_frame_rect,
    _check_orientation,
    _check_portraits,
    check_template,
    check_template_dir,
)
from worlds.tiled import (
    CAST_ART_FIELDS,
    CHARACTERS_DIRNAME,
    CHARACTERS_FILENAME,
    TiledWorldConfig,
    list_templates,
    template_dir,
)

# Examples included on purpose: a reference map's whole job is to be a worked
# example of the contract, so it is the last thing that should be exempt from it.
TEMPLATES = list_templates(include_examples=True)


def test_templates_are_discoverable() -> None:
    """A world has to be buildable on something."""
    assert TEMPLATES, "no world templates found under worlds/templates/"


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_satisfies_contract(template: str) -> None:
    problems = check_template(template)
    assert not problems, "\n".join([f"template {template!r}:", *problems])


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_loads_as_a_world_config(template: str) -> None:
    """The contract check reads the file; this proves the engine's own reader agrees."""
    config = TiledWorldConfig(template=template)

    entities = config.get_places()
    assert entities, f"{template}: no locations"
    assert all(entity.connections for entity in entities.values()), (
        f"{template}: some locations have no connections and are unreachable"
    )

    context = config.to_runtime_context()
    assert context["template"] == template
    assert context["world_name"], f"{template}: empty world_name"
    assert context["era_name"], f"{template}: empty era_name"
    assert context["calendar"], f"{template}: empty calendar"
    assert config.get_world_description(), f"{template}: empty world_description"

    # Aliases must resolve to real locations — a stale alias silently sends an
    # agent nowhere.
    for alias, canonical in config.get_location_aliases().items():
        assert canonical in entities, f"{template}: alias {alias!r} → unknown {canonical!r}"


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_ships_a_cast(template: str) -> None:
    """A map comes with the people who walk it, and the renderer needs no knowledge
    of them beyond this manifest.

    The contract check already proves the manifest is internally sound (every
    bracket filled, every file present, every frame name real). What this adds is
    that the reader the backend actually uses agrees, and that the manifest is
    self-describing enough for a renderer that hardcodes nothing: it must state its
    own scale, since only the art knows how tall it was drawn.
    """
    manifest = TiledWorldConfig(template=template).render_characters()
    assert manifest is not None, f"{template}: no cast art"

    assert isinstance(manifest.get("scale"), (int, float)) and manifest["scale"] > 0, (
        f"{template}: the manifest must state its own scale — the renderer has no "
        "default to fall back on, and a default would be a guess about someone's art"
    )
    bodies = manifest["bodies"]
    atlases = manifest["atlases"]
    for gender in GENDERS:
        for bracket in AGE_BRACKETS:
            body = bodies[gender][bracket]
            assert body in atlases, f"{template}: {gender}/{bracket} → unknown atlas {body!r}"
    for pose in REQUIRED_POSES:
        assert pose in manifest["frames"], f"{template}: pose {pose!r} is unmapped"


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_is_isometric(template: str) -> None:
    """One projection, and it is not a preference.

    A figure's heading is one of the four ISOMETRIC directions and the cast art is
    drawn per heading, so no other projection has poses to render: a top-down map
    would draw its ground correctly with every person on it facing a diagonal while
    walking an axis. That is the state this pins shut."""
    doc = TiledWorldConfig(template=template).render_map()
    assert doc is not None
    assert doc["orientation"] == REQUIRED_ORIENTATION, (
        f"{template}: {doc['orientation']!r} — the cast has no poses for it"
    )


@pytest.mark.parametrize("orientation", ["orthogonal", "hexagonal", "staggered", "", None])
def test_a_map_in_any_other_projection_is_refused(orientation: str | None) -> None:
    """The projection is enforced AT THE DOOR, which is the whole point of fixing it.

    With the renderer's orientation branches gone, an unrefused non-isometric map is
    not a map that renders badly — it is one silently projected as if it had declared
    isometric, misplacing every tile and every location. So the check must reject it
    rather than the renderer discovering it, and an unset orientation counts too: a
    missing field must not read as consent."""
    problems = _check_orientation({"orientation": orientation} if orientation is not None else {})
    assert problems, f"{orientation!r} was accepted"
    assert REQUIRED_ORIENTATION in problems[0]  # the message says what to do instead


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_art_is_self_contained(template: str) -> None:
    """A template holds the art its map names, so it is a drop-in directory."""
    config = TiledWorldConfig(template=template)
    root = config.render_assets_dir()
    render_map = config.render_map()
    assert render_map is not None

    tilesets = render_map["tilesets"]
    assert tilesets, f"{template}: map names no tilesets"
    for tileset in tilesets:
        image = tileset.get("image")
        assert image, (
            f"{template}: tileset {tileset.get('name') or tileset.get('source')!r} has no "
            "image — the backend can neither freeze nor serve it"
        )
        assert (root / image).is_file(), f"{template}: missing art {image}"

    # The cast's sheets resolve off the SAME root as the map's tilesets — one asset
    # namespace per world, so one freeze and one route reach all of it.
    manifest = config.render_characters()
    assert manifest is not None
    for key, entry in manifest["atlases"].items():
        for field in CAST_ART_FIELDS:
            if field not in entry:
                continue
            assert (root / entry[field]).is_file(), (
                f"{template}: cast atlas {key!r} names missing {field} {entry[field]!r}"
            )


def test_a_verdict_never_names_the_directory_it_ran_in(tmp_path) -> None:
    """An import runs this against a staging directory and shows the verdict to the
    author, who has no use for a path inside the server's temp tree."""
    problems = check_template_dir(tmp_path)

    assert len(problems) == 1, problems
    assert str(tmp_path) not in problems[0]


def test_no_two_maps_call_themselves_the_same_thing() -> None:
    """`world_name` is what the map picker shows; the directory name never reaches a
    user. Two maps sharing one are two rows nobody can tell apart — the import refuses
    it, and this holds the same line for a map copied in by hand."""
    seen: dict[str, str] = {}
    for template in TEMPLATES:
        name = TiledWorldConfig(template=template).to_runtime_context()["world_name"]
        assert name not in seen, f"{template!r} and {seen[name]!r} both call themselves {name!r}"
        seen[str(name)] = template


# ---- portraits: a close-up of the same person, standing the same way ---------

PORTRAIT_TEMPLATES = [
    t for t in TEMPLATES
    if sum("portrait" in e for e in json.loads((template_dir(t) / CHARACTERS_FILENAME).read_text())["atlases"].values()) >= 2
]


def _cast_copy(template: str, tmp_path):
    """The template's cast directory, copied so a test can tamper with it."""
    shutil.copytree(template_dir(template) / CHARACTERS_DIRNAME, tmp_path / CHARACTERS_DIRNAME)
    manifest = json.loads((tmp_path / CHARACTERS_FILENAME).read_text())
    with_portrait = [k for k, e in manifest["atlases"].items() if "portrait" in e]
    return manifest, with_portrait


def _refused(problems: list[str], key: str, why: str) -> bool:
    return any(repr(key) in p and why in p for p in problems)


@pytest.mark.parametrize("template", PORTRAIT_TEMPLATES)
def test_another_bodys_portrait_is_refused(template: str, tmp_path) -> None:
    """The drift a portrait check exists to prevent: a close-up of someone else."""
    manifest, (a, b, *_) = _cast_copy(template, tmp_path)
    atlases = manifest["atlases"]
    atlases[a]["portrait"] = atlases[b]["portrait"]
    problems = _check_portraits(tmp_path, atlases, manifest["frames"]["idle"])
    assert _refused(problems, a, "is not this body"), problems
    assert not any(repr(b) in p for p in problems), problems


@pytest.mark.parametrize("template", PORTRAIT_TEMPLATES)
def test_someone_else_in_the_right_stance_is_refused(template: str, tmp_path) -> None:
    """The right outline filled with another person's face, hair and clothes: the case an
    outline-only check lets through."""
    manifest, (a, b, *_) = _cast_copy(template, tmp_path)
    atlases = manifest["atlases"]
    own = tmp_path / atlases[a]["portrait"]
    with Image.open(own) as mine, Image.open(tmp_path / atlases[b]["portrait"]) as theirs:
        swapped = theirs.convert("RGBA").resize(mine.size)
        swapped.putalpha(mine.getchannel("A"))
    swapped.save(own)
    problems = _check_portraits(tmp_path, atlases, manifest["frames"]["idle"])
    assert _refused(problems, a, "is not this body"), problems


@pytest.mark.parametrize("template", PORTRAIT_TEMPLATES)
def test_the_same_body_in_another_pose_is_refused(template: str, tmp_path) -> None:
    """Same person, same clothes, not standing: a walk frame blown up to portrait size."""
    manifest, (a, *_) = _cast_copy(template, tmp_path)
    entry = manifest["atlases"][a]
    walk = manifest["frames"]["walk"]["SE"][1]
    x, y, w, h = _atlas_frame_rect(tmp_path / entry["atlas"], walk)
    with Image.open(tmp_path / entry["image"]) as sheet:
        frame = sheet.convert("RGBA").crop((x, y, x + w, y + h))
    frame.resize((w * 5, h * 5), Image.Resampling.LANCZOS).save(tmp_path / entry["portrait"])
    problems = _check_portraits(tmp_path, manifest["atlases"], manifest["frames"]["idle"])
    assert _refused(problems, a, "is not this body"), problems


@pytest.mark.parametrize("template", PORTRAIT_TEMPLATES)
def test_a_portrait_without_alpha_is_refused(template: str, tmp_path) -> None:
    manifest, (a, *_) = _cast_copy(template, tmp_path)
    path = tmp_path / manifest["atlases"][a]["portrait"]
    with Image.open(path) as im:
        im.convert("RGB").save(path)
    problems = _check_portraits(tmp_path, manifest["atlases"], manifest["frames"]["idle"])
    assert _refused(problems, a, "no alpha"), problems


@pytest.mark.parametrize("template", PORTRAIT_TEMPLATES)
def test_the_same_person_standing_differently_is_refused(template: str, tmp_path) -> None:
    """Same pixels, other stance: a gap cut between the legs keeps every tone the layout check
    compares, so only the outline can catch it."""
    manifest, (a, *_) = _cast_copy(template, tmp_path)
    path = tmp_path / manifest["atlases"][a]["portrait"]
    with Image.open(path) as im:
        figure = im.convert("RGBA")
    x0, y0, x1, y1 = figure.getchannel("A").getbbox()
    w, h = x1 - x0, y1 - y0
    alpha = figure.getchannel("A")
    alpha.paste(0, (x0 + int(w * 0.3), y0 + int(h * 0.45), x0 + int(w * 0.7), y1 - int(h * 0.02)))
    figure.putalpha(alpha)
    figure.save(path)
    problems = _check_portraits(tmp_path, manifest["atlases"], manifest["frames"]["idle"])
    assert _refused(problems, a, "is not this body"), problems
