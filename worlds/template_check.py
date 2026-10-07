"""The template contract: what a ``worlds/templates/<name>/`` directory must hold.

A world's map is authored in Tiled, outside this codebase, and then dropped in as
a directory. Nothing about that authoring step can be assumed — which is why the
contract the engine and the renderer both rely on is written down HERE and
checked, rather than living as folklore about "how the Chang'an map does it".
Every rule below is one way a real, plausibly-authored map broke something:

* a non-isometric map         → the cast can only face the four isometric
                                headings, so a top-down city renders correctly
                                underfoot with everyone in it walking sideways
                                while facing a diagonal
* missing map properties      → the world builds with no name, era or description,
                                and dates itself on the wrong calendar
* an external ``.tsj``        → carries no ``image``, so the backend can neither
                                freeze it nor serve the frontend a URL for it —
                                those tiles render as holes (see
                                ``interaction/api/maps.py``'s asset routes)
* duplicate layer names       → a name that picks out two layers picks out
                                neither, and the map becomes one nobody can edit
                                with any confidence about where a change lands
* a malformed location object → ``TiledWorldConfig.get_places`` raises,
                                or silently drops the location from the world
* an unreachable location     → agents can be placed somewhere they can never
                                leave, or never reach
* a location off the road
  network                     → the renderer's A* has no cell to start from, so
                                every journey to or from it straight-lines through
                                the buildings
* a location with no open
  ground inside its rectangle → the map says people are here and the art leaves
                                nowhere for them to be, so everyone there is
                                staged on whatever ground is nearest OUTSIDE it
* a broken character manifest → the world renders with no people in it, or with
                                a pose that silently never draws (a frame name
                                the atlas does not carry)
* a portrait of someone else,
  or standing another way     → the close-up shows a different person from the
                                figure walking the map, or its callouts miss the body

Checks are pure data inspection: no Tiled, no Phaser, no LLM. They are run over
every template by ``tests/unit/test_world_templates.py``, so adding a map that
violates the contract fails the suite rather than the simulation.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from worlds.connections import declared_connections, derive_connections, main_component
from worlds.ground import GroundGrids, derive_ground
from worlds.tiled import (
    CHARACTERS_FILENAME, MAP_FILENAME, read_template_asset, template_dir, tiled_properties,
)

# How much open ground a location must have INSIDE its own rectangle. Four is one
# figure plus a little room, the floor at which a small gate room still stages a
# meeting. Set by what a rectangle must hold, not by what current art scores.
MIN_STANDING_CELLS = 4

# How far the renderer will look for road when it puts a location onto the network
# — ``pathfinding.snap`` searches rings out to r = 15. A location with no network
# cell inside that reach is one A* has no start for. Mirrored rather than guessed:
# a laxer number here passes maps the renderer cannot route on, a stricter one
# fails maps it handles.
SNAP_REACH = 15

# Map properties every template must carry. Each one is read by production code:
# see ``TiledWorldConfig.to_runtime_context`` / ``get_world_description`` /
# ``get_places``. A default in the reader is a guard against corruption,
# not a licence to leave the property out — an unset ``calendar`` silently dates
# a modern city by the classical Chinese calendar.
REQUIRED_MAP_PROPERTIES = (
    "world_name",
    "era_name",
    "calendar",
    "world_description",
    "start_month",
    "start_day",
    "start_hour",
    "seconds_per_tile",
)

# The one object layer a template exposes locations through. The parser finds
# location objects by their ``type``, not by the layer's name, so this is a
# UNIFORMITY rule rather than a functional one: every map is laid out the same
# way, so an author opening any template sees the same structure.
PLACE_LAYER = "place"

LOCATION_TYPE = "location"

# The one projection this renderer draws. Not a preference — a body's heading is one of the
# four ISOMETRIC directions (``skins.ts``: NE/SE/SW/NW) and the art is authored per heading,
# so on a top-down map every figure would walk an axis while facing a diagonal. A second
# projection would need a second heading vocabulary and a second art contract, so it is fixed
# and checked at the door: a map declaring anything else is refused rather than drawn as if it
# had declared this one (which would misplace every tile and location).
REQUIRED_ORIENTATION = "isometric"

# Per-location properties. ``capacity`` / ``is_public`` have parser defaults but
# are required here for the same reason as above: a template states its world.
REQUIRED_LOCATION_PROPERTIES = (
    "location_id",
    "description",
    "connections",
    "capacity",
    "is_public",
)


# ---- the character manifest's closed vocabularies -------------------------
#
# All three mirror the renderer and are duplicated here on purpose: this file is
# the only thing standing between a mistyped manifest and a world that renders
# with nobody in it, and a check that read the vocabulary from the file it is
# checking would pass anything.

# Every pose the renderer asks a body for. The list is closed by the engine, not
# by taste: it is the image of ``Deed`` under ``DEED_BODY``
# (frontend/src/phaser/deedPose.ts) plus the two states no deed names — standing
# and fallen. Art with a pose missing has an action it cannot show.
REQUIRED_POSES = (
    "idle",      # standing (also MOVE/REST, which name no pose of their own)
    "walk",      # driven by the token actually moving, not by a deed
    "down",      # vitality 0 — the one pose that stays on screen indefinitely
    "attack",    # strike / destroy / exert — one arm thrown, whatever it meets
    "hurt",      # the recoil of whoever was struck
    "shove",     # restrain: hands laid on, no blow
    "duck",      # the target of a restraint, anything that missed, and covert
    "hold",      # seize
    "interact",  # operate / work
    "talk",      # talk
    "show",      # send_message: the thing held out
)

# The two axes a body is chosen on (frontend/src/phaser/skins.ts). The renderer
# owns where the age boundaries fall; a manifest only has to name every bracket,
# because an agent whose bracket is absent has no body to draw.
GENDERS = ("male", "female")
AGE_BRACKETS = ("child", "young", "middle", "elder")

# A portrait is the body's close-up: the same person in the same clothes, standing the way the SE
# idle frame stands. It need not be that frame redrawn (a repaint is fine), so it is held to what
# carries over: the outline (another pose or heading fails it, and it keeps the chart's callouts on
# the body) and the coarse light-dark layout of face, hair and clothes (another person fails it).
# Don't drop the second: outline alone lets a similarly built neighbour through. Repaints score
# ≤ 8 on it, the nearest other body 11. Face detail and style are left to the eye (the cast bench).
PORTRAIT_MIN_OVERLAP = 0.94
PORTRAIT_MAX_TONE_DIFF = 9.0  # mean luminance difference at _TONE_PROBE, 0–255
_TONE_PROBE = (12, 32)  # small enough that texture averages out and only the layout remains
PORTRAIT_FACING = "SE"
_OUTLINE_PROBE = (48, 128)  # the common size both outlines are compared at
# Alpha at or below this is cut-out residue (generators leave a faint haze over the whole canvas),
# not body. Counting it frames the entire canvas and compares nothing.
PORTRAIT_ALPHA_FLOOR = 8

# Isometric travel directions, screen-relative. A pose may be given as one frame
# (the renderer mirrors it for west-facing travel) or as all four of these.
DIRECTIONS = ("NE", "SE", "SW", "NW")



def _atlas_frame_names(path: Path) -> set[str] | None:
    """Frame names in a Starling/Kenney XML atlas, or None if it cannot be read."""
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return None
    return {
        name for sub in root.iter("SubTexture")
        if (name := str(sub.get("name") or ""))
    }


def _atlas_frame_rect(path: Path, frame: str) -> tuple[int, int, int, int] | None:
    """(x, y, width, height) of one frame in a Starling/Kenney XML atlas."""
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError):
        return None
    for sub in root.iter("SubTexture"):
        if sub.get("name") == frame:
            try:
                x, y, w, h = (int(sub.get(k) or "") for k in ("x", "y", "width", "height"))
            except ValueError:
                return None
            return x, y, w, h
    return None


def _opaque(image: Image.Image) -> Image.Image | None:
    """The figure cropped to its opaque bounds (alpha above the residue floor), or None if empty."""
    rgba = image.convert("RGBA")
    box = rgba.getchannel("A").point(lambda a: 255 if a > PORTRAIT_ALPHA_FLOOR else 0).getbbox()
    return rgba.crop(box) if box else None


def _outline_overlap(a: Image.Image, b: Image.Image) -> float:
    """How much two figures' silhouettes coincide once both are scaled to the same box."""
    masks = [
        fig.resize(_OUTLINE_PROBE, Image.Resampling.BILINEAR).getchannel("A").point(lambda v: 255 if v > 127 else 0)
        for fig in (a, b)
    ]
    union = ImageChops.lighter(*masks).histogram()[255]
    return ImageChops.multiply(*masks).histogram()[255] / union if union else 0.0


def _tone_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean luminance difference of two figures' coarse light-dark layout."""
    smalls = [fig.resize(_TONE_PROBE, Image.Resampling.BOX) for fig in (a, b)]
    shared = ImageChops.multiply(*(s.getchannel("A").point(lambda v: 255 if v > 200 else 0) for s in smalls))
    if not shared.histogram()[255]:
        return 255.0
    return ImageStat.Stat(ImageChops.difference(*(s.convert("L") for s in smalls)), shared).mean[0]


def _portrait_drift(portrait: Path, sheet: Path, rect: tuple[int, int, int, int]) -> str | None:
    """Why this portrait is not a close-up of the body standing in its idle frame, or None."""
    with Image.open(portrait) as raw:
        if "A" not in raw.getbands():
            return "has no alpha channel; a portrait is a cut-out figure"
        figure = _opaque(raw)
    if figure is None:
        return "is fully transparent"
    x, y, w, h = rect
    with Image.open(sheet) as atlas:
        frame = _opaque(atlas.crop((x, y, x + w, y + h)))
    if frame is None:
        return "cannot be compared: the idle frame is empty"

    overlap = _outline_overlap(figure, frame)
    tone = _tone_diff(figure, frame)
    if overlap < PORTRAIT_MIN_OVERLAP or tone > PORTRAIT_MAX_TONE_DIFF:
        return (
            f"is not this body standing as its {PORTRAIT_FACING} idle frame stands (outline overlap "
            f"{overlap:.3f}, needs ≥ {PORTRAIT_MIN_OVERLAP}; light-dark layout difference {tone:.1f}, "
            f"needs ≤ {PORTRAIT_MAX_TONE_DIFF}) — another pose or heading moves the outline, another "
            "person the layout"
        )
    return None


def _pose_frames(spec: Any) -> tuple[list[str], list[str]]:
    """Flatten one pose entry to (frame names it references, problems with its shape).

    A pose is one frame, an animation, or either of those given per direction —
    the four shapes the art itself decides between (side-view art has no
    directions to give; isometric art draws all four).
    """
    if isinstance(spec, str):
        return [spec], []
    if isinstance(spec, list):
        if spec and all(isinstance(f, str) for f in spec):
            return list(spec), []
        return [], ["is an empty or non-string animation"]
    if isinstance(spec, dict):
        missing = [d for d in DIRECTIONS if d not in spec]
        if missing:
            return [], [f"is per-direction but omits {missing}"]
        frames: list[str] = []
        problems: list[str] = []
        for direction in DIRECTIONS:
            got, bad = _pose_frames(spec[direction])
            frames.extend(got)
            problems.extend(f"direction {direction} {p}" for p in bad)
        return frames, problems
    return [], ["is neither a frame name, an animation, nor a per-direction map"]


def _check_characters(root: Path) -> list[str]:
    """The cast's art: present, complete, and actually carrying the frames it names.

    A map with no people is not a lesser map, it is an empty one — so a missing
    manifest is an error here rather than a soft fallback in the renderer, where
    it would show up as a world nobody is standing in.
    """
    path = root / CHARACTERS_FILENAME
    if not path.is_file():
        return [
            f"no {CHARACTERS_FILENAME}; a template supplies the cast that walks its "
            "map (see worlds/templates/CHARACTER_ASSET_SPEC.md)"
        ]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"{CHARACTERS_FILENAME} is not valid JSON: {exc}"]
    if not isinstance(manifest, dict):
        return [f"{CHARACTERS_FILENAME} must hold an object"]

    problems: list[str] = []
    scale = manifest.get("scale")
    if not isinstance(scale, (int, float)) or not 0 < float(scale) <= 4:
        problems.append(
            f"characters: scale must be a positive number (got {scale!r}); it converts "
            "the art's own pixels to screen pixels, so only the art can state it"
        )

    atlases = manifest.get("atlases")
    if not isinstance(atlases, dict) or not atlases:
        return problems + ["characters: 'atlases' must map a body key to its art files"]

    frames_by_atlas: dict[str, set[str]] = {}
    for key, entry in atlases.items():
        if not isinstance(entry, dict):
            problems.append(f"characters: atlas {key!r} is not an object")
            continue
        image = str(entry.get("image") or "")
        descriptor = str(entry.get("atlas") or "")
        if not image or not descriptor:
            problems.append(f"characters: atlas {key!r} must name both 'image' and 'atlas'")
            continue
        for named in (image, descriptor):
            if not (root / named).is_file():
                problems.append(f"characters: atlas {key!r} names a missing file: {named}")
        names = _atlas_frame_names(root / descriptor)
        if names is None:
            problems.append(f"characters: atlas {key!r} descriptor is unreadable: {descriptor}")
        else:
            frames_by_atlas[key] = names

    bodies = manifest.get("bodies")
    if not isinstance(bodies, dict):
        problems.append("characters: 'bodies' must map gender → age bracket → atlas key")
        bodies = {}
    for gender in GENDERS:
        by_age = bodies.get(gender)
        if not isinstance(by_age, dict):
            problems.append(f"characters: bodies is missing gender {gender!r}")
            continue
        for bracket in AGE_BRACKETS:
            key = by_age.get(bracket)
            if not key:
                problems.append(
                    f"characters: no body for {gender}/{bracket} — an agent in that "
                    "bracket would have nothing to draw. Reuse a neighbouring "
                    "bracket's body if the art has no finer split"
                )
            elif key not in atlases:
                problems.append(f"characters: body {gender}/{bracket} names unknown atlas {key!r}")

    frames = manifest.get("frames")
    if not isinstance(frames, dict):
        return problems + ["characters: 'frames' must map each pose to its frame name(s)"]

    problems.extend(_check_portraits(root, atlases, frames.get("idle")))
    for pose in REQUIRED_POSES:
        if pose not in frames:
            problems.append(
                f"characters: pose {pose!r} is unmapped; the renderer asks every body "
                "for it, so the action it shows would silently draw nothing"
            )
            continue
        named, shape_problems = _pose_frames(frames[pose])
        problems.extend(f"characters: pose {pose!r} {p}" for p in shape_problems)
        # The check that matters: a frame name no atlas carries costs nothing at
        # load time and shows nothing at draw time.
        for atlas_key, available in frames_by_atlas.items():
            for frame in named:
                if frame not in available:
                    problems.append(
                        f"characters: pose {pose!r} names frame {frame!r}, which atlas "
                        f"{atlas_key!r} does not carry"
                    )
    return problems


def _check_portraits(root: Path, atlases: dict[str, Any], idle: Any) -> list[str]:
    """Each optional portrait exists and is a close-up of its body, standing as in the SE idle frame."""
    if isinstance(idle, dict):
        idle = idle.get(PORTRAIT_FACING)
    if isinstance(idle, list):
        idle = idle[0] if idle else None
    problems: list[str] = []
    for key, entry in atlases.items():
        if not isinstance(entry, dict) or not (named := str(entry.get("portrait") or "")):
            continue
        if not (root / named).is_file():
            problems.append(f"characters: atlas {key!r} names a missing portrait: {named}")
            continue
        sheet = root / str(entry.get("image") or "")
        descriptor = root / str(entry.get("atlas") or "")
        rect = _atlas_frame_rect(descriptor, idle) if isinstance(idle, str) else None
        if rect is None or not sheet.is_file():
            problems.append(f"characters: atlas {key!r} portrait has no idle frame to be checked against")
            continue
        if why := _portrait_drift(root / named, sheet, rect):
            problems.append(f"characters: atlas {key!r} portrait {why}")
    return problems


def _tile_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer in layers:
        if layer.get("type") == "group":
            out.extend(_tile_layers(layer.get("layers", [])))
        elif layer.get("type") == "tilelayer":
            out.append(layer)
    return out


def _object_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer in layers:
        if layer.get("type") == "group":
            out.extend(_object_layers(layer.get("layers", [])))
        elif layer.get("type") == "objectgroup":
            out.append(layer)
    return out


def _check_tilesets(doc: dict[str, Any], root: Path) -> list[str]:
    problems: list[str] = []
    for tileset in doc.get("tilesets", []):
        if "source" in tileset:
            problems.append(
                f"tileset references an external file ({tileset['source']}); "
                "export it embedded — an external tileset carries no image path, "
                "so the backend cannot freeze or serve its art"
            )
            continue
        image = str(tileset.get("image") or "")
        if not image:
            problems.append(f"tileset {tileset.get('name')!r} names no image")
        elif not (root / image).is_file():
            problems.append(f"tileset image is missing from the template: {image}")
    return problems


def _check_layers(doc: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    names = [layer["name"] for layer in _tile_layers(doc.get("layers", []))]
    for name, count in Counter(names).items():
        if count > 1:
            problems.append(
                f"tile layer name {name!r} is used {count} times; layer names must "
                "be unique across the whole map, groups included — a name shared by "
                "two layers tells neither apart when the map is edited"
            )
    place_layers = [
        layer for layer in _object_layers(doc.get("layers", []))
        if layer["name"] == PLACE_LAYER
    ]
    if len(place_layers) != 1:
        problems.append(
            f"expected exactly one object layer named {PLACE_LAYER!r}, "
            f"found {len(place_layers)}"
        )
    return problems


def _locations(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        obj
        for layer in _object_layers(doc.get("layers", []))
        for obj in layer.get("objects", [])
        if obj.get("type") == LOCATION_TYPE
    ]


def _check_orientation(doc: dict[str, Any]) -> list[str]:
    """The map must be drawn in the one projection the renderer draws."""
    orientation = str(doc.get("orientation") or "")
    if orientation == REQUIRED_ORIENTATION:
        return []
    return [
        f"orientation is {orientation or 'unset'!r}, but only {REQUIRED_ORIENTATION!r} "
        "is rendered — the cast is drawn per isometric heading (NE/SE/SW/NW) and "
        "has no poses for any other projection. Redraw the map isometric "
        "(128x64 recommended); see worlds/templates/README.md section 2"
    ]


def _check_locations(doc: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    objects = _locations(doc)
    if not objects:
        problems.append("no location objects (the engine has nowhere to simulate)")
        return problems

    # Tiled projects an isometric map's objects into a space whose unit on BOTH
    # axes is the tile HEIGHT (``IsometricRenderer``: ``tileX = x / tileHeight``).
    # Reading them in tile widths would halve every derived distance, and with it
    # every movement cost — ``worlds/tiled.py`` divides by the same field.
    unit = float(doc.get("tileheight", 32)) or 32.0
    cols = float(doc.get("width", 0))
    rows = float(doc.get("height", 0))

    graph: dict[str, list[str]] = {}
    for obj in objects:
        label = str(obj.get("name") or obj.get("id"))
        props = tiled_properties(obj)
        for key in REQUIRED_LOCATION_PROPERTIES:
            if key not in props:
                problems.append(f"location {label!r} is missing property {key!r}")
        location_id = str(props.get("location_id") or "").strip()
        if not location_id:
            problems.append(f"location {label!r} has an empty location_id")
            continue
        if not str(obj.get("name") or "").strip():
            problems.append(f"location {location_id!r} has no display name")
        if location_id in graph:
            problems.append(f"duplicate location_id: {location_id!r}")
        width = float(obj.get("width", 0))
        height = float(obj.get("height", 0))
        if width <= 0 or height <= 0:
            problems.append(
                f"location {location_id!r} has a zero-size rect; a location is an "
                "area agents stand in, not a point marker"
            )
        x0 = float(obj.get("x", 0)) / unit
        y0 = float(obj.get("y", 0)) / unit
        if x0 < 0 or y0 < 0 or x0 + width / unit > cols or y0 + height / unit > rows:
            problems.append(f"location {location_id!r} lies outside the map bounds")
        graph[location_id] = [
            part.split(":")[0].strip()
            for part in str(props.get("connections", "")).split(",")
            if part.split(":")[0].strip()
        ]

    for source, targets in graph.items():
        for target in targets:
            if target not in graph:
                problems.append(f"{source!r} connects to unknown location {target!r}")
            elif source not in graph[target]:
                problems.append(
                    f"connection {source!r} → {target!r} is one-way; "
                    "connections must be declared on both locations"
                )

    if graph:
        start = next(iter(graph))
        seen = {start}
        stack = [start]
        while stack:
            for neighbour in graph[stack.pop()]:
                if neighbour in graph and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if len(seen) != len(graph):
            stranded = sorted(set(graph) - seen)
            problems.append(f"locations are unreachable from the rest of the map: {stranded}")
    return problems


def _ground(doc: dict[str, Any], root: Path) -> GroundGrids | None:
    """The two grids the renderer will derive, or None if the art cannot be read.

    Derived by ``worlds.ground`` — the SAME code the renderer's staging and its A*
    network run on, so a check here is a check on what actually ships. A check on a
    separately-reimplemented grid would drift from it and start passing maps the
    renderer cannot use, which is the failure this module exists to prevent.
    """

    try:
        return derive_ground(doc, lambda image: read_template_asset(root, image))
    except Exception:  # noqa: BLE001 — a broken sheet is reported by _check_tilesets
        return None


def _components(grid: list[list[bool]], cols: int, rows: int) -> list[int]:
    """Sizes of the 4-connected true-regions, largest first."""
    seen = [[False] * cols for _ in range(rows)]
    out: list[int] = []
    for y0 in range(rows):
        for x0 in range(cols):
            if not grid[y0][x0] or seen[y0][x0]:
                continue
            seen[y0][x0] = True
            stack = [(x0, y0)]
            size = 0
            while stack:
                cx, cy = stack.pop()
                size += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            out.append(size)
    return sorted(out, reverse=True)


def _check_roads(doc: dict[str, Any], ground: GroundGrids | None) -> list[str]:
    """Every location must be able to get onto the road network.

    Stated as REACHABILITY, not as "the network is one piece": an art-derived network
    is pixel-accurate and strands a few fringe cells wherever road art clips a corner
    or runs off the edge, with no location touching them. Failing a map for that is a
    false alarm, and a checker that cries wolf is one an author learns to skip.

    What genuinely breaks is a location the renderer cannot put ON the network:
    ``pathfinding.snap`` gives up, A* has no start cell, and every journey to or
    from that place becomes a straight line across the buildings. A shattered map
    fails here too, because its locations end up stranded off the network.
    """
    if ground is None or not any(any(row) for row in ground.walkable):
        return []
    main = main_component(ground)
    if not main:
        return []
    unit = float(doc.get("tileheight", 32)) or 32.0
    stranded: list[str] = []
    for obj in _locations(doc):
        gx = float(obj.get("x", 0)) / unit + float(obj.get("width", 0)) / unit / 2
        gy = float(obj.get("y", 0)) / unit + float(obj.get("height", 0)) / unit / 2
        ax, ay = int(gx), int(gy)
        if not any(
            (ax + dx, ay + dy) in main
            for r in range(SNAP_REACH + 1)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
        ):
            stranded.append(str(tiled_properties(obj).get("location_id") or obj.get("name") or "?"))
    if stranded:
        total = sum(_components(ground.walkable, ground.cols, ground.rows))
        return [
            f"locations with no road network within reach: {', '.join(sorted(stranded))} "
            f"(the largest connected network is {len(main)} cells of {total}); the "
            "renderer cannot put them on it, so every journey to or from them "
            "straight-lines through the buildings. Tag every road tile with "
            "road=true, and check the road art actually joins up"
        ]
    return []


def _check_connections(doc: dict[str, Any], root: Path) -> list[str]:
    """No declared connection may contradict the ground it is drawn on.

    ``connections`` and the map art are two statements about the same fact — which
    places are next to each other — and only one of them is checkable, so this is
    where they are held together. An edge like "东宫"—"太极宫" across a palace wall
    makes the sim price the move as a short walk while the ground needs a 45-cell walk
    round by "玄武门", and the renderer has no honest way to draw it.

    Only the DECLARED side fails. An edge the map cannot support is always a defect —
    the author is asserting a passage that is not drawn. The reverse (ground allows
    it, the template stays silent) is a legitimate authoring choice and is reported
    by ``python -m worlds.connections <template>``, not failed here.
    """
    try:
        graph, pairs = derive_connections(doc, root)
    except Exception:  # noqa: BLE001 — unreadable art is reported by _check_tilesets
        return []
    if not graph:
        return []
    names = {
        str(tiled_properties(o).get("location_id") or ""): str(o.get("name") or "")
        for o in _locations(doc)
    }
    label = lambda lid: names.get(lid) or lid  # noqa: E731
    problems: list[str] = []
    for source, targets in declared_connections(doc).items():
        for target in targets:
            if source >= target or target not in graph:
                continue  # unordered pair, once; unknown ids are _check_locations' business
            walk, through = pairs.get((source, target), (None, set()))
            if walk is None:
                problems.append(
                    f"connection {label(source)!r} — {label(target)!r} has no walkable route "
                    "at all; the ground between them is solid"
                )
            elif through:
                via = " / ".join(sorted(label(t) for t in through))
                problems.append(
                    f"connection {label(source)!r} — {label(target)!r} is not adjacent on the "
                    f"map: the shortest walk is {walk} cells and goes through {via}. Declare it "
                    f"as the two legs it really is, or draw the passage the edge claims "
                    f"(python -m worlds.connections <template> --write rewrites the field)"
                )
    return problems


def _check_standing_room(doc: dict[str, Any], ground: GroundGrids | None) -> list[str]:
    """Every location needs open ground of its own to put people on.

    A location rectangle is drawn over a whole ward, and a ward can be solid
    housing with its streets left OUTSIDE the rectangle — in which case the map
    says people are here and the art leaves nowhere for them to be. The renderer
    widens its search past the rectangle rather than stand anyone on a roof, so
    this is not fatal — just wrong, and otherwise only discoverable by running the
    map and looking at it.
    """
    if ground is None:
        return []
    unit = float(doc.get("tileheight", 32)) or 32.0
    short: list[tuple[int, str]] = []
    for obj in _locations(doc):
        gx0, gy0 = float(obj.get("x", 0)) / unit, float(obj.get("y", 0)) / unit
        gw, gh = float(obj.get("width", 0)) / unit, float(obj.get("height", 0)) / unit
        cells = [
            (x, y)
            for y in range(max(0, int(gy0)), min(ground.rows, int(gy0 + gh) + 1))
            for x in range(max(0, int(gx0)), min(ground.cols, int(gx0 + gw) + 1))
            if gx0 <= x + 0.5 <= gx0 + gw and gy0 <= y + 0.5 <= gy0 + gh
        ]
        room = ground.standable_in(cells)
        if room < MIN_STANDING_CELLS:
            name = str(tiled_properties(obj).get("location_id") or obj.get("name") or "?")
            short.append((room, name))
    if not short:
        return []
    listed = ", ".join(f"{name} ({room})" for room, name in sorted(short))
    return [
        f"locations with fewer than {MIN_STANDING_CELLS} cells of open ground inside "
        f"their own rectangle: {listed}. Everyone there is staged on whatever ground "
        "is nearest OUTSIDE the rectangle, because the art leaves them nowhere to "
        "stand inside it — move the rectangle onto the street or courtyard, or widen "
        "it until it takes some in"
    ]


def check_template_dir(root: Path) -> list[str]:
    """Everything wrong with the template in *root*, as human-readable lines (empty = valid).

    No message may name ``root``: an import runs this against a staging directory
    before the template is allowed to land, and the path it ran in is an internal
    coordinate the author has no use for (``worlds/template_import.py``).
    """
    path = root / MAP_FILENAME
    if not path.is_file():
        return [f"no {MAP_FILENAME} at the template root"]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"{MAP_FILENAME} is not valid JSON: {exc}"]

    problems: list[str] = []
    properties = tiled_properties(doc)
    problems.extend(
        f"missing map property: {key!r}"
        for key in REQUIRED_MAP_PROPERTIES
        if key not in properties
    )
    problems.extend(_check_orientation(doc))
    problems.extend(_check_tilesets(doc, root))
    problems.extend(_check_layers(doc))
    problems.extend(_check_locations(doc))
    ground = _ground(doc, root)
    problems.extend(_check_roads(doc, ground))
    problems.extend(_check_connections(doc, root))
    problems.extend(_check_standing_room(doc, ground))
    problems.extend(_check_characters(root))
    return problems


def check_template(template: str) -> list[str]:
    """The same, for a template installed under ``worlds/templates/``."""
    return check_template_dir(template_dir(template))
