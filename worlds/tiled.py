"""WorldConfig sourced from a Tiled (.tmj) template.

The tmj IS the world's spatial representation (see the Tiled-world design): its
location objects (``type="location"``) define the locations + weighted connections
the engine simulates on, and map-level properties carry the substrate metadata. There are no
hand-coded ``WorldConfig`` classes — the map is authored/edited in Tiled, not
Python, and both the engine (here) and the 2D renderer read the same file.

Only STABLE infrastructure (locations) is read from the template. Theme cast
(agents/entities) is added at initialization from ThemeAnalysis, not baked into
the reusable template.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.place import Place
from core.interfaces.world_config import WorldConfig

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# A template is a SELF-CONTAINED DIRECTORY: the map plus the art it names, laid
# out exactly as the Tiled project that authored it. Adding a setting is copying
# one directory in; removing it is deleting one — there is no second place where
# half of a map lives. The map file has a fixed name because the directory
# already carries the template's identity.
MAP_FILENAME = "map.tmj"

# The cast's art manifest, beside the map inside the same directory. A body is
# dressed for a PERIOD and a period is what a map is of, so the figures are part
# of the same drop-in artifact as the ground: frozen with it, served with it, and
# replaced with it. See ``worlds/templates/CHARACTER_ASSET_SPEC.md``.
CHARACTERS_DIRNAME = "characters"
CHARACTERS_FILENAME = f"{CHARACTERS_DIRNAME}/characters.json"
# The files one body's entry under ``atlases`` names, by field. Whatever freezes the cast and
# whatever serves it read this one list, so a new kind of art can't reach one and miss the other.
# ``portrait`` is optional: a high-resolution close-up of the body, standing as its idle frame does.
CAST_ART_FIELDS = ("image", "atlas", "portrait")

# Fallback when the map doesn't set `seconds_per_tile`. It guards against broken maps; it isn't a
# shortcut. How long a tile takes is each map's own fact (a tile can be a corridor or a whole ward,
# orders of magnitude apart), and leaving it out distorts travel time across the map. template_check rejects that.
DEFAULT_SECONDS_PER_TILE = 180


def tiled_properties(holder: dict[str, Any]) -> dict[str, Any]:
    """A Tiled map / layer / object / tile's ``properties`` list as a name → value dict."""
    return {p.get("name"): p.get("value") for p in holder.get("properties") or []}


def template_dir(template: str) -> Path:
    """Directory holding a template's map and its assets.

    The name is confined to a single path segment: it reaches this function from
    a world's persisted config, and a value like ``../../etc`` must resolve to a
    missing template rather than to somewhere else on disk.
    """
    resolved = (TEMPLATES_DIR / template).resolve()
    if resolved.parent != TEMPLATES_DIR.resolve():
        return TEMPLATES_DIR / "__invalid__"
    return resolved


def template_asset_path(root: Path, image: str) -> Path | None:
    """The file ``image`` names inside ``root``, or None when it is missing or escapes ``root``.

    ``image`` comes from a map document or a URL, so ``..`` and absolute paths are
    expected input: resolving first and then requiring the result to stay under
    ``root`` is the check that keeps them from reading elsewhere on disk.
    """
    path = (root / image).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path


def read_template_asset(root: Path, image: str) -> bytes | None:
    path = template_asset_path(root, image)
    return path.read_bytes() if path is not None else None


# A leading underscore marks a template as an EXAMPLE: a reference map that
# exists to be read and tested against, not to build a world on. It is a real
# template in every other respect — it must satisfy the same contract, and the
# suite holds it to it — but it is not offered as somewhere a story could happen.
EXAMPLE_PREFIX = "_"


def list_templates(*, include_examples: bool = False) -> list[str]:
    """Templates a world can be built on, sorted by name.

    A template is discovered by being a directory that holds a map — that is the
    whole registration mechanism, and deliberately so: adding a setting is
    copying one directory in, with nothing to also declare in code or config.

    ``include_examples`` widens this to reference maps as well. Only the contract
    checks want that: what a reference map is FOR is to be a worked example of
    the rules, so it has to be held to them — but a placeholder city drawn in
    flat color has no business in the list a user picks a world from.
    """
    if not TEMPLATES_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in TEMPLATES_DIR.iterdir()
        if entry.is_dir()
        and (entry / MAP_FILENAME).is_file()
        and (include_examples or not entry.name.startswith(EXAMPLE_PREFIX))
    )


def _parse_neighbors(raw: object) -> list[str]:
    """Parse a ``connections`` string → the list of neighbor location ids.

    Connections are EXPLICIT edges (which rooms link), authored as a comma list
    (``"donggong,xuanwu_gate,market"``). The travel COST is not authored — it is
    derived from the map geometry (see ``get_places``). Any ``:weight`` suffix
    is tolerated and ignored, so older templates still load.
    """
    out: list[str] = []
    for part in str(raw).split(","):
        nbr = part.split(":")[0].strip()
        if nbr:
            out.append(nbr)
    return out


def _parse_aliases(raw: object) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(raw).split(","):
        part = part.strip()
        if ":" not in part:
            continue
        alias, _, canon = part.partition(":")
        if alias.strip() and canon.strip():
            out[alias.strip()] = canon.strip()
    return out


@ProviderFactory.register("tiled", kind=ComponentKind.WORLD)
class TiledWorldConfig(WorldConfig):
    """A ``WorldConfig`` parsed from a Tiled ``.tmj`` template.

    Location objects (``type="location"``, in any object layer) become ``Place``s;
    map properties supply description / era / clock / aliases. Parsing is lazy and
    cached, and the instance stays ``deepcopy``-able and its places JSON-serializable
    — the two hard requirements ``WorldInitializer`` places on a template.

    Multi-template seam: a template is picked **by name**, per world — the API
    caller names one, or the build reads the theme and chooses (see
    ``TemplateSelector``). Adding a new setting (e.g. a modern-city map) is just
    dropping ``worlds/templates/<name>/`` — no new code, no per-template subclass.

    ``template`` is required and has no default: every world names the map it is
    built on. A name baked in here would be one setting silently standing in
    wherever a caller failed to choose — and a world on the wrong substrate is
    wrong under every step that follows, so that must be an error instead.

    The clock (``start_month`` / ``start_day`` / ``start_hour``) is authored on the
    map itself. There is deliberately **no** deployment-level override: one config
    cannot know what hour every map opens at, so an override would silently force
    a modern city to start on a Tang-dynasty morning.
    """

    def __init__(self, *, template: str) -> None:
        self._template = str(template)
        self._path = template_dir(self._template) / MAP_FILENAME
        self._map: dict[str, Any] | None = None
        self._places_cache: dict[str, Place] = {}

    # ---- lazy tmj read -----------------------------------------------------
    def _tiled_map(self) -> dict[str, Any]:
        if self._map is None:
            self._map = json.loads(self._path.read_text(encoding="utf-8"))
        return self._map

    def _map_props(self) -> dict[str, Any]:
        return tiled_properties(self._tiled_map())

    def _location_objects(self) -> list[dict[str, Any]]:
        """Every ``type="location"`` object, from any object layer at any depth.

        Tiled lets an author file layers into GROUPS, and a map organized that way
        nests its object layer under a ``"type": "group"`` entry rather than listing
        it at the top level — so the walk has to recurse or it silently finds no
        locations at all on an otherwise valid template.
        """
        objects: list[dict[str, Any]] = []

        def walk(layers: list[dict[str, Any]]) -> None:
            for layer in layers:
                if layer.get("type") == "group":
                    walk(layer.get("layers", []))
                elif layer.get("type") == "objectgroup":
                    objects.extend(
                        o for o in layer.get("objects", []) if o.get("type") == "location"
                    )

        walk(self._tiled_map().get("layers", []))
        return objects

    def _object_unit_px(self) -> float:
        """Pixels per grid step in the object coordinate space.

        Tiled measures object x/y in PIXELS, and an ISOMETRIC map — the only
        projection a template may declare (``template_check.REQUIRED_ORIENTATION``)
        — projects them into a space whose unit on BOTH axes is the tile HEIGHT
        (Tiled's ``IsometricRenderer``: ``tileX = x / tileHeight``). Dividing by the
        width instead halves every derived distance, and with it every movement
        cost, so the height is the divisor on both axes rather than the obvious one.
        """
        return float(self._tiled_map().get("tileheight", 32)) or 32.0

    # ---- WorldConfig contract ---------------------------------------------
    def render_map(self) -> dict[str, Any] | None:
        """The raw tmj document — frozen per world at build so the renderer keeps
        rendering the map the world was built on, regardless of later template edits."""
        return self._tiled_map()

    def render_characters(self) -> dict[str, Any] | None:
        """The cast's art manifest, or None if this template ships no figures.

        Read lazily and never cached: it is touched once per world at build (to be
        frozen) and once per request on the fallback path, so a cache would only
        hold a template edit stale.
        """
        path = self._path.parent / CHARACTERS_FILENAME
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def render_assets_dir(self) -> Path:
        """The template's own directory — the map and the manifest both name their
        art relative to it, so one root serves both."""
        return self._path.parent

    def get_places(self) -> dict[str, Place]:
        if self._places_cache:
            return self._places_cache

        # An edge's cost is how long the walk takes in seconds: MANHATTAN tile distance between the
        # two linked rooms' object centers × `seconds_per_tile`. Manhattan (not Euclidean) because the
        # generated corridors are axis-aligned L-shapes, so this equals the tile path the renderer
        # walks. Symmetric (same centers both ways).
        #
        # Don't convert to steps here: a route's step count comes from its total time, rounded once
        # (``engine/executors/movement.py``). Rounding each edge up to a step makes travel time
        # "hops × step length", so a 40-minute walk across four rooms costs a day in a 6h/step world.
        seconds_per_tile = float(
            self._map_props().get("seconds_per_tile") or DEFAULT_SECONDS_PER_TILE
        )
        tile_px = self._object_unit_px()

        raw: dict[str, dict[str, Any]] = {}
        for obj in self._location_objects():
            props = tiled_properties(obj)
            location_id = str(props.get("location_id") or "").strip()
            if not location_id:
                continue
            raw[location_id] = {
                "name": str(obj.get("name") or location_id),
                "description": str(props.get("description", "")),
                "capacity": int(props.get("capacity", 50)),
                "is_public": bool(props.get("is_public", True)),
                "cx": (float(obj.get("x", 0)) + float(obj.get("width", 0)) / 2) / tile_px,
                "cy": (float(obj.get("y", 0)) + float(obj.get("height", 0)) / 2) / tile_px,
                "neighbors": _parse_neighbors(props.get("connections", "")),
            }
        if not raw:
            raise ValueError(f"Tiled template has no location objects: {self._path}")

        places: dict[str, Place] = {}
        for lid, d in raw.items():
            connections: dict[str, int] = {}
            for nbr in d["neighbors"]:
                if nbr not in raw:
                    continue
                dist = abs(d["cx"] - raw[nbr]["cx"]) + abs(d["cy"] - raw[nbr]["cy"])
                travel_seconds = dist * seconds_per_tile
                connections[nbr] = round(travel_seconds)
            places[lid] = Place(
                place_id=lid,
                name=d["name"],
                description=d["description"],
                connections=connections,
                is_public=d["is_public"],
                capacity=d["capacity"],
            )
        self._places_cache = places
        return places

    def get_location_aliases(self) -> dict[str, str]:
        return _parse_aliases(self._map_props().get("location_aliases", ""))

    def get_world_description(self) -> str:
        return str(self._map_props().get("world_description", ""))

    def to_runtime_context(self) -> dict[str, object]:
        props = self._map_props()
        return {
            "world_name": str(props.get("world_name", "")),
            "era_name": str(props.get("era_name", "")),
            # How this setting NAMES its dates ("正月初一" vs "3月4日"). A property of
            # the world, authored on its map — see engine.clock.CalendarStyle.
            "calendar": str(props.get("calendar", "") or "classical_cn"),
            "start_month": int(props.get("start_month", 1)),
            "start_day": int(props.get("start_day", 1)),
            "start_hour": int(props.get("start_hour", 6)),
            # Records which Tiled template this world was built on, so the map API
            # can serve the same .tmj the frontend renders (single source of truth).
            "template": self._template,
        }
