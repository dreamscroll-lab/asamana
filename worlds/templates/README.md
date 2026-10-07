# World Map Specification

English | [简体中文](README.zh-CN.md)

**A world map includes map assets and character assets.**

One map can be reused by many different narrative themes, so map assets should never be tied to any particular theme.

A map provides terrain, roads, buildings, static infrastructure, vegetation and character assets. The narrative engine handles character movement and interactions. **Keep map assets simple and reusable across themes.**

---

## 1. Layout of a map

```
worlds/templates/<map-name>/
├── map.tmj          # fixed file name
├── tilesets/        # map art; paths follow the relative paths in the .tmj
└── characters/      # character art — see CHARACTER_ASSET_SPEC.md
    ├── characters.json
    └── *.png + *.xml
```

**Installing a map**: copy the directory into `worlds/templates/` and run the validator below, or zip it and click "Import map" in the **map workbench** (`#/lab`). Import runs the same validator **before writing map files**. If validation fails, it lists the problems and does not install the map.

```bash
.venv/bin/python -c "from worlds.template_check import check_template; print(check_template('your-directory-name') or 'OK')"
```

**Refresh the page and the map shows up in the map picker.** The list is rescanned from disk on every request, so there's no need to restart the service.

**Deleting**: click "Delete" in the map workbench, or just delete the directory. **Worlds that are already built are unaffected**: when a world is initialized, it **freezes** its own copy of the map, its art and its character assets, and reads only that copy at runtime. Deleting a map only means you can no longer build new worlds from it.

**`<map-name>`**: `^[a-z][a-z0-9_]{0,63}$`, a single directory level (`metro`, `changan_iso`). It is a **code-level identifier only and never shown in the UI**; the UI shows `world_name`. We recommend opening and editing maps in Tiled.

**Both names must be unique, and import enforces this:**

- The **directory name** is already taken → rejected. **Import never overwrites an existing map.**
- The **`world_name`** is already used by another map → rejected.

---

## 2. Projection and drawing

### Isometric projection only

Only the isometric projection is supported.

Character frames use four isometric directions. Other map projections do not align with these directions.

### Grid size: 128×64 (2:1) recommended, not required

2:1 is the standard for isometric art. We recommend 128×64, though 64×32 works just as well.

### Tilesets must be embedded

External `.tsj` references are not allowed. An external tileset has no `image` field in the `.tmj`, so the server can neither freeze it nor serve it, and those tiles render as holes. In Tiled: Tilesets panel → right-click → Embed in map.

Art **must live inside the map directory** and be referenced by relative path. Tile size doesn't have to match the grid size; tall buildings, for example, are meant to span several cells.

### Layer names are globally unique

Layer names must be unique across the whole map, including layers in different groups.
We recommend grouping layers by purpose (`base` / `structures` / `environment`); group names play no part in any logic.

---

## 3. Map properties (Tiled: Map → Map Properties)

**All eight are required.**

| Property            | Type   | Description |
| ------------------- | ------ | ----------- |
| `world_name`        | string | **The world map's name**, 2–6 Chinese characters. This is the name shown in the map picker. |
| `era_name`          | string | The era name, fed into the narrative engine, e.g. `大唐` / `当代` / `2087年`. |
| `calendar`          | string | How dates are written: `classical_cn` (正月初一) or `modern` (3月4日). |
| `world_description` | string | Description used to select a suitable map during world initialization. Describe the map accurately without tying it to a particular narrative theme. |
| `start_month`       | int    | 1–12 |
| `start_day`         | int    | 1–30 |
| `start_hour`        | int    | 0–23 |
| `seconds_per_tile`  | int    | **How many seconds it takes to cross one tile** |
| `location_aliases`  | string | *Optional*. `alias:location_id,…`, a fallback for colloquial references. |

### Travel time per tile

`seconds_per_tile` is how long a character takes to move across one tile. Estimate the real-world distance a tile represents, then a typical travel speed, and derive the value from those. For example, Tang-dynasty Chang'an is traversed on foot (one tile ≈ 200 m → `144`), while a modern city has cars and a subway (same tile size → `60`).

---

## 4. Locations

There must be **exactly one** object layer named `place` (other object layers are allowed, but locations are read only from `place`).

Every object in that layer must meet these requirements:

| Requirement         | Notes |
| ------------------- | ----- |
| `type` = `location` | The "Class" field in Tiled |
| Has a name          | The object's Name is the **place name used in the narrative** (e.g. 「市立医院」, "City Hospital"). |
| Non-zero rectangle  | A location is an **area**, not a point marker |
| Inside the map      |       |

There are five properties, four of which you fill in:

| Property      | Description |
| ------------- | ----------- |
| `location_id` | Lowercase English with underscores, **unique across the map**. |
| `description` | One sentence on what this place is and what it looks like. |
| `connections` | **Leave it out**: it is computed and written automatically on import (section 5). |
| `capacity`    | Maximum number of people it holds |
| `is_public`   | Whether the location is open to the public. Characters consider access restrictions when deciding whether to enter; this is not a hard movement constraint. |

**Granularity**: maps provide **stable infrastructure**; the narrative engine adds people, items and story events. Name locations by function or geography (「城东工业园」 "East Industrial Park", 「市立医院」 "City Hospital"), avoiding names or conflicts tied to a theme. We recommend **20–25 locations**, each connected to **3–5** neighbors.

**The whole map must be connected**: every location must be reachable from every other, or characters will get stuck.

---

## 5. Connections

**Do not set `connections` manually.** Import derives them from the map's walkable terrain and writes the property automatically, replacing any supplied value. No empty placeholder is needed.

The criterion is the **path**, not the distance:

> A and B are connected ⟺ the **shortest walkable route** between them doesn't enter the rectangle of any **third location**. If the shortest path from A to B goes straight there, A and B each list the other in `connections`. If the shortest path passes through C (A <-> C <-> B), A and B do not list each other.

Locations are **areas**, not points: characters walk to a location's edge, not its center.

Roads, walls and obstacles already define where characters can move. Deriving connections from this geometry keeps the location graph consistent with the map.

> If you copy a directory in by hand (instead of importing it), run this yourself:
>
> ```bash
> python -m worlds.connections <map-name>            # just show the derived result
> python -m worlds.connections <map-name> --write     # write it back (changes only this field, keeps the Tiled format)
> ```

---

## 6. Walking and standing

### Walkable road surface: pick one mode, never mix

- **Allowlist**: mark walkable road surface tiles with `road = true` (characters can move across them).
- **Blocklist**: mark obstacle tiles with `collides = true` (characters can neither move across nor stand on them).

As soon as **any** tile in the map has `road`, the whole map is treated as an allowlist: every unmarked cell is unwalkable.

> **The most common mistake: every walkable road surface tile must be marked, not just one.** A street is often assembled from several tiles of the same tileset, and if you miss some of them, the whole street stops being walkable.

`collides` applies in both modes: it means "no one can be on this cell".

### Standing areas: usually inferred from artwork

The renderer reads your art directly. A tile **taller than the grid** is something standing up (a house, a city wall, a tree canopy), and it occupies the ground cells its pixels actually cover. A tile **exactly one cell high** is the ground itself (paving, walkable road surface, grass) and never occupies a cell. So when you draw a house, the ground under and behind it automatically becomes off-limits without a single marker.

**Mark flat surfaces that characters cannot stand on**, typically water, with `collides`. These tiles are one cell high, so their dimensions do not distinguish them from walkable ground. Mark water tiles only; mixed shoreline tiles should remain available for standing.

### Two things the validator checks here

- **No location may be isolated: each must connect to the walkable road surface.** If it can't be reached, pathfinding has no starting point, and every trip to or from it becomes a straight line through buildings.
- **Each location's own rectangle must contain at least 4 open cells.** Otherwise people get placed **outside** the rectangle.

---

## 7. Characters

Character art is part of the map. The full specification is in
**[CHARACTER_ASSET_SPEC.md](CHARACTER_ASSET_SPEC.md)**.

Character assets are required. A missing `characters/` directory fails validation.

---

## 8. What the validator rejects

| Reported problem | See |
| ---------------- | --- |
| Projection isn't isometric | Section 2 |
| Tileset is an external reference / has no image / image not found | Section 2 |
| Duplicate layer names / not exactly one `place` layer | Sections 2, 4 |
| Missing map properties | Section 3 |
| No locations / missing properties / empty or duplicate id / no name / zero size / out of bounds | Section 4 |
| Unreachable location (map not connected) | Sections 4, 5 |
| Hand-written connections the ground doesn't support (no path / passes through a third location / one-way / points to a nonexistent location) | Section 5 |
| Location can't reach the road network | Section 6 |
| Not enough open cells in a location's rectangle | Section 6 |
| Character manifest missing or invalid | CHARACTER_ASSET_SPEC.md |

---

## 9. Building a map from scratch

### Decide what kind of place it is

Choose the map's name, era, calendar format and spatial layout before selecting artwork. Record the name, era and calendar in the map properties (section 3), and describe the layout in `world_description`.

### Gather both sets of art

- **Map**: a 2:1 isometric tileset
- **Characters**: character assets that match the map (see [CHARACTER_ASSET_SPEC.md](CHARACTER_ASSET_SPEC.md))

> To get the map working first, you can **temporarily** copy an existing `characters/` directory. Once the map passes validation, make character assets that belong to this world.

### Create it in Tiled

- Orientation: **Isometric**
- Tile size: **128 × 64**
- Map size: both existing maps are **48 × 56 tiles** (larger maps are fine too).

### Lay the ground, in layers

The existing maps are grouped as `base` (ground / walkable road surface / water), `structures` (infrastructure, walls, buildings), `environment` (trees, vegetation and decoration) and `regions` (object layer). Group names play no part in any logic; they exist only to make editing easier. **Drawing order is rendering order, stacked bottom to top.**

### Mark the walkable road surface — mind where the marks go

`road` / `collides` go on **tiles in the tileset**, not on map layers: in Tiled, open the tileset, select the tile, and add the custom property `road` (bool) = true or `collides` (bool) = true.

### Draw the locations

Create an **object layer** that must be named `place` (it can sit in any group). Draw rectangles on it. For each rectangle, set Class to `location`, set Name to the location's name, then add the four properties (section 4). Leave out `connections`.

### Validate as you go

Run the validator once the ground is laid, then address the reported problems as you refine the map.

### Check the result

After importing the map, open the **map workbench** (`#/lab`). The "Cast" tab lays out every character asset so you can preview them, including their animations. The "Scenes" tab is the **real renderer**: what you see there is exactly what you'll see once a real narrative is running.

---

## 10. Asset provenance and license

The map and character art bundled with the repository (`changan_iso/`, `metro/`) was created by AI and humans together and, like the code, is released under [Apache-2.0](../../LICENSE). When contributing a new map, make sure you have the right to release all of its art under the same license.
