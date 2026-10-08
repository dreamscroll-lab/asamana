# Character Asset Specification

English | [简体中文](CHARACTER_ASSET_SPEC.zh-CN.md)

**Character assets must match the map's art style.**

This specification defines character coverage (gender × age band), animation frames, facing directions, size, anchors, color regions and file formats. **Body shape, clothing, shoes and hairstyles are unrestricted.**

---

## 1. Deliverables

**8 bodies × 76 frames each; one PNG + one XML per body, plus one manifest.**

```
characters/
├── characters.json     # manifest, fixed file name
├── <body>.png          # atlas, one per body
├── <body>.xml          # frame descriptions (Starling / Kenney format)
└── <body>Portrait.png  # optional: portrait, see section 7
```

---

## 2. Character coverage

**Two axes, eight bodies.**

| Axis   | Values |
| ------ | ------ |
| Gender | `male` / `female` |
| Age    | `child` / `young` / `middle` / `elder` |

**All four age bands are required** for each gender. If the asset set does not distinguish every band, adjacent bands may **share** a body; no band may be left without one.

**These are the only two axes.** Don't add your own class or occupation axis.

| Band     | Age   |
| -------- | ----- |
| `child`  | 0–13  |
| `young`  | 14–37 |
| `middle` | 38–57 |
| `elder`  | 58+   |

**Body shape isn't prescribed, but the four bands must be distinguishable at actual in-game scale.** How you distinguish them is up to you.

---

## 3. Animation frames: 11 poses, 76 frames

**The artist chooses frame names** and records them in the manifest's `frames` table. The renderer uses that table rather than hard-coded names. The second column lists names used by the existing maps for reference.

| pose       | Current frame names (`{dir}` = NE/SE/SW/NW) | Frames | What the body is doing |
| ---------- | ------------------------------------------- | ------ | ---------------------- |
| `walk`     | `walk{dir}0..3`                             | 16     | Walk cycle |
| `idle`     | `idle{dir}`                                 | 4      | Standing in a relaxed posture |
| `down`     | `fallFaceUp{dir}` / `fallFaceDown{dir}`     | 8      | Down on the ground |
| `attack`   | `swing{dir}0..2`                            | 12     | Wind-up → swing → recovery, **an unarmed arm swing** |
| `interact` | `operate{dir}0..1`                          | 8      | Working on something in front, both hands busy |
| `talk`     | `talk{dir}Closed` / `talk{dir}Open`         | 8      | Facing the other person and talking, one hand raised to gesture |
| `shove`    | `push{dir}`                                 | 4      | Two-handed push / grab, **not a strike** |
| `duck`     | `duck{dir}`                                 | 4      | Crouching low with knees bent and weight lowered |
| `hold`     | `take{dir}`                                 | 4      | Bending to pick something up / holding on |
| `show`     | `give{dir}`                                 | 4      | Handing something forward, arms extended |
| `hurt`     | `hit{dir}`                                  | 4      | The moment of being hit: upper body thrown back, off balance |
|            |                                             | **76** |  |

Several frames for a pose make an animation; a single frame makes a still pose. Both are valid.

**Bodies must be reusable across characters and stories, and poses across behaviors. Keep both generic; avoid details tied to a specific narrative action.**

---

## 4. Facing directions

The four facings are named by **screen direction**: `NE` (up-right, back to the camera), `SE` (down-right, facing the camera), `SW` (down-left, facing the camera), `NW` (up-left, back to the camera). Isometric 2:1, matching the map grid.

**Directional actions require all four facings**: the engine turns characters toward their targets before playing a pose. During `talk`, for example, both characters must face each other.

**Mirroring flips a character left or right; it cannot switch between front and back views.** Drawing only east-facing frames leaves no back view for north-facing characters. With targets distributed roughly evenly across directions, about half of directional actions would show characters facing away from their targets.

`duck` / `hurt` / `down` have no facing requirement, but for consistency we recommend giving them all four facings.

---

## 5. Canvas and anchor

- **All frames in one asset set share the same size**; the exact size is up to the artist, and scaling is declared by the manifest's `scale`.
- **Anchor: bottom-center of the frame = midpoint between the feet = center of the isometric cell.** All frames share the same anchor and must not shift when switching poses. **Anchor drift is the most common asset defect.** For `down`, align the projected center of mass to this anchor rather than the feet's standing position.
- **Leave margin on all sides of each frame** so limbs aren't clipped by the canvas during swings or falls.

**Characters are rendered at roughly 3× the map's real-world scale for readability.** At a realistic scale on the current maps, a character would be about 12 px tall with a 2 px head, making facing directions indistinguishable. For maps of this size, **the recommended `scale` range is 0.22–0.36**; both existing maps use `0.28`. Adjust the value to suit the map.

---

## 6. Color requirements for tinting

**Do not paint character identity colors into the assets.** The renderer applies them at runtime by **classifying pixels by color**:

| Order | Condition            | Classified as | Handling |
| ----- | -------------------- | ------------- | -------- |
| 1     | Value V ≤ 0.08       | Outline       | Kept as is |
| 2     | Hue H ≤ 60° (warm zone) | Skin / brown hair / leather / neutral gray-white | Kept as is |
| 3     | Everything else      | **Clothing**  | Re-tinted to the identity color |

Four rules follow:

1. **Use cool or neutral colors for the main garments** (blue, cyan, green, purple, gray, white, black). **Avoid ochre, sienna, vermilion, orange and camel**: these warm hues are classified as skin and are not tinted, making characters harder to distinguish. This is especially easy to overlook with historical clothing. **For Tang-dynasty costumes, replace reddish ochre with cyan, green, indigo, muted pink or pale yellow-gray of the same value.**
2. **Paint the clothing's mid-tone at V = 0.70.** Folds and highlights vary around that value. The engine normalizes to 0.70 before re-tinting, so the resulting mid-tones land exactly on the identity color while the light and shadow are fully preserved. **Paint it too dark and the identity color won't come through.**
3. **Skin falls in the warm zone** (H ≤ 60°), so paint all skin tones normally. **Hair**: browns ✅ / neutral black (dark gray-black with R=G=B) ✅ / neutral white and gray-white ✅ / **black with a blue or purple cast ❌** (it gets classified as clothing and re-tinted).
4. **Don't prepare different palettes for different characters.** Paint **one version** of each body; individual variation comes from the engine's tinting.

> Color values cannot distinguish body parts: the same color (#bf7958) may represent skin or hair. The warm hue range is therefore treated as a single region.

---

## 7. Manifest `characters.json`

Declare asset mappings and rendering settings in this manifest:

```jsonc
{
  "scale": 0.28,         // frame pixels → screen pixels (see section 5 for why)
  "walk_fps": 8,         // walk cycle frame rate
  "bodies": {            // gender → age band → atlas key (all four bands required; may be shared)
    "male":   {"child": "…ChildMale", "young": "…", "middle": "…", "elder": "…"},
    "female": {"child": "…", "young": "…", "middle": "…", "elder": "…"}
  },
  "atlases": {           // atlas key → files, paths relative to the map directory
    "…ChildMale": {"image": "characters/x.png", "atlas": "characters/x.xml",
                   "portrait": "characters/xPortrait.png"}   // portrait is optional
  },
  "frames": {            // pose → frame names. All three forms are valid:
    "idle":  {"NE": "idleNE", "SE": "idleSE", "SW": "idleSW", "NW": "idleNW"},   // four-facing still
    "walk":  {"NE": ["walkNE0", "…"], "SE": […], "SW": […], "NW": […]},          // four-facing animation
    "duck":  "duckSE"                                                            // single facing (mirrored by the engine)
  }
}
```

Use 32-bit RGBA PNGs with non-premultiplied alpha. XML `SubTexture` coordinates must stay within the image bounds.

Each `down` entry takes one frame name, though the artwork may include both face-up and face-down falls. Preserve **facing continuity**: use face-down for north-facing characters (back to the camera) and face-up for south-facing characters (toward the camera).

The `_comment` field is only for noting what this art set is for and its known deviations; the renderer ignores it.

### Portrait `portrait` (optional)

The character cognition overlay displays an enlarged figure. Atlas frames (about 190 px tall) become blurry at this size, so each body may include a separate high-resolution portrait. **The portrait need not be the idle frame itself, but it must match it**:

- **Same person**: same face, hairstyle, build and apparent age.
- **Same outfit**: same cut and palette. The color rules are the same as section 6; the engine tints the portrait's identity color with the same rules.
- **Same stance**: match the body's `idle` SE frame, standing naturally and facing the camera at a slight angle to the right. The overlay's callout lines target the head, chest, hands and feet based on this stance; a different pose or facing misaligns them.
- **Same art style** as the map and atlas.
- **PNG with a transparent background**, cropped to the figure. Figures should be ≥ 1000 px tall (children ≥ 850 px).

The validator crops the portrait and idle SE frame to the figure, then checks **silhouette overlap ≥ 0.94** and **similar overall light/dark distribution**. These checks reject mismatched poses, facings and figures. Facial details and art style require visual review: compare the images side by side in the map workbench's "Cast" tab. Without a portrait, the overlay uses the idle SE frame, which appears blurry at this size.

---

## 8. Acceptance

Run the validator first (it checks the characters too):

```bash
.venv/bin/python -c "from worlds.template_check import check_template; print(check_template('map-directory-name') or 'OK')"
```

It automatically checks that all four `bodies` bands are present, every `atlases` file exists, all 11 poses are present, and every frame name actually exists in its atlas.

**Review the remaining requirements visually.** Open the **map workbench**'s "Cast" tab, which renders every pose in section 3, and check the following:

- **The characters' style matches the map.**
- **All frames share the same anchor**: turn on "Stack"; the midpoint between the feet must not drift as frames are stacked.
- The four age bands are **distinguishable** at actual scale.
- Limbs in swing and fall frames are **not clipped by the canvas**: turn on "Canvas" to see the frame borders.
- Single-facing poses all face **east**; mirrored characters have no fixed asymmetric features (check the "mirrored" badge).
- Characters **don't cover the buildings at their feet**: switch to "In game" to view at real scale.

The two color rules (V≈0.70 and the hue zone from section 6) can't be checked in the UI; verify them in your drawing software.

---

## Appendix: the two existing maps

| Map           | Atlas keys                                            | Frame size | `scale` | `walk_fps` |
| ------------- | ----------------------------------------------------- | ---------- | ------- | ---------- |
| `changan_iso` | `tangCommoner{Child,Young,Middle,Elder}{Male,Female}` | 192 × 256  | 0.28    | 8          |
| `metro`       | `metro{Child,Young,Middle,Elder}{Male,Female}`        | 192 × 256  | 0.28    | 8          |

Both provide all four facings and 76 frames with the same frame names, so the `frames` table can be copied as is.

Portraits: all 16 bodies have one. Most are the source images from when the atlases were generated; `metro`'s `ChildMale` was upscaled from its idle frame (the figure is only 636 px tall, on the small side).

### Known deviations from this specification

Check these deviations before creating replacement assets. **Follow this specification when existing assets conflict with it.**

| Item                                  | Required          | changan_iso | metro |
| ------------------------------------- | ----------------- | ----------- | ----- |
| Clothing mid-tone                     | V = 0.70          | 0.42–0.56   | **0.11–0.33** |
| `young` / `middle` distinguishable    | Yes               | **No**      | **No** |
| `elder` distinguishable               | Yes               | 5% shorter + stooped | **Same height as young**, stoop is the only cue |
| East/west facings                     | All four facings drawn separately | **Pixel-level mirrors** | Same |

- **Clothing value** is too low in both sets, especially `metro`: the tinted identity color reaches only 20–50% of its intended brightness, making similarly colored characters harder to distinguish. This is the highest-priority asset fix.
- **`young` / `middle` can't be told apart in either set** (the middle-aged torso is actually 7–13% narrower than the young one). Body shape itself isn't prescribed; not being distinguishable is the problem.
- **East/west facings are mirrored in both sets**, so about half of the 76 frames add no new information. Mirroring also reverses the crossed collar's overlap, making it historically incorrect in one direction. This is accepted for now because the collar occupies only a few pixels at game scale.
- **Outstanding issue**: the movement frames still have noticeable problems. They don't affect functionality or the narrative, so they can be ignored for now.
