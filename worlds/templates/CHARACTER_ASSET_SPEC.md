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

## 4. Facings

The four facings are named by **screen direction**: `NE` (up-right, back to the camera), `SE` (down-right, facing the camera), `SW` (down-left, facing the camera), `NW` (up-left, back to the camera). Isometric 2:1, matching the map grid.

**Directional actions require all four facings**: the engine turns characters toward their targets before playing a pose. During `talk`, for example, both characters must face each other.

**Mirroring can only flip a character left/right, not turn them around**: if you draw only the east-facing frames, a character facing north is drawn showing their face. Since targets are spread roughly evenly across the four directions, about half of all directional actions would show the character with their back to the target and their face to the camera.

`duck` / `hurt` / `down` have no facing requirement, but for consistency we recommend giving them all four facings.

---

## 5. Canvas and anchor

- **All frames in one asset set share the same size**; the exact size is up to the artist, and scaling is declared by the manifest's `scale`.
- **Anchor: bottom-center of the frame = midpoint between the feet = center of the isometric cell.** All frames share the same anchor and must not jump when switching poses. **This is the most common delivery defect.** `down` also aligns its "projected center of mass" to this anchor; don't position it by where the feet would be when standing.
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

1. **The main body of the clothing must be a cool or neutral color** (blue, cyan, green, purple, gray, white, black). **Never paint the main garments ochre, sienna, vermilion, orange or camel**: they fall into the warm zone and get classified as skin, so the character **loses its identity color entirely** and becomes indistinguishable from others on the map. ⚠️ This is especially dangerous for period costume: the intuitive palette for Tang-dynasty clothing sits almost entirely in the warm zone. **When drawing Tang costume, deliberately swap the reddish-ochre you'd reach for with cyan, green, indigo, lotus-root pink or pale yellow-gray of the same value.**
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

`down` takes only one frame name, while the art may include both face-up and face-down falls. Choose by **facing continuity**: a character who dies facing north (back to the camera) gets face-down, one who dies facing south (toward the camera) gets face-up, so the fallen character keeps showing the audience the same side they showed in life.

The `_comment` field is only for noting what this art set is for and its known deviations; the renderer ignores it.

### Portrait `portrait` (optional)

The character cognition overlay displays an enlarged figure. Atlas frames (about 190 px tall) become blurry at this size, so each body may include a separate high-resolution portrait. **The portrait need not be the idle frame itself, but it must match it**:

- **Same person**: same face, hairstyle, build and apparent age.
- **Same outfit**: same cut and palette. The color rules are the same as section 6; the engine tints the portrait's identity color with the same rules.
- **Same stance**: like this body's `idle` SE frame, facing the camera turned slightly right, standing naturally. The overlay's callout lines land on the head, chest, hands and feet based on this stance; change the facing or pose and they miss.
- **Same art style** as the map and atlas.
- **Transparent-background** PNG (with alpha), cropped to the figure. Figures should be ≥ 1000 px tall (children ≥ 850 px).

The validator crops both the portrait and the idle SE frame to the figure and compares two things: **silhouette overlap ≥ 0.94** (a different pose or facing is rejected) and **similar large-scale light/dark distribution** (a different person is rejected). Facial detail and art style can't be judged by machine: compare the portrait and the idle frame side by side in the "Cast" tab of the map workbench and confirm them yourself. A body without a portrait falls back to the idleSE frame, which will look blurry.

---

## 8. Acceptance

Run the validator first (it checks the characters too):

```bash
.venv/bin/python -c "from worlds.template_check import check_template; print(check_template('map-directory-name') or 'OK')"
```

It automatically checks that all four `bodies` bands are present, every `atlases` file exists, all 11 poses are present, and every frame name actually exists in its atlas.

**The rest is checked by eye.** Open the "Cast" tab of the **map workbench**, which renders the whole table from section 3. Things to verify:

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

- **Clothing value** is too dark in both, metro especially: the tinted identity color comes out at only 20–50% of what was intended, making similarly colored characters even harder to tell apart. This is the most important one to fix.
- **`young` / `middle` can't be told apart in either set** (the middle-aged torso is actually 7–13% narrower than the young one). Body shape itself isn't prescribed; not being distinguishable is the problem.
- **East/west facings are mirrors in both sets**, so about half of the 76 frames add no new information. For period costume there's one more catch: the crossed collar is necessarily wrapped left-over-right on one side, the wrong way. At game scale the collar is only a few pixels, so this is accepted for now.
- **Outstanding issue**: the movement frames still have noticeable problems. They don't affect functionality or the narrative, so they can be ignored for now.
