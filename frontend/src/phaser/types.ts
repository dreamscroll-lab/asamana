// Shared Phaser render-layer types + constants used across the scene and its
// extracted modules (effects, pathfinding, skins).

import Phaser from "phaser";

import type { Dir } from "./skins";

// Phaser rasterizes text to a texture at 1× by default → soft on Hi-DPI screens
// and when the camera zooms out to fit the map. Rasterize at the device pixel
// ratio (capped) so labels stay crisp regardless of display or zoom.
export const TEXT_RES = Math.min(4, Math.max(2, Math.ceil((window.devicePixelRatio || 1) * 2)));

// The map's one typeface. Every text uses this constant: a text that omits it silently falls
// back to Phaser's built-in default face.
export const FONT = '"PingFang SC", system-ui, sans-serif';

// How wide a floating body of text may run before it wraps — one width for the bubble, the
// chip and the broadcast subtitle. Differences between them would go unnoticed, because no
// two of them ever appear at the same moment.
export const WRAP_W = 210;

// Wrap width for prose over the map that isn't a speech bubble (the float and the hover card).
// Wider than WRAP_W since an errand's whole line of speech can go through both; unwrapped, a
// 54-character errand at 12px runs 650px. Both plates size from their text's own box.
export const NOTE_WRAP_W = 300;

// An action's caption above the token and result chip below it are one voice reporting one
// deed, so they share size and face. A spoken line keeps its own size (see BUBBLE_VARIANTS):
// dialogue is the map's focal text.
export const TEXT_ACTION = "11px";

export interface AgentToken {
  container: Phaser.GameObjects.Container;
  sprite: Phaser.GameObjects.Sprite; // the humanoid body — a body from the world's own
  // character set (gender + age bracket); its garment is dyed by the identity color.
  skinId: string; // body key in the world's CharacterSet; reassigned when demographics arrive
  posed: boolean; // an action pose is being held — update() must not yank it back to idle
  // or overwrite it with the walk cycle. Cleared when the pose resolves.
  plate: Phaser.GameObjects.Container; // the compact nameplate (pip + name + health) as
  // ONE unit below the figure — moved/toggled together (LOD, co-located roster).
  label: Phaser.GameObjects.Text; // name (identity-colored), inside the plate. ALWAYS
  // drawn — the one thing on the plate that never moves into the pocket or waits for a
  // click, because "who is this" is the question a map answers before any other.
  pouch: Phaser.GameObjects.Text; // the pocket's handle beside the name: "✦3 ☞" closed,
  // "✦3 ☟" open, hidden when empty-handed. Carried items don't fit on the one-row plate;
  // they live in `pocket`.
  condPip: Phaser.GameObjects.Ellipse; // the always-on half of the condition: a small amber
  // mark beside the head whenever `condText` is non-empty, readable at any zoom. Persistent
  // (unlike actionChip/postBadge), so clearEphemeral must not wipe it. Amber, never
  // tint/alpha/the red OVER badge: those mean death (applyDeathVisual).
  condText: string; // the condition itself, cached for the pocket and for `sync` when focus
  // changes with no step in flight. Not drawn on the nameplate (see pocket.ts `sync`).
  pocket?: Phaser.GameObjects.Container; // the open pocket panel: condition in full, then
  // the carried things with their states. A child of `container` and not ephemeral: it stays
  // open across steps, rebuilt in place. At most one is open on the map; see PocketLayer.
  headY: number; // container-local y of the top of this figure's head (negative; feet ≈ 0).
  // Measured from the art, since body height arrives per world in the manifest's `scale`; a
  // fixed offset above the feet would put the bubble through a taller figure's chest.
  color: number; // stable role/main color — tints this agent's name tag in bubbles
  baseAlpha: number; // how solid this figure is before the x-ray (what focus and death say).
  // Only the scene's update() writes the container's alpha, multiplying this by the x-ray
  // factor; a second writer would flick a dimmed corpse back to full opacity behind a tree.
  floor: number; // y below which a downward bubble must start, to clear the name roster (0 = none)
  // OVER badge, toggled by vitality (playback-safe). A container: rounded plate + label,
  // since a Text's own backgroundColor can only be a hard rectangle.
  deadMark?: Phaser.GameObjects.Container;
  vitTrack: Phaser.GameObjects.Rectangle; // vitality bar backing (hidden at full health)
  vitFill: Phaser.GameObjects.Rectangle; // vitality bar fill, scaled by vitality
  focusRing: Phaser.GameObjects.Ellipse; // purple ground ring shown when this agent is focused
  actionChip?: Phaser.GameObjects.Container; // step-scoped "result" pill under the token, a
  // child of `container`. Shows this step's outcome + success/failure until clearEphemeral
  // clears it before the next step (cross-step history lives in the feed).
  // The raw chip content, cached so applyFocus can re-render the chip when focus changes
  // (focused → full text, others → compact) without the per-step action data. `progress`
  // (0..1) turns the accent underline into a progress bar for the middle beat of a long act.
  chipData?: {
    icon: string;
    text: string;
    tone: "ok" | "fail" | "cut";
    progress?: number;
    // What he is doing, on a muted line above `text` (how it turned out). The closing beat of
    // a long act has no intent bubble, so without this the chip says only the result. Both
    // lines share the one-line glance budget when unfocused, as `text` is clipped.
    intent?: string;
  };
  // The post that reached this agent this step, cached like chipData: focus decides whether it
  // opens itself or waits to be clicked, and focus can change after the beat has played.
  postData?: { from: string; ink: number; said: string }[];
  postBadge?: Phaser.GameObjects.Container; // ✉ + count over the head; click opens the post
  locationId: string; // last-known location id (for the spotlight in-region test); "" in transit
  dead: boolean; // cached so applyFocus can restore the dead-dim alpha
  hurt: boolean; // cached vitality-bar "should show" state (LOD gates it by zoom)
  vit: number; // vitality the bar is CURRENTLY drawn at — what the step reports may be withheld
  phase: number; // per-agent idle-bob phase so a crowd doesn't bob in unison
  dir: Dir; // isometric heading (NE/SE/SW/NW). A ±1 left/right flag can't show a figure
  // walking away from the camera.
  px: number; // previous-frame container x/y (NaN until first seen) — drives walk anim
  py: number; //   + heading by detecting real movement, no per-path bookkeeping
}

// ---------------------------------------------------------------------------
// The token's anatomy, the one shape every readable surface is made of, and the renderer's
// single definition of death. They live here because they belong to the map's furniture,
// not to whichever module draws them.
// ---------------------------------------------------------------------------

// The heading a figure starts on, before it has walked anywhere: toward the camera,
// so a standing cast faces the viewer rather than showing it their backs.
export const INITIAL_DIR: Dir = "SE";

// Where the figure's FEET sit inside its token container. The sprite's origin is its frame's
// bottom-center, so this is also the sprite's y — and the baseline every fixture measures from.
export const SPRITE_FOOT_Y = 6;

// Gap between the top of a figure's head and the nameplate above it. The anchor is
// `tok.headY`, measured from the art (see measureHeadY), not a fixed offset.
export const PLATE_GAP = 5;

// How far the nameplate's own contents reach ABOVE its anchor: name at 0, health at -8,
// carried items at -18, plus half a line. A property of the plate's internal layout, so it
// stays a constant — unlike the figure's height, which is data.
export const PLATE_RISE = 24;

export const PLATE_R = 5; // corner radius — a soft box, not a rounded blob

// The OVER badge worn by a fallen body. Smaller radius than PLATE_R since it is half a
// nameplate's height. Red is reserved for it: no other plate on the map uses red.
export const DEAD_MARK_R = 3;
export const DEAD_MARK_FILL = 0xb3202c;
export const DEAD_MARK_EDGE = 0x7a1119;
export const DEAD_MARK_PAD_X = 4;
export const DEAD_MARK_PAD_Y = 1;

/**
 * One thing somebody is holding: what it is, and what state it is in.
 *
 * The state is why this isn't a bare name: otherwise an item's state (a snapped bowstring, a
 * sealed letter) would vanish the moment someone picked it up. The pocket shows it.
 */
export interface CarriedItem {
  name: string;
  state: string;
  // Nobody standing beside him can perceive this one (`is_public: false` on the wire). Shown
  // only in the pocket, the one surface that lists what he holds.
  concealed?: boolean;
  description?: string;
  content?: string;
}

/**
 * A PLATE: drop shadow, fill, border. Every surface on this map that the reader is meant to
 * stop and READ is made of this one shape.
 *
 * Drawn because Phaser's `Text.backgroundColor` paints only a sharp, borderless slab, which on
 * a pixel tilemap reads as a hole in the world; the shadow and border make it an object.
 *
 * The text on top keeps its `padding` (the plate is sized around it) but drops `setStroke`: an
 * outline around an 11px CJK glyph silts up its strokes, and the plate already gives contrast.
 */
export function drawPlate(
  g: Phaser.GameObjects.Graphics,
  x: number, y: number, w: number, h: number,
  fill: number, border: number, radius = PLATE_R, alpha = 0.97,
): void {
  g.fillStyle(0x000000, 0.35);
  g.fillRoundedRect(x + 2, y + 3, w, h, radius);
  g.fillStyle(fill, alpha);
  g.fillRoundedRect(x, y, w, h, radius);
  g.lineStyle(2, border, 1);
  g.strokeRoundedRect(x, y, w, h, radius);
}
