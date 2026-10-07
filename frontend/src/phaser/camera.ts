/**
 * The viewer's eye on the map: zoom limits, wheel / pinch / drag, the follow flight, and fitting the
 * whole map to the canvas.
 *
 * While following, the camera is the FOCUS's; any hand gesture takes it back and tells React
 * (onGrabCamera). What the focus IS stays with the scene, which hands it in on every call.
 */

import Phaser from "phaser";

import type { MapFocus } from "../types";
import { THEME } from "../lib/theme";
import type { CharacterSet } from "./skins";
import type { AgentToken } from "./types";
import type { WorldMap } from "./worldMap";

// The tightest a FOLLOW shot may be framed, in tiles across: a single agent is a point and
// would otherwise be framed with nothing around him, which is what makes him worth watching.
const FOLLOW_MIN_TILES = 6;
// Pull toward the aim per second of playback, as an exponential rate: per-frame fractions would
// run faster on a 120Hz screen and lag behind walkers at 2×. High, every step of a walk jolts the
// shot; low, the subject moves within the frame first. Zoom is slower still — a shot that breathes
// in and out reads as a fault, and the framing box changes size every step.
const CAM_PAN_RATE = 3.4;
const CAM_ZOOM_RATE = 1.5;
// Wheel zoom is exponential in scroll distance (a mouse detent ≈ 100px), so one detent is a
// fixed factor and a trackpad's small deltas creep.
const WHEEL_NOTCH = 100;
const ZOOM_PER_NOTCH = 1.12;
const ZOOM_RATE = Math.log(ZOOM_PER_NOTCH) / WHEEL_NOTCH;

/** The slice of the scene the camera needs: the canvas and input, the map, the cast's art scale. */
export interface CameraHost extends Phaser.Scene {
  readonly map: WorldMap;
  readonly cast: CharacterSet;
  token(id: string): AgentToken | undefined;
}

export class MapCamera {
  // Once the user adjusts, a resize won't yank the view back to fit.
  private minZoom = 0.1;
  private maxZoom = 4;
  private pinchPrev = 0; // last two-pointer distance (0 = not currently pinching)
  private dragFrom: { x: number; y: number } | null = null; // pan anchor (screen px)
  private userAdjusted = false;
  // Whether the camera is being flown; the aim itself is recomputed every frame (followTarget).
  private following = false;
  // One-shot: the next follow frame lands its aim instead of easing into it. Set when a
  // followed subject is cut to a new position (see the scene's moveAgent), consumed by fly().
  private cameraCut = false;
  private downPos: { x: number; y: number } | null = null; // pointer-down for click-vs-drag
  private pinched = false; // a pinch happened this gesture → suppress the click-select

  constructor(
    private host: CameraHost,
    // Fired when the user takes the camera by hand while following; React turns follow off.
    private onGrabCamera?: () => void,
  ) {}

  /**
   * Fit the whole map, refit on resize, and take the pointer: wheel + pinch zoom (to the cursor /
   * pinch midpoint) and drag-to-pan, all in screen space and clamped to [minZoom, maxZoom]. A press
   * calls `onPress`; a release after negligible movement with no pinch is a click → `onClick`.
   */
  mount(onPress: () => void, onClick: (p: Phaser.Input.Pointer) => void): void {
    this.fit();
    this.host.scale.on("resize", () => this.fit());
    this.host.game.canvas.style.touchAction = "none"; // let the scene own pinch, not the browser
    this.host.input.addPointer(1); // enable a 2nd pointer so pinch works
    const cam = this.host.cameras.main;
    this.host.input.on("wheel", (p: Phaser.Input.Pointer, _o: unknown, _dx: number, dy: number) => {
      this.zoomAtScreen(Math.exp(-this.wheelPixels(p, dy) * ZOOM_RATE), p.x, p.y);
    });
    this.host.input.on("pointerdown", (p: Phaser.Input.Pointer) => {
      onPress();
      this.downPos = { x: p.x, y: p.y };
      this.pinched = false;
    });
    this.host.input.on("pointermove", (p: Phaser.Input.Pointer) => {
      const p1 = this.host.input.pointer1;
      const p2 = this.host.input.pointer2;
      if (p1.isDown && p2.isDown) {
        // Two-finger pinch (touch): zoom by the change in finger distance.
        const d = Phaser.Math.Distance.Between(p1.x, p1.y, p2.x, p2.y);
        if (this.pinchPrev > 0 && d > 0) {
          this.zoomAtScreen(d / this.pinchPrev, (p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        }
        this.pinchPrev = d;
        this.pinched = true;
        this.dragFrom = null;
      } else if (p.isDown) {
        // Drag-pan — gate on THIS pointer (works for mouse, which is not pointer1,
        // and single touch). Track the anchor manually so the delta is exact.
        if (this.dragFrom) {
          cam.scrollX -= (p.x - this.dragFrom.x) / cam.zoom;
          cam.scrollY -= (p.y - this.dragFrom.y) / cam.zoom;
          this.userAdjusted = true;
          this.releaseFollow(); // a drag is an explicit camera grab → stop following
        }
        this.dragFrom = { x: p.x, y: p.y };
      }
    });
    const endInput = (p: Phaser.Input.Pointer) => {
      // A click (negligible movement since down, no pinch) selects; a drag/pinch panned.
      if (!this.pinched && this.downPos && Phaser.Math.Distance.Between(this.downPos.x, this.downPos.y, p.x, p.y) < 6) {
        onClick(p);
      }
      this.downPos = null;
      this.pinchPrev = 0;
      this.dragFrom = null;
    };
    this.host.input.on("pointerup", endInput);
    this.host.input.on("pointerupoutside", endInput);
  }

  /** Re-fit to the whole map, discarding the user's zoom/pan. */
  reset(): void {
    this.userAdjusted = false;
    this.fit();
  }

  /** Zoom by `factor` about the centre of the canvas — the wheel's zoom, for a button. */
  zoomBy(factor: number): void {
    this.zoomAtScreen(factor, this.host.scale.width / 2, this.host.scale.height / 2);
  }

  /** The followed subject was cut to a new position: land the next follow frame instead of easing. */
  cut(): void {
    this.cameraCut = true;
  }

  /**
   * Called every frame: re-aim at where the subjects ARE and ease toward it (see CAM_PAN_RATE).
   * `playSec`: playback seconds since the last frame, i.e. wall time × the animation speed.
   */
  fly(focus: MapFocus, active: boolean, playSec: number): void {
    if (!this.following) return;
    const want = this.followTarget(focus, active);
    if (!want) return;
    const cam = this.host.cameras.main;
    const cut = this.cameraCut;
    this.cameraCut = false;
    const pan = cut ? 1 : 1 - Math.exp(-playSec * CAM_PAN_RATE);
    const zoom = cut ? 1 : 1 - Math.exp(-playSec * CAM_ZOOM_RATE);
    cam.setZoom(Phaser.Math.Linear(cam.zoom, want.zoom, zoom));
    cam.centerOn(
      Phaser.Math.Linear(cam.midPoint.x, want.cx, pan),
      Phaser.Math.Linear(cam.midPoint.y, want.cy, pan),
    );
  }

  /**
   * Where the camera wants to be, to frame the focus (subjects' bounding box + the
   * spotlighted location), padded. Null = nothing to follow.
   *
   * Recomputed EVERY FRAME, never snapshotted: a target fixed at step start leaves a walker
   * behind, and one re-issued on a timer makes the camera converge, jump, converge again.
   */
  private followTarget(focus: MapFocus, active: boolean): { cx: number; cy: number; zoom: number } | null {
    if (!focus.follow || !active) return null;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const grow = (x: number, y: number) => {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    };
    for (const id of focus.agents) {
      const tok = this.host.token(id);
      if (tok) grow(tok.container.x, tok.container.y);
    }
    if (focus.place) {
      const room = this.host.map.rooms.get(focus.place.id);
      if (room) {
        for (const [cx, cy] of [
          [room.gx0, room.gy0], [room.gx0 + room.gw, room.gy0],
          [room.gx0 + room.gw, room.gy0 + room.gh], [room.gx0, room.gy0 + room.gh],
        ] as const) {
          const p = this.host.map.iso(cx, cy);
          grow(p.x, p.y);
        }
      }
    }
    if (!isFinite(minX)) return null;
    const pad = 130;
    const w = maxX - minX + pad * 2;
    const h = maxY - minY + pad * 2;
    const zoom = Phaser.Math.Clamp(
      Math.min(this.host.scale.width / w, this.host.scale.height / h),
      this.minZoom,
      Math.min(this.maxZoom, this.host.scale.width / (this.host.map.tileW * FOLLOW_MIN_TILES)),
    );
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, zoom };
  }

  /** Take (or release) the camera for the current focus; the aiming is fly()'s. */
  followFocus(focus: MapFocus, active: boolean): void {
    this.following = focus.follow && active;
    if (this.following) this.userAdjusted = true;
  }

  /**
   * One wheel event's scroll distance in pixels (normalised from lines/pages), capped at one
   * detent. Don't use a fixed step per event: a trackpad's dozens of events per swipe would
   * slam between the zoom limits.
   */
  private wheelPixels(p: Phaser.Input.Pointer, dy: number): number {
    const mode = (p.event as WheelEvent | undefined)?.deltaMode ?? 0;
    const px = mode === 1 ? dy * 16 : mode === 2 ? dy * this.host.scale.height : dy;
    return Phaser.Math.Clamp(px, -WHEEL_NOTCH, WHEEL_NOTCH);
  }

  // Zoom by `factor` while keeping the world point under (sx, sy) fixed on screen.
  private zoomAtScreen(factor: number, sx: number, sy: number): void {
    const cam = this.host.cameras.main;
    const z = Phaser.Math.Clamp(cam.zoom * factor, this.minZoom, this.maxZoom);
    if (z === cam.zoom) return;
    this.userAdjusted = true;
    this.releaseFollow(); // zooming by hand is an explicit camera grab → stop following
    const before = cam.getWorldPoint(sx, sy);
    cam.setZoom(z);
    const after = cam.getWorldPoint(sx, sy);
    cam.scrollX += before.x - after.x;
    cam.scrollY += before.y - after.y;
  }

  // A hand gesture stops the flight NOW (so this frame's fly() won't yank the view back) and
  // tells React to flip follow off. No-op when not following.
  private releaseFollow(): void {
    if (!this.following) return;
    this.following = false;
    this.onGrabCamera?.();
  }

  private fit(): void {
    const { worldW, worldH } = this.host.map;
    const cam = this.host.cameras.main;
    cam.setBackgroundColor(THEME.bg);
    const fit = Math.min(this.host.scale.width / worldW, this.host.scale.height / worldH) * 0.98 || 1;
    this.minZoom = fit * 0.8; // can't zoom out much past the whole-map fit
    // Up to the character art's native resolution (1/scale) — past it there is only blur.
    // `fit * 6` is the floor so a small map still zooms usefully.
    this.maxZoom = Math.max(fit * 6, 1 / this.host.cast.scale);
    // Bounds centred and grown to the most-zoomed-out viewport. NOT tight to the map: at fit
    // zoom the viewport is taller than the ~2:1 map and Phaser's clamp would top-align it.
    const viewW = this.host.scale.width / this.minZoom;
    const viewH = this.host.scale.height / this.minZoom;
    const bw = Math.max(worldW, viewW);
    const bh = Math.max(worldH, viewH);
    cam.setBounds(worldW / 2 - bw / 2, worldH / 2 - bh / 2, bw, bh);
    if (this.userAdjusted) {
      // Respect the user's current zoom/pan; just keep it within the new bounds.
      cam.setZoom(Phaser.Math.Clamp(cam.zoom, this.minZoom, this.maxZoom));
    } else {
      cam.setZoom(fit);
      cam.centerOn(worldW / 2, worldH / 2);
    }
  }
}
