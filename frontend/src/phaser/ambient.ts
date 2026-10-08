/**
 * Ambient life: a day/night wash over the ground, and lamplight under the places people are in.
 *
 * Continuous, not per-step ephemeral — never cleared between steps. Nothing else belongs here:
 * effects keyed on one map's tile names (a water shimmer, say) are stand-ins for art.
 */

import Phaser from "phaser";

import type { WorldMap } from "./worldMap";

// The wash is cut larger than the viewport: it is re-hung before the camera's move is drawn,
// so a fast pan would otherwise expose a hairline of untinted map at one edge.
const AMBIENT_OVERSCAN = 1.08;
// Lamplight for the dark hours. Warm ADD-blended amber reads as a lamp burning through the
// blue night tint, not a lighter patch of it.
const LAMP_TEX = "lamp-glow";
const LAMP_TEX_R = 64; // texture radius in px; the pool's real size comes from setScale
const LAMP_COLOR = 0xffc879;
// Above the wash (1) and below the room names (5): a lamp lights the GROUND, never the labels.
const LAMP_DEPTH = 2;

/** The slice of the scene the ambient layer needs: somewhere to draw, and the map's rooms. */
export interface AmbientHost extends Phaser.Scene {
  readonly map: WorldMap;
}

export class AmbientLight {
  private dayNight?: Phaser.GameObjects.Rectangle;
  private lastHour = -1; // last world hour applied, so the same hour is not re-applied
  // Reusable glow pool; see refreshLamps.
  private lamps: Phaser.GameObjects.Image[] = [];
  private litRooms = new Map<string, number>(); // room id → how many are standing there
  private lampStrength = 0; // 0 by day, 1 at deep night; set with the wash

  constructor(private host: AmbientHost) {}

  mount(): void {
    // Day/night wash: a screen-space tint over the GROUND only. Depth 1, just above terrain
    // and under room names (5), place spotlight (6) and entity markers (8): a wash over the
    // labels would cap how dark night may get at how legible they stay.
    //
    // Sized to the canvas and re-hung every frame (follow). Don't use a fixed world rectangle
    // with `setScrollFactor(0)`: that cancels pan but not zoom, so zoomed out the night stops
    // in a hard line across the screen.
    //
    // Strength is the OBJECT alpha (set by setHour), starting at 0, so it can only tint.
    this.dayNight = this.host.add
      .rectangle(0, 0, this.host.scale.width * AMBIENT_OVERSCAN, this.host.scale.height * AMBIENT_OVERSCAN, 0x1a1a4a, 1)
      .setOrigin(0.5)
      .setDepth(1)
      .setAlpha(0);
    // The canvas takes the panel's shape (see MapStage), so the wash has to follow it.
    this.host.scale.on("resize", () =>
      this.dayNight?.setSize(this.host.scale.width * AMBIENT_OVERSCAN, this.host.scale.height * AMBIENT_OVERSCAN),
    );

    // One soft disc of stacked translucent circles — Phaser's Graphics has no radial gradient.
    if (!this.host.textures.exists(LAMP_TEX)) {
      const g = this.host.make.graphics({ x: 0, y: 0 }, false);
      for (let r = LAMP_TEX_R; r > 0; r -= 1) {
        g.fillStyle(0xffffff, Math.pow(1 - r / LAMP_TEX_R, 2.4) * 0.045);
        g.fillCircle(LAMP_TEX_R, LAMP_TEX_R, r);
      }
      g.generateTexture(LAMP_TEX, LAMP_TEX_R * 2, LAMP_TEX_R * 2);
      g.destroy();
    }
  }

  /** Which rooms are inhabited this step, and by how many. Relights the lamps. */
  lightRooms(crowds: Map<string, number>): void {
    this.litRooms = crowds;
    this.refreshLamps();
  }

  /**
   * A warm pool on the ground of every INHABITED location while the hour is dark. Lit rooms
   * are where the cast stands this step (`litRooms`, from Staging.layoutRooms — the dead and
   * those in transit light nothing), so the night shows where the city is awake.
   */
  private refreshLamps(): void {
    let i = 0;
    if (this.lampStrength > 0) {
      for (const [roomId, crowd] of this.litRooms) {
        const room = this.host.map.rooms.get(roomId);
        if (!room) continue;
        const lamp =
          this.lamps[i] ??
          (this.lamps[i] = this.host.add
            .image(0, 0, LAMP_TEX)
            .setDepth(LAMP_DEPTH)
            .setBlendMode(Phaser.BlendModes.ADD)
            .setTint(LAMP_COLOR));
        i++;
        // Sized to the room's iso footprint; a crowd widens it a little.
        const halfW = ((room.gw + room.gh) / 4) * this.host.map.tileW;
        const spread = Math.min(1.5, 1 + 0.07 * (crowd - 1));
        const sx = (Math.max(halfW, this.host.map.tileW * 1.6) * spread) / LAMP_TEX_R;
        lamp
          .setPosition(room.sx, room.sy)
          // Squashed to the iso ground plane.
          .setScale(sx, sx * (this.host.map.tileH / this.host.map.tileW))
          .setAlpha(this.lampStrength * Math.min(0.85, 0.45 + 0.1 * crowd))
          .setVisible(true);
      }
    }
    for (; i < this.lamps.length; i++) this.lamps[i].setVisible(false);
  }

  /**
   * Day/night wash from the code-layer hour (`StepEvent.world_time.hour`). NEVER parse the
   * narrative label for the hour: that binds the renderer to one theme's calendar, and other
   * worlds would freeze the wash on its last tint.
   */
  setHour(hour: number | null | undefined): void {
    if (!this.dayNight || hour == null) return;
    // No hour → keep the current wash; resetting would fight the real tint every other render.
    if (!Number.isFinite(hour) || hour === this.lastHour) return;
    this.lastHour = hour;
    // Night is saturated blue, not gray, so terrain keeps its drawing. Strong enough to notice
    // unprompted — affordable only because the wash sits under every label (see mount).
    // `lamp` rides the same buckets: a second time-of-day table would drift.
    let color = 0x3a4a86;
    let alpha = 0;
    let lamp = 0;
    if (hour >= 6 && hour < 10) { color = 0xe8a850; alpha = 0.26; lamp = 0.20; } // dawn (warm, lamps guttering out)
    else if (hour >= 10 && hour < 16) { color = 0xffffff; alpha = 0.0; lamp = 0; } // daylight (clear)
    else if (hour >= 16 && hour < 18) { color = 0xe8b868; alpha = 0.18; lamp = 0; } // afternoon (faint warm)
    else if (hour >= 18 && hour < 20) { color = 0xd97a34; alpha = 0.40; lamp = 0.55; } // dusk (amber, lamps lit)
    else if (hour >= 20) { color = 0x2a3a72; alpha = 0.52; lamp = 0.90; } // night (blue)
    else { color = 0x1e2a5c; alpha = 0.64; lamp = 1.0; } // 0–5 deep night (blue, not black)
    // Set directly, never tweened: a tween restarted on every render stays pinned near its start.
    this.dayNight.setFillStyle(color).setAlpha(alpha);
    this.lampStrength = lamp;
    // Relight now, or the lamps keep the previous hour's setting for a whole step.
    this.refreshLamps();
  }

  /**
   * Re-hang the wash on the viewport; counter-scaling by 1/zoom makes it screen-space. Called
   * last in the scene's update(), so it reads this frame's camera.
   */
  follow(): void {
    if (!this.dayNight) return;
    const cam = this.host.cameras.main;
    this.dayNight.setPosition(cam.midPoint.x, cam.midPoint.y).setScale(1 / cam.zoom);
  }
}
