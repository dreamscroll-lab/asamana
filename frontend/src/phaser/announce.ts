/**
 * A broadcast on the map: its WORDS (a notice over the place, or the world banner) and its FACE (the
 * weather it reports). When each plays is the scene's call; how it looks is decided here.
 */

import Phaser from "phaser";

import { mix, toHex } from "../lib/color";
import type { BroadcastSummary } from "../types";
import { type FxHost, emanateRings } from "./effects";
import type { MapHud } from "./hud";
import { FONT, PLATE_R, TEXT_RES, WRAP_W, drawPlate } from "./types";
import { type WeatherArea, type WeatherHost, type WeatherSpell, playWeather } from "./weather";
import type { Room, WorldMap } from "./worldMap";

// How a notice comes and goes — shared by the world banner and the place plate. Both must
// fade in, hold and leave: a plate that never leaves stacks under the next notice.
const NOTICE_IN = 400;
const NOTICE_HOLD = 2800;
const NOTICE_OUT = 700;

/** The slice of the scene the announcer needs, on top of the shared FX and weather ones. */
export interface AnnounceHost extends FxHost, WeatherHost {
  readonly map: WorldMap;
  tweenP(config: Phaser.Types.Tweens.TweenBuilderConfig): Promise<void>;
}

export class Announcer {
  // The weather currently falling. NOT in the scene's `ephemeral` (destroyed at the next
  // step): a spell is retired instead — emission stops, the air clears, it disposes itself.
  private weather: WeatherSpell[] = [];
  // Active world-wide broadcast. fx/fy are viewport FRACTIONS (resize-robust); a is
  // tween-driven alpha; pinBanner applies them.
  private broadcastHud?: { obj: Phaser.GameObjects.Container; fx: number; fy: number; a: number };

  constructor(private host: AnnounceHost, private hud: MapHud) {}

  /**
   * Called every frame: pin the world-wide broadcast (a world-space object) to a fixed screen
   * point at constant size — getWorldPoint for position, 1/zoom to cancel the camera scale.
   */
  pinBanner(): void {
    if (!this.broadcastHud) return;
    const p = this.broadcastHud;
    if (p.obj.active) {
      const cam = this.host.cameras.main;
      const wp = cam.getWorldPoint(p.fx * this.host.scale.width, p.fy * this.host.scale.height);
      p.obj.setPosition(wp.x, wp.y).setScale(1 / cam.zoom).setAlpha(p.a);
    } else {
      this.broadcastHud = undefined;
    }
  }

  // The visible face of a world change (rain, fire, smoke). The engine sends a FACT from a
  // closed vocabulary; its look is decided only in weather.ts. Scale comes from `severity`.
  // Returns whether anything drew, so the caller can stand the broadcast's words down.
  playPhenomenon(b: BroadcastSummary, room: Room | undefined): boolean {
    const phenomenon = String(b.phenomenon ?? "");
    if (!phenomenon || phenomenon === "none") return false;
    const spell = playWeather(
      this.host,
      phenomenon,
      String(b.severity ?? "medium"),
      this.weatherArea(room),
      { tilePx: this.host.map.tileH, zoom: this.host.cameras.main.zoom },
    );
    if (spell) this.weather.push(spell);
    return spell !== null;
  }

  /**
   * Let go of the weather on screen. Emptied, not awaited: each retiring spell owns its end
   * and destroys its objects, so several may fade at once without leaking.
   */
  retireWeather(): void {
    for (const spell of this.weather) spell.retire();
    this.weather = [];
  }

  /** Where a phenomenon belongs — ONE outline, never two code paths.
   *
   *  Scoped: the ward's projected diamond. Don't emit across its bounding box: that is about
   *  twice its area and rains on the neighbours.
   *
   *  Global: the WHOLE MAP's outline, not the camera's view — otherwise the storm moves
   *  whenever somebody drags the map. */
  private weatherArea(room: Room | undefined): WeatherArea {
    const corners = room
      ? [
          this.host.map.iso(room.gx0, room.gy0),
          this.host.map.iso(room.gx0 + room.gw, room.gy0),
          this.host.map.iso(room.gx0 + room.gw, room.gy0 + room.gh),
          this.host.map.iso(room.gx0, room.gy0 + room.gh),
        ]
      : [
          this.host.map.iso(0, 0),
          this.host.map.iso(this.host.map.cols, 0),
          this.host.map.iso(this.host.map.cols, this.host.map.rows),
          this.host.map.iso(0, this.host.map.rows),
        ];
    const xs = corners.map((c) => c.x);
    const ys = corners.map((c) => c.y);
    const box = {
      x: Math.min(...xs),
      y: Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
    };
    return {
      footprint: corners,
      box,
      // Only the quake reads this: the camera is the VIEWER's eye, so ground shaking out of
      // sight must not shake it (see weatherPlan.quake).
      onScreen: Phaser.Geom.Rectangle.Overlaps(
        this.host.cameras.main.worldView,
        new Phaser.Geom.Rectangle(box.x, box.y, box.width, box.height),
      ),
    };
  }

  // A broadcast's WORDS — the map's only public announcement (`world_events` are not drawn;
  // see TiledWorldScene.draw). Location-scoped → a subtitle over the place + ground waves;
  // world-wide → playWorldBroadcast. The phenomenon is played separately, before the deeds;
  // callers skip this for a broadcast whose weather drew.
  playBroadcast(b: Pick<BroadcastSummary, "content" | "location_scope">, from?: { name: string; ink: number }): Promise<void> {
    const content = String(b.content ?? "");
    if (!content) return Promise.resolve();
    const room = this.host.map.rooms.get(String(b.location_scope ?? ""));
    if (!room) return this.playWorldBroadcast(content, from);
    // -46 clears the room's own name (which reaches up to about sy - 20).
    const text = this.host.add
      .text(room.sx, room.sy - 46, content, {
        fontFamily: FONT,
        fontSize: "14px",
        fontStyle: "italic",
        color: "#c2e4d0",
        padding: { x: 11, y: 6 },
        // Left-aligned: a notice is read from its margin; centred it reads as decoration.
        align: "left",
        wordWrap: { width: WRAP_W, useAdvancedWrap: true },
      })
      .setOrigin(0.5, 1)
      .setDepth(46)
      .setResolution(TEXT_RES);
    this.host.pushEphemeral(text);
    const parts: (Phaser.GameObjects.Text | Phaser.GameObjects.Graphics)[] = [text];
    // An eyebrow naming the voice, ONLY when somebody said it. Unlike the world banner this
    // needs no channel label: the place's own name sits right beneath it.
    if (from) {
      const tag = this.host.add
        .text(text.x - text.width / 2 + 11, text.y - text.height + 3, `◈ ${from.name} ◈`, {
          fontFamily: FONT,
          fontSize: "12px",
          fontStyle: "bold",
          color: toHex(mix(from.ink, 0xffffff, 0.45)),
          padding: { x: 0, y: 3 },
        })
        .setOrigin(0, 1)
        .setDepth(46)
        .setResolution(TEXT_RES);
      this.host.pushEphemeral(tag);
      // ONE hand-drawn plate around both rows: plateBehind fits a single Text, and two plates read as two notices.
      const bx = text.x - text.width / 2;
      const bw = Math.max(text.width, tag.width + 22);
      const by = tag.y - tag.height;
      const g = this.host.add.graphics().setDepth(45);
      drawPlate(g, bx, by, bw, text.y - by, 0x0e1a14, 0x3f6a55);
      this.host.pushEphemeral(g);
      parts.push(tag, g);
    } else {
      parts.push(this.hud.plateBehind(text, 0x0e1a14, 0x3f6a55));
    }
    emanateRings(this.host, room.sx, room.sy, 0x5a9a7a, { count: 4, maxR: 110, duration: 1500, stagger: 260, width: 3 });
    // In, hold, out — the banner's arc, faded as one object.
    parts.forEach((o) => o.setAlpha(0));
    this.host.tweens.add({ targets: parts, alpha: 1, duration: NOTICE_IN, ease: "Sine.easeOut" });
    return this.host.delayP(NOTICE_IN + NOTICE_HOLD).then(() =>
      this.host.tweenP({ targets: parts, alpha: 0, duration: NOTICE_OUT, ease: "Sine.easeIn" }),
    );
  }

  /**
   * A world-wide broadcast: a generic channel, not any setting's ritual (CLAUDE.md Rule 7) — a
   * notice in the top band, arriving on one pulse, pinned by pinBanner. It STAYS PUT (drifting
   * down, it would cover the figures) and carries ONE signature, so it doesn't out-shout its sentence.
   *
   * `from`: a character proclaiming puts his name on the eyebrow; same surface, not a second banner.
   */
  private playWorldBroadcast(content: string, from?: { name: string; ink: number }): Promise<void> {
    const w = this.host.scale.width;
    const g = 0x8fe6c0; // broadcast channel hue (soft teal-green light)

    const label = this.host.add
      .text(0, 0, content, {
        fontFamily: FONT,
        fontSize: "22px",
        fontStyle: "bold",
        color: "#eafff6",
        align: "left",
        wordWrap: { width: Math.min(w * 0.6, 540), useAdvancedWrap: true },
      })
      .setOrigin(0, 0.5)
      .setResolution(TEXT_RES);
    // Generic channel eyebrow (not theme content), so the sentence isn't unlabelled floating text.
    const tag = this.host.add
      .text(0, 0, from ? `◈ ${from.name}信息 ◈` : "◈ 世界广播 ◈", {
        fontFamily: FONT,
        fontSize: "12px",
        fontStyle: "bold",
        // Signed → the speaker's own colour; unsigned → the channel hue.
        color: from ? toHex(mix(from.ink, 0xffffff, 0.45)) : "#9fd9c2",
      })
      // Left-aligned: centred, a label reads as a tiny second line of content.
      .setOrigin(0, 0.5)
      .setResolution(TEXT_RES);

    const padX = 22;
    const padY = 11;
    const gap = 5;
    const bw = Math.max(label.width, tag.width) + padX * 2;
    const bh = tag.height + gap + label.height + padY * 2;
    const top = -bh / 2;
    tag.setPosition(-bw / 2 + padX, top + padY + tag.height / 2);
    label.setPosition(-bw / 2 + padX, top + padY + tag.height + gap + label.height / 2);
    const plate = this.host.add.graphics();
    drawPlate(plate, -bw / 2, top, bw, bh, 0x0d1a15, 0x4e9a7c);

    // The one signature: the plate's own border pushing outward once. Redrawn per frame, not
    // scaled, so the stroke stays hairline and corners keep their radius. Not an ellipse: that
    // reads as a second shape arriving.
    const pulse = this.host.add.graphics().setBlendMode(Phaser.BlendModes.ADD);
    const wave = { e: 0, a: 0.7 };
    this.host.tweens.add({
      targets: wave,
      e: 26,
      a: 0,
      duration: 1000,
      ease: "Cubic.easeOut",
      // The tween outlives a graphic destroyed by a scrub — draw only while it exists.
      onUpdate: () => {
        if (!pulse.active) return;
        pulse.clear();
        pulse.lineStyle(2, g, wave.a);
        pulse.strokeRoundedRect(
          -bw / 2 - wave.e, top - wave.e, bw + wave.e * 2, bh + wave.e * 2, PLATE_R + wave.e * 0.6,
        );
      },
    });

    const obj = this.host.add.container(0, 0, [pulse, plate, tag, label]).setDepth(120);
    this.host.pushEphemeral(obj);

    const p = { obj, fx: 0.5, fy: 0.09, a: 0 };
    this.broadcastHud = p;

    this.host.tweens.add({ targets: p, a: 1, duration: NOTICE_IN, ease: "Sine.easeOut" });

    return this.host.delayP(NOTICE_HOLD).then(() =>
      this.host.tweenP({ targets: p, a: 0, duration: NOTICE_OUT, ease: "Sine.easeIn" }),
    );
  }
}
