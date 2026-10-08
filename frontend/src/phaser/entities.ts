/**
 * THE THINGS: what stands in the world that is not a person. How each is drawn, where it
 * stands, its badge, and the beat it plays when it changes. Also the ledger of WHO HOLDS WHAT:
 * an item's `presence` is a place or a pair of hands, and a pocket is that ledger read back.
 *
 * WITHHOLDING: a step reports the world AFTER the step, so applied up front every consequence
 * shows before its cause. A change some deed this step causes is HELD: layout and markers get a
 * doctored snapshot (`effectiveEntities`) and the beat releases it as it lands (`revealEntities`).
 */

import Phaser from "phaser";

import type { EntityView } from "../types";
import { type FxHost, flyMessageArc } from "./effects";
import { inkOn } from "./hud";
import { mix, shade } from "../lib/color";
import { type CarriedItem, FONT, TEXT_RES, drawPlate } from "./types";
import type { Room, WorldMap } from "./worldMap";

// The landmark's stone: ONE base color per tier that `shade()` lights into faces.
const STONE = 0x6b7185;         // footing
const STONE_MID = 0x7d8497;     // shaft
const STONE_CAP = 0x9aa2b6;     // cap: lightest, it catches the most sky
const STONE_EDGE = 0x2b2f3a;
const LANDMARK_ACCENT = 0x8f9bb0;
const HALO_FLASH = 0.5;         // one-shot peak when a landmark's state changes

/**
 * Did anything the map DRAWS about this thing change? Only the fields markers read: a change
 * elsewhere would withhold a beat for nothing visible. A new thing has changed: it appears.
 */
function entityChanged(prev: EntityView | undefined, next: EntityView): boolean {
  if (!prev) return true;
  return (
    prev.presence !== next.presence ||
    prev.presence_ref !== next.presence_ref ||
    prev.name !== next.name ||
    prev.state !== next.state
  );
}

/**
 * The slice of the scene a thing needs. Deliberately not the HUD itself: callbacks keep this
 * module from drawing whatever it likes over the map.
 */
export interface EntityHost extends FxHost {
  readonly map: WorldMap;
  /** Deal this step's things onto the ground their rooms can spare. */
  layoutFixtures(entities: Record<string, EntityView>): void;
  /** Announce a change where it happened: a thing wrecked, a state turned. */
  floatNote(x: number, y: number, text: string, accent: number): void;
  /** What people carry has changed; every pouch handle and open pocket is stale. */
  refreshCarried(): void;
  /** The hover card over a thing: one card on the map at a time, shared with the NPCs'. */
  showEntityTip(x: number, y: number, entity: EntityView): void;
  hideTip(): void;
}

export class EntityLayer {
  /** Who is holding what, by agent id, rebuilt every step. Written here; the pockets only read it. */
  readonly carriedBy = new Map<string, CarriedItem[]>();

  /** By entity id. Public: effects (joltNearbyEntities) and the staging read them. */
  readonly entityMarkers = new Map<string, Phaser.GameObjects.Container>();

  /** Where each thing was dealt this step. Written by the STAGING, read here to place markers. */
  readonly entitySlotOf = new Map<string, { x: number; y: number; gx: number; gy: number }>();
  /** The snapshot the markers are currently DRAWN from, not the one the step reported. */
  private lastEntityStates: Record<string, EntityView> = {};
  private stepEntities: Record<string, EntityView> = {};
  /** Things whose reported change is being held back until its cause plays. */
  private withheldEntities = new Set<string>();
  /**
   * Things some deed this step BROUGHT INTO THE WORLD; they get one arrival when their beat lands.
   *
   * Needs both halves: the step names it (`causedIds`) AND the map never drew it. Diffing alone
   * fires on every thing made since the last drawn step after a scrub; `causedIds` alone fires
   * on a thing that merely changed. Deleted as drawn, so a re-run does not replay the arrival.
   */
  private bornEntities = new Set<string>();

  /** Which things are still holding a change back; read by the beat scheduler. */
  get withheld(): ReadonlySet<string> {
    return this.withheldEntities;
  }

  constructor(private host: EntityHost) {}

  /**
   * Take this step's report, hold back every change a deed this step will cause, and return the
   * doctored view downstream works from. The decision stays here: the caller knows which things
   * the deeds name, this layer knows which would visibly change, and neither half is the test.
   */
  beginStep(entities: Record<string, EntityView>, causedIds: Set<string>): Record<string, EntityView> {
    this.stepEntities = entities;
    this.withheldEntities = new Set(
      Object.keys(entities).filter(
        (eid) => causedIds.has(eid) && entityChanged(this.lastEntityStates[eid], entities[eid]),
      ),
    );
    this.bornEntities = new Set(
      Object.keys(entities).filter((eid) => causedIds.has(eid) && !this.lastEntityStates[eid]),
    );
    return this.effectiveEntities();
  }

  /** Land every change whose causing beat never played (focus-gated out, say). */
  revealWithheld(): void {
    this.revealEntities([...this.withheldEntities]);
  }

  // Float the loss and remove the marker: the one place the instant path, the deferred path
  // (playTrack) and the safety net all say goodbye. A destroyed thing never keeps its marker
  // across steps, so a marker present here is always a fresh loss.
  destroyEntityMarker(eid: string, name: string): void {
    const marker = this.entityMarkers.get(eid);
    if (!marker) return;
    this.host.floatNote(marker.x, marker.y - 20, `${name} 已损毁`, 0xc65f4e);
    marker.destroy();
    this.entityMarkers.delete(eid);
  }

  /**
   * The world as the map should DRAW it right now: this step's report, with every
   * withheld entity rolled back to what is already on screen (or absent entirely, for
   * one that only appears this step).
   *
   * The withholding lives here, in one place, rather than as a flag threaded through
   * the marker code — which is what keeps it general. applyEntities and everything under
   * it work on whatever map they are handed and never learn that a change is pending, so
   * presence, holder, position and state all defer through the same path, and revealing
   * a change is just handing over a map with one more truth in it.
   */
  effectiveEntities(): Record<string, EntityView> {
    if (!this.withheldEntities.size) return this.stepEntities;
    const out = { ...this.stepEntities };
    for (const eid of this.withheldEntities) {
      const shown = this.lastEntityStates[eid];
      if (shown) out[eid] = shown;
      else delete out[eid];
    }
    return out;
  }

  /**
   * Land withheld changes — the deed that causes them has played (or never will).
   *
   * Re-runs the fixture layout as well as the markers: a thing SET DOWN this step was
   * "held" in the withheld view and so was given no ground, and without a fresh pass it
   * would appear at its room's bare anchor. layoutFixtures is a pure function of rooms and
   * ids — it does not consult who is standing where — so re-running it mid-step is safe,
   * and re-seating a room's other fixtures by one cell is the same shuffle they already do
   * whenever a thing enters or leaves that room between steps.
   */
  revealEntities(eids: string[]): void {
    let any = false;
    for (const eid of eids) any = this.withheldEntities.delete(eid) || any;
    if (!any) return;
    const shown = this.effectiveEntities();
    this.host.layoutFixtures(shown);
    this.applyEntities(shown);
  }

  applyEntities(entities: Record<string, EntityView>): void {
    // Runs before lastEntityStates is overwritten, so this still reads what was drawn before.
    const prevOf = (eid: string): EntityView | undefined => this.lastEntityStates[eid];
    const items = new Map<string, [string, EntityView][]>();
    const landmarks = new Map<string, [string, EntityView][]>();
    // The card is free-standing, and pointerout never fires on a destroyed object.
    this.host.hideTip();
    this.carriedBy.clear();
    for (const [eid, e] of Object.entries(entities)) {
      if (e.presence === "destroyed") {
        this.destroyEntityMarker(eid, e.name);
        continue;
      }
      if (e.presence === "held") {
        // Held items leave the map for the holder's pocket (carriedBy); a marker would only
        // stack on his token. Destroyed, not hidden: a marker exists iff the thing is in the world.
        this.entityMarkers.get(eid)?.destroy();
        this.entityMarkers.delete(eid);
        const holderId = e.presence_ref;
        // Changing hands is the only part of a hand-off the map can show; without this the
        // thing just blinks between pockets. Reuses the letter's arc (one gesture, one
        // vocabulary). Fire and forget: a step that ends first just cuts it.
        const wasHeldBy = prevOf(eid)?.presence === "held" ? prevOf(eid)?.presence_ref : "";
        if (holderId && wasHeldBy && wasHeldBy !== holderId) {
          const from = this.host.token(wasHeldBy);
          const to = this.host.token(holderId);
          if (from && to) {
            void flyMessageArc(
              this.host, from.container.x, from.container.y,
              to.container.x, to.container.y, { color: to.color },
            );
          }
        }
        if (holderId) {
          this.carriedBy.set(holderId, [
            ...(this.carriedBy.get(holderId) ?? []),
            {
              name: e.name, state: e.state ?? "", concealed: e.is_public === false,
              description: e.description ?? "", content: e.content ?? "",
            },
          ]);
          // Made and kept: never on the ground, so announce it at his token (the pocket panel
          // is usually shut).
          if (this.bornEntities.delete(eid)) this.noteAtHolder(holderId, `${e.name} 已入手`);
        }
        continue;
      }
      const locId = e.presence_ref;
      if (!locId) continue;
      const bucket = e.entity_type === "landmark" ? landmarks : items;
      const list = bucket.get(locId) ?? [];
      list.push([eid, e]);
      bucket.set(locId, list);
    }
    // Both forms stand where layoutFixtures put them; a room with no reachable ground falls back
    // to its anchor. Branches on entity_type only, never on free-form `state`/`name` (Rule 7);
    // an unrecognized type gets the item form.
    for (const [bucket, make] of [
      [items, (eid: string) => this.makeItemMarker(eid)],
      [landmarks, (eid: string) => this.makeLandmarkMarker(eid)],
    ] as const) {
      for (const [locId, list] of bucket) {
        const room = this.host.map.rooms.get(locId);
        if (!room) continue;
        for (const [eid, e] of list) {
          const at = this.entitySlotOf.get(eid) ?? { x: room.sx, y: room.sy + 8 };
          // Restore scale: an arrival tween (playArrival) cut off mid-way would otherwise leave
          // the marker permanently shrunk.
          const marker = this.entityMarkers.get(eid) ?? make(eid);
          marker.setVisible(true).setPosition(at.x, at.y).setScale(1);
          this.refreshEntityMarker(marker, room, eid, e);
          if (this.bornEntities.delete(eid)) this.playArrival(marker, room, e.name);
        }
      }
    }
    // What the map now SHOWS (doctored while withheld), so the state-change note fires on reveal.
    this.lastEntityStates = { ...entities };
    // Whoever rebuilds carriedBy pushes it to the view: a mid-step reveal skips applyFocus.
    this.host.refreshCarried();
  }

  // ---- the two fixture forms ------------------------------------------------
  //
  // ONE marker set serves every world, so neither form depicts a KIND of object, and neither
  // branches on free-form `state`/`name` (Rule 7). Both are iso solids lit through `shade()` on
  // the map's one light direction, like buildings and tokens, so they belong to the world.
  //
  //           | form                    | animation                   | temperature
  //   item    | lifted 4-point star     | glow breathes, always       | warm gold
  //   landmark| three stacked iso tiers | still; flashes on a state turn | cool stone

  // One iso box: a 2:1 diamond top at `topY`, extruded down to `baseY`, lit from one base color.
  isoBox(
    g: Phaser.GameObjects.Graphics,
    cx: number, w: number, h: number, topY: number, baseY: number,
    base: number, edge: number,
  ): void {
    const top = [
      { x: cx, y: topY - h }, { x: cx + w, y: topY },
      { x: cx, y: topY + h }, { x: cx - w, y: topY },
    ];
    // West face darkest, east face mid, top lightest: one light direction for the whole map.
    g.fillStyle(shade(base, 0.62), 1).fillPoints(
      [{ x: cx - w, y: topY }, { x: cx, y: topY + h }, { x: cx, y: baseY + h }, { x: cx - w, y: baseY }], true,
    );
    g.fillStyle(shade(base, 0.82), 1).fillPoints(
      [{ x: cx, y: topY + h }, { x: cx + w, y: topY }, { x: cx + w, y: baseY }, { x: cx, y: baseY + h }], true,
    );
    g.fillStyle(base, 1).fillPoints(top, true);
    g.lineStyle(1.1, edge, 0.85).strokePoints(top, true);
  }

  // A lifted gold gem with a breathing glow. Items are sparse plot props, so all of them pulse:
  // which one matters is a narrative call the renderer cannot make theme-neutrally.
  makeItemMarker(eid: string): Phaser.GameObjects.Container {
    const shadow = this.host.add.ellipse(0, 7, 16, 8, 0x000000, 0.28);
    const glow = this.host.add.star(0, -3, 4, 7, 16, 0xffd97a).setAlpha(0.16);
    const pulse = this.host.tweens.add({
      targets: glow,
      alpha: { from: 0.1, to: 0.34 },
      scale: { from: 0.82, to: 1.14 },
      duration: 1200, yoyo: true, repeat: -1, ease: "Sine.easeInOut",
    });
    // Phaser doesn't drop a tween when its target is destroyed, and this one repeats forever.
    // Hooking DESTROY covers every removal path.
    glow.once(Phaser.GameObjects.Events.DESTROY, () => pulse.remove());
    const icon = this.host.add.star(0, -3, 4, 5, 11, 0xd9a441).setStrokeStyle(1.5, 0x0a0e1a);
    return this.mountEntityMarker(eid, [shadow, glow, icon], 16, 0xe8c06a, new Phaser.Geom.Circle(0, 0, 16));
  }

  // A LANDMARK (gate, shrine, notice board): fixed room fabric, so BUILT as footing, shaft and
  // overhanging cap. One block reads as a boulder; the overhang says "someone made this".
  // Taller, wider and cooler than an item so the two forms differ at a glance. No animation.
  makeLandmarkMarker(eid: string): Phaser.GameObjects.Container {
    // Resting alpha 0: the halo exists only so a state change can flash (flashFixture).
    const halo = this.host.add.graphics({ x: 0, y: -0.8 });
    halo.fillStyle(LANDMARK_ACCENT, 1).fillPoints(
      [{ x: 0, y: -8.8 }, { x: 17.6, y: 0 }, { x: 0, y: 8.8 }, { x: -17.6, y: 0 }], true,
    );
    halo.setAlpha(0);
    const g = this.host.add.graphics();
    g.fillStyle(0x000000, 0.3).fillEllipse(0, 1.6, 27.2, 13.6);
    this.isoBox(g, 0, 12, 6, -2.4, 1.6, STONE, STONE_EDGE);
    this.isoBox(g, 0, 7.6, 3.8, -13.6, -3.2, STONE_MID, STONE_EDGE);
    this.isoBox(g, 0, 10.4, 5.2, -19.2, -14.4, STONE_CAP, STONE_EDGE); // overhangs the shaft
    const marker = this.mountEntityMarker(eid, [halo, g], 21, LANDMARK_ACCENT, new Phaser.Geom.Circle(0, -9.6, 16));
    marker.setData("halo", halo).setData("haloRest", 0);
    return marker;
  }

  // Shared plumbing for both forms: plate, name and state badge (laid out by layoutEntityLabel).
  // `accent` fills the badge and tints the plate's border.
  mountEntityMarker(
    eid: string,
    parts: Phaser.GameObjects.GameObject[],
    labelY: number,
    accent: number,
    hit: Phaser.Geom.Circle,
  ): Phaser.GameObjects.Container {
    const font = { fontFamily: FONT };
    const plate = this.host.add.graphics();
    const name = this.host.add
      .text(0, labelY, "", { ...font, fontSize: "13px", color: "#f2efe4" })
      .setOrigin(0, 0.5)
      .setResolution(TEXT_RES);
    const state = this.host.add
      .text(0, labelY, "", { ...font, fontSize: "11px", fontStyle: "bold", color: inkOn(accent) })
      .setOrigin(0, 0.5)
      .setResolution(TEXT_RES);
    const marker = this.host.add.container(0, 0, [...parts, plate, name, state]).setDepth(8);
    marker.setData("plate", plate);
    marker.setData("nameText", name);
    marker.setData("stateText", state);
    marker.setData("accent", accent);
    marker.setData("labelY", labelY);
    // Interactive for the hover card; clicks on things select nothing (see handleClick).
    marker.setInteractive(hit, Phaser.Geom.Circle.Contains);
    marker.setData("entityId", eid);
    // Read what the map SHOWS, not the report: a withheld change must not leak into the card.
    marker.on("pointerover", () => {
      const e = this.lastEntityStates[eid];
      // Above the hit circle, so the upward-growing card never buries the name plate.
      if (e) this.host.showEntityTip(marker.x, marker.y + hit.y - hit.radius, e);
    });
    marker.on("pointerout", () => this.host.hideTip());
    this.entityMarkers.set(eid, marker);
    return marker;
  }

  // Name on a dark plate, state on its own accent BADGE (the narratively loaded half). Never
  // colored by what the state means: the vocabulary is the theme's (Rule 7).
  layoutEntityLabel(marker: Phaser.GameObjects.Container, name: string, state: string): void {
    const plate = marker.getData("plate") as Phaser.GameObjects.Graphics;
    const nameText = marker.getData("nameText") as Phaser.GameObjects.Text;
    const stateText = marker.getData("stateText") as Phaser.GameObjects.Text;
    const accent = marker.getData("accent") as number;
    const labelY = marker.getData("labelY") as number;

    nameText.setText(name);
    stateText.setText(state);
    const hasState = state.trim().length > 0;
    const padX = 5, gap = 6, plPad = 5;
    const nameW = nameText.width;
    const badgeW = hasState ? stateText.width + padX * 2 : 0;
    const total = nameW + (hasState ? gap + badgeW : 0);
    const h = Math.max(nameText.height, stateText.height) + 2;
    const x0 = -total / 2;

    plate.clear();
    drawPlate(plate, x0 - plPad, labelY - h / 2 - 2, total + plPad * 2, h + 4, 0x12131c, mix(0x12131c, accent, 0.5));
    nameText.setPosition(x0, labelY);
    if (hasState) {
      const badgeX = x0 + nameW + gap;
      plate.fillStyle(accent, 1);
      plate.fillRoundedRect(badgeX, labelY - h / 2, badgeW, h, 4);
      stateText.setColor(inkOn(accent)).setPosition(badgeX + padX, labelY).setVisible(true);
    } else {
      stateText.setVisible(false);
    }
  }

  refreshEntityMarker(
    marker: Phaser.GameObjects.Container,
    room: Room,
    eid: string,
    e: EntityView,
  ): void {
    this.layoutEntityLabel(marker, e.name, e.state);
    const prev = this.lastEntityStates[eid];
    if (prev && prev.state !== e.state) {
      this.host.floatNote(room.sx, room.sy - 26, `${e.name}：${prev.state}→${e.state}`, 0xd9a441);
      this.flashFixture(marker);
    }
  }

  /**
   * A thing ARRIVES on the ground: it swells into its slot and says its name once. The mirror
   * of `destroyEntityMarker`. Scales rather than fades, so it does not read as a redraw.
   */
  playArrival(marker: Phaser.GameObjects.Container, room: Room, name: string): void {
    this.host.floatNote(room.sx, room.sy - 26, `${name} 已出现`, 0xe8c06a);
    marker.setScale(0);
    this.host.tweens.add({
      targets: marker,
      scale: { from: 0, to: 1 },
      duration: 420,
      ease: "Back.easeOut",
    });
  }

  /** A thing arrives in someone's hands: said over the holder. Silent when he has no token. */
  noteAtHolder(holderId: string, text: string): void {
    const tok = this.host.token(holderId);
    if (!tok) return;
    this.host.floatNote(tok.container.x, tok.container.y - 34, text, 0xe8c06a);
  }

  // One-shot swell of a LANDMARK's halo alongside the floated note: the note says what changed,
  // the swell says WHICH thing. No-op for an item (no `halo`): flashing would kill its breathing tween.
  flashFixture(marker: Phaser.GameObjects.Container): void {
    const halo = marker.getData("halo") as Phaser.GameObjects.Graphics | undefined;
    if (!halo) return;
    this.host.tweens.killTweensOf(halo);
    halo.setAlpha(0).setScale(1);
    this.host.tweens.add({
      targets: halo,
      alpha: { from: HALO_FLASH, to: 0 },
      scale: { from: 1.5, to: 1 },
      duration: 900,
      ease: "Cubic.easeOut",
    });
  }
}
