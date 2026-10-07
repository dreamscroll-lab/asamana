/**
 * Where everyone rests on the map this step: the per-room force layout that seats the cast (drawn
 * together by relationship and by what they are doing to each other), the ground each fixture takes,
 * and the nameplates that keep co-located figures legible.
 *
 * Render-layer only. It never travels back: the simulation has no intra-location geometry and must not
 * acquire one from a picture of itself.
 */

import Phaser from "phaser";

import type { ActionSummary, AgentStateSummary, EntityView, GraphEdge } from "../types";
import { ActionType, actedOnAgents, actedOnEntities, actedOnNpcs, reachedAgents } from "../lib/contract";
import type { EntityLayer } from "./entities";
import { type AgentToken, PLATE_GAP } from "./types";
import type { Room, WorldMap } from "./worldMap";

// How hard two figures dealing with each other this step are drawn together, against affinity's
// 0.16. Decisively stronger: an assault is often between people whose affinity pushes them apart,
// and the blow still lands within arm's reach. Only a preference for where in the room the
// encounter settles; adjacency itself is guaranteed by assignSlots, which seats engaged groups
// (and anyone acting on a fixture) first, since quantising onto cells can separate them.
const ENGAGE_PULL = 0.55;
// Unordered pair key, so a relation or an encounter is one entry however it is named.
const pairKey = (a: string, b: string): string => (a < b ? `${a}|${b}` : `${b}|${a}`);
/**
 * The PEOPLE a beat is about — who its actor turns toward and stands beside, primary first.
 *
 * Agents before NPCs, so a deed naming both turns to the one who can answer; receivers last: a
 * hand-over is done to the thing, and its receiver is the only person it names.
 */
export const subjectPeopleIds = (act: ActionSummary): string[] => [
  ...actedOnAgents(act.target), ...actedOnNpcs(act.target), ...reachedAgents(act.target),
];
/**
 * The THINGS a beat is about — what its actor turns toward and stands beside.
 *
 * Aim first, effect second: `affected_entity_ids` (what the step changed) is empty when a man
 * heaved at a gate and it held, while `target.acts_on` survives failure. The fallback is still
 * needed: a seize names its prize only through the change (its target may have been a person).
 */
export const subjectEntityIds = (act: ActionSummary): string[] => {
  const aimed = actedOnEntities(act.target);
  return aimed.length ? aimed : act.affected_entity_ids ?? [];
};

/** The slice of the scene the layout needs: the map, the tokens, and who is no longer on it. */
export interface StagingHost {
  readonly map: WorldMap;
  token(id: string): AgentToken | undefined;
  /** Reported dead with no token left on the map — takes no room. */
  gone(state: AgentStateSummary): boolean;
}

export class Staging {
  // Symmetric per-pair affinity in [-1,1] (avg of both directions' trust+affection),
  // fed from the backend relationship graph. Drives the resting force-layout: allies
  // (affinity > 0) draw together, rivals (< 0) push apart. Key = sorted "idA|idB".
  private relAffinity = new Map<string, number>();
  // Per-step computed resting slot (screen px) for each co-located agent, from
  // layoutRooms(); positionFor() reads it. Cleared + rebuilt each render.
  private slotOf = new Map<string, { x: number; y: number }>();
  // room id → which of that room's stand cells the fixtures took, so the cast doesn't
  // get assigned ground something is already standing on.
  private fixtureCells = new Map<string, Set<number>>();

  constructor(private host: StagingHost, private entities: EntityLayer) {}

  /**
   * Feed the step's relationship edges (`StepEvent.relations`). Collapsed to a
   * symmetric per-pair affinity in [-1,1] that the resting layout uses to draw
   * allies together and push rivals apart.
   */
  setRelations(edges: GraphEdge[]): void {
    const acc = new Map<string, { sum: number; n: number }>();
    for (const e of edges) {
      if (!e.from_id || !e.to_id) continue;
      const key = e.from_id < e.to_id ? `${e.from_id}|${e.to_id}` : `${e.to_id}|${e.from_id}`;
      // trust + affection each ~[-1,1]; average → a single closeness signal.
      const v = Math.max(-1, Math.min(1, ((e.trust ?? 0) + (e.affection ?? 0)) / 2));
      const cur = acc.get(key) ?? { sum: 0, n: 0 };
      cur.sum += v;
      cur.n += 1;
      acc.set(key, cur);
    }
    this.relAffinity.clear();
    for (const [key, { sum, n }] of acc) this.relAffinity.set(key, sum / n);
  }

  private affinity(a: string, b: string): number {
    return this.relAffinity.get(pairKey(a, b)) ?? 0;
  }

  /**
   * Compute every room's resting slots for this step and cache them in slotOf. Returns how many
   * are standing in each room — the one count the lamps are lit from, so the two never disagree.
   *
   * A small deterministic force sim in tile space per room: baseline repulsion keeps figures
   * apart, affinity draws allies together and pushes rivals apart, engagement draws whoever is
   * dealing with each other into arm's reach (otherwise a conversation plays out from opposite
   * corners), and a main character is anchored toward the room centre. Clamped to the room
   * footprint; iso-projected to screen at the end. Render-layer only (see the file header).
   *
   * Called once per render, before the scene's moveAgent and layoutLabels read positionFor.
   */
  layoutRooms(
    states: AgentStateSummary[],
    actions: ActionSummary[],
    entities: Record<string, EntityView>,
  ): Map<string, number> {
    this.slotOf.clear();
    const byRoom = new Map<string, AgentStateSummary[]>();
    for (const s of states) {
      if (s.transit) continue;
      // The long dead take no room, or each would hold a cell where he died for the rest of
      // the run. Whoever is falling this step still has a token and is still seated.
      if (this.host.gone(s)) continue;
      const roomId = s.location_id;
      if (!this.host.map.rooms.has(roomId)) continue;
      (byRoom.get(roomId) ?? byRoom.set(roomId, []).get(roomId)!).push(s);
    }
    // The fixtures take their ground first — see layoutFixtures. The cast is then laid
    // out around them, and reads their cells for the anchor below.
    this.layoutFixtures(entities);
    // Who is dealing with whom this step, symmetric. A not_executed action draws no one: pulling
    // him toward someone he never reached would assert the opposite of what happened.
    const engaged = new Set<string>();
    // Where the thing he is about to act on stands, for an actor acting on a thing, not a person.
    const anchor = new Map<string, { gx: number; gy: number }>();
    for (const a of actions) {
      if (a.not_executed) continue;
      // Read off the target, not the action type: whoever a deed names is dealt with in person.
      // The one exception is a letter — its recipients are reached at a distance, and
      // directedMessages draws the envelope arcing between them. Someone named but in another room
      // is left out by the per-room layout below.
      if (a.action_type === ActionType.send_message) continue;
      const people = subjectPeopleIds(a);
      for (const t of people) {
        if (t !== a.agent_id) engaged.add(pairKey(a.agent_id, t));
      }
      if (people.length) continue; // a person outranks a thing
      for (const eid of subjectEntityIds(a)) {
        const slot = this.entities.entitySlotOf.get(eid);
        if (!slot) continue;
        anchor.set(a.agent_id, { gx: slot.gx + 0.5, gy: slot.gy + 0.5 });
        break; // the primary one, as faceEntity does
      }
    }
    for (const [roomId, list] of byRoom) {
      const room = this.host.map.rooms.get(roomId)!;
      const agents = [...list].sort((a, b) => (a.agent_id < b.agent_id ? -1 : 1));
      const n = agents.length;
      if (n === 1) {
        // Alone in the room, he stands at its centre, or beside the thing he is about to act on.
        const a = anchor.get(agents[0].agent_id);
        const at = a
          ? {
              gx: Phaser.Math.Clamp(a.gx, room.gx - room.gw / 2, room.gx + room.gw / 2),
              gy: Phaser.Math.Clamp(a.gy, room.gy - room.gh / 2, room.gy + room.gh / 2),
            }
          : { gx: room.gx, gy: room.gy };
        this.assignSlots(room, [agents[0].agent_id], [at], engaged, [a]);
        continue;
      }
      const halfW = Math.max(0.7, room.gw / 2 - 1);
      const halfH = Math.max(0.7, room.gh / 2 - 1);
      // Seed on a deterministic ring (no RNG → identical inputs give identical, stable
      // layouts, so co-located agents don't jitter between steps).
      const px = agents.map((_, i) => Math.cos((i / n) * Math.PI * 2) * halfW * 0.55);
      const py = agents.map((_, i) => Math.sin((i / n) * Math.PI * 2) * halfH * 0.55);
      const minSep = Math.min(1, Math.max(0.7, (2 * Math.min(halfW, halfH)) / Math.max(1, n - 1)));
      for (let iter = 0; iter < 70; iter++) {
        const fx = new Array(n).fill(0);
        const fy = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            let dx = px[i] - px[j];
            let dy = py[i] - py[j];
            let d = Math.hypot(dx, dy) || 0.001;
            dx /= d;
            dy /= d;
            // Baseline repulsion inside the personal-space radius.
            if (d < minSep) {
              const push = (minSep - d) * 0.5;
              fx[i] += dx * push; fy[i] += dy * push;
              fx[j] -= dx * push; fy[j] -= dy * push;
            }
            // Engagement replaces affinity rather than adding to it: additive, an assault
            // between enemies would settle halfway apart, the case this term exists to fix.
            if (engaged.has(pairKey(agents[i].agent_id, agents[j].agent_id))) {
              const pull = (d - minSep) * ENGAGE_PULL;
              fx[i] -= dx * pull; fy[i] -= dy * pull;
              fx[j] += dx * pull; fy[j] += dy * pull;
              continue;
            }
            // Affinity: >0 attracts toward a comfortable distance, <0 repels further.
            const aff = this.affinity(agents[i].agent_id, agents[j].agent_id);
            if (aff !== 0) {
              const pull = aff * (d - minSep) * 0.16; // + closes gap, − opens it
              fx[i] -= dx * pull; fy[i] -= dy * pull;
              fx[j] += dx * pull; fy[j] += dy * pull;
            }
          }
          // Drawn to the thing he is about to act on as strongly as to a person he deals with;
          // overrides the main-character pull below.
          const at = anchor.get(agents[i].agent_id);
          if (at) {
            fx[i] += (at.gx - room.gx - px[i]) * ENGAGE_PULL;
            fy[i] += (at.gy - room.gy - py[i]) * ENGAGE_PULL;
            continue;
          }
          // Main character eases toward the room centre → the focal anchor.
          if (agents[i].is_main_character) {
            fx[i] -= px[i] * 0.05; fy[i] -= py[i] * 0.05;
          }
        }
        for (let i = 0; i < n; i++) {
          px[i] = Math.max(-halfW, Math.min(halfW, px[i] + fx[i]));
          py[i] = Math.max(-halfH, Math.min(halfH, py[i] + fy[i]));
        }
      }
      this.assignSlots(
        room,
        agents.map((a) => a.agent_id),
        px.map((x, i) => ({ gx: room.gx + x, gy: room.gy + py[i] })),
        engaged,
        agents.map((ag) => anchor.get(ag.agent_id)),
      );
    }
    return new Map([...byRoom].map(([roomId, list]) => [roomId, list.length]));
  }

  /**
   * Put each figure on the standable cell nearest where the layout wanted it.
   *
   * The force sim solves in continuous tile space, where a fractional position can land on a
   * roof; this quantises it onto open ground, one figure per cell, greedy nearest-first in the
   * caller's stable (id) order so the arrangement is identical step after step.
   */
  private assignSlots(
    room: Room, ids: string[], want: { gx: number; gy: number }[], engaged: Set<string>,
    anchorAt: ({ gx: number; gy: number } | undefined)[] = [],
  ): void {
    const cells = room.stand;
    if (!cells.length) {
      // A room with no ground within reach at all: fall back to its own anchor.
      for (const id of ids) this.slotOf.set(id, { x: room.sx, y: room.sy + 8 });
      return;
    }
    const taken = new Set<number>(this.fixtureCells.get(room.id));
    const place = (i: number, c: number) => {
      taken.add(c);
      this.slotOf.set(ids[i], this.host.map.iso(cells[c].x + 0.5, cells[c].y + 0.5));
    };
    const cost = (c: number, i: number) =>
      (cells[c].x + 0.5 - want[i].gx) ** 2 + (cells[c].y + 0.5 - want[i].gy) ** 2;
    const nearestFree = (i: number): number => {
      let best = -1;
      let bestD = Infinity;
      for (let c = 0; c < cells.length; c++) {
        if (taken.has(c)) continue;
        const d = cost(c, i);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    };

    // Engaged figures are seated first, as groups: a seed, then breadth-first everyone beside
    // someone he deals with, so a man three people are dealing with has all three at his side.
    // Seated a pair at a time, the second visitor would find the pair done and stand wherever
    // the sim left him, whatever ENGAGE_PULL is.
    //
    // Overlap rule: when the room can't seat someone beside his partner, he shares the partner's
    // cell rather than being separated. Closeness is the invariant; one figure per cell only a
    // tie-breaker.
    const seated = new Set<number>();
    const engagedWith = (i: number, k: number) => engaged.has(pairKey(ids[i], ids[k]));
    const grow = (frontier: number[]) => {
      for (let f = 0; f < frontier.length; f++) {
        const at = this.slotOf.get(ids[frontier[f]])!;
        for (let k = 0; k < ids.length; k++) {
          if (seated.has(k) || !engagedWith(frontier[f], k)) continue;
          const c = this.nearestCellBeside(cells, taken, at, (x) => cost(x, k));
          seated.add(k);
          const p = c < 0 ? null : this.host.map.iso(cells[c].x + 0.5, cells[c].y + 0.5);
          if (p && Phaser.Math.Distance.BetweenPoints(p, at) <= this.besideReach()) place(k, c);
          else this.slotOf.set(ids[k], at);
          frontier.push(k);
        }
      }
    };
    for (let i = 0; i < ids.length; i++) {
      if (seated.has(i)) continue;
      const j = ids.findIndex((_, k) => k !== i && !seated.has(k) && engagedWith(i, k));
      if (j < 0) continue;
      // A member at work on a thing seeds the group: the thing can't step aside, the others can.
      const group = [i];
      for (let g = 0; g < group.length; g++) {
        for (let k = 0; k < ids.length; k++) {
          if (!seated.has(k) && !group.includes(k) && engagedWith(group[g], k)) group.push(k);
        }
      }
      const m = group.find((k) => anchorAt[k]);
      if (m !== undefined) {
        const at = anchorAt[m]!;
        const c = this.nearestCellBeside(cells, taken, this.host.map.iso(at.gx, at.gy), (x) => cost(x, m));
        if (c < 0) continue; // nothing free at all — the overflow rule below
        seated.add(m);
        place(m, c);
        grow([m]);
        continue;
      }
      // Otherwise the first pair is one decision over two adjacent cells, so the seed itself
      // can't be stranded somewhere with no free cell beside it.
      const best = this.bestAdjacentPair(cells, taken, (c) => cost(c, i), (c) => cost(c, j));
      if (!best) continue;
      seated.add(i); seated.add(j);
      place(i, best.a);
      if (best.b === best.a) this.slotOf.set(ids[j], this.host.map.iso(cells[best.a].x + 0.5, cells[best.a].y + 0.5));
      else place(j, best.b);
      grow([i, j]);
    }

    // Anchored figures next: the same guarantee for a man acting on a thing, several of them
    // sharing its ring. After the groups (a person outranks a thing) and before the rest.
    for (let i = 0; i < ids.length; i++) {
      const at = anchorAt[i];
      if (seated.has(i) || !at) continue;
      const c = this.nearestCellBeside(cells, taken, this.host.map.iso(at.gx, at.gy), (x) => cost(x, i));
      if (c < 0) continue;
      seated.add(i);
      place(i, c);
    }

    // Everyone else, nearest-free in the caller's stable order. More figures than cells is not
    // an error — a gate room has seven — so the overflow wraps and they stand on each other
    // rather than being flung across the map.
    ids.forEach((_, i) => {
      if (seated.has(i)) return;
      const c = nearestFree(i);
      place(i, c < 0 ? i % cells.length : c);
    });
  }

  /** How far apart on screen two cells may be and still read as side by side. */
  private besideReach(): number {
    return Math.hypot(this.host.map.tileW / 2, this.host.map.tileH / 2) * 1.05;
  }

  /**
   * The free cell that best suits a wanted position while still standing BESIDE a fixed point
   * — for a figure who must end up within reach of the thing he is acting on.
   *
   * Adjacency is the screen-space test bestAdjacentPair uses; the thing's own cell is already
   * taken by layoutFixtures, so this picks from the ring around it. Falls back to the nearest
   * free cell anywhere when that ring is full rather than standing on the object's marker.
   */
  private nearestCellBeside(
    cells: { x: number; y: number }[], taken: Set<number>,
    at: { x: number; y: number }, cost: (c: number) => number,
  ): number {
    const maxPx = this.besideReach();
    let beside = -1;
    let besideCost = Infinity;
    let any = -1;
    let anyCost = Infinity;
    for (let c = 0; c < cells.length; c++) {
      if (taken.has(c)) continue;
      const p = this.host.map.iso(cells[c].x + 0.5, cells[c].y + 0.5);
      const k = cost(c);
      if (k < anyCost) { anyCost = k; any = c; }
      if (Phaser.Math.Distance.BetweenPoints(p, at) <= maxPx && k < besideCost) {
        besideCost = k;
        beside = c;
      }
    }
    return beside >= 0 ? beside : any;
  }

  /**
   * The pair of free cells, adjacent to one another, that best suits two wanted positions.
   *
   * Adjacency is measured on screen, since that is what must read as together: on a 128×64 iso
   * grid ±1 in gx or gy is 71.6px, the (+1,+1) diagonal 64px, the (+1,−1) diagonal 128px, which a
   * Chebyshev "within one cell" test would admit. The threshold derives from the map's tile size.
   *
   * Returns `{a, b}` with `b === a` when no adjacent free pair exists: the caller then seats
   * both on that one cell (see the overlap rule). Null only when nothing is free at all.
   */
  private bestAdjacentPair(
    cells: { x: number; y: number }[], taken: Set<number>,
    costA: (c: number) => number, costB: (c: number) => number,
  ): { a: number; b: number } | null {
    const maxPx = this.besideReach();
    const free: number[] = [];
    for (let c = 0; c < cells.length; c++) if (!taken.has(c)) free.push(c);
    if (!free.length) return null;
    let best: { a: number; b: number } | null = null;
    let bestCost = Infinity;
    for (const a of free) {
      const pa = this.host.map.iso(cells[a].x + 0.5, cells[a].y + 0.5);
      for (const b of free) {
        if (b === a) continue;
        const pb = this.host.map.iso(cells[b].x + 0.5, cells[b].y + 0.5);
        if (Phaser.Math.Distance.BetweenPoints(pa, pb) > maxPx) continue;
        const total = costA(a) + costB(b);
        if (total < bestCost) { bestCost = total; best = { a, b }; }
      }
    }
    if (best) return best;
    // Nothing adjacent is free → share the single cell that best suits the two of them.
    let solo = free[0];
    let soloCost = Infinity;
    for (const c of free) {
      const t = costA(c) + costB(c);
      if (t < soloCost) { soloCost = t; solo = c; }
    }
    return { a: solo, b: solo };
  }

  /**
   * Give every thing standing in a room a cell of its own, from the far end of that
   * room's ground.
   *
   *  1. Fixtures take the edge, the cast keeps the middle: a thing doesn't step aside.
   *  2. A thing's cell depends on the room and its id, not on who is present, so a chest
   *     doesn't shuffle every time someone walks in.
   *
   * Held and destroyed things have no marker and get no cell. Called with the map as drawn,
   * not as the step reported it, so a withheld change is laid out in the visible state.
   */
  layoutFixtures(entities: Record<string, EntityView>): void {
    this.entities.entitySlotOf.clear();
    this.fixtureCells.clear();
    const byLoc = new Map<string, string[]>();
    for (const [eid, e] of Object.entries(entities)) {
      if (e.presence === "held" || e.presence === "destroyed") continue;
      const locId = e.presence_ref ?? "";
      if (!this.host.map.rooms.has(locId)) continue;
      (byLoc.get(locId) ?? byLoc.set(locId, []).get(locId)!).push(eid);
    }
    for (const [locId, eids] of byLoc) {
      const room = this.host.map.rooms.get(locId)!;
      if (!room.stand.length) continue;
      // Fill inward from the last cell still inside the room; the near half is kept for the cast.
      const edge = Math.max(1, room.inRect || room.stand.length);
      const band = Math.max(1, edge - Math.ceil(edge / 2));
      const mine = this.fixtureCells.get(locId) ?? new Set<number>();
      [...eids].sort().forEach((eid, i) => {
        const at = edge - 1 - (i % band);
        const cell = room.stand[at];
        mine.add(at);
        const p = this.host.map.iso(cell.x + 0.5, cell.y + 0.5);
        this.entities.entitySlotOf.set(eid, { x: p.x, y: p.y, gx: cell.x, gy: cell.y });
      });
      this.fixtureCells.set(locId, mine);
    }
  }

  // Resting slot only (from the cached layoutRooms result); in-transit agents are
  // driven by Walker.walkTransit (path-follow).
  positionFor(
    state: AgentStateSummary,
    _all: AgentStateSummary[],
  ): { x: number; y: number } | null {
    const slot = this.slotOf.get(state.agent_id);
    if (slot) return slot;
    const room = this.host.map.rooms.get(state.location_id);
    if (!room) return null;
    const cell = room.stand[0];
    return cell ? this.host.map.iso(cell.x + 0.5, cell.y + 0.5) : { x: room.sx, y: room.sy + 8 };
  }

  // Keep each nameplate hugging its OWN figure from above (attached, not a detached
  // roster). Co-located figures can crowd, so de-collide only vertically: sort by
  // screen-x and lift alternating plates a row higher, so horizontal neighbours don't
  // overprint.
  layoutLabels(states: AgentStateSummary[]): void {
    for (const s of states) {
      const tok = this.host.token(s.agent_id);
      if (tok) {
        tok.plate.setPosition(0, tok.headY - PLATE_GAP); // just above THIS figure's head
        tok.floor = 0;
      }
    }
    const byRoom = new Map<string, AgentToken[]>();
    for (const s of states) {
      if (s.transit) continue;
      const roomId = s.location_id;
      const tok = this.host.token(s.agent_id);
      if (!this.host.map.rooms.has(roomId) || !tok) continue;
      (byRoom.get(roomId) ?? byRoom.set(roomId, []).get(roomId)!).push(tok);
    }
    for (const toks of byRoom.values()) {
      if (toks.length <= 1) continue;
      toks.sort((a, b) => a.container.x - b.container.x);
      toks.forEach((tok, i) => tok.plate.setPosition(0, tok.headY - PLATE_GAP - (i % 2) * 13));
    }
  }
}
