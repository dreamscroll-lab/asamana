/**
 * How a figure gets from one place to another on foot: the road network, room-to-room and point-to-point
 * routes through it, and walking a route at a constant speed. There is one way to travel, and it goes
 * through the pathfinder.
 */

import Phaser from "phaser";

import type { AgentStateSummary } from "../types";
import { type FxHost, footDust } from "./effects";
import { astar, computeMainComponent, foldStairs, snap } from "./pathfinding";
import { routeFraction } from "./transitProgress";
import type { AgentToken } from "./types";
import type { WorldMap } from "./worldMap";

// How long a figure takes to cross one screen pixel on foot: a speed, not a duration, so
// distance stays legible on the clock. Paced for a walk: faster reads as hurrying, a claim
// about the character that only the simulation should make.
const WALK_MS_PER_PX = 9.0;

/** The slice of the scene walking needs, on top of the shared FX one. */
export interface WalkHost extends FxHost {
  readonly map: WorldMap;
  /** A superseded render is winding down: no walk may take time. */
  readonly unwinding: boolean;
  tweenP(config: Phaser.Types.Tweens.TweenBuilderConfig): Promise<void>;
}

export class Walker {
  private mainComponent = new Set<number>();
  private pathCache = new Map<string, { x: number; y: number }[] | null>();
  // Per agent: the trip last drawn and how far along it. The scene only walks consecutive steps
  // (a seek places instead, see placeOnTransit), so the same trip means the next step of it.
  private transitProgress = new Map<string, { trip: string; p: number }>();

  /** `placed`: the scene's set of agents already positioned on the map (no fly-in from 0,0). */
  constructor(private host: WalkHost, private placed: Set<string>) {}

  /** Build the road network off the map and put every room's route anchor on it. */
  mount(): void {
    this.mainComponent = computeMainComponent(this.host.map.blocked, this.host.map.cols, this.host.map.rows);
    // Snap each room's A* anchor to the nearest reachable (road-network) tile so
    // transit paths always have a walkable endpoint, and stage its people on ground that
    // same network reaches — see findStanding for why the two must agree.
    for (const room of this.host.map.rooms.values()) {
      const [ax, ay] = snap(Math.floor(room.gx), Math.floor(room.gy), this.host.map.blocked, this.host.map.cols, this.host.map.rows, this.mainComponent);
      room.atx = ax;
      room.aty = ay;
      this.host.map.findStanding(room, this.mainComponent);
    }
  }

  /** He is no longer mid-transit: the next journey starts from where he stands. */
  clearTransit(agentId: string): void {
    this.transitProgress.delete(agentId);
  }

  // Cached wall-avoiding pixel route between two rooms (ends pinned to centres).
  private transitPixelPath(fromId: string, toId: string): { x: number; y: number }[] | null {
    const cacheKey = `${fromId}::${toId}`;
    if (this.pathCache.has(cacheKey)) return this.pathCache.get(cacheKey)!;
    const from = this.host.map.rooms.get(fromId);
    const to = this.host.map.rooms.get(toId);
    let result: { x: number; y: number }[] | null = null;
    if (from && to) {
      const path = astar([from.atx, from.aty], [to.atx, to.aty], this.host.map.blocked, this.host.map.cols, this.host.map.rows, this.host.map.toll);
      if (path && path.length >= 2) {
        // Folded into long axis-aligned runs before projecting: a 4-connected A* route that
        // runs diagonally is a staircase (correct, but it would turn the figure every tile).
        // Folded by elbow, never by diagonal: the cast can only be drawn along the grid axes
        // (see foldStairs). The cell test must also avoid `toll` ground (hidden by the art,
        // which A* only crosses when it must), or folding would put the figure back on a roof.
        const passable = (x: number, y: number): boolean =>
          !this.host.map.blocked[y]?.[x] && (this.host.map.toll[y]?.[x] ?? 0) === 0;
        result = foldStairs(path, passable).map(([tx, ty]) => this.host.map.iso(tx + 0.5, ty + 0.5));
      }
    }
    this.pathCache.set(cacheKey, result);
    return result;
  }

  // Full wall-avoiding pixel route for a transit, following its logical waypoint
  // `path` (through intermediate rooms) by concatenating each leg's A* route, with
  // each leg's pixel length and the step it ends on. Falls back to a direct from→to
  // route (one leg, reached at total_steps) if any leg is unroutable.
  private transitRoutePixels(
    t: NonNullable<AgentStateSummary["transit"]>,
  ): { pts: { x: number; y: number }[]; legLengths: number[]; arrivals: number[] } | null {
    const direct = () => {
      const leg = this.transitPixelPath(t.from_location_id ?? "", t.to_location_id ?? "");
      return leg ? { pts: leg, legLengths: [polylineLength(leg)], arrivals: [0, t.total_steps] } : null;
    };
    if (t.path.length < 2) return direct();
    const pts: { x: number; y: number }[] = [];
    const legLengths: number[] = [];
    for (let i = 0; i < t.path.length - 1; i++) {
      const leg = this.transitPixelPath(t.path[i], t.path[i + 1]);
      if (!leg) return direct();
      legLengths.push(polylineLength(leg));
      // Drop the shared endpoint between consecutive legs so it isn't duplicated.
      pts.push(...(pts.length ? leg.slice(1) : leg));
    }
    return { pts, legLengths, arrivals: t.arrivals };
  }

  // Move a mover ALONG its route for the portion covered this step (prevP→curP),
  // so it follows the road instead of straight-lining across walls. He stands on a
  // waypoint on the step the engine does (see routeFraction). The route runs between room
  // anchors, not where he stands: the first step walks him from his spot onto it, and
  // `standAt` (the step that lands him) walks him off it into his place, all in one walk.
  walkTransit(
    tok: AgentToken,
    agentId: string,
    t: NonNullable<AgentStateSummary["transit"]>,
    standAt: { x: number; y: number } | null = null,
  ): Promise<void> {
    const { path, curP, trip } = this.progressOf(t);
    this.host.killTweens(tok.container);
    const last = this.transitProgress.get(agentId);
    this.transitProgress.set(agentId, { trip, p: curP });
    if (!path) {
      const from = this.host.map.rooms.get(t.from_location_id ?? "");
      const to = this.host.map.rooms.get(t.to_location_id ?? "");
      if (from && to) {
        return this.host.tweenP({
          targets: tok.container,
          x: from.sx + (to.sx - from.sx) * curP,
          y: from.sy + (to.sy - from.sy) * curP,
          duration: 1200, ease: "Sine.easeInOut",
        });
      }
      return Promise.resolve();
    }
    const startingTransit = last?.trip !== trip;
    let pts = this.subPath(path, startingTransit ? 0 : last.p, curP);
    if (!this.placed.has(agentId)) {
      // First sighting: snap onto the route rather than fly in from (0,0).
      tok.container.setPosition(pts[0].x, pts[0].y);
      this.placed.add(agentId);
    } else if (startingTransit) {
      // Don't snap to the route start: he'd jump from his spot in the room to its anchor.
      pts = [...this.stepBetween({ x: tok.container.x, y: tok.container.y }, pts[0]), ...pts.slice(1)];
    }
    if (startingTransit) footDust(this.host, pts[0].x, pts[0].y + 8); // departure puff
    if (curP >= 1 && standAt) pts = [...pts, ...this.stepBetween(pts[pts.length - 1], standAt).slice(1)];
    if (curP >= 1) footDust(this.host, pts[pts.length - 1].x, pts[pts.length - 1].y + 8); // arrival puff
    return this.walkAlong(tok, pts);
  }

  /**
   * Put a mover where this step has him on his trip, without walking: the step drawn before was
   * not the one before this, so no walk between the two ever happened. Records the progress, so
   * the next step continues from here.
   */
  placeOnTransit(
    tok: AgentToken,
    agentId: string,
    t: NonNullable<AgentStateSummary["transit"]>,
  ): void {
    const { path, curP, trip } = this.progressOf(t);
    this.host.killTweens(tok.container);
    this.transitProgress.set(agentId, { trip, p: curP });
    let at: { x: number; y: number } | null = path ? this.subPath(path, curP, curP).at(-1)! : null;
    if (!at) {
      const from = this.host.map.rooms.get(t.from_location_id ?? "");
      const to = this.host.map.rooms.get(t.to_location_id ?? "");
      if (from && to) at = { x: from.sx + (to.sx - from.sx) * curP, y: from.sy + (to.sy - from.sy) * curP };
    }
    if (!at) return;
    tok.container.setPosition(at.x, at.y);
    this.placed.add(agentId);
  }

  private progressOf(t: NonNullable<AgentStateSummary["transit"]>): {
    path: { x: number; y: number }[] | null;
    curP: number;
    trip: string;
  } {
    const route = this.transitRoutePixels(t);
    const curP = route
      ? routeFraction(route.legLengths, route.arrivals, t.elapsed_steps)
      : Math.max(0, Math.min(1, t.elapsed_steps / t.total_steps));
    return { path: route?.pts ?? null, curP, trip: `${t.path.join(">")}/${t.total_steps}` };
  }

  // On foot between a spot in a room and that room's anchor. A straight line when there is no
  // ground route: both ends are in the same room, so it can't cross a wall.
  private stepBetween(from: { x: number; y: number }, to: { x: number; y: number }): { x: number; y: number }[] {
    return this.groundRoute(from, to) ?? [from, to];
  }

  /**
   * Walk a figure along a pixel polyline at constant speed. The one way a token travels:
   * both the multi-step transit above and the scene's plain relocation come through here.
   *
   * Time follows the distance walked, floored so a short hop registers and capped so a march
   * never outstays the step. Each leg gets time in proportion to its length; an even split
   * would make the figure crawl across short legs and bolt down long ones.
   */
  walkAlong(tok: AgentToken, pts: { x: number; y: number }[]): Promise<void> {
    if (this.host.unwinding) return Promise.resolve();
    if (pts.length < 2) {
      return this.host.tweenP({ targets: tok.container, x: pts[0].x, y: pts[0].y, duration: 400, ease: "Linear" });
    }
    const legs = pts.slice(1).map((p, i) => Phaser.Math.Distance.BetweenPoints(pts[i], p));
    const dist = legs.reduce((sum, d) => sum + d, 0);
    const total = Phaser.Math.Clamp(dist * WALK_MS_PER_PX, 1400, 20000);
    return new Promise((resolve) => {
      const settle = this.host.awaitTweens(tok.container, () => {
        this.host.removeCanceler(cancel);
        resolve();
      });
      const cancel = () => this.host.killTweens(tok.container);
      this.host.addCanceler(cancel);
      this.host.tweens.chain({
        targets: tok.container,
        tweens: pts.slice(1).map((p, i) => ({
          x: p.x,
          y: p.y,
          duration: Math.max(1, (legs[i] / (dist || 1)) * total),
          ease: "Linear",
        })),
        onComplete: settle,
        onStop: settle,
      });
    });
  }

  /**
   * A wall-avoiding pixel route between two arbitrary screen points.
   *
   * Like `transitPixelPath`, but between arbitrary points rather than room anchors: a figure
   * who turns up somewhere else this step starts where he stood and ends on a resting slot.
   * The ends are read off the screen and snapped to standable ground.
   *
   * Returns null when either end is off the grid or the search finds nothing; the caller
   * then has no route to walk and places him directly.
   */
  groundRoute(
    from: { x: number; y: number },
    to: { x: number; y: number },
  ): { x: number; y: number }[] | null {
    const a = this.host.map.unproject(from.x, from.y);
    const b = this.host.map.unproject(to.x, to.y);
    const cell = (g: { gx: number; gy: number }): [number, number] =>
      snap(Math.floor(g.gx), Math.floor(g.gy), this.host.map.blocked, this.host.map.cols, this.host.map.rows, this.mainComponent);
    const start = cell(a);
    const goal = cell(b);
    if (start[0] === goal[0] && start[1] === goal[1]) return null;
    const path = astar(start, goal, this.host.map.blocked, this.host.map.cols, this.host.map.rows, this.host.map.toll);
    if (!path || path.length < 2) return null;
    const passable = (x: number, y: number): boolean =>
      !this.host.map.blocked[y]?.[x] && (this.host.map.toll[y]?.[x] ?? 0) === 0;
    const pts = foldStairs(path, passable).map(([tx, ty]) => this.host.map.iso(tx + 0.5, ty + 0.5));
    // Pin the ends to where he actually is and is going; the search runs on whole cells, so
    // otherwise he would jump to a cell centre at both ends of every walk.
    pts[0] = { x: from.x, y: from.y };
    pts[pts.length - 1] = { x: to.x, y: to.y };
    return pts;
  }

  // The polyline covering [p0, p1] of a route (bracketed by interpolated ends).
  private subPath(
    path: { x: number; y: number }[],
    p0: number,
    p1: number,
  ): { x: number; y: number }[] {
    if (path.length < 2) return path.slice();
    const seg = path.slice(1).map((pt, i) => Math.hypot(pt.x - path[i].x, pt.y - path[i].y));
    const total = seg.reduce((s, d) => s + d, 0) || 1;
    const locate = (p: number): { point: { x: number; y: number }; seg: number } => {
      let target = Math.max(0, Math.min(1, p)) * total;
      for (let i = 0; i < seg.length; i++) {
        if (target <= seg[i] || i === seg.length - 1) {
          const a = path[i];
          const b = path[i + 1];
          const f = seg[i] > 0 ? target / seg[i] : 0;
          return { point: { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f }, seg: i };
        }
        target -= seg[i];
      }
      return { point: path[path.length - 1], seg: seg.length - 1 };
    };
    const s0 = locate(p0);
    const s1 = locate(p1);
    const pts = [s0.point];
    for (let i = s0.seg + 1; i <= s1.seg; i++) pts.push(path[i]);
    pts.push(s1.point);
    return pts;
  }
}

function polylineLength(pts: { x: number; y: number }[]): number {
  let d = 0;
  for (let i = 1; i < pts.length; i++) d += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  return d;
}
