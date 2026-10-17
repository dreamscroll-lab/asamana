import type { AgentToken } from "./types";
import type { WalkHost, Walker } from "./walking";
import type { Viewer } from "../lib/viewer";

const PRESENCE_WALK_MS = 2000;
const PRESENCE_FADE_IN_MS = 600;
const PRESENCE_FADE_OUT_MS = 600;

export interface PresenceView {
  viewer: Viewer;
  location: string;
}

export function continuesPresence(previous: PresenceView | null, current: PresenceView, cut: boolean): boolean {
  return !cut && current.viewer !== null && !!current.location
    && previous?.viewer === current.viewer && previous.location === current.location;
}

/** Departing bodies keep drawing, but leave the scene's logical token registry immediately. */
export class Presence {
  readonly departing = new Map<string, AgentToken>();
  readonly entering = new Set<AgentToken>();

  constructor(private host: WalkHost, private walker: Walker) {}

  async enter(tok: AgentToken, at: { x: number; y: number }): Promise<void> {
    this.entering.add(tok);
    const route = this.walker.presenceRoute(tok.locationId, at)?.reverse();
    const start = route?.[0] ?? at;
    tok.container.setPosition(start.x, start.y).setAlpha(0);
    tok.px = NaN;
    tok.py = NaN;
    tok.presenceAlpha = 0;
    try {
      await Promise.all([
        this.host.tweenP({ targets: tok, presenceAlpha: 1, duration: PRESENCE_FADE_IN_MS }),
        route ? this.walker.walkAlong(tok, route, PRESENCE_WALK_MS) : Promise.resolve(),
      ]);
    } finally {
      tok.presenceAlpha = 1;
      this.entering.delete(tok);
    }
  }

  async leave(id: string, tok: AgentToken, motion: "walk" | "fade" = "walk"): Promise<void> {
    this.departing.set(id, tok);
    tok.container.disableInteractive();
    const route = motion === "walk" ? this.walker.presenceRoute(tok.locationId, tok.container) : null;
    try {
      await Promise.all([
        this.host.tweenP({
          targets: tok, presenceAlpha: 0, duration: PRESENCE_FADE_OUT_MS,
          delay: route ? PRESENCE_WALK_MS - PRESENCE_FADE_OUT_MS : 0,
        }),
        route ? this.walker.walkAlong(tok, route, PRESENCE_WALK_MS) : Promise.resolve(),
      ]);
    } finally {
      this.host.killTweens(tok);
      this.host.killTweens(tok.container);
      tok.container.destroy();
      this.departing.delete(id);
    }
  }
}
