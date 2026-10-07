// Live world feed over WebSocket. Reconnects on drop. The backend pushes
// status / snapshot / step frames (see interaction/api/observe.py).

import type { WsFrame } from "../types";
import { API_BASE } from "./client";

// Same origin as the page by default; if VITE_API_BASE points elsewhere, use its
// host and map http(s) → ws(s).
function wsBase(): string {
  if (API_BASE) return API_BASE.replace(/^http/, "ws");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}`;
}

const STABLE_MS = 5000;

export class WorldSocket {
  private ws: WebSocket | null = null;
  private closed = false;
  private retry = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private openedAt: number | null = null;

  constructor(
    private worldId: string,
    private onFrame: (frame: WsFrame) => void,
    private onOpen?: () => void,
    private onClose?: () => void,
  ) {}

  connect(): void {
    this.ws = new WebSocket(`${wsBase()}/ws/${this.worldId}`);
    this.ws.onopen = () => {
      this.openedAt = Date.now();
      this.onOpen?.();
    };
    this.ws.onmessage = (ev) => {
      try {
        this.onFrame(JSON.parse(ev.data) as WsFrame);
      } catch {
        /* ignore malformed frame */
      }
    };
    this.ws.onclose = () => {
      this.onClose?.();
      if (this.closed) return;
      // Back off from scratch only after a connection that held. Resetting on open would retry
      // every second forever against a server that accepts and then closes (a deleted world).
      if (this.openedAt !== null && Date.now() - this.openedAt >= STABLE_MS) this.retry = 0;
      this.openedAt = null;
      // Exponential-ish backoff, capped at 5s.
      this.retry = Math.min(this.retry + 1, 5);
      // Re-check `closed` when the timer fires: close() can land after scheduling, and the
      // reconnect would open a socket nobody closes while the server keeps serializing
      // every step to it.
      this.retryTimer = setTimeout(() => {
        this.retryTimer = null;
        if (!this.closed) this.connect();
      }, this.retry * 1000);
    };
    this.ws.onerror = () => this.ws?.close();
  }

  close(): void {
    this.closed = true;
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.ws?.close();
  }
}
