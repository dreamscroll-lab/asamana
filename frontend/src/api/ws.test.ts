import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WorldSocket } from "./ws";

/** A fake socket with just enough for WorldSocket: it records whether it was closed. */
class FakeSocket {
  static live: FakeSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeSocket.live.push(this);
  }

  close(): void {
    this.closed = true;
    this.onclose?.();
  }
}

beforeEach(() => {
  FakeSocket.live = [];
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", FakeSocket);
  vi.stubGlobal("location", { protocol: "http:", host: "localhost:8080" });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("WorldSocket reconnect", () => {
  it("leaves no socket behind when the page closes during the retry wait", () => {
    // Disconnect → reconnect scheduled → user navigates away within those 1–5 s.
    // The scheduled reconnect must not open a socket nobody would ever close.
    const socket = new WorldSocket("w1", () => {});
    socket.connect();
    FakeSocket.live[0].onclose?.();   // backend restart / network blip

    socket.close();
    vi.advanceTimersByTime(10_000);

    expect(FakeSocket.live).toHaveLength(1);
    expect(FakeSocket.live[0].closed).toBe(true);
  });

  it("still reconnects when the page is still open", () => {
    const socket = new WorldSocket("w1", () => {});
    socket.connect();
    FakeSocket.live[0].onclose?.();

    vi.advanceTimersByTime(10_000);

    expect(FakeSocket.live).toHaveLength(2);   // reconnect happens as usual
  });

  it("backs off against a server that accepts and then closes at once", () => {
    const socket = new WorldSocket("w1", () => {});
    socket.connect();
    for (let i = 0; i < 4; i++) {
      const ws = FakeSocket.live.at(-1)!;
      ws.onopen?.();
      ws.onclose?.();
      vi.advanceTimersByTime(10_000);
    }
    // Waits grow 1, 2, 3, 4 s instead of restarting at 1 s on every accept.
    const before = FakeSocket.live.length;
    FakeSocket.live.at(-1)!.onopen?.();
    FakeSocket.live.at(-1)!.onclose?.();
    vi.advanceTimersByTime(4_999);
    expect(FakeSocket.live).toHaveLength(before);
    socket.close();
  });

  it("starts the backoff over after a connection that held", () => {
    const socket = new WorldSocket("w1", () => {});
    socket.connect();
    FakeSocket.live[0].onclose?.();
    vi.advanceTimersByTime(10_000);
    const ws = FakeSocket.live.at(-1)!;
    ws.onopen?.();
    vi.advanceTimersByTime(60_000);
    ws.onclose?.();
    vi.advanceTimersByTime(1_000);
    expect(FakeSocket.live).toHaveLength(3);
    socket.close();
  });
});
