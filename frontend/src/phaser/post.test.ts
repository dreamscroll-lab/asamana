import { describe, expect, it, vi } from "vitest";

import type { MessageSummary } from "../types";
import type { Scoped } from "../lib/step";
import type { PostHost } from "./post";
import type { AgentToken } from "./types";

vi.mock("phaser", () => ({ default: {} }));
vi.mock("./effects", () => ({ arrivingMessage: vi.fn().mockResolvedValue(undefined) }));
Object.assign(globalThis, { window: { devicePixelRatio: 2 } });
const { PostLayer } = await import("./post");
const { arrivingMessage } = await import("./effects");

interface PostRendering {
  renderPostBadge(tok: AgentToken): void;
  openPost(tok: AgentToken): Promise<void>;
}

describe("offscreen post identity", () => {
  it("does not let the letter badge bypass a reader's interaction gate", () => {
    const badge = { setSize: vi.fn().mockReturnThis(), setInteractive: vi.fn().mockReturnThis(), on: vi.fn().mockReturnThis() };
    const label = { width: 30, height: 14, setOrigin: vi.fn().mockReturnThis(), setResolution: vi.fn().mockReturnThis() };
    const graphics = {
      fillStyle: vi.fn(), fillRoundedRect: vi.fn(), lineStyle: vi.fn(), strokeRoundedRect: vi.fn(),
    };
    const selectable = vi.fn(() => false);
    const host = {
      canSelect: selectable,
      add: { text: () => label, graphics: () => graphics, container: () => badge },
    } as unknown as PostHost;
    const tok = {
      container: { add: vi.fn(), getData: () => "reader" },
      color: 0x123456, headY: -40, postData: [{ from: "Sender", ink: 0x123456, said: "A letter" }],
    } as unknown as AgentToken;
    const layer = new PostLayer(host, {} as never, {} as never, {} as never);
    const rendering = layer as unknown as PostRendering;
    const open = vi.spyOn(rendering, "openPost").mockResolvedValue(undefined);
    rendering.renderPostBadge(tok);
    const click = badge.on.mock.calls[0][1];
    const pointer = { event: { stopPropagation: vi.fn() } };
    const event = { stopPropagation: vi.fn() };
    click(pointer, 0, 0, event);
    expect(open).not.toHaveBeenCalled();
    selectable.mockReturnValue(true);
    click(pointer, 0, 0, event);
    expect(open).toHaveBeenCalledWith(tok);
  });

  it("uses world identity for absent senders and reserves unsigned ink for narrator", async () => {
    const reader = { dead: false, postData: [] } as unknown as AgentToken;
    const host = {
      renderGen: 1,
      token: (id: string) => id === "reader" ? reader : undefined,
      identityColor: vi.fn(() => 0x123456),
      focusActive: () => true,
      isFocused: () => false,
      canSelect: () => true,
    } as unknown as PostHost;
    const heard = { playBroadcast: vi.fn() };
    const layer = new PostLayer(host, {} as never, {} as never, heard as never);
    vi.spyOn(layer as unknown as PostRendering, "renderPostBadge").mockImplementation(() => {});
    const message = (sender_id: string, scope: string): Scoped<MessageSummary> => ({
      sender_id, sender_name: sender_id, receiver_ids: ["reader"], scope, content: "A letter", spoken: "",
    }) as Scoped<MessageSummary>;
    await layer.playDeliveries([
      message("offscreen", "direct"), message("narrator", "direct"), message("offscreen", "world"),
    ], 1);
    expect(reader.postData?.map((letter) => letter.ink)).toEqual([0x123456, 0x8a94a8]);
    expect(heard.playBroadcast).toHaveBeenCalledWith(
      { content: "A letter", location_id: null }, { name: "offscreen", ink: 0x123456 },
    );
    expect(host.identityColor).not.toHaveBeenCalledWith("narrator");
  });

  it("keeps a signed arrival in the sender's identity color without a visible sender token", async () => {
    const reader = { dead: false, postData: [] } as unknown as AgentToken;
    const host = {
      renderGen: 1,
      token: (id: string) => id === "reader" ? reader : undefined,
      identityColor: () => 0x123456,
      focusActive: () => true,
      isFocused: (id: string) => id === "reader",
      canSelect: () => true,
    } as unknown as PostHost;
    const layer = new PostLayer(host, {} as never, {} as never, {} as never);
    vi.spyOn(layer as unknown as PostRendering, "renderPostBadge").mockImplementation(() => {});
    vi.spyOn(layer as unknown as PostRendering, "openPost").mockResolvedValue(undefined);
    await layer.playDeliveries([{
      sender_id: "offscreen", sender_name: "Sender", receiver_ids: ["reader"], scope: "direct",
      content: "A letter", spoken: "",
    } as Scoped<MessageSummary>], 1);
    expect(arrivingMessage).toHaveBeenCalledWith(host, reader, { color: 0x123456 });
  });
});
