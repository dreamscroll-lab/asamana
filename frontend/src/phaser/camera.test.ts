import { describe, expect, it, vi } from "vitest";
vi.mock("phaser", () => ({ default: { Math: {
  Clamp: (value: number, min: number, max: number) => Math.min(max, Math.max(min, value)),
  Distance: { Between: (x1: number, y1: number, x2: number, y2: number) => Math.hypot(x2 - x1, y2 - y1) },
} } }));
import { MapCamera, type CameraHost } from "./camera";

function mounted() {
  const events = new Map<string, (...args: unknown[]) => void>();
  const canvas = { style: {} };
  const cam = {
    zoom: 1, scrollX: 0, scrollY: 0,
    setBackgroundColor: vi.fn(), setBounds: vi.fn(), centerOn: vi.fn(),
    setZoom: vi.fn((zoom: number) => { cam.zoom = zoom; }),
    getWorldPoint: (x: number, y: number) => ({ x, y }),
  };
  const host = {
    map: { worldW: 1000, worldH: 1000, tileW: 32 }, cast: { scale: 1 },
    game: { canvas }, cameras: { main: cam },
    scale: { width: 1000, height: 1000, on: vi.fn() },
    input: {
      on: (name: string, handler: (...args: unknown[]) => void) => events.set(name, handler),
      addPointer: vi.fn(), mouse: { disableContextMenu: vi.fn() },
      pointer1: { isDown: false }, pointer2: { isDown: false },
    },
  };
  const click = vi.fn();
  const context = vi.fn();
  new MapCamera(host as unknown as CameraHost).mount(vi.fn(), click, context);
  cam.setZoom.mockClear();
  const pointer = (target: object, x = 10, y = 10, right = false) => ({
    event: { target }, x, y, button: right ? 2 : 0, isDown: true,
    leftButtonDown: () => !right, rightButtonDown: () => right,
  });
  const fire = (event: string, ...args: unknown[]) => events.get(event)!(...args);
  return { canvas, cam, click, context, pointer, fire };
}

describe("map camera DOM input boundary", () => {
  it("ignores overlay form presses, motion, wheel and releases", () => {
    const h = mounted();
    const form = {};
    h.fire("pointerdown", h.pointer(form));
    h.fire("pointermove", h.pointer(h.canvas, 20));
    h.fire("pointermove", h.pointer(h.canvas, 80));
    h.fire("wheel", h.pointer(form), [], 0, 100);
    h.fire("pointerup", h.pointer(form));
    expect(h.cam.scrollX).toBe(0);
    expect(h.cam.setZoom).not.toHaveBeenCalled();
    expect(h.click).not.toHaveBeenCalled();
  });
  it("retains left-click selection and right-click actions on the canvas", () => {
    const h = mounted();
    h.fire("pointerdown", h.pointer(h.canvas));
    h.fire("pointerup", h.pointer(h.canvas));
    expect(h.click).toHaveBeenCalledOnce();
    h.fire("pointerdown", h.pointer(h.canvas, 10, 10, true));
    h.fire("pointerup", h.pointer(h.canvas, 10, 10, true));
    expect(h.context).toHaveBeenCalledOnce();
    expect(h.click).toHaveBeenCalledOnce();
  });
  it("cleans up a drag released outside and never selects through an overlay", () => {
    const h = mounted();
    h.fire("pointerdown", h.pointer(h.canvas));
    h.fire("pointermove", h.pointer(h.canvas, 20));
    h.fire("pointermove", h.pointer(h.canvas, 40));
    expect(h.cam.scrollX).toBeLessThan(0);
    h.fire("pointerupoutside", h.pointer({}));
    const scroll = h.cam.scrollX;
    h.fire("pointermove", h.pointer(h.canvas, 80));
    expect(h.cam.scrollX).toBe(scroll);
    expect(h.click).not.toHaveBeenCalled();
    h.fire("pointerdown", h.pointer(h.canvas));
    h.fire("pointerup", h.pointer(h.canvas));
    expect(h.click).toHaveBeenCalledOnce();
  });
});
