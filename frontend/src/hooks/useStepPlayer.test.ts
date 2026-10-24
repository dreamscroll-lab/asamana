import { beforeEach, describe, expect, it, vi } from "vitest";

const hooks = vi.hoisted(() => ({
  cursor: 0,
  values: new Map<number, unknown>(),
  effects: new Map<number, { deps: unknown[]; cleanup?: () => void }>(),
  pending: [] as (() => void)[],
  dirty: false,
}));

vi.mock("react", () => ({
  useState: (initial: unknown) => {
    const slot = hooks.cursor++;
    if (!hooks.values.has(slot)) hooks.values.set(slot, initial);
    return [hooks.values.get(slot), (update: unknown) => {
      const previous = hooks.values.get(slot);
      const next = typeof update === "function" ? update(previous) : update;
      if (!Object.is(previous, next)) {
        hooks.values.set(slot, next);
        hooks.dirty = true;
      }
    }];
  },
  useRef: (initial: unknown) => {
    const slot = hooks.cursor++;
    if (!hooks.values.has(slot)) hooks.values.set(slot, { current: initial });
    return hooks.values.get(slot);
  },
  useMemo: (compute: () => unknown) => compute(),
  useEffect: (effect: () => (() => void) | undefined, deps: unknown[]) => {
    const slot = hooks.cursor++;
    const previous = hooks.effects.get(slot);
    if (previous && deps.every((value, index) => Object.is(value, previous.deps[index]))) return;
    hooks.pending.push(() => {
      previous?.cleanup?.();
      hooks.effects.set(slot, { deps, cleanup: effect() });
    });
  },
}));

vi.mock("../api/client", () => ({ api: { listSteps: vi.fn(), getStep: vi.fn(), graph: vi.fn() } }));

import { api } from "../api/client";
import type { StepView } from "../types";
import { useStepPlayer, type StepPlayerControls } from "./useStepPlayer";

const step = (number: number) => ({ step: number } as StepView);

let mode: "live" | "playback";
let controls: StepPlayerControls;

function render(): void {
  let attempts = 0;
  do {
    if (++attempts > 30) throw new Error("Hook did not settle");
    hooks.dirty = false;
    hooks.cursor = 0;
    controls = useStepPlayer("world", mode, false);
    const effects = hooks.pending.splice(0);
    effects.forEach((effect) => effect());
  } while (hooks.dirty);
}

async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  render();
}

beforeEach(() => {
  hooks.cursor = 0;
  hooks.values.clear();
  hooks.effects.clear();
  hooks.pending = [];
  hooks.dirty = false;
  vi.resetAllMocks();
  mode = "live";
});

describe("step playback consistency", () => {
  it("keeps playback advancing after seeking back while another step is loading", async () => {
    mode = "playback";
    vi.mocked(api.listSteps).mockResolvedValue([1, 2, 3]);
    const waiting = new Promise<StepView>(() => {});
    vi.mocked(api.getStep).mockImplementation(async (_id, number) => number === 2 ? waiting : step(number));
    render();
    await flush();
    await flush();
    controls.goTo(0);
    render();
    await flush();
    expect(controls.displayStep?.step).toBe(1);
    controls.doneRendering();
    render();

    controls.goTo(1);
    render();
    expect(controls.displayStep?.step).toBe(1);
    controls.goTo(0);
    render();
    controls.togglePlay();
    render();

    expect(controls.playing).toBe(true);
    expect(controls.pbIndex).toBe(1);
  });

  it("tells the story as steps arrive and releases consumed snapshots", () => {
    render();
    controls.onLiveStep(step(1));
    render();
    controls.onLiveStep(step(2));
    controls.onLiveStep(step(3));
    render();
    expect(controls.displayStep?.step).toBe(1);
    // The feed shows how far the world has really run; the map catches up on its own.
    expect(controls.feed.map((entry) => entry.step)).toEqual([3, 2, 1]);

    controls.doneRendering();
    render();
    expect(controls.displayStep?.step).toBe(2);
    controls.doneRendering();
    render();
    expect(controls.displayStep?.step).toBe(3);
    // The live queue owns only the current snapshot after every queued step has been consumed.
    expect(hooks.values.get(0)).toEqual([controls.displayStep]);
    controls.onLiveStep(step(1));
    render();
    expect(hooks.values.get(0)).toEqual([controls.displayStep]);
    expect(controls.feed.map((entry) => entry.step)).toEqual([3, 2, 1]);
  });
});
