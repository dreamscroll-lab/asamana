import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SetupView, WorldMeta } from "../types";

const state = vi.hoisted(() => ({
  cursor: 0, values: new Map<number, unknown>(), effects: [] as (() => unknown)[],
}));
const global = vi.hoisted(() => ({
  cast: vi.fn(() => ({ cast: {}, status: "ready", error: null })),
  places: vi.fn(() => ({ places: {}, status: "ready", error: null })),
  npcs: vi.fn(() => ({ npcs: {}, status: "ready", error: null })),
  reloadCast: vi.fn(), reloadPlaces: vi.fn(), reloadNpcs: vi.fn(),
}));
const controls = vi.hoisted(() => ({
  navigate: vi.fn(), confirm: undefined as (() => Promise<void>) | undefined, confirmDisabled: false,
}));
vi.mock("react/jsx-dev-runtime", async (original) => {
  const runtime = await original<typeof import("react/jsx-dev-runtime")>();
  const jsxDEV: typeof runtime.jsxDEV = (...args) => {
    const [type, props] = args;
    if (type === "button" && props && typeof props === "object" && "children" in props
      && props.children === "确认并进入叙事 →" && "onClick" in props) {
      controls.confirm = props.onClick as () => Promise<void>;
      controls.confirmDisabled = "disabled" in props && Boolean(props.disabled);
    }
    return runtime.jsxDEV(...args);
  };
  return { ...runtime, jsxDEV };
});
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useState: (initial: unknown) => {
    const slot = state.cursor++;
    return [state.values.has(slot) ? state.values.get(slot) : initial, vi.fn()];
  },
  useEffect: (effect: () => unknown) => { state.effects.push(effect); },
}));
vi.mock("react-router-dom", () => ({ useParams: () => ({ id: "world" }), useNavigate: () => controls.navigate }));
vi.mock("../lib/cast", () => ({ useCast: global.cast, loadCast: vi.fn(), reloadCast: global.reloadCast }));
vi.mock("../lib/places", async (original) => ({
  ...await original<typeof import("../lib/places")>(),
  usePlaces: global.places, loadPlaces: vi.fn(), reloadPlaces: global.reloadPlaces,
}));
vi.mock("../lib/npcs", () => ({ useNpcs: global.npcs, loadNpcs: vi.fn(), reloadNpcs: global.reloadNpcs }));
vi.mock("../lib/worldsContext", () => ({ useWorlds: () => ({ worlds: [], refresh: vi.fn() }) }));
vi.mock("../lib/vocabulary", () => ({ useVocabulary: () => ({ emotions: [], needs: [] }) }));
vi.mock("../components/RelationshipGraph", () => ({ default: () => null }));
vi.mock("../components/EditableWorldName", () => ({ default: () => null }));
vi.mock("../api/client", () => ({ API_BASE: "", api: {
  getWorld: vi.fn(), agents: vi.fn(), places: vi.fn(), getSetup: vi.fn(), agentProfile: vi.fn(), confirmWorld: vi.fn(),
} }));

import { api } from "../api/client";
import ReviewView from "./ReviewView";
import WorldView from "./WorldView";

const meta: WorldMeta = {
  world_id: "world", theme: "草稿", world_name: "草稿世界", description: "", status: "unknown",
  created_at: null, current_step: 0, main_agent_names: [], seconds_per_step: 3600,
  run_state: null, confirmed: false, actor_character_id: null, actor_exited: false,
  actor_pace: "every_step", director_reviewed: false,
};
const setup: SetupView = {
  entities: [], npcs: [],
  seconds_per_step: 3600, historical_events: [], relations: [],
  characters: [{
    agent_id: "a", name: "甲", role: "新身份", age: 30, gender: "女", background: "新背景",
    appearance: "", color: "#123456", core_traits: [], core_values: [], self_image: "",
    life_goal: "", secret: "保存响应里的秘密", is_main_character: true, initial_location: "hall",
    initial_emotion: { emotion: "neutral", intensity: 0.2, valence: 0 },
    long_term_goals: [], hard_constraints: [], needs: [],
  }],
};

beforeEach(() => {
  vi.clearAllMocks();
  state.cursor = 0;
  state.values.clear();
  state.effects = [];
  controls.confirm = undefined;
  vi.mocked(api.getWorld).mockResolvedValue(meta);
  vi.mocked(api.agents).mockResolvedValue([]);
  vi.mocked(api.places).mockResolvedValue([]);
  vi.mocked(api.getSetup).mockResolvedValue(setup);
});
const expectNoGlobalIdentityReads = () => {
  expect(global.cast).not.toHaveBeenCalled();
  expect(global.places).not.toHaveBeenCalled();
  expect(global.npcs).not.toHaveBeenCalled();
};

describe("world identity phase boundary", () => {
  it("keeps observation identity hooks unmounted until metadata is confirmed", () => {
    renderToStaticMarkup(<WorldView />);
    expectNoGlobalIdentityReads();
    state.cursor = 0;
    state.values.set(0, meta);
    renderToStaticMarkup(<WorldView />);
    expectNoGlobalIdentityReads();
  });

  it("loads draft public resources directly without requesting private setup or profiles", async () => {
    state.values.set(0, meta);
    renderToStaticMarkup(<ReviewView />);
    state.effects.forEach((effect) => effect());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(api.agents).toHaveBeenCalledWith("world");
    expect(api.places).toHaveBeenCalledWith("world");
    expect(api.getSetup).not.toHaveBeenCalled();
    expect(api.agentProfile).not.toHaveBeenCalled();
    expectNoGlobalIdentityReads();
  });

  it("renders director identities and dossiers from the complete setup response", () => {
    state.values.set(0, { ...meta, director_reviewed: true });
    state.values.set(5, setup);
    const html = renderToStaticMarkup(<ReviewView />);
    expect(html).toContain("新身份");
    expect(html).toContain("保存响应里的秘密");
    expect(api.agentProfile).not.toHaveBeenCalled();
    expectNoGlobalIdentityReads();
  });

  it("blocks world confirmation while an entity or NPC draft is being edited", () => {
    state.values.set(0, { ...meta, director_reviewed: true });
    state.values.set(5, setup);
    state.values.set(12, true);
    renderToStaticMarkup(<ReviewView />);
    expect(controls.confirmDisabled).toBe(true);
    state.cursor = 0;
    state.values.set(12, false);
    renderToStaticMarkup(<ReviewView />);
    expect(controls.confirmDisabled).toBe(false);
  });

  it("uses global identities for confirmed director review", () => {
    state.values.set(0, { ...meta, confirmed: true });
    renderToStaticMarkup(<ReviewView />);
    expect(global.cast).toHaveBeenCalledWith("world");
    expect(global.places).toHaveBeenCalledWith("world");
  });

  it("does not mount director review while an actor performs", () => {
    state.values.set(0, { ...meta, confirmed: true, actor_character_id: "a" });
    renderToStaticMarkup(<ReviewView />);
    expect(api.getSetup).not.toHaveBeenCalled();
    expectNoGlobalIdentityReads();
  });

  it("waits for confirmation and all three identity refreshes before entering observation", async () => {
    function deferred<T>() {
      let resolve!: (value: T) => void;
      let reject!: (failure: Error) => void;
      const promise = new Promise<T>((accept, fail) => { resolve = accept; reject = fail; });
      return { promise, resolve, reject };
    }
    const confirmed = deferred<WorldMeta>();
    const cast = deferred<Record<string, never>>();
    const places = deferred<Record<string, never>>();
    const npcs = deferred<Record<string, never>>();
    vi.mocked(api.confirmWorld).mockReturnValueOnce(confirmed.promise);
    global.reloadCast.mockReturnValueOnce(cast.promise);
    global.reloadPlaces.mockReturnValueOnce(places.promise);
    global.reloadNpcs.mockReturnValueOnce(npcs.promise);
    state.values.set(0, { ...meta, director_reviewed: true });
    renderToStaticMarkup(<ReviewView />);
    expect(controls.confirm).toBeTypeOf("function");

    const confirmation = controls.confirm!();
    expect(api.confirmWorld).toHaveBeenCalledWith("world", undefined);
    expect(global.reloadCast).not.toHaveBeenCalled();
    expect(global.reloadPlaces).not.toHaveBeenCalled();
    expect(global.reloadNpcs).not.toHaveBeenCalled();
    expect(controls.navigate).not.toHaveBeenCalled();

    confirmed.resolve({ ...meta, confirmed: true });
    await Promise.resolve();
    expect(global.reloadCast).toHaveBeenCalledWith("world");
    expect(global.reloadPlaces).toHaveBeenCalledWith("world");
    expect(global.reloadNpcs).toHaveBeenCalledWith("world");
    expect(controls.navigate).not.toHaveBeenCalled();

    cast.resolve({});
    await Promise.resolve();
    expect(controls.navigate).not.toHaveBeenCalled();
    npcs.reject(new Error("Offline"));
    await Promise.resolve();
    expect(controls.navigate).not.toHaveBeenCalled();

    places.resolve({});
    await confirmation;
    expect(controls.navigate).toHaveBeenCalledExactlyOnceWith("/worlds/world");
  });
});
