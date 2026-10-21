import { Children, isValidElement, type ReactElement, type ReactNode, type TextareaHTMLAttributes } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ActorEntry, ActorTurn } from "../../types";

const state = vi.hoisted(() => ({ values: [] as unknown[] }));
const submission = vi.hoisted(() => ({ run: vi.fn(async (fn: () => Promise<string>) => fn()) }));
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useEffect: vi.fn(),
  useMemo: (fn: () => unknown) => fn(),
  useRef: (current: unknown) => ({ current }),
  useState: (initial: unknown) => [state.values.length ? state.values.shift() : initial, vi.fn()],
}));
vi.mock("../../lib/cast", () => ({ useCast: () => ({ cast: {} }) }));
vi.mock("./MindPanel", () => ({ default: () => null }));
vi.mock("./submission", () => ({ useActorSubmission: () => ({ busy: false, message: "", run: submission.run }) }));
import { api } from "../../api/client";
import ActorConsole from "./ActorConsole";
import ConversationWindow from "./ConversationWindow";

const turn: ActorTurn = {
  turn_id: "t", agent_id: "self", step: 1, kind: "decide",
  briefing: { location: null, people: [], npcs: [], places: [], entities: [] },
  doing: "", invitation: null, decided: "", conversation: null, reason: "",
};
const rejected: ActorEntry[] = [
  { kind: "asked", entry_id: "asked", time_label: "", step: 1, turn },
  { kind: "said", entry_id: "said", time_label: "", step: 1, turn_id: "t", words: "唉", decision_summary: null, reason: "unreadable", reading: "", intent: null },
];

type Element = ReactElement<{ children?: ReactNode }>;
function elements(node: ReactNode): Element[] {
  return Children.toArray(node).flatMap((child) => {
    if (!isValidElement(child)) return [];
    const element = child as Element;
    return [element, ...elements(element.props.children)];
  });
}

function consoleTree(entries: ActorEntry[] = []) {
  state.values.push(null, "every_step", "唉", null, null, null);
  return ActorConsole({ worldId: "world", selfId: "self", entries, turn, runState: "running", step: 1, pace: "every_step", immersive: true });
}

beforeEach(() => {
  state.values.length = 0;
  submission.run.mockClear();
  vi.restoreAllMocks();
});

describe("a reading of words awaiting confirmation", () => {
  it("shows no duration for doing nothing", () => {
    const markup = renderToStaticMarkup(consoleTree([
      { kind: "asked", entry_id: "asked", time_label: "", step: 1, turn },
      { kind: "said", entry_id: "said", time_label: "", step: 1, turn_id: "t", words: "先等等看", reason: null, reading: "", intent: null,
        decision_summary: { decision_summary_id: "d", description: "", operation: "idle", target: null,
          estimated_duration_seconds: 0, parameters: {} } },
    ]));
    expect(markup).toContain("⏸ 这次先不动");
    expect(markup).not.toContain("预计耗时");
  });
});

describe("actor intent rejection feedback", () => {
  it("emphasizes the recorded rejection once in the thread", () => {
    const markup = renderToStaticMarkup(consoleTree(rejected));
    expect(markup.split("没读懂这句话，换个说法试试。")).toHaveLength(2);
    expect(markup).toContain('role="alert"');
    expect(markup).toContain("border-rose-400/30");
  });
  it("shows where the reading got stuck under the rejection", () => {
    const [asked, said] = rejected;
    const markup = renderToStaticMarkup(consoleTree([asked, { ...said, reading: "只是一声叹息，没说要做什么" } as ActorEntry]));
    expect(markup).toContain("没读懂这句话，换个说法试试。");
    expect(markup).toContain("只是一声叹息，没说要做什么");
  });
  it.each(["unreadable", "stale_intent"])("only gives unrecorded %s rejections an immediate notice", async (reason) => {
    vi.spyOn(api, "actorIntent").mockResolvedValue({ status: "rejected", reason, decision_summary: null });
    const tree = consoleTree();
    const input = elements(tree).find((element) => element.type === "textarea") as ReactElement<TextareaHTMLAttributes<HTMLTextAreaElement>>;
    input.props.onKeyDown!({ key: "Enter", shiftKey: false, nativeEvent: { isComposing: false }, preventDefault: vi.fn() } as unknown as Parameters<NonNullable<typeof input.props.onKeyDown>>[0]);
    expect(api.actorIntent).toHaveBeenCalledOnce();
    const notice = await submission.run.mock.results[0].value;
    expect(notice).toBe(reason === "unreadable" ? "" : "回合或指令已经变化，请重新提交。");
  });
  it("keeps request failures available to the immediate error handler", async () => {
    vi.spyOn(api, "actorIntent").mockRejectedValue(new Error("Network unavailable"));
    const input = elements(consoleTree()).find((element) => element.type === "textarea") as ReactElement<TextareaHTMLAttributes<HTMLTextAreaElement>>;
    input.props.onKeyDown!({ key: "Enter", shiftKey: false, nativeEvent: { isComposing: false }, preventDefault: vi.fn() } as unknown as Parameters<NonNullable<typeof input.props.onKeyDown>>[0]);
    await expect(submission.run.mock.results[0].value).rejects.toThrow("Network unavailable");
  });
});

describe("conversation close control", () => {
  it.each([false, true])("only offers closing after the conversation ends (%s)", (over) => {
    const onClose = vi.fn();
    const tree = ConversationWindow({
      worldId: "world", selfId: "self", self: undefined, otherId: "other", other: undefined,
      ink: () => "#ffffff", dialogue: [], conversation: null, over, onClose, onAnswer: vi.fn(),
    });
    const controls = elements(tree).filter((element) => element.type === "button" && (element.props as { "aria-label"?: string })["aria-label"] === "关闭");
    expect(controls).toHaveLength(over ? 1 : 0);
    if (over) {
      const header = elements(tree).find((element) => element.type === "header")!;
      expect(renderToStaticMarkup(header)).toContain('aria-label="关闭"');
      (controls[0].props as { onClick: () => void }).onClick();
      expect(onClose).toHaveBeenCalledOnce();
      expect(renderToStaticMarkup(tree)).not.toContain(">关闭<");
    }
  });
});
