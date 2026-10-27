import { Children, isValidElement, type HTMLAttributes, type ReactElement, type ReactNode, type TextareaHTMLAttributes } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ActorOperation, ActorOperationParameters, ActorTurn } from "../../types";

const { stateValues } = vi.hoisted(() => ({ stateValues: [] as unknown[] }));
vi.mock("react", async (importOriginal) => ({
  ...await importOriginal<typeof import("react")>(),
  useEffect: vi.fn(),
  useRef: (current: unknown) => ({ current }),
  useState: (initial: unknown) => [stateValues.length ? stateValues.shift() : initial, vi.fn()],
}));
const { publicCast } = vi.hoisted(() => ({ publicCast: {} as Record<string, { agent_id: string; name: string }> }));
vi.mock("../../lib/cast", () => ({ useCast: () => ({ cast: publicCast }) }));
vi.mock("./submission", () => ({
  useActorSubmission: () => ({ busy: false, run: vi.fn() }),
}));
import MapOperations from "./MapOperations";

function panel(target: { kind: "agent" | "location"; id: string } = { kind: "location", id: "room" }) {
  const onClose = vi.fn();
  const turn: ActorTurn = {
    turn_id: "turn", agent_id: "self", step: 1, kind: "decide",
    briefing: { location: null, people: [], npcs: [], places: [], entities: [] },
    doing: "", invitation: null, decided: "", conversation: null, reason: "",
  };
  const overlay = MapOperations({
    worldId: "world", turn, selection: { target, x: 10, y: 10 }, onClose,
  }) as ReactElement<HTMLAttributes<HTMLDivElement>>;
  return { onClose, overlay };
}

describe("map operation context-menu boundary", () => {
  it("names self from the existing public cast while preserving other-person perception boundaries", () => {
    delete publicCast.self;
    expect(renderToStaticMarkup(panel({ kind: "agent", id: "self" }).overlay)).toContain("自己");
    publicCast.self = { agent_id: "self", name: "李世民" };
    publicCast.hidden = { agent_id: "hidden", name: "不该泄露的角色" };
    expect(renderToStaticMarkup(panel({ kind: "agent", id: "self" }).overlay)).toContain("李世民");
    const unknown = renderToStaticMarkup(panel({ kind: "agent", id: "hidden" }).overlay);
    expect(unknown).toContain("某人");
    expect(unknown).not.toContain(publicCast.hidden.name);
  });
  it("suppresses native menus retargeted to the overlay or its form children", () => {
    const { overlay } = panel();
    for (const target of [{ tagName: "DIV" }, { tagName: "TEXTAREA" }]) {
      const event = new Event("contextmenu", { bubbles: true, cancelable: true });
      Object.defineProperty(event, "target", { value: target });
      overlay.props.onContextMenu!(event as unknown as Parameters<NonNullable<typeof overlay.props.onContextMenu>>[0]);
      expect(event.defaultPrevented).toBe(true);
    }
  });
  it("keeps the overlay mounted on secondary press until contextmenu is handled", () => {
    const { overlay, onClose } = panel();
    overlay.props.onPointerDown!({ button: 2 } as Parameters<NonNullable<typeof overlay.props.onPointerDown>>[0]);
    expect(onClose).not.toHaveBeenCalled();
    const preventDefault = vi.fn();
    overlay.props.onPointerDown!({ button: 0, preventDefault } as unknown as Parameters<NonNullable<typeof overlay.props.onPointerDown>>[0]);
    expect(onClose).toHaveBeenCalledOnce();
    expect(preventDefault).not.toHaveBeenCalled();
  });
});


type FormElement = ReactElement<{ children?: ReactNode; required?: boolean; disabled?: boolean }>;
function formElements(node: ReactNode): FormElement[] {
  return Children.toArray(node).flatMap((child) => {
    if (!isValidElement(child)) return [];
    const element = child as FormElement;
    return [element, ...formElements(element.props.children)];
  });
}

function operationForm(operation: ActorOperation, required_parameters: string[], parameters: ActorOperationParameters = {}, description = "") {
  stateValues.push({ turn_id: "turn", target: { kind: "location", id: "room" }, operations: [
    { operation, available: true, reason: null, required_parameters },
  ] }, operation, parameters, description, "");
  const { overlay } = panel();
  const elements = formElements(overlay);
  const field = (label: string) => {
    const element = elements.find((candidate) => candidate.type === "label" && Children.toArray(candidate.props.children).filter((child) => typeof child === "string").join("") === label);
    expect(element, `Missing field ${label}`).toBeDefined();
    return formElements(element!.props.children).find((candidate) => ["select", "textarea", "input"].includes(String(candidate.type)))!;
  };
  const confirm = elements.find((candidate) => candidate.type === "button" && candidate.props.children === "确认")!;
  return { field, confirm, markup: renderToStaticMarkup(overlay), elements };
}

describe("map operation backend-owned requirements", () => {
  it("keeps physical description labels, HTML validation and submit gating consistent", () => {
    const optional = operationForm("physical", []);
    expect(optional.field("描述（可选）").props.required).toBe(false);
    expect(optional.confirm.props.disabled).toBe(false);
    const required = operationForm("physical", ["description"]);
    expect(required.field("描述").props.required).toBe(true);
    expect(required.confirm.props.disabled).toBe(true);
    expect(operationForm("physical", ["description"], {}, "轻敲门板").confirm.props.disabled).toBe(false);
  });
  it("can make broadcast content optional or required without changing operation", () => {
    const optional = operationForm("broadcast", []);
    expect(optional.field("广播内容（可选）").props.required).toBe(false);
    expect(optional.confirm.props.disabled).toBe(false);
    const required = operationForm("broadcast", ["message"]);
    expect(required.field("广播内容").props.required).toBe(true);
    expect(required.confirm.props.disabled).toBe(true);
    expect(operationForm("broadcast", ["message"], { message: "请留步" }).confirm.props.disabled).toBe(false);
    expect(required.markup).not.toContain("描述");
  });
  it.each(["work", "covert", "send_message"] as const)("requires the backend-declared description for %s", (operation) => {
    const missing = operationForm(operation, ["description"]);
    expect(missing.field("描述").props.required).toBe(true);
    expect(missing.confirm.props.disabled).toBe(true);
    expect(operationForm(operation, ["description"], {}, "明确具体行动").confirm.props.disabled).toBe(false);
  });
  it("only asks for the backend-required errand description", () => {
    const required = operationForm("errand", ["description"]);
    expect(required.field("描述").props.required).toBe(true);
    expect(required.markup).not.toContain("<select");
    expect(required.markup).not.toContain('type="checkbox"');
    expect(required.markup).not.toContain("传话原文");
    expect(required.confirm.props.disabled).toBe(true);
    expect(operationForm("errand", ["description"], {}, "去院子看看").confirm.props.disabled).toBe(false);
  });
  it.each([
    ["broadcast", ["message"], "广播内容"], ["errand", ["description"], "描述"],
  ] as const)("uses the chat keyboard convention for %s", (operation, required, label) => {
    const input = operationForm(operation, [...required]).field(label) as ReactElement<TextareaHTMLAttributes<HTMLTextAreaElement>>;
    for (const [shiftKey, isComposing, submits] of [[false, false, true], [true, false, false], [false, true, false]]) {
      const requestSubmit = vi.fn();
      const preventDefault = vi.fn();
      input.props.onKeyDown!({
        key: "Enter", shiftKey, nativeEvent: { isComposing }, preventDefault,
        currentTarget: { form: { requestSubmit } },
      } as unknown as Parameters<NonNullable<typeof input.props.onKeyDown>>[0]);
      expect(requestSubmit).toHaveBeenCalledTimes(submits ? 1 : 0);
      expect(preventDefault).toHaveBeenCalledTimes(submits ? 1 : 0);
    }
  });
  it("uses an accessible close icon in the header", () => {
    const { markup } = operationForm("physical", ["description"]);
    expect(markup).toContain('aria-label="关闭"');
    expect(markup).not.toContain(">关闭<");
  });
});

describe("message signature", () => {
  it.each([
    ["broadcast", false, ["message"], { message: "请留步" }], ["broadcast", true, ["message"], { message: "请留步" }],
    ["send_message", false, ["description"], {}], ["send_message", true, ["description"], {}],
  ] as const)("shows the selected anonymity state for %s (%s)", (operation, anonymous, required, parameters) => {
    const { markup } = operationForm(operation, [...required], { ...parameters, anonymous });
    expect(markup).toContain("匿名发送");
    expect(markup).toContain('type="checkbox"');
    expect(markup.includes('checked=""')).toBe(anonymous);
  });
});
