import { Children, isValidElement, type ReactElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ActorTurn } from "../../types";

const state = vi.hoisted(() => ({ thought: "" }));
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useState: () => [state.thought, vi.fn()],
}));
vi.mock("../../hooks/useAutoGrow", () => ({ useAutoGrow: () => undefined }));
import InterruptQuestion from "./InterruptQuestion";

type Element = ReactElement<{ children?: ReactNode; onClick?: () => void; disabled?: boolean; maxLength?: number }>;
function elements(node: ReactNode): Element[] {
  return Children.toArray(node).flatMap((child) => {
    if (!isValidElement(child)) return [];
    const element = child as Element;
    return [element, ...elements(element.props.children)];
  });
}

function question(thought = "", open = true, busy = false) {
  state.thought = thought;
  const onAnswer = vi.fn();
  const turn: ActorTurn = {
    turn_id: "turn", agent_id: "self", step: 2, kind: "interrupt", briefing: null,
    doing: "整理手头的事", reason: "有消息传来", invitation: null, decided: "", conversation: null,
  };
  const tree = InterruptQuestion({ turn, open, busy, maxLength: 500, onAnswer });
  return { onAnswer, tree, fields: elements(tree) };
}

describe("interrupt thought", () => {
  it("allows either decision without a thought", () => {
    const { fields, onAnswer, tree } = question("  ");
    expect(renderToStaticMarkup(tree)).toContain("想法（可选）");
    const buttons = fields.filter((element) => element.type === "button");
    buttons[0].props.onClick!();
    expect(onAnswer).toHaveBeenLastCalledWith({ interrupt: false });
    buttons[1].props.onClick!();
    expect(onAnswer).toHaveBeenLastCalledWith({ interrupt: true });
  });
  it("submits the actor's thought with the decision", () => {
    const { fields, onAnswer } = question(" 我得先处理来信 ");
    const buttons = fields.filter((element) => element.type === "button");
    buttons[1].props.onClick!();
    expect(onAnswer).toHaveBeenLastCalledWith({ interrupt: true, thought: "我得先处理来信" });
    buttons[0].props.onClick!();
    expect(onAnswer).toHaveBeenLastCalledWith({ interrupt: false, thought: "我得先处理来信" });
  });
  it("locks inputs while submitting and removes them when the turn closes", () => {
    const { fields } = question("", true, true);
    const input = fields.find((element) => element.type === "textarea")!;
    expect(input.props.maxLength).toBe(500);
    expect(input.props.disabled).toBe(true);
    expect(fields.filter((element) => element.type === "button").every((element) => element.props.disabled)).toBe(true);
    expect(question("", false).fields.some((element) => ["button", "textarea"].includes(String(element.type)))).toBe(false);
  });
});
