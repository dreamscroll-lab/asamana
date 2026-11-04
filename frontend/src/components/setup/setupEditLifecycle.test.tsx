import { Children, isValidElement, type ReactElement, type ReactNode } from "react";
import { beforeEach, expect, it, vi } from "vitest";

const hooks = vi.hoisted(() => ({ values: [] as unknown[], index: 0 }));
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useState: (initial: unknown) => {
    const index = hooks.index++;
    if (!(index in hooks.values)) hooks.values[index] = initial;
    return [hooks.values[index], (next: unknown) => {
      hooks.values[index] = typeof next === "function" ? next(hooks.values[index]) : next;
    }];
  },
}));
import HistoryPanel from "./HistoryPanel";
import RelationPanel from "./RelationPanel";
type Element = ReactElement<Record<string, unknown> & { children?: ReactNode }>;
function elements(node: ReactNode): Element[] {
  if (!isValidElement(node)) return [];
  const element = node as Element;
  return [element, ...Children.toArray(element.props.children).flatMap(elements)];
}
beforeEach(() => { hooks.values = []; hooks.index = 0; });
it.each(["history", "relations"])("%s preserves the editing gate on failure and releases it on save or cancel", async (kind) => {
  const onSave = vi.fn().mockRejectedValueOnce(new Error("Save failed"));
  const onEditingChange = vi.fn();
  const props = { cast: {}, editable: true, onSave, onEditingChange };
  const render = () => {
    hooks.index = 0;
    return elements(kind === "history"
      ? HistoryPanel({ ...props, events: [] })
      : RelationPanel({ ...props, relations: [] }));
  };
  const click = (label: string) => {
    const button = render().find((e) => e.type === "button" && e.props.children === label)!;
    (button.props.onClick as () => void)();
  };
  click("✎ 编辑");
  expect(onEditingChange).toHaveBeenLastCalledWith(true);
  click("保存");
  await Promise.resolve(); await Promise.resolve();
  expect(render().some((e) => e.props.children === "Save failed")).toBe(true);
  expect(onEditingChange).toHaveBeenLastCalledWith(true);
  onSave.mockResolvedValueOnce(undefined);
  click("保存");
  await Promise.resolve(); await Promise.resolve();
  expect(onEditingChange).toHaveBeenLastCalledWith(false);
  click("✎ 编辑");
  click("取消");
  expect(onEditingChange).toHaveBeenLastCalledWith(false);
  expect(onSave).toHaveBeenCalledTimes(2);
});
