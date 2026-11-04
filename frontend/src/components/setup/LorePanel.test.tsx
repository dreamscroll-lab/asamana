import { Children, isValidElement, type FormEvent, type ReactElement, type ReactNode } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import type { SetupView } from "../../types";
import { SETUP_INPUT } from "./styles";

const hooks = vi.hoisted(() => ({ values: [] as unknown[], index: 0 }));
vi.mock("react", async (original) => ({
  ...await original<typeof import("react")>(),
  useState: (initial: unknown) => {
    const index = hooks.index++;
    if (!(index in hooks.values)) hooks.values[index] = initial;
    return [hooks.values[index], (value: unknown) => { hooks.values[index] = value; }];
  },
}));
import LorePanel from "./LorePanel";
type Element = ReactElement<Record<string, unknown> & { children?: ReactNode }>;
function elements(node: ReactNode): Element[] {
  if (!isValidElement(node)) return [];
  const element = node as Element;
  return [element, ...Children.toArray(element.props.children).flatMap(elements)];
}
const setup: SetupView = {
  narrative_theme: "信任", seconds_per_step: 3600,
  opening_time: { era_name: "", start_year: 1, start_month: 1, start_day: 1, start_hour: 6 },
  characters: [], entities: [], npcs: [], relations: [], historical_events: [],
};
beforeEach(() => { hooks.values = []; hooks.index = 0; });
it("uses shared inputs and native validation, preserving the draft when saving fails", async () => {
  const onSave = vi.fn().mockRejectedValueOnce(new Error("Save failed"));
  const onEditingChange = vi.fn();
  const render = () => {
    hooks.index = 0;
    return elements(LorePanel({
      setup, editable: true, onSave, onEditingChange,
      renderHeader: (pace) => <header>{pace}</header>,
    }));
  };
  (render().find((e) => e.type === "button")!.props.onClick as () => void)();
  const controls = render().filter((e) => e.type === "input" || e.type === "textarea");
  expect(controls.every((e) => String(e.props.className).includes(SETUP_INPUT))).toBe(true);
  expect(controls.filter((e) => e.props.type === "number").every((e) => e.props.required)).toBe(true);
  expect(render().filter((e) => e.type === "select")).toHaveLength(1);
  const header = render().find((e) => e.type === "header")!;
  expect(elements(header).filter((e) => e.type === "select")).toHaveLength(1);
  const form = render().find((e) => e.type === "form")!;
  expect(elements(form).some((e) => e.type === "select")).toBe(false);
  const pace = render().find((e) => e.type === "select")!;
  expect(String(pace.props.className)).toContain(SETUP_INPUT);
  (pace.props.onChange as (e: { target: { value: string } }) => void)({ target: { value: "2" } });
  const submit = () => (render().find((e) => e.type === "form")!.props.onSubmit as (e: FormEvent) => void)(
    { preventDefault: vi.fn() } as unknown as FormEvent,
  );
  submit();
  await Promise.resolve(); await Promise.resolve();
  expect(render().some((e) => e.props.children === "Save failed")).toBe(true);
  expect(onEditingChange).toHaveBeenLastCalledWith(true);
  onSave.mockResolvedValueOnce(undefined);
  submit();
  await Promise.resolve(); await Promise.resolve();
  expect(onSave).toHaveBeenLastCalledWith({
    narrative_theme: setup.narrative_theme, opening_time: setup.opening_time, seconds_per_step: 7200,
  });
  expect(onEditingChange).toHaveBeenLastCalledWith(false);
});
it("shows confirmed setup without edit controls", () => {
  expect(elements(LorePanel({ setup, editable: false, onSave: vi.fn(), onEditingChange: vi.fn() }))
    .some((e) => e.type === "button")).toBe(false);
});

it("displays the opening as a date and a padded clock time", () => {
  const dated = { ...setup, opening_time: {
    era_name: "武德", start_year: 9, start_month: 6, start_day: 1, start_hour: 6,
  } };
  const tree = elements(LorePanel({
    setup: dated, editable: false, onSave: vi.fn(), onEditingChange: vi.fn(),
  }));
  expect(tree.some((e) => e.props.children === "武德 9年6月1日")).toBe(true);
  expect(tree.some((e) => e.props.children === "06:00")).toBe(true);
});

it("cancels the header pace change with the rest of the world settings", () => {
  const onSave = vi.fn();
  const onEditingChange = vi.fn();
  const render = () => {
    hooks.index = 0;
    return elements(LorePanel({ setup, editable: true, onSave, onEditingChange }));
  };
  (render().find((e) => e.type === "button")!.props.onClick as () => void)();
  (render().find((e) => e.type === "select")!.props.onChange as (e: { target: { value: string } }) => void)(
    { target: { value: "4" } },
  );
  expect(render().find((e) => e.type === "select")!.props.value).toBe(4);
  (render().find((e) => e.type === "button" && e.props.children === "取消")!.props.onClick as () => void)();
  expect(render().some((e) => e.type === "select")).toBe(false);
  expect(onSave).not.toHaveBeenCalled();
  expect(onEditingChange).toHaveBeenLastCalledWith(false);
  (render().find((e) => e.type === "button")!.props.onClick as () => void)();
  expect(render().find((e) => e.type === "select")!.props.value).toBe(1);
});
