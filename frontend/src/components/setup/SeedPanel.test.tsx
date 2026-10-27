import { Children, isValidElement, type ReactElement, type ReactNode, type FormEvent } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EntitySetupView, NpcSetupView } from "../../types";

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
import SeedPanel from "./SeedPanel";

type Element = ReactElement<Record<string, unknown> & { children?: ReactNode }>;
function elements(node: ReactNode): Element[] {
  if (!isValidElement(node)) return [];
  const element = node as Element;
  return [element, ...Children.toArray(element.props.children).flatMap(elements)];
}
const entity: EntitySetupView = {
  entity_id: "seal", name: "玉玺", entity_type: "item", initial_state: "intact",
  description: "信物", content: "受命于天", location: "palace",
};
const npc: NpcSetupView = {
  npc_id: "guard", name: "守卫", gender: "男", age: 30, description: "认得路", location: "palace",
};
function panel(editable = true, entities: EntitySetupView[] = [entity], npcs: NpcSetupView[] = [npc]) {
  const onSaveEntity = vi.fn().mockResolvedValue(undefined);
  const onSaveNpc = vi.fn().mockResolvedValue(undefined);
  const onEditingChange = vi.fn();
  const onAddEntity = vi.fn().mockResolvedValue(undefined);
  const onAddNpc = vi.fn().mockResolvedValue(undefined);
  const onRemoveEntity = vi.fn().mockResolvedValue(undefined);
  const onRemoveNpc = vi.fn().mockResolvedValue(undefined);
  function render() {
    hooks.index = 0;
    return <>
      {SeedPanel({ kind: "entity", entities, places: {}, editable, onSave: onSaveEntity, onAdd: onAddEntity, onRemove: onRemoveEntity, onEditingChange })}
      {SeedPanel({ kind: "npc", npcs, places: {}, editable, onSave: onSaveNpc, onAdd: onAddNpc, onRemove: onRemoveNpc, onEditingChange })}
    </>;
  }
  return { render, onSaveEntity, onSaveNpc, onAddEntity, onAddNpc, onRemoveEntity, onRemoveNpc, onEditingChange };
}
const click = (element: Element) => (element.props.onClick as () => void)();
const submit = (element: Element) => (element.props.onSubmit as (e: FormEvent) => void)({ preventDefault: vi.fn() } as unknown as FormEvent);
const buttons = (tree: ReactNode) => elements(tree).filter((e) => e.type === "button" && e.props.children === "编辑");
const flush = async () => { await Promise.resolve(); await Promise.resolve(); };
beforeEach(() => {
  hooks.values = []; hooks.index = 0;
  vi.stubGlobal("window", { confirm: vi.fn().mockReturnValue(true) });
});
function changeField(tree: ReactNode, label: string, value: string) {
  const field = elements(tree).find((e) => e.type === "label"
    && Children.toArray(e.props.children).some((child) => typeof child === "string" && child.trim() === label))!;
  const control = elements(field).find((e) => e.type === "input" || e.type === "textarea" || e.type === "select")!;
  (control.props.onChange as (event: { target: { value: string } }) => void)({ target: { value } });
}

describe("setup entities and NPCs", () => {
  it("shows both records with no edit controls after confirmation", () => {
    const tree = panel(false).render();
    const markup = renderToStaticMarkup(tree);
    expect(markup).toContain("玉玺");
    expect(markup).toContain("受命于天");
    expect(markup).toContain("守卫");
    expect(buttons(tree)).toHaveLength(0);
    expect(elements(tree).some((e) => e.type === "button")).toBe(false);
  });

  it("lists fixed objects before portable items without changing the setup order", () => {
    const entities: EntitySetupView[] = [
      entity,
      { ...entity, entity_id: "gate", name: "城门", entity_type: "landmark" },
      { ...entity, entity_id: "letter", name: "密信" },
      { ...entity, entity_id: "tower", name: "高塔", entity_type: "landmark" },
    ];
    const tree = panel(false, entities).render();
    const names = elements(tree).filter((e) => e.type === "strong").map((e) => e.props.children);
    expect(names).toEqual(["城门", "高塔", "玉玺", "密信", "守卫"]);
    expect(entities.map((e) => e.entity_id)).toEqual(["seal", "gate", "letter", "tower"]);
  });

  it("saves entity fields without revising its identity and releases the confirmation gate", async () => {
    const p = panel();
    click(buttons(p.render())[0]);
    expect(p.onEditingChange).toHaveBeenCalledWith(true);
    const tree = p.render();
    const cards = elements(tree).filter((e) => e.type === "article");
    expect(elements(cards[0]).some((e) => e.type === "form")).toBe(true);
    expect(elements(cards[1]).some((e) => e.type === "form")).toBe(false);
    const content = elements(tree).find((e) => e.type === "textarea" && e.props.value === entity.content)!;
    (content.props.onChange as (e: { target: { value: string } }) => void)({ target: { value: "新的正文" } });
    submit(elements(p.render()).find((e) => e.type === "form")!);
    await flush();
    expect(p.onSaveEntity).toHaveBeenCalledWith("seal", {
      entity_type: "item", initial_state: "intact", description: "信物", content: "新的正文", location: "palace",
    });
    expect(p.onSaveNpc).not.toHaveBeenCalled();
    expect(p.onEditingChange).toHaveBeenLastCalledWith(false);
  });

  it("keeps a failed NPC draft available for retry and cancels without a write", async () => {
    const p = panel();
    p.onSaveNpc.mockRejectedValueOnce(new Error("Save failed"));
    click(buttons(p.render())[1]);
    const cards = elements(p.render()).filter((e) => e.type === "article");
    expect(elements(cards[0]).some((e) => e.type === "form")).toBe(false);
    expect(elements(cards[1]).some((e) => e.type === "form")).toBe(true);
    submit(elements(p.render()).find((e) => e.type === "form")!);
    await flush();
    expect(p.onSaveNpc).toHaveBeenCalledWith("guard", {
      gender: "男", age: 30, description: "认得路", location: "palace",
    });
    expect(renderToStaticMarkup(p.render())).toContain("Save failed");
    expect(p.onEditingChange).toHaveBeenLastCalledWith(true);
    click(elements(p.render()).find((e) => e.type === "button" && e.props.children === "取消")!);
    expect(p.onEditingChange).toHaveBeenLastCalledWith(false);
    expect(p.onSaveNpc).toHaveBeenCalledTimes(1);
    expect(elements(p.render()).some((e) => e.type === "form")).toBe(false);
  });

  it("adds a fixed object through the shared form without submitting an id", async () => {
    const p = panel();
    click(elements(p.render()).find((e) => e.type === "button" && renderToStaticMarkup(e).includes("添加物件"))!);
    expect(p.onEditingChange).toHaveBeenLastCalledWith(true);
    changeField(p.render(), "名称", "新告示");
    changeField(p.render(), "初始地点", "palace");
    changeField(p.render(), "内容", "今日闭门");
    submit(elements(p.render()).find((e) => e.type === "form")!);
    await flush();
    expect(p.onAddEntity).toHaveBeenCalledWith({
      name: "新告示", entity_type: "landmark", initial_state: "intact",
      description: "", content: "今日闭门", location: "palace",
    });
    expect(p.onSaveEntity).not.toHaveBeenCalled();
    expect(p.onEditingChange).toHaveBeenLastCalledWith(false);
    expect(elements(p.render()).some((e) => e.type === "form")).toBe(false);
  });

  it("keeps a new NPC draft after a failed save and cancels without adding it", async () => {
    const p = panel();
    p.onAddNpc.mockRejectedValueOnce(new Error("NPC quota reached"));
    click(elements(p.render()).find((e) => e.type === "button" && renderToStaticMarkup(e).includes("添加 NPC"))!);
    changeField(p.render(), "名称", "新信使");
    changeField(p.render(), "性别", "女");
    changeField(p.render(), "年龄", "25");
    changeField(p.render(), "初始地点", "palace");
    changeField(p.render(), "描述", "认得路");
    submit(elements(p.render()).find((e) => e.type === "form")!);
    await flush();
    expect(p.onAddNpc).toHaveBeenCalledWith({
      name: "新信使", gender: "女", age: 25, location: "palace", description: "认得路",
    });
    expect(renderToStaticMarkup(p.render())).toContain("NPC quota reached");
    expect(p.onEditingChange).toHaveBeenLastCalledWith(true);
    expect(elements(p.render()).some((e) => e.type === "input" && e.props.value === "新信使")).toBe(true);
    click(elements(p.render()).find((e) => e.type === "button" && e.props.children === "取消")!);
    expect(p.onEditingChange).toHaveBeenLastCalledWith(false);
    expect(p.onAddNpc).toHaveBeenCalledTimes(1);
  });

  it("deletes only the selected record and does nothing when deletion is cancelled", async () => {
    const p = panel();
    let deletes = elements(p.render()).filter((e) => e.type === "button" && e.props.children === "删除");
    vi.mocked(window.confirm).mockReturnValueOnce(false);
    click(deletes[0]);
    await flush();
    expect(p.onRemoveEntity).not.toHaveBeenCalled();
    click(deletes[0]);
    await flush();
    expect(p.onRemoveEntity).toHaveBeenCalledWith("seal");
    expect(p.onRemoveNpc).not.toHaveBeenCalled();
    expect(p.onEditingChange).toHaveBeenLastCalledWith(false);
    deletes = elements(p.render()).filter((e) => e.type === "button" && e.props.children === "删除");
    click(deletes[1]);
    await flush();
    expect(p.onRemoveNpc).toHaveBeenCalledWith("guard");
  });

  it("shows a deletion failure without hiding the record and releases the confirmation gate", async () => {
    const p = panel();
    p.onRemoveEntity.mockRejectedValueOnce(new Error("Delete failed"));
    click(elements(p.render()).find((e) => e.type === "button" && e.props.children === "删除")!);
    await flush();
    const markup = renderToStaticMarkup(p.render());
    expect(markup).toContain("Delete failed");
    expect(markup).toContain("玉玺");
    expect(p.onEditingChange).toHaveBeenLastCalledWith(false);
  });


  it("requires a gender of a new NPC with a disabled placeholder and only male or female choices", () => {
    const p = panel();
    click(elements(p.render()).find((e) => e.type === "button" && renderToStaticMarkup(e).includes("添加 NPC"))!);
    const genderSelect = elements(p.render()).find((e) => e.type === "select" && e.props.value === ""
      && elements(e).some((option) => option.type === "option" && option.props.value === "男"))!;
    expect(genderSelect.props.required).toBe(true);
    const options = elements(genderSelect).filter((e) => e.type === "option");
    expect(options.map((e) => e.props.children)).toEqual(["请选择性别", "男", "女"]);
    expect(options[0].props.disabled).toBe(true);
    changeField(p.render(), "性别", "女");
    const selected = elements(p.render()).find((e) => e.type === "select" && e.props.value === "女")!;
    expect(elements(selected).filter((e) => e.type === "option").map((e) => e.props.children)).toEqual(["男", "女"]);
  });

  it("edits an NPC the build left without a gender without making the director guess one", async () => {
    const p = panel(true, [entity], [{ ...npc, gender: "" }]);
    click(buttons(p.render())[1]);
    const genderSelect = elements(p.render()).find((e) => e.type === "select" && e.props.value === "")!;
    expect(genderSelect.props.required).toBe(false);
    changeField(p.render(), "描述", "认得路，识字");
    submit(elements(p.render()).find((e) => e.type === "form")!);
    await flush();
    expect(p.onSaveNpc).toHaveBeenCalledWith("guard", {
      gender: "", age: 30, description: "认得路，识字", location: "palace",
    });
  });

});
