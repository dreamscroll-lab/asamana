import { useState, type FormEvent } from "react";
import type { EntityRevision, EntitySetupView, NewEntity, NewNpc, NpcRevision, NpcSetupView } from "../../types";
import { DEFAULT_ENTITY_STATE } from "../../lib/entity";
import { placeName, type Places } from "../../lib/places";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { SETUP_INPUT } from "./styles";

const INPUT = `${SETUP_INPUT} block w-full mt-1`;
type Editing = { kind: "entity"; draft: EntitySetupView } | { kind: "npc"; draft: NpcSetupView };

type Props = {
  places: Places;
  editable: boolean;
  onEditingChange: (editing: boolean) => void;
  onRemove: (id: string) => Promise<void>;
} & (
  | {
    kind: "entity";
    entities: EntitySetupView[];
    onSave: (id: string, revision: EntityRevision) => Promise<void>;
    onAdd: (entity: NewEntity) => Promise<void>;
  }
  | {
    kind: "npc";
    npcs: NpcSetupView[];
    onSave: (id: string, revision: NpcRevision) => Promise<void>;
    onAdd: (npc: NewNpc) => Promise<void>;
  }
);

export default function SeedPanel(props: Props) {
  const { places, editable, onEditingChange } = props;
  const [editing, setEditing] = useState<Editing | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  function edit(next: Editing | null) {
    setEditing(next);
    setError("");
    onEditingChange(next !== null);
  }

  function update(patch: Partial<EntitySetupView> & Partial<NpcSetupView>) {
    setEditing((current) => current && ({ ...current, draft: { ...current.draft, ...patch } } as Editing));
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!editing || busy) return;
    setBusy(true);
    setError("");
    try {
      if (editing.kind === "entity" && props.kind === "entity") {
        const { entity_id, ...entry } = editing.draft;
        if (!entity_id) await props.onAdd(entry);
        else {
          const { name: _name, ...revision } = entry;
          await props.onSave(entity_id, revision);
        }
      } else if (editing.kind === "npc" && props.kind === "npc") {
        const { npc_id, ...entry } = editing.draft;
        if (!npc_id) await props.onAdd(entry);
        else {
          const { name: _name, ...revision } = entry;
          await props.onSave(npc_id, revision);
        }
      }
      edit(null);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  function add() {
    if (props.kind === "entity") edit({ kind: "entity", draft: {
      entity_id: "", name: "", entity_type: "landmark", initial_state: DEFAULT_ENTITY_STATE,
      description: "", content: "", location: "",
    } });
    else edit({ kind: "npc", draft: {
      npc_id: "", name: "", gender: "", age: NaN, description: "", location: "",
    } });
  }

  async function remove(id: string, name: string) {
    if (busy || editing || !window.confirm(`删除${props.kind === "entity" ? "物件" : " NPC"}「${name}」？`)) return;
    setBusy(true);
    setError("");
    onEditingChange(true);
    try {
      await props.onRemove(id);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
      onEditingChange(false);
    }
  }

  const isNew = editing !== null && (editing.kind === "entity" ? !editing.draft.entity_id : !editing.draft.npc_id);
  const addLabel = props.kind === "entity" ? "添加物件" : "添加 NPC";

  const orderedEntities = props.kind === "entity" ? [...props.entities].sort(
    (a, b) => Number(a.entity_type === "item") - Number(b.entity_type === "item"),
  ) : [];

  const editor = editing && <form onSubmit={(event) => void save(event)} className="space-y-2 text-xs">
    <h5 className="m-0 text-xs font-semibold text-slate-300">{isNew ? addLabel : `编辑 · ${editing.draft.name}`}</h5>
    <fieldset disabled={busy} className="m-0 min-w-0 border-0 p-0 space-y-2">
      {isNew && <label className="block text-[11px] text-slate-400">名称 <span className="text-slate-500">· 添加后不可修改</span>
        <input required autoFocus className={INPUT} value={editing.draft.name} onChange={(e) => update({ name: e.target.value })} />
      </label>}
      {editing.kind === "entity" ? <>
        <label className="block text-[11px] text-slate-400">类型<select className={INPUT} value={editing.draft.entity_type} onChange={(e) => update({ entity_type: e.target.value as EntitySetupView["entity_type"] })}>
          <option value="landmark">固定物件</option><option value="item">可携带物品</option>
        </select></label>
        <label className="block text-[11px] text-slate-400">初始状态<input required className={INPUT} value={editing.draft.initial_state} onChange={(e) => update({ initial_state: e.target.value })} /></label>
        <label className="block text-[11px] text-slate-400">内容<textarea rows={4} className={`${INPUT} resize-y`} value={editing.draft.content} onChange={(e) => update({ content: e.target.value })} /></label>
      </> : <>
        <label className="block text-[11px] text-slate-400">性别<select required={isNew} className={INPUT} value={editing.draft.gender} onChange={(e) => update({ gender: e.target.value })}>
          {!editing.draft.gender && <option value="" disabled>请选择性别</option>}
          <option value="男">男</option><option value="女">女</option>
        </select></label>
        <label className="block text-[11px] text-slate-400">年龄<input required type="number" min={0} step={1} className={INPUT} value={Number.isNaN(editing.draft.age) ? "" : editing.draft.age} onChange={(e) => update({ age: e.target.value === "" ? NaN : Number(e.target.value) })} /></label>
      </>}
      <label className="block text-[11px] text-slate-400">初始地点<select required className={INPUT} value={editing.draft.location} onChange={(e) => update({ location: e.target.value })}>
        {!editing.draft.location && <option value="" disabled>请选择初始地点</option>}
        {Object.values(places).map((place) => <option key={place.location_id} value={place.location_id}>{place.name}</option>)}
      </select></label>
      <label className="block text-[11px] text-slate-400">描述<textarea required={editing.kind === "npc"} rows={3} className={`${INPUT} resize-y`} value={editing.draft.description} onChange={(e) => update({ description: e.target.value })} /></label>
      {error && <p role="alert" className="text-[11px] text-rose-400">{error}</p>}
      <div className="flex justify-end gap-2"><button type="button" className={BTN} onClick={() => edit(null)}>取消</button><button className={BTN_PRIMARY}>{busy ? "保存中…" : "保存"}</button></div>
    </fieldset>
  </form>;

  const footer = <>
    {isNew && <article className="bg-slate-950/50 border border-slate-800/70 rounded-lg p-3 text-xs space-y-2">{editor}</article>}
    {editable && !editing && <button type="button" disabled={busy} onClick={add} className={`${BTN} w-full`}>＋ {addLabel}</button>}
    {!editing && error && <p role="alert" className="text-[11px] text-rose-400">{error}</p>}
  </>;

  return props.kind === "entity" ? <section className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg space-y-3">
    <div className="flex items-center justify-between">
      <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
        <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">📦 世界物件</span>
      </h4>
      <span className="text-xs text-slate-500">{props.entities.length} 件</span>
    </div>
    {!props.entities.length && <p className="text-xs text-slate-500">暂无物件</p>}
    {orderedEntities.map((entity) => <article key={entity.entity_id} className="bg-slate-950/50 border border-slate-800/70 rounded-lg p-3 text-xs space-y-2">
      {editing?.kind === "entity" && editing.draft.entity_id === entity.entity_id ? editor : <>
        <div className="flex items-center justify-between gap-2 text-slate-300"><strong className="font-semibold">{entity.name}</strong>
          {editable && <div className="flex items-center gap-2 shrink-0">
            <button className={BTN} disabled={busy || editing !== null} onClick={() => edit({ kind: "entity", draft: { ...entity } })}>编辑</button>
            <button className="text-[11px] text-rose-400 hover:text-rose-300 disabled:opacity-40" disabled={busy || editing !== null} onClick={() => void remove(entity.entity_id, entity.name)}>删除</button>
          </div>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={`inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium ${entity.entity_type === "item"
            ? "border-sky-500/30 bg-sky-500/10 text-sky-300"
            : "border-amber-500/30 bg-amber-500/10 text-amber-300"}`}>
            {entity.entity_type === "item" ? "可携带物品" : "固定物件"}
          </span>
          <span className="text-[10px] text-slate-500">{entity.initial_state} · {placeName(places, entity.location)}</span>
        </div>
        <p className="whitespace-pre-wrap text-slate-200 leading-relaxed">{entity.description}</p>
        {entity.content && <p className="whitespace-pre-wrap text-[11px] text-slate-500 leading-relaxed">内容：{entity.content}</p>}
      </>}
    </article>)}
    {footer}
  </section> : <section className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg space-y-3">
    <div className="flex items-center justify-between">
      <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
        <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">👤 NPC</span>
      </h4>
      <span className="text-xs text-slate-500">{props.npcs.length} 人</span>
    </div>
    <p className="text-[11px] leading-relaxed text-slate-500">NPC 可由角色指派执行传话、现场查看和物品运送等任务。执行时按指定内容行动，不自主交谈或临场改变任务。</p>
    {!props.npcs.length && <p className="text-xs text-slate-500">暂无 NPC</p>}
    {props.npcs.map((npc) => <article key={npc.npc_id} className="bg-slate-950/50 border border-slate-800/70 rounded-lg p-3 text-xs space-y-2">
      {editing?.kind === "npc" && editing.draft.npc_id === npc.npc_id ? editor : <>
        <div className="flex items-center justify-between gap-2 text-slate-300"><strong className="font-semibold">{npc.name}</strong>
          {editable && <div className="flex items-center gap-2 shrink-0">
            <button className={BTN} disabled={busy || editing !== null} onClick={() => edit({ kind: "npc", draft: { ...npc } })}>编辑</button>
            <button className="text-[11px] text-rose-400 hover:text-rose-300 disabled:opacity-40" disabled={busy || editing !== null} onClick={() => void remove(npc.npc_id, npc.name)}>删除</button>
          </div>}
        </div>
        <p className="text-[10px] text-slate-500">{npc.gender || "性别未知"} · {npc.age} 岁 · {placeName(places, npc.location)}</p>
        <p className="whitespace-pre-wrap text-slate-200 leading-relaxed">{npc.description}</p>
      </>}
    </article>)}
    {footer}
  </section>;
}
