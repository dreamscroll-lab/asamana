// The world's places, each with the sub-places that belong to it. A place's name and description
// come from the map and stay read-only; its sub-places are part of the world's setup: revisable
// until the world is confirmed, read-only after, one place at a time.
import { type FormEvent, useState } from "react";

import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { SETUP_INPUT } from "./styles";
import type { Places } from "../../lib/places";
import type { SubPlaceSetupView } from "../../types";

interface PlaceRow {
  location_id: string;
  name: string;
  description: string;
}

/** Every place in map order, then any place a sub-place names that the table lacks (as 某地). */
export function placeRows(places: Places, subPlaces: readonly SubPlaceSetupView[]): PlaceRow[] {
  const rows: PlaceRow[] = Object.values(places).map((p) => ({
    location_id: p.location_id, name: p.name || "某地", description: p.description,
  }));
  const known = new Set(rows.map((r) => r.location_id));
  for (const s of subPlaces) {
    if (known.has(s.location_id)) continue;
    known.add(s.location_id);
    rows.push({ location_id: s.location_id, name: "某地", description: "" });
  }
  return rows;
}

/** Each place's sub-place names, in setup order. */
export function namesByPlace(subPlaces: readonly SubPlaceSetupView[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const s of subPlaces) (groups[s.location_id] ??= []).push(s.name);
  return groups;
}

/** The whole sub-place list with one place's names replaced and every other place's kept: the
 *  save body of a one-place edit (the backend replaces the list whole). */
export function withPlaceNames(
  rows: readonly PlaceRow[], subPlaces: readonly SubPlaceSetupView[], placeId: string, names: readonly string[],
): SubPlaceSetupView[] {
  const saved = namesByPlace(subPlaces);
  return rows.flatMap((r) =>
    (r.location_id === placeId ? names.map((n) => n.trim()) : saved[r.location_id] ?? [])
      .map((name) => ({ location_id: r.location_id, name })),
  );
}

/** The saved names an edit no longer has: removed, or renamed (a rename is a new sub-place). */
export function droppedNames(saved: readonly string[], draft: readonly string[]): string[] {
  const kept = new Set(draft.map((n) => n.trim()));
  return saved.filter((n) => !kept.has(n));
}

export default function PlacePanel({
  subPlaces,
  places,
  editable,
  onSave,
  onEditingChange,
}: {
  subPlaces: SubPlaceSetupView[];
  places: Places;
  editable: boolean;
  onEditingChange: (editing: boolean) => void;
  // Rejects with the backend's reason; the drafts stay so they can be fixed.
  onSave: (subPlaces: SubPlaceSetupView[]) => Promise<void>;
}) {
  // One place is edited at a time: its id and its draft sub-place names.
  const [draft, setDraft] = useState<{ placeId: string; names: string[] } | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const rows = placeRows(places, subPlaces);
  const saved = namesByPlace(subPlaces);
  // Collapsed, only the places that have sub-places show, plus the one being edited.
  const withSubPlaces = rows.filter((r) => saved[r.location_id]?.length || r.location_id === draft?.placeId);
  const collapsible = rows.length > withSubPlaces.length;
  const shown = expanded ? rows : withSubPlaces;

  const setNames = (names: string[]) => setDraft((d) => d && { ...d, names });

  function edit(next: { placeId: string; names: string[] } | null) {
    setDraft(next);
    setError("");
    onEditingChange(next !== null);
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!draft || busy) return;
    setBusy(true);
    setError("");
    try {
      await onSave(withPlaceNames(rows, subPlaces, draft.placeId, draft.names));
      edit(null);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="bg-slate-900 border border-slate-800/80 rounded-2xl p-4 shadow-lg space-y-3">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">
          <span className="inline-flex items-center rounded-full bg-slate-800/60 border border-slate-700/60 px-4 py-1.5">📍 地点</span>
        </h4>
        <span className="text-xs text-slate-500">{subPlaces.length} 个子地点</span>
      </div>
      {shown.length === 0 && <p className="text-xs text-slate-500">暂无子地点</p>}
      {shown.map((row) => {
        const names = saved[row.location_id] ?? [];
        return (
          <article key={row.location_id} className="bg-slate-950/50 border border-slate-800/70 rounded-lg p-3 text-xs space-y-2">
            {draft?.placeId === row.location_id ? (
              <form onSubmit={(event) => void save(event)} className="space-y-2 text-xs">
                <h5 className="m-0 text-xs font-semibold text-slate-300">编辑 · {row.name}</h5>
                {row.description && <p className="whitespace-pre-wrap text-slate-500 leading-relaxed">{row.description}</p>}
                <fieldset disabled={busy} className="m-0 min-w-0 border-0 p-0 space-y-2">
                  {draft.names.map((name, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <input
                        required
                        aria-label={`${row.name}的子地点名`}
                        value={name}
                        placeholder="子地点名"
                        onChange={(e) => setNames(draft.names.map((n, j) => (j === i ? e.target.value : n)))}
                        className={`${SETUP_INPUT} flex-1`}
                      />
                      <button
                        type="button"
                        onClick={() => setNames(draft.names.filter((_, j) => j !== i))}
                        className="text-[11px] text-rose-400 hover:text-rose-300 disabled:opacity-40"
                      >
                        删除
                      </button>
                    </div>
                  ))}
                  <button type="button" onClick={() => setNames([...draft.names, ""])} className={`${BTN} w-full`}>＋ 添加子地点</button>
                </fieldset>
                {droppedNames(names, draft.names).length > 0 && (
                  // Nobody records who started at a sub-place, so dropping one moves no one.
                  <p role="note" className="text-[11px] text-amber-300/90 leading-relaxed">
                    删去或改名的子地点不会带走原本在那里的人物、NPC 和物件，如有需要，请到对应卡片修改地点。
                  </p>
                )}
                {error && <p role="alert" className="text-[11px] text-rose-400">{error}</p>}
                <div className="flex justify-end gap-2">
                  <button type="button" className={BTN} onClick={() => edit(null)}>取消</button>
                  <button className={BTN_PRIMARY} disabled={busy}>{busy ? "保存中…" : "保存"}</button>
                </div>
              </form>
            ) : (
              <>
                <div className="flex items-center justify-between gap-2 text-slate-300">
                  <strong className="font-semibold">{row.name}</strong>
                  {editable && (
                    <button
                      className={BTN}
                      disabled={busy || draft !== null}
                      onClick={() => edit({ placeId: row.location_id, names: [...names] })}
                    >
                      编辑
                    </button>
                  )}
                </div>
                {row.description && <p className="whitespace-pre-wrap text-slate-200 leading-relaxed">{row.description}</p>}
                {names.length > 0 && (
                  <p className="text-[11px] text-slate-500 leading-relaxed">子地点：{names.join("、")}</p>
                )}
              </>
            )}
          </article>
        );
      })}
      {collapsible && (
        <button type="button" onClick={() => setExpanded((e) => !e)} className={`${BTN} w-full`}>
          {expanded ? "收起" : `展开全部 ${rows.length} 个地点`}
        </button>
      )}
    </section>
  );
}
