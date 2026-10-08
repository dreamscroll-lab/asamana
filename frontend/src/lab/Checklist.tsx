import { useEffect, useMemo, useRef, useState } from "react";

import type { LabScene } from "./scenes";
import { markKey, type Marks } from "./marks";

interface Props {
  /** Already filtered by `query` — the view above owns the needle, because the arrow
   *  keys walk this same list and must not step onto a scene the search has hidden. */
  scenes: LabScene[];
  template: string;
  marks: Marks;
  pickId: string | null;
  onPick: (id: string) => void;
  /** Only to decide whether a folded group should open anyway — the box itself is pinned
   *  above this list, out of the scroll. */
  query: string;
}

const DOT: Record<string, string> = {
  pass: "bg-teal-400",
  fail: "bg-rose-400",
};

/** The scene list: search, foldable groups, and each scene's mark — sixty-odd entries don't fit on a screen. */
export default function Checklist({
  scenes,
  template,
  marks,
  pickId,
  onPick,
  query,
}: Props) {
  const [folded, setFolded] = useState<ReadonlySet<string>>(() => new Set());
  const searching = query.trim().length > 0;
  // A scene can be picked off-screen (arrow key, address bar), so scroll the list to it.
  const picked = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    picked.current?.scrollIntoView({ block: "nearest" });
  }, [pickId]);

  const groups = useMemo(() => {
    const order: string[] = [];
    const byGroup = new Map<string, LabScene[]>();
    for (const s of scenes) {
      if (!byGroup.has(s.group)) {
        byGroup.set(s.group, []);
        order.push(s.group);
      }
      byGroup.get(s.group)!.push(s);
    }
    return order.map((g) => ({ group: g, items: byGroup.get(g)! }));
  }, [scenes]);

  return (
    <>
      {!scenes.length && <p className="text-[11px] text-slate-600 px-1 py-3">No matching scenes.</p>}

      {groups.map(({ group, items }) => {
        // A search is a request to see the matches, so it overrides a fold.
        const open = !folded.has(group) || searching;
        const done = items.filter((s) => marks[markKey(template, s.id)]).length;
        return (
          <div key={group} className="mb-1.5">
            <button
              onClick={() =>
                setFolded((prev) => {
                  const next = new Set(prev);
                  next.has(group) ? next.delete(group) : next.add(group);
                  return next;
                })
              }
              className="w-full flex items-center gap-1.5 px-1 py-1 text-[10px] uppercase tracking-wider text-slate-500 hover:text-slate-300 transition"
            >
              <span className={`transition ${open ? "rotate-90" : ""}`}>›</span>
              <span>{group}</span>
              <span className="ml-auto tabular-nums text-slate-600">
                {done}/{items.length}
              </span>
            </button>
            {open && (
              <div className="flex flex-col gap-0.5">
                {items.map((s) => {
                  const mark = marks[markKey(template, s.id)];
                  return (
                    <button
                      key={s.id}
                      ref={pickId === s.id ? picked : undefined}
                      onClick={() => onPick(s.id)}
                      className={`flex items-center gap-1.5 text-left text-[12px] pl-2 pr-2 py-1 rounded-md border transition ${
                        pickId === s.id
                          ? "bg-indigo-600/20 border-indigo-500/60 text-indigo-100"
                          : "bg-transparent border-transparent text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full shrink-0 ${mark ? DOT[mark] : "bg-slate-700"}`}
                      />
                      <span className="min-w-0 truncate">{s.title}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}
