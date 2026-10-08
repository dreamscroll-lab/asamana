// The world's name, editable in place wherever it is shown (sidebar, review card,
// observation bar), so the save path exists once.
//
// Every interactive element stops propagation: the sidebar row navigates on click, so
// clicking into the input would otherwise leave the page.
import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import { MAX_WORLD_NAME_LEN } from "../lib/contract";
import type { WorldMeta } from "../types";

export default function EditableWorldName({
  worldId,
  name,
  className = "",
  onRenamed,
}: {
  worldId: string;
  name: string;
  className?: string;
  // Fires with the world's fresh metadata; the caller decides what to refresh.
  onRenamed: (meta: WorldMeta) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(name);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Start from the current name, not a stale draft from a cancelled edit.
  useEffect(() => {
    if (editing) {
      setDraft(name);
      setError("");
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing, name]);

  async function save() {
    const next = draft.trim();
    if (!next || next === name) {
      setEditing(false);
      return;
    }
    setBusy(true);
    try {
      const meta = await api.renameWorld(worldId, next);
      onRenamed(meta);
      setEditing(false);
    } catch (e) {
      // Stay in edit mode so the user's text is still there to fix or abandon.
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!editing) {
    return (
      <span className="group/name inline-flex items-center gap-1.5 min-w-0">
        <span title={name} className={`truncate ${className}`}>
          {name}
        </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setEditing(true);
          }}
          title="重命名"
          // Two hover scopes: `group/name` is this component's, the unscoped one answers a
          // surrounding row, so the rename appears together with its delete.
          className="shrink-0 opacity-0 group-hover/name:opacity-100 group-hover:opacity-100 focus:opacity-100 text-slate-500 hover:text-slate-200 transition text-xs leading-none"
        >
          ✎
        </button>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-2 min-w-0" onClick={(e) => e.stopPropagation()}>
      <input
        ref={inputRef}
        value={draft}
        maxLength={MAX_WORLD_NAME_LEN}
        disabled={busy}
        onChange={(e) => setDraft(e.target.value)}
        // Blur saves; Esc leaves without saving.
        onBlur={() => void save()}
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
          else if (e.key === "Escape") setEditing(false);
        }}
        className={`min-w-0 bg-slate-950 border border-indigo-700/70 rounded-lg px-2 py-0.5 text-slate-100 outline-none focus:border-indigo-500 disabled:opacity-50 ${className}`}
      />
      {error && <span className="shrink-0 text-[11px] text-rose-400">{error}</span>}
    </span>
  );
}
