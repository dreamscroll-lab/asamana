import { useEffect, useRef, useState } from "react";

import {
  importTemplate,
  suggestTemplateName,
  TEMPLATE_NAME_RE,
  type ImportOutcome,
} from "./importTemplate";

/**
 * Import map: drop an archive in and have the contract judge it before it lands.
 *
 * A popover, not a dialog or a layout band: a refusal is a list of things to fix, which a dialog's
 * dismiss button would get in the way of re-reading, and a band would push the whole bench down.
 */
export default function TemplateImport({
  onClose,
  onImported,
}: {
  onClose: () => void;
  onImported: (template: string, files: number) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<ImportOutcome | null>(null);

  const panel = useRef<HTMLDivElement>(null);
  const valid = TEMPLATE_NAME_RE.test(name);

  // Close on outside click and Escape. The trigger is excluded, or its own toggle would reopen
  // the panel this handler just closed.
  useEffect(() => {
    const away = (e: MouseEvent) => {
      const el = e.target as HTMLElement | null;
      if (panel.current?.contains(el ?? null) || el?.closest("[data-import-trigger]")) return;
      onClose();
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [onClose]);

  function choose(picked: File | null) {
    setFile(picked);
    setOutcome(null);
    if (picked) setName(suggestTemplateName(picked.name));
  }

  async function send() {
    if (!file || !valid) return;
    setBusy(true);
    setOutcome(null);
    const result = await importTemplate(name, file);
    setBusy(false);
    setOutcome(result);
    if (result.ok) onImported(name, result.files);
  }

  const megabytes = file ? (file.size / (1024 * 1024)).toFixed(1) : "";

  return (
    <div
      ref={panel}
      className="absolute right-4 top-full mt-2 z-50 w-[36rem] max-w-[calc(100vw-2rem)] rounded-xl border border-slate-700/80 bg-slate-900/80 backdrop-blur-xl shadow-2xl shadow-black/50 p-4"
    >
      <div className="flex items-center gap-2">
        <input
          type="file"
          accept=".zip,application/zip"
          onChange={(e) => choose(e.target.files?.[0] ?? null)}
          className="min-w-0 flex-1 text-[11px] text-slate-400 file:mr-2 file:rounded-md file:border file:border-slate-700 file:bg-slate-900 file:px-2 file:py-1 file:text-[11px] file:text-slate-300 hover:file:border-slate-600"
        />
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Map directory name"
          className={`w-40 shrink-0 text-[12px] bg-slate-950 border rounded-md px-2 py-1 text-slate-200 placeholder:text-slate-600 focus:outline-none ${
            name && !valid ? "border-rose-700" : "border-slate-800 focus:border-indigo-500/70"
          }`}
        />
        <button
          onClick={send}
          disabled={!file || !valid || busy}
          className="text-[11px] px-2.5 py-1 rounded-md border border-indigo-600 bg-indigo-700 text-white disabled:opacity-40 disabled:border-slate-800 disabled:bg-slate-900"
        >
          {busy ? `Uploading and checking…${megabytes && ` (${megabytes} MB)`}` : "Import"}
        </button>
        <button
          onClick={onClose}
          className="text-[11px] px-2 py-1 rounded-md border border-slate-800 text-slate-400 hover:border-slate-700"
        >
          Cancel
        </button>
      </div>

      <p className="mt-2 text-[10px] text-slate-600 leading-relaxed">
        Directory name: starts with a lowercase letter; lowercase letters, digits and underscores only. A leading underscore is reserved for reference samples and can't be imported.
      </p>

      {name && !valid && (
        <p className="mt-1.5 text-[11px] text-rose-400">Invalid name; fix it to import.</p>
      )}

      {outcome?.ok && (
        <p className="mt-1.5 text-[11px] text-teal-300">
          Imported "{name}": {outcome.files} files written, {outcome.connections} connections derived from the map.
        </p>
      )}

      {outcome && !outcome.ok && outcome.kind === "conflict" && (
        <p className="mt-1.5 text-[11px] text-amber-300">
          {outcome.message}
        </p>
      )}

      {outcome && !outcome.ok && outcome.kind === "error" && (
        <p className="mt-1.5 text-[11px] text-rose-400">{outcome.message}</p>
      )}

      {outcome && !outcome.ok && outcome.kind === "rejected" && (
        <div className="mt-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] text-rose-300">
              Map failed validation ({outcome.problems.length} problems); nothing was written
            </span>
            <button
              onClick={() => navigator.clipboard?.writeText(outcome.problems.join("\n"))}
              className="text-[10px] px-1.5 py-0.5 rounded border border-slate-800 text-slate-400 hover:border-slate-700"
            >
              Copy all
            </button>
            <span className="text-[10px] text-slate-600">
              Each one is traced in README.md §8
            </span>
          </div>
          {/* Preflight is off, so the marker and indent are set explicitly: a bare <ul> draws a
              second bullet beside the one in the text, 40px in. */}
          <ul className="max-h-[40vh] overflow-y-auto space-y-1 pr-1 list-disc pl-4 marker:text-rose-400/60">
            {outcome.problems.map((line, i) => (
              <li key={i} className="text-[11px] font-mono leading-snug text-rose-200/80">
                {line}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
