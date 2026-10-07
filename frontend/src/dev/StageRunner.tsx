// Stage suite: a workbench for the scenario library. View, edit, run a few, and read every LLM
// call of the run.
//
// Split with Prompt replay: replay edits the prompt text recorded in a trace (code changes don't
// affect it); this re-runs the production prompt builder, so it tests the code.
//
// Don't add "preview the new prompt without running": a mid-pipeline stage's prompt is built
// from upstream LLM output (`run_decision` = `_build_internal_context` -> `decide`), and stubbing
// the upstream yields something structurally right and entirely fake. The only zero-cost view is
// what the last run left on disk (timestamped).
//
// Edits land directly in the git-tracked `tuning/scenarios/<stage>.json`. Don't save a separate
// draft: the CLI and CI read that file and would never see it.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";

import CallCard, { Labeled } from "./CallCard";
import { devApi } from "./api";
import type {
  PromptMessage,
  StageCatalog,
  StageReport,
  StageScenario,
  StageSpec,
  TraceCall,
} from "./types";
import {
  AlphaTag,
  Collapse,
  Empty,
  ErrorLine,
  Field,
  JobLog,
  JsonBlock,
  Mono,
  btn,
  btnGhost,
  input,
  panel,
  prettyJson,
  useJob,
} from "./ui";

/** Any file in a scenario directory beyond these common artifacts is shown as raw JSON. Each
 *  stage also writes its own payload (event.json / interrupt.json …); enumerating them would
 *  miss whichever one is added next. */
const WELL_KNOWN = new Set(["llm_calls", "prompt", "checks", "judge", "input"]);

/** Sentinel for "being created, not yet on disk". It contains a space, so it can't collide with
 * a valid scenario name (names don't allow spaces). */
const NEW = "new scenario";

const SKELETON = {
  description: "",
  criteria_focus: [] as string[],
  expect: "",
  scenario: {},
};

/** Split a scenario entry into name and the rest: the name has strict rules and is the report
 * directory name, so it gets its own input. */
function splitEntry(entry: StageScenario): { name: string; body: string } {
  const { name, ...rest } = entry;
  return { name: String(name ?? ""), body: JSON.stringify(rest, null, 2) };
}

export default function StageRunner({ world }: { world: string }) {
  const modelKeys = useModelKeys();
  const [catalog, setCatalog] = useState<StageCatalog | null>(null);
  const [stageKey, setStageKey] = useState("");
  const [report, setReport] = useState<StageReport | null>(null);
  const [file, setFile] = useState<{
    path: string;
    scenarios: StageScenario[];
    extras: string[];
  } | null>(null);

  // The entry being edited (keyed by its on-disk name; NEW = a new draft) + two edit buffers.
  const [active, setActive] = useState("");
  const [nameDraft, setNameDraft] = useState("");
  const [bodyDraft, setBodyDraft] = useState("");
  // Checked = which ones to run. That's independent of which one is being edited, so one uses
  // checkboxes and the other a row click, and they look different too.
  const [checked, setChecked] = useState<Set<string>>(new Set());
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState("");

  useEffect(() => {
    devApi
      .stages()
      .then((c) => {
        setCatalog(c);
        setStageKey((k) => k || c.stages[0]?.key || "");
      })
      .catch(() => setCatalog({ stages: [], runs: {} }));
  }, []);

  const spec: StageSpec | undefined = useMemo(
    () => catalog?.stages.find((s) => s.key === stageKey),
    [catalog, stageKey],
  );

  // Each fetch takes a ticket; a response whose ticket is stale is dropped, since responses can
  // come back out of order.
  //
  // For loadFile this is more than cosmetic: with stage A's scenarios on screen while stageKey is
  // already B, Save would PUT to `…/stages/B/scenarios/<A's scenario name>` and corrupt a
  // git-tracked corpus file.
  const ticket = useRef(0);

  // A missing report is the normal case (this world hasn't run this stage yet), so it's just
  // null, not an error.
  const loadReport = useCallback(async () => {
    if (!stageKey) return;
    const mine = ++ticket.current;
    const next = await devApi.stageReport(stageKey, world).catch(() => null);
    if (mine === ticket.current) setReport(next);
  }, [stageKey, world]);

  const loadFile = useCallback(
    async (keep?: string) => {
      if (!stageKey) return;
      const mine = ++ticket.current;
      try {
        const d = await devApi.stageScenarios(stageKey);
        if (mine !== ticket.current) return;
        const { scenarios, ...rest } = d.file;
        const list = scenarios ?? [];
        setFile({ path: d.path, scenarios: list, extras: Object.keys(rest) });
        const target = keep && list.some((s) => s.name === keep) ? keep : "";
        setActive(target);
        if (target) {
          const split = splitEntry(list.find((s) => s.name === target)!);
          setNameDraft(split.name);
          setBodyDraft(split.body);
        } else {
          setNameDraft("");
          setBodyDraft("");
        }
      } catch (e) {
        if (mine !== ticket.current) return;
        setFile(null);
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [stageKey],
  );

  useEffect(() => {
    void loadReport();
  }, [loadReport]);

  useEffect(() => {
    // On a stage switch, clear the old stage's state right away instead of waiting for new data.
    // Otherwise a Save in the gap writes an old-stage scenario name into the new stage's file
    // (see ticket). The UI shows Loading while file is null.
    setFile(null);
    setActive("");
    setChecked(new Set());
    setError("");
    setConfirmDelete("");
    void loadFile();
  }, [loadFile]);

  const original = useMemo(
    () => (active && active !== NEW ? file?.scenarios.find((s) => s.name === active) : undefined),
    [file, active],
  );

  const dirty = useMemo(() => {
    if (active === NEW) return true;
    if (!original) return false;
    const split = splitEntry(original);
    return nameDraft !== split.name || bodyDraft.trim() !== split.body;
  }, [active, original, nameDraft, bodyDraft]);

  const open = (name: string) => {
    const s = file?.scenarios.find((x) => x.name === name);
    if (!s) return;
    const split = splitEntry(s);
    setActive(name);
    setNameDraft(split.name);
    setBodyDraft(split.body);
    setError("");
    setConfirmDelete("");
  };

  const startNew = (from?: StageScenario) => {
    const base = from ? { ...from } : { name: "", ...SKELETON };
    const split = splitEntry(base as StageScenario);
    setActive(NEW);
    setNameDraft(from ? `${from.name}_copy` : "");
    setBodyDraft(split.body);
    setError("");
    setConfirmDelete("");
  };

  /** Name + the rest, composed into a full scenario. If body also has a name, the input wins. */
  const composed = (): StageScenario | null => {
    try {
      const rest = bodyDraft.trim() ? JSON.parse(bodyDraft) : {};
      if (!rest || typeof rest !== "object" || Array.isArray(rest)) {
        setError("The scenario body must be a JSON object.");
        return null;
      }
      delete (rest as Record<string, unknown>).name;
      return { name: nameDraft.trim(), ...rest };
    } catch (e) {
      setError(`Not valid JSON — ${e instanceof Error ? e.message : String(e)}`);
      return null;
    }
  };

  const guard = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const save = () => {
    const entry = composed();
    if (!entry) return;
    void guard(async () => {
      const res =
        active === NEW
          ? await devApi.createScenario(stageKey, entry)
          : await devApi.saveScenario(stageKey, active, entry);
      await loadFile(res.name);
    });
  };

  const remove = (name: string) =>
    guard(async () => {
      await devApi.deleteScenario(stageKey, name);
      setChecked((c) => {
        const next = new Set(c);
        next.delete(name);
        return next;
      });
      setConfirmDelete("");
      await loadFile();
    });

  const { job, error: jobError, start, running } = useJob(loadReport);
  const picked = useMemo(
    () => (file?.scenarios ?? []).map((s) => s.name).filter((n) => checked.has(n)),
    [file, checked],
  );
  const runCount = picked.length || file?.scenarios.length || 0;

  const activeFiles = active && active !== NEW ? report?.scenarios?.[active] : undefined;
  const activeRow = report?.summary?.scenarios?.find((r) => r.name === active);

  const toggle = (name: string) =>
    setChecked((c) => {
      const next = new Set(c);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  return (
    <div className="space-y-3">
      <div className={`${panel} px-4 py-3`}>
        <div className="mb-2 flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-200">
          Stage
          <AlphaTag />
          <span className="font-normal text-slate-500">
            (restore this world, inject the scenario, then the same production cognition path; the
            prompt is reassembled by the code as it stands now)
          </span>
        </div>
        <p className="mb-2 max-w-4xl text-[11px] leading-relaxed text-amber-400/80">
          <strong className="font-semibold">
            This feature hasn't been carefully verified. Use it with caution.
          </strong>{" "}
          Alpha — the suite runs, but its scoring is still settling: not every stage judge has been
          brought onto the same deduction scale the audit uses, so scores are comparable within a
          stage over time, not across stages. Treat a number here as a signal to go read the run, not
          as a baseline to defend.
        </p>
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Stage">
            <select className={input} value={stageKey} onChange={(e) => setStageKey(e.target.value)}>
              {(catalog?.stages ?? []).map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label} ({s.key}){s.judged ? "" : " · deterministic only"}
                </option>
              ))}
            </select>
          </Field>
          {spec && (
            <span className="text-[11px] text-slate-500">
              Criteria: {spec.criteria.length ? spec.criteria.join(" · ") : "(no judge)"}
            </span>
          )}
          {file && (
            <span className="ml-auto text-[11px] text-slate-600">
              <span className="font-mono">{file.path}</span> · {file.scenarios.length} scenarios
              {file.extras.length ? ` · also carries ${file.extras.join(", ")}` : ""}
            </span>
          )}
        </div>
      </div>

      {!file ? (
        <Empty>Loading scenarios…</Empty>
      ) : (
        <>
          <div className={`${panel} flex flex-wrap items-center gap-3 px-4 py-3`}>
            <button
              className={btn}
              disabled={running || runCount === 0 || !modelKeys}
              title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
              onClick={() =>
                start(() =>
                  devApi.runStage(
                    stageKey,
                    picked.length
                      ? { world_id: world, scenario_names: picked }
                      : { world_id: world },
                  ),
                )
              }
            >
              {running
                ? "Running…"
                : picked.length
                  ? `Run ${picked.length} selected`
                  : `Run all ${runCount}`}
            </button>
            <span className="text-[11px] text-slate-500">
              {runCount} scenario{runCount === 1 ? "" : "s"} = {runCount} stage run
              {runCount === 1 ? "" : "s"}
              {spec?.judged ? ` + ${runCount} judge call${runCount === 1 ? "" : "s"}` : ""}
            </span>
            <code className="ml-auto font-mono text-[11px] text-slate-600">
              python -m tuning validate {stageKey} {world}
              {picked.length ? ` --scenario ${picked.join(",")}` : ""}
            </code>
          </div>
          <ErrorLine error={jobError} />
          <JobLog job={job} />

          <div className="flex gap-3">
            <ScenarioList
              scenarios={file.scenarios}
              active={active}
              checked={checked}
              report={report}
              onOpen={open}
              onToggle={toggle}
              onAll={(on) => setChecked(on ? new Set(file.scenarios.map((s) => s.name)) : new Set())}
              onNew={() => startNew()}
            />
            <div className="min-w-0 flex-1 space-y-3">
              {!active ? (
                <Empty>Pick a scenario to see what it injects, edit it, or add a new one.</Empty>
              ) : (
                <>
                  <Editor
                    isNew={active === NEW}
                    original={original}
                    nameDraft={nameDraft}
                    setNameDraft={setNameDraft}
                    bodyDraft={bodyDraft}
                    setBodyDraft={setBodyDraft}
                    dirty={dirty}
                    busy={busy}
                    error={error}
                    filePath={file.path}
                    confirmDelete={confirmDelete === active}
                    onSave={save}
                    onRevert={() => (active === NEW ? setActive("") : open(active))}
                    onDuplicate={() => original && startNew(original)}
                    onAskDelete={() => setConfirmDelete(active)}
                    onCancelDelete={() => setConfirmDelete("")}
                    onDelete={() => void remove(active)}
                  />
                  {active !== NEW && (
                    <LastRun
                      files={activeFiles}
                      row={activeRow}
                      generatedAt={report?.summary?.generated_at}
                      fresh={job?.status === "completed"}
                    />
                  )}
                </>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ScenarioList({
  scenarios,
  active,
  checked,
  report,
  onOpen,
  onToggle,
  onAll,
  onNew,
}: {
  scenarios: StageScenario[];
  active: string;
  checked: Set<string>;
  report: StageReport | null;
  onOpen: (name: string) => void;
  onToggle: (name: string) => void;
  onAll: (on: boolean) => void;
  onNew: () => void;
}) {
  const all = scenarios.length > 0 && scenarios.every((s) => checked.has(s.name));
  return (
    <aside className="w-72 shrink-0 space-y-1">
      <div className="flex items-center gap-2 px-1 pb-1">
        <input
          type="checkbox"
          checked={all}
          onChange={(e) => onAll(e.target.checked)}
          className="accent-indigo-500"
          title="Select all / none"
        />
        <span className="text-[10px] uppercase tracking-wide text-slate-500">
          {checked.size ? `${checked.size} selected` : "select to run"}
        </span>
        <button className={`${btnGhost} ml-auto`} onClick={onNew}>
          + New
        </button>
      </div>
      {scenarios.map((s) => {
        const row = report?.summary?.scenarios?.find((r) => r.name === s.name);
        const bad = row && (!row.deterministic_passed || row.issues.length > 0);
        return (
          <div
            key={s.name}
            className={`flex gap-2 rounded-lg border px-2 py-2 transition ${
              active === s.name
                ? "border-indigo-500/40 bg-indigo-600/15"
                : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
            }`}
          >
            <input
              type="checkbox"
              checked={checked.has(s.name)}
              onChange={() => onToggle(s.name)}
              className="mt-0.5 accent-indigo-500"
            />
            <button onClick={() => onOpen(s.name)} className="min-w-0 flex-1 text-left">
              <div className="flex items-center gap-2">
                <span className="truncate text-xs font-medium text-slate-200">{s.name}</span>
                {row && (
                  <span
                    className={`ml-auto shrink-0 text-[10px] ${
                      bad ? "text-amber-400" : "text-emerald-500"
                    }`}
                  >
                    {bad ? "issues" : "ran"}
                  </span>
                )}
              </div>
              {s.description && (
                <div className="mt-0.5 text-[10px] leading-snug text-slate-500">{s.description}</div>
              )}
              {!!s.criteria_focus?.length && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {s.criteria_focus.map((c) => (
                    <span
                      key={c}
                      className="rounded border border-slate-800 px-1 text-[9px] text-slate-500"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              )}
            </button>
          </div>
        );
      })}
    </aside>
  );
}

/**
 * Scenario editor = a name input + a JSON textarea. The name is separate because it is also the
 * report directory name (`validation/<stage>/<name>/`), so a rename should be visible.
 *
 * Don't build a per-stage form for the rest: the `scenario` keys differ per stage, so that's
 * fourteen schemas to keep in sync with phase_harness, and any lag silently drops fields.
 */
function Editor({
  isNew,
  original,
  nameDraft,
  setNameDraft,
  bodyDraft,
  setBodyDraft,
  dirty,
  busy,
  error,
  filePath,
  confirmDelete,
  onSave,
  onRevert,
  onDuplicate,
  onAskDelete,
  onCancelDelete,
  onDelete,
}: {
  isNew: boolean;
  original?: StageScenario;
  nameDraft: string;
  setNameDraft: (v: string) => void;
  bodyDraft: string;
  setBodyDraft: (v: string) => void;
  dirty: boolean;
  busy: boolean;
  error: string;
  filePath: string;
  confirmDelete: boolean;
  onSave: () => void;
  onRevert: () => void;
  onDuplicate: () => void;
  onAskDelete: () => void;
  onCancelDelete: () => void;
  onDelete: () => void;
}) {
  return (
    <div className={`${panel} px-4 py-3`}>
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold text-slate-200">
          {isNew ? "New scenario" : "Scenario"}
        </span>
        <span className="text-[11px] text-slate-500">
          what gets injected into the restored world before the stage runs
        </span>
        {dirty && !isNew && (
          <span className="rounded border border-amber-700 bg-amber-950/40 px-1.5 text-[10px] text-amber-400">
            unsaved
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <Field label="Name (also the report directory)">
          <input
            className={`${input} w-64 font-mono`}
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            placeholder="letters, digits, _ . -"
          />
        </Field>
        <button className={btn} disabled={busy || !dirty} onClick={onSave}>
          {busy ? "Saving…" : isNew ? "Create" : "Save"}
        </button>
        <button className={btnGhost} disabled={busy || !dirty} onClick={onRevert}>
          {isNew ? "Discard" : "Revert"}
        </button>
        {!isNew && (
          <button className={btnGhost} disabled={busy} onClick={onDuplicate}>
            Duplicate
          </button>
        )}
        {!isNew &&
          (confirmDelete ? (
            <span className="flex items-center gap-2">
              <button
                className="rounded-lg border border-rose-700 bg-rose-950/50 px-3 py-1.5 text-xs font-semibold text-rose-300 hover:bg-rose-900/50"
                disabled={busy}
                onClick={onDelete}
              >
                Delete for real
              </button>
              <button className={btnGhost} onClick={onCancelDelete}>
                Cancel
              </button>
            </span>
          ) : (
            <button className={btnGhost} disabled={busy} onClick={onAskDelete}>
              Delete
            </button>
          ))}
        <span className="ml-auto text-[11px] text-slate-600">
          saves into <span className="font-mono">{filePath}</span> — git tracks it
        </span>
      </div>

      {original?.expect && (
        <div className="mt-2">
          <Labeled label="expect — what the judge is told to look for">
            <Mono className="max-h-28">{original.expect}</Mono>
          </Labeled>
        </div>
      )}
      <textarea
        value={bodyDraft}
        onChange={(e) => setBodyDraft(e.target.value)}
        spellCheck={false}
        className={`${input} mt-2 w-full font-mono leading-relaxed`}
        style={{ minHeight: 300 }}
      />
      <ErrorLine error={error} />
    </div>
  );
}

/**
 * What this scenario produced on its last run: every LLM call, deterministic checks, judge scores.
 * The title carries the run's timestamp ("This run" after a run): after a code change without a
 * re-run, the prompt shown came from the old code.
 */
function LastRun({
  files,
  row,
  generatedAt,
  fresh,
}: {
  files?: Record<string, unknown>;
  row?: {
    scores: Record<string, number>;
    overall?: string;
    deterministic_passed: boolean;
    issues: string[];
  };
  generatedAt?: string;
  fresh?: boolean;
}) {
  if (!files) {
    return (
      <Empty>
        This scenario has not been run against this world yet — hit “Run” to see every LLM call the
        current code makes for it.
      </Empty>
    );
  }
  const calls = (files.llm_calls as TraceCall[] | undefined) ?? [];
  const prompt = files.prompt as Record<string, unknown> | undefined;
  return (
    <div className={`${panel} px-4 py-3`}>
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <span className="text-xs font-semibold text-slate-200">
          {fresh ? "This run" : "Last run"}
        </span>
        {!fresh && generatedAt && (
          <span className="text-[11px] text-amber-500/80">
            {generatedAt} — reassembled only when you run again
          </span>
        )}
        {row &&
          Object.entries(row.scores).map(([k, v]) => (
            <span key={k} className="text-[11px] text-slate-500">
              {k} <b className="text-slate-200">{v}</b>
            </span>
          ))}
        {row && !row.deterministic_passed && (
          <span className="text-[11px] text-rose-400">deterministic failed</span>
        )}
      </div>
      {row?.overall && <div className="mb-2 text-xs text-slate-400">{row.overall}</div>}
      {!!row?.issues.length && (
        <ul className="mb-2 list-disc space-y-0.5 pl-5 text-[11px] text-amber-400">
          {row.issues.map((i, n) => (
            <li key={n}>{i}</li>
          ))}
        </ul>
      )}

      {/* Every LLM call this run made: upstream cognition, this stage's own, and the final judge.
          Uses the trace page's renderer (the on-disk record shape is the same), so scene / stage
          / agent / tokens / thinking / failure markers come for free. */}
      {calls.length > 0 && (
        <div className="mb-3">
          <div className="mb-1 text-[11px] uppercase tracking-wide text-indigo-300">
            Every LLM call in this run ({calls.length}) — upstream cognition, this stage, then the
            judge
          </div>
          <div className="space-y-1">
            {calls.map((c, i) => (
              <CallCard key={c.call_id || i} call={c} open={calls.length === 1} />
            ))}
          </div>
        </div>
      )}

      {prompt && <PromptBlock payload={prompt} />}
      <div className="mt-2 space-y-1">
        {Object.entries(files)
          .filter(([stem]) => stem !== "prompt" && stem !== "llm_calls")
          .map(([stem, value]) => (
            <details
              key={stem}
              className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-1.5"
            >
              <summary className="cursor-pointer text-[11px] text-slate-400">
                {stem}
                {WELL_KNOWN.has(stem) ? "" : " (stage-specific payload)"}
              </summary>
              <JsonBlock value={value} />
            </details>
          ))}
      </div>
    </div>
  );
}

/**
 * `prompt.json` = this stage's own calls, grouped however each suite chooses (by agent, goal or
 * event), so the shape varies. No fixed key names here: any object with `prompt` (a message array)
 * and `response` renders as a call; everything else is shown raw. For all calls, see the section
 * above.
 */
function PromptBlock({ payload }: { payload: Record<string, unknown> }) {
  const calls = collectCalls(payload);
  if (!calls.length) return <JsonBlock value={payload} />;
  return (
    <div className="space-y-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">
        This stage&rsquo;s own call{calls.length === 1 ? "" : "s"} ({calls.length})
      </div>
      {calls.map((c, i) => (
        <Collapse
          key={i}
          title={
            <span className="text-xs text-slate-200">
              {c.label || `call #${i + 1}`}
              <span className="ml-2 text-[10px] text-slate-500">
                {c.prompt.map((m) => m.role).join(" + ")} → response
              </span>
            </span>
          }
        >
          <div className="space-y-1.5">
            {c.prompt.map((m, n) => (
              <Labeled key={n} label={m.role}>
                <Mono className="max-h-96">{m.content}</Mono>
              </Labeled>
            ))}
            <Labeled label="response">
              <Mono accent className="max-h-96">
                {prettyJson(c.response) || "—"}
              </Mono>
            </Labeled>
          </div>
        </Collapse>
      ))}
    </div>
  );
}

interface FoundCall {
  label: string;
  prompt: PromptMessage[];
  response: string;
}

function collectCalls(node: unknown, label = ""): FoundCall[] {
  if (Array.isArray(node)) return node.flatMap((n, i) => collectCalls(n, label || `#${i + 1}`));
  if (!node || typeof node !== "object") return [];
  const obj = node as Record<string, unknown>;
  if (Array.isArray(obj.prompt) && obj.prompt.every((m) => m && typeof m === "object" && "role" in m)) {
    const who = obj.agent_name ?? obj.agent_id ?? obj.target_name ?? obj.name ?? label;
    return [
      {
        label: String(who ?? ""),
        prompt: obj.prompt as PromptMessage[],
        response: String(obj.response ?? ""),
      },
    ];
  }
  return Object.entries(obj).flatMap(([k, v]) => collectCalls(v, k));
}
