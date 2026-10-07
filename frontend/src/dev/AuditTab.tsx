// Audit: reads traces after the fact and has an LLM judge score the whole world across five
// scopes x a fixed set of director dimensions.
//
// This screen only audits: run panel on top, report below. Prompt-tuning tools don't belong
// here; they live in the debug tab. Mixing them in would make one control do two jobs.

import { useCallback, useEffect, useState } from "react";

import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";

import { devApi } from "./api";
import type {
  AuditDimensionCard,
  AuditEntry,
  AuditMetric,
  AuditReport,
  PromptMessage,
} from "./types";
import {
  Collapse,
  Empty,
  ErrorLine,
  Field,
  JobLog,
  Mono,
  Score,
  btn,
  btnGhost,
  input,
  panel,
  prettyJson,
  useJob,
} from "./ui";

/**
 * What each of the five scopes is for, shared by the run panel's checkboxes and the report's
 * section headings. A scope should carry only dimensions it is the only witness of: dimension
 * scores are averaged across scopes, so one it can't see clearly adds noise, not evidence.
 *
 * Don't read names from the report's `scopes_meta[k].label`: it is frozen into summary.json at
 * audit time, so a rename would show old and new names side by side. Only a key not listed here
 * falls back to it.
 *
 * Order = what to tick first: cheap ones first, per-cell fan-outs last (`costly`). `cost` shows
 * the judge-call fan-out, since one wrong tick is hundreds of LLM calls. `alias` is the CLI's
 * `--scope` name.
 */
const SCOPE_ROWS: {
  alias: string;
  key: string;
  label: string;
  what: string;
  only: string;
  cost: string;
  costly?: boolean;
  caveat?: string;
}[] = [
  {
    alias: "init",
    key: "initialization",
    label: "Initialization",
    what: "The starting world, before anything runs",
    only: "The only look at the world at rest \u2014 do the personas, the relationships and the backstory timeline fit together?",
    cost: "1 judge call total",
  },
  {
    alias: "sams",
    key: "single_agent_multi_step",
    label: "1 agent · N steps",
    what: "One agent, measured against their earlier self",
    only: "The only view of a single agent changing over time: how mood and goals shift, whether they spin in circles, slip out of character, or genuinely grow.",
    cost: "1 judge call per agent",
  },
  {
    alias: "mams",
    key: "multi_agent_multi_step",
    label: "N agents · N steps",
    what: "The whole run, read as a story",
    only: "The only view of where it is all heading \u2014 does it converge, does it build, does it surprise, does anyone come back from the dead or time run backwards. Reads what cognition produced (mood, goals, relationships), never how it got there.",
    cost: "1 judge call total",
  },
  {
    alias: "sass",
    key: "single_agent_single_step",
    label: "1 agent · 1 step",
    what: "A single act of thinking, output against its own input",
    only: "The only scope that sees what each call was actually handed, so it alone can catch a fact the model made up. Audits the machinery, not the story \u2014 scored on its own terms, and kept out of the world score.",
    cost: "1 judge call per agent × step",
    costly: true,
  },
  {
    alias: "mass",
    key: "multi_agent_single_step",
    label: "N agents · 1 step",
    what: "One moment, across the cast: who knows what",
    only: "The only view of what each agent actually perceived, so it alone can catch information leaking between them, a reaction aimed at the wrong thing, or two actions that cannot both be happening.",
    cost: "1 judge call per step",
    costly: true,
    caveat:
      "The window is a single step, but actions often run across several: a decision here may have nothing after it, a result nothing before it. Neither means nothing happened. It is also why this scope leaves effectiveness to the wider ones and scores only what one step can settle \u2014 who knew what, and what conflicts.",
  },
];

const SCOPE_BY_KEY: Record<string, (typeof SCOPE_ROWS)[number]> = Object.fromEntries(
  SCOPE_ROWS.map((r) => [r.key, r]),
);

const labelOf = (key: string, fallback?: string) => SCOPE_BY_KEY[key]?.label ?? fallback ?? key;
const aliasOf = (key: string) => SCOPE_BY_KEY[key]?.alias ?? key;

const pct = (v: number) => `${Math.round(v * 100)}%`;

const WEIGHT_NOTE =
  "A dimension pulls its designed weight only when all its evidence is in. Half the witnesses " +
  "missing, half the pull — the estimate itself is not discounted, the confidence in it is.";

const FIDELITY_NOTE =
  "Whether each call reasoned from what it was actually handed. A world can be flawless here and dull, " +
  "or invent a few facts and still tell a good story — so it stands beside the world score, never inside it.";

/** "sams 75% · mams 25%" — who supplied this dimension's score, after normalising over present scopes. */
const sourcesOf = (d: AuditDimensionCard) => {
  const parts = Object.entries(d.sources ?? {}).map(
    ([k, v]) => `${aliasOf(k)} ${pct(v.share)} (scored ${v.score}, n=${v.samples})`,
  );
  // Three reasons a scope contributes nothing, each calling for something different:
  // never run (silent here), judged under an older metric set (re-run it), or asked and
  // declined (go look at the criteria). Collapsing them hides which one you are looking at.
  const quiet = (d.abstained ?? []).map(aliasOf);
  const outdated = (d.stale ?? []).map(aliasOf);
  return [
    parts.length ? parts.join(" · ") : "not audited",
    outdated.length ? `never put to: ${outdated.join(", ")} (older metric set)` : "",
    quiet.length ? `asked but declined: ${quiet.join(", ")}` : "",
  ]
    .filter(Boolean)
    .join("\n");
};

export default function AuditTab({ world }: { world: string }) {
  const [report, setReport] = useState<AuditReport | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    try {
      setReport(await devApi.audit(world));
    } catch (e) {
      setReport(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [world]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="space-y-4">
      <RunPanel world={world} onDone={load} />
      <ScoringNote />
      <ErrorLine error={error} />
      {report?.summary ? <Report report={report} /> : !error && <Empty>Loading audit report…</Empty>}
      {error && (
        <Empty>
          No audit report for this world yet. Use “Run audit” above, or the CLI:
          <div className="mt-2 font-mono text-slate-400">python -m tuning audit {world}</div>
        </Empty>
      )}
    </div>
  );
}

function RunPanel({ world, onDone }: { world: string; onDone: () => void }) {
  const modelKeys = useModelKeys();
  const [scopes, setScopes] = useState<string[]>(
    SCOPE_ROWS.filter((r) => !r.costly).map((r) => r.alias),
  );
  const [agents, setAgents] = useState("");
  const [steps, setSteps] = useState("");
  const [judgeModel, setJudgeModel] = useState("");
  const [judgeProvider, setJudgeProvider] = useState("");
  const [clearing, setClearing] = useState(false);
  const [clearError, setClearError] = useState("");
  const { job, error, start, running } = useJob(onDone);

  // Which judge and model this deployment uses is the backend's call; we only prefill it.
  // Leaving it blank runs the same model.
  useEffect(() => {
    let live = true;
    devApi
      .auditJudge()
      .then((j) => {
        if (!live) return;
        setJudgeProvider(j.provider);
        setJudgeModel(j.model);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  const toggle = (k: string) =>
    setScopes((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]));

  async function clearReport() {
    if (!window.confirm(`Delete the audit report for ${world}? The next run starts from scratch.`)) return;
    setClearError("");
    setClearing(true);
    try {
      await devApi.clearAudit(world);
      onDone();
    } catch (e) {
      setClearError(e instanceof Error ? e.message : String(e));
    } finally {
      setClearing(false);
    }
  }

  return (
    <div className={`${panel} px-4 py-3`}>
      <div className="mb-1 text-xs font-semibold text-slate-200">
        Run audit
        <span className="ml-2 font-normal text-slate-500">(spawns python -m tuning audit)</span>
      </div>
      <p className="mb-3 max-w-4xl text-[11px] leading-relaxed text-slate-500">
        Each scope checks <b className="text-slate-300">a different kind of consistency</b>, at a
        different scale. What makes a scope worth running is what it alone can see, so these are not
        coarse and fine settings of one dial — none stands in for another. Cost varies enormously: the
        first three are the usual pick (one judge call per agent, plus two for the world); the last two
        fan out per cell, adding roughly 280 and 40 calls on a 40-step, 7-agent world.
      </p>
      <div className="mb-3 space-y-1.5">
        {SCOPE_ROWS.map((r, i) => (
          <label
            key={r.alias}
            className={`flex cursor-pointer items-start gap-2 rounded-lg border px-2.5 py-1.5 transition ${
              scopes.includes(r.alias)
                ? "border-indigo-500/40 bg-indigo-600/10"
                : "border-slate-800 hover:border-slate-700"
            } ${r.costly && !SCOPE_ROWS[i - 1]?.costly ? "mt-3" : ""}`}
          >
            <input
              type="checkbox"
              checked={scopes.includes(r.alias)}
              onChange={() => toggle(r.alias)}
              className="mt-0.5 accent-indigo-500"
            />
            <span className="min-w-0 flex-1">
              <span className="flex flex-wrap items-center gap-2">
                <b className="text-xs text-slate-200">{r.label}</b>
                <span className="font-mono text-[10px] text-slate-600">{r.alias}</span>
                <span
                  className={`rounded border px-1.5 text-[10px] ${
                    r.costly
                      ? "border-amber-700 bg-amber-950/40 text-amber-400"
                      : "border-slate-800 text-slate-500"
                  }`}
                >
                  {r.costly ? "⚠ " : ""}
                  {r.cost}
                </span>
              </span>
              <span className="mt-0.5 block text-[11px] leading-snug text-slate-300">{r.what}</span>
              <span className="mt-0.5 block text-[11px] leading-snug text-slate-500">{r.only}</span>
              {r.caveat && (
                <span className="mt-1 block text-[11px] leading-snug text-amber-500/90">
                  ⚠ {r.caveat}
                </span>
              )}
            </span>
          </label>
        ))}
      </div>
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Agents (id or name, comma-separated)">
          <input className={`${input} w-48`} value={agents} onChange={(e) => setAgents(e.target.value)} />
        </Field>
        <Field label="Steps (1-10 / 3,5)">
          <input className={`${input} w-32`} value={steps} onChange={(e) => setSteps(e.target.value)} />
        </Field>
        {/* Vendor is read-only: endpoint and credentials hang off the provider declaration, so
            switching vendors is a deployment change, not a per-run one. */}
        <Field label="Judge vendor">
          <div className={`${input} w-32 cursor-default text-slate-400`}>{judgeProvider || "—"}</div>
        </Field>
        <Field label="Judge model">
          <input
            className={`${input} w-44`}
            placeholder="blank = configured default"
            value={judgeModel}
            onChange={(e) => setJudgeModel(e.target.value)}
          />
        </Field>
        <button
          className={btn}
          disabled={running || scopes.length === 0 || !modelKeys}
          title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
          onClick={() =>
            start(() =>
              devApi.runAudit(world, {
                scopes,
                agents: agents.trim() ? agents.split(",").map((s) => s.trim()).filter(Boolean) : undefined,
                steps: steps.trim() || undefined,
                judge_model: judgeModel.trim() || undefined,
              }),
            )
          }
        >
          {running ? "Auditing…" : "Run audit"}
        </button>
        {scopes.length === 0 && (
          <span className="text-[11px] text-slate-500">Pick at least one scope.</span>
        )}
        {/* Clear sits in the run panel because it's a pre-rerun step: per agent/step scores merge
            incrementally by key, and the world score re-aggregates every scope file on disk, so
            without clearing, the previous round's entries count toward this one. */}
        <button
          className={`${btnGhost} ml-auto hover:border-rose-800 hover:text-rose-300`}
          disabled={running || clearing}
          onClick={() => void clearReport()}
        >
          {clearing ? "Clearing\u2026" : "Clear report"}
        </button>
      </div>
      <ErrorLine error={error} />
      <ErrorLine error={clearError} />
      <JobLog job={job} />
    </div>
  );
}

function TierTag({ category }: { category: string }) {
  const basic = category === "basic";
  return (
    <span
      className={`ml-1 rounded border px-1 text-[9px] ${
        basic ? "border-amber-600 text-amber-500" : "border-indigo-500 text-indigo-300"
      }`}
    >
      {basic ? "basic" : "elevated"}
    </span>
  );
}

function DimPill({
  score,
  name,
  category,
  trailing,
  inspect,
}: {
  score: number | null | undefined;
  name: string;
  category: string;
  trailing?: string;
  inspect?: string;
}) {
  return (
    <span
      title={inspect}
      className="inline-flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1"
    >
      <Score value={score} />
      <span className="text-[11px] text-slate-200">{name}</span>
      <TierTag category={category} />
      {trailing && <span className="text-[10px] text-slate-500">{trailing}</span>}
    </span>
  );
}

/** Scope with unit=none: scores = {metricId: score}, one pill per dimension. */
function MetricRow({
  scores,
  metrics,
}: {
  scores: Record<string, number | null> | undefined;
  metrics: AuditMetric[];
}) {
  if (!metrics.length) return null;
  return (
    <div className="my-1.5 flex flex-wrap gap-2">
      {metrics.map((m) => (
        <DimPill
          key={m.id}
          score={scores?.[m.id]}
          name={m.name}
          category={m.category}
          trailing={`w${m.weight}`}
          inspect={m.inspect}
        />
      ))}
    </div>
  );
}

/** Scope with unit=stage/step: scores = {unit: {metricId: score}} -> score matrix. */
function MetricTable({
  scores,
  metrics,
}: {
  scores: Record<string, Record<string, number | null>> | undefined;
  metrics: AuditMetric[];
}) {
  const units = Object.keys(scores ?? {});
  if (!units.length) return <span className="text-xs text-slate-500">(not scored)</span>;
  return (
    <div className="overflow-x-auto">
      <table className="text-[11px]">
        <thead>
          <tr>
            <th />
            {metrics.map((m) => (
              <th key={m.id} title={m.inspect} className="px-1.5 pb-1 font-semibold text-slate-400">
                {m.name}
                <TierTag category={m.category} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {units.map((u) => (
            <tr key={u}>
              <td className="pr-3 text-slate-500">{u}</td>
              {metrics.map((m) => (
                <td key={m.id} className="px-1 py-0.5 text-center">
                  <Score value={scores?.[u]?.[m.id]} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Rationale({ text }: { text?: string }) {
  return (
    <div className="my-1 rounded-r border-l-2 border-indigo-500/60 bg-slate-950/50 px-2.5 py-1.5 text-xs text-slate-400">
      <b className="text-slate-200">Rationale</b>　{text || "—"}
    </div>
  );
}

function JudgeCall({ call }: { call?: { prompt: PromptMessage[]; response: string } }) {
  if (!call || (!call.prompt && !call.response)) return null;
  return (
    <details className="mt-2 border-t border-dashed border-slate-800 pt-1.5">
      <summary className="cursor-pointer text-[11px] text-slate-500">
        🔧 The LLM call behind this score (prompt + response)
      </summary>
      <div className="mt-1 space-y-1">
        {(call.prompt || []).map((m, i) => (
          <div key={i}>
            <div className="text-[10px] uppercase text-slate-500">{m.role}</div>
            <Mono className="max-h-72">{m.content}</Mono>
          </div>
        ))}
        <div className="text-[10px] uppercase text-slate-500">response</div>
        <Mono accent className="max-h-72">
          {prettyJson(call.response) || "—"}
        </Mono>
      </div>
    </details>
  );
}

/** Section header = name + purpose (same source as the run panel, see SCOPE_ROWS); scopes with
 * a window limitation get an inline caveat. */
function ScopeSection({
  scopeKey,
  total,
  children,
}: {
  scopeKey: string;
  total: number | null | undefined;
  children: React.ReactNode;
}) {
  const row = SCOPE_BY_KEY[scopeKey];
  const edge = total == null ? "border-l-slate-700" : total >= 85 ? "border-l-emerald-600" : total >= 70 ? "border-l-amber-600" : "border-l-rose-700";
  return (
    <section className={`${panel} border-l-4 ${edge} my-4 px-4 py-3`}>
      <h3 className="flex items-center gap-2 text-base font-bold text-slate-100">
        <Score value={total} />
        {labelOf(scopeKey)}
      </h3>
      {row && <p className="mb-2 mt-1 text-[11px] leading-snug text-slate-400">{row.what}</p>}
      {row?.caveat && (
        <p className="mb-2 rounded border border-amber-800/60 bg-amber-950/30 px-2 py-1 text-[11px] leading-snug text-amber-400/90">
          ⚠ {row.caveat}
        </p>
      )}
      {children}
    </section>
  );
}

/**
 * What a deduction score does and doesn't guarantee. Not inside the report: an unaudited world
 * would never show it, yet it's needed before running. Open by default: folded it's one line of
 * small text nobody notices.
 */
function ScoringNote() {
  return (
    <Collapse
      open
      title={
        <span className="text-slate-300">
          How scoring works — <b className="text-slate-100">everything here is a deduction</b>
        </span>
      }
    >
      <div className="max-w-4xl space-y-2.5 text-[11px] leading-relaxed text-slate-400">
        <p>
          Every dimension starts at <b className="text-slate-200">100</b> and can only lose points.
          The judge gets a list of named failure modes — <i>goldfish amnesia</i>,{" "}
          <i>invented facts</i>, <i>spinning in place</i>, <i>the most clichéd possible plot</i> — and
          deducts when it hits one. Nothing earns points back. Three reasons it is built this way:
        </p>
        <ol className="ml-4 list-decimal space-y-1.5">
          <li>
            <b className="text-slate-200">A judge is far better at catching a specific fault than at
            rating overall quality.</b>{" "}
            “Does this contradict what he did in step 12?” has an answer you can point at. “How good
            is this story out of 100?” does not — ask it and you get the model’s mood. Only the first
            kind of question gives a number that means the same thing twice.
          </li>
          <li>
            <b className="text-slate-200">It measures a floor, not a ceiling.</b> A high score is not
            a claim that the world is good. It is a claim that nothing visibly broke: no
            contradictions, nothing conjured out of thin air, nobody forgetting what they just did.
            That is the guarantee worth asking an automated check for.
          </li>
          <li>
            <b className="text-slate-200">It leaves the interesting part alone.</b> Above the floor,
            the judge has no opinion. A run is never marked down for failing to be clever — only for
            being flatly, demonstrably bad. Grading the ceiling would mean writing down in advance
            what a good story looks like, and then the system would be scored on how closely it
            imitated that. This project exists to find out what emerges, so the bar is “did anything
            actually go wrong”, and everything above it is left open.
          </li>
        </ol>
        <p className="border-t border-slate-800 pt-2">
          Two things follow, and you will see both in the report:{" "}
          <b className="text-slate-200">“can’t tell” is null, never zero</b> — a judge with nothing to
          go on abstains, and the dimension steps out of the weighting instead of dragging the score
          down. And <b className="text-slate-200">the criteria list faults, never model answers</b> —
          the judge is told outright not to deduct because the output differs from what it would have
          written.
        </p>
      </div>
    </Collapse>
  );
}

function Report({ report }: { report: AuditReport }) {
  const s = report.summary!;
  const meta = s.scopes_meta ?? {};
  const totals = s.scope_totals ?? {};
  const calls = report.calls ?? {};
  const metricsOf = (k: string) => meta[k]?.metrics ?? [];
  const unitOf = (k: string) => meta[k]?.unit ?? "none";
  const callOf = (k: string, entryKey: string) => calls[k]?.[entryKey];

  const body = (k: string, entry: AuditEntry | null | undefined) =>
    unitOf(k) === "none" ? (
      <MetricRow scores={entry?.scores as Record<string, number | null>} metrics={metricsOf(k)} />
    ) : (
      <MetricTable
        scores={entry?.scores as Record<string, Record<string, number | null>>}
        metrics={metricsOf(k)}
      />
    );

  const cards = (
    k: string,
    obj: Record<string, AuditEntry> | null | undefined,
    title: (key: string, e: AuditEntry) => string,
  ) =>
    Object.entries(obj ?? {}).map(([key, entry]) => (
      <Collapse
        key={key}
        title={
          <span className="flex items-center gap-2">
            <Score value={entry.total} />
            <b className="text-slate-100">{title(key, entry)}</b>
          </span>
        }
      >
        <Rationale text={entry.rationale} />
        {(entry.dropped ?? []).length > 0 && (
          <p className="my-1 text-[11px] text-amber-400/90">
            ⚠ Judge omitted {entry.dropped!.join(", ")} — the prompt asks for null, not a missing key.
          </p>
        )}
        {body(k, entry)}
        <JudgeCall call={callOf(k, key)} />
      </Collapse>
    ));

  const init = report.initialization;
  const mm = report.multi_agent_multi_step;

  return (
    <div>
      <div className="rounded-xl border border-indigo-500/30 bg-indigo-950/20 px-4 py-3">
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-[10px] font-semibold uppercase tracking-widest text-indigo-300">
            World score
          </span>
          <Score value={s.world_score} big />
          {s.coverage != null && (
            <span
              className="text-[11px] text-slate-400"
              title={`Share of the defined evidence weight that was actually audited. Two audits only compare at equal coverage. ${WEIGHT_NOTE}`}
            >
              evidence coverage {pct(s.coverage)}
            </span>
          )}
          <span className="h-5 w-px bg-slate-800" />
          <span className="text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            Fidelity
          </span>
          <Score value={s.fidelity_score} />
          <span className="text-[11px] text-slate-500" title={FIDELITY_NOTE}>
            engineering, not in world score
          </span>
          <span className="ml-auto text-[11px] text-slate-500">
            judge={s.judge_model || "?"} · {s.generated_at || ""}
          </span>
        </div>
        {(s.stale_scopes ?? []).length > 0 && (
          <p className="mt-2 rounded border border-amber-800/60 bg-amber-950/30 px-2 py-1 text-[11px] leading-snug text-amber-400/90">
            ⚠ Scored under an older metric set: {s.stale_scopes.map((k) => labelOf(k)).join(", ")}. Their scores
            still count, but the dimensions added since were never put to them — re-run those scopes.
          </p>
        )}
        {s.mixed_provenance && (
          <p className="mt-2 rounded border border-amber-800/60 bg-amber-950/30 px-2 py-1 text-[11px] leading-snug text-amber-400/90">
            ⚠ The scopes feeding this world score were judged by different models — incremental
            re-runs stitched their evidence together.
          </p>
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          {Object.entries(s.dimensions ?? {}).map(([id, d]) => (
            <DimPill
              key={id}
              score={d.score}
              name={d.name}
              category={d.category}
              trailing={
                d.coverage < 1
                  ? `${d.effective_weight}% of ${d.weight}% · cov ${pct(d.coverage)}`
                  : `${d.weight}%`
              }
              inspect={`${d.inspect}\n\nEvidence: ${sourcesOf(d)}`}
            />
          ))}
        </div>
        <div className="mt-3 border-t border-slate-800 pt-2 text-[11px] text-slate-500">
          Per-scope total (each over its own dimensions — not comparable across scopes):{" "}
          {Object.entries(meta)
            .map(([k, g]) => `${labelOf(k, g.label)} ${totals[k] ?? "N/A"}`)
            .join("　·　")}
        </div>
      </div>


      <ScopeSection scopeKey="initialization" total={totals.initialization}>
        {(init?.det ?? []).length > 0 && (
          <div className="my-1 text-xs text-amber-400">
            <b>Relation reciprocity precheck:</b>
            {(init?.det ?? []).map((f) => f.detail).join("；")}
          </div>
        )}
        <Rationale text={init?.rationale} />
        <MetricRow
          scores={init?.scores as Record<string, number | null>}
          metrics={metricsOf("initialization")}
        />
        <JudgeCall call={callOf("initialization", "_")} />
      </ScopeSection>

      <ScopeSection
        scopeKey="multi_agent_multi_step"
        total={totals.multi_agent_multi_step}
      >
        <Rationale text={mm?.rationale} />
        <MetricRow
          scores={mm?.scores as Record<string, number | null>}
          metrics={metricsOf("multi_agent_multi_step")}
        />
        <JudgeCall call={callOf("multi_agent_multi_step", "_")} />
      </ScopeSection>

      <ScopeSection
        scopeKey="multi_agent_single_step"
        total={totals.multi_agent_single_step}
      >
        {cards("multi_agent_single_step", report.multi_agent_single_step, (step) => `step ${step}`)}
      </ScopeSection>

      <ScopeSection
        scopeKey="single_agent_multi_step"
        total={totals.single_agent_multi_step}
      >
        {cards("single_agent_multi_step", report.single_agent_multi_step, (aid, e) => e.name || aid)}
      </ScopeSection>

      <ScopeSection
        scopeKey="single_agent_single_step"
        total={totals.single_agent_single_step}
      >
        {cards(
          "single_agent_single_step",
          report.single_agent_single_step,
          (_k, e) => `${e.name || e.agent} · step ${e.step}`,
        )}
      </ScopeSection>
    </div>
  );
}
