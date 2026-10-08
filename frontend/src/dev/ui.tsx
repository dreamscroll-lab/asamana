// Shared primitives for the developer tool. The three tabs read very differently
// (a call list, a score card, a prompt bench) but they are the same instrument panel
// underneath, so the chrome lives here instead of being re-typed per tab.

import { useCallback, useEffect, useRef, useState } from "react";

import { devApi } from "./api";
import type { CallAggregate, DevJob } from "./types";

export const panel = "rounded-xl border border-slate-800 bg-slate-900/60";
export const input =
  "bg-slate-950/70 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 " +
  "placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/70";
export const btn =
  "px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 border border-indigo-500 " +
  "text-white hover:bg-indigo-500 transition disabled:opacity-40 disabled:cursor-not-allowed";
export const btnGhost =
  "px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-900 border border-slate-800 " +
  "text-slate-300 hover:border-slate-700 hover:text-slate-100 transition disabled:opacity-40";

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wide text-slate-500">{label}</span>
      {children}
    </label>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950/60 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`px-3 py-1 rounded-md text-xs transition ${
            value === o.value
              ? "bg-indigo-600/25 text-indigo-200 border border-indigo-500/40"
              : "text-slate-400 hover:text-slate-200 border border-transparent"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, bad }: { label: string; value: React.ReactNode; bad?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className={`text-base font-semibold ${bad ? "text-rose-400" : "text-slate-100"}`}>
        {value}
      </span>
      <span className="text-[10px] uppercase tracking-wide text-slate-500">{label}</span>
    </div>
  );
}

export const fmt = (n: number | null | undefined) => (n ?? 0).toLocaleString();
export const pct = (r: number | null | undefined) => `${((r ?? 0) * 100).toFixed(1)}%`;

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 90) return `${s.toFixed(1)} s`;
  return `${(s / 60).toFixed(1)} min`;
}

/** LLM JSON arrives as one unbroken line; pretty-print it so line breaks exist to render.
 *  Falls back to the raw text whenever it is not valid JSON — a truncated response must
 *  stay visible exactly as received, which is the whole point of looking at it. */
export function prettyJson(text: string | null | undefined): string {
  if (typeof text !== "string" || !text.trim()) return text ?? "";
  const body = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
  try {
    return JSON.stringify(JSON.parse(body), null, 2);
  } catch {
    /* fall through to the brace-slice attempt */
  }
  const start = body.search(/[{[]/);
  if (start >= 0) {
    const close = body[start] === "{" ? "}" : "]";
    const end = body.lastIndexOf(close);
    if (end > start) {
      try {
        return JSON.stringify(JSON.parse(body.slice(start, end + 1)), null, 2);
      } catch {
        /* not JSON after all */
      }
    }
  }
  return text;
}

export function Mono({
  children,
  accent,
  className = "",
}: {
  children: React.ReactNode;
  accent?: boolean;
  className?: string;
}) {
  return (
    <pre
      className={`whitespace-pre-wrap break-words rounded-lg border px-3 py-2 text-[11px] leading-relaxed
        font-mono text-slate-300 bg-slate-950/80 overflow-auto
        ${accent ? "border-indigo-500/40" : "border-slate-800"} ${className}`}
    >
      {children}
    </pre>
  );
}

export function JsonBlock({ value, max = "max-h-72" }: { value: unknown; max?: string }) {
  return <Mono className={max}>{JSON.stringify(value, null, 2)}</Mono>;
}

/** A tool that isn't settled: its measures may change, so don't treat its numbers as a baseline. */
export function AlphaTag({ title }: { title?: string }) {
  return (
    <span
      title={title ?? "Alpha — still settling. Scores may shift as the criteria change."}
      className="rounded border border-amber-700/70 bg-amber-950/40 px-1 py-px text-[9px] font-semibold uppercase tracking-wide text-amber-400"
    >
      alpha
    </span>
  );
}

/** A labeled, collapsible section. Native <details> — the dev tool has no need for
 *  a controlled accordion, and native keeps keyboard/find-in-page working. */
export function Collapse({
  title,
  children,
  open,
  tone = "default",
}: {
  title: React.ReactNode;
  children: React.ReactNode;
  open?: boolean;
  tone?: "default" | "bad" | "warn";
}) {
  const edge =
    tone === "bad"
      ? "border-l-2 border-l-rose-500/70"
      : tone === "warn"
        ? "border-l-2 border-l-amber-500/70"
        : "";
  return (
    <details open={open} className={`${panel} ${edge} mb-2 overflow-hidden`}>
      <summary className="cursor-pointer list-none px-3 py-2 text-xs hover:bg-slate-800/40">
        {title}
      </summary>
      <div className="px-3 pb-3 pt-1">{children}</div>
    </details>
  );
}

export function Empty({ children }: { children: React.ReactNode }) {
  return <div className="py-10 text-center text-xs text-slate-500">{children}</div>;
}

export function ErrorLine({ error }: { error: string }) {
  return error ? <div className="text-[11px] text-rose-400">✗ {error}</div> : null;
}

/** 0–100 scale, monotone: high green / mid amber / low red. `null` = not scored
 *  (a judge failure or an unrun scope) — which is NOT the same as a legitimate 0. */
export function scoreTone(s: number | null | undefined) {
  if (s == null) return "bg-slate-700 text-slate-300";
  if (s >= 85) return "bg-emerald-600 text-emerald-50";
  if (s >= 70) return "bg-amber-600 text-amber-50";
  return "bg-rose-700 text-rose-50";
}

export function Score({ value, big }: { value: number | null | undefined; big?: boolean }) {
  const txt = value == null ? "N/A" : big ? Number(value).toFixed(1) : String(Math.round(value));
  return (
    <span
      className={`inline-block rounded font-bold ${scoreTone(value)} ${
        big ? "px-3 py-1 text-xl" : "px-2 py-0.5 text-[11px]"
      }`}
    >
      {txt}
    </span>
  );
}

export function AggregateBand({
  label,
  total,
  extra,
}: {
  label: string;
  total: CallAggregate;
  extra?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-8 gap-y-3 rounded-xl border border-indigo-500/30 bg-indigo-950/20 px-4 py-3">
      <span className="text-[10px] font-semibold uppercase tracking-widest text-indigo-300">
        {label}
      </span>
      <Stat label="LLM calls" value={fmt(total.calls)} />
      <Stat
        label="call failures"
        value={`${fmt(total.failures)} (${pct(total.failure_rate)})`}
        bad={total.failures > 0}
      />
      <Stat
        label="parse failures"
        value={`${fmt(total.parse_failures)} (${pct(total.parse_failure_rate)})`}
        bad={total.parse_failures > 0}
      />
      <Stat
        label="discarded"
        value={`${fmt(total.discarded)} (${pct(total.discard_rate)})`}
        bad={total.discarded > 0}
      />
      <Stat label="input tokens" value={fmt(total.input_tokens)} />
      <Stat label="output tokens" value={fmt(total.output_tokens)} />
      <Stat label="LLM latency" value={fmtMs(total.latency_ms)} />
      {total.wall_ms != null && <Stat label="wall time" value={fmtMs(total.wall_ms)} />}
      {extra}
    </div>
  );
}

/**
 * Poll one backgrounded `python -m tuning …` job until it stops running.
 *
 * The job writes its report to disk and the client re-fetches on completion, so the
 * only thing streamed is the log tail — which is also the only way to see WHY a run is
 * slow (it is a judge call per scenario, and the child prints as it goes).
 */
export function useJob(onDone: () => void) {
  const [job, setJob] = useState<DevJob | null>(null);
  const [error, setError] = useState("");
  const timer = useRef<number>();
  const done = useRef(onDone);
  done.current = onDone;
  // Cleanup can't just cancel the scheduled timeout: if unmount races an in-flight devApi.job(),
  // its continuation schedules another one, so polling keeps running on a dead component and
  // finally calls done() to setState. Hence a "stop" flag that the continuation checks itself.
  const stopped = useRef(false);

  useEffect(() => {
    stopped.current = false;
    return () => {
      stopped.current = true;
      window.clearTimeout(timer.current);
    };
  }, []);

  const poll = useCallback((jobId: string) => {
    const tick = async () => {
      try {
        const j = await devApi.job(jobId);
        if (stopped.current) return;
        setJob(j);
        if (j.status === "running") {
          timer.current = window.setTimeout(tick, 2000);
        } else {
          done.current();
        }
      } catch (e) {
        if (stopped.current) return;
        setError(String(e instanceof Error ? e.message : e));
      }
    };
    void tick();
  }, []);

  const start = useCallback(
    async (launch: () => Promise<DevJob>) => {
      setError("");
      try {
        const j = await launch();
        setJob(j);
        poll(j.job_id);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [poll],
  );

  return { job, error, start, running: job?.status === "running" };
}

export function JobLog({ job }: { job: DevJob | null }) {
  if (!job) return null;
  const tone =
    job.status === "failed" ? "text-rose-400" : job.status === "completed" ? "text-emerald-400" : "text-indigo-300";
  return (
    <div className="mt-2">
      <div className={`text-[11px] ${tone}`}>
        {job.status === "running" ? "Running…" : job.status === "completed" ? "✓ Done" : `✗ Failed (rc=${job.returncode})`}
        <span className="ml-2 font-mono text-slate-500">{job.cmd}</span>
      </div>
      {job.log_tail.length > 0 && <Mono className="mt-1 max-h-44">{job.log_tail.join("\n")}</Mono>}
    </div>
  );
}
