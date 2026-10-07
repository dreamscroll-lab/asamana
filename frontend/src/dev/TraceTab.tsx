// Trace: every LLM call a world made, plus their totals.
//
// Two orthogonal axes, one control each: Scope = which batch of calls (Runtime / Build); Group by
// = how that batch is bucketed (step / stage / agent, passed straight through as the backend's
// group_by). Build has no steps, its stage is always world_init and it has no agent, so it only
// groups by scene and those three filters don't apply.
//
// Don't send another axis's value through `groupBy`, and don't hide a filter based on the current
// grouping: either makes one control act as two axes.
//
// Call id and search narrow the current batch under either Scope; neither is a separate axis.
// Call id matches by prefix (you may only have 8 chars from a chip or log line); search needs
// every whitespace-separated word in the prompt / response / thinking / error / extra.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import CallCard, { isAnyFail } from "./CallCard";
import { devApi, type Grouping } from "./api";
import type { CallGroup, CallsPage, TraceDimensions } from "./types";
import {
  AggregateBand,
  Empty,
  ErrorLine,
  Field,
  Segmented,
  fmt,
  fmtMs,
  input,
  panel,
  pct,
} from "./ui";

type Scope = "runtime" | "build";

const SCOPES: { value: Scope; label: string }[] = [
  { value: "runtime", label: "Runtime" },
  { value: "build", label: "Build" },
];

const GROUP_BYS: { value: Grouping; label: string }[] = [
  { value: "step", label: "Step" },
  { value: "stage", label: "Stage" },
  { value: "agent", label: "Agent" },
];

export default function TraceTab({
  world,
  onPickCallId,
}: {
  world: string;
  onPickCallId: (id: string) => void;
}) {
  const [scope, setScope] = useState<Scope>("runtime");
  const [groupBy, setGroupBy] = useState<Grouping>("step");
  const [dims, setDims] = useState<TraceDimensions | null>(null);
  const [page, setPage] = useState<CallsPage | null>(null);
  const [step, setStep] = useState("");
  const [stage, setStage] = useState("");
  const [agentId, setAgentId] = useState("");
  const [callId, setCallId] = useState("");
  const [callIdQuery, setCallIdQuery] = useState("");
  const [keyword, setKeyword] = useState("");
  const [keywordQuery, setKeywordQuery] = useState("");
  const [failOnly, setFailOnly] = useState(false);
  const [error, setError] = useState("");

  // An id is 32 chars; a request per keystroke is waste. Query once typing stops.
  useEffect(() => {
    const t = setTimeout(() => setCallIdQuery(callId.trim()), 300);
    return () => clearTimeout(t);
  }, [callId]);

  useEffect(() => {
    const t = setTimeout(() => setKeywordQuery(keyword.trim()), 300);
    return () => clearTimeout(t);
  }, [keyword]);

  useEffect(() => {
    setDims(null);
    setStep("");
    setStage("");
    setAgentId("");
    setCallId("");
    setKeyword("");
    let live = true;
    devApi
      .dimensions(world)
      .then((d) => {
        if (live) setDims(d);
      })
      .catch((e) => {
        if (live) setError(String(e.message ?? e));
      });
    return () => {
      live = false;
    };
  }, [world]);

  // Each fetch takes a ticket; a stale response is dropped. /trace/calls can be much slower than
  // the 300ms debounce, so an earlier broad query arriving late would silently overwrite a later
  // narrow one, with no spinner to hint it's stale.
  const ticket = useRef(0);

  const load = useCallback(async () => {
    const mine = ++ticket.current;
    setError("");
    setPage(null);
    try {
      const next =
        scope === "build"
          ? await devApi.buildCalls(world, { callId: callIdQuery, q: keywordQuery })
          : await devApi.calls(world, {
              groupBy,
              step,
              stage,
              agentId,
              callId: callIdQuery,
              q: keywordQuery,
            });
      if (mine === ticket.current) setPage(next);
    } catch (e) {
      if (mine === ticket.current) setError(e instanceof Error ? e.message : String(e));
    }
  }, [world, scope, groupBy, step, stage, agentId, callIdQuery, keywordQuery]);

  useEffect(() => {
    void load();
  }, [load]);

  const groups: CallGroup[] = useMemo(() => {
    const raw = page?.groups ?? [];
    if (!failOnly) return raw;
    return raw
      .map((g) => ({ ...g, items: g.items.filter(isAnyFail) }))
      .filter((g) => g.items.length > 0);
  }, [page, failOnly]);

  const runtime = scope === "runtime";

  return (
    <div className="space-y-3">
      {dims && <AggregateBand label="World total" total={dims.totals} />}

      <div className={`${panel} flex flex-wrap items-end gap-4 px-4 py-3`}>
        <Field label="Scope">
          <Segmented value={scope} options={SCOPES} onChange={setScope} />
        </Field>
        <Field label="Group by">
          {runtime ? (
            <Segmented value={groupBy} options={GROUP_BYS} onChange={setGroupBy} />
          ) : (
            // Build has only one meaningful axis, scene. Say so rather than leave a dead control.
            <span className="px-1 py-1.5 text-xs text-slate-500">Scene (build has no steps)</span>
          )}
        </Field>
        {runtime && (
          <>
            <Field label="Step">
              <select className={input} value={step} onChange={(e) => setStep(e.target.value)}>
                <option value="">All steps</option>
                {(dims?.steps ?? []).map((s) => (
                  <option key={s.step} value={String(s.step)}>
                    step {s.step} · {s.calls} calls · {fmtMs(s.wall_ms)}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Stage">
              <select className={input} value={stage} onChange={(e) => setStage(e.target.value)}>
                <option value="">All stages</option>
                {(dims?.stages ?? []).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Agent">
              <select className={input} value={agentId} onChange={(e) => setAgentId(e.target.value)}>
                <option value="">All agents</option>
                {(dims?.agents ?? []).map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name}
                  </option>
                ))}
              </select>
            </Field>
          </>
        )}
        <Field label="Call ID">
          <input
            className={`${input} w-40 font-mono`}
            value={callId}
            placeholder="id or prefix"
            onChange={(e) => setCallId(e.target.value)}
          />
        </Field>
        <Field label="Search">
          <input
            className={`${input} w-56`}
            value={keyword}
            placeholder="keywords in prompt / response"
            onChange={(e) => setKeyword(e.target.value)}
          />
        </Field>
        <label className="flex items-center gap-2 whitespace-nowrap text-xs text-slate-300">
          <input
            type="checkbox"
            checked={failOnly}
            onChange={(e) => setFailOnly(e.target.checked)}
            className="accent-rose-500"
          />
          Failures only
        </label>
        {page && (
          <span className="ml-auto text-[11px] text-slate-500">
            This view: {fmt(page.total.calls)} calls · in {fmt(page.total.input_tokens)} · out{" "}
            {fmt(page.total.output_tokens)} · {fmtMs(page.total.latency_ms)}
          </span>
        )}
      </div>

      <ErrorLine error={error} />

      {!page && !error && <Empty>Loading…</Empty>}
      {page && groups.length === 0 && (
        <Empty>{failOnly ? "No failures in this view. 🎉" : "No calls match this view."}</Empty>
      )}
      {groups.map((g) => (
        <GroupBlock
          key={g.key}
          group={g}
          axis={runtime ? groupBy : "scene"}
          openAll={failOnly || callIdQuery !== "" || keywordQuery !== ""}
          onPickCallId={onPickCallId}
        />
      ))}
    </div>
  );
}

function GroupBlock({
  group,
  axis,
  openAll,
  onPickCallId,
}: {
  group: CallGroup;
  axis: Grouping | "scene";
  openAll: boolean;
  onPickCallId: (id: string) => void;
}) {
  // Expand by default when filtered to failures, a call id or a keyword: the list is short then,
  // and those are the ones you want to see.
  const [open, setOpen] = useState(openAll);
  useEffect(() => setOpen(openAll), [openAll]);
  // The prefix says which axis this row is a cell of; "31" or "decision" alone doesn't.
  const prefix = axis === "step" && group.key !== "build" ? "step " : "";
  // The backend files agent-less calls under key "—". Shown as-is it looks like a name (a lone
  // dash among Chinese names doesn't read as "no agent"), so spell it out.
  const label = axis === "agent" && group.key === "—" ? "(no agent)" : group.label;
  return (
    <div className={`${panel} overflow-hidden`}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-4 px-4 py-2.5 text-left hover:bg-slate-800/40"
      >
        <span className="min-w-40 text-sm font-semibold text-slate-100">
          <span className="mr-2 inline-block h-2 w-2 rounded-full bg-indigo-400" />
          {prefix}
          {label}
        </span>
        <span className="ml-auto flex flex-wrap items-center gap-4 text-[11px] text-slate-500">
          <span>
            <b className="text-slate-200">{fmt(group.calls)}</b> calls
          </span>
          {group.failures > 0 && (
            <span className="text-rose-400">
              ✗<b>{group.failures}</b> call ({pct(group.failure_rate)})
            </span>
          )}
          {group.parse_failures > 0 && (
            <span className="text-amber-400">
              ✗<b>{group.parse_failures}</b> parse ({pct(group.parse_failure_rate)})
            </span>
          )}
          {group.discarded > 0 && (
            <span className="text-fuchsia-400">
              🚫<b>{group.discarded}</b> ({pct(group.discard_rate)})
            </span>
          )}
          <span>in <b className="text-slate-200">{fmt(group.input_tokens)}</b></span>
          <span>out <b className="text-slate-200">{fmt(group.output_tokens)}</b></span>
          <span>{fmtMs(group.latency_ms)}</span>
        </span>
      </button>
      {open && (
        <div className="space-y-1 border-t border-slate-800 px-2 py-2">
          {group.items.map((c, i) => (
            <CallCard key={c.call_id || i} call={c} onPickCallId={onPickCallId} />
          ))}
        </div>
      )}
    </div>
  );
}
