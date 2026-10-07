// Recall inspector: send a query into an agent's memory and see the whole funnel (candidates ->
// relevance floor -> R/I/R ranking -> MMR dedup -> top_k), not just the final few.
// Each candidate's dense score, three factors and drop reason are shown, since "which ones came
// back" can't tell which stage failed, and the floor should be calibrated from data. The
// diagnostic read doesn't touch memories, so it won't disturb recency.

import { useCallback, useEffect, useState } from "react";

import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";

import { devApi } from "./api";
import type { RecallResult } from "./types";
import { Empty, ErrorLine, Field, Stat, btn, input, panel } from "./ui";

export default function RecallInspector({ world }: { world: string }) {
  const modelKeys = useModelKeys();
  const [agents, setAgents] = useState<{ id: string; name: string }[]>([]);
  const [agentId, setAgentId] = useState("");
  const [about, setAbout] = useState("");
  const [query, setQuery] = useState("");
  const [stream, setStream] = useState("");
  const [topK, setTopK] = useState(5);
  const [floor, setFloor] = useState(0.2);
  const [data, setData] = useState<RecallResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setData(null);
    setAgentId("");
    setAbout("");
    devApi
      .recallAgents(world)
      .then((d) => {
        setAgents(d.agents);
        setAgentId(d.agents[0]?.id ?? "");
      })
      .catch(() => setAgents([]));
  }, [world]);

  const run = useCallback(async () => {
    if (!query.trim()) {
      setError("query is required");
      return;
    }
    setBusy(true);
    setError("");
    try {
      setData(
        await devApi.recall(world, {
          agent_id: agentId,
          query: query.trim(),
          related_agent_id: about || null,
          stream: stream || null,
          top_k: topK,
          floor,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [world, agentId, query, about, stream, topK, floor]);

  const opts = agents.map((a) => (
    <option key={a.id} value={a.id}>
      {a.name}
    </option>
  ));

  return (
    <div className="space-y-3">
      <div className={`${panel} px-4 py-3`}>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Whose memory">
            <select className={`${input} w-40`} value={agentId} onChange={(e) => setAgentId(e.target.value)}>
              {opts}
            </select>
          </Field>
          <Field label="Query">
            <input
              className={`${input} w-80`}
              placeholder="what to retrieve"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
            />
          </Field>
          <Field label="About whom">
            <select className={`${input} w-40`} value={about} onChange={(e) => setAbout(e.target.value)}>
              <option value="">(unscoped — pure semantic search)</option>
              {opts}
            </select>
          </Field>
          <Field label="Stream">
            <select className={`${input} w-32`} value={stream} onChange={(e) => setStream(e.target.value)}>
              <option value="">both</option>
              <option value="factual">factual</option>
              <option value="experiential">experiential</option>
            </select>
          </Field>
          <Field label="Top k">
            <input
              type="number"
              min={1}
              max={50}
              className={`${input} w-20`}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value, 10) || 5)}
            />
          </Field>
          <Field label="Floor">
            <input
              type="number"
              step={0.05}
              min={-1}
              max={1}
              className={`${input} w-24`}
              value={floor}
              onChange={(e) => setFloor(parseFloat(e.target.value))}
            />
          </Field>
          <button
            className={btn}
            disabled={busy || !modelKeys}
            title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
            onClick={run}
          >
            {busy ? "Retrieving…" : "Retrieve"}
          </button>
        </div>
        <div className="mt-2 text-[11px] text-slate-500">
          “About whom” constrains the candidate set by related_agents — “what I know about this
          person” is an indexing question, not a semantic one. A diagnostic read never touches the
          memories, so it cannot disturb recency.
        </div>
        <ErrorLine error={error} />
      </div>

      {!data && <Empty>Type a query, then hit “Retrieve”.</Empty>}
      {data && data.candidates.length === 0 && (
        <Empty>
          No candidates at all.{" "}
          {data.related_agent_id
            ? "That identity scope is empty — this agent holds no memory involving that person. (“I don’t know them” is the right answer here, not a bug.)"
            : "Everything fell below the relevance floor."}
        </Empty>
      )}
      {data && data.candidates.length > 0 && <Funnel data={data} />}
    </div>
  );
}

function Funnel({ data }: { data: RecallResult }) {
  const s = data.stats;
  return (
    <div className={`${panel} px-4 py-3`}>
      <div className="flex flex-wrap gap-x-8 gap-y-2">
        <Stat label="candidates" value={s.candidates} />
        <Stat label="selected" value={s.selected} />
        <Stat label="cut by floor" value={s.dropped_by_floor} />
        <Stat label="cut by MMR" value={s.dropped_by_mmr} />
        <Stat label="cut by top_k" value={s.dropped_by_top_k} />
        <Stat label="dense range" value={`${s.dense_min ?? "—"} ~ ${s.dense_max ?? "—"}`} />
      </div>
      <div className="mt-2 text-[11px] text-slate-500">
        dense is raw cosine — the floor compares against exactly this. If “irrelevant” memories
        score about as high as the relevant ones, an absolute threshold has no discriminating power
        in this corpus: constrain by identity scope rather than raising the floor.
      </div>
      <div className="mt-3 -mx-2 overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="whitespace-nowrap text-left text-[10px] uppercase text-slate-500">
              <th className="px-2 py-1">Outcome</th>
              <th className="px-2 py-1">Score</th>
              <th className="px-2 py-1">dense</th>
              <th className="px-2 py-1">R/Re/I</th>
              <th className="px-2 py-1">Origin</th>
              <th className="px-2 py-1">About</th>
              <th className="px-2 py-1">Content</th>
            </tr>
          </thead>
          <tbody>
            {data.candidates.map((c) => (
              <tr key={c.id} className="border-t border-slate-800 align-top">
                <td className="whitespace-nowrap px-2 py-1.5">
                  {c.selected ? (
                    <span className="text-emerald-400">✓ selected</span>
                  ) : (
                    <span className="text-slate-500">
                      ✕{" "}
                      {c.dropped === "floor"
                        ? "below floor"
                        : c.dropped === "mmr_duplicate"
                          ? "near-duplicate"
                          : "out of slots"}
                    </span>
                  )}
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 tabular-nums text-slate-200">
                  {c.score.toFixed(3)}
                </td>
                <td
                  className="whitespace-nowrap px-2 py-1.5 tabular-nums text-slate-300"
                  title="raw cosine"
                >
                  {c.dense.toFixed(3)}
                </td>
                <td
                  className="whitespace-nowrap px-2 py-1.5 tabular-nums text-slate-500"
                  title="normalised relevance / recency / importance"
                >
                  {c.relevance_n.toFixed(2)} / {c.recency_n.toFixed(2)} / {c.importance_n.toFixed(2)}
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 text-slate-500">
                  {c.stream}·{c.kind}·s{c.created_step}
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 text-slate-400">
                  {c.related_agents.length ? (
                    c.related_agents.map((a) => a.name).join("、")
                  ) : (
                    <span className="text-slate-600">unowned</span>
                  )}
                </td>
                <td className="w-full px-2 py-1.5 text-slate-300">{c.content}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
