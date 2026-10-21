// One LLM call, opened up: tags → call id → extra → prompt → thinking → response.
// Shared by the trace list and the prompt bench, so "what a call looks like" is
// defined once. A beat that landed without a call (BeatCard) is shown the same way, minus
// everything a request has.

import { Collapse, Mono, fmt, prettyJson } from "./ui";
import type { TraceBeat, TraceCall } from "./types";

const ACTOR_TONE = "border-sky-800 bg-sky-950/40 text-sky-300";

const isCallFail = (c: TraceCall) => c.ok === false;
const isParseFail = (c: TraceCall) => c.ok !== false && c.parse_ok === false;
// core.interfaces.trace.SET_ASIDE_REASONS: output set aside for what happened around it (a
// person passed it over, its character was drawn into someone else's act). Not adopted, but
// nothing was wrong with it, so it is no failure.
const SET_ASIDE: Record<string, string> = { superseded: "NOT CHOSEN", conscripted: "CONSCRIPTED" };
const setAside = (c: TraceCall) => (c.adopted === false ? SET_ASIDE[c.reject_reason] : undefined);
const isSetAside = (c: TraceCall) => setAside(c) !== undefined;
// The quiet one: the provider answered, the JSON parsed — and the engine threw the result
// away anyway (a required field missing, a hallucinated index…). Nothing in ok/parse_ok
// shows it, so without this the burned beat looks like a perfectly healthy call.
const isDiscarded = (c: TraceCall) => c.adopted === false && !isSetAside(c);
export const isAnyFail = (c: TraceCall) => isCallFail(c) || isParseFail(c) || isDiscarded(c);

function Tag({ children, tone = "" }: { children: React.ReactNode; tone?: string }) {
  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[10px] ${
        tone || "border-slate-800 bg-slate-950/60 text-slate-400"
      }`}
    >
      {children}
    </span>
  );
}

export default function CallCard({
  call,
  open,
  onPickCallId,
}: {
  call: TraceCall;
  open?: boolean;
  onPickCallId?: (id: string) => void;
}) {
  const tone = isCallFail(call) ? "bad" : isParseFail(call) || isDiscarded(call) ? "warn" : "default";
  const head = (
    <span className="flex flex-wrap items-center gap-1.5">
      {isCallFail(call) && <Tag tone="border-rose-800 bg-rose-950/60 text-rose-300">✗ CALL FAILED</Tag>}
      {isParseFail(call) && <Tag tone="border-amber-800 bg-amber-950/50 text-amber-300">⚠ PARSE FAILED</Tag>}
      {isDiscarded(call) && (
        <Tag tone="border-fuchsia-800 bg-fuchsia-950/40 text-fuchsia-300">
          🚫 DISCARDED{call.reject_reason ? `: ${call.reject_reason}` : ""}
        </Tag>
      )}
      {call.actor && <Tag tone={ACTOR_TONE}>ACTOR</Tag>}
      {isSetAside(call) && <Tag>{setAside(call)}</Tag>}
      <VerdictTag extra={call.extra} />
      <Tag tone="border-indigo-800 bg-indigo-950/40 text-indigo-300">{call.scene}</Tag>
      <Tag>{call.step != null ? `step ${call.step}` : "build"}</Tag>
      <Tag>{call.stage}</Tag>
      {call.agent_name && <Tag tone="border-emerald-900 bg-emerald-950/30 text-emerald-300">{call.agent_name}</Tag>}
      <Tag>{call.model}</Tag>
      {call.call_id && (
        <button
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onPickCallId?.(call.call_id);
            void navigator.clipboard?.writeText(call.call_id);
          }}
          title={`${call.call_id} — click to copy${onPickCallId ? " and open in Prompt replay (new tab)" : ""}`}
          className="rounded border border-slate-800 bg-slate-950/60 px-1.5 py-0.5 font-mono text-[10px] text-slate-500 hover:text-indigo-300"
        >
          #{call.call_id.slice(0, 8)}
        </button>
      )}
      <span className="text-[10px] text-slate-500">
        in {fmt(call.input_tokens)} · out {fmt(call.output_tokens)} · {Math.round(call.latency_ms)}ms
      </span>
    </span>
  );

  const extraEntries = Object.entries(call.extra || {});
  return (
    <Collapse title={head} open={open} tone={tone}>
      <div className="space-y-2">
        {call.call_id && (
          <Labeled label="call id">
            <Mono>{call.call_id}</Mono>
          </Labeled>
        )}
        {/* Generic diagnostic annotations (LLMCallTrace.extra) — key/value, so any field
            added via annotate_call() shows up here with no front-end change. */}
        {extraEntries.length > 0 && (
          <Labeled label="extra">
            <Mono>{extraEntries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join("\n")}</Mono>
          </Labeled>
        )}
        {(call.prompt_messages || []).map((m, i) => (
          <Labeled key={i} label={m.role}>
            <Mono className="max-h-96">{m.content}</Mono>
          </Labeled>
        ))}
        {isCallFail(call) && call.error && (
          <Labeled label="error">
            <Mono className="text-rose-300">{call.error}</Mono>
          </Labeled>
        )}
        {/* Thinking sits BEFORE the response: it is where the answer came from, and reading
            order should match. Its token count is here because it shares max_tokens with the
            answer — a truncation is eaten from this side first. */}
        {!isCallFail(call) && call.thinking && (
          <Labeled label={`thinking${call.thinking_tokens ? ` (${call.thinking_tokens} tok)` : ""}`}>
            <Mono className="max-h-72">{call.thinking}</Mono>
          </Labeled>
        )}
        {!isCallFail(call) && (
          <Labeled
            label={`response${
              isParseFail(call)
                ? " (unparseable)"
                : isDiscarded(call)
                  ? ` (discarded — ${call.reject_reason || "not adopted"})`
                  : isSetAside(call)
                    ? ` (${setAside(call)?.toLowerCase()})`
                    : ""
            }`}
          >
            <Mono accent className="max-h-96">
              {prettyJson(call.response_content)}
            </Mono>
          </Labeled>
        )}
      </div>
    </Collapse>
  );
}

// The arbitration verdict the runtime stamps on a decision (``extra.verdict``): what the world
// did with the intent. Executed is the usual case and goes untagged; the rest are world outcomes,
// not failures, except start_failed, the machine breaking.
function VerdictTag({ extra }: { extra: Record<string, unknown> | undefined }) {
  const verdict = extra?.verdict;
  if (typeof verdict !== "string" || verdict === "executed") return null;
  const label = verdict.replace(/_/g, " ").toUpperCase();
  return verdict === "start_failed" ? (
    <Tag tone="border-amber-800 bg-amber-950/50 text-amber-300">⚠ {label}</Tag>
  ) : (
    <Tag>{label}</Tag>
  );
}

export function BeatCard({ beat }: { beat: TraceBeat }) {
  const head = (
    <span className="flex flex-wrap items-center gap-1.5">
      {beat.actor && <Tag tone={ACTOR_TONE}>ACTOR</Tag>}
      <Tag tone="border-teal-800 bg-teal-950/40 text-teal-300">beat · {beat.beat_kind}</Tag>
      <VerdictTag extra={beat.extra} />
      <Tag>step {beat.step}</Tag>
      <Tag>{beat.stage}</Tag>
      {beat.agent_name && <Tag tone="border-emerald-900 bg-emerald-950/30 text-emerald-300">{beat.agent_name}</Tag>}
      <span className="text-[10px] text-slate-500">no LLM request</span>
    </span>
  );
  const extraEntries = Object.entries(beat.extra || {});
  return (
    <Collapse title={head}>
      <div className="space-y-2">
        {extraEntries.length > 0 && (
          <Labeled label="extra">
            <Mono>{extraEntries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join("\n")}</Mono>
          </Labeled>
        )}
        <Labeled label="payload">
          <Mono accent className="max-h-96">
            {JSON.stringify(beat.payload, null, 2)}
          </Mono>
        </Labeled>
      </div>
    </Collapse>
  );
}

export function Labeled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-0.5 text-[10px] uppercase tracking-wide text-amber-500/80">{label}</div>
      {children}
    </div>
  );
}
