// Prompt replay: load an LLM call that already happened, edit the prompt by hand, and re-run it
// straight against the provider.
//
// It replays the prompt text recorded in the trace, so code changes don't affect it; to test a
// changed prompt builder, use the Stage suite.
// Replay deliberately writes no trace: probing shouldn't pollute a world's record.

import { useCallback, useEffect, useState } from "react";

import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";

import { Labeled } from "./CallCard";
import { devApi } from "./api";
import type { PromptMessage, ReplayResult, TraceCall } from "./types";
import { ErrorLine, Field, Mono, btn, btnGhost, input, panel, prettyJson } from "./ui";

export default function PromptReplay({
  world,
  seedCallId,
}: {
  world: string;
  seedCallId: string;
}) {
  const modelKeys = useModelKeys();
  const [callId, setCallId] = useState(seedCallId);
  const [call, setCall] = useState<TraceCall | null>(null);
  const [messages, setMessages] = useState<PromptMessage[]>([]);
  const [scene, setScene] = useState("");
  const [model, setModel] = useState("");
  // Endpoint-dialect overrides (JSON), applied key by key over the scene's params. Switching
  // models often needs them: some models only accept enable_thinking: true, and sending the
  // scene's false gets a 400.
  const [paramsText, setParamsText] = useState("");
  const [temp, setTemp] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);
  const [jsonMode, setJsonMode] = useState(false);
  const [result, setResult] = useState<ReplayResult | null>(null);
  const [error, setError] = useState("");
  const [runError, setRunError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(
    async (id: string) => {
      const cid = id.trim();
      if (!cid) return;
      setError("");
      setResult(null);
      try {
        const c = await devApi.call(world, cid);
        setCall(c);
        setMessages(c.prompt_messages ?? []);
        setScene(c.scene ?? "");
        setModel("");
        setParamsText("");
        setTemp(c.temperature ?? 0.7);
        setMaxTokens(c.max_tokens ?? 1000);
        setJsonMode(!!c.json_mode);
      } catch (e) {
        setCall(null);
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [world],
  );

  // Arriving from a #call_id link on the trace page loads it straight away, no extra Load click.
  useEffect(() => {
    setCallId(seedCallId);
    if (seedCallId) void load(seedCallId);
  }, [seedCallId, load]);

  const run = async () => {
    // Clear the previous result first: a stale success left up after a failure reads as if this
    // run succeeded.
    setResult(null);
    setRunError("");
    let params: Record<string, unknown> | null = null;
    if (paramsText.trim()) {
      try {
        const parsed: unknown = JSON.parse(paramsText);
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
          throw new Error("must be a JSON object");
        }
        params = parsed as Record<string, unknown>;
      } catch (e) {
        setRunError(`Params: ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }
    setBusy(true);
    try {
      setResult(
        await devApi.replay({
          scene: scene.trim() || null,
          model: model.trim() || null,
          params,
          messages,
          temperature: temp,
          max_tokens: maxTokens,
          json_mode: jsonMode,
        }),
      );
    } catch (e) {
      setRunError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className={`${panel} px-4 py-3`}>
        <div className="mb-2 text-xs font-semibold text-slate-200">
          Load a call
          <span className="ml-2 font-normal text-slate-500">
            (clicking a call’s #id on the Trace tab sends it here; a replay is never written
            to the trace)
          </span>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <Field label="call id">
            <input
              className={`${input} w-80 font-mono`}
              placeholder="paste a call id"
              value={callId}
              onChange={(e) => setCallId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && load(callId)}
            />
          </Field>
          <button className={btnGhost} onClick={() => load(callId)}>
            Load
          </button>
          {call && (
            <span className="text-[11px] text-slate-500">
              {call.scene} · {call.stage} · {call.model}
              {call.agent_name ? ` · ${call.agent_name}` : ""}
              {call.step != null ? ` · step ${call.step}` : " · build"}
            </span>
          )}
        </div>
        <ErrorLine error={error} />
      </div>

      {call && (
        <>
          <div className={`${panel} space-y-2 px-4 py-3`}>
            {messages.map((m, i) => (
              <Labeled key={i} label={m.role}>
                <textarea
                  value={m.content}
                  onChange={(e) =>
                    setMessages((ms) => ms.map((x, n) => (n === i ? { ...x, content: e.target.value } : x)))
                  }
                  className={`${input} w-full font-mono leading-relaxed`}
                  style={{ minHeight: m.role === "system" ? 220 : 140 }}
                />
              </Labeled>
            ))}
            <div className="flex flex-wrap items-end gap-3 border-t border-slate-800 pt-3">
              <Field label="Scene">
                <input className={`${input} w-48`} value={scene} onChange={(e) => setScene(e.target.value)} />
              </Field>
              <Field label="Model override">
                <input
                  className={`${input} w-44`}
                  placeholder="blank = the scene’s configured model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                />
              </Field>
              <Field label="Params override">
                <textarea
                  rows={3}
                  className={`${input} w-72 resize-y font-mono leading-relaxed`}
                  placeholder='JSON, e.g. {"enable_thinking": true}'
                  title="Merged key by key over the scene's endpoint params and sent as-is"
                  value={paramsText}
                  onChange={(e) => setParamsText(e.target.value)}
                />
              </Field>
              <Field label="Temp">
                <input
                  type="number"
                  step={0.1}
                  min={0}
                  max={2}
                  className={`${input} w-20`}
                  value={temp}
                  onChange={(e) => setTemp(parseFloat(e.target.value) || 0)}
                />
              </Field>
              <Field label="Max tokens">
                <input
                  type="number"
                  step={10}
                  min={1}
                  className={`${input} w-24`}
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 1)}
                />
              </Field>
              <label
                className="flex items-center gap-1.5 text-xs text-slate-300"
                title="Whether production ran this call in JSON mode — restored from the trace"
              >
                <input
                  type="checkbox"
                  checked={jsonMode}
                  onChange={(e) => setJsonMode(e.target.checked)}
                  className="accent-indigo-500"
                />
                json_mode
              </label>
              <button
                className={btn}
                disabled={busy || !modelKeys}
                title={modelKeys ? undefined : NO_MODEL_KEYS_HINT}
                onClick={run}
              >
                {busy ? "Calling…" : "Call LLM"}
              </button>
              {result && (
                <span className="text-[11px] text-emerald-400">
                  ✓ {result.model} · in {result.input_tokens} · out {result.output_tokens} ·{" "}
                  {result.latency_ms}ms
                  {Object.keys(result.params ?? {}).length > 0 && (
                    <span className="ml-1 font-mono text-slate-500">· {JSON.stringify(result.params)}</span>
                  )}
                </span>
              )}
            </div>
            <ErrorLine error={runError} />
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
                Recorded response
              </div>
              <Mono className="max-h-[32rem]">{prettyJson(call.response_content) || "—"}</Mono>
            </div>
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
                New response
              </div>
              <Mono accent className="max-h-[32rem]">
                {result ? prettyJson(result.content) || "(empty)" : "—"}
              </Mono>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
