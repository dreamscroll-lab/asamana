// 🎬 Director console: exercise the director's LLM path outside the world, break it, run it again.
//
// Three stages. The middle one is the same replay route Prompt replay uses, so the prompt can be
// edited and the model swapped freely:
//   ① /director/prompt     the prompt production would send (not sent)
//   ② /api/llm/replay      sent straight to the provider (no trace written)
//   ③ /director/interpret  run the raw response through production validation: is it accepted,
//                          and what shape does it land as
//
// The world doesn't move: nothing is injected or queued. To actually land something in the
// world, use the product path POST /direct.
// The response box is editable: hand-write a response (e.g. an out-of-range person index like
// #99) and send it to ③ to test the validation layer without spending tokens. That layer is
// where the director path most often goes wrong.

import { useEffect, useState } from "react";

import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";
import { Labeled } from "./CallCard";
import { devApi } from "./api";
import type { DirectorPrompt, DirectorVerdict, PromptMessage } from "./types";
import { ErrorLine, Field, btn, btnGhost, input, panel, prettyJson } from "./ui";

export default function DirectorConsole({ world }: { world: string }) {
  // Every stage needs the world's live session, and a session needs the model keys.
  const modelKeys = useModelKeys();
  const keyHint = modelKeys ? undefined : NO_MODEL_KEYS_HINT;
  const [text, setText] = useState("");
  const [prompt, setPrompt] = useState<DirectorPrompt | null>(null);
  const [messages, setMessages] = useState<PromptMessage[]>([]);
  const [model, setModel] = useState("");
  const [temp, setTemp] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1000);
  const [response, setResponse] = useState("");
  const [verdict, setVerdict] = useState<DirectorVerdict | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setPrompt(null);
    setMessages([]);
    setResponse("");
    setVerdict(null);
    setStatus("");
  }, [world]);

  const build = async (): Promise<DirectorPrompt | null> => {
    if (!text.trim()) {
      setError("Write a directive first");
      return null;
    }
    setError("");
    setStatus("Assembling prompt…");
    const p = await devApi.directorPrompt(world, text.trim());
    setPrompt(p);
    setMessages(p.messages);
    setTemp(p.temperature);
    setMaxTokens(p.max_tokens);
    setStatus("✓ prompt assembled (edit it, then call)");
    return p;
  };

  const call = async (p: DirectorPrompt | null, msgs: PromptMessage[]): Promise<string | null> => {
    if (!p) {
      setError("Assemble the prompt first");
      return null;
    }
    setStatus("Calling…");
    const res = await devApi.replay({
      scene: p.scene,
      model: model.trim() || null,
      messages: msgs,
      temperature: temp,
      max_tokens: maxTokens,
      json_mode: p.json_mode,
    });
    const content = prettyJson(res.content);
    setResponse(content);
    setStatus(`✓ ${res.model} · in ${res.input_tokens} · out ${res.output_tokens} · ${res.latency_ms}ms`);
    return content;
  };

  const interpret = async (raw: string) => {
    if (!raw.trim()) {
      setError("Nothing to validate");
      return;
    }
    setVerdict(await devApi.directorInterpret(world, raw));
  };

  const guard = (fn: () => Promise<unknown>) => async () => {
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
    }
  };

  return (
    <div className="space-y-3">
      <div className={`${panel} px-4 py-3`}>
        <div className="mb-2 text-xs font-semibold text-slate-200">
          Directive
          <span className="ml-2 font-normal text-slate-500">
            (assemble prompt → call the LLM → run production validation; nothing is injected,
            queued, or traced)
          </span>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Say in one plain sentence what you want to happen in this world"
          className={`${input} min-h-16 w-full`}
        />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button
            className={btn}
            disabled={!modelKeys}
            title={keyHint}
            onClick={guard(async () => {
              const p = await build();
              if (!p) return;
              const c = await call(p, p.messages);
              if (c) await interpret(c);
            })}
          >
            Run all three (assemble → call → validate)
          </button>
          <button className={btnGhost} disabled={!modelKeys} title={keyHint} onClick={guard(build)}>
            Assemble prompt only
          </button>
          {status && <span className="text-[11px] text-slate-500">{status}</span>}
        </div>
        <ErrorLine error={error} />
      </div>

      {prompt && (
        <div className={`${panel} space-y-2 px-4 py-3`}>
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Scene">
              <input className={`${input} w-48`} value={prompt.scene} readOnly />
            </Field>
            <Field label="Model override">
              <input
                className={`${input} w-44`}
                placeholder="blank = the scene’s configured model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
              />
            </Field>
            <Field label="Temp">
              <input
                type="number"
                step={0.1}
                className={`${input} w-20`}
                value={temp}
                onChange={(e) => setTemp(parseFloat(e.target.value) || 0)}
              />
            </Field>
            <Field label="Max tokens">
              <input
                type="number"
                step={10}
                className={`${input} w-24`}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 1)}
              />
            </Field>
            <span className="text-[11px] text-slate-500">
              World parked at step {prompt.step} · {prompt.world_time.label} · json_mode=
              {String(prompt.json_mode)}
            </span>
          </div>
          {messages.map((m, i) => (
            <Labeled key={i} label={m.role}>
              <textarea
                value={m.content}
                onChange={(e) =>
                  setMessages((ms) => ms.map((x, n) => (n === i ? { ...x, content: e.target.value } : x)))
                }
                className={`${input} w-full font-mono leading-relaxed`}
                style={{ minHeight: m.role === "system" ? 200 : 140 }}
              />
            </Labeled>
          ))}
          <Menus menus={prompt.menus} />
          <button className={btnGhost} onClick={guard(() => call(prompt, messages))}>
            ② Call LLM (re-run as often as you like after editing)
          </button>
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        <div className={`${panel} px-4 py-3`}>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
            LLM response (editable — hand-write one and send it straight to validation)
          </div>
          <textarea
            value={response}
            onChange={(e) => setResponse(e.target.value)}
            className={`${input} min-h-56 w-full font-mono`}
          />
          <button
            className={`${btnGhost} mt-2`}
            disabled={!modelKeys}
            title={keyHint}
            onClick={guard(() => interpret(response))}
          >
            ③ Validate this response
          </button>
        </div>
        <div className={`${panel} px-4 py-3`}>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-slate-500">
            Production verdict
          </div>
          {verdict ? (
            <Verdict verdict={verdict} />
          ) : (
            <div className="text-xs text-slate-500">Not validated yet.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function Menus({ menus }: { menus: DirectorPrompt["menus"] }) {
  const table = (title: string, m?: Record<string, string>) => {
    const rows = Object.entries(m ?? {});
    return (
      <div className="min-w-40">
        <div className="text-[10px] uppercase text-slate-500">{title}</div>
        {rows.length ? (
          rows.map(([i, name]) => (
            <div key={i} className="text-[11px] text-slate-300">
              #{i} {name}
            </div>
          ))
        ) : (
          <div className="text-[11px] text-slate-600">(none)</div>
        )}
      </div>
    );
  };
  return (
    <details className="rounded-lg border border-slate-800 bg-slate-950/40 px-3 py-1.5">
      <summary className="cursor-pointer text-[11px] text-slate-400">
        Candidate menus (index → name) — the LLM only emits indices, so this is where you
        check who it actually pointed at
      </summary>
      <div className="mt-2 flex flex-wrap gap-6">
        {table("Cast", menus?.cast)}
        {table("Locations", menus?.location)}
        {table("Entities", menus?.entity)}
      </div>
    </details>
  );
}

function Verdict({ verdict }: { verdict: DirectorVerdict }) {
  const p = verdict.plan;
  return (
    <div className="space-y-2 text-xs">
      <span
        className={`inline-block rounded px-2 py-0.5 font-bold ${
          verdict.accepted ? "bg-emerald-600 text-emerald-50" : "bg-rose-700 text-rose-50"
        }`}
      >
        {verdict.accepted ? "Would be accepted" : "Would be rejected"}
      </span>
      <div className="text-slate-300">
        {verdict.accepted ? (
          <>
            This beat would be: <b className="text-slate-100">{verdict.preview}</b>
          </>
        ) : (
          <>
            What the director is told: <b className="text-slate-100">{verdict.reason}</b>
          </>
        )}
      </div>
      {p && (
        <>
          <div className="text-[10px] text-slate-500">
            Landing points (indices resolved to names) · channels:{" "}
            {(p.channels ?? []).join(" + ") || "—"}
          </div>
          <table className="w-full">
            <tbody>
              {p.broadcast && (
                <Row kind="broadcast">
                  {p.broadcast.location}
                  <span className="text-slate-500">
                    {p.broadcast.severity} · {p.broadcast.phenomenon}
                  </span>
                  <br />
                  {p.broadcast.content}
                </Row>
              )}
              {p.message && (
                <Row kind="message">
                  → {(p.message.recipients ?? []).join("、") || "(no recipients)"}
                  <span className="text-slate-500">{p.message.urgency}</span>
                  <br />
                  {p.message.content}
                </Row>
              )}
              {(p.mutations ?? []).map((m, i) => (
                <Row key={i} kind={m.kind}>
                  {m.target}　<span className="text-slate-500">{m.detail}</span>
                  <br />
                  <span className="text-slate-500">Bystanders see: </span>
                  {m.observation}
                </Row>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

function Row({ kind, children }: { kind: string; children: React.ReactNode }) {
  return (
    <tr>
      <td className="py-1 pr-3 align-top text-slate-500">{kind}</td>
      <td className="py-1 text-slate-300">{children}</td>
    </tr>
  );
}
