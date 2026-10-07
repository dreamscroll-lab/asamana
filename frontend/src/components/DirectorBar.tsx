/**
 * DirectorBar — the one place a human reaches into a running world.
 *
 * A single free-text box, on purpose: form fields (target / channel / severity) would cap
 * what you can say; the engine parses a sentence and says plainly when it cannot.
 *
 * It floats top-right over the map: a band would cost the stage height for a mostly empty
 * box, the map owns the other corners (focus toolbar top-left, control strip bottom), and
 * an isometric diamond leaves both upper corners clear at any fit-to-view zoom.
 *
 * One button, not 注入 + 推一步: whether a step is needed depends on the run loop, which
 * only the backend knows for certain, so it decides and an accepted directive always lands.
 *
 * 1. A refusal is an answer, not an error: "落实不了" is a normal 200 with a reason, shown
 *    as advice about the sentence.
 * 2. Never leave a submitted sentence unaccounted for — accepted, refused or transport
 *    failure, something is said about it.
 *
 * The record below the box is an index read from `GET /directives`, because the feed only
 * holds steps you watched. A row is 「what I said → what the engine did」; the receipt
 * (delivered / pressed / decided / interrupted) stays on the feed card, since it means
 * nothing without the step around it. Refusals are kept locally and not persisted: they
 * changed nothing in the world.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { api } from "../api/client";
import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";
import type { Intervention } from "../types";

type Outcome =
  | { kind: "idle" }
  | { kind: "working" }
  | { kind: "queued"; preview: string }
  | { kind: "refused"; reason: string };

/** A sentence this session that never became anything — lives and dies with the tab. */
type Refused = { text: string; reason: string };

/** Past this height the input scrolls internally; any taller and it covers the map. */
const INPUT_MAX_PX = 132;

/** Collapsed and expanded share this corner so collapsing doesn't move the entry point.
 *  LiveWorldMap's focus toolbar leaves this width free (pr-[27rem] there). */
const DOCK = "absolute top-2 right-2 z-30";

/** `step` is only a refresh trigger (receipts arrive at the end of the step). Never
 *  rendered: 「第 N 步」 is a code-layer ordinal. */
export default function DirectorBar({ worldId, step }: { worldId: string; step: number | null }) {
  // Collapsed by default: a rarely used tool shouldn't cover the map that is usually watched.
  const [collapsed, setCollapsed] = useState(true);
  const [text, setText] = useState("");
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const [landed, setLanded] = useState<Intervention[]>([]);
  const [refused, setRefused] = useState<Refused[]>([]);
  const box = useRef<HTMLTextAreaElement>(null);
  const modelKeys = useModelKeys();

  // Auto-grow: reset to zero before measuring scrollHeight, otherwise it only grows and never
  // shrinks.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_PX)}px`;
  }, [text, collapsed]);

  const reload = useCallback(async () => {
    try {
      setLanded(await api.listDirectives(worldId));
    } catch {
      // Failing to read history shouldn't block the input.
    }
  }, [worldId]);

  // Only read while expanded: collapsed, nobody is looking at the record.
  useEffect(() => {
    if (!collapsed) void reload();
  }, [collapsed, step, reload]);

  async function submit() {
    const value = text.trim();
    if (!modelKeys || !value || outcome.kind === "working") return;
    setOutcome({ kind: "working" });
    try {
      const res = await api.direct(worldId, value);
      if (res.accepted) {
        setOutcome({ kind: "queued", preview: res.preview });
        setText(""); // it is committed; leaving it in the box invites a double submit
        // The backend steps a stopped world itself, so the snapshot is usually written by now;
        // in a running world the effect above catches up on the next step.
        void reload();
      } else {
        // Keep the text: the whole point of a reason is that they can edit and retry.
        setOutcome({ kind: "refused", reason: res.reason });
        setRefused((prev) => [{ text: value, reason: res.reason }, ...prev]);
      }
    } catch (e) {
      setOutcome({ kind: "refused", reason: String(e) });
    }
  }

  if (collapsed) {
    return (
      <button
        className={`${DOCK} w-9 h-9 grid place-items-center rounded-full text-sm bg-slate-950/75 border border-slate-800/70 backdrop-blur-md shadow-lg shadow-black/40 hover:border-indigo-700/70 transition`}
        onClick={() => setCollapsed(false)}
        title="展开导演台"
      >
        🎬
      </button>
    );
  }

  return (
    <section className={`${DOCK} w-[min(26rem,calc(100%-1rem))] rounded-2xl border border-slate-800/70 bg-slate-950/75 backdrop-blur-md shadow-xl shadow-black/40 px-3 py-2.5 space-y-2`}>
      <div className="flex items-center justify-between">
        <span className="text-sm" title="导演台">🎬</span>
        <button
          className="text-[11px] text-slate-500 hover:text-slate-200 transition"
          onClick={() => setCollapsed(true)}
          title="收起导演台"
        >
          收起 ⌄
        </button>
      </div>

      <textarea
        ref={box}
        value={text}
        rows={1}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          // Enter submits, Shift+Enter adds a newline, like a chat box. Enter during IME
          // composition confirms a candidate; it doesn't submit.
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            void submit();
          }
        }}
        className="w-full resize-none overflow-y-auto bg-slate-950/70 border border-slate-800 rounded-lg px-3 py-2 text-xs leading-relaxed text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/70"
        placeholder="写下你想让这个世界发生什么…"
        maxLength={500}
      />

      <div className="flex items-center gap-3">
        <span className="text-[10px] text-slate-600">Enter 送出 · Shift+Enter 换行</span>
        <button
          className="ml-auto px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 border border-indigo-500 text-white hover:bg-indigo-500 transition disabled:opacity-40 disabled:cursor-not-allowed"
          disabled={!modelKeys || !text.trim() || outcome.kind === "working"}
          onClick={() => void submit()}
          title={modelKeys ? "注入这句话；世界若没在跑，会就地推进一步让它落定" : NO_MODEL_KEYS_HINT}
        >
          {outcome.kind === "working" ? "解析中…" : "注入"}
        </button>
      </div>

      {outcome.kind === "refused" && (
        // Advice about the sentence, not a failure — amber, not error red.
        <p className="text-[11px] text-amber-300/80">落实不了：{outcome.reason}</p>
      )}
      {outcome.kind === "queued" && (
        // Accepted means it lands; the backend guarantees it.
        <p className="text-[11px] text-indigo-300/80">已注入：{outcome.preview}</p>
      )}

      {(refused.length > 0 || landed.length > 0) && (
        // Newest on top: this panel answers "what did I just do".
        <div className="max-h-48 overflow-y-auto space-y-1 border-t border-slate-800/70 pt-2">
          {/* Every row: what you said on top, the result below — a refusal is a result too. */}
          {refused.map((r, i) => (
            <div key={`no-${i}`} className="text-[11px] leading-snug opacity-70">
              <div className="text-slate-300">{r.text}</div>
              <div className="text-amber-300/70">落实不了：{r.reason}</div>
            </div>
          ))}
          {[...landed].reverse().map((it) => (
            <div key={`${it.step}-${it.directive_text}`} className="text-[11px] leading-snug">
              <div className="flex gap-2">
                {/* The world's own time, not "第 N 步": that's a scheduling coordinate. */}
                <span className="shrink-0 text-slate-600">{it.time_label}</span>
                <span className="truncate text-slate-300" title={it.directive_text}>
                  {it.directive_text}
                </span>
              </div>
              <div className="text-indigo-300/70">{it.narrative}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

