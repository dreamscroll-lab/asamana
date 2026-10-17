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
 * The history is an index read from `GET /directives`, because the feed only
 * holds steps you watched. A row is 「what I said → what the engine did」; the receipt
 * (delivered / pressed / decided / interrupted) stays on the feed card, since it means
 * nothing without the step around it. Refusals are kept locally and not persisted: they
 * changed nothing in the world.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { api } from "../api/client";
import { NO_MODEL_KEYS_HINT, useModelKeys } from "../lib/deployment";
import type { Intervention } from "../types";
import SendButton, { SEND_HINT, isSendKey } from "./SendButton";
import { buildThread, nextOccurrence, type Submission, type SubmissionState } from "./director/thread";
import {
  CONSOLE_USER_MESSAGE, CONSOLE_WORLD_MESSAGE, CONSOLE_COLLAPSED, CONSOLE_PANEL, CONSOLE_HEADER,
  CONSOLE_CLOSE, CONSOLE_THREAD, CONSOLE_FOOTER, CONSOLE_INPUT,
} from "./consoleStyles";

/** Past this height the input scrolls internally; any taller and it covers the map. */
const INPUT_MAX_PX = 132;
const FOLLOW_BOTTOM_PX = 48;
const MAX_DIRECTIVE_CHARS = 500;

/** `step` is only a refresh trigger (receipts arrive at the end of the step). Never
 *  rendered: 「第 N 步」 is a code-layer ordinal. */
export default function DirectorBar({ worldId, step }: { worldId: string; step: number | null }) {
  // Collapsed by default: a rarely used tool shouldn't cover the map that is usually watched.
  const [collapsed, setCollapsed] = useState(true);
  const [text, setText] = useState("");
  const [landed, setLanded] = useState<Intervention[]>([]);
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const submissionId = useRef(0);
  const submitting = useRef(false);
  const reloadVersion = useRef(0);
  const history = useRef<Intervention[] | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const box = useRef<HTMLTextAreaElement>(null);
  const modelKeys = useModelKeys();
  const working = submissions.some((item) => item.state.kind === "working");
  const items = buildThread(landed, submissions);

  // Auto-grow: reset to zero before measuring scrollHeight, otherwise it only grows and never
  // shrinks.
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_PX)}px`;
  }, [text, collapsed]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && followLatest.current) el.scrollTop = el.scrollHeight;
  }, [landed, submissions, collapsed]);

  const reload = useCallback(async () => {
    const version = ++reloadVersion.current;
    try {
      const records = await api.listDirectives(worldId);
      if (version === reloadVersion.current) {
        history.current = records;
        setLanded(records);
      }
      return records;
    } catch {
      // Failing to read history shouldn't block the input.
      return null;
    }
  }, [worldId]);

  // Only read while expanded: collapsed, nobody is looking at the record.
  useEffect(() => {
    if (!collapsed) void reload();
  }, [collapsed, step, reload]);

  async function submit() {
    const value = text.trim();
    if (!modelKeys || !value || submitting.current) return;
    submitting.current = true;
    followLatest.current = true;
    const id = ++submissionId.current;
    setSubmissions((previous) => [{
      id, text: value, occurrence: null, after: history.current?.length ?? null, state: { kind: "working" },
    }, ...previous]);
    setText("");
    const settle = (state: SubmissionState): void => {
      setSubmissions((previous) => previous.map((item) => item.id === id ? { ...item, state } : item));
    };
    try {
      // Only history read before POST can distinguish this sentence from an identical earlier one.
      const baseline = history.current ?? await reload();
      setSubmissions((previous) => previous.map((item) => item.id === id ? {
        ...item, after: baseline?.length ?? null,
        occurrence: nextOccurrence(value, baseline, previous.filter((other) => other.id !== id)),
      } : item));
      const res = await api.direct(worldId, value);
      if (res.accepted) {
        settle({ kind: "accepted", preview: res.preview });
        // The backend steps a stopped world itself, so the snapshot is usually written by now;
        // in a running world the effect above catches up on the next step.
        void reload();
      } else {
        settle({ kind: "refused", reason: res.reason });
        setText((draft) => draft || value);
      }
    } catch (e) {
      settle({ kind: "failed", reason: String(e) });
      setText((draft) => draft || value);
    } finally {
      submitting.current = false;
    }
  }

  if (collapsed) {
    return (
      <button
        className={CONSOLE_COLLAPSED}
        onClick={() => { followLatest.current = true; setCollapsed(false); }}
        title="展开导演台"
      >
        🎬
      </button>
    );
  }

  return (
    <section className={`${CONSOLE_PANEL} flex`}>
      <div className={CONSOLE_HEADER}>
        <div className="flex items-start gap-2">
          <span className="text-sm pt-1" title="导演台">🎬</span>
          <span className="min-w-0 flex-1 pt-1 text-[13px] text-slate-300">导演台</span>
          <button
            className={CONSOLE_CLOSE}
            onClick={() => setCollapsed(true)}
            title="收起导演台"
          >
            收起 ⌄
          </button>
        </div>
      </div>

      <div ref={scroller} className={CONSOLE_THREAD} onScroll={(event) => {
        const el = event.currentTarget;
        followLatest.current = el.scrollHeight - el.scrollTop - el.clientHeight <= FOLLOW_BOTTOM_PX;
      }}>
        {items.map((item) => {
          if (item.kind === "landed") {
            const it = item.intervention;
            return (
              <div key={item.key} className="space-y-2">
                <div className="pt-1 text-center text-[11px] text-slate-500">{it.time_label}</div>
                <div className="flex justify-end">
                  <div className={`${CONSOLE_USER_MESSAGE} whitespace-pre-wrap`}>{it.directive_text}</div>
                </div>
                <div className={`${CONSOLE_WORLD_MESSAGE} whitespace-pre-wrap`}>{it.narrative}</div>
              </div>
            );
          }
          const { text: words, state } = item.submission;
          const status = {
            working: "解析中…", accepted: "已注入", refused: "未注入", failed: "结果未确认",
          }[state.kind];
          return (
            <div key={item.key} className="space-y-2">
              <div className="flex justify-end">
                <div className={`${CONSOLE_USER_MESSAGE} whitespace-pre-wrap ${state.kind === "working" ? "opacity-60" : ""}`}>{words}</div>
              </div>
              <div role="status" className="text-right text-[10px] text-slate-500">
                {status}
              </div>
              {state.kind === "accepted" && (
                <div className={`${CONSOLE_WORLD_MESSAGE} whitespace-pre-wrap`}>{state.preview}</div>
              )}
              {state.kind === "refused" && (
                <div className={`${CONSOLE_WORLD_MESSAGE} whitespace-pre-wrap text-amber-300/80`}>落实不了：{state.reason}</div>
              )}
              {state.kind === "failed" && (
                <div className={`${CONSOLE_WORLD_MESSAGE} whitespace-pre-wrap text-rose-300`}>未能确认注入结果：{state.reason}</div>
              )}
            </div>
          );
        })}
      </div>

      <div className={CONSOLE_FOOTER}>
        <textarea
          ref={box}
          value={text}
          rows={2}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (!isSendKey(e)) return;
            e.preventDefault();
            void submit();
          }}
          className={`${CONSOLE_INPUT} overflow-y-auto`}
          placeholder="写下你想让这个世界发生什么…"
          maxLength={MAX_DIRECTIVE_CHARS}
        />

        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-600">{SEND_HINT}</span>
          <span className="ml-auto text-[10px] text-slate-600 tabular-nums">
            {[...text].length}/{MAX_DIRECTIVE_CHARS}
          </span>
          <SendButton
            disabled={!modelKeys || !text.trim()}
            busy={working}
            onClick={() => void submit()}
            title={modelKeys ? "注入这句话；世界若没在跑，会就地推进一步让它落定" : NO_MODEL_KEYS_HINT}
          />
        </div>
      </div>
    </section>
  );
}

