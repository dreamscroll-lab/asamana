/**
 * ConversationWindow — a live conversation of the performed character's, over the map.
 *
 * The header names the other side with their avatar and name in their identity color, as the
 * character knows them (gender, age, role), and what they are to the character (its own side of
 * the relation). Each dialogue turn carries its speaker's avatar, which says who spoke it. The
 * dialogue comes from the actor's thread; the input is open only while a SPEAK turn waits for the
 * character's dialogue turn, with the turns left and the turn's length cap in view. Once the
 * conversation is over the actor closes the window; the conversation stays in the thread as a card.
 *
 * The window is bounded by the map it sits over and only its dialogue scrolls, so the input is
 * never pushed out of view. The other side's turns arriving while it is open are typed out
 * (Typewriter); turns already there when it opened are shown whole.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../../api/client";
import { avatarStyle } from "../../lib/avatar";
import { DIALOGUE_TYPE_MS } from "../../lib/dialogue";
import type { ActorDialogueTurn, AgentProfile, BriefRelation, Conversation, DialogueTurn } from "../../types";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import SendButton, { SEND_HINT, isSendKey } from "./SendButton";

export default function ConversationWindow({
  worldId,
  selfId,
  self,
  otherId,
  other,
  ink,
  dialogue,
  conversation,
  over,
  onAnswer,
  onClose,
}: {
  worldId: string;
  selfId: string;
  // Profiles; undefined while they load.
  self: AgentProfile | undefined;
  otherId: string;
  other: AgentProfile | undefined;
  ink: (id: string) => string;
  dialogue: DialogueTurn[];
  // The open SPEAK turn's conversation; null while it is not the character's line.
  conversation: Conversation | null;
  over: boolean;
  // Resolves to a refusal's text, "" when the dialogue turn was taken.
  onAnswer: (line: ActorDialogueTurn) => Promise<string>;
  onClose: () => void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [relation, setRelation] = useState<BriefRelation | null>(null);
  // The dialogue turn just sent, shown at once until the thread brings it back: `after` is how
  // many of the character's turns there were when it was sent.
  const [pending, setPending] = useState<{ line: string; after: number } | null>(null);
  const mineCount = dialogue.filter((t) => t.speaker_id === selfId).length;
  useEffect(() => {
    if (pending && mineCount > pending.after) setPending(null);
  }, [pending, mineCount]);
  const scroller = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  // The first turn typed out: the other side's latest turn when the window opened (the window
  // opens as the conversation goes live, so that turn has just arrived), and every turn after it.
  const typedFrom = useRef<number | null>(null);
  if (typedFrom.current === null) {
    const latestOther = dialogue.map((t) => t.speaker_id !== selfId).lastIndexOf(true);
    typedFrom.current = latestOther >= 0 ? latestOther : dialogue.length;
  }

  useEffect(() => {
    let alive = true;
    api
      .actorMind(worldId)
      .then((m) => alive && setRelation(m.relations.find((r) => r.agent_id === otherId) ?? null))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [worldId, otherId]);

  const followBottom = () => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  };
  useEffect(followBottom, [dialogue.length, conversation, over, pending]);

  // The character's dialogue turn is asked for: the cursor goes where the words go.
  useEffect(() => {
    if (conversation) input.current?.focus();
  }, [conversation]);

  const otherName = other?.name || conversation?.other.name || "对方";
  const maxChars = conversation?.max_turn_chars ?? 0;
  const length = [...text.trim()].length;
  const speaking = conversation !== null && !busy;

  const send = async (end: boolean) => {
    const line = text.trim();
    if (!end && !line) return;
    setBusy(true);
    setMessage("");
    if (line) setPending({ line, after: mineCount });
    setText("");
    // A dialogue turn not taken goes back into the box, to be said again.
    const giveBack = (why: string) => {
      setMessage(why);
      setPending(null);
      setText(line);
    };
    try {
      const refusal = await onAnswer({ line, end });
      if (refusal) giveBack(refusal);
    } catch (e) {
      giveBack(String(e));
    } finally {
      setBusy(false);
    }
  };

  const identity = [
    other?.gender || conversation?.other.gender,
    other?.age != null ? `${other.age}岁` : "",
    other?.role,
  ].filter(Boolean).join(" · ");

  return (
    <div className="absolute inset-0 z-40 flex items-center justify-center bg-slate-950/55 p-3 backdrop-blur-[2px] sm:p-6">
      <section className="relative flex max-h-full w-[min(34rem,100%)] flex-col overflow-hidden rounded-2xl border border-slate-700/50 bg-gradient-to-b from-slate-900/95 to-slate-950/95 shadow-[0_24px_60px_-12px_rgba(0,0,0,0.75)] ring-1 ring-white/5">
        <div className="h-px shrink-0 bg-gradient-to-r from-transparent via-indigo-300/25 to-transparent" />

        <header className="shrink-0 px-5 pt-4 pb-3">
          <div className="flex items-center gap-3">
            <Avatar id={otherId} name={otherName} color={other?.color} size="lg" />
            <div className="min-w-0 flex-1">
              <div className="truncate text-[16px] font-bold tracking-wide" style={{ color: ink(otherId) }}>
                {otherName}
              </div>
              {identity && <div className="mt-0.5 truncate text-[12px] text-slate-400">{identity}</div>}
            </div>
          </div>
          <RelationStrip relation={relation} />
        </header>

        <div
          ref={scroller}
          className="min-h-[8rem] flex-1 space-y-3 overflow-y-auto border-y border-slate-800/60 bg-slate-950/40 px-4 py-4"
        >
          {dialogue.length === 0 && !over && conversation && (
            <p className="py-6 text-center text-[12px] text-slate-500">轮到你先开口。</p>
          )}
          {dialogue.map((t, i) => (
            <Bubble
              key={i}
              turn={t}
              mine={t.speaker_id === selfId}
              color={t.speaker_id === selfId ? self?.color : other?.color}
              ink={ink(t.speaker_id)}
              typed={t.speaker_id !== selfId && i >= (typedFrom.current ?? 0)}
              onType={followBottom}
            />
          ))}
          {pending && (
            <Bubble
              turn={{ speaker_id: selfId, speaker: self?.name ?? "", line: pending.line }}
              mine
              pending
              color={self?.color}
              ink={ink(selfId)}
              typed={false}
              onType={followBottom}
            />
          )}
          {!over && !conversation && <Thinking id={otherId} name={otherName} color={other?.color} />}
          {over && (
            <div className="flex items-center gap-3 pt-1 text-[11px] tracking-[0.2em] text-slate-500">
              <span className="h-px flex-1 bg-slate-800" />
              对话结束
              <span className="h-px flex-1 bg-slate-800" />
            </div>
          )}
        </div>

        <footer className="shrink-0 px-4 pt-3 pb-4">
          {message && <p className="mb-2 text-[12px] text-rose-400">{message}</p>}
          {over ? (
            <div className="flex justify-end">
              <button className={BTN_PRIMARY} onClick={onClose}>关闭</button>
            </div>
          ) : (
            <>
              <div
                className={`rounded-xl border bg-slate-950/80 transition ${
                  speaking ? "border-indigo-500/50 shadow-[0_0_0_3px_rgba(99,102,241,0.12)]" : "border-slate-800"
                }`}
              >
                <textarea
                  ref={input}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => {
                    if (!isSendKey(e)) return;
                    e.preventDefault();
                    if (speaking) void send(false);
                  }}
                  disabled={!speaking}
                  maxLength={maxChars || undefined}
                  rows={2}
                  placeholder={conversation ? `对${otherName}说…` : `${otherName}正在说…`}
                  className="block w-full resize-none bg-transparent px-3 pt-2.5 pb-1 text-[13px] leading-relaxed text-slate-100 placeholder:text-slate-600 focus:outline-none disabled:opacity-50"
                />
                <div className="flex items-center gap-2 px-3 pb-2">
                  <span className="text-[11px] text-slate-500">{conversation ? turnsHint(conversation) : ""}</span>
                  {conversation && (
                    <span
                      className={`ml-auto text-[10px] tabular-nums ${length >= maxChars ? "text-amber-400" : "text-slate-600"}`}
                    >
                      {length}/{maxChars}
                    </span>
                  )}
                </div>
              </div>
              <div className="mt-2.5 flex items-center gap-2">
                <span className="text-[10px] text-slate-600">{SEND_HINT}</span>
                <button
                  className={`${BTN} ml-auto`}
                  disabled={!speaking}
                  onClick={() => void send(true)}
                  title="说出框里的话（可以不写）后结束对话"
                >
                  结束
                </button>
                <SendButton disabled={!speaking || !text.trim()} busy={busy} onClick={() => void send(false)} />
              </div>
            </>
          )}
        </footer>
      </section>
    </div>
  );
}

/** A character's avatar, as everywhere else: their identity color, their name's first character. */
function Avatar({ id, name, color, size }: { id: string; name: string; color: string | undefined; size: "sm" | "lg" }) {
  const box = { sm: "h-7 w-7 rounded-lg text-[12px]", lg: "h-12 w-12 rounded-2xl text-[21px]" }[size];
  return (
    <span
      title={name}
      className={`${box} grid shrink-0 place-items-center border border-slate-100/10 font-black text-white shadow-[0_6px_16px_rgba(0,0,0,0.45)]`}
      style={avatarStyle(color, id)}
    >
      {name[0]}
    </span>
  );
}

function Bubble({
  turn,
  mine,
  color,
  ink,
  typed,
  onType,
  pending = false,
}: {
  turn: DialogueTurn;
  mine: boolean;
  // Sent, not yet back from the thread.
  pending?: boolean;
  color: string | undefined;
  ink: string;
  typed: boolean;
  onType: () => void;
}) {
  return (
    <div className={`flex items-end gap-2 transition-opacity ${mine ? "flex-row-reverse" : ""} ${pending ? "opacity-60" : ""}`}>
      <Avatar id={turn.speaker_id} name={turn.speaker} color={color} size="sm" />
      <div className={`flex max-w-[78%] flex-col ${mine ? "items-end" : "items-start"}`}>
        <div
          className={`px-3.5 py-2 text-[13px] leading-relaxed text-slate-100 shadow-[0_4px_12px_-6px_rgba(0,0,0,0.7)] ${
            mine
              ? "rounded-2xl rounded-br-md bg-indigo-600/40 ring-1 ring-indigo-400/20"
              : "rounded-2xl rounded-bl-md border-l-2 bg-slate-800/70 ring-1 ring-white/5"
          }`}
          style={mine ? undefined : { borderLeftColor: ink }}
        >
          {typed ? <Typewriter text={turn.line} onType={onType} /> : turn.line}
        </div>
      </div>
    </div>
  );
}

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** A dialogue turn's line typed out character by character at a steady pace; a click shows it
 *  whole. */
function Typewriter({ text, onType }: { text: string; onType: () => void }) {
  const chars = useMemo(() => [...text], [text]);
  const [shown, setShown] = useState(() => (prefersReducedMotion() ? chars.length : 0));
  const done = shown >= chars.length;

  useEffect(() => {
    if (done) return;
    const timer = setTimeout(() => setShown((n) => n + 1), DIALOGUE_TYPE_MS);
    return () => clearTimeout(timer);
  }, [shown, done, chars]);
  useEffect(onType, [shown, onType]);

  return (
    <span onClick={() => setShown(chars.length)} className={done ? "" : "cursor-pointer"} title={done ? undefined : "点一下看全句"}>
      {chars.slice(0, shown).join("")}
      {!done && <span className="ml-px inline-block h-[1em] w-[2px] translate-y-[2px] animate-pulse bg-slate-300/80" />}
    </span>
  );
}

/** The other side is finding its words. */
function Thinking({ id, name, color }: { id: string; name: string; color: string | undefined }) {
  return (
    <div className="flex items-end gap-2" title={`${name}在想…`}>
      <Avatar id={id} name={name} color={color} size="sm" />
      <div className="flex gap-1 rounded-2xl rounded-bl-md bg-slate-800/70 px-3.5 py-3 ring-1 ring-white/5">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
    </div>
  );
}

/** How much of the conversation is left, this turn included. */
function turnsHint(c: Conversation): string {
  return c.last ? "最后一轮：说完对话就结束" : `还剩 ${c.turns_left} 轮`;
}

/** What the other side is to the character: its own labels, and where trust and affection stand. */
function RelationStrip({ relation }: { relation: BriefRelation | null }) {
  if (!relation) {
    return <div className="mt-3 text-[12px] text-slate-500">与我 · 素不相识</div>;
  }
  return (
    <div className="mt-3 space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
        <span className="text-slate-500">与我</span>
        {relation.labels.map((label) => (
          <span key={label} className="rounded-full border border-slate-700/70 bg-slate-800/60 px-2 py-0.5 text-[11px] text-slate-200">
            {label}
          </span>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3">
        <Meter label="信任" value={relation.trust} min={0} max={1} tone="bg-teal-400/80" />
        <Meter label="好感" value={relation.affection} min={-1} max={1} tone={relation.affection >= 0 ? "bg-rose-400/80" : "bg-slate-400/80"} />
      </div>
      {relation.summary && <p className="text-[11px] leading-relaxed text-slate-500">{relation.summary}</p>}
    </div>
  );
}

function Meter({ label, value, min, max, tone }: { label: string; value: number; min: number; max: number; tone: string }) {
  const ratio = Math.min(1, Math.max(0, (value - min) / (max - min)));
  return (
    <div className="flex items-center gap-2 text-[11px] text-slate-500">
      <span className="shrink-0">{label}</span>
      <div className="h-1 flex-1 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full rounded-full ${tone}`} style={{ width: `${Math.round(ratio * 100)}%` }} />
      </div>
      <span className="w-9 shrink-0 text-right tabular-nums text-slate-400">
        {min < 0 && value >= 0 ? "+" : ""}
        {value.toFixed(2)}
      </span>
    </div>
  );
}
