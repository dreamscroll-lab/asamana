/**
 * ActorConsole — the actor's conversation with the world, floating over the map's top-right (the
 * corner the director console takes in director mode) and collapsible the same way.
 *
 * Top to bottom: the character's mind (MindPanel, folded to its tabs) and the actor's pace; the
 * thread (thread.ts): what reached the character, folded; the world's questions as cards; the
 * actor's words and the world's action summary of them, confirmed on the card; each act with its
 * progress, and how it ended as the character remembers it; each live conversation as a folded
 * card; then the actor's words box. A live conversation going on is spoken in its own window over
 * the map (ConversationWindow), which the actor closes once it is over.
 *
 * The actor speaks only when the world asks what to do; at any other time they can only break
 * off an act spanning steps. The character's body, belongings and surroundings are the map's.
 *
 * A new turn opens the console even if it was collapsed: the world is waiting on it, so it must
 * not stay hidden behind an icon.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { identityOf } from "../../lib/worldTable";
import ActorIcon from "../ActorIcon";
import { api } from "../../api/client";
import { useCast } from "../../lib/cast";
import { identityColor, toHex } from "../../lib/color";
import { durationLabel } from "../../lib/duration";
import type {
  ActorPublicAnswer, ActorBriefing, ActorActionSummary, ActorEntry, ActorDialogueTurn, ActorMind, ActorPace, ActorTurn, CharacterIdentity, RunState,
} from "../../types";
import { ActionType } from "../../lib/contract";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import {
  CONSOLE_USER_MESSAGE, CONSOLE_WORLD_MESSAGE, CONSOLE_RAISED, CONSOLE_COLLAPSED, CONSOLE_PANEL, CONSOLE_HEADER,
  CONSOLE_CLOSE, CONSOLE_THREAD, CONSOLE_FOOTER, CONSOLE_INPUT,
} from "../consoleStyles";
import Transcript from "../narrative/Transcript";
import ConversationWindow from "./ConversationWindow";
import MindPanel from "./MindPanel";
import InterruptQuestion from "./InterruptQuestion";
import { goalChoices, type GoalRow } from "./goals";
import SendButton, { SEND_HINT, isSendKey } from "../SendButton";
import { useActorSubmission } from "./submission";
import { describeActionSummary, describeIntent, reasonText } from "./actionSummary";
import { breakableAct, buildThread, conversationsOf, liveConversation, type ThreadItem } from "./thread";

// The backend's cap on what the actor says at once (interaction/api/actor.py).
const MAX_WORDS = 500;

// Only the pressing levels are worth a mark; the rest read as plain words.
const URGENCY_MARK: Record<string, string> = { high: "（急）", critical: "（十万火急）" };


// A person's name in their identity color.
type Ink = (id: string) => string;

export default function ActorConsole({
  worldId,
  selfId,
  entries,
  turn,
  runState,
  step,
  pace: initialPace,
  immersive,
}: {
  worldId: string;
  // The performed character.
  selfId: string;
  // The actor's thread, in order.
  entries: ActorEntry[];
  turn: ActorTurn | null;
  runState: RunState;
  // The latest step shown: a new one may have moved the character's mind.
  step: number;
  pace: ActorPace;
  immersive: boolean;
}) {
  const [collapsedOverride, setCollapsed] = useState<boolean | null>(null);
  const collapsed = collapsedOverride ?? !immersive;
  const [pace, setPace] = useState(initialPace);
  const [text, setText] = useState("");
  const { busy, message, run } = useActorSubmission();
  const [mind, setMind] = useState<ActorMind | null>(null);
  // Words just said, shown at once until their action summary comes back as a `said` entry: `after` is
  // how many there were when they were said.
  const [pending, setPending] = useState<{ words: string; after: number } | null>(null);
  const saidCount = entries.filter((e) => e.kind === "said").length;
  useEffect(() => {
    if (pending && saidCount > pending.after) setPending(null);
  }, [pending, saidCount]);
  const scroller = useRef<HTMLDivElement>(null);

  const turnId = turn?.turn_id ?? null;
  const currentTurn = useRef(turnId);
  currentTurn.current = turnId;
  const running = runState === "running" || runState === "stopping";
  const deciding = turn?.kind === "decide";
  const items = useMemo(() => buildThread(entries, turn), [entries, turn]);
  const latestSaid = [...items].reverse().find((item) => item.kind === "said");
  useEffect(() => { if (latestSaid) setCollapsed(false); }, [latestSaid?.key]);
  const breakable = breakableAct(items);
  const choices = useMemo(() => mind ? goalChoices(mind) : [], [mind]);
  const speaking = turn?.kind === "speak" ? turn.conversation : null;
  const conversations = useMemo(() => conversationsOf(items), [items]);
  const live = liveConversation(items);

  // The conversation in the window, by its place among the thread's conversations, and whom
  // with. When the character speaks first it has no turn in the thread yet, so its place is the
  // next one; until it is there it is not over. Kept until the actor closes the window, unless
  // another conversation goes live.
  const [windowed, setWindowed] = useState<{ index: number; otherId: string } | null>(null);
  const shown = windowed ? conversations[windowed.index] ?? null : null;
  const shownOver = shown?.over ?? false;
  useEffect(() => {
    if (!speaking && !live) return;
    if (windowed && !shownOver) return;
    const otherId =
      speaking?.other.agent_id ?? live?.dialogue.find((t) => t.speaker_id !== selfId)?.speaker_id;
    if (!otherId) return;
    setWindowed({ index: live ? conversations.indexOf(live) : conversations.length, otherId });
  }, [speaking, live, windowed, shownOver, selfId, conversations]);

  const { cast } = useCast(worldId);
  const ink: Ink = (id) => toHex(identityColor(identityOf(id, cast)?.color, id));

  useEffect(() => {
    // A conversation asks turn after turn in its own window; only other turns open the console.
    if (turnId && turn?.kind !== "speak") setCollapsed(false);
    setPending(null);
  }, [turnId, turn?.kind]);

  // Follow the newest line, as a chat does.
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items.length, turnId, collapsed, pending]);

  // Each answer comes back over the socket as entries of the thread; only a refusal is said here.
  const say = (choice?: string) => run(async () => {
    if (!turn || turn.kind !== "decide") return "";
    const started = turn.turn_id;
    const words = (choice ?? text).trim();
    if (!words) return "";
    setPending({ words, after: saidCount });
    if (choice === undefined) setText("");
    try {
      const result = await api.actorIntent(worldId, { turn_id: turn!.turn_id, text: words });
      if (currentTurn.current !== started) return "";
      if (result.status === "rejected") {
        setPending(null);
        return result.reason === "stale_intent" ? reasonText(result.reason) : "";
      }
    } catch (e) {
      // Not taken: the words go back into the box, to be said again.
      if (currentTurn.current === started) {
        setPending(null);
        if (choice === undefined) setText(words);
      }
      throw e;
    }
    return "";
  });
  const answer = (a: ActorPublicAnswer) => run(async () => {
    if (!turn) return "";
    const r = await api.answerActor(worldId, turn.turn_id, a);
    return r.accepted ? "" : reasonText(r.reason);
  });
  const speak = async (line: ActorDialogueTurn): Promise<string> => {
    if (!turn) return "";
    const r = await api.answerActor(worldId, turn.turn_id, line);
    return r.accepted ? "" : reasonText(r.reason);
  };
  const breakOff = () => run(async () => {
    const r = await api.breakOffActor(worldId);
    return r.accepted ? "" : reasonText(r.reason);
  });
  const switchPace = (next: ActorPace) => run(async () => {
    await api.setActorPace(worldId, next);
    setPace(next);
    return "";
  });

  return (
    <>
      {windowed && (
        <ConversationWindow
          worldId={worldId}
          selfId={selfId}
          self={identityOf(selfId, cast)}
          otherId={windowed.otherId}
          other={identityOf(windowed.otherId, cast)}
          ink={ink}
          dialogue={shown?.dialogue ?? []}
          conversation={speaking}
          over={!speaking && shownOver}
          onAnswer={speak}
          onClose={() => setWindowed(null)}
        />
      )}
      {collapsed && (
        <button
          className={CONSOLE_COLLAPSED}
          onClick={() => setCollapsed(false)}
          title="展开演员台"
        >
          <ActorIcon />
          {turn && <span className="absolute top-0 right-0 w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse" />}
        </button>
      )}
      {/* Collapsing only hides the console: unsent words and an unsaved edit to the mind survive it. */}
      <section
        className={`${CONSOLE_PANEL} ${collapsed ? "hidden" : "flex"}`}
      >
        <div className={CONSOLE_HEADER}>
          <div className="flex items-start gap-2">
            <span className="text-sm pt-1" title="演员台"><ActorIcon /></span>
            <div className="min-w-0 flex-1 max-h-72 overflow-y-auto">
              <MindPanel worldId={worldId} refreshKey={`${step}:${turnId ?? ""}`} turnId={turnId} running={running}
                immediate={turn?.kind === "decide" || turn?.kind === "interrupt"} mind={mind} onMindChange={setMind} />
            </div>
            <button
              className={CONSOLE_CLOSE}
              onClick={() => setCollapsed(true)}
              title="收起演员台"
            >
              收起 ⌄
            </button>
          </div>
          <label className="flex items-center gap-1.5 text-[11px] text-slate-400" title="打开后，角色闲着时只在有事找上门、或手上的事刚结束时才问你">
            <input
              type="checkbox"
              checked={pace === "when_reached"}
              disabled={busy}
              onChange={(e) => void switchPace(e.target.checked ? "when_reached" : "every_step")}
            />
            有事才问我
          </label>
        </div>

        <div ref={scroller} className={CONSOLE_THREAD}>
          {items.length === 0 && (
            <p className="text-[12px] text-slate-600">在这里与世界互动...</p>
          )}
          {items.map((item) => (
            <Line
              key={item.key}
              item={item}
              busy={busy}
              selfId={selfId}
              self={{ agent_id: selfId, name: identityOf(selfId, cast)?.name ?? "自己" }}
              ink={ink}
              onAnswer={answer}
              onBreakOff={breakOff}
              canBreakOff={breakable?.key === item.key && !breakable.breakingOff}
              choices={choices}
              onChoose={(words) => void say(words)}
            />
          ))}
          {pending && (
            <div className="flex justify-end">
              <div className={`${CONSOLE_USER_MESSAGE} opacity-60`}>{pending.words}</div>
            </div>
          )}
          {!turn && running && <p className="text-[12px] text-slate-500">世界在忙碌着...</p>}
        </div>

        <div className={CONSOLE_FOOTER}>
          {message && <p className="text-[12px] text-rose-400">{message}</p>}
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (!isSendKey(e)) return;
              e.preventDefault();
              if (deciding && !busy) void say();
            }}
            disabled={!deciding}
            maxLength={MAX_WORDS}
            rows={2}
            placeholder={deciding ? "做你想做的任何事儿～" : "别着急，再等等，世界很忙的～～"}
            className={CONSOLE_INPUT}
          />
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-600">{SEND_HINT}</span>
            <span className="ml-auto text-[10px] text-slate-600 tabular-nums">
              {[...text].length}/{MAX_WORDS}
            </span>
            <SendButton disabled={!deciding || !text.trim()} busy={busy} onClick={() => void say()} />
          </div>
        </div>
      </section>
    </>
  );
}

// ----- lines ---------------------------------------------------------------------------------

function Line({
  item,
  busy,
  selfId,
  self,
  ink,
  onAnswer,
  onBreakOff,
  canBreakOff,
  choices,
  onChoose,
}: {
  item: ThreadItem;
  busy: boolean;
  selfId: string;
  self: Pick<CharacterIdentity, "agent_id" | "name">;
  ink: Ink;
  onAnswer: (a: ActorPublicAnswer) => void;
  onBreakOff: () => void;
  canBreakOff: boolean;
  choices: GoalRow[];
  onChoose: (words: string) => void;
}) {
  switch (item.kind) {
    case "time":
      return <div className="pt-1 text-center text-[11px] text-slate-500">{item.label}</div>;
    case "perceived":
      return <Perceived item={item} />;
    case "asked":
      return <Asked item={item} busy={busy} ink={ink} onAnswer={onAnswer} choices={choices} onChoose={onChoose} />;
    case "said":
      return <Said item={item} busy={busy} self={self} onAnswer={onAnswer} />;
    case "chose":
      return (
        <div className="flex justify-end">
          <div className={CONSOLE_USER_MESSAGE}>
            <div>{CHOICE_TEXT[item.choice]}</div>
            {item.description && <p className="text-xs opacity-80 whitespace-pre-wrap">{item.description}</p>}
          </div>
        </div>
      );
    case "act":
      return <Act item={item} busy={busy} canBreakOff={canBreakOff} onBreakOff={onBreakOff} />;
    case "ended":
      return <Ended item={item} />;
    case "conversation":
      return <ConversationCard item={item} selfId={selfId} ink={ink} />;
  }
}

const CHOICE_TEXT: Record<Extract<ThreadItem, { kind: "chose" }>["choice"], string> = {
  idle: "⏸ 这次先不动",
  carry_on: "▶ 继续手上的事",
  break_off: "✋ 停下手上的事",
  accept: "✔ 接受",
  refuse: "✘ 拒绝",
};

/** What reached the character: folded to a line, open while it is what the world is asking about. */
function Perceived({ item }: { item: Extract<ThreadItem, { kind: "perceived" }> }) {
  const [toggled, setToggled] = useState<boolean | null>(null);
  const open = toggled ?? item.asking;
  const count = item.messages.length + item.broadcasts.length + item.acted_upon.length;
  return (
    <div className={`${CONSOLE_WORLD_MESSAGE} ${item.asking ? "border-amber-800/60" : ""}`}>
      <button type="button" onClick={() => setToggled(!open)} className="flex w-full items-baseline gap-2 text-left">
        <span className="text-[12px] text-slate-400">
          <span className="text-slate-600">{open ? "▾" : "▸"} </span>最新动态
        </span>
        <span className="ml-auto text-[11px] text-slate-500">{count} 件</span>
      </button>
      {open && (
        <ul className="mt-1 space-y-0.5">
          {item.messages.map((m, i) => (
            <li key={`m${i}`}>
              <span className="text-slate-500">✉ </span>
              {m.sender_name && <span className="text-slate-400">{m.sender_name}：</span>}
              {m.content}
              {URGENCY_MARK[m.urgency] && <span className="text-amber-400">{URGENCY_MARK[m.urgency]}</span>}
            </li>
          ))}
          {item.broadcasts.map((b, i) => (
            <li key={`b${i}`}><span className="text-slate-500">📣 </span>{b}</li>
          ))}
          {item.acted_upon.map((fact, i) => (
            <li key={`a${i}`}><span className="text-slate-500">⚡ </span>{fact}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** A question from the world. A decision asked and since answered leaves no card: its answer says it. */
function Asked({
  item,
  busy,
  ink,
  onAnswer,
  choices,
  onChoose,
}: {
  item: Extract<ThreadItem, { kind: "asked" }>;
  busy: boolean;
  ink: Ink;
  onAnswer: (a: ActorPublicAnswer) => void;
  choices: GoalRow[];
  onChoose: (words: string) => void;
}) {
  const { turn, open } = item;
  if (turn.kind === "invite") {
    const inviter = turn.invitation?.inviter;
    const who = inviter?.name || "有人";
    return (
      <div className={`${CONSOLE_WORLD_MESSAGE} ${open ? "border-amber-800/60" : "opacity-70"} space-y-2`}>
        <p>
          <span style={inviter ? { color: ink(inviter.agent_id) } : undefined}>{who}</span>
          {turn.invitation?.action_type === ActionType.talk ? "想与你交谈。" : "邀你一起行动。"}
          {turn.decided && <span className="text-slate-400">接受邀请就要放下：{turn.decided}。</span>}
        </p>
        {open && (
          <div className="flex gap-2">
            <button className={BTN} disabled={busy} onClick={() => onAnswer({ accept: true })}>✔ 接受</button>
            <button className={BTN} disabled={busy} onClick={() => onAnswer({ accept: false })}>✘ 拒绝</button>
          </div>
        )}
      </div>
    );
  }
  if (turn.kind === "decide") {
    if (!open) return null;
    return (
      <div className={`${CONSOLE_WORLD_MESSAGE} border-indigo-800/60 space-y-2`}>
        <p>{turn.reason && `${turn.reason}。`}想做什么？</p>
        {choices.length > 0 && (
          <div className="space-y-1">
            <p className="text-[11px] text-slate-500">也可以继续这些事 · 选择后确认行动</p>
            {choices.map((g, i) => (
              <button key={g.id ?? `new-${i}`} className={`${BTN} block w-full text-left whitespace-normal break-words`}
                disabled={busy || [...g.text].length > MAX_WORDS} onClick={() => onChoose(g.text)}>
                <span className="text-slate-500 tabular-nums">{i + 1}. </span>
                <span className="text-[11px] text-slate-500">短期目标 · </span>
                {g.text}
                {g.detail?.time_hint && <span className={`text-[11px] ${g.detail.attention ? "text-amber-300" : "text-slate-500"}`}>{" "}{g.detail.time_hint}</span>}
              </button>
            ))}
          </div>
        )}
        <button className={BTN} disabled={busy} onClick={() => onAnswer({ idle: true })}>
          ⏸ 这次先不动
        </button>
        {turn.briefing && <Reference briefing={turn.briefing} />}
      </div>
    );
  }
  return <InterruptQuestion turn={turn} open={open} busy={busy} maxLength={MAX_WORDS} onAnswer={onAnswer} />;
}

function Said({
  item,
  busy,
  self,
  onAnswer,
}: {
  item: Extract<ThreadItem, { kind: "said" }>;
  busy: boolean;
  self: Pick<CharacterIdentity, "agent_id" | "name">;
  onAnswer: (a: ActorPublicAnswer) => void;
}) {
  const submitted = item.intent && item.briefing ? describeIntent(item.intent, item.briefing, self) : null;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-end">
        <div className={CONSOLE_USER_MESSAGE}>{submitted ? <>
          <div>{submitted.headline}</div>
          {submitted.details.map((detail, index) => <div key={index} className="text-xs opacity-80">{detail}</div>)}
        </> : item.words}</div>
      </div>
      {item.action_summary && item.briefing ? (
        <ActionSummaryCard
          actionSummary={item.action_summary}
          briefing={item.briefing}
          self={self}
          state={item.state}
          busy={busy}
          onConfirm={() => onAnswer({ action_summary_id: item.action_summary!.action_summary_id })}
        />
      ) : (
        <div role="alert" className={`${CONSOLE_WORLD_MESSAGE} border border-rose-400/30 bg-rose-950/30 text-rose-200`}>{reasonText(item.reason)}</div>
      )}
    </div>
  );
}

function ActionSummaryCard({
  actionSummary,
  briefing,
  state,
  busy,
  self,
  onConfirm,
}: {
  actionSummary: ActorActionSummary;
  briefing: ActorBriefing;
  state: Extract<ThreadItem, { kind: "said" }>["state"];
  busy: boolean;
  self: Pick<CharacterIdentity, "agent_id" | "name">;
  onConfirm: () => void;
}) {
  const text = describeActionSummary(actionSummary, briefing, self);
  return (
    <div
      className={`max-w-[92%] rounded-xl rounded-tl-sm border px-3 py-2 ${CONSOLE_RAISED} ${
        state === "declined" ? "border-slate-800/60 bg-slate-900/30 opacity-60" : "border-indigo-800/50 bg-indigo-950/40"
      }`}
    >
      <div className="text-[13px] font-semibold text-slate-100">{text.headline}</div>
      <div className="text-[12px] leading-relaxed text-slate-300">预计耗时：{actionSummary.estimated_duration_seconds / 3600}h</div>
      {text.details.map((d, i) => (
        <div key={i} className="text-[12px] leading-relaxed text-slate-300">{d}</div>
      ))}
      <div className="mt-1.5 flex items-center justify-end gap-2">
        {state === "pending" && (
          <button className={BTN_PRIMARY} disabled={busy} onClick={onConfirm}>确认</button>
        )}
        {state === "confirmed" && <span className="text-[12px] text-teal-400">☑ 已确认</span>}
        {state === "declined" && <span className="text-[12px] text-slate-500">☐ 没接受</span>}
      </div>
    </div>
  );
}

function Act({
  item,
  busy,
  canBreakOff,
  onBreakOff,
}: {
  item: Extract<ThreadItem, { kind: "act" }>;
  busy: boolean;
  canBreakOff: boolean;
  onBreakOff: () => void;
}) {
  const ratio = item.total > 0 ? item.elapsed / item.total : 1;
  return (
    <div className={CONSOLE_WORLD_MESSAGE}>
      <div>
        {!item.ended && <span className="text-slate-400">正在：</span>}
        {item.description || "（一件事）"}
      </div>
      <div className="mt-1.5 flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800">
          <div
            className={`h-full rounded-full ${item.ended ? "bg-teal-500/70" : "bg-indigo-500/80"}`}
            style={{ width: `${Math.round(ratio * 100)}%` }}
          />
        </div>
        {canBreakOff && (
          <button className={BTN} disabled={busy} onClick={onBreakOff} title="在下一刻停下手上的事">
            ✋ 中断
          </button>
        )}
        {item.breakingOff && !item.ended && <span className="text-[11px] text-amber-400">即将停下</span>}
      </div>
    </div>
  );
}

function Ended({ item }: { item: Extract<ThreadItem, { kind: "ended" }> }) {
  if (!item.fact) return <div className={`${CONSOLE_WORLD_MESSAGE} text-slate-500`}>（没有下文）</div>;
  const mark = !item.happened
    ? <span className="text-slate-500">未成事 · </span>
    : item.succeeded
      ? <span className="text-teal-400">✓ </span>
      : <span className="text-rose-400">✗ </span>;
  return (
    <div className={`${CONSOLE_WORLD_MESSAGE} ${item.happened ? "" : "text-slate-400"}`}>
      {mark}
      {item.fact}
    </div>
  );
}

/** A live conversation: both sides named in their colors, folded to that line. */
function ConversationCard({
  item,
  selfId,
  ink,
}: {
  item: Extract<ThreadItem, { kind: "conversation" }>;
  selfId: string;
  ink: Ink;
}) {
  const [open, setOpen] = useState(false);
  const names = new Map<string, string>();
  for (const t of item.dialogue) names.set(t.speaker_id, t.speaker);
  // The character first.
  const sides = [...names].sort(([a], [b]) => Number(b === selfId) - Number(a === selfId));
  return (
    <div className={CONSOLE_WORLD_MESSAGE}>
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full items-baseline gap-2 text-left">
        <span className="text-[12px]">
          <span className="text-slate-600">{open ? "▾" : "▸"} </span>💬{" "}
          {sides.map(([id, name], i) => (
            <span key={id}>
              {i > 0 && <span className="text-slate-500"> ⇄ </span>}
              <span className="font-semibold" style={{ color: ink(id) }}>{name}</span>
            </span>
          ))}
          {sides.length === 0 && <span className="text-slate-400">一场没说出话的交谈</span>}
        </span>
        <span className="ml-auto text-[11px] text-slate-500">
          {item.over ? `${item.dialogue.length} 轮` : "交谈中"}
        </span>
      </button>
      {open && <Transcript turns={item.dialogue} inkOf={ink} />}
    </div>
  );
}

/** What a decision can reach beyond what is in front of the character (that is the map's). */
function Reference({ briefing: b }: { briefing: ActorBriefing }) {
  const [open, setOpen] = useState(false);
  const places = b.places.map((p) => `${p.name}${p.travel_seconds ? `（${durationLabel(p.travel_seconds)}路程）` : ""}`);
  const absent = b.people.filter((p) => !p.present).map((p) => `${p.name}${p.gender ? `（${p.gender}）` : ""}`);
  if (!places.length && !absent.length) return null;
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="text-[11px] font-semibold text-slate-400 hover:text-slate-200 transition"
      >
        <span className="text-slate-600">{open ? "▾" : "▸"}</span> 行动参考
      </button>
      {open && (
        <div className="mt-1 space-y-1 text-[12px] leading-relaxed text-slate-300">
          {places.length > 0 && <div><span className="text-slate-500">能去的地方：</span>{places.join("、")}</div>}
          {absent.length > 0 && <div><span className="text-slate-500">能传信的人：</span>{absent.join("、")}</div>}
        </div>
      )}
    </div>
  );
}
