/**
 * The actor's conversation with the world, read off the actor's thread (`ActorEntry`): one line
 * per thing that passed between them, in the order the backend recorded it.
 *
 * Derived here, never stored: whether a reading was confirmed or declined, how far an act has
 * come, whether it can still be broken off, which perceptions belong to the question open now,
 * and where the world's time is shown: once per step, above the first line that step adds. The
 * backend's entries are the only facts.
 */

import type {
  ActorAnswer, ActorBriefing, ActorDecision, ActorEntry, ActorTurn, BriefMessage, DialogueTurn,
} from "../../types";

// A reading waiting for the actor while its turn is open; confirmed when the turn was answered
// with it; declined otherwise (a later reading, another answer, or the turn closing unanswered).
export type ReadingState = "pending" | "confirmed" | "declined";

export type ThreadItem =
  // The world's time for the lines after it, up to the next one.
  | { kind: "time"; key: string; label: string }
  // What reached the character. `asking`: it is what the open INTERRUPT question is about.
  | { kind: "perceived"; key: string; messages: BriefMessage[]; broadcasts: string[]; acted_upon: string[]; asking: boolean }
  // A question the world asked; `open` while it still waits for the actor.
  | { kind: "asked"; key: string; turn: ActorTurn; open: boolean }
  // The actor's words, and the world's reading of them or why there is none.
  | { kind: "said"; key: string; words: string; reading: ActorDecision | null; reason: string | null; briefing: ActorBriefing | null; state: ReadingState }
  // An answer that is a choice of its own, not the confirming of a reading.
  | { kind: "chose"; key: string; choice: "idle" | "carry_on" | "break_off" | "accept" | "refuse" | "act"; description: string }
  // An act of the character's, from its start: `elapsed` of `total` steps, `ended` once it is over.
  | { kind: "act"; key: string; description: string; elapsed: number; total: number; ended: boolean; breakingOff: boolean }
  | { kind: "ended"; key: string; fact: string; succeeded: boolean; happened: boolean }
  // A live conversation, its dialogue turns in order; `over` once it has ended.
  | { kind: "conversation"; key: string; dialogue: DialogueTurn[]; over: boolean };

/** The thread as lines, oldest first. `turn` is the turn open now (null when none). */
export function buildThread(entries: ActorEntry[], turn: ActorTurn | null): ThreadItem[] {
  const answered = new Map<string, ActorAnswer>();
  const lastReading = new Map<string, string>();
  const briefings = new Map<string, ActorBriefing | null>();
  for (const e of entries) {
    if (e.kind === "answered") answered.set(e.turn_id, e.answer);
    if (e.kind === "said" && e.reading) lastReading.set(e.turn_id, e.entry_id);
    if (e.kind === "asked") briefings.set(e.turn.turn_id, e.turn.briefing);
  }
  const lastAnswer = entries.map((e) => e.kind).lastIndexOf("answered");
  const items: ThreadItem[] = [];
  let act: Extract<ThreadItem, { kind: "act" }> | null = null;
  let timedStep: number | null = null;
  // A line of `e`'s, under its step's time. A line merged into one already shown, or an entry
  // that only updates one, adds no time: what it adds reads under the time already shown.
  const push = (e: ActorEntry, item: ThreadItem) => {
    if (e.step !== timedStep && e.time_label) {
      items.push({ kind: "time", key: `time-${e.entry_id}`, label: e.time_label });
      timedStep = e.step;
    }
    items.push(item);
  };

  entries.forEach((e, i) => {
    // Whatever follows a conversation in the thread comes after it ended.
    const trailing = items[items.length - 1];
    if (e.kind !== "dialogue_turn" && trailing?.kind === "conversation") trailing.over = true;
    switch (e.kind) {
      case "perceived": {
        const asking = turn?.kind === "interrupt" && i > lastAnswer;
        const prev = items[items.length - 1];
        // What reached the character with nothing else between is one card, however many steps
        // it came over.
        if (prev?.kind === "perceived") {
          items[items.length - 1] = {
            ...prev,
            messages: [...prev.messages, ...e.messages],
            broadcasts: [...prev.broadcasts, ...e.broadcasts],
            acted_upon: [...prev.acted_upon, ...e.acted_upon],
            asking: prev.asking || asking,
          };
        } else {
          push(e, {
            kind: "perceived", key: e.entry_id, messages: e.messages, broadcasts: e.broadcasts,
            acted_upon: e.acted_upon, asking,
          });
        }
        break;
      }
      case "asked":
        push(e, { kind: "asked", key: e.entry_id, turn: e.turn, open: turn?.turn_id === e.turn.turn_id });
        break;
      case "said": {
        const answer = answered.get(e.turn_id);
        const latest = lastReading.get(e.turn_id) === e.entry_id;
        const state: ReadingState =
          latest && answer && "action" in answer && answer.action ? "confirmed"
          : latest && !answer && turn?.turn_id === e.turn_id ? "pending"
          : "declined";
        push(e, {
          kind: "said", key: e.entry_id, words: e.words, reading: e.reading, reason: e.reason,
          briefing: briefings.get(e.turn_id) ?? null, state,
        });
        break;
      }
      case "answered": {
        const a = e.answer;
        if ("interrupt" in a) {
          push(e, { kind: "chose", key: e.entry_id, choice: a.interrupt ? "break_off" : "carry_on", description: "" });
        } else if ("accept" in a) {
          push(e, { kind: "chose", key: e.entry_id, choice: a.accept ? "accept" : "refuse", description: "" });
        } else if ("end" in a) {
          // A dialogue turn is recorded as a `dialogue_turn` entry, never as an answer.
        } else if (a.idle) {
          push(e, { kind: "chose", key: e.entry_id, choice: "idle", description: "" });
        } else if (a.action && !lastReading.has(e.turn_id)) {
          // Answered with an action no reading carried: say what it was.
          push(e, { kind: "chose", key: e.entry_id, choice: "act", description: a.action.action_description });
        }
        break;
      }
      case "break_off":
        push(e, { kind: "chose", key: e.entry_id, choice: "break_off", description: "" });
        if (act && !act.ended) act.breakingOff = true;
        break;
      case "began":
        // Its start is its first step.
        act = {
          kind: "act", key: e.entry_id, description: e.description, total: e.total_steps,
          elapsed: 1, ended: false, breakingOff: false,
        };
        push(e, act);
        break;
      case "progressed":
        if (act && !act.ended) {
          act.elapsed = e.elapsed_steps;
          act.total = e.total_steps;
        }
        break;
      case "dialogue_turn": {
        const prev = items[items.length - 1];
        const conversation: Extract<ThreadItem, { kind: "conversation" }> =
          prev?.kind === "conversation" && !prev.over
            ? prev
            : { kind: "conversation", key: e.entry_id, dialogue: [], over: false };
        if (conversation !== prev) push(e, conversation);
        if (e.dialogue_turn.line) conversation.dialogue.push(e.dialogue_turn);
        if (e.end) conversation.over = true;
        break;
      }
      case "ended":
        if (act && !act.ended) {
          act.ended = true;
          act.elapsed = act.total;
        }
        push(e, { kind: "ended", key: e.entry_id, fact: e.fact, succeeded: e.succeeded, happened: e.happened });
        break;
    }
  });
  return items;
}

/** The act under way that can still be broken off: one spanning steps, not yet ended. */
export function breakableAct(items: ThreadItem[]): Extract<ThreadItem, { kind: "act" }> | null {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item.kind === "act") return !item.ended && item.total > 1 ? item : null;
  }
  return null;
}

/**
 * The live conversation going on now: the thread's last item when it is a conversation not yet
 * over, or none yet when the character's first dialogue turn is being asked for.
 */
export function liveConversation(items: ThreadItem[]): Extract<ThreadItem, { kind: "conversation" }> | null {
  const last = items[items.length - 1];
  return last?.kind === "conversation" && !last.over ? last : null;
}

/** The thread's live conversations, oldest first. */
export function conversationsOf(items: ThreadItem[]): Extract<ThreadItem, { kind: "conversation" }>[] {
  return items.flatMap((i) => (i.kind === "conversation" ? [i] : []));
}

/** `entries` with `incoming` added, each once: a live entry may also come with a fetched history. */
export function mergeEntries(entries: ActorEntry[], incoming: ActorEntry[]): ActorEntry[] {
  const seen = new Set(entries.map((e) => e.entry_id));
  const fresh = incoming.filter((e) => !seen.has(e.entry_id));
  return fresh.length ? [...entries, ...fresh] : entries;
}
