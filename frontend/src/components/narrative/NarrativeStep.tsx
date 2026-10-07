// One step of the narrative feed: the step's own header, and its four channels folded into
// a single seq-ordered stream of rows.
//
// This file is the ASSEMBLY: which rows exist, in what order, and which a focus collapses
// (never drops; see `involves` in feed.ts). A row's look belongs to its own module; what the
// wire means belongs to feed.ts.
import { useState } from "react";

import { Phase } from "../../lib/contract";
import type { NpcStateSummary, StepEvent } from "../../types";
import { npcStandingLine } from "../../lib/npc";
import ActionRow from "./ActionRow";
import BroadcastRow from "./BroadcastRow";
import CollapsedRow from "./CollapsedRow";
import EventRow from "./EventRow";
import MessageRow from "./MessageRow";
import { stepCast } from "./cast";
import { bornInStep, buildFeed, involves, itemKey } from "./feed";
import { collapsedOf } from "./meta";

export default function NarrativeStep({
  step,
  focusAgents = [],
  isolated = false,
  onToggleIsolate,
}: {
  step: StepEvent;
  focusAgents?: string[];
  // Whether the feed is showing this step ALONE; owned by the feed, which sees the other steps.
  isolated?: boolean;
  onToggleIsolate?: (step: number) => void;
}) {
  // Beats the reader opened despite the focus. Owned by the card, so it survives a focus
  // change: an expanded beat is one you asked to see.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());
  // What this step brought into the world (see feed.bornInStep).
  const born = bornInStep(step);
  const focusSet = new Set(focusAgents);
  const focusActive = focusSet.size > 0;
  const cast = stepCast(step);
  const feed = buildFeed(step);
  // One transcript rides on every participant's record; the first card in feed order that
  // prints it claims it. Decided here because it's a question about the whole list.
  const spoken = new Set<string>();

  return (
    <div className="relative pl-5">
      <span className="absolute left-0 top-[6px] w-2 h-2 rounded-full bg-indigo-500/80 ring-4 ring-slate-900" />
      <span className="absolute left-[3.5px] top-4 -bottom-3 w-px bg-slate-800/80" />

      {/* The world's hour leads; the engine's counter trails it, dimmer. The label is shown
          whole, never split: its form belongs to the theme's calendar.

          STICKY within this step's card, so "which step am I reading" is always answered,
          and it is the ISOLATE control, because that question is when you want to isolate.
          Opaque with a z-index: a pinned header sits over its own beats.

          A compact one-line form, not the WorldClock plate. */}
      <div className="sticky top-0 z-10 -mx-1 -mt-1 px-1 pt-1 pb-1 bg-slate-900 border-b border-slate-800/60">
        <button
          type="button"
          onClick={() => onToggleIsolate?.(step.step)}
          disabled={!onToggleIsolate}
          title={isolated ? "退出，看全部步" : "只看这一步"}
          className="group w-full flex items-baseline gap-1.5 text-left text-[11px] font-medium tracking-wide rounded transition enabled:hover:bg-slate-800/40 disabled:cursor-default focus:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500/50"
        >
          <span className="text-amber-200/70">{step.world_time.label}</span>
          <span className="text-slate-600">· 第 {step.step} 步</span>
          {onToggleIsolate && (
            <span
              className={`ml-auto shrink-0 text-[10px] ${
                isolated ? "text-indigo-300" : "text-slate-700 group-hover:text-slate-400"
              }`}
            >
              {isolated ? "只看此步 ✕" : "只看此步"}
            </span>
          )}
        </button>
      </div>

      {/* A step where nothing happened SAYS SO (e.g. every decision failed and was skipped).
          Blank would look like a broken feed; rows would dress the absence up as deeds. */}
      {feed.length === 0 && (
        <div className="mt-2 text-[13px] text-slate-600 italic">本步无人行动。</div>
      )}

      {/* World events, actions, messages and broadcasts in seq (emission) order. */}
      <div className="mt-2 space-y-2.5">
        {feed.map((item, feedIdx) => {
          // Same key in both branches, so expanding a row swaps it in place without a remount.
          const key = itemKey(item, feedIdx);
          if (focusActive && !involves(item, focusSet) && !expanded.has(key)) {
            const { icon, text } = collapsedOf(item);
            return (
              <CollapsedRow
                key={key}
                icon={icon}
                text={text}
                onExpand={() => setExpanded((prev) => new Set(prev).add(key))}
              />
            );
          }

          if (item.kind === "event") return <EventRow key={key} ev={item.data} />;
          if (item.kind === "message") return <MessageRow key={key} m={item.data} cast={cast} />;
          if (item.kind === "broadcast") return <BroadcastRow key={key} b={item.data} />;

          // The dialogue rides on whichever participant's record happens to carry it, and a
          // tick never replays it.
          const turns = (item.parts.find((p) => (p.dialogue ?? []).length)?.dialogue ?? []).filter((t) => t.line);
          const sig = turns.map((t) => `${t.speaker}:${t.line}`).join("|");
          const showConversation =
            turns.length > 0 && !spoken.has(sig) && item.data.phase !== Phase.ongoing_tick;
          if (showConversation) spoken.add(sig);
          return (
            <ActionRow
              key={key}
              act={item.data}
              parts={item.parts}
              led={item.led}
              cast={cast}
              turns={turns}
              showConversation={showConversation}
              born={born}
              entities={step.entities}
            />
          );
        })}
      </div>

      <NpcStandingRow npcs={step.npcs ?? []} roll={step.step === 0} />
    </div>
  );
}

/**
 * What the bodies-without-minds are up to this step, as one block after the stream: it is the
 * step's closing state and has no `seq`, so it has no place in the event order.
 *
 * One body per line, bare (no avatar, chip or heading) so it doesn't out-shout the cast. Don't
 * join them with interpuncts: the wrap would break names in the middle.
 *
 * Idle ones are not listed, and with nobody busy the row disappears, except at step 0, where
 * the world and its cast are introduced.
 *
 * The present-tense phrasing comes from lib/npc, shared with the map's hover; see there on why
 * it must not read as an event.
 */
function NpcStandingRow({ npcs, roll }: { npcs: NpcStateSummary[]; roll: boolean }): JSX.Element | null {
  const rows = npcs
    .map((npc) => [npc.npc_id, npcStandingLine(npc, roll)] as const)
    .filter(([, line]) => line.length > 0);
  if (!rows.length) return null;
  return (
    <div className="mt-2.5 flex items-baseline gap-2 text-[12px] text-slate-500">
      <span className="shrink-0 text-slate-600">NPC</span>
      <div className="min-w-0 space-y-0.5">
        {rows.map(([id, line]) => (
          <div key={id}>{line}</div>
        ))}
      </div>
    </div>
  );
}
