// One step of the narrative feed: the step's own header, and what happened in it as a single
// seq-ordered stream of rows.
//
// This file is the ASSEMBLY: which rows exist, in what order, and which a focus collapses
// (never drops; see `involves` in feed.ts). A row's look belongs to its own module; what the
// wire means belongs to feed.ts.
import { useState } from "react";

import { identityOf } from "../../lib/worldTable";
import type { Npcs } from "../../lib/npcs";
import type { Cast } from "../../lib/cast";
import type { Places } from "../../lib/places";
import type { StepCast } from "./cast";
import { Phase } from "../../lib/contract";
import type { AgentView, NpcView, StepView } from "../../types";
import { type Placed, indexStep } from "../../lib/step";
import { npcStandingLine } from "../../lib/npc";
import type { Viewer } from "../../lib/viewer";
import ActionRow from "./ActionRow";
import BroadcastRow from "./BroadcastRow";
import CollapsedRow from "./CollapsedRow";
import InjectionRow from "./InjectionRow";
import MessageRow from "./MessageRow";
import ObservationRow from "./ObservationRow";
import { stepCast } from "./cast";
import { bornInStep, buildFeed, involves, itemKey } from "./feed";
import { collapsedOf } from "./meta";

export default function NarrativeStep({
  step,
  cast: publicCast,
  places,
  npcs,
  focusAgents = [],
  isolated = false,
  onToggleIsolate,
  viewer = null,
}: {
  step: StepView;
  cast: Cast;
  places: Places;
  npcs: Npcs;
  focusAgents?: string[];
  // Whether the feed is showing this step ALONE; owned by the feed, which sees the other steps.
  isolated?: boolean;
  onToggleIsolate?: (step: number) => void;
  // Whose view the step is filtered to; null for the director.
  viewer?: Viewer;
}) {
  // Beats the reader opened despite the focus. Owned by the card, so it survives a focus
  // change: an expanded beat is one you asked to see.
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set());
  // What this step brought into the world (see feed.bornInStep).
  const born = bornInStep(step);
  const focusSet = new Set(focusAgents);
  const focusActive = focusSet.size > 0;
  const cast = stepCast(step, publicCast, places, npcs);
  const feed = buildFeed(step, viewer);
  const index = indexStep(step);
  // A transcript prints once even if two beats carry it; the first card in feed order claims
  // it. Decided here because it's a question about the whole list.
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

      {/* Step 0 is the world as built: nothing has happened yet, so it introduces who stands
          where. Any later step where nothing happened SAYS SO (e.g. every decision failed and was
          skipped): blank would look like a broken feed, rows would dress the absence up as deeds. */}
      {step.step === 0 ? (
        <AgentRollRow agents={Object.values(index.agents)} cast={cast} />
      ) : feed.length === 0 && (
        <div className="mt-2 text-[13px] text-slate-600 italic">本步无人行动。</div>
      )}

      {/* Injections, actions, messages and broadcasts in seq (emission) order. */}
      <div className="mt-2 space-y-2.5">
        {feed.map((item) => {
          // Same key in both branches, so expanding a row swaps it in place without a remount.
          const key = itemKey(item);
          if (focusActive && !involves(item, focusSet) && !expanded.has(key)) {
            const { icon, text } = collapsedOf(item, cast);
            return (
              <CollapsedRow
                key={key}
                icon={icon}
                text={text}
                onExpand={() => setExpanded((prev) => new Set(prev).add(key))}
              />
            );
          }

          if (item.kind === "injection") return <InjectionRow key={key} injection={item.data} />;
          if (item.kind === "message") return <MessageRow key={key} m={item.data} cast={cast} />;
          if (item.kind === "observation") return <ObservationRow key={key} o={item.data} cast={cast} />;
          if (item.kind === "broadcast") {
            return <BroadcastRow key={key} b={item.data} place={cast.placeOf(item.data.location_id)} />;
          }

          // A tick never replays the dialogue.
          const turns = (item.shared?.dialogue ?? []).filter((t) => t.line);
          const sig = turns.map((t) => `${t.speaker}:${t.line}`).join("|");
          const showConversation =
            turns.length > 0 && !spoken.has(sig) && item.data.phase !== Phase.ongoing_tick;
          if (showConversation) spoken.add(sig);
          return (
            <ActionRow
              key={key}
              act={item.data}
              privateParts={item.private_parts}
              cast={cast}
              turns={turns}
              showConversation={showConversation}
              born={born}
              entities={index.entities}
              shared={item.shared}
              actionDescription={item.action_description}
              mine={viewer !== null && item.data.participant_ids.includes(viewer)}
            />
          );
        })}
      </div>

      <NpcStandingRow npcs={index.npcs} roll={step.step === 0} cast={cast} identities={npcs} />
    </div>
  );
}

/** The opening roll call: every character and where they stand, main characters first. */
function AgentRollRow({ agents, cast }: { agents: Placed<AgentView>[]; cast: StepCast }): JSX.Element | null {
  if (!agents.length) return null;
  const ordered = [...agents].sort((a, b) => Number(b.is_main_character) - Number(a.is_main_character));
  return (
    <div className="mt-2 space-y-0.5 text-[13px] text-slate-400">
      {ordered.map((a) => (
        <div key={a.agent_id}>
          <span style={{ color: cast.inkOf(a.agent_id) }} className="font-medium">{cast.nameOf(a.agent_id)}</span>
          {a.is_main_character && <span className="text-[10px] text-indigo-400"> ★</span>}
          <span className="text-slate-600">（{cast.placeOf(a.location_id)}）</span>
        </div>
      ))}
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
function NpcStandingRow({ npcs, roll, cast, identities }: { npcs: Placed<NpcView>[]; roll: boolean; cast: StepCast; identities: Npcs }): JSX.Element | null {
  const rows = npcs
    .map((npc) => [npc.npc_id, npcStandingLine(npc, identityOf(npc.npc_id, identities), cast.placeOf(npc.location_id), roll)] as const)
    .filter(([, line]) => line.length > 0);
  if (!rows.length) return null;
  return (
    <div className="mt-2.5 flex items-baseline gap-2 text-[12px] text-slate-500">
      <span className="shrink-0 text-slate-600">NPC</span>
      <div className="min-w-0 space-y-0.5">
        {rows.map(([id, line]) => (
          <div key={id} style={{ color: cast.inkOf(id) }}>{line}</div>
        ))}
      </div>
    </div>
  );
}
