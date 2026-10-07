import { ActionType, Phase, actedOnAgents, claimedAgents } from "../../lib/contract";
import { avatarStyle } from "../../lib/avatar";
import type { ActionSummary, DialogueTurn, EntityView } from "../../types";
import type { StepCast } from "./cast";
import { metaOf } from "./meta";
import Who from "./Who";

// Fallback palette for a speaker the world carries no identity colour for — per
// CONVERSATION, so the two sides of an exchange at least stay apart from each other. Never
// reached when the speaker is a known character: they wear their own colour.
const SPEAKER_COLORS = [
  "text-indigo-300",
  "text-teal-300",
  "text-rose-300",
  "text-amber-300",
  "text-sky-300",
  "text-fuchsia-300",
];

function Conversation({ turns, cast }: { turns: DialogueTurn[]; cast: StepCast }) {
  const speakers = [...new Set(turns.map((t) => t.speaker).filter(Boolean))];
  const colorOf = (s: string) => SPEAKER_COLORS[Math.max(0, speakers.indexOf(s)) % SPEAKER_COLORS.length];
  return (
    <div className="mt-2 space-y-1.5 rounded-lg bg-slate-950/50 border border-slate-800/60 px-3 py-2.5">
      {turns.map((t, i) => {
        const idColor = cast.inkOf(t.speaker_id);
        return (
          <div key={i} className="text-sm leading-relaxed">
            {t.speaker && (
              <span
                className={`font-semibold ${idColor ? "" : colorOf(t.speaker)}`}
                style={idColor ? { color: idColor } : undefined}
              >
                {t.speaker}
              </span>
            )}
            {t.speaker && <span className="text-slate-600">：</span>}
            <span className="text-slate-200">{t.line}</span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * ONE execution, however many people were in it.
 *
 * `act` is the initiator's record; `parts` is every participant's record, initiator first when
 * `led`. `showConversation` is decided by the feed: one transcript rides on every participant's
 * record, so whether it was already shown is a question about the whole list.
 */
export default function ActionRow({
  act,
  parts,
  led,
  cast,
  turns,
  showConversation,
  born,
  entities,
}: {
  act: ActionSummary;
  parts: ActionSummary[];
  led: boolean;
  cast: StepCast;
  turns: DialogueTurn[];
  showConversation: boolean;
  // What this STEP brought into the world, and the entity table to name it by. Only together
  // with `affected_entity_ids` does it say what this deed made.
  born?: ReadonlySet<string>;
  entities?: Record<string, EntityView>;
}) {
  // What THIS deed made: touched and born. A thing it merely changed is touched but not born.
  const made = (act.affected_entity_ids ?? [])
    .filter((eid) => born?.has(eid) && entities?.[eid])
    .map((eid) => entities![eid]);
  // Whoever the act was DONE TO is in `target.acts_on` and earns the arrow; whoever merely had
  // their turn spent on it is in `target.claims`. A move drags people along with no arrow:
  // 「A → B」 would report a journey as an act upon a man.
  const aimedAt = new Set(actedOnAgents(act.target));
  const others = led ? parts.slice(1) : [];
  // Worded like movement._company_clause: neutral 「带着」, since the same mechanism serves both
  // dragging a man off and carrying the wounded clear.
  const claimed = new Set(claimedAgents(act.target));
  const carried = others.filter((p) => claimed.has(p.agent_id) && !aimedAt.has(p.agent_id));
  // Show direction only when the records attest it; otherwise cast order is just arrival order
  // and an arrow would invent who approached whom.
  const directed = others.some((p) => aimedAt.has(p.agent_id));
  const isInterrupt = act.phase === Phase.interrupt;
  // The MIDDLE beat of a multi-step act. Don't give it a bespoke light row: it would read as a
  // stray tail of the card above and drift from the shared markup. It shows only a 进行中 badge
  // and a progress bar, never dialogue, 心声 or outcome (those play at the opening and close).
  const tick = act.phase === Phase.ongoing_tick;
  const tickPct = tick && act.total_steps > 0
    ? Math.min(100, Math.round((act.elapsed_steps / act.total_steps) * 100))
    : 0;
  const meta = metaOf(act);
  // Whoever broke it off — it rides on every participant's record of the execution.
  const breaker = parts.find((p) => p.interrupted_by)?.interrupted_by ?? "";
  // Whoever heard it without being in it; not in the avatar stack or name line, which are for
  // participants. Nameless ones are dropped: never print an id.
  const overheard = (act.overheard_by ?? [])
    .map((id) => ({ id, name: cast.nameOf(id) }))
    .filter((o) => o.name);

  // A multi-step act reaches us as three beats, and each says only what is NEW.
  //   opening (begin) — the intent. Its `outcome` ("X着手做…") only restates it, so it is
  //     NOT printed as a result.
  //   done (ongoing_complete) — the RESULT, and the result leads: the intent was already said
  //     steps ago.
  // A `settled` beat opened and closed here: it prints its result and wears no 开始/结束 badge.
  const opening = act.phase === Phase.begin;
  const done = act.phase === Phase.ongoing_complete;
  // not_executed: the act never engaged the world (target absent/busy, no path). A non-event,
  // so the whole card recedes rather than looking like a deed with a grey badge.
  const nonEvent = act.not_executed && !isInterrupt;
  // A talk's result is the conversation itself, rendered structurally below — so its
  // outcome (gist + verbatim transcript) must never become prose.
  const spokenFor = act.action_type === ActionType.talk && turns.length > 0;
  const showOutcome =
    !!act.outcome &&
    act.phase !== Phase.initialization && // step-0 "初始化完成" is placement noise, not a result
    // An opening's or a tick's outcome only restates the body; the result comes on the close.
    !opening &&
    !tick &&
    !spokenFor;

  return (
    <div
      className={`rounded-xl border p-3 ${
        nonEvent
          ? "bg-slate-900/20 border-slate-800/40 opacity-60"
          : "bg-slate-900/40 border-slate-800/60"
      }`}
    >
      <div className="flex items-center gap-2 flex-wrap">
        {/* Everyone who was in it, each in their own identity colour. The fallback gradient
            is keyed on the person's id, not the beat's position, or the same person would
            change face between rows of one step. */}
        <span className="flex -space-x-1.5 shrink-0">
          {parts.map((p) => (
            <span
              key={p.agent_id}
              style={avatarStyle(cast.inkOf(p.agent_id), p.agent_id)}
              className={`w-6 h-6 rounded-lg grid place-items-center text-white text-[11px] font-black ring-2 ring-slate-900`}
            >
              {p.agent_name[0]}
            </span>
          ))}
        </span>
        {/* The initiator leads: A → B is A seeking B out, a different event from the
            reverse. The others read dimmer; the sentence belongs to the one who started it. */}
        <strong className="text-sm text-slate-100">
          {directed ? (
            <>
              {parts[0].agent_name}
              <span className="mx-1 font-normal text-slate-600">→</span>
              <span className="font-semibold text-slate-300">
                {others.filter((p) => aimedAt.has(p.agent_id)).map((p) => p.agent_name).join(" · ")}
              </span>
            </>
          ) : carried.length > 0 ? (
            parts[0].agent_name
          ) : (
            parts.map((p) => p.agent_name).join(" · ")
          )}
        </strong>
        {parts.some((p) => p.is_main_character) && <span className="text-[10px] text-indigo-400">★</span>}
        <span className="text-xs text-slate-500">📍 {cast.placeOf(act)}</span>
        <span className={`ml-auto text-[10px] px-1.5 py-0.5 rounded-md border ${meta.chip}`}>
          {meta.icon} {meta.label}
        </span>
        {/* Mark the running act and its conclusion, so the result visibly belongs to
            something begun steps ago. */}
        {opening && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-sky-300/80 bg-sky-950/40 border border-sky-900/50">
            ⏳ 开始{act.duration_label ? ` · 预计${act.duration_label}` : ""}
          </span>
        )}
        {done && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-slate-400 bg-slate-800/40 border border-slate-700/50">
            ⏳ 结束
          </span>
        )}
        {/* The middle beat's marker, in the same ⏳ language as 开始/结束 — one act's three
            beats read as a sequence, not three unrelated states. */}
        {tick && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-sky-300/80 bg-sky-950/40 border border-sky-900/50">
            ⏳ 进行中
          </span>
        )}
        {/* The condition he acts under. Not a card of its own: the deed that imposed it
            already reported it. Amber, matching the character card. */}
        {cast.conditionOf(act.agent_id) && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-amber-500/80 bg-amber-950/30 border border-amber-900/40">
            {cast.conditionOf(act.agent_id)}
          </span>
        )}
        {/* An interrupt adds WHO CUT IT, which matters for a joint action (the initiator
            walking out differs from the other man walking out). A solo action omits it: the
            interrupter is the actor himself. The why is his 心声 below. */}
        {isInterrupt && breaker && parts.length > 1 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-slate-300 bg-slate-800/50 border border-slate-600/60">
            被 <Who cast={cast} id={breaker} name={cast.nameOf(breaker) || "某人"} /> 打断
          </span>
        )}
        {/* Taken along. Not on the name line (the deed isn't theirs), but their face stays in
            the stack: unlike an overhearer, they are in this execution. */}
        {carried.length > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-slate-400 bg-slate-800/40 border border-slate-700/50">
            带着 {carried.map((p, i) => (
              <span key={p.agent_id}>
                {i > 0 && " · "}
                <Who cast={cast} id={p.agent_id} name={p.agent_name} />
              </span>
            ))}
          </span>
        )}
        {/* Overhearers: they keep a memory of this, so it must answer to an event on screen.
            Named, not given a face: a face in the stack would read as taking part. */}
        {overheard.length > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-md text-slate-400 bg-slate-800/40 border border-slate-700/50">
            👂 {overheard.map((o, i) => (
              <span key={o.id}>
                {i > 0 && " · "}
                <Who cast={cast} id={o.id} name={o.name} />
              </span>
            ))} 在旁听着
          </span>
        )}
        {/* not_executed = the intent never engaged the world (target absent/busy,
            no path) → a muted grey "未成事", NOT the rose "未果" of a real defeat. */}
        {act.not_executed && !isInterrupt ? (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md text-slate-400 bg-slate-800/50 border border-slate-700/50">
            未成事
          </span>
        ) : (
          act.succeeded === false && !isInterrupt && (
            <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md text-rose-300 bg-rose-950/50 border border-rose-900/60">
              未果
            </span>
          )
        )}
        {/* Was he seen: a covert act's second question, independent of 未果, so amber
            rather than rose. Only COVERT sets it, and only being noticed gets a mark. */}
        {act.detected && (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-md text-amber-300 bg-amber-950/50 border border-amber-900/60">
            被察觉
          </span>
        )}
      </div>

      {/* Every beat says WHAT he is doing; the opening adds how long, the close how it went. */}
      {act.action_description && (
        <p className={`mt-2 text-sm leading-relaxed ${nonEvent ? "text-slate-400" : "text-slate-200"}`}>
          {act.action_description}
        </p>
      )}

      {/* How far along: the middle beat's only payload beyond the description. */}
      {tick && (
        <div className="mt-2 h-1 rounded-full bg-slate-800/80 overflow-hidden">
          <div className="h-full rounded-full bg-sky-500/60" style={{ width: `${tickPct}%` }} />
        </div>
      )}

      {/* WHY he did it: the deliberation behind the deed. Feed-only; the map has no minds.
          Not filtered by is_main_character: every agent decides through the same LLM
          cognition (CLAUDE.md §5).

          Unlike the outcome, 心声 is per person (two people in one conversation think
          different things), named when there is more than one. Empty on every beat that
          did not decide: a tick, a later completion, a conscripted participant, an
          interrupt (its reason lives in its outcome). */}
      {parts
        .filter((p) => !tick && p.inner_monologue)
        .map((p) => (
          <p key={p.agent_id} className="mt-1.5 text-[13px] italic leading-relaxed text-slate-400">
            <span className="not-italic text-slate-600">
              心声{parts.length > 1 ? ` · ${p.agent_name}` : ""} ·{" "}
            </span>
            {p.inner_monologue}
          </p>
        ))}

      {showConversation && <Conversation turns={turns} cast={cast} />}

      {showOutcome && (
        <p className="mt-2 pt-2 border-t border-slate-800/50 text-[13px] leading-relaxed text-slate-400">
          <span className="text-emerald-500/70">↳ </span>
          {/* The whole record, including privileged detail (an interrupt's quoted reason).
              A talk's transcript never lands here: showOutcome drops the outcome when the
              structured `dialogue` renders it. */}
          {act.outcome}
        </p>
      )}

      {/* WHY it failed, verbatim from its own field. Empty on success, on an interrupt
          (the ✂ badge says it), and when adjudication never happened. */}
      {/* What the deed LEFT BEHIND, on its own line so it can be found without reading the
          prose. Where it went is half the fact: pocketed and set down are different futures. */}
      {made.length > 0 && (
        <p className="mt-1.5 text-[13px] leading-relaxed text-amber-300/80">
          <span className="text-amber-400/70">✦ </span>
          留下：
          {made.map((e, i) => (
            <span key={e.name + i}>
              {i > 0 && "、"}
              {e.name}
              <span className="text-amber-300/50">
                （{e.presence === "held" ? "随身" : "放在此地"}
                {e.is_public === false && "，不公开"}）
              </span>
            </span>
          ))}
        </p>
      )}
      {act.failure_reason && (
        <p className="mt-1.5 text-[13px] leading-relaxed text-rose-400/80">
          <span className="text-rose-500/70">✗ </span>
          {act.failure_reason}
        </p>
      )}
    </div>
  );
}
