"""Goals: the goal record (short- and long-term), the short-term queue's membership and eviction,
due dates, and parsing the judge's verdicts on them.

``NeedEngine`` decides when goals are generated and asks the judge how they went; this module owns
what a goal is and how the queue holding them behaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

from core.duration import describe_duration
from core.interfaces.llm import extract_json
from core.logging import get_logger

if TYPE_CHECKING:
    from agent.need import NeedType

_logger = get_logger(__name__)


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class GoalOrigin(str, Enum):
    """Where a goal came from, which decides what keeps it in the queue (see _evict_over_cap).

    - COGNITIVE: a plan derived from "my needs + situation"; being pushed out by a newer plan is fine.
    - RESIDUE: unfinished business left when an action ends (a request I accepted, the errand I
      came for). It already happened, so it leaves only once resolved, not because I have new plans.

    RESIDUE is what carries unfinished business into the next step's motivation instead of
    leaving it to a memory that may or may not be recalled.
    """

    COGNITIVE = "cognitive"
    RESIDUE = "residue"


@dataclass
class GoalEntity:

    id: str
    text: str
    goal_type: str  # "short_term" | "long_term"
    status: GoalStatus = GoalStatus.ACTIVE
    related_need: NeedType | None = None  # which need this goal serves
    created_step: int = 0
    # The agreed time (absolute step), set only when the goal text fixes a time. It lets rendering
    # state "is it time yet" instead of leaving the agent to work it out from prose, and it is where
    # the time allowance starts (is_past_allowance). Whether a due goal is still worth doing is the
    # judge's call; code steps in only once the make-up allowance is spent.
    #
    # Written once at creation, never updated: the model re-estimates "how long left" each step,
    # and "3 more steps" every time would slide due forward forever. An in-world reschedule
    # expires the old goal at its original time and enters a new one.
    due_step: int | None = None
    last_evaluated_step: int | None = None
    progress_summary: str = ""
    origin: GoalOrigin = GoalOrigin.COGNITIVE


# Short-term goals are a capped FIFO queue: membership is rule-maintained, pursuit is the decision
# stage's call. The one short_term_goal_entities list holds both the live queue (ACTIVE /
# INTERRUPTED, front=oldest) and a capped recent history (COMPLETED / FAILED) shown to the
# generation prompt. Regeneration appends, never overwrites; over capacity the oldest is marked
# FAILED rather than silently dropped, so it still shows up in the history.
_SHORT_TERM_GOAL_CAP: int = 6
_SHORT_TERM_GOAL_HISTORY_CAP: int = 6

# Literal-similarity (SequenceMatcher.ratio) dedup threshold on enqueue. High on purpose: it only
# catches near-verbatim repeats; semantic duplicates are the generation prompt's job. The ratio is
# literal, so a low threshold would merge distinct intentions that look alike, and a false merge
# loses an intention.
_SHORT_TERM_GOAL_DEDUP_RATIO: float = 0.60
# Admitted goals above this similarity are logged for threshold tuning; no part in the decision.
_SHORT_TERM_GOAL_DEDUP_WATCH_FLOOR: float = 0.5

_NON_TERMINAL_GOAL_STATUSES: tuple[GoalStatus, ...] = (GoalStatus.ACTIVE, GoalStatus.INTERRUPTED)

# A short-term goal's time allowance (see is_past_allowance).
#
# Without an agreed time it runs from creation: a hard floor that catches a judge that won't let
# go of a goal the world won't let finish ("soothe someone"), which would otherwise lock the agent
# into repeating itself. 6 ≥ the longest legitimate multi-step action, so slow goals aren't hit.
#
# With one, the same allowance starts at due:
# ① Don't reap before due: a waiting appointment isn't going in circles, and steps vs world hours
#    differ (at 1 hour per step an overnight appointment would be reaped before it came due).
# ② Past due is not void: a missed appointment deserves a full make-up allowance, not a step or
#    two, which would amount to voiding it on expiry.
_SHORT_TERM_GOAL_STALL_STEPS: int = 6

# Residue limits live only in the prompt; don't truncate residue text, a half-cut sentence would
# reach the goal queue and decision prompt. residue_reason goes only into the trace, so capping it
# is safe.
_RESIDUE_REASON_MAX_CHARS: int = 120


def is_live_goal(goal: GoalEntity) -> bool:
    return goal.status in _NON_TERMINAL_GOAL_STATUSES


def live_goals(entities: Sequence[GoalEntity]) -> List[GoalEntity]:
    """The live queue: non-terminal goals in enqueue order, front=oldest."""
    return [g for g in entities if is_live_goal(g)]


# "Advanceable" (ACTIVE only) is not is_live_goal's "queue member" (includes INTERRUPTED):
# membership drives display, dedup and eviction; advanceability drives progress re-evaluation,
# supply and need bias. Conflating them misjudges interrupted goals and miscounts supply.
def is_active_goal(goal: GoalEntity) -> bool:
    return goal.status == GoalStatus.ACTIVE


def active_goals(entities: Sequence[GoalEntity]) -> List[GoalEntity]:
    """Advanceable goals (ACTIVE only), for supply and progress."""
    return [g for g in entities if is_active_goal(g)]


def goal_age_hint(goal: GoalEntity, step: int, seconds_per_step: int) -> str:
    """Stall evidence for the judge ("going in circles → set aside"): the natural duration since
    the goal was set, never the step integer.

    Not for goals with a set time (goal_due_hint covers those): "已历…仍未了结" of an appointment
    not yet due would be false, and the judge's set-aside rule keys on that mark.
    """
    if goal.due_step is not None:
        return ""
    elapsed = max(0, step - goal.created_step)
    if elapsed <= 0:
        return ""
    # describe_duration already adds the "约" prefix; don't repeat it.
    return f"（已历{describe_duration(elapsed, seconds_per_step)}仍未了结）"


def parse_goal_item(item: object) -> tuple[str, int | None] | None:
    """The single parser for a "goal text + optional deadline" item (short-term goals and residue).

    Returns ``(text, hours until the agreed time)``, or None for empty text. Hours, not steps:
    step is a code-layer coordinate and code does the conversion. A missing or invalid deadline
    means None, the normal answer, never a reason to drop the goal.
    """
    if isinstance(item, str):
        return (text, None) if (text := item.strip()) else None
    if not isinstance(item, dict):
        return None
    if not (text := str(item.get("text", "") or "").strip()):
        return None
    try:
        hours = int(float(item["due_in_hours"]))
    except (KeyError, TypeError, ValueError):
        return (text, None)
    return (text, hours if hours > 0 else None)


def _due_step_from_hours(current_step: int, hours: int | None, seconds_per_step: int) -> int | None:
    """"Hours from now" → absolute step; the inverse of describe_duration.

    Round down: late suggests slack that isn't there (and round() sends 3.5 → 4). At least one
    step: half a step away is "due next step", not "due now".
    """
    if not hours or hours <= 0:
        return None
    return current_step + max(1, int(hours * 3600 / max(1, seconds_per_step)))


def is_due(goal: GoalEntity, step: int) -> bool:
    """Due goals are listed separately."""
    return goal.due_step is not None and goal.due_step <= step


def is_past_allowance(goal: GoalEntity, step: int) -> bool:
    """The reaper's only test: the two clocks described at _SHORT_TERM_GOAL_STALL_STEPS.
    Same clock as goal_age_hint / goal_due_hint, so the mark the judge sees decides the reaping;
    don't let them drift apart."""
    start = goal.due_step if goal.due_step is not None else goal.created_step
    return step - start >= _SHORT_TERM_GOAL_STALL_STEPS


def goal_due_hint(goal: GoalEntity, step: int, seconds_per_step: int) -> str:
    """State the agreed time relative to now as a fact, in natural durations. Mutually
    exclusive with goal_age_hint: goals with a set time use only this one."""
    due = goal.due_step
    if due is None:
        return ""
    if due > step:
        return f"（约定在{describe_duration(due - step, seconds_per_step)}之后）"
    if due == step:
        return "（约定就在此刻）"
    return f"（约定时刻已过{describe_duration(step - due, seconds_per_step)}）"


# Definition of the marks the two hints above produce; keep the wording in sync. The audit needs
# it, or its judge reads engine-stamped marks as narrative rather than facts to score on.
GOAL_TIME_MARK_DEFINITION = (
    "目标文本后的时间标记由引擎盖上、非角色所写："
    "「（约定在…之后）」= 这条目标约定的时刻还没到；「（约定就在此刻）」= 正好到点；"
    "「（约定时刻已过…）」= 时刻过了仍未了结；「（已历…仍未了结）」= 从立下到此刻已过多久"
)


def _goal_sort_key(goal: GoalEntity, order: int) -> tuple[int, int, int]:
    """Goals with a set time first, by due time; the rest keep queue order. In plain creation
    order a due goal would read as the oldest and get buried."""
    if goal.due_step is None:
        return (1, 0, order)
    return (0, goal.due_step, order)


def order_goals_for_prompt(goals: Sequence[GoalEntity]) -> List[GoalEntity]:
    """Order goals before rendering (single source, shared by decision / personality / judge)."""
    return [g for _, g in sorted(
        ((_goal_sort_key(g, i), g) for i, g in enumerate(goals)), key=lambda kv: kv[0],
    )]


def _next_short_term_seq(entities: Sequence[GoalEntity]) -> int:
    """Stateless unique-id suffix: ids need only be unique among current entities, so a suffix
    freed by trimming may be reused."""
    seq = 0
    for g in entities:
        if g.goal_type == "short_term":
            tail = g.id.rsplit("-", 1)[-1]
            if tail.isdigit():
                seq = max(seq, int(tail))
    return seq + 1


def _evict_over_cap(entities: List[GoalEntity], current_step: int) -> None:
    """Evict the oldest over-capacity goals in place, marked FAILED, not dropped.

    Oldest COGNITIVE first; RESIDUE only when the live queue is all RESIDUE, or "what I promised"
    gets pushed out by two fresh routine goals (see GoalOrigin). RESIDUE stays bounded by this cap
    and the time-allowance reaping in evaluate_goal_progress.
    """
    live = live_goals(entities)
    while len(live) > _SHORT_TERM_GOAL_CAP:
        victim = next((g for g in live if g.origin == GoalOrigin.COGNITIVE), live[0])
        live.remove(victim)
        victim.status = GoalStatus.FAILED
        victim.last_evaluated_step = current_step
        victim.progress_summary = (
            "搁置——终究没能顾上。"
            if victim.origin == GoalOrigin.RESIDUE
            else "搁置——未及跟进便被新的事挤下。"
        )


def trim_goal_history(entities: List[GoalEntity]) -> None:
    """Keep only the latest _SHORT_TERM_GOAL_HISTORY_CAP terminal goals, in place."""
    terminal_idxs = [i for i, g in enumerate(entities) if not is_live_goal(g)]
    drop = len(terminal_idxs) - _SHORT_TERM_GOAL_HISTORY_CAP
    if drop <= 0:
        return
    for i in sorted(terminal_idxs[:drop], reverse=True):
        del entities[i]


def enqueue_goals(
    entities: Sequence[GoalEntity],
    new_goals: Sequence[str | tuple[str, int | None]],
    *,
    dominant_need: NeedType | None,
    current_step: int,
    seconds_per_step: int = 3600,
    origin: GoalOrigin = GoalOrigin.COGNITIVE,
) -> List[GoalEntity]:
    """Append new short-term goals to the live queue (FIFO); pure, returns a new list.

    Near-verbatim repeats of a live goal are skipped so they can't push out older intentions;
    over capacity the oldest is evicted (_evict_over_cap); the history is then trimmed.
    ``new_goals`` items are bare text or ``(text, hours from now)``.
    """
    result = list(entities)
    seq = _next_short_term_seq(result)
    live_texts = {g.text for g in result if is_live_goal(g)}
    for raw in new_goals:
        text, due_hours = (raw, None) if isinstance(raw, str) else raw
        text = str(text).strip()
        if not text:
            continue
        ratio, matched = _max_text_similarity(text, live_texts)
        if ratio >= _SHORT_TERM_GOAL_DEDUP_RATIO:
            # Info level: dedup is rare and is the main signal for tuning the threshold.
            _logger.info(
                "short_term_goal_dedup_skip",
                extra={
                    "ratio": round(ratio, 3),
                    "exact": ratio >= 1.0,
                    "threshold": _SHORT_TERM_GOAL_DEDUP_RATIO,
                    "candidate": text,
                    "matched": matched,
                },
            )
            continue
        if ratio >= _SHORT_TERM_GOAL_DEDUP_WATCH_FLOOR:
            _logger.info(
                "short_term_goal_dedup_near_admit",
                extra={
                    "ratio": round(ratio, 3),
                    "threshold": _SHORT_TERM_GOAL_DEDUP_RATIO,
                    "candidate": text,
                    "matched": matched,
                },
            )
        result.append(
            GoalEntity(
                id=f"stg-{current_step}-{seq}",
                text=text,
                goal_type="short_term",
                related_need=dominant_need,
                created_step=current_step,
                due_step=_due_step_from_hours(current_step, due_hours, seconds_per_step),
                origin=origin,
            )
        )
        seq += 1
        live_texts.add(text)
        _evict_over_cap(result, current_step)
    trim_goal_history(result)
    return result


def _max_text_similarity(text: str, candidates: Iterable[str]) -> tuple[float, Optional[str]]:
    """(ratio, matched_text) for the literally most similar candidate; (0.0, None) if none."""
    best_ratio = 0.0
    best_match: Optional[str] = None
    for other in candidates:
        ratio = SequenceMatcher(None, text, other).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = other
    return best_ratio, best_match


def text_to_long_term_goal_entity(text: str) -> GoalEntity:
    # related_need=None (wildcard): which need a goal serves is a subjective reading. Don't guess
    # by keyword here (Rule 7); a need tag, if ever wanted, should come from the authoring LLM.
    return GoalEntity(
        id=f"ltg-{abs(hash(text)) % 100000}",
        text=text,
        goal_type="long_term",
        related_need=None,
        created_step=0,
    )


def parse_goal_evaluation(content: str, goals: List[GoalEntity], step: int) -> None:
    """Apply the judge's verdict {"goals": [{"reason", "index" (1-based), "status"}]} in place.

    Only index + status drive the update; reason is kept as progress_summary. Malformed entries
    are skipped.
    """
    _STATUS_MAP = {
        "completed": GoalStatus.COMPLETED, "已完成": GoalStatus.COMPLETED,
        "active": GoalStatus.ACTIVE,       "进行中": GoalStatus.ACTIVE,
        "interrupted": GoalStatus.INTERRUPTED, "被中断": GoalStatus.INTERRUPTED,
        "failed": GoalStatus.FAILED,       "失败": GoalStatus.FAILED,
    }
    try:
        data = extract_json(content)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    entries = data.get("goals", [])
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            pos = int(entry.get("index")) - 1  # 1-based → 0-based
        except (TypeError, ValueError):
            continue
        if not 0 <= pos < len(goals):
            continue
        new_status = _STATUS_MAP.get(str(entry.get("status", "")).strip().lower())
        if new_status is None:
            continue
        goals[pos].status = new_status
        goals[pos].last_evaluated_step = step
        reason = entry.get("reason")
        if reason:
            # The prompt caps this; don't slice it here, that would cut a persisted summary
            # mid-sentence.
            goals[pos].progress_summary = str(reason).strip()


def parse_residue(content: str) -> tuple[List[tuple[str, int | None]], str]:
    """Extract this action's residue (new unfinished business) and the judge's residue_reason.

    An empty list is a legitimate answer (most actions leave nothing), so every anomaly also falls
    back to it (Rule 1 tier-1: better nothing than an invented intention that gets persisted).
    residue_reason makes the judge look at the evidence first, curbing filler items, and goes into
    the trace for auditing over-generation; enqueuing ignores it.
    """
    try:
        data = extract_json(content)
    except Exception:
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    reason = str(data.get("residue_reason", "") or "").strip()[:_RESIDUE_REASON_MAX_CHARS]
    raw = data.get("residue", [])
    if not isinstance(raw, list):
        return [], reason
    out: List[tuple[str, int | None]] = []
    for item in raw:
        if parsed := parse_goal_item(item):
            out.append(parsed)
    return out, reason
