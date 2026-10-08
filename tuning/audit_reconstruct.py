"""Reconstruct a browsable audit view from a world's persisted runtime traces.

Auditing is post-hoc and offline: it only reads the ``LLMCallTrace`` records already persisted in
``data/traces/<world_id>/`` (prompt + response + stage + agent_id + step) and never reruns the
simulation. This module regroups the flat JSONL calls into the shapes the three-part audit needs:

- ``init``          — the build stage's world_building / cast_design / persona outputs (part one).
- ``per_agent``     — each agent's per-stage trajectory across all steps (part two: across / along).
- ``per_step``      — each agent's stage outputs within a step (part three: one step across agents).
- ``world_sequence``— the world skeleton across steps (time + injected events); each agent's cell goes through ``cells_at``.

The audit's basic unit is the step: everything that happened to one body in one step.
Occasionally two things land in one step (an interrupt tears down an in-flight execution and
settles it on the spot, then a new round of cognition runs). Multi-valued slots are then stored as
lists in arrival order; see ``summarize``.

Parsing always goes through ``extract_json_object`` (never raises; prose responses like
memory_summarization are kept as-is) and tolerates gaps: a missing field in the audit view is one
less piece of evidence and shouldn't crash the whole audit. Ids are allowed here (this is purely
code-layer offline analysis, and reading ``LLMCallTrace`` legitimately involves ids; see the
exception in CLAUDE.md's narrative/code layer boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.interfaces.llm import IndexedRef, extract_json_object
from core.interfaces.trace import LLMCallTrace, stage_sort_key
from engine.clock import SECONDS_PER_HOUR
from interaction.models import WorldTimeView
from providers.trace.file import BufferedJsonlTraceSink

# Every input/state field the audit reads is attached by the cognition path to that stage's LLM call
# through annotate_call into LLMCallTrace.extra (contract: core.context.annotate_call → LLMRouter →
# LLMCallTrace.extra). The audit reads structured values from extra only and never regexes the
# prompt: that would let prompt wording changes and the prefix-cache system/user split affect the
# audit. Producers of each extra key (renaming a key means changing both sides):
#   every call in the five cognition beats: given_facts (every fact put in front of this call,
#                     each with a channel prefix)
#                     ← agent/decision.py _build_decision_prompt (produced from the same source as the user prompt)
#                     ← agent/need.py motivation goal generation / goal progress verdict
#                     ← engine/executors/* adjudication points (via engine/executors/base.py declared_facts)
#                     ← agent/agent.py perception emotion / feedback self-assessment
#                     sass's "did the output add facts from nowhere" reads only this, without
#                     splitting the prompt or knowing any slot names or candidate lists. If a new
#                     input isn't declared, the audit will flag sourced statements as fabricated.
#   decision call:    dropped_indices (indices filled in but outside the list; silently dropped at runtime, invisible unless reported)
#                                                                      ← agent/decision.py _Slots
#   every cognition beat: location (where he was on this beat; time belongs to the step, not here)
#   perception call:  prior_mood / relations / perceived
#                                                                      ← agent/agent.py _assess_perception_emotion
#   interrupt call:   interrupt_doing / interrupt_trigger              ← agent/agent.py evaluate_interrupt
#   motivation call:  dominant_need / prior_goals                      ← agent/need.py motivation goal generation
#   decision call:    action_menu / person_candidates / destination_candidates /
#                     item_candidates / npc_candidates                 ← agent/decision.py _attempt_selection
#                     verdict (what arbitration did with this intent) / spans_steps (multi-step start, from the execution)
#                                                                      ← engine/runtime.py commit point
#   action call:      action_owner / conscripted_input / talk_role / acted_upon ← engine/executors/*
#                     action_description / action_initiator / action_participants (which action
#                     landed this step, who started it, who was there)  ← engine/executors/base.py
#                                                                        execution_annotations, attached at adjudication
#   feedback call:    goal_texts / residue (goal progress) / feedback_action_* (appraisal)
#                                                                      ← agent/need.py; agent/agent.py _llm_emotion
#   relation evolution call: relation_targets (index → who the other party is) ← agent/relation_evolution.py
#   long-term goal review call: long_term_goals_new (the version the engine finally adopted; empty = direction unchanged) ← agent/agent.py revise_long_term_goals
#   every action-related call (action / feedback / memory): execution_id ← engine/execution_processor.py adjudication + landing,
#                                                                        engine/interrupt_coordinator.py interrupt settlement
#   every call in the step-start settlement phase: settles_prior_action (this segment settles the action
#                           already in hand, not this step's new round) ← engine/runtime.py, three places at step start
# ``execution_id`` is the engine's own key for joining action records; the audit doesn't infer from it.
# "What happened to this intent" is stated outright by the verdict arbitration stamps, not reverse-
# engineered from whether there's an id, whether there's an action call, or what talk_role is.
# Candidate/target maps (action_menu / *_candidates / goal_texts) are keyed by stringified indices:
# JSON persistence turns int keys into str, so the parser always looks up str(idx).


def _menu_lookup(mapping: Any, idx: Any) -> str | None:
    """Look up a value by index in a {str(index): value} map; idx may be int/str; missing/invalid/out-of-range → None."""
    if idx is None or not isinstance(mapping, dict):
        return None
    return mapping.get(str(idx))


# Each target slot in the decision output → the candidate list it binds to. Listed per slot, not
# merged: the list is part of the field name (single source of truth: agent/decision.py's output
# schema and parse branches). Merging would render "pin someone down" as the entity with the same
# index in another list, wrong in both entity and kind. A missing channel means the action has no
# object in the audit and the judge can't tell whether it worked.
#   (slot name, list key, render prefix)
_TARGET_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("person_indices",           "person_candidates",      ""),
    ("destination_index",        "destination_candidates", "→"),
    ("physical_person_index",    "person_candidates",      ""),
    ("physical_npc_index",       "npc_candidates",         ""),
    ("physical_entity_index",    "item_candidates",        "[物]"),
    ("physical_recipient_index", "person_candidates",      "交予:"),
    ("move_carry_indices",       "person_candidates",      "带走:"),
    ("errand_npc_index",         "npc_candidates",         "遣:"),
    ("errand_destination_index", "destination_candidates", "→"),
    ("errand_item_index",        "item_candidates",        "[物]"),
    ("errand_recipient_index",   "person_candidates",      "说与:"),
)


def _resolve_targets(extra: dict[str, Any], out: dict) -> str | None:
    """Translate the decision output's index targets into names / locations / things / errand-runners using the candidate maps in extra.

    Index → name comes from the extra attached at the decision stage, not the prompt; index bases match
    what decision.py prints (people/locations/things/NPCs 1-based, action_menu 0-based). Lists are
    split per slot; see ``_TARGET_SLOTS``.
    """
    parts: list[str] = []
    for slot, menu_key, prefix in _TARGET_SLOTS:
        raw = out.get(slot)
        idxs = raw if isinstance(raw, list) else ([] if raw is None else [raw])
        for idx in idxs:
            if (nm := _menu_lookup(extra.get(menu_key), idx)) and f"{prefix}{nm}" not in parts:
                parts.append(f"{prefix}{nm}")
    return "、".join(parts) or None


# ---------------------------------------------------------------------------
# View models
# ---------------------------------------------------------------------------

@dataclass
class StageCall:
    """One LLM call for a (agent, step) at a given cognition stage."""

    stage: str
    scene: str
    prompt: list[dict[str, str]]
    output: Any  # parsed dict, or raw prose string when the response is not JSON
    parse_ok: bool | None
    ok: bool
    # Did the engine actually use this output? False = a well-formed response the consumer
    # threw away, so the beat produced nothing — invisible to ok/parse_ok, and the judge
    # must not read the discarded output as if it had shaped the world. None = no verdict.
    adopted: bool | None = None
    reject_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)  # code-layer diagnostics (annotate_call)


@dataclass
class AgentStepView:
    """Everything that happened to one body in one step; the step is the audit's basic unit.

    Occasionally two things land in one step (an interrupt tears down an in-flight execution and
    settles it on the spot, then a new round of cognition runs; or the agent is pulled into someone
    else's action in the step it settles). ``summarize`` stores such multi-valued slots as lists in
    arrival order. Don't merge them (merging pairs A's result with B's inputs), and don't split them
    into numbered "beats" (that needs machinery whose only job is reassembling them).
    """

    step: int
    world_time: str
    calls: list[StageCall] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    agent_id: str
    name: str
    steps: list[AgentStepView] = field(default_factory=list)


@dataclass
class WorldAuditView:
    world_id: str
    init: dict[str, Any]
    per_agent: dict[str, AgentTrajectory]
    per_step: dict[int, dict[str, list[StageCall]]]
    world_sequence: list[dict[str, Any]]
    agent_names: dict[str, str]

    def cells_at(self, step: int) -> list[tuple[str, str, AgentStepView]]:
        """Every body present this step → (agent_id, name, its cell).

        All three per-step scopes take their cells from here (mass per character, mams per step ×
        character, the world sequence) rather than each digging through ``per_agent``. With two ways to
        get the same cells, one drifts (keying on names silently collides characters with the same name,
        and dropping agent_id loses the link back to the persona).
        """
        out: list[tuple[str, str, AgentStepView]] = []
        for aid in self.per_step.get(step, {}):
            traj = self.per_agent.get(aid)
            if traj is None:
                continue
            sv = next((x for x in traj.steps if x.step == step), None)
            if sv is not None:
                out.append((aid, traj.name, sv))
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(call: LLMCallTrace) -> Any:
    """Parsed dict for JSON responses; raw string for prose (parse_ok is None)."""
    obj = extract_json_object(call.response_content)
    return obj if obj is not None else call.response_content


# Output of the step-end maintenance phase: runs after every execution has settled, at most once per step, and belongs to no action.
_STEP_LEVEL_STAGES = {"relation_evolution", "long_term_goals", "reflection"}

# When an action is discarded, the agent's motivation output and decision for this step are void and
# shouldn't be injected. Injecting an intent that never executed would only make the judge compare
# it with someone else's action result below and flag "decision inconsistent with action".
_DISCARDED_VERDICTS = {"conscripted", "stood_down"}


def summarize(calls: list[StageCall], agent_name: str = "") -> dict[str, Any]:
    """One block per step: everything that happened to this body this step.

    The unit is the step, not the "beat". But inside a step there is a real order: the engine first
    settles the action in hand (an interrupt tears it down on the spot, feedback lands), then runs a
    new round of cognition. Mixing the two in one set of slots would pair the new decision with the
    interrupted action's result and have two emotions share one "before".

    Which segment a call belongs to is stated by the engine (``extra.settles_prior_action``, stamped
    by engine/runtime at three places at step start). The audit doesn't guess from arrival order or
    stage name: memory writes go through an async queue, so the settled action's memory prose arrives
    late, after the new round's calls, while this step's perception memory arrives before the
    interrupt. Any order-based rule breaks in both places, and would silently misalign everything the
    moment the engine reorders its phases.

    ``agent_name`` lets the action stage drop the other party's first-person memory (a two-party TALK
    traces both sides' facts under the initiator's agent_id).
    """
    carried = [c for c in calls if c.extra.get("settles_prior_action")]
    fresh = [c for c in calls if not c.extra.get("settles_prior_action")]
    if not carried or not fresh:  # nothing carried over, or the whole step is settlement — neither needs splitting
        return _fold(calls, agent_name)
    summary = _fold(fresh, agent_name)
    summary["carried"] = _fold(carried, agent_name)
    return summary


def _fold(calls: list[StageCall], agent_name: str) -> dict[str, Any]:
    """Fold one segment's calls (within one phase) into a summary block. Two things can still land in
    one phase (pulled into someone else's action in the step it settles); multi-valued slots are then
    listed one by one, not merged: merging pairs A's result with B's inputs."""
    s: dict[str, Any] = {}
    discarded: list[str] = []
    for c in calls:
        if c.adopted is False:
            # Output that never entered the world stays out of the summary: nothing happened. A discarded decision is reported explicitly (see the end).
            if c.stage == "decision":
                discarded.append(c.reject_reason or "未注明")
            continue
        out = c.output if isinstance(c.output, dict) else {}
        # Where he is, regardless of beat: perception, motivation, decision and feedback each declare that
        # beat's location; take the first to arrive. Relying on perception alone misses a whole class of
        # cells: steps where perception wasn't scheduled (only settling an old action, only feedback). He's
        # still somewhere, but "他在：" can't be rendered, and whole locations vanish from the view.
        if not s.get("location") and c.extra.get("location"):
            s["location"] = c.extra["location"]
        if c.stage in _STEP_LEVEL_STAGES:
            s[c.stage] = out or c.output
            if c.stage == "long_term_goals" and isinstance(s[c.stage], dict):
                # Whether anything changed is stated by the engine (agent.revise_long_term_goals's post-call tag): goals
                # the LLM writes are discarded whole in two cases (empty array = direction unchanged, identical to the
                # old ones), which the LLM output alone can't distinguish.
                s[c.stage]["applied"] = c.extra.get("long_term_goals_new") or []
            if c.stage == "relation_evolution":
                # target_index → who the other party is (extra.relation_targets, attached by agent/relation_evolution).
                # Judging "is this relation reversal abrupt" requires knowing whom it's about; an index alone says nothing.
                for u in (out.get("updates") or []) if isinstance(out, dict) else []:
                    if isinstance(u, dict):
                        u["target"] = _menu_lookup(c.extra.get("relation_targets"), u.get("target_index"))
        elif c.stage == "perception":
            s["emotion"] = {"primary": out.get("emotion"), "intensity": out.get("intensity"),
                            "valence": out.get("valence")}
            s["perception_reason"] = out.get("reason")  # which perceived signal stirred it
            if out.get("need_activation"):
                s["need_activation"] = out["need_activation"]
            # Entering mood / relations / perceived information: structured inputs attached to extra by the perception call.
            for k in ("prior_mood", "relations", "perceived"):
                if c.extra.get(k) is not None:
                    s[k] = c.extra[k]
        elif c.stage == "interrupt":
            s["interrupt"] = {
                "thought": out.get("thought"), "interrupt": out.get("interrupt"),
                "doing": c.extra.get("interrupt_doing"),      # input: what he's doing
                "trigger": c.extra.get("interrupt_trigger"),  # input: the sudden signal
            }
        elif c.stage == "motivation":
            # New short-term goals = what the engine actually enqueued after dedup (extra.short_term_goals_new, attached by need.py).
            # Near-duplicates removed by enqueue_goals' literal dedup aren't included, so the audit won't mistake them for new goals (→ false "idling").
            new_goals = c.extra.get("short_term_goals_new")
            s["short_term_goals"] = new_goals if new_goals is not None else out.get("goals")
            s["motivation_thought"] = out.get("thought")
            for k in ("dominant_need", "prior_goals"):
                if c.extra.get(k) is not None:
                    s[k] = c.extra[k]
        elif c.stage == "decision":
            # Arbitration verdict (extra.verdict, stamped by engine/runtime at the commit point): what the world
            # did with this intent. Discarded ones (conscripted / yielded) aren't injected at all, nor are this
            # step's motivation and goals (the engine already marked them not adopted; the adopted check above
            # filters them). An intent that never executed would only be compared with someone else's action
            # result below and flagged as fabrication / a broken chain.
            if c.extra.get("verdict") in _DISCARDED_VERDICTS:
                s.pop("short_term_goals", None)
                s.pop("motivation_thought", None)
                s["intent_dropped"] = c.extra["verdict"]
                continue
            s["decision"] = {
                "action_type": _menu_lookup(c.extra.get("action_menu"), out.get("selected_index")),
                "selected_index": out.get("selected_index"),
                "action_description": out.get("action_description"),
                "expected_outcome": out.get("expected_outcome"),
                "inner_monologue": out.get("inner_monologue"),
                "target": _resolve_targets(c.extra, out),  # index → name / location / thing (via the extra candidate maps)
                # The exact words actually sent (god's-eye). SEND_MESSAGE and ERRAND each have their own slot; to
                # the audit they're the same thing: which words went out this step.
                "message_content": out.get("message_content") or out.get("errand_message"),
                # Multi-step actions are marked only at the start: remaining steps come from the execution (not
                # the decision's self-reported estimated_steps, which is an estimate). Later steps render as usual
                # and readers can see it's the same action continuing.
                "spans_steps": c.extra.get("spans_steps"),
                "rejected": c.extra.get("verdict") == "rejected",
            }
        elif c.stage == "action":
            _fold_action(s, c, out, agent_name)
        elif c.stage == "feedback":
            if c.scene == "need_goal_generation":
                goals = out.get("goals")
                texts = c.extra.get("goal_texts") or {}  # {str(index): goal text}, attached by the feedback call
                if isinstance(goals, list) and isinstance(texts, dict):
                    for g in goals:
                        if isinstance(g, dict) and str(g.get("index")) in texts:
                            g["text"] = texts[str(g["index"])]
                _append(s, "goal_progress", goals)
                _append(s, "residue", c.extra.get("residue"))
            else:
                _append(s, "appraisal", {"emotion": out.get("emotion"),
                                         "intensity": out.get("intensity"),
                                         "valence": out.get("valence"),
                                         "of": c.extra.get("feedback_action_desc")})
                # When the action stage has no result for this agent (in a two-party TALK the other party's fact is
                # attached to the initiator's id; see the TODO), fall back to this agent's result attached to extra by feedback.
                if not s.get("results") and (actual := c.extra.get("feedback_action_actual")):
                    # There's no action call on this path (the action failed before adjudication), so no participant list;
                    # feedback's own description says which action it was.
                    _append(s, "results", {"did": c.extra.get("feedback_action_desc"), "fact": actual})
        elif c.stage == "memory":
            if c.scene == "memory_importance":
                _append(s, "memory_importance", out.get("score"))
            else:  # memory_summarization → first-person memory prose
                _append(s, "memory", c.output if isinstance(c.output, str) else None)
    # Nothing adopted = nothing happened this step, and that has to be said: a bare "no decision" reads as missing data or mechanical repetition.
    if discarded and "decision" not in s:
        s["decision_discarded"] = discarded[0]
    return s


def _append(s: dict[str, Any], key: str, value: Any) -> None:
    """Append multi-valued slots in arrival order. A step can hold two (interrupt settlement + the newly
    started action); the later write must not overwrite the earlier. Overwriting would make an
    emotion / memory / goal progress vanish, and the judge, seeing only what remains, couldn't tell anything was missing."""
    if value in (None, [], ""):
        return
    s.setdefault(key, []).append(value)


def _fold_action(s: dict[str, Any], c: StageCall, out: dict, agent_name: str) -> None:
    """Action stage: the result from this agent's viewpoint + whether this step he was invited in or acted upon.

    A two-party TALK produces 3 action calls in one step (the dialogue + each side's first-person
    memory), all under the initiator's agent_id. 【我是谁】 is used to drop the other party's memory,
    keeping only this agent's view + the neutral dialogue.

    Each result carries its own action (did/initiator/participants, extra attached at adjudication):
    an action in a step doesn't necessarily match that step's decision (a multi-step action settles in
    the step it finishes; a conscripted agent has no decision of its own). Without carrying it, those
    steps would be left with an ownerless result. Carrying it with the result also keeps two things
    in one step from being mixed up.
    """
    if c.extra.get("talk_role") == "addressee":
        # Invited in: what he did this step wasn't his own decision. Who invited him to talk + the topic come from extra.
        _append(s, "joined", c.extra.get("conscripted_input"))
    parts = c.extra.get("action_participants") or []
    if len(parts) == 2 and agent_name in parts:
        # In a two-person action, the participant list says who the other party is, so the dialogue line
        # needn't fall back to "对方". With three or more there's no single "other"; leave it empty. Also
        # empty when the two share a name and the list can't single out the other.
        if peer := next((p for p in parts if p != agent_name), None):
            s.setdefault("peer", peer)
    owner = c.extra.get("action_owner")
    if owner is not None and agent_name and owner != agent_name:
        # The other party's first-person memory: not this agent's, but its 【我是谁】 is exactly the counterpart's name, so keep it as the other side of the dialogue.
        s.setdefault("peer", owner)
        return
    # Acted upon: this agent is the object of someone else's physical action (owner == self, so it isn't dropped above).
    if au := c.extra.get("acted_upon"):
        _append(s, "acted_upon", au)
    fields = {k: out.get(k) for k in
              ("outcome", "fact", "dialogue", "success", "achieved", "detected") if k in out}
    if fields:
        whose = {k: v for k, v in (("did", c.extra.get("action_description")),
                                   ("initiator", c.extra.get("action_initiator")),
                                   ("participants", parts)) if v}
        _append(s, "results", {**whose, **fields})


def _build_init(build_calls: list[LLMCallTrace]) -> tuple[dict[str, Any], dict[str, str]]:
    """Assemble the initialization view + an agent_id→name map from build.jsonl calls."""
    world_building: dict[str, Any] = {}
    cast_roles: Any = None
    personas: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for c in build_calls:
        obj = extract_json_object(c.response_content)
        if not isinstance(obj, dict):
            continue
        if c.scene == "world_building":
            world_building = obj
        elif c.scene == "cast_design":
            cast_roles = obj.get("roles")
        elif c.scene == "persona_generation":
            # Identity comes from the trace's own agent_id / agent_name, not the echo: the model miscopies ids,
            # and production (AgentGenerator._definition_from_payload) doesn't trust the echoed name / agent_id either.
            aid = c.agent_id or obj.get("agent_id")
            name = c.extra.get("agent_name") or obj.get("name")
            if aid:
                obj["agent_id"] = aid
            if name:
                obj["name"] = name
            personas.append(obj)
            if aid and name:
                names[str(aid)] = str(name)
    # Step duration is set by this build itself. The audit reads the raw LLM output, whose unit is hours
    # (production quantizes to seconds in _analysis_from_payload), so the same conversion is applied
    # here to express the trajectory's estimated_steps as natural durations; the narrative layer has no "step" unit.
    time_config = world_building.get("world_time_config") or {}
    hours_per_step = time_config.get("hours_per_step") if isinstance(time_config, dict) else None
    seconds_per_step = (
        int(hours_per_step) * SECONDS_PER_HOUR if isinstance(hours_per_step, int) else None
    )
    # Roles name their figure by roster number, as CastDesigner._parse reads them.
    roster = IndexedRef(
        str(f.get("name")) for f in world_building.get("key_figures") or [] if isinstance(f, dict)
    )
    for role in cast_roles if isinstance(cast_roles, list) else []:
        if isinstance(role, dict) and (resolved := roster.resolve([role.get("index")])):
            role["name"] = resolved[0]
    init = {
        "world_name": world_building.get("world_name"),
        # Opening things and errand-runners: like personas, they're the initial world fixed at build time
        # for the whole run. The audit needs to see them to judge whether the opening has anything to fight
        # over or anyone to send; a world with no affordances can only move forward through talk.
        "world_entity_seeds": world_building.get("world_entity_seeds") or [],
        "npcs": world_building.get("npcs") or [],
        "core_tension": world_building.get("core_tension"),
        "narrative_theme": world_building.get("narrative_theme"),
        "key_figures": world_building.get("key_figures") or [],
        "initial_relations": world_building.get("initial_relations") or [],
        "historical_events": world_building.get("historical_events") or [],
        "world_time_config": world_building.get("world_time_config") or {},
        "seconds_per_step": seconds_per_step,
        "cast_roles": cast_roles or [],
        "personas": personas,
    }
    return init, names


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def reconstruct_world(
    world_id: str,
    *,
    trace_dir: str = "data/traces",
    steps: list[int] | None = None,
) -> WorldAuditView:
    """Read a world's traces and reshape them for the audit judges.

    ``steps`` (optional) restricts the single/multi-step views to a subset; the
    initialization view always reads the full build segment.
    """
    sink = BufferedJsonlTraceSink(base_dir=trace_dir)
    all_calls = sink.read_calls(world_id)

    build_calls = [c for c in all_calls if c.step is None or c.stage == "world_init"]
    init, names = _build_init(build_calls)

    world_times = {
        s.step: WorldTimeView.from_payload(s.world_time).label
        for s in sink.read_step_summaries(world_id)
    }
    step_ids = steps if steps is not None else sink.list_steps(world_id)
    step_set = set(step_ids)

    per_agent: dict[str, AgentTrajectory] = {}
    per_step: dict[int, dict[str, list[StageCall]]] = {}
    # world-level calls (agent_id is None): event / director, grouped by step. pressure hangs under the
    # agent being assessed, so it's not a world-level call.
    world_calls: dict[int, list[StageCall]] = {}

    for c in all_calls:
        if c.step is None or c.step not in step_set:
            continue
        sc = StageCall(
            stage=c.stage, scene=c.scene, prompt=c.prompt_messages,
            output=_parse(c), parse_ok=c.parse_ok, ok=c.ok,
            adopted=c.adopted, reject_reason=c.reject_reason,
            extra=c.extra or {},
        )
        if not c.agent_id:
            world_calls.setdefault(c.step, []).append(sc)
            continue
        # TODO(trace-attribution): a two-party TALK traces both sides' first-person facts under the
        # initiator's agent_id, so the target's fact is missing from its action stage. Fix that in the
        # trace layer, not here. Meanwhile the audit falls back to feedback_action_actual (attached under
        # this agent by the feedback call); once the trace is fixed the fallback stops firing on its own.
        per_step.setdefault(c.step, {}).setdefault(c.agent_id, []).append(sc)
        traj = per_agent.get(c.agent_id)
        if traj is None:
            traj = AgentTrajectory(agent_id=c.agent_id, name=names.get(c.agent_id, c.agent_id))
            per_agent[c.agent_id] = traj

    # Fold each agent's calls into AgentStepViews ordered by step. Arrival order is the true order within
    # a step; stage order is only for aligning the display with the cognition loop, not for inferring what came first.
    agent_step_calls: dict[str, dict[int, list[StageCall]]] = {}
    for step, by_agent in per_step.items():
        for aid, calls in by_agent.items():
            agent_step_calls.setdefault(aid, {})[step] = calls
    for aid, by_step in agent_step_calls.items():
        traj = per_agent[aid]
        for step in sorted(by_step):
            arrived = by_step[step]
            traj.steps.append(AgentStepView(
                step=step, world_time=world_times.get(step, ""),
                calls=sorted(arrived, key=lambda c: stage_sort_key(c.stage)),
                summary=summarize(arrived, traj.name),
            ))

    # World skeleton: each step's time + injected events. Agents' cells aren't copied here (see cells_at).
    world_sequence: list[dict[str, Any]] = []
    for step in sorted(step_set):
        wcalls = world_calls.get(step, [])
        world_sequence.append({
            "step": step,
            "world_time": world_times.get(step, ""),
            "events": [c.output for c in wcalls if c.stage == "event"],
        })

    return WorldAuditView(
        world_id=world_id,
        init=init,
        per_agent=per_agent,
        per_step=per_step,
        world_sequence=world_sequence,
        agent_names=names,
    )
