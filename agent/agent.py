"""Agent aggregate root."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

_PERCEPTION_MEMORY_TOP_K = 5  # memories per stream (factual + experiential) entering LLM context

from agent.decision import DecisionEngine, DecisionStatus
from agent.memory import MemorySystem
from agent.memory_types import Memory, MemoryImportance, MemoryStream
from agent.reflection import ReflectionEngine
from agent.relation_evolution import RelationEvolution
from agent.goals import GoalEntity, GoalStatus
from agent.need import NeedEngine, NeedEvaluation, NeedType
from agent.perception import InternalContext, PerceptionPacket, RetrievalQuery
from agent.perception_emotion import (
    PerceptionAppraisal, build_perception_emotion_prompt, emotion_from_payload,
    parse_perception_emotion_response,
)
from agent.personality import (
    AgentActivityStatus,
    activity_status_for,
    EmotionState,
    EmotionType,
    PersonalityLayer,
    StateLayer,
)
from agent.motivation import ExternalGoal
from core.context import (
    GivenFacts, annotate_active_call, annotate_call, note_active_call_adoption,
    observe_stage,
)
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.interfaces.trace import Stage
from core.logging import get_logger
from agent.relation import (
    PerceivedRelation,
    RelationSystem,
    render_relation_context,
)
from core.interfaces.action import ActionResult, ActionType, AgentAction, TargetAgentEffect
from core.interfaces.agent_store import AgentRelation, AgentState, AgentStoreProvider
from core.interfaces.condition import condition_to_dict
from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, PerceivedIdentity, Situation, SpatialPerception
from core.prompts import (
    EMOTION_INTENSITY_DEFINITION,
    EMOTION_VALENCE_DEFINITION,
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    DECEASED_MARK,
    MEMORY_ORDER_HINT,
    URGENCY_TO_STRENGTH,
    SituationVoice,
    order_memories_chrono,
    condition_line,
    person_referent,
    render_memory,
    render_memory_lines,
    render_perceived_signals,
    render_situation_header,
    situation_location,
)
from agent.perception_layer import PerceptionMemoryLayer


logger = get_logger(__name__)

_KNOWN_RELATION_LIMIT = 8  # max related-but-absent people merged into perception per step, by salience
_GOAL_EVAL_RECENT_FACTUAL_K = 6  # recent factual memories fed to short-term goal completion (trajectory evidence for multi-step completion)
_INTERRUPT_RECENT_FACTUAL_K = 5  # recent factual memories fed to interrupt evaluation (places the event in a local arc: escalation or noise)
# Recent factual memories fed to feedback emotion as backdrop: the same outcome feels different after
# what came before (a third loss != the first). Focus on the new outcome comes from the block header's
# "backdrop only" framing, not from keeping this count low.
_FEEDBACK_EMOTION_RECENT_FACTUAL_K = 6
# Recent factual memories fed to perception emotion as background, to judge continuation/escalation.
# Must exclude the current step: its perceptions are already in the signal block and would repeat.
_PERCEPTION_EMOTION_RECENT_FACTUAL_K = 6
# Recent-memory supplement for decide: vector recall misses just-happened events unrelated to the scene
# (see _supplement_recent). More factual (objective anchors), fewer experiential (feelings make decide drift).
_DECIDE_RECENT_FACTUAL_SUPPLEMENT_K = 5
_DECIDE_RECENT_EXPERIENTIAL_SUPPLEMENT_K = 2
# Forgetting period for foiled counts: a key untouched for this many steps is dropped. Keep close to
# _FOILED_LOOKBACK_STEPS (how long the same matter stays visible in the decision prompt).
_FOILED_MISS_AGE_STEPS = 5


def _foiled_key(action: "AgentAction") -> "tuple[str, str]":
    """Merge key for an action that failed to happen: (action type, object acted on).

    MemorySystem.recent_foiled_attempts' primary "same thing keeps not working" test: stable
    across rewording, unlike literal similarity.

    Object-less actions (WORK/REST/COVERT) get a half key (empty object). It means "same kind of
    act", not "same matter", so only merged display uses it (backed by the stricter
    ``_FOILED_MERGE_MIN_RATIO_NO_OBJECT``); arbitration initiative counting and clearing on
    success accept only full keys (``_foiled_key_has_object``).

    Keys on ``acts_on``, not everyone in the target: an escorting move acts on the destination,
    and keying on the escorted person would merge "take A to X" with "take A to Y".

    Multiple referents use the sorted full set: the first would drift with LLM output order, and
    a different recipient set is a different matter (not merging falls back to itemized listing).
    """
    target = action.target
    ref = ",".join(sorted(target.acted_on_ids)) if target is not None else ""
    action_type = action.action_type
    return (getattr(action_type, "value", str(action_type)), ref)


def _foiled_key_has_object(key: "tuple[str, str]") -> bool:
    """A key identifies a specific matter only if its object half is non-empty."""
    return bool(key[1])


def _supplement_recent(
    recalled: List[Memory], recent: Iterable[Memory | None], *, limit: int
) -> List[Memory]:
    """Append recent memories missing from vector recall (deduped by id), giving decide anchors
    for autobiographical continuity.

    Recall candidates pass the hard RETRIEVAL_SCORE_FLOOR gate (MemorySystem.retrieve) before recency
    weighting, so a just-happened but off-topic event is dropped; this adds back only those.

    *recent* is chronological ascending; its latest *limit* new entries are appended after
    ``recalled``. Order doesn't matter: decide re-sorts with order_memories_chrono.
    """
    seen = {m.id for m in recalled}
    extras: List[Memory] = []
    for m in recent:
        if m is None or m.id in seen:
            continue
        seen.add(m.id)
        extras.append(m)
    if limit and len(extras) > limit:
        extras = extras[-limit:]  # ascending order → the tail is the latest `limit`
    return recalled + extras


@dataclass(frozen=True)
class AgentStepPlan:
    """Read-only plan produced before environment arbitration."""

    agent_id: str
    step: int
    spatial: SpatialPerception
    inbox: List[Message]
    broadcasts: List[Broadcast]
    need_evaluation: NeedEvaluation
    # None for both NO_ACTION (deliberate inaction) and FAILED (decision LLM unavailable); the
    # runtime skips both identically with zero state change. Only ``decision_status`` tells them
    # apart, and only at the code layer.
    action: AgentAction | None
    decision_status: DecisionStatus = DecisionStatus.ACTED
    consumed_external_goals: List[ExternalGoal] = field(default_factory=list)
    # How many recent beats this action's matter failed to land (Agent._foiled_misses_for);
    # arbitration owns the initiative threshold. Keyed by the action, not the agent, so the boost
    # follows only the blocked intent.
    foiled_misses: int = 0


@dataclass
class Agent:
    """Single autonomous agent."""

    world_id: str
    agent_id: str
    personality: PersonalityLayer
    decision_engine: DecisionEngine
    memory_system: MemorySystem
    need_engine: NeedEngine
    relation_system: RelationSystem
    agent_store: AgentStoreProvider
    # For the agent's own feedback/interrupt/perception-emotion cognition: the same shared router
    # the subsystems receive, injected directly rather than borrowed from a subordinate component.
    llm_router: LLMRouter
    pending_external_goals: List[ExternalGoal] = field(default_factory=list)
    # Narrative tier, not a cognition-mode switch: every agent's cognition gates use the LLM
    # (CLAUDE.md §5).
    is_main_character: bool = False
    # Turns memory created_step into relative recency (render_memory).
    seconds_per_step: int = 3600
    # The memory prefix's "today/yesterday" counts midnights crossed, which a duration alone can't give.
    world_start_second_of_day: int = 0
    is_active: bool = True
    # Narrative-only cause-of-death phrase (no id/step) set by _trigger_death, read by
    # DeathHandler.process_new_deaths for the death notice. None while alive.
    death_cause: str | None = None
    reflection_engine: ReflectionEngine | None = None
    # Scheduled periodically by the runtime.
    relation_evolution: RelationEvolution | None = None

    def __post_init__(self) -> None:
        self._perception_layer = PerceptionMemoryLayer(
            memory_system=self.memory_system,
            relation_system=self.relation_system,
            agent_id=self.agent_id,
            is_main_character=self.is_main_character,
        )
        # Perceived agent_id → PerceivedIdentity, for record_event's Mapping form; refilled from
        # spatial.visible_agents each perceive, not persisted. Name and gender share one map so
        # they can't drift apart.
        self._known_agents: Dict[str, PerceivedIdentity] = {}
        # This step's "when and where I am" for paths with no plan context (commit/finalize/
        # interrupt) to inject as their header. Perception is the only legitimate source of names/
        # time (agent/ may not use the directory). Cache only the Situation, not the whole
        # SpatialPerception, so stale visible_agents/ambient can't be misread later. Not persisted.
        self._situation: Situation = Situation()
        # This step's dead, for "deceased" marks at plan and feedback time; refreshed each step like
        # _situation. Written by perceive_step and note_deaths.
        self._dead_agent_ids: "frozenset[str]" = frozenset()
        # _foiled_key → (foiled count in window, step of last foil), feeding arbitration initiative.
        # Means "N times recently", not "in a row": other actions in between push the same matter
        # and must not reset it. Only the matter actually happening clears it (_clear_foiled).
        # Separate from MemorySystem._recent_foiled, which restarts on rewording and gives no
        # decidable N. Not persisted: the foiled buffer isn't snapshotted either, and restoring only
        # the count would boost an intent the agent no longer remembers being blocked on.
        self._foiled_misses: Dict[tuple[str, str], tuple[int, int]] = {}

    def remember_agent(self, agent_id: str, identity: PerceivedIdentity) -> None:
        """Cache an agent's perceived name + gender.

        Empty names are ignored, and an empty gender keeps the known one: a message sender carries
        only a name and mustn't clear a gender seen in person.
        """
        if not identity.name:
            return
        known = self._known_agents.get(agent_id)
        gender = identity.gender or (known.gender if known else "")
        self._known_agents[agent_id] = PerceivedIdentity(name=identity.name, gender=gender)

    def _resolve_agent_names(self, agent_ids) -> Dict[str, str]:
        """agent_id → display name for memory prose; unknown ids become "某人" (someone), never a
        bare id (a layer leak). Bare names only: person_referent's parentheses are for rosters.
        """
        return {
            aid: ((who.name if (who := self._known_agents.get(aid)) else "") or "某人")
            for aid in agent_ids
        }

    async def _resolve_relation_context(self, agent_ids) -> str:
        """Build the "my relations with the people involved" block for memory writes.

        Skips empty default relations. Names come from the agent's own knowledge (_known_agents /
        relation.to_name); unknowns get a descriptive referent, never an id.
        """
        named: list[tuple[str, AgentRelation]] = []
        seen: set[str] = set()
        for aid in agent_ids:
            if not aid or aid in seen:
                continue
            seen.add(aid)
            rel = await self.relation_system.load_existing(aid)
            if rel is None:
                continue
            if rel.interaction_count == 0 and not rel.labels and not rel.history_summary:
                continue
            known = self._known_agents.get(aid)
            name = (known.name if known else "") or rel.to_name or ""
            if not name:
                name = "某位与我有往来的人"
            gender = (known.gender if known else "") or rel.to_gender
            marks = (DECEASED_MARK,) if aid in self._dead_agent_ids else ()
            named.append((person_referent(name, gender, *marks), rel))
        return render_relation_context(named)

    async def _event_people_context(self, agent_ids) -> tuple[Dict[str, str], str]:
        """{id: display name} plus the relations block for a memory write.

        Assembled here, where both the name cache and relation_system live, so MemorySystem takes
        only precomputed state and never depends on RelationSystem.
        """
        return (
            self._resolve_agent_names(agent_ids),
            await self._resolve_relation_context(agent_ids),
        )

    def set_active(self, active: bool) -> None:
        self.is_active = active

    async def perceive_step(
        self,
        *,
        spatial: SpatialPerception,
        inbox: List[Message],
        broadcasts: List[Broadcast],
        step: int,
        dead_ids: "frozenset[str]" = frozenset(),
    ) -> None:
        """Single entry point for turning perception into memory, every step for every active agent.

        The runtime calls it after pressure_evaluator and before InterruptCoordinator.evaluate_interrupts. External
        pressure (pending_external_goals) is not an event and is never written to memory; plan_step
        consumes it as a live signal.
        """
        for vid, presence in spatial.visible_agents.items():
            # Identity only: presence.situation is per-step state and would become a stale belief
            # in this persistent cache.
            self.remember_agent(vid, presence.identity)
        for msg in inbox:
            self.remember_agent(
                msg.sender_id,
                PerceivedIdentity(name=getattr(msg, "sender_name", "") or ""),
            )
        # See the __init__ comment; ``refresh_situation`` updates it if he moves again within the beat.
        self._situation = Situation.from_spatial(spatial)
        self._dead_agent_ids = dead_ids
        logger.debug(
            "agent_perceive_step",
            extra={
                "agent_id": self.agent_id,
                "step": step,
                "location": spatial.location_id,
                "visible_agents": len(spatial.visible_agent_ids),
                "inbox": len(inbox),
                "broadcasts": len(broadcasts),
            },
        )
        await self._perception_layer.record(
            spatial=spatial,
            inbox=inbox,
            broadcasts=broadcasts,
            personality=self.personality,
            step=step,
        )

    async def plan_step(
        self,
        *,
        step: int,
        spatial: SpatialPerception,
        inbox: List[Message],
        broadcasts: List[Broadcast],
    ) -> AgentStepPlan:
        """Plan one cognition/action loop without persistence side effects.

        ``step`` comes from the world clock; the agent keeps no step counter of its own, so
        planning can't drift from the clock after a restore.
        """

        plan_start = time.perf_counter()
        current_step = step
        inbox_list = list(inbox)
        broadcasts_list = list(broadcasts)
        consumed_external_goals = list(self.pending_external_goals)
        awareness_start = time.perf_counter()
        internal_context = await self._build_internal_context(
            spatial=spatial,
            inbox=inbox_list,
            broadcasts=broadcasts_list,
            current_step=current_step,
            consumed_external_goals=consumed_external_goals,
        )
        awareness_ms = round((time.perf_counter() - awareness_start) * 1000.0, 2)
        packet = PerceptionPacket(
            agent_id=self.agent_id,
            step=current_step,
            spatial=spatial,
            inbox=inbox_list,
            broadcasts=broadcasts_list,
            internal_context=internal_context,
        )
        decide_start = time.perf_counter()
        with observe_stage(Stage.DECISION, agent_id=self.agent_id):
            decision = await self.decision_engine.decide(
                personality=self.personality,
                packet=packet,
            )
        decide_ms = round((time.perf_counter() - decide_start) * 1000.0, 2)
        action = decision.action
        if action is not None:
            action.agent_id = self.agent_id
        elif decision.status is DecisionStatus.NO_ACTION:
            # Deliberate inaction: the runtime skips it exactly like a failure; only the log level differs.
            logger.info(
                "agent_step_no_action",
                extra={"agent_id": self.agent_id, "step": current_step},
            )
        else:
            # Decision LLM failed even after retry; the runtime skips this agent's step.
            logger.warning(
                "agent_step_no_decision",
                extra={"agent_id": self.agent_id, "step": current_step},
            )
        logger.info(
            "agent_plan_step",
            extra={
                "agent_id": self.agent_id,
                "step": current_step,
                "is_main_character": self.is_main_character,
                "awareness_ms": awareness_ms,
                "decide_ms": decide_ms,
                "elapsed_ms": round((time.perf_counter() - plan_start) * 1000.0, 2),
            },
        )
        return AgentStepPlan(
            agent_id=self.agent_id,
            step=current_step,
            spatial=spatial,
            inbox=inbox_list,
            broadcasts=broadcasts_list,
            need_evaluation=internal_context.need_evaluation,
            action=action,
            decision_status=decision.status,
            consumed_external_goals=consumed_external_goals,
            foiled_misses=self._foiled_misses_for(action, current_step),
        )

    async def _perceive_relevant_relations(
        self, spatial: SpatialPerception, inbox: List[Message], *, current_step: int,
    ) -> "List[PerceivedRelation]":
        """Perceive relations with people in view, agent senders of my messages, and top related
        absent people.

        A remote sender's relation shapes how the message lands. Bias uses last step's emotion,
        avoiding a loop with the perception emotion being computed.
        """
        relevant_ids = list(spatial.visible_agent_ids)
        relevant: Dict[str, PerceivedIdentity] = {
            aid: presence.identity for aid, presence in spatial.visible_agents.items()
        }
        for msg in inbox:
            sid = msg.sender_id
            # Senders without cognition (e.g. an errand runner reporting back) get no relation:
            # there's no one on their end, and it would put them in ``_message_roster``'s
            # contactable list as an undeliverable recipient.
            # Test the sender's own ``sender_is_agent`` stamp; don't add a second id list: one it
            # misses would persist a relation with an "unknown source".
            if not getattr(msg, "sender_is_agent", True):
                continue
            if not sid or sid in relevant_ids:
                continue
            relevant_ids.append(sid)
            # Senders carry only a name; a gender seen earlier comes from relation.to_gender.
            relevant.setdefault(sid, PerceivedIdentity(
                name=getattr(msg, "sender_name", "") or "某人",
                gender=(known.gender if (known := self._known_agents.get(sid)) else ""),
            ))
        # Related-but-absent people (top-N by salience) still weigh on cognition, and SEND_MESSAGE
        # must be able to target them.
        for rel in await self.relation_system.significant_relations(limit=_KNOWN_RELATION_LIMIT):
            tid = rel.to_id
            if not tid or tid == self.agent_id or tid in relevant_ids:
                continue
            relevant_ids.append(tid)
            relevant.setdefault(tid, PerceivedIdentity(
                name=rel.to_name or "某人", gender=rel.to_gender,
            ))
        memory_biases = {
            aid: self.memory_system.related_memory_bias(target_agent_id=aid, current_step=current_step)
            for aid in relevant_ids
        }
        perceived = await self.relation_system.perceive_many(
            relevant_ids,
            emotion=self.personality.state.emotion,
            memory_biases=memory_biases,
            identities=relevant,
        )
        # Relations to the dead stay as remembered bonds but are flagged deceased: downstream drops
        # them from contact candidates and marks them "（已死亡）". Transient, derived per step.
        if self._dead_agent_ids:
            for rel in perceived:
                if rel.target_agent_id in self._dead_agent_ids:
                    rel.deceased = True
        return perceived

    def _recent_factual_lines(self, step: int, *, top_k: int, before_step: bool = False) -> list[str]:
        """Recent factual memories as chronological lines; the one fetch-and-render shared by the
        cognition prompts.

        ``before_step`` drops this step's entries (already in the signal block) and those lacking a
        factual; 8 extra are fetched so top_k survives the filter.
        """
        fetched = self.memory_system.sample_recent_events(step, top_k=top_k + 8 if before_step else top_k)
        if before_step:
            fetched = [pair for pair in fetched if pair[0] is not None and pair[0].created_step < step][:top_k]
        return [
            text
            for fact, _exp in order_memories_chrono(fetched, key=lambda pair: pair[0] or pair[1])
            if (text := self._render_recent_event(fact, None, current_step=step))
        ]

    def _render_recent_event(
        self, fact: "Memory | None", exp: "Memory | None", *, current_step: int
    ) -> str:
        """Render one recent event as the factual plus "my reading" (experiential) in parentheses.

        Fact plus reading gives an objective anchor and a subjective lean, preventing goal
        repetition/drift. The experiential renders alone only when the factual is missing (before
        a cold restart warms up).
        """
        if fact is not None:
            line = render_memory(
                fact, now_step=current_step, seconds_per_step=self.seconds_per_step,
                world_start_second_of_day=self.world_start_second_of_day)
            reading = (exp.stored_content or "").strip() if exp is not None else ""
            return f"{line}（我当时的主观理解与感受：{reading}）" if reading else line
        if exp is not None:
            return render_memory(
                exp, now_step=current_step, seconds_per_step=self.seconds_per_step,
                world_start_second_of_day=self.world_start_second_of_day)
        return ""

    async def _build_internal_context(
        self,
        *,
        spatial: SpatialPerception,
        inbox: List[Message],
        broadcasts: List[Broadcast],
        current_step: int,
        consumed_external_goals: List[ExternalGoal] | None = None,
    ) -> InternalContext:
        retrieval_query = self._build_retrieval_query(
            spatial=spatial,
            inbox=inbox,
            broadcasts=broadcasts,
        )
        # Recent first-hand memories (in-memory scan) for goal generation, kept as their own prompt
        # section rather than mixed into the retrieval query. Both streams: feelings alone make
        # short-term goals repeat/drift. sample_recent_events returns importance order, so sort
        # chronologically for MEMORY_ORDER_HINT; always via order_memories_chrono.
        recent_events = order_memories_chrono(
            self.memory_system.sample_recent_events(current_step, top_k=10),
            key=lambda pair: pair[0] or pair[1],
        )
        recent_memory_texts = [
            text
            for fact, exp in recent_events
            if (text := self._render_recent_event(fact, exp, current_step=current_step))
        ]
        # Recent attempts that didn't go through (transient, never persisted): fetched once, read by
        # motivation (rewrite blocked intents) and decide (don't hit the same wall).
        recent_foiled = self.memory_system.recent_foiled_attempts(current_step)
        logger.debug(
            "agent_retrieval_context",
            extra={
                "agent_id": self.agent_id,
                "step": current_step,
                "spatial_context": retrieval_query.spatial_context[:120],
                "message_context": retrieval_query.message_context[:120],
                "broadcast_context": retrieval_query.broadcast_context[:120],
                "factual_query": retrieval_query.as_primary()[:160],
                "recent_memory_count": len(recent_memory_texts),
            },
        )
        perceived_signal_texts = render_perceived_signals(
            spatial=spatial, inbox=inbox, broadcasts=broadcasts
        )
        # Read-only: the runtime rewrites ``pending_external_goals`` every step. Don't clear it
        # here: agents the cadence gate skips would never clear and would keep a stale pressure.
        external_goals_raw = consumed_external_goals if consumed_external_goals is not None else list(self.pending_external_goals)
        # Before the perception-emotion appraisal, which reads them.
        perceived_relations = await self._perceive_relevant_relations(
            spatial, inbox, current_step=current_step
        )
        # Instinctive emotion + per-need activation, not persisted. need_activation is the
        # authoritative situational signal for need scoring; empty → needs compete on disposition
        # (I×W) + structured runtime_adjustments only.
        with observe_stage(Stage.PERCEPTION, agent_id=self.agent_id):
            appraisal = await self._assess_perception_emotion(
                spatial, inbox, broadcasts, external_goals_raw, perceived_relations,
            )
        if appraisal.emotion is None:
            # Benign: no signal or no emotion. LLM failures log "perception_emotion_llm_failed".
            logger.debug(
                "perception_emotion_none",
                extra={"agent_id": self.agent_id, "step": current_step},
            )
        perception_emotion = appraisal.emotion or self.personality.state.emotion
        with observe_stage(Stage.MOTIVATION, agent_id=self.agent_id):
            need_evaluation = await self.need_engine.run(
                current_step=current_step,
                personality=self.personality,
                visible_agents=spatial.visible_agent_ids,
                pending_messages=len(inbox),
                emotion=perception_emotion,
                need_relevance=appraisal.need_activation,
                world_time_hour=spatial.world_time_hour,
                situation=Situation.from_spatial(spatial),
                perceived_signal_texts=perceived_signal_texts,
                recent_memory_texts=recent_memory_texts,
                recent_foiled_texts=recent_foiled,
                external_goals=external_goals_raw,
                perceived_relations=perceived_relations,
            )
        logger.debug(
            "agent_motivation",
            extra={
                "agent_id": self.agent_id,
                "step": current_step,
                "dominant_need": (
                    need_evaluation.dominant_need.value
                    if need_evaluation.dominant_need is not None else None
                ),
                "active_need_count": len(need_evaluation.active_needs),
                "short_term_goals": len(need_evaluation.short_term_goals),
                "long_term_goals": len(need_evaluation.long_term_goals),
            },
        )
        retrieval_result = await self.memory_system.retrieve_both(
            retrieval_query,
            current_step=current_step,
            top_k_each=_PERCEPTION_MEMORY_TOP_K,
            before_step=current_step,  # this step's perceptions are already in the PerceptionPacket
        )
        factual_memories = [m for m in retrieval_result.events if m.stream == MemoryStream.FACTUAL]
        experiential_memories = [m for m in retrieval_result.events if m.stream == MemoryStream.EXPERIENTIAL]
        # Add recent anchors recall missed (see _supplement_recent), reusing recent_events.
        factual_memories = _supplement_recent(
            factual_memories,
            (fact for fact, _exp in recent_events),
            limit=_DECIDE_RECENT_FACTUAL_SUPPLEMENT_K,
        )
        experiential_memories = _supplement_recent(
            experiential_memories,
            (exp for _fact, exp in recent_events),
            limit=_DECIDE_RECENT_EXPERIENTIAL_SUPPLEMENT_K,
        )
        return InternalContext(
            emotion=perception_emotion,
            dominant_need=need_evaluation.dominant_need,
            active_needs=need_evaluation.active_needs,
            short_term_goals=list(need_evaluation.short_term_goals),
            long_term_goals=list(need_evaluation.long_term_goals),
            factual_memories=factual_memories,
            experiential_memories=experiential_memories,
            relevant_relations=perceived_relations,
            need_evaluation=need_evaluation,
            insights=list(retrieval_result.insights),
            period_summaries=list(retrieval_result.period_summaries),
            insight_sources=dict(retrieval_result.insight_sources),
            # Never persisted, so absent from recall; decide needs it to change tactics.
            recent_foiled_attempts=recent_foiled,
        )

    def apply_vitality_damage(self, delta: float, *, step: int, death_cause: str | None) -> None:
        """The single entry point for vitality changes (positive = damage, negative = recovery).

        Don't call ``personality.apply_vitality_damage`` alone: vitality would hit zero without
        ``is_active`` flipping, leaving a corpse that keeps acting."""
        self.personality.apply_vitality_damage(delta)
        if not self.personality.is_alive:
            self._trigger_death(step, cause=death_cause)

    def _trigger_death(self, step: int, cause: str | None = None) -> None:
        """Agent-internal death: only flip the active flag and record the cause phrase.

        Never write an "I died" memory: there is no experiencing subject after death. World-side
        handling belongs to DeathHandler.process_new_deaths."""
        self.set_active(False)
        self.death_cause = cause
        logger.info("agent_died", extra={"agent_id": self.agent_id, "step": step})

    async def _appraise_outcome(
        self, result: ActionResult, dominant_need: NeedType | None
    ) -> "EmotionState | None":
        """How the outcome feels (in-character LLM for every agent), kept out of _apply_feedback,
        which only applies it.

        Returns None on failure (fallback tier-1): the current mood stays rather than an emotion
        synthesized from ``succeeded`` that the character never felt.
        """
        with observe_stage(Stage.FEEDBACK, agent_id=self.agent_id):
            return await self._llm_emotion(result, dominant_need)

    @staticmethod
    def _memory_importance_override(result: ActionResult) -> "MemoryImportance | float | None":
        """Importance for own actions whose weight a structured signal fixes, skipping the
        importance LLM; None → record_event scores as usual.

        - MOVE/REST: reliably mundane → LOW (also makes _should_write_experiential skip the rewrite).
        - SEND_MESSAGE: the sender's urgency, via the shared URGENCY_TO_STRENGTH table rather than
          a new mapping."""

        action = result.action
        if action.action_type in (ActionType.MOVE, ActionType.REST):
            return MemoryImportance.LOW
        if action.action_type == ActionType.SEND_MESSAGE:
            return URGENCY_TO_STRENGTH[action.urgency]
        return None

    async def _apply_feedback(
        self,
        *,
        result: ActionResult,
        dominant_need: NeedType | None,
        emotion: EmotionState | None,
    ) -> None:
        """Write the already-appraised emotion (from _appraise_outcome) and the action's declared
        effects into agent state.

        ``emotion=None`` (appraisal failed) keeps the current mood; need/relation/memory still
        land. Memory importance comes from record_event's ImportanceEvaluator, a separate
        functional LLM call."""
        logger.debug(
            "agent_feedback_applied",
            extra={
                "agent_id": self.agent_id,
                "step": result.action.step,
                "action_type": result.action.action_type.value,
                "succeeded": result.succeeded,
                "emotion": emotion.primary.value if emotion is not None else None,
                "intensity": round(emotion.intensity, 3) if emotion is not None else None,
                "valence": round(emotion.valence, 3) if emotion is not None else None,
                "dominant_need": dominant_need.value if dominant_need is not None else None,
                "relation_updates": len(result.relation_updates),
            },
        )
        self.personality.complete_action(
            step=result.action.step,
            description=result.action.action_description,
            result_summary=result.outcome,
            succeeded=result.succeeded,
            emotion=emotion,
        )
        # personality.state.need_intensities is the only source of truth for need intensity.
        new_intensities = self.need_engine.evolve_intensities(
            self.personality.state.need_intensities,
            dominant_need=dominant_need,
            succeeded=result.succeeded,
        )
        self.personality.update_need_intensities(new_intensities)

        # Keep this serial: apply_interaction is a store read-modify-write, so concurrent updates
        # to the same target would be lost; with at most one entry today, parallelism gains nothing.
        for _tid, trust_d, affection_d in result.relation_updates:
            await self.relation_system.apply_interaction(
                _tid,
                trust_delta=trust_d,
                affection_delta=affection_d,
                step=result.action.step,
            )

        # raw_memory must be first-person. Never fall back to the third-person outcome: it would put
        # a named third-person line into "my" memory, recalled forever via embedding.
        raw_memory = result.factual_memory or f"我{result.action.action_description}"
        if result.not_executed:
            # A non-event (preconditions failed): never persist it; it would dilute retrieval and
            # feed relation_evolution/reflection a relation that never happened. Keep only a
            # transient trace so decide can change tactics after repeated failure.
            self._note_foiled(result.action, step=result.action.step, text=raw_memory)
        else:
            # The intent landed, so the foiled trace ends; an in-world failure still counts as
            # having happened.
            self._clear_foiled(result.action)
            # Display names keep the experiential rewrite from inventing who was involved.
            related_names, relation_ctx = await self._event_people_context(
                result.action.target.acted_on_agents
            )
            dominant_need_label = (
                f"{dominant_need.value}（{self.personality.need_label(dominant_need)}）"
                if dominant_need is not None else ""
            )
            # The same event feels different at court and in a study.
            await self.memory_system.record_event(
                current_step=result.action.step,
                raw_content=raw_memory,
                experiential_content=raw_memory,
                personality=self.personality,
                related_agents=related_names,
                related_relations_text=relation_ctx,
                dominant_need_label=dominant_need_label,
                triggered_by=result.action.action_type.value,
                importance=self._memory_importance_override(result),
                situation=self._situation,
            )

        if result.vitality_damage != 0.0:
            # The cause phrase must be a predicate that reads after the name (like "died of his
            # wounds"), not a whole outcome sentence.
            self.apply_vitality_damage(
                result.vitality_damage, step=result.action.step, death_cause="生命耗尽。"
            )

        # The actor's own ongoing condition: same three states and set-over-clear precedence as
        # apply_target_effect. This is the only place a bound agent who breaks free can clear it.
        if result.actor_condition_set is not None:
            self.personality.set_condition(result.actor_condition_set)
        elif result.actor_condition_cleared:
            self.personality.clear_condition()

    async def _record_completion(
        self, *, result: ActionResult, dominant_need: NeedType | None, step: int,
    ) -> None:
        """The single feedback sink for every action, reached only via finalize_ongoing_action:
        appraise → apply → judge goal progress. The runtime persists the result at step end.

        Logs exactly one "agent_feedback_recorded" line per feedback, so "decisions == feedbacks"
        is auditable.
        """
        logger.info(
            "agent_feedback_recorded",
            extra={
                "agent_id": self.agent_id,
                "step": step,
                "action_type": result.action.action_type.value,
                "succeeded": result.succeeded,
                "adjudication_failed": result.adjudication_failed,
            },
        )
        if result.adjudication_failed:
            # Infra failure, not an in-world outcome → null step: skip all writeback, only reset to
            # idle. A genuine in-world failure has adjudication_failed=False and is remembered.
            self.personality.update_action_result(
                step=step,
                action=result.action.action_description,
                result="",
                succeeded=False,
            )
            return
        # appraise → apply_feedback and the goal judge run concurrently (the judge only reads).
        # Goals are written back only after the join: apply_feedback's importance prompt reads
        # goals, and an early write-back would make it see pre- or post-action goals unpredictably.
        # A bare gather would cancel an in-flight apply_feedback → half-applied personality.
        # not_executed skips the judge: nothing happened, and it would be tempted to invent progress.
        async def _appraise_and_apply() -> None:
            emotion = await self._appraise_outcome(result, dominant_need)
            await self._apply_feedback(
                result=result, dominant_need=dominant_need, emotion=emotion
            )

        slots: list[tuple[str, Awaitable[Any]]] = [("apply_feedback", _appraise_and_apply())]
        if not result.not_executed:
            slots.append(("judge_goals", self._judge_goal_progress(
                step=step, result=result
            )))
        outcomes = await asyncio.gather(
            *(coro for _, coro in slots), return_exceptions=True
        )
        judged: List[GoalEntity] | None = None
        for (slot, _coro), outcome in zip(slots, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "record_completion_slot_failed",
                    extra={
                        "agent_id": self.agent_id,
                        "step": step,
                        "slot": slot,
                        "error": str(outcome),
                    },
                )
            elif slot == "judge_goals":
                judged = outcome
        if judged is not None:
            try:
                await self._land_goal_progress(judged, step=step)
            except Exception as exc:  # noqa: BLE001 — Rule 1: a failed goal write-back must not kill the run
                logger.warning(
                    "record_completion_slot_failed",
                    extra={
                        "agent_id": self.agent_id,
                        "step": step,
                        "slot": "land_goals",
                        "error": str(exc),
                    },
                )

    async def revise_long_term_goals(self, *, step: int) -> None:
        """Re-examine long-term goals from the agent's own recent memories (both streams) and sync
        the result to personality. Skips with no recent memories: no change for its own sake.

        NeedEngine doesn't touch personality and MemorySystem doesn't enter NeedEngine; Agent
        assembles the memory text and owns need → personality consistency.
        """
        recent = await self.memory_system.sample_recent(step, lookback=30, top_k=15)
        if not recent:
            return
        updated = await self.need_engine.revise_long_term_goals(
            personality=self.personality,
            recent_memory_texts=render_memory_lines(
                recent, now_step=step, seconds_per_step=self.seconds_per_step,
                world_start_second_of_day=self.world_start_second_of_day,
            ),
            is_main_character=self.is_main_character,
            situation=self._situation,
        )
        # Record what was actually adopted: otherwise an audit reads goals the engine discarded
        # (None = unchanged/identical/failed) as "he changed direction".
        annotate_active_call(long_term_goals_new=updated or [])
        if updated is None:
            return
        # personality keeps state of existing entities whose text matches.
        self.personality.set_long_term_goals(updated)

    def _resolve_dominant_need_from_state(self) -> NeedType | None:
        """The persisted dominant need as a NeedType, for multi-step finalize where the
        originating plan is long gone."""
        raw = self.personality.state.dominant_need
        if raw is None:
            return None
        try:
            return NeedType(raw)
        except ValueError:
            return None

    async def _llm_emotion(
        self,
        result: ActionResult,
        dominant_need: NeedType | None,
    ) -> "EmotionState | None":
        """In-character first-person emotion for an outcome, for every agent (main characters
        retry once). None on failure: the caller keeps the current mood.
        """
        llm = self.llm_router
        # The agent's own build-time need label, not a generic Maslow entry that would flatten
        # every agent's motivation into the same line.
        need_label = (
            f"{dominant_need.value}（{self.personality.need_label(dominant_need)}）"
            if dominant_need is not None else "无"
        )
        emotion_types = EmotionType.prompt_list()
        succeeded_str = "成功" if result.succeeded else "失败"
        expected_part = (
            f"\n我原本期望：{result.expected_outcome}" if result.expected_outcome else ""
        )
        # The prior mood gets its own block marked "backdrop, not the answer", so the persona uses
        # include_emotion=False.
        cur_emotion = self.personality.state.emotion
        mood_block = (
            f"\n\n【我此刻的心境】（这是我做这件事之前就有的既有情绪，是底色而非答案）\n我此前感到{cur_emotion.summary()}。"
            if cur_emotion.intensity > 0.3 else ""
        )
        # The same setback has a different tone at court and on a battlefield.
        _header = render_situation_header(self._situation, voice=SituationVoice.FIRST)
        # Failing feels different when bound; same block as the time-place anchor, as in decision.
        _condition = condition_line(
            self.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=result.action.step, seconds_per_step=self.seconds_per_step, lead="",
        )
        _situation_lines = [line for line in (_header, _condition) if line]
        situation_block = "\n".join(_situation_lines) + "\n\n" if _situation_lines else ""
        # Backdrop (see _FEEDBACK_EMOTION_RECENT_FACTUAL_K). The current action isn't in memory yet
        # (record_event runs later in _apply_feedback), so it can't appear twice.
        recent_factuals = self._recent_factual_lines(
            result.action.step, top_k=_FEEDBACK_EMOTION_RECENT_FACTUAL_K
        )
        recent_block = (
            f"\n【我近来的经历（客观发生；{MEMORY_ORDER_HINT}）——只作情绪的连续底色，"
            "我的情绪仍只就下面那一件事而来，别对这些旧事重新生情】\n"
            + "\n".join(f"- {t}" for t in recent_factuals)
            + "\n"
            if recent_factuals else ""
        )
        # The same outcome satisfies against a rival and shames against kin.
        involved_relation_ctx = await self._resolve_relation_context(
            result.action.target.acted_on_agents
        )
        relation_block = (
            f"\n【我与这件事涉及之人的关系】\n{involved_relation_ctx}\n"
            if involved_relation_ctx else ""
        )
        # Prefix-cache split: system is byte-identical across calls (emotion_types is a constant
        # enum); user carries this call's situation/persona/experience/need.

        system_prompt = f"""\
你此刻完全代入一个角色，以第一人称「我」感受一件刚发生的事在此刻心里激起的情绪。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【任务】
就这件事此刻在我心里激起的情绪——这是结果落定那一下的真实感受，不是事后权衡。

【字段定义】
  - {EMOTION_INTENSITY_DEFINITION}
  - {EMOTION_VALENCE_DEFINITION}

【强制约束，必须遵守】
 1.以我的性格去感受这件事，不要套用"成功就该高兴、失败就该沮丧"的模板——同一个结果，不同的人会有不同的滋味。
 2.既有心境只是底色：这件事可能强化它、扭转它、或几乎不动它；不要照搬既有情绪当输出。
 3.强度与这件事对我的分量相称，不要为了有戏剧性而夸大，也不要把无关紧要的小事说得惊心动魄。
 4.禁止揣测没有发生过的客观事实。
 5.我需要结合【我刚刚的经历】的所有内容综合判断，不要仅仅依靠单一局部内容，不要捏造和揣测客观事实。
 6.先在 reason 里把"我是被这件事的哪一点触动、为什么是这种情绪"点清楚，再据此给后面三项——情绪由这个触点推出，不是先挑个戏剧化的情绪再补理由。
 7.严格输出以下 JSON，不要任何多余内容（reason 必须排在最前）：
{{"reason": "<第一人称，我是被这件事的哪一点触动、是否与我预期有落差、为什么会是这种情绪，先点清再给下面三项，20-40字>", "emotion": "从以下选一：{emotion_types}", "intensity": 0.0~1.0, "valence": -1.0~1.0}}"""
        user_prompt = f"""\
{situation_block}【我是谁】（我固有的身份、性格、价值观；这是我这个人）
{self.personality.to_prompt_context(include_goals=False, include_emotion=False)}{mood_block}
{recent_block}{relation_block}
【我刚刚的经历】（此刻就这一件事，我的情绪只就它而来）
我打算做：{result.action.action_description}
结果：{succeeded_str}
实际发生：{result.factual_memory}{expected_part}
我当时最迫切的需求：{need_label}

我依上面说定的规矩与 JSON 格式说出此刻的情绪与落差，只输出 JSON、不写任何多余内容。"""
        scene = LLMScene.AGENT_DECISION_MAIN
        try:
            llm_start = time.perf_counter()
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            # reason ≤40 chars (often ~50 in practice) ≈ 75 tok + emotion enum + 3 numbers + JSON
            # overhead ≈ 115 tok. reason comes first, so a tight budget truncates the conclusion fields.
            # Trace extra mirrors the "【我刚刚的经历】" lines so audits needn't regex the prompt.
            feedback_extra = {
                "given_facts": GivenFacts()
                .add("此刻何时何地", _header)
                .add("我打算做", result.action.action_description)
                .add("结果", succeeded_str)          # same word as in the prompt above
                .add("实际发生", result.factual_memory),
                # Same key as the perception beat's: steps without perception rely on it for location.
                "location": situation_location(self._situation),
                "feedback_action_desc": result.action.action_description,
                "feedback_action_success": bool(result.succeeded),
                "feedback_action_actual": result.factual_memory,
            }
            if self.is_main_character:
                with annotate_call(**feedback_extra):
                    response = await llm.complete_with_retry(
                        scene, messages, temperature=0.7, max_tokens=output_budget(115), json_mode=True,
                    )
            else:
                with annotate_call(**feedback_extra):
                    response = await llm.complete(
                        scene, messages, temperature=0.7, max_tokens=output_budget(115), json_mode=True,
                    )
            logger.info(
                "llm_call",
                extra={
                    "scene": scene.value,
                    "agent_id": self.agent_id,
                    "elapsed_ms": round((time.perf_counter() - llm_start) * 1000.0, 2),
                    "purpose": "feedback_emotion",
                },
            )
            data = extract_json(response.content)
            emotion = emotion_from_payload(data, triggered_by=result.factual_memory)
            if emotion is None:
                # Valid JSON but no emotion → discard; note it, or the trace would look clean.
                note_active_call_adoption(False, "missing_emotion")
                return None
            note_active_call_adoption(True)
            return emotion
        except Exception as exc:
            logger.warning(
                "feedback_emotion_llm_failed",
                extra={"agent_id": self.agent_id, "error": str(exc)},
            )
            # No-op if the call itself failed; matters for valid JSON with a wrong field type.
            note_active_call_adoption(False, "unusable_payload")
            return None

    async def evaluate_interrupt(
        self,
        *,
        step: int,
        reason: str,
        current_action_desc: str,
        intent: str,
        progress_hint: str,
    ) -> tuple[bool, str]:
        """Decide in character whether to interrupt the current ongoing action.

        Returns (should_interrupt, thought); ``thought`` is the first-person deliberation the
        verdict falls out of, and it colors the interrupt outcome narrative. ``progress_hint`` is
        narrative text, never a step count.

        ``intent`` (``participant_intent``; empty when conscripted) shows the action's aim: without
        it, someone already heading somewhere would abandon the trip on "come here at once".

        Persona uses ``include_goals=False``: the goal queue is already embodied by
        ``current_action_desc``, and here it would only expose a stale FIFO. Recent factual memory
        is injected instead: whether the event escalates or is noise turns on what just happened.
        """
        llm = self.llm_router
        persona = self.personality.to_prompt_context(include_goals=False, include_emotion=True)
        recent_factuals = self._recent_factual_lines(step, top_k=_INTERRUPT_RECENT_FACTUAL_K)
        recent_block = (
            f"\n【我近来的经历（客观发生；{MEMORY_ORDER_HINT}）】\n"
            + "\n".join(f"- {t}" for t in recent_factuals)
            + "\n"
            if recent_factuals else ""
        )
        # A study late at night and a procession at court weigh differently.
        _header = render_situation_header(self._situation, voice=SituationVoice.FIRST)
        # Without the condition, someone tied up doesn't know he can't move, and his monologue
        # (persisted via format_interrupt_reason_3p) would fabricate facts about himself.
        _condition = condition_line(
            self.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=step, seconds_per_step=self.seconds_per_step, lead="",
        )
        _situation_lines = [line for line in (_header, _condition) if line]
        situation_block = "\n".join(_situation_lines) + "\n\n" if _situation_lines else ""
        intent_line = f"\n{intent}" if intent else ""
        # Prefix-cache split: system is invariant; user carries persona/current action/sudden event.
        system_prompt = """\
你此刻完全代入一个角色，以第一人称「我」权衡「要不要立刻放下手头的事」。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【现在，以"我"的视角】
先在心里把这件事过一遍：它和我此刻在做的、我在意的、我的目标比，够不够分量让我立刻放下手头的事？想清楚了再决定。
「放下」是指手头这件事到此作罢：已花的功夫白费，我下一拍从头另作打算。所以：
- 手头这件事本来就通向那件突然发生的事所要的，我不放下，接着做完就是
- 只有它非得让位、或者已经没有意义了，才值得放下
我不会这样做：
- 不会一惊一乍、为了戏剧性就丢下手头的事；也不会麻木地无视真正要紧的事
- 不会跳出"我"去旁观评估，就是我自己在权衡
- 不会把内心独白写成一个情绪标签（如"我很愤怒"），而要写出我具体在掂量什么
- 不会捏造不存在的客观事实。
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON，不要任何多余内容（thought 在前，先想后判）：
{"thought": "我此刻的权衡与决定，一两句，≤60字", "interrupt": true或false}
"""
        user_prompt = f"""\
{situation_block}【我是谁】
{persona}
{recent_block}
【我正在做的事】
「{current_action_desc}」，{progress_hint}{intent_line}

【突然发生了】
{reason}

我依上面说定的规矩与 JSON 格式给出这一拍的权衡与决定，只输出 JSON、不写任何多余内容。
"""
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        try:
            with observe_stage(Stage.INTERRUPT, agent_id=self.agent_id), annotate_call(
                interrupt_doing=current_action_desc,
                interrupt_trigger=reason,
            ):
                if self.is_main_character:
                    resp = await llm.complete_with_retry(
                        LLMScene.AGENT_INTERRUPT_DECISION,
                        messages,
                        temperature=0.7,
                        max_tokens=output_budget(105),  # thought ≤60 chars ≈90 + bool + JSON ≈ 105 tok
                    json_mode=True,
                    )
                else:
                    resp = await llm.complete(
                        LLMScene.AGENT_INTERRUPT_DECISION,
                        messages,
                        temperature=0.7,
                        max_tokens=output_budget(105),  # thought ≤60 chars ≈90 + bool + JSON ≈ 105 tok
                    json_mode=True,
                    )
            data = extract_json(resp.content)
            # Don't slice: the thought is quoted in the persisted outcome
            # (format_interrupt_reason_3p), and a cut would leave a broken clause forever. The
            # prompt's ≤60 chars owns the length.
            return coerce_bool(data.get("interrupt"), False), str(data.get("thought", ""))
        except Exception as exc:
            # Don't interrupt: interrupting is destructive, while the signal is already in memory
            # and pressure stays in pending_external_goals for the next re-plan.
            logger.warning(
                "interrupt_llm_failed",
                extra={"agent_id": self.agent_id, "step": step, "error": repr(exc)},
            )
            return False, ""

    def _build_retrieval_query(
        self,
        *,
        spatial: SpatialPerception,
        inbox: List[Message],
        broadcasts: List[Broadcast],
    ) -> RetrievalQuery:
        # The query is embedded against memory prose, so it must be natural text: names and
        # content, no ids, `key:` tags, timestamps or machine separators like " | ". Own goals stay
        # out (see RetrievalQuery).
        spatial_parts: List[str] = [spatial.location_view.name]
        spatial_parts.extend(
            p.identity.name for aid in spatial.visible_agent_ids
            if (p := spatial.visible_agents.get(aid)) and p.identity.name
        )
        spatial_parts.extend(e.name for e in spatial.visible_entities)
        spatial_parts.extend(e.content for e in spatial.ambient_events)

        message_context = "，".join(m.content for m in inbox) if inbox else ""
        broadcast_context = "，".join(b.content for b in broadcasts) if broadcasts else ""

        return RetrievalQuery(
            spatial_context="，".join(p for p in spatial_parts if p),
            message_context=message_context,
            broadcast_context=broadcast_context,
        )

    async def begin_ongoing_step(
        self, *, plan: AgentStepPlan, estimated_steps: int | None = None
    ) -> None:
        """Begin half of the begin→finalize path: mark the action started, writing no memory.

        ``estimated_steps`` overrides the LLM's estimate with the executor-authoritative duration
        (e.g. MOVE distance).
        """
        self.personality.update_location(step=plan.step, location=plan.spatial.location_id)
        # Don't write need_intensities here: active_needs[].intensity is this step's salience
        # score, not the stored intensity, and would hand the survival check values in the wrong unit.
        self.personality.apply_need_state(
            step=plan.step,
            active_needs=[need.type.value for need in plan.need_evaluation.active_needs],
            dominant_need=(
                plan.need_evaluation.dominant_need.value
                if plan.need_evaluation.dominant_need is not None
                else None
            ),
            long_term_goals=plan.need_evaluation.long_term_goals,
            short_term_goal_entities=list(plan.need_evaluation.short_term_goal_entities),
        )
        self.personality.begin_action(
            step=plan.step,
            description=plan.action.action_description,
            activity_status=activity_status_for(plan.action.action_type),
            target=plan.action.target.single_acted_on_agent,
            estimated_steps=estimated_steps if estimated_steps is not None else plan.action.estimated_steps,
        )

    async def join_ongoing_action(
        self,
        *,
        description: str,
        target_id: str,
        activity_status: AgentActivityStatus,
        estimated_steps: int,
        step: int,
    ) -> None:
        """Passively join a multi-step action initiated by another agent (SocialExecutor's
        second participant).
        """
        # No apply_need_state: being conscripted discards this step's own goals, so he doesn't
        # bring a separate agenda into the conversation; the cost is one wasted goal generation.
        self.personality.begin_action(
            step=step,
            description=description,
            activity_status=activity_status,
            target=target_id,
            estimated_steps=estimated_steps,
        )

    def _note_foiled(self, action: "AgentAction", *, step: int, text: str) -> None:
        """The single entry point when an intent fails to land: a transient trace for decide and a
        count for arbitration, sharing ``_foiled_key`` so both agree on "same matter".

        Covers unmet preconditions and decisions discarded by conscription; either way, going
        earlier helps him initiate instead of being absorbed.
        """
        key = _foiled_key(action)
        self.memory_system.note_foiled_attempt(
            step=step, text=text, key=key, gist=action.action_description,
        )
        if not _foiled_key_has_object(key):
            return          # object-less actions (WORK/REST) contend for no body; the initiative boost is meaningless
        self._foiled_misses[key] = (self._foiled_misses.get(key, (0, 0))[0] + 1, step)

    def _clear_foiled(self, action: "AgentAction") -> None:
        """The matter actually happened: clear both the arbitration count and decide's trace.

        The only clearing criterion. Being invited into the same conversation counts (same key);
        "he did something else" doesn't, being merely an absence of evidence. Missing the trace
        half would leave an accomplished matter in decide as "I meant to … but couldn't". The trace
        is cleared even for object-less keys (``clear_foiled_attempts`` matches like merging).
        """
        key = _foiled_key(action)
        if _foiled_key_has_object(key):
            self._foiled_misses.pop(key, None)
        self.memory_system.clear_foiled_attempts(key, action.action_description)

    def _foiled_misses_for(self, action: "AgentAction | None", step: int) -> int:
        """How many recent beats this action's matter failed to land; also prunes stale keys
        (called once per step at plan time).
        """

        for key, (_count, last_step) in list(self._foiled_misses.items()):
            if step - last_step > _FOILED_MISS_AGE_STEPS:
                del self._foiled_misses[key]
        if action is None:
            return 0
        key = _foiled_key(action)
        if not _foiled_key_has_object(key):
            return 0
        return self._foiled_misses.get(key, (0, 0))[0]

    def defer_decided_intent(self, action: "AgentAction", step: int) -> None:
        """Keep the intent of an agent who decided this step but was drawn into someone else's
        action; the signals behind it lived one step, so otherwise it is lost for good.

        Goes into the foiled buffer (same kind of thing as ``not_executed``), not the short-term
        goal queue: the queue would hold a concrete action, which the goal prompt forbids, and
        decide would replay it as a debt again and again.

        Accepted cost: the buffer's window (_FOILED_LOOKBACK_STEPS) isn't snapshotted, so an intent
        not picked up within it disappears; its driving signal only lived one step anyway.
        """
        text = (action.action_description or "").strip()
        if not text:
            return
        # No "我本想" ("I meant to") prefix: action_description is already first-person.
        self._note_foiled(
            action, step=step, text=f"{text}——却被卷入他人的事，这一行动没能做成。",
        )

    async def apply_target_effect(
        self,
        effect: TargetAgentEffect,
        from_agent_id: str | None,
        step: int,
        situation: Situation | None = None,
    ) -> None:
        """Apply inbound consequences from this agent's (B's) perspective.

        B is often the more affected party, so B gets a full dual-stream memory. The emotion is
        applied before record_event so B's experiential rewrite is colored by it.

        ``situation=None`` stamps the memory with the last perceived situation, right for executor
        feedback. The director's channel lands before perception, so it passes the correct one for
        this write only rather than writing it into the agent (``_situation`` is set only by
        ``perceive_step`` and ``refresh_situation``).

        ``effect.overheard``: content B merely heard; its importance is judged rather than
        asserted HIGH.

        ``from_agent_id=None``: an impersonal force (e.g. the director); no actor name or relation,
        but B is still hurt and still remembers. Director wounds must go through here too: a
        private vitality write would skip the memory, giving one event two sets of physics.
        """
        if effect.emotion_type is not None:
            self.personality.update_emotion(
                primary=effect.emotion_type,
                intensity=effect.emotion_intensity,
                valence=effect.emotion_valence,
                triggered_by=effect.factual_memory,
            )
        # An empty fact is a producer that failed (Rule 1 tier-1): record nothing, or the
        # experiential rewrite would invent what happened from an empty event.
        if effect.factual_memory:
            # Display names keep the experiential rewrite from inventing the actor.
            actors = [from_agent_id] if from_agent_id else []
            related_names, relation_ctx = await self._event_people_context(actors)
            await self.memory_system.record_event(
                current_step=step,
                raw_content=effect.factual_memory,
                experiential_content=effect.factual_memory,
                personality=self.personality,
                related_agents=related_names,
                related_relations_text=relation_ctx,
                triggered_by="target_effect",
                importance=None if effect.overheard else MemoryImportance.HIGH,
                situation=situation if situation is not None else self._situation,
            )
        if effect.relation_toward_actor is not None:
            actor_id, trust_d, affection_d = effect.relation_toward_actor
            await self.relation_system.apply_interaction(
                actor_id,
                trust_delta=trust_d,
                affection_delta=affection_d,
                step=step,
            )
        # != 0, not > 0: negative heals, and > 0 would silently drop every recovery.
        if effect.vitality_damage != 0:
            self.apply_vitality_damage(
                effect.vitality_damage, step=step, death_cause=effect.death_cause
            )
        # Set wins over clear: both at once means the adjudication contradicted itself, and "what
        # state he's in now" is the safer reading.
        if effect.condition_set is not None:
            self.personality.set_condition(effect.condition_set)
        elif effect.condition_cleared:
            self.personality.clear_condition()

    def expire_condition(self, step: int) -> bool:
        """Clear a self-limiting condition once it expires (the drug wears off); returns whether
        it was cleared.

        Must run before the step's spatial perception is assembled, or someone who woke up still
        looks restrained. ``until_step is None`` never expires: ropes don't untie themselves, and
        a timeout would let an unattended prisoner walk free.
        """
        condition = self.personality.state.condition
        if condition is None or condition.until_step is None:
            return False
        if step < condition.until_step:
            return False
        self.personality.clear_condition()
        return True

    async def _assess_perception_emotion(
        self,
        spatial: SpatialPerception,
        inbox: List[Message],
        broadcasts: List[Broadcast],
        external_goals: List[ExternalGoal],
        perceived_relations: "List[PerceivedRelation]",
    ) -> PerceptionAppraisal:
        """Appraise the perceived situation (LLM for every agent): an instinctive emotion plus
        per-need activation, not persisted. Empty on failure or no signal, so the caller carries
        the current emotion forward.
        """
        step = spatial.current_step
        recent_memory_texts = self._recent_factual_lines(
            step, top_k=_PERCEPTION_EMOTION_RECENT_FACTUAL_K, before_step=True
        )
        built = build_perception_emotion_prompt(
            personality=self.personality,
            spatial=spatial,
            inbox=inbox,
            broadcasts=broadcasts,
            external_goals=external_goals,
            perceived_relations=perceived_relations,
            recent_memory_texts=recent_memory_texts,
            seconds_per_step=self.seconds_per_step,
        )
        if built is None:
            return PerceptionAppraisal(emotion=None, need_activation={})
        system_prompt, user_prompt, signals, given_facts = built
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        # Trace extra. Recent experience is deliberately left out of "perceived": it's background,
        # not a trigger, and counting it would leave audits unable to tell an emotion grounded in
        # current signals from one made up out of old grudges.
        cur_emotion = self.personality.state.emotion
        extra_fields: dict[str, Any] = {
            # Exactly what the builder laid out; re-rendering here would disagree with it (count
            # caps, omitted sections) and report things the model never saw.
            "given_facts": given_facts,
            # Place only: time is per step, place differs per person.
            "location": situation_location(spatial),
            "perceived": list(signals),
            "relations": [
                {
                    "name": r.target_agent_name,
                    "labels": "/".join(r.labels),
                    "trust": r.trust,
                    "affection": r.affection,
                }
                for r in (perceived_relations or [])
            ],
        }
        if cur_emotion.intensity > 0.3:
            extra_fields["prior_mood"] = cur_emotion.summary()
            extra_fields["given_facts"].insert(0, f"进来时的心境：{cur_emotion.summary()}")
        try:
            llm = self.llm_router
            if self.is_main_character:
                with annotate_call(**extra_fields):
                    response = await llm.complete_with_retry(
                        LLMScene.AGENT_DECISION_MAIN,
                        messages,
                        temperature=0.5,
                        max_tokens=output_budget(130),  # reason ≤20 chars + emotion + 2 numbers + need_activation (≤5) ≈ 130 tok
                        json_mode=True,
                    )
            else:
                with annotate_call(**extra_fields):
                    response = await llm.complete(
                        LLMScene.AGENT_DECISION_MAIN,
                        messages,
                        temperature=0.5,
                        max_tokens=output_budget(130),  # reason ≤20 chars + emotion + 2 numbers + need_activation (≤5) ≈ 130 tok
                        json_mode=True,
                    )
            return parse_perception_emotion_response(response.content)
        except Exception as exc:
            # Warn, or an outage is indistinguishable from the benign "no new reaction" path.
            logger.warning(
                "perception_emotion_llm_failed",
                extra={"agent_id": self.agent_id, "error": str(exc)},
            )
            return PerceptionAppraisal(emotion=None, need_activation={})

    def note_deaths(self, dead_ids: "frozenset[str]") -> None:
        """Update this step's set of the dead with deaths after perception (decay, lethal
        actions), or they'd render as alive for the rest of the step.
        """
        self._dead_agent_ids = dead_ids

    def refresh_situation(self, spatial: "SpatialPerception") -> None:
        """Re-anchor the situation after the world moved him within a beat (a MOVE lands), or
        feedback would write "I am at A" alongside "I have arrived at B".

        Always derived from ``SpatialPerception``, as in ``perceive_step``, never assembled by hand.
        """
        self._situation = Situation.from_spatial(spatial)

    async def finalize_ongoing_action(
        self, *, result: ActionResult, step: int
    ) -> None:
        """Complete an action (naturally or interrupted): write final memories, update
        relations/needs, reset to IDLE.
        """
        landed_at = (
            result.action.target.acted_on_place if result.action.target is not None else None
        )
        if result.action.action_type == ActionType.MOVE and landed_at:
            self.personality.update_location(step=step, location=landed_at)

        await self._record_completion(
            result=result,
            dominant_need=self._resolve_dominant_need_from_state(),
            step=step,
        )

    async def _judge_goal_progress(
        self, *, step: int, result: ActionResult
    ) -> List[GoalEntity]:
        """Judge short-term goal progress; returns the updated table WITHOUT writing it back
        (``_land_goal_progress`` commits it after the join in ``_record_completion``)."""
        # Trajectory evidence for multi-step completion. Read before any await, so this step's
        # action memory isn't in it; the action reaches the judge explicitly.
        recent_factuals = self._recent_factual_lines(step, top_k=_GOAL_EVAL_RECENT_FACTUAL_K)
        succeeded_str = "成功" if result.succeeded else "失败"
        action_result = f"{result.outcome}({succeeded_str})"
        with observe_stage(Stage.FEEDBACK, agent_id=self.agent_id):
            return await self.need_engine.evaluate_goal_progress(
                step=step,
                action_description=result.action.action_description,
                action_result=action_result,
                personality=self.personality,
                situation=self._situation,
                recent_factuals=recent_factuals,
                # For a MOVE, the errand at the destination lives here; the description only says he set off.
                expected_outcome=result.expected_outcome,
            )

    async def _land_goal_progress(self, updated: List[GoalEntity], *, step: int) -> None:
        """Commit a judged goal table and write newly INTERRUPTED goals to memory.

        COMPLETED isn't recorded: the completing action already has its own memory, and a flat
        "completed X" entry would crowd it in retrieval. INTERRUPTED is: a thwarted goal often has
        no action memory (he was pulled into something else)."""
        # Don't use last_evaluated_step == step to find new ones: this can run twice per step
        # (interrupted action plus its replacement), which would write the same interrupt twice.
        already_interrupted = {
            goal.id
            for goal in self.personality.state.short_term_goal_entities
            if goal.status == GoalStatus.INTERRUPTED
        }
        self.personality.set_short_term_goal_entities(updated)
        # The returned table still holds terminal goals from the FIFO.
        for goal in updated:
            if goal.status == GoalStatus.INTERRUPTED and goal.id not in already_interrupted:
                await self.memory_system.record_event(
                    current_step=step,
                    raw_content=f"短期目标被中断，尚未完成：{goal.text}",
                    experiential_content=f"短期目标 {goal.text!r} 被打断。",
                    personality=self.personality,
                    related_agents=[],
                    triggered_by="goal_interrupted",
                    importance=MemoryImportance.LOW,
                    situation=self._situation,
                )

    async def persist_state(self, state: StateLayer) -> None:
        await self.agent_store.save_agent_state(
            self.world_id,
            self.agent_id,
            AgentState(
                world_id=self.world_id,
                agent_id=self.agent_id,
                updated_step=state.step,
                current_emotion=state.emotion.primary,
                emotion_intensity=state.emotion.intensity,
                emotion_valence=state.emotion.valence,
                emotion_triggered_by=state.emotion.triggered_by,
                active_needs=list(state.active_needs),
                dominant_need=state.dominant_need,
                long_term_goals=list(state.long_term_goals),
                short_term_goals=list(state.short_term_goals),
                current_location=state.current_location,
                activity_status=state.activity_status.value,
                activity_target=state.activity_target,
                action_status=state.action_status.value,
                current_action=state.current_action,
                action_remaining_steps=state.action_remaining_steps,
                last_action=state.last_action,
                last_action_result=state.last_action_result,
                last_action_succeeded=state.last_action_succeeded,
                short_term_goal_entities=[
                    {
                        "id": g.id,
                        "text": g.text,
                        "goal_type": g.goal_type,
                        "status": g.status if isinstance(g.status, str) else g.status.value,
                        "related_need": g.related_need.value if g.related_need is not None else None,
                        "created_step": g.created_step,
                        "due_step": g.due_step,
                        "last_evaluated_step": g.last_evaluated_step,
                        "progress_summary": g.progress_summary,
                        "origin": g.origin if isinstance(g.origin, str) else g.origin.value,
                    }
                    for g in state.short_term_goal_entities
                ],
                need_intensities=dict(state.need_intensities),
                vitality=state.vitality,
                condition=condition_to_dict(state.condition),
                last_decision_step=state.last_decision_step,
            )
        )


def snapshot_agent_state(agent: "Agent") -> Dict[str, Any]:
    """Serialize agent state for world snapshots."""

    state = agent.personality.state
    return {
        "agent_id": agent.agent_id,
        "agent_name": agent.personality.soul.name,
        "is_main_character": agent.is_main_character,
        # Immutable identity colour — one write here reaches both the map
        # (AgentStateSummary) and the relationship graph (get_graph node).
        "color": agent.personality.soul.color,
        "location_id": state.current_location,
        "current_location": state.current_location,
        "emotion": {
            # .value: json.dumps launders the str enum on disk, but the live path hands this dict
            # to the read model, where str() gives "EmotionType.FRUSTRATION". See
            # tests/unit/test_state_payload_is_jsonable.py.
            "primary": state.emotion.primary.value,
            "intensity": state.emotion.intensity,
            "valence": state.emotion.valence,
        },
        "emotion_intensity": state.emotion.intensity,
        "emotion_valence": state.emotion.valence,
        "activity_status": state.activity_status.value,
        "dominant_need": state.dominant_need,
        "current_action": state.current_action,
        "action_remaining_steps": state.action_remaining_steps,
        "last_action": state.last_action,
        "last_action_result": state.last_action_result,
        "last_action_succeeded": state.last_action_succeeded,
        "long_term_goals": list(state.long_term_goals),
        "short_term_goals": list(state.short_term_goals),
        "short_term_goal_entities": [
            {
                "text": g.text,
                "status": g.status.value if hasattr(g.status, "value") else g.status,
                # Whether the agent set this goal itself or it is unfinished business left by
                # experience; see GoalOrigin.
                "origin": g.origin.value if hasattr(g.origin, "value") else g.origin,
            }
            for g in state.short_term_goal_entities
        ],
        "vitality": state.vitality,
        # A plain JSON dict (or None), never a BodyCondition instance. This payload feeds both
        # snapshots and the read model; test_state_payload_is_jsonable rejects live objects/enums.
        "condition": condition_to_dict(state.condition),
        "is_active": agent.is_active,
        # Copy: the store may hold the payload by reference before the snapshot is written, and
        # aliasing live state would let past snapshots change as the agent evolves (same reason
        # goals use list(...) above).
        "active_needs": list(state.active_needs),
        "need_intensities": dict(state.need_intensities),
    }
