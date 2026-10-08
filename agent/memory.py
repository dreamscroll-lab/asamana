"""Dual-stream memory system."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable, Collection, Mapping
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from agent.personality import (
    SECRET_LABEL,
    EmotionType,
    PersonalityLayer,
)
from agent.importance_evaluator import ImportanceEvaluator, render_related_people
from agent.memory_ranking import (
    RETRIEVAL_SCORE_FLOOR, RETRIEVAL_WEIGHTS, VECTOR_WEIGHTS, RankedMemory, score_and_rank,
    simple_similarity, mmr_select,
)
from agent.memory_types import (
    Memory, MemoryImportance, MemoryKind, MemoryStream, RetrievalResult, coerce_importance,
    importance_level,
)
from agent.memory_write_queue import MemoryWriteQueue
from core.context import observe_stage, observe_step
from core.interfaces.embedding import EmbeddingProvider
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.interfaces.trace import Stage
from core.interfaces.perception import Situation
from core.prompts import (
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    KEEP_ABSOLUTE_TIME_RULE,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    MEMORY_ORDER_HINT,
    SituationVoice,
    condition_line,
    order_memories_chrono,
    render_memory,
    render_situation_header,
)
from core.interfaces.vector_store import SearchResult, VectorStoreProvider
from core.logging import get_logger


logger = get_logger(__name__)


COMPRESSION_TRIGGER_COUNT = 30  # compression only runs once this many candidates have accumulated
COMPRESSION_TIME_WINDOW_STEPS = 20  # time-window granularity for clustering
MIN_AGE_STEPS = 20  # a memory must be at least this old (since created_step) to become a candidate
EMOTIONAL_ANCHOR_VALENCE = 0.7  # experiential at or above this |valence| is never decayed or deleted
# Recent-event window (steps) shared by sample_recent_events and warm_recent_memories, so warming
# always covers what is sampled after a cold restart. Far below MIN_AGE_STEPS, so factual memories
# inside it are never compressed away.
RECENT_EVENT_LOOKBACK_STEPS = 10
# Ring buffer of not_executed attempts (see note_foiled_attempt): never persisted or embedded; only
# decide reads it, for its "switch approach after repeated failures" instruction. LOOKBACK caps age.
_FOILED_BUFFER_MAXLEN = 8
_FOILED_LOOKBACK_STEPS = 5
# Similarity guard (SequenceMatcher.ratio on gist) for merging foiled attempts. The structural key
# (action_type, target) is the primary merge test, since wording varies; similarity only splits
# same-key entries that clearly differ. It never merges across keys, so a mis-set value only costs a
# missed merge. Don't lower it to 0.20: the case it exists for scores about 0.21 on the gist scale.
_FOILED_MERGE_MIN_RATIO = 0.25
# Guard for object-less actions (WORK/REST/COVERT): only the type half of the key exists (see
# agent._foiled_key), so similarity must decide "same thing" and the bar is higher. Don't lower it:
# a false merge feeds a false count into the prompt's "switch strategy after 3+ failures" gate.
_FOILED_MERGE_MIN_RATIO_NO_OBJECT = 0.45
# Recent factual memories given to the inner monologue as continuity background. Keep it small: with
# more, the LLM retells past events as new. Factual only: feeding past feelings back amplifies drift.
_EXPERIENTIAL_RECENT_FACTUAL_K = 3

def memory_collection_name(world_id: str, agent_id: str, stream: MemoryStream) -> str:
    return f"{world_id}:{agent_id}:memory:{stream.value}"


def _normalize_related_agents(
    value: "Mapping[str, str] | Sequence[str] | None",
) -> tuple[list[str], dict[str, str] | None]:
    """Accept an ``{agent_id: name}`` mapping (names for prompt rendering) or a plain id sequence
    (no names); return ``(id_list, directory_or_none)``."""
    if value is None:
        return [], None
    if isinstance(value, Mapping):
        directory = dict(value)
        return list(directory.keys()), directory
    return list(value), None


class MemorySystem:
    """Per-agent memory subsystem: one entry point for both streams, all three kinds, R/I/R
    ranking and the memory lifecycle.

    Invariants
    ----------
    1. Importance is LLM-judged for every agent (CLAUDE.md §5); the structured rule is only the
       failure floor. `is_main_character` is a narrative tier. It sets the experiential write
       threshold and never chooses LLM vs. rules.
    2. Writes never touch personality (executor/feedback boundary). MemorySystem only changes
       memory. Emotion, relation and state changes belong to the caller (the feedback path in
       agent.py).
    3. Failures don't stop the simulation (Rule 1 / Rule 6). LLM calls and vector-store reads
       return a conservative default on failure instead of raising. Write failures are logged at
       error, not swallowed.
    4. Asymmetric writes. `record_event` always writes factual, and experiential only when
       _should_write_experiential passes, linked by event_group_id.
    5. Only Compression and Forget events may delete memories. Decay never deletes.
       _is_emotional_anchor protects CRITICAL memories, insights and high-|valence| experiential
       memories from every deletion path.
    6. kind has three values: event (a concrete experience), insight (from reflection) and
       summary (from compression). event_group_id links the two streams of one event;
       source_ids lists the memories a memory was derived from.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        embedding: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        *,
        world_id: str,
        agent_id: str,
        is_main_character: bool = False,
        time_label_for: Callable[[int], str] | None = None,
        world_start_second_of_day: int = 0,
        retrieval_score_floor: float = RETRIEVAL_SCORE_FLOOR,
        seconds_per_step: int = 1,
    ) -> None:
        self._llm_router = llm_router
        self._embedding = embedding
        self._vector_store = vector_store
        self._world_id = world_id
        self._agent_id = agent_id
        # Narrative tier; drives the experiential write threshold (_should_write_experiential).
        self._is_main_character = is_main_character
        self._importance = ImportanceEvaluator(
            llm_router=self._llm_router,
            agent_id=self._agent_id,
        )
        # Tests on the in_memory hash embedding (random scores) set this to 0.0 via a fixture.
        self._retrieval_score_floor = retrieval_score_floor
        # step → world-calendar label for compression summaries' time span; None = no clock (tests).
        self._time_label_for = time_label_for
        # The only conversion from steps to narrative durations (see core.prompts.render_memory).
        self._seconds_per_step = max(1, int(seconds_per_step))
        # Seconds since midnight at world start: render_memory needs it to find day boundaries
        # ("today/yesterday"), which a duration alone can't tell.
        self._world_start_second_of_day = world_start_second_of_day
        self._entries: Dict[str, Memory] = {}
        # Transient traces of actions that didn't happen (foiled attempts, intents displaced by
        # conscription). Not in _entries or the vector store, so deep cognition (relation_evolution,
        # reflection, summary, goal-eval) can't see them; only decide/motivation read them via
        # recent_foiled_attempts. Empty after restore, which is harmless.
        #
        # Entries are (step, first-person membrane-safe text, key, gist):
        # - text is for display, gist for comparison: every text shares the same "I meant to… but
        #   couldn't" wrapper, which would inflate similarity and defeat the merge guard.
        # - key = (action_type, acted_on) is the primary merge criterion; None means no key and the
        #   entry never merges.
        self._recent_foiled: "deque[tuple[int, str, tuple[str, str] | None, str]]" = deque(
            maxlen=_FOILED_BUFFER_MAXLEN
        )
        # Monotonic counter that keeps summary / insight ids unique. Don't use len(_entries): it
        # drops after deletions and ids would collide.
        self._seq: int = 0
        # Fire-and-forget queue for the slow work (experiential rewrite, embed, upsert), drained in
        # order by one worker so the step never blocks. Id reservation and the _entries insert stay
        # synchronous before enqueuing. The worker starts on the first enqueue, inside an event loop.
        self._write_q = MemoryWriteQueue(agent_id=self._agent_id)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def drain_writes(self) -> None:
        """Barrier before maintenance (compression/reflection). See MemoryWriteQueue.drain."""
        await self._write_q.drain()

    async def aclose(self) -> None:
        """Drain and stop the background write worker when run() exits. See MemoryWriteQueue.close."""
        await self._write_q.close()

    async def write(
        self,
        raw_content: str,
        stream: MemoryStream,
        *,
        personality: PersonalityLayer,
        related_agents: "Mapping[str, str] | Sequence[str] | None" = None,
        related_relations_text: str = "",
        triggered_by: str | None = None,
        importance: MemoryImportance | float | None = None,
        current_step: int | None = None,
        metadata: dict[str, Any] | None = None,
        decay_score: float = 1.0,
        event_group_id: str | None = None,
        dominant_need_label: str = "",
        situation: Situation = Situation(),
    ) -> Memory | None:
        """Write one memory through its stream's pipeline; event_group_id binds it to the other stream."""

        step = personality.state.step if current_step is None else current_step
        related_list, related_directory = _normalize_related_agents(related_agents)
        md: dict[str, Any] = {"world_id": self._world_id, **(metadata or {})}

        if stream == MemoryStream.FACTUAL:
            # Verbatim: an LLM rewrite would add subjective judgment, which belongs to the
            # experiential stream.
            stored_content = raw_content
            emotion_valence = 0.0
            emotion_label = "objective"
        else:
            stored_content = await self._write_experiential(
                raw_content=raw_content,
                personality=personality,
                related_agent_directory=related_directory,
                related_relations_text=related_relations_text,
                situation=situation,
                now_step=step,
            )
            if len(stored_content) == 0:
                return
            emotion_valence = personality.state.emotion.valence
            emotion_label = personality.state.emotion.primary

        if importance is not None:
            memory_importance = coerce_importance(importance)
        else:
            memory_importance = await self._importance.evaluate(
                raw_content,
                personality=personality,
                related_agents=related_list,
                related_directory=related_directory,
                related_relations_text=related_relations_text,
                dominant_need_label=dominant_need_label,
            )

        memory = Memory(
            id=f"{self._agent_id}-{stream.value}-{step}-{self._next_seq()}",
            agent_id=self._agent_id,
            stream=stream,
            raw_content=raw_content,
            stored_content=stored_content,
            importance=memory_importance,
            created_step=step,
            metadata=md,
            emotion_valence=emotion_valence,
            emotion_label=emotion_label,
            related_agents=related_list,
            triggered_by=triggered_by,
            decay_score=decay_score,
            kind=MemoryKind.EVENT,
            event_group_id=event_group_id,
        )
        await self._persist(memory)
        return memory

    async def record_event(
        self,
        *,
        current_step: int,
        raw_content: str,
        personality: PersonalityLayer,
        related_agents: "Mapping[str, str] | Sequence[str] | None" = None,
        related_relations_text: str = "",
        triggered_by: str | None = None,
        importance: MemoryImportance | float | None = None,
        experiential_content: str | None = None,
        metadata: dict[str, Any] | None = None,
        decay_score: float = 1.0,
        dominant_need_label: str = "",
        situation: Situation = Situation(),
    ) -> List[Memory]:
        """Write factual (always) and experiential (only if _should_write_experiential) for one
        event, bound by a shared event_group_id."""

        related_list, related_directory = _normalize_related_agents(related_agents)
        event_group_id = f"{self._agent_id}-evt-{current_step}-{self._next_seq()}"

        # Importance is resolved synchronously: the experiential gate and retrieval ranking both need
        # the real score, and a rule-based provisional one would drop the inner stream of events that
        # weigh on the agent more than objective rules say. Only the rewrite and embed/upsert, which
        # don't affect decision or recall correctness, go to the background.
        if importance is not None:
            resolved_importance: float = coerce_importance(importance)
        else:
            resolved_importance = await self._importance.evaluate(
                raw_content,
                personality=personality,
                related_agents=related_list,
                related_directory=related_directory,
                related_relations_text=related_relations_text,
                dominant_need_label=dominant_need_label,
            )
        write_experiential = self._should_write_experiential(
            personality=personality, importance_score=resolved_importance,
        )

        factual = self._provisional_memory(
            raw_content, MemoryStream.FACTUAL, resolved_importance, event_group_id,
            current_step, related_list, triggered_by, metadata, decay_score, personality,
        )
        self._entries[factual.id] = factual
        provisioned = [factual]
        experiential: "Memory | None" = None
        recent_factual_block = ""
        if write_experiential:
            # Id reserved now, but not in _entries until the rewrite lands (_refine_event): before
            # that its content would just duplicate the factual.
            experiential = self._provisional_memory(
                experiential_content or raw_content, MemoryStream.EXPERIENTIAL, resolved_importance,
                event_group_id, current_step, related_list, triggered_by, metadata, decay_score, personality,
            )
            provisioned.append(experiential)
            # Built now, not in _refine_event: that may run steps later, when "recent" would include
            # later events.
            recent_factual_block = self._build_recent_factual_continuity_block(
                current_step=current_step, exclude_event_group_id=event_group_id,
            )

        self._write_q.enqueue(self._refine_event(
            factual=factual,
            experiential=experiential,
            personality=personality,
            related_directory=related_directory,
            related_relations_text=related_relations_text,
            situation=situation,
            recent_factual_block=recent_factual_block,
        ))
        return provisioned

    def _build_recent_factual_continuity_block(
        self, *, current_step: int, exclude_event_group_id: str | None,
    ) -> str:
        """The last few FACTUAL memories, oldest first, rendered via render_memory as the
        monologue's continuity background.

        Excludes the current event: its factual is already in _entries and would otherwise appear
        as "earlier", duplicating the "【刚发生的事】" block. Failure → "" (the prompt omits the
        section).
        """
        try:
            pairs = self.sample_recent_events(
                current_step, top_k=_EXPERIENTIAL_RECENT_FACTUAL_K + 1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "recent_factual_continuity_sample_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return ""
        ordered = order_memories_chrono(pairs, key=lambda pair: pair[0] or pair[1])
        lines: list[str] = []
        for fact, _exp in ordered:
            if fact is None:
                continue
            if exclude_event_group_id and fact.event_group_id == exclude_event_group_id:
                continue
            rendered = render_memory(
                fact,
                now_step=current_step,
                seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day,
            )
            if rendered:
                lines.append(f"- {rendered}")
            if len(lines) >= _EXPERIENTIAL_RECENT_FACTUAL_K:
                break
        return "\n".join(lines)

    def _provisional_memory(
        self,
        raw_content: str,
        stream: MemoryStream,
        importance: float,
        event_group_id: str,
        step: int,
        related_list: "Sequence[str]",
        triggered_by: str | None,
        metadata: dict[str, Any] | None,
        decay_score: float,
        personality: PersonalityLayer,
    ) -> Memory:
        """Build a memory with stored_content=raw as a placeholder until the background rewrite.
        The id is reserved here, synchronously, so ids stay monotonic and the streams paired."""
        md: dict[str, Any] = {"world_id": self._world_id, **(metadata or {})}
        if stream == MemoryStream.FACTUAL:
            emotion_valence, emotion_label = 0.0, "objective"
        else:
            emotion_valence = personality.state.emotion.valence
            emotion_label = personality.state.emotion.primary
        return Memory(
            id=f"{self._agent_id}-{stream.value}-{step}-{self._next_seq()}",
            agent_id=self._agent_id,
            stream=stream,
            raw_content=raw_content,
            stored_content=raw_content,
            importance=importance,
            created_step=step,
            metadata=md,
            emotion_valence=emotion_valence,
            emotion_label=emotion_label,
            related_agents=list(related_list),
            triggered_by=triggered_by,
            decay_score=decay_score,
            kind=MemoryKind.EVENT,
            event_group_id=event_group_id,
        )

    async def _refine_event(
        self,
        *,
        factual: Memory,
        experiential: "Memory | None",
        personality: PersonalityLayer,
        related_directory: dict[str, str] | None,
        related_relations_text: str,
        situation: Situation,
        recent_factual_block: str = "",
    ) -> None:
        """Background half of record_event: persist the factual, then rewrite and persist the
        experiential. A failed rewrite writes no inner stream.

        Runs queued, possibly steps later, so the trace is attributed to the event's
        ``created_step``; the rest of the context comes from enqueue time (see
        ``MemoryWriteQueue.enqueue``).
        """
        with observe_step(factual.created_step):
            await self._persist_vector(factual)
            if experiential is not None:
                rewritten = await self._write_experiential(
                    raw_content=experiential.raw_content,
                    personality=personality,
                    related_agent_directory=related_directory,
                    related_relations_text=related_relations_text,
                    situation=situation,
                    recent_factual_block=recent_factual_block,
                    # Condition durations are measured as of the event, not the rewrite.
                    now_step=factual.created_step,
                )
                if rewritten:
                    # Inserting from the background is safe: one sequential worker, and sync
                    # readers iterate atomically.
                    experiential.stored_content = rewritten
                    self._entries[experiential.id] = experiential
                    await self._persist_vector(experiential)

    def _should_write_experiential(
        self,
        *,
        personality: PersonalityLayer,
        importance_score: float,
    ) -> bool:
        """Asymmetric write gate, tiered by narrative level (CLAUDE.md §5).

        Everyone needs importance ≥ 0.4: strong emotion about an event that doesn't matter would
        invent significance. Then emotion.intensity ≥ 0.4 for main characters, ≥ 0.6 for background
        agents (not a narrative focus; saves LLM calls).

        importance_score must be the resolved score. Don't accept None as 0.0: "unknown" is not
        "zero".
        """
        if importance_score < 0.4:
            return False
        if self._is_main_character:
            return float(personality.state.emotion.intensity) >= 0.4
        return float(personality.state.emotion.intensity) >= 0.6

    async def recall_about_agent(
        self,
        target_agent_id: str,
        query: str,
        *,
        current_step: int,
        top_k: int = 5,
        stream: MemoryStream | None = None,
        exclude_ids: Collection[str] = (),
    ) -> List[Memory]:
        """Identity-scoped retrieval of "what I know about someone", scoped by
        ``Memory.related_agents``.

        Who a memory is about is a structured fact; don't answer it by vector search on a name: in a
        single-theme world that collapses into "memories most like the current topic", often about
        someone else. An empty scope returns []; don't fall back to global search, since "I barely
        know this person" beats an invented relationship.

        ``query`` only ranks within the scope: pass the current topic/intent, not the name.
        """
        return await self.retrieve(
            query,
            current_step=current_step,
            top_k=top_k,
            stream=stream,
            related_agent_id=target_agent_id,
            exclude_ids=exclude_ids,
        )

    async def retrieve(
        self,
        query: str,
        *,
        current_step: int,
        top_k: int = 10,
        stream: MemoryStream | None = None,
        related_agent_id: str | None = None,
        filters: dict[str, Any] | None = None,
        before_step: int | None = None,
        trace: list[RankedMemory] | None = None,
        touch: bool = True,
        exclude_ids: Collection[str] = (),
    ) -> List[Memory]:
        """Recall relevant memories, optionally from one stream.

        - ``related_agent_id``: identity scope, pushed down to the store as a membership predicate.
          A post-filter would run after the store already cut top_k by similarity. See
          ``recall_about_agent``.
        - ``trace``: collects every candidate, dropped ones included, with per-factor scores, for
          dev tools.
        - ``touch``: False skips writing back last_accessed_step; diagnostic reads must not raise
          recency.
        - ``exclude_ids``: removed before the top_k cut. Don't over-fetch and drop afterwards: the
          extras would be touched without being used.
        - Decay floor: below 0.2 the final score is multiplied by decay_score, so an old memory
          still resurfaces when importance and relevance are both high.

        No vector bias for dominant_need: motivation reaches recall through the query text and
        write-time importance.
        """

        # embed("") yields a degenerate vector that would return arbitrary memories.
        if not query or not query.strip():
            return []

        retrieve_start = time.perf_counter()
        query_embedding = await self._embedding.embed(query)
        streams = [stream] if stream is not None else [MemoryStream.FACTUAL, MemoryStream.EXPERIENTIAL]
        weights = RETRIEVAL_WEIGHTS

        scope = dict(filters or {})
        if related_agent_id is not None:
            scope["related_agents"] = related_agent_id

        candidates: list[tuple[Memory, SearchResult]] = []
        for active_stream in streams:
            dense_weight, sparse_weight = VECTOR_WEIGHTS
            results = await self._vector_store.search(
                collection=self._namespace(active_stream),
                dense_vector=query_embedding.dense,
                sparse_vector=query_embedding.sparse,
                top_k=max(top_k * 3, 6),
                filters=scope or None,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            for result in results:
                memory = self._memory_from_result(result)
                if memory is None:
                    continue
                if before_step is not None and memory.created_step >= before_step:
                    continue
                if memory.id in exclude_ids:
                    continue
                # The floor compares fused, not dense: an exact proper-noun hit can be high on sparse
                # but middling on dense. It must run before min-max normalization, which erases
                # absolute relevance.
                if float(result.score) < self._retrieval_score_floor:
                    if trace is not None:
                        trace.append(
                            RankedMemory(
                                memory=memory,
                                dense=float(result.dense_score),
                                sparse=float(result.sparse_score),
                                fused=float(result.score),
                                dropped="floor",
                            )
                        )
                    continue
                candidates.append((memory, result))

        # Normalize over the merged pool, never per stream: per-stream min-max would give each
        # stream's best hit relevance 1.0, so a weak match ties with an exact hit in the other stream.
        ranked = score_and_rank(candidates, current_step, weights)

        # mmr_select annotates drop reasons in place, so ranked is then the full funnel.
        selected = mmr_select(ranked, top_k=top_k)
        if trace is not None:
            trace.extend(ranked)
        if touch:
            for item in selected:
                item.memory.touch(current_step)
                await self._persist_payload(item.memory)
        logger.debug(
            "memory_retrieve",
            extra={
                "agent_id": self._agent_id,
                "stream": stream.value if stream is not None else "both",
                "query": query[:120],
                "query_count": len(streams),
                "top_k": top_k,
                "scoped_to_agent": related_agent_id or "",
                "candidate_count": len(ranked),
                "result_count": len(selected),
                "scores": [round(item.score, 3) for item in selected],
                "elapsed_ms": round((time.perf_counter() - retrieve_start) * 1000.0, 2),
            },
        )
        return [item.memory for item in selected]

    async def retrieve_both(
        self,
        query: str,
        *,
        current_step: int,
        top_k_each: int = 5,
        related_agent_id: str | None = None,
        before_step: int | None = None,
    ) -> RetrievalResult:
        """Retrieve both streams (factual_query / experiential_query) into a RetrievalResult grouped
        by kind; see _compose_retrieval_result for linking."""
        from agent.perception import RetrievalQuery

        both_start = time.perf_counter()
        if isinstance(query, RetrievalQuery):
            factual_query: str = query.as_primary()
            experiential_query: str = query.as_combined()
        else:
            factual_query = experiential_query = query

        # One empty side is handled by retrieve's own empty-query guard.
        if not factual_query.strip() and not experiential_query.strip():
            return RetrievalResult()

        factual_result, experiential_result = await asyncio.gather(
            self.retrieve(
                factual_query,
                current_step=current_step,
                top_k=top_k_each,
                stream=MemoryStream.FACTUAL,
                related_agent_id=related_agent_id,
                before_step=before_step,
            ),
            self.retrieve(
                experiential_query,
                current_step=current_step,
                top_k=top_k_each,
                stream=MemoryStream.EXPERIENTIAL,
                related_agent_id=related_agent_id,
                before_step=before_step,
            ),
            return_exceptions=True,
        )
        factual: List[Memory] = [] if isinstance(factual_result, BaseException) else factual_result
        experiential: List[Memory] = [] if isinstance(experiential_result, BaseException) else experiential_result

        result = self._compose_retrieval_result(factual=factual, experiential=experiential)
        logger.debug(
            "memory_retrieve",
            extra={
                "agent_id": self._agent_id,
                "stream": "both",
                "query_count": 2,
                "top_k": top_k_each,
                "events": len(result.events),
                "insights": len(result.insights),
                "summaries": len(result.period_summaries),
                "elapsed_ms": round((time.perf_counter() - both_start) * 1000.0, 2),
            },
        )
        return result

    def _compose_retrieval_result(
        self,
        *,
        factual: List[Memory],
        experiential: List[Memory],
    ) -> RetrievalResult:
        """Group dual-stream results by kind, with link-aware dedup:
        - factual + experiential sharing an event_group_id are both kept (complementary);
        - an insight's sources that were also hit move into insight_sources (≤ 3) instead of taking
          top-k slots.
        """
        events: List[Memory] = []
        insights: List[Memory] = []
        period_summaries: List[Memory] = []
        all_pool: dict[str, Memory] = {}
        for memory in [*factual, *experiential]:
            all_pool[memory.id] = memory

        insight_source_ids: set[str] = set()
        for memory in all_pool.values():
            if memory.kind == MemoryKind.INSIGHT and memory.source_ids:
                insight_source_ids.update(memory.source_ids)

        for memory in all_pool.values():
            if memory.kind == MemoryKind.EVENT:
                if memory.id in insight_source_ids:
                    continue
                events.append(memory)
            elif memory.kind == MemoryKind.INSIGHT:
                insights.append(memory)
            elif memory.kind == MemoryKind.SUMMARY:
                period_summaries.append(memory)

        # memory id → the paired memory id in the other stream
        stream_links: Dict[str, str] = {}
        by_group: dict[str, list[Memory]] = {}
        for memory in events:
            if memory.event_group_id:
                by_group.setdefault(memory.event_group_id, []).append(memory)
        for group_id, members in by_group.items():
            if len(members) >= 2:
                for m in members:
                    other = next((other for other in members if other.id != m.id and other.stream != m.stream), None)
                    if other is not None:
                        stream_links[m.id] = other.id

        insight_sources: Dict[str, List[Memory]] = {}
        for ins in insights:
            if not ins.source_ids:
                continue
            sources = [all_pool[sid] for sid in ins.source_ids if sid in all_pool]
            if sources:
                insight_sources[ins.id] = sources[:3]

        return RetrievalResult(
            events=events,
            insights=insights,
            period_summaries=period_summaries,
            stream_links=stream_links,
            insight_sources=insight_sources,
        )

    def related_memory_bias(self, *, target_agent_id: str, current_step: int, lookback_steps: int = 8) -> float:
        """Mean valence of recent experiential memories involving another agent."""

        relevant = [
            memory
            for memory in self._entries.values()
            if memory.stream == MemoryStream.EXPERIENTIAL
            and target_agent_id in memory.related_agents
            and current_step - memory.created_step <= lookback_steps
        ]
        if not relevant:
            return 0.0
        return sum(memory.emotion_valence for memory in relevant) / len(relevant)

    def note_foiled_attempt(
        self,
        *,
        step: int,
        text: str,
        key: "tuple[str, str] | None" = None,
        gist: str = "",
    ) -> None:
        """Record an action that didn't happen: a ``not_executed`` foiled attempt, or a decision
        dropped because arbitration conscripted the agent (see Agent.defer_decided_intent).

        Never persisted: nothing changed in the world, and a templated "I meant to X but couldn't"
        memory would dilute retrieval and feed relation_evolution a dealing that never happened.

        text must be first-person and membrane-safe; it goes straight into the decide prompt.
        ``key``/``gist`` are optional merge inputs (none → never merges). ``gist`` should be the
        ``action_description``, not ``text`` (see ``_recent_foiled``).
        """
        cleaned = (text or "").strip()
        if cleaned:
            self._recent_foiled.append((step, cleaned, key, (gist or cleaned).strip()))

    def clear_foiled_attempts(self, key: "tuple[str, str]", gist: str = "") -> None:
        """The action finally happened: remove its foiled entries.

        Must use the same criterion as merging, or counting and clearing drift apart. A full key
        clears regardless of content: every buffered failure is about reach (busy / absent / no
        path / conscripted), so landing the key means the obstacle is gone. An object-less key also
        requires ``gist`` similarity ≥ ``_FOILED_MERGE_MIN_RATIO_NO_OBJECT``.
        """
        if not key:
            return
        if key[1]:
            kept = [item for item in self._recent_foiled if item[2] != key]
        else:
            kept = [
                item for item in self._recent_foiled
                if item[2] != key
                or simple_similarity(item[3], gist) < _FOILED_MERGE_MIN_RATIO_NO_OBJECT
            ]
        if len(kept) != len(self._recent_foiled):
            self._recent_foiled.clear()
            self._recent_foiled.extend(kept)

    def recent_foiled_attempts(
        self, current_step: int, *, window: int = _FOILED_LOOKBACK_STEPS
    ) -> list[str]:
        """Actions that didn't happen within the last ``window`` steps, one line per distinct
        attempt with a count, oldest first and rendered via render_memory like normal memories.

        Why merge: repeating the same blocked intent once per step makes the model more likely to
        retry and loop; one line with a count lets the "switch approach" instruction fire.

        Merge rules:
        1. filter to the window;
        2. entries with the same ``key`` form a group; each ``key is None`` entry is its own group;
        3. similarity guard: compare each ``gist`` with the group's representative only (all-pairs
           wouldn't be transitive) and split below ``_FOILED_MERGE_MIN_RATIO`` (full key) or
           ``_FOILED_MERGE_MIN_RATIO_NO_OBJECT`` (object-less key);
        4. representative = the entry with the largest step (ties: the later write);
        5. groups sorted by their representative's step, ascending;
        6. a count is attached only when ``n >= 2``.

        Sort by step explicitly: append order matching step order is only a coincidence of the call
        sites.
        """
        threshold = current_step - window
        recent = sorted(
            (item for item in self._recent_foiled if item[0] >= threshold),
            key=lambda item: item[0],
        )

        # Newest first, so the first entry of each group is its representative.
        groups: list[dict] = []
        for step, text, key, gist in reversed(recent):
            floor = (
                _FOILED_MERGE_MIN_RATIO if key and key[1]
                else _FOILED_MERGE_MIN_RATIO_NO_OBJECT
            )
            for group in groups:
                if key is None or group["key"] != key:
                    continue
                if simple_similarity(gist, group["gist"]) < floor:
                    continue        # same key but clearly different content: two separate attempts
                group["count"] += 1
                break
            else:
                groups.append({"step": step, "text": text, "key": key, "gist": gist, "count": 1})

        groups.sort(key=lambda g: g["step"])
        # render_memory is duck-typed (stored_content / created_step / kind). The count is a
        # render-time aggregate, so it is appended after rendering, never into stored_content.
        out: list[str] = []
        for group in groups:
            line = render_memory(
                SimpleNamespace(stored_content=group["text"], created_step=group["step"], kind=MemoryKind.EVENT),
                now_step=current_step,
                seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day,
            )
            if group["count"] >= 2:
                # "Recently", not "in a row": the attempts aren't known to be consecutive.
                line = f"{line}近来同一件事我已试过{group['count']}次。"
            out.append(line)
        return out

    def sample_recent_events(
        self, current_step: int, *, lookback: int = RECENT_EVENT_LOOKBACK_STEPS, top_k: int = 10
    ) -> list[tuple[Memory | None, Memory | None]]:
        """Top-k recent EVENTS (last ``lookback`` steps), each paired as
        ``(factual, experiential)`` by ``event_group_id``, ranked by max importance.

        Short-term goals need both what happened and how the agent read it: with only the feeling
        stream they repeat or drift with emotion. Experiential may be None; factual is None only in
        the edge case where just the experiential is in _entries.

        Scans _entries only (warm_recent_memories fills it after a restart); insights and summaries
        are excluded.
        """
        threshold = current_step - lookback
        groups: dict[str, dict[MemoryStream, Memory]] = {}
        for m in self._entries.values():
            if m.kind != MemoryKind.EVENT or m.created_step < threshold:
                continue
            groups.setdefault(m.event_group_id or m.id, {})[m.stream] = m
        ranked = sorted(
            groups.values(),
            key=lambda slot: max(float(m.importance) for m in slot.values()),
            reverse=True,
        )[:top_k]
        return [
            (slot.get(MemoryStream.FACTUAL), slot.get(MemoryStream.EXPERIENTIAL))
            for slot in ranked
        ]

    async def sample_recent(
        self, current_step: int, *, lookback: int = 30, top_k: int = 8
    ) -> list[Memory]:
        """Top-k most important recent memories across both streams, read from the vector store.

        Long-term goal revision must see objective facts (e.g. a protected figure has died), and
        ``_entries`` only holds the last RECENT_EVENT_LOOKBACK_STEPS after a restore, a narrower
        window than ``lookback``. Warms ``_entries`` as a side effect; a failed stream read is skipped.
        """
        threshold = current_step - lookback
        candidates: list[Memory] = []
        for stream in MemoryStream:
            try:
                results = await self._vector_store.list_all(self._namespace(stream))
            except Exception as exc:
                logger.warning(
                    "sample_recent_list_failed",
                    extra={"agent_id": self._agent_id, "stream": stream.value, "error": str(exc)},
                )
                continue
            for result in results:
                memory = self._memory_from_result(result)  # warms _entries (side effect)
                if memory is not None and memory.created_step >= threshold:
                    candidates.append(memory)
        return sorted(candidates, key=lambda m: float(m.importance), reverse=True)[:top_k]

    def collect_recent_related_agents(
        self, *, current_step: int, lookback_steps: int
    ) -> set[str]:
        """Agent ids in memories within the window (scans `_entries` only)."""
        threshold = current_step - lookback_steps
        return {
            aid
            for memory in self._entries.values()
            if memory.created_step >= threshold
            for aid in memory.related_agents
        }

    def list_recent_memories_mentioning(
        self, target_agent_id: str, *, current_step: int, lookback_steps: int
    ) -> list[Memory]:
        """Memories within the window involving ``target_agent_id``, oldest first (scans `_entries`
        only)."""
        threshold = current_step - lookback_steps
        relevant = [
            memory
            for memory in self._entries.values()
            if memory.created_step >= threshold
            and target_agent_id in memory.related_agents
        ]
        return sorted(relevant, key=lambda m: m.created_step)

    async def warm_recent_memories(
        self, current_step: int, lookback_steps: int = RECENT_EVENT_LOOKBACK_STEPS
    ) -> None:
        """After a snapshot restore, load both streams' recent memories into _entries so the
        _entries-scanning readers (related_memory_bias, sample_recent_events,
        collect_recent_related_agents) see pre-restore memories from the first step."""
        threshold = current_step - lookback_steps
        for stream in MemoryStream:
            try:
                all_results = await self._vector_store.list_all(self._namespace(stream))
            except Exception as exc:
                logger.warning(
                    "warm_recent_list_failed",
                    extra={"agent_id": self._agent_id, "stream": stream.value, "error": str(exc)},
                )
                continue
            for result in all_results:
                memory = self._memory_from_result(result)
                if memory is not None and memory.created_step >= threshold:
                    self._entries[memory.id] = memory

    async def ensure_collections(self) -> None:
        for stream in MemoryStream:
            await self._vector_store.create_collection(
                collection=self._namespace(stream),
                dimension=self._embedding.dimension,
            )

    async def purge_runtime_memories(self) -> int:
        """On reset, delete runtime memories (``created_step > 0``) and keep the build-time
        backstory seeds (``created_step <= 0``, see ``_write_historical_memories``). Returns the
        count removed."""
        removed = 0
        for stream in MemoryStream:
            collection = self._namespace(stream)
            try:
                results = await self._vector_store.list_all(collection)
            except Exception as exc:
                # Error, not warning: the reset then keeps this stream's runtime memories.
                logger.error(
                    "purge_runtime_list_failed",
                    extra={"agent_id": self._agent_id, "stream": stream.value, "error": str(exc)},
                )
                continue
            for result in results:
                created = int(result.payload.get("created_step", result.payload.get("step", 0)) or 0)
                if created <= 0:
                    continue
                try:
                    await self._vector_store.delete(collection, result.id)
                    self._entries.pop(result.id, None)
                    removed += 1
                except Exception as exc:
                    logger.warning(
                        "purge_runtime_memory_failed",
                        extra={"agent_id": self._agent_id, "memory_id": result.id, "error": str(exc)},
                    )
        return removed

    async def seed_factual_memory(
        self,
        *,
        current_step: int,
        raw_content: str,
        related_agents: Sequence[str] | None = None,
        triggered_by: str | None = None,
        importance: MemoryImportance | float | None = None,
        metadata: dict[str, Any] | None = None,
        decay_score: float = 1.0,
    ) -> Memory:
        """Persist an objective factual seed without LLM rewriting or experiential echo.

        Raises if persisting fails (Rule 2): the build-time backstory has no second chance to be
        written, unlike runtime writes (Rule 1, see _persist_vector).
        """

        memory = Memory(
            id=f"{self._agent_id}-factual-seed-{current_step}-{self._next_seq()}",
            agent_id=self._agent_id,
            stream=MemoryStream.FACTUAL,
            raw_content=raw_content,
            stored_content=raw_content,
            importance=coerce_importance(importance or MemoryImportance.MEDIUM),
            created_step=current_step,
            metadata={"world_id": self._world_id, **(metadata or {})},
            emotion_valence=0.0,
            emotion_label="objective",
            related_agents=list(related_agents or []),
            triggered_by=triggered_by,
            decay_score=decay_score,
        )
        if not await self._persist(memory):
            raise RuntimeError(
                f"Seed memory {memory.id!r} could not be persisted to the vector store."
            )
        return memory

    async def apply_decay(self, current_step: int) -> None:
        """Decay non-anchor memories. Decay lowers how easily a memory is recalled; it never
        deletes.

        Why fade: unrevisited memories should recall weakly, so recent events stand out and earlier
        periods recede as a character changes; R/I/R ranking alone leaves everything equally sharp.

        Why never delete: deletion is irreversible, and retrieval_count doesn't measure whether the
        agent cares; a trauma unrecalled for 200 steps must still surface on a cue. Compression and
        Forget events (e.g. MEMORY_LOSS) are the only deletion paths, so every loss has a cause in
        the story.

        Emotional anchors (_is_emotional_anchor) don't decay: they carry identity and motivation.
        """

        for stream in MemoryStream:
            collection = self._namespace(stream)
            try:
                results = await self._vector_store.list_all(collection)
            except Exception as exc:  # noqa: BLE001 — Rule 6: on read failure, log a warning and keep going
                # Nothing above this catches up to runtime.run_step: skipping one round of decay is
                # harmless, losing the whole step is not.
                logger.warning(
                    "memory_decay_list_failed",
                    extra={"agent_id": self._agent_id, "stream": stream.value, "error": str(exc)},
                )
                continue
            for result in results:
                memory = self._memory_from_result(result)
                if memory is None:
                    continue
                if _is_emotional_anchor(memory):
                    continue
                base_decay = 0.95
                consolidation_bonus = min(memory.retrieval_count * 0.01, 0.04)
                decay_rate = base_decay + consolidation_bonus
                if stream == MemoryStream.EXPERIENTIAL:
                    decay_rate = min(decay_rate + 0.03, 0.99)
                memory.decay_score *= decay_rate
                # TODO(clarity_degradation): when decay_score crosses 0.5 / 0.3, have the LLM rewrite
                # stored_content into a vaguer version. That changes the text, so it must go through
                # _persist (re-embed), not this payload write-back.
                await self._persist_payload(memory)

    async def compress(
        self,
        stream: MemoryStream,
        *,
        personality: PersonalityLayer | None = None,
        current_step: int | None = None,
    ) -> int:
        """Compress old FACTUAL memories into objective summaries: the last stage of aging and the
        only bulk-deletion path. Experiential is never compressed (decay fades it, Reflection
        distills it), so other streams return 0.

        - Candidates: _is_compression_candidate; runs only at ≥ COMPRESSION_TRIGGER_COUNT.
        - Clusters: _cluster_for_compression.
        - source_ids keeps every member id, for the insight fixup.
        - Members are deleted only after the summary persists.

        Returns the number of summaries produced.
        """
        if stream != MemoryStream.FACTUAL:
            return 0

        collection = self._namespace(stream)
        candidates: list[Memory] = []
        ref_step = current_step if current_step is not None else 0
        for result in await self._vector_store.list_all(collection):
            memory = self._memory_from_result(result)
            if memory is None:
                continue
            if not _is_compression_candidate(memory, current_step=ref_step):
                continue
            candidates.append(memory)

        if len(candidates) < COMPRESSION_TRIGGER_COUNT:
            return 0

        clusters = _cluster_for_compression(candidates)
        produced = 0
        for cluster in clusters:
            if len(cluster) < 3 or len(cluster) > 10:
                continue
            summary = await self._summarize_cluster_for_stream(cluster, stream=stream)
            if summary is None:
                # Keep the originals and retry next interval (Rule 1).
                logger.warning(
                    "compression_cluster_skipped",
                    extra={"agent_id": self._agent_id, "stream": stream.value},
                )
                continue
            representative = cluster[-1]
            cluster_importance = max(memory.importance for memory in cluster)
            summary_memory = Memory(
                id=f"{self._agent_id}-{stream.value}-summary-{representative.created_step}-{self._next_seq()}",
                agent_id=self._agent_id,
                stream=stream,
                raw_content=summary,
                stored_content=summary,
                importance=min(0.95, cluster_importance * 0.8),
                created_step=representative.created_step,
                metadata={"world_id": self._world_id},
                emotion_valence=_average([memory.emotion_valence for memory in cluster]),
                emotion_label="summary",
                related_agents=sorted({agent_id for memory in cluster for agent_id in memory.related_agents}),
                triggered_by="compression",
                decay_score=0.6,
                kind=MemoryKind.SUMMARY,
                source_ids=[memory.id for memory in cluster],
            )
            try:
                await self._persist(summary_memory)
            except Exception as exc:
                logger.error(
                    "compression_summary_persist_failed",
                    extra={"agent_id": self._agent_id, "stream": stream.value, "error": str(exc)},
                )
                # Roll back; the source memories stay.
                self._entries.pop(summary_memory.id, None)
                continue

            deleted_ids: set[str] = set()
            for memory in cluster:
                try:
                    await self._vector_store.delete(collection, memory.id)
                    self._entries.pop(memory.id, None)
                    deleted_ids.add(memory.id)
                except Exception as exc:
                    logger.error(
                        "compression_source_delete_failed",
                        extra={
                            "agent_id": self._agent_id,
                            "memory_id": memory.id,
                            "error": str(exc),
                        },
                    )
            if deleted_ids:
                await self._fixup_insight_sources(
                    deleted_ids=deleted_ids, summary_id=summary_memory.id
                )
            produced += 1

        if produced > 0:
            logger.info(
                "memory_compression",
                extra={
                    "agent_id": self._agent_id,
                    "stream": stream.value,
                    "summaries_created": produced,
                    "candidates": len(candidates),
                },
            )
        return produced

    async def candidates_for_reflection(
        self,
        *,
        current_step: int,
        lookback_steps: int,
        max_candidates: int,
        max_depth: int,
    ) -> list[Memory]:
        """EXPERIENTIAL events and insights for Reflection, ranked by importance × (|valence| + 0.1).

        Warms _entries from the store first, so it works after a restore. Only events are
        windowed; insights are lasting beliefs. reflection_depth must be < max_depth.
        """
        collection = self._namespace(MemoryStream.EXPERIENTIAL)
        try:
            results = await self._vector_store.list_all(collection)
        except Exception as exc:
            logger.warning(
                "reflection_candidates_list_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return []
        for result in results:
            self._memory_from_result(result)  # warms _entries (side effect)

        event_threshold = current_step - lookback_steps
        candidates = [
            memory
            for memory in self._entries.values()
            if memory.stream == MemoryStream.EXPERIENTIAL
            and memory.kind in (MemoryKind.EVENT, MemoryKind.INSIGHT)
            and memory.reflection_depth < max_depth
            and (memory.kind == MemoryKind.INSIGHT or memory.created_step >= event_threshold)
        ]
        candidates.sort(
            key=lambda m: float(m.importance) * (abs(m.emotion_valence) + 0.1),
            reverse=True,
        )
        return candidates[:max_candidates]

    def build_insight(
        self,
        *,
        current_step: int,
        text: str,
        related_agents: Sequence[str],
        importance: float,
        emotion_valence: float,
        source_ids: Sequence[str],
        reflection_depth: int,
    ) -> Memory:
        """Build an insight Memory with an id from the monotonic counter; the caller persists it.

        Ids come from here because len(_entries) drops after deletions and would collide.
        """
        return Memory(
            id=f"{self._agent_id}-insight-{current_step}-{self._next_seq()}",
            agent_id=self._agent_id,
            stream=MemoryStream.EXPERIENTIAL,
            raw_content=text,
            stored_content=text,
            importance=importance,
            created_step=current_step,
            metadata={"world_id": self._world_id, "kind": MemoryKind.INSIGHT.value},
            emotion_valence=emotion_valence,
            emotion_label="insight",
            related_agents=list(related_agents),
            triggered_by="reflection",
            decay_score=1.0,
            kind=MemoryKind.INSIGHT,
            source_ids=list(source_ids),
            reflection_depth=reflection_depth,
        )

    async def _fixup_insight_sources(
        self, *, deleted_ids: set[str], summary_id: str
    ) -> None:
        """Re-point insights' source_ids from compressed sources to the new summary (deduplicated),
        so an insight follows its evidence instead of dangling."""
        collection = self._namespace(MemoryStream.EXPERIENTIAL)
        try:
            all_results = await self._vector_store.list_all(collection)
        except Exception as exc:
            logger.warning(
                "insight_source_fixup_list_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return
        for result in all_results:
            memory = self._memory_from_result(result)
            if memory is None or memory.kind != MemoryKind.INSIGHT:
                continue
            if not memory.source_ids:
                continue
            if not any(sid in deleted_ids for sid in memory.source_ids):
                continue
            new_sources: list[str] = []
            seen: set[str] = set()
            replaced = False
            for sid in memory.source_ids:
                if sid in deleted_ids:
                    if summary_id not in seen:
                        new_sources.append(summary_id)
                        seen.add(summary_id)
                    replaced = True
                else:
                    if sid not in seen:
                        new_sources.append(sid)
                        seen.add(sid)
            if replaced:
                memory.source_ids = new_sources
                try:
                    await self._persist_payload(memory)
                except Exception as exc:
                    logger.warning(
                        "insight_source_fixup_persist_failed",
                        extra={"agent_id": self._agent_id, "memory_id": memory.id, "error": repr(exc)},
                    )

    async def _summarize_cluster_for_stream(
        self, memories: Sequence[Memory], *, stream: MemoryStream
    ) -> str | None:
        """Summarize a cluster of aged FACTUAL memories into one objective summary via the LLM.

        No rule-based fallback: concatenating and truncating would corrupt the content. Failure →
        None, and the caller keeps the originals.
        """
        time_range = self._span_time_range(memories)
        # Oldest first, so the prompt's MEMORY_ORDER_HINT holds.
        contents = "\n".join(
            f"- {memory.stored_content}" for memory in order_memories_chrono(memories)
        )
        system_prompt, user_prompt = self._build_summary_prompt(
            stream, time_range=time_range, contents=contents
        )
        # Not via _complete_memory_prompt (the prose-only monologue channel): this call parses JSON,
        # so it must request json_mode itself (see test_json_mode).
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        try:
            # reason (≤100 chars ≈150) + points (at most five, each ≤40 chars ≈60 + 5 structure)
            response = await self._llm_router.complete(
                LLMScene.MEMORY_SUMMARIZATION, messages, max_tokens=output_budget(490), json_mode=True,
            )
            if _is_unusable_llm_content(response.content):
                return None
            # extract_json raises; uncaught, one bad cluster would abort the whole compression round.
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "memory_summary_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return None
        if not isinstance(data, dict):
            return None
        # Persist points only; reason is dropped so the reasoning is never embedded.
        points = data.get("points")
        if not isinstance(points, list):
            return None
        cleaned = [str(p).strip() for p in points if str(p).strip()]
        if not cleaned:
            return None
        return "\n".join(f"- {p}" for p in cleaned)

    def _span_time_range(self, memories: Sequence[Memory]) -> str:
        """The cluster's span as a world-calendar range "<start> 至 <end>"; "" without a clock."""
        if self._time_label_for is None:
            return ""
        start_step = min(m.created_step for m in memories)
        end_step = max(m.created_step for m in memories)
        start_label = self._time_label_for(start_step)
        if start_step == end_step:
            return start_label
        return f"{start_label} 至 {self._time_label_for(end_step)}"

    def _build_summary_prompt(
        self, stream: MemoryStream, *, time_range: str, contents: str
    ) -> tuple[str, str]:
        """Compression summary prompt (factual only; ``stream`` is unused). Invariant role, task,
        constraints and schema go in system for the prefix cache; the records go in user."""
        period = f"（{time_range}）" if time_range else ""
        system = """\
【角色】你是一个客观的记忆归档处理器，把同一时期的若干事实记录精简归档。

【任务】先在心里判断：哪些记录重复、可合并、可提炼成更概括的一条；再据此压缩。
方法是**精简、去冗余、合并同类、提炼共性**——不是改写成故事，是把零散事实压实。
把这段时期收拢成**尽量少的几条**要点（能一条说清就只写一条，至多三五条），逐条写进 points；
每条都要短。**越精简越好**。

【约束】
- 保持客观第三人称陈述，忠实于记录：不渲染、不评判、不替任何人代入情绪。
- 只做精简合并，丢弃无用细节，保留 who / when / where / general what。
- 不得引入记录中未出现的人物或事件。
""" + KEEP_ABSOLUTE_TIME_RULE + """

【输出】严格输出以下 JSON，不要任何多余内容（reason 仅供你先想后写，会被忽略；摘要内容全在 points）：
{"reason": "哪些可合并/可提炼（一句，≤100字）", "points": ["简短的一条，≤40字", "..."]}"""
        user = f"""\
【输入】这段时期{period}的若干观察记录（客观、无上下文，{MEMORY_ORDER_HINT}）：
{contents}

依【角色/任务/约束】把上述记录压实，严格按约定的 JSON 输出，不写任何多余内容。"""
        return system, user

    async def _persist(self, memory: Memory) -> bool:
        """Cache insert + embed + upsert, for new or changed text (unchanged text uses
        _persist_payload). Returns whether it was persisted; only build-time callers should read it."""
        self._entries[memory.id] = memory
        return await self._persist_vector(memory)

    async def _persist_payload(self, memory: Memory) -> None:
        """Write back a memory whose text is unchanged (touch, decay) without re-embedding. Falls
        back to a full embed + upsert only if the record is missing from the store."""
        self._entries[memory.id] = memory
        try:
            updated = await self._vector_store.update_payload(
                collection=self._namespace(memory.stream),
                id=memory.id,
                payload=self._memory_to_payload(memory),
            )
        except Exception as exc:  # noqa: BLE001 — Rule 1: a transient failure only loses this write-back
            logger.warning(
                "memory_payload_update_failed",
                extra={"agent_id": self._agent_id, "memory_id": memory.id, "error": repr(exc)},
            )
            return
        if not updated:
            await self._persist_vector(memory)

    async def _persist_vector(self, memory: Memory) -> bool:
        """Embed and upsert only; the caller owns _entries. Returns whether it was persisted:
        runtime callers carry on regardless (Rule 1), seed_factual_memory raises (Rule 2)."""
        _t_embed = time.perf_counter()
        try:
            embedding = await self._embedding.embed(memory.stored_content)
            embed_ms = round((time.perf_counter() - _t_embed) * 1000, 1)
            _t_upsert = time.perf_counter()
            await self._vector_store.upsert(
                collection=self._namespace(memory.stream),
                id=memory.id,
                dense_vector=embedding.dense,
                sparse_vector=embedding.sparse,
                payload=self._memory_to_payload(memory),
            )
        except Exception as exc:  # noqa: BLE001 — Rule 1: a transient failure loses only this persist
            # The memory stays usable in _entries; only its embedding recall is lost. One 429 from
            # the embedding API must not kill the world loop.
            # repr, not str: a timeout's str is empty.
            logger.warning(
                "memory_persist_failed",
                extra={"agent_id": self._agent_id, "memory_id": memory.id, "error": repr(exc)},
            )
            return False
        upsert_ms = round((time.perf_counter() - _t_upsert) * 1000, 1)
        logger.debug(
            "memory_persist",
            extra={
                "agent_id": self._agent_id,
                "stream": memory.stream.value,
                "importance": round(float(memory.importance), 3),
                "embed_ms": embed_ms,
                "upsert_ms": upsert_ms,
                "content_len": len(memory.stored_content),
            },
        )
        return True

    def _namespace(self, stream: MemoryStream) -> str:
        return memory_collection_name(self._world_id, self._agent_id, stream)

    def _memory_to_payload(self, memory: Memory) -> dict[str, Any]:
        return {
            "id": memory.id,
            "world_id": self._world_id,
            "agent_id": memory.agent_id,
            "stream": memory.stream.value,
            "raw_content": memory.raw_content,
            "stored_content": memory.stored_content,
            "text": memory.stored_content,
            "importance": float(memory.importance),
            "created_step": memory.created_step,
            "step": memory.created_step,
            "metadata_json": json.dumps(memory.metadata, ensure_ascii=False),
            "emotion_valence": memory.emotion_valence,
            "emotion_label": memory.emotion_label,
            # A real list, not a JSON string: identity-scoped retrieval needs the store's membership
            # predicate (filters={"related_agents": <id>}).
            "related_agents": list(memory.related_agents),
            "triggered_by": memory.triggered_by or "",
            "last_accessed_step": memory.last_accessed_step,
            "decay_score": memory.decay_score,
            "retrieval_count": memory.retrieval_count,
            "kind": memory.kind.value,
            "event_group_id": memory.event_group_id or "",
            "source_ids_json": json.dumps(memory.source_ids, ensure_ascii=False),
            "reflection_depth": memory.reflection_depth,
        }

    def _memory_from_result(self, result: SearchResult) -> Memory | None:
        if result.id in self._entries:
            return self._entries[result.id]
        payload = result.payload
        if not payload:
            return None
        try:
            kind_raw = str(payload.get("kind") or "")
            kind = MemoryKind(kind_raw) if kind_raw in {k.value for k in MemoryKind} else MemoryKind.EVENT
            memory = Memory(
                id=str(payload.get("id") or result.id),
                agent_id=str(payload.get("agent_id") or self._agent_id),
                stream=MemoryStream(str(payload.get("stream"))),
                raw_content=str(payload.get("raw_content") or payload.get("text") or ""),
                stored_content=str(payload.get("stored_content") or payload.get("text") or ""),
                importance=coerce_importance(float(payload.get("importance", 0.5))),
                created_step=int(payload.get("created_step", payload.get("step", 0))),
                metadata=_loads_dict(payload.get("metadata_json")),
                emotion_valence=float(payload.get("emotion_valence", 0.0)),
                emotion_label=str(payload.get("emotion_label") or EmotionType.NEUTRAL.value),
                related_agents=_loads_list(payload.get("related_agents")),
                triggered_by=str(payload.get("triggered_by") or "") or None,
                last_accessed_step=int(payload.get("last_accessed_step", -1)),
                decay_score=float(payload.get("decay_score", 1.0)),
                retrieval_count=int(payload.get("retrieval_count", 0)),
                kind=kind,
                event_group_id=str(payload.get("event_group_id") or "") or None,
                source_ids=_loads_list(payload.get("source_ids_json")),
                reflection_depth=int(payload.get("reflection_depth", 0)),
            )
        except (TypeError, ValueError):
            return None
        self._entries[memory.id] = memory
        return memory

    async def _write_experiential(
        self,
        *,
        raw_content: str,
        personality: PersonalityLayer,
        related_agent_directory: dict[str, str] | None = None,
        related_relations_text: str = "",
        situation: Situation = Situation(),
        recent_factual_block: str = "",
        now_step: int = 0,
    ) -> str:
        """Rewrite an event as a first-person monologue shaped by personality and current emotion.

        It only interprets and feels, and doesn't retell: FACTUAL already holds the objective
        account, and retelling would make the streams overlap.

        - ``related_agent_directory``: the closed set of people the monologue may mention.
        - ``recent_factual_block``: continuity background, not to be retold; "" omits the section.
        Returns "" on failure.
        """
        soul = personality.soul
        emotion = personality.state.emotion
        # life_goal, not state.long_term_goals: planning detail doesn't help "what did this touch in me".
        life_goal = soul.life_goal or "无"
        secret_line = f"\n我的{SECRET_LABEL}：{soul.secret}" if soul.secret else ""

        people_block = ""
        if related_agent_directory:
            names = render_related_people(related_agent_directory)
            people_block = f"【涉及相关人】只能提及：{names}（不得引入未列出的人）\n\n"
        relation_block = ""
        if related_relations_text:
            relation_block = f"【我与他们的关系】\n{related_relations_text}\n\n"
        # Placed right before "【刚发生的事】", which must stay next to the generation point to
        # outweigh the background.
        recent_factual_context_block = ""
        if recent_factual_block:
            recent_factual_context_block = (
                "【此前近来发生的事（仅供衔接理解，不要复述、不要当作刚发生的事）】\n"
                f"（{MEMORY_ORDER_HINT}）\n"
                f"{recent_factual_block}\n\n"
            )

        situation_header = render_situation_header(situation, voice=SituationVoice.FIRST)
        # Without the condition a bound character writes "I clenched my fist", and that memory
        # persists.
        condition_block = condition_line(
            personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        system_prompt = """\
你此刻完全代入一个角色，以第一人称「我」把刚发生的事写进心里。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【我的主观理解与感受】
用第一人称凝练成一段内心独白（一段、不分点），只写两件事：
 1. **我怎么理解刚发生的事**：带着我的性格、价值观、底线、追求，和此刻这般情绪等，去主观理解刚发生的这件事——
    我从中读到了什么、最在意的是哪一点、它在我眼里到底算怎么回事。
 2. **我的感受**：在这般解读之下，我心里的真实感受是什么？。

【约束】
 - 不提及【涉及相关人】之外的任何角色。
 - 不贴标签、不喊口号、不堆套话；不跳出"我"。
 - 提到我认识的人时,不妨带上其名字便于日后回想;若我惯以称谓或「他/她」相称,可把名字附在其后(形如「称谓（名字）」)。这只是便于回想的建议,不必生硬套用、更不必每处都加括号;陌生人用描述,不臆造名字。
 - 如果【刚发生的事】实在没什么可由我解读、可记的，可以什么都不写。什么都不写比写无意义的内容更加有价值。如果【刚发生的事】让我感觉到困惑，我一定想想为什么会这样。
 - **不发空泛的感慨**：解读要死死咬住这件具体的事、落在我具体在意的那一点上；不要借题发挥
   讲人生哲理、不要把它抽象拔高成一通玄虚的大道理（这是最常见的跑偏）。
 - **变深浅**：戳中我的事浓墨写，温和的事淡淡记，平淡的事如实平淡。宁可写得近似客观陈述，
   也绝不为显得深刻而无中生有地拔高情绪或意义（那是涌现叙事的毒药）。
 - 不要过度脑补没有发生过的事情，宁可直接复述刚刚发生过的事情，也不要胡编乱造。
 - 要区分已经发生、正在发生、将要发生的事情：把“我现在要去超市”写成“我去了超市”、把“我命令他帮我收拾行李”写成“他已经帮我收拾行李”，都是禁止的。
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + "\n" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """
 - **这段独白只是关于【刚发生的事】的**：如果给出了【此前近来发生的事】，那只是为了让我
   理解自己此刻的心境不至断裂，是背景衔接；**不要**把它们当作"刚发生"来复述、评论、或
   当作独白的对象——只有【刚发生的事】才是我此刻要在心里落笔的那件事。

【格式】一段，不超过 200 字（宁可更短，不必写满）；只输出独白本身，不要任何说明。"""
        user_prompt = f"""\
{situation_header}

【我是谁】
我是{soul.name}。
核心性格：{soul.traits_text()}
核心价值观：{soul.values_text()}
行为底线：{soul.constraints_text()}{secret_line}
自我认知：{soul.self_image or '无'}
毕生追求：{life_goal}

【此刻我的心绪】
{emotion.summary()}{condition_block}

{people_block}{relation_block}{recent_factual_context_block}【刚发生的事（客观经过）】
{raw_content}

我依上面说定的规矩与格式，写出我此刻的内心独白，只输出独白本身，不写任何多余内容。"""
        # Callers may not have set per-agent context, so tag it explicitly.
        with observe_stage(Stage.MEMORY, agent_id=self._agent_id):
            # The monologue's "不超过 200 字" (≤200 chars) ≈ 300 tok (including possible bracketed
            # dates)
            content = await self._complete_memory_prompt(
                system_prompt, user_prompt, max_tokens=output_budget(300),
            )
        if _is_unusable_llm_content(content):
            # Don't fall back to raw: it would give the same event two slots in recall.
            return ""
        return content

    async def _complete_memory_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
    ) -> str:
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        try:
            # Prose-only channel for the monologue. A call that parses its reply must not route
            # through here: that would hide its decoding mode from the guard in test_json_mode.
            response = await self._llm_router.complete(
                LLMScene.MEMORY_SUMMARIZATION,
                messages,
                max_tokens=max_tokens,
            )
            return response.content.strip()
        except Exception as exc:
            logger.warning(
                "memory_llm_call_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return ""


def _is_unusable_llm_content(content: str) -> bool:
    lowered = content.strip().lower()
    return not lowered or lowered in {"mock response", "default mock response"}


def _loads_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _loads_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        loaded = json.loads(value)
        return [str(item) for item in loaded] if isinstance(loaded, list) else []
    return []


def _is_emotional_anchor(memory: Memory) -> bool:
    """CRITICAL, insights and high-|valence| experiential are never decayed or compressed."""
    if importance_level(memory.importance) == MemoryImportance.CRITICAL:
        return True
    if memory.kind == MemoryKind.INSIGHT:
        return True
    if (
        memory.stream == MemoryStream.EXPERIENTIAL
        and abs(memory.emotion_valence) >= EMOTIONAL_ANCHOR_VALENCE
    ):
        return True
    return False


def _is_compression_candidate(memory: Memory, *, current_step: int) -> bool:
    """Only unimportant, faded, emotionally flat, old events that aren't emotional anchors."""
    if memory.kind != MemoryKind.EVENT:
        return False
    if _is_emotional_anchor(memory):
        return False
    if float(memory.importance) >= 0.4:
        return False
    if memory.decay_score >= 0.2:
        return False
    if abs(memory.emotion_valence) >= 0.5:
        return False
    if (current_step - memory.created_step) < MIN_AGE_STEPS:
        return False
    return True


def _cluster_for_compression(candidates: Sequence[Memory]) -> list[list[Memory]]:
    """Bucket by COMPRESSION_TIME_WINDOW_STEPS window, then by first related agent ("_solo" if
    none); groups over 10 are split in time order."""
    by_window: dict[int, dict[str, list[Memory]]] = {}
    for memory in sorted(candidates, key=lambda m: m.created_step):
        window = memory.created_step // COMPRESSION_TIME_WINDOW_STEPS
        key = sorted(memory.related_agents)[0] if memory.related_agents else "_solo"
        by_window.setdefault(window, {}).setdefault(key, []).append(memory)
    clusters: list[list[Memory]] = []
    for window_groups in by_window.values():
        for group in window_groups.values():
            if len(group) <= 10:
                clusters.append(group)
            else:
                for start in range(0, len(group), 10):
                    chunk = group[start : start + 10]
                    if len(chunk) >= 3:
                        clusters.append(chunk)
    return clusters


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
