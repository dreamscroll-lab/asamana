"""Reflection layer: distills recent experiential memories into insights.

See ReflectionEngine for the contract (what is generated directly vs changed indirectly, input
filter, grounding, depth cap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from agent.memory import MemorySystem
from agent.memory_types import Memory, MemoryKind
from agent.personality import PersonalityLayer
from core.coerce import coerce_float
from core.interfaces.llm import IndexedRef, LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.interfaces.perception import Situation
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    MEMORY_ORDER_HINT,
    SituationVoice,
    order_memories_chrono,
    render_memory,
    render_situation_header,
)


logger = get_logger(__name__)


DEFAULT_LOOKBACK_STEPS = 30
DEFAULT_INSIGHT_CANDIDATES = 10
INSIGHT_BASE_IMPORTANCE = 0.6
INSIGHT_CONFIDENCE_BONUS_PER_SOURCE = 0.05
MAX_INSIGHT_IMPORTANCE = 0.9
# event (0) → insight v1 (1) → insight v2 (2); deeper ones are not candidates. Three levels cover
# "first judgment → confirmation → revision"; higher abstractions add little.
MAX_REFLECTION_DEPTH = 2


@dataclass
class ReflectionResult:
    """The insight memories written by one reflection."""

    new_insights: List[Memory] = field(default_factory=list)


class ReflectionEngine:
    """Reflection: where personality change in the narrative happens.

    From the agent's first-person view, it turns what stayed with them (high importance /
    strong emotion) into insight memories about a person, a relationship or an event; the
    subjective counterpart of Compression (which summarizes aged, low-importance memory for all
    agents and deletes its sources; reflection keeps sources as grounds).

    Only observable things are judged directly. Self-image, values and personality are never
    asserted ("I've become braver" has no anchor and drifts); they reshape indirectly as grounded
    insights are retrieved into later prompts. Change in the self goes through understanding of
    the world.

    Insights rank high without an extra scoring multiplier (that would double-count): they're
    written with high importance (base 0.6 + per-source bonus), protected from decay by
    _is_emotional_anchor, and their source events fold in as insight_sources rather than taking
    separate slots in _compose_retrieval_result.

    Contract:
    1. Input: EXPERIENTIAL; kind event or insight (summaries have lost their detail);
       reflection_depth < MAX_REFLECTION_DEPTH.
    2. Grounding: every insight cites at least one event; beliefs derived only from old insights
       drift from reality and are dropped.
    3. Sources are cited by index (IndexedRef); an insight with no valid source is discarded.
    4. importance grows slightly with len(source_ids).
    5. Personality is never modified directly: an LLM rewriting traits, with no cumulative cap,
       consistency check or cooldown, piles up contradictory, hard-to-debug changes.
    6. A failed LLM call returns an empty ReflectionResult (Rule 1); never raises.
    7. When Compression deletes a cited memory, MemorySystem._fixup_insight_sources repoints
       source_ids to the summary.
    8. A new insight never deletes an old one; the newer usually wins on importance × recency,
       but strong relevance can still bring back "the simple view he once held".
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        memory_system: MemorySystem,
        personality: PersonalityLayer,
        *,
        agent_id: str,
        lookback_steps: int = DEFAULT_LOOKBACK_STEPS,
        max_candidates: int = DEFAULT_INSIGHT_CANDIDATES,
        seconds_per_step: int = 3600,
        time_label_for: Callable[[int], str] | None = None,
        world_start_second_of_day: int = 0,
    ) -> None:
        self._llm_router = llm_router
        self._memory_system = memory_system
        self._personality = personality
        self._agent_id = agent_id
        self._lookback_steps = lookback_steps
        self._max_candidates = max_candidates
        self._seconds_per_step = seconds_per_step
        # Step → world-calendar time, for dated judgments. Without it the header is omitted, and
        # the prompt says not to write dates when no time is given.
        self._time_label_for = time_label_for
        # "today/yesterday" counts midnights crossed, which duration alone can't give.
        self._world_start_second_of_day = world_start_second_of_day

    async def reflect(self, current_step: int) -> ReflectionResult:
        """Run one reflection cycle. On failure, returns an empty result per Rule 1; never raises."""

        candidates = await self._memory_system.candidates_for_reflection(
            current_step=current_step,
            lookback_steps=self._lookback_steps,
            max_candidates=self._max_candidates,
            max_depth=MAX_REFLECTION_DEPTH,
        )
        if not candidates:
            return ReflectionResult()

        # Oldest first, to show patterns over time. ref and _build_prompt share this list, so
        # indices match what is shown.
        candidates = order_memories_chrono(candidates)
        ref = IndexedRef([memory.id for memory in candidates])
        system_prompt, user_prompt = self._build_prompt(candidates, current_step=current_step)
        try:
            response = await self._llm_router.complete_with_retry(
                LLMScene.MEMORY_REFLECTION,
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                temperature=0.6,
                # insights[]: one item = text (≤150 chars ≈ 225) + importance (3) + source_indices
                # (~10) + structure (10) ≈ 248 tok; the prompt's cap of 4 items
                max_tokens=output_budget(992),
            json_mode=True,
            )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "reflection_llm_failed",
                extra={"agent_id": self._agent_id, "step": current_step, "error": str(exc)},
            )
            return ReflectionResult()

        insights_payload = data.get("insights") if isinstance(data, dict) else None
        if not isinstance(insights_payload, list):
            return ReflectionResult()

        candidate_by_id = {memory.id: memory for memory in candidates}
        new_insights: List[Memory] = []
        for raw in insights_payload:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            raw_indices = raw.get("source_indices") or []
            if not isinstance(raw_indices, list):
                continue
            filtered_sources = ref.resolve(raw_indices)
            if not filtered_sources:
                continue

            # Grounding (contract 2).
            source_memories = [candidate_by_id[sid] for sid in filtered_sources]
            if not any(s.kind == MemoryKind.EVENT for s in source_memories):
                logger.debug(
                    "reflection_insight_dropped_no_event_grounding",
                    extra={
                        "agent_id": self._agent_id,
                        "step": current_step,
                        "source_ids": filtered_sources,
                    },
                )
                continue

            base_score = coerce_float(
                raw.get("importance"), default=INSIGHT_BASE_IMPORTANCE, minimum=0.0, maximum=1.0
            )
            importance = min(
                MAX_INSIGHT_IMPORTANCE,
                base_score + INSIGHT_CONFIDENCE_BONUS_PER_SOURCE * len(filtered_sources),
            )
            related_set: set[str] = set()
            valences: list[float] = []
            for source in source_memories:
                related_set.update(source.related_agents)
                valences.append(source.emotion_valence)
            avg_valence = sum(valences) / len(valences) if valences else 0.0
            # depth = max(source.depth) + 1. It stays ≤ MAX_REFLECTION_DEPTH because
            # candidates_for_reflection only returns sources with depth < MAX.
            new_depth = max((s.reflection_depth for s in source_memories), default=0) + 1

            insight_memory = await self._write_insight(
                current_step=current_step,
                text=text,
                source_ids=filtered_sources,
                related_agents=sorted(related_set),
                importance=importance,
                emotion_valence=avg_valence,
                reflection_depth=new_depth,
            )
            if insight_memory is not None:
                new_insights.append(insight_memory)

        if new_insights:
            logger.info(
                "reflection_produced_insights",
                extra={
                    "agent_id": self._agent_id,
                    "step": current_step,
                    "insights": len(new_insights),
                },
            )
        return ReflectionResult(new_insights=new_insights)

    def _build_prompt(
        self, candidates: Sequence[Memory], *, current_step: int
    ) -> tuple[str, str]:
        # Each candidate gets a number, an [event]/[insight] tag and relative recency (events, via
        # render_memory); the LLM returns source_indices.
        listing = "\n".join(
            f"#{i + 1} [{memory.kind.value}] "
            f"{render_memory(memory, now_step=current_step, seconds_per_step=self._seconds_per_step, world_start_second_of_day=self._world_start_second_of_day)}"
            for i, memory in enumerate(candidates)
        )
        n = len(candidates)
        # Only the time: looking back doesn't depend on where I am. Omitted when there is no clock.
        _now_label = self._time_label_for(current_step) if self._time_label_for else ""
        time_header = (
            f"{render_situation_header(Situation(time_label=_now_label), voice=SituationVoice.FIRST)}\n\n"
            if _now_label else ""
        )
        # include_emotion=True: reflection doesn't generate emotion, so mood is a legitimate input.
        # The count n varies per call, so it goes in the user prompt, not system.
        system_prompt = f"""\
你此刻完全代入一个角色，以第一人称「我」把近来留在心里的事回看一遍、写出真正新的认知。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【此刻的我】我带着此刻的心绪停下来，回看近来留在心里的这些事——想看清我对**某个人、
某段关系、某件事，或我自己**，形成了什么新的认知，把原先的看法又看深了一层，或干脆推翻了它。
我此刻的心情会染色我怎么解读它们，这是真实的，不必假装中立。
两条底线：
 一、每一点体悟都必须扎在至少一件真正发生过的事（[event]）上——凭空的大道理，我自己也不信；
     哪怕是关于"我自己"的认知，也得落在具体的人与事上，不空喊"我变了"。
 二、**若回看下来这段时间确实没什么真正值得记下的新认知，我就直接交白卷（insights 给空数组）**——
     平淡的日子本就未必有什么了悟，硬挤一条连我自己都不信的洞察，远不如什么都不写。

【我先问自己】（想过再写，不必把答案写出来）：
- 这些事里，哪个人、哪件事最戳中我？我对他/它，看清了什么？
- 它为什么留在我心里、动摇或印证了我原先的什么看法？
- 有哪条旧念（[insight]）该被这次的经历修正、加深、甚至推翻？
- 有哪点是值得我从此记牢？
- 我记着的这些事里，有没有对不上的地方——日子前后颠倒，或某个我记得已经死了的人又出现？
  若有，我要想清楚哪一版才是真的。

【我落笔时避开】：
- 不贴标签、不喊口号、不堆套话——只说我真切想明白的。
- 不跳出"我"，不写成旁人对我的评点；也不空喊"我变了"——认知是落在人和事上的，
  我是什么样的人，留给这些认知日后自己显形。
- 不无中生有：只谈「我回看的素材」里真有的人和事。
- 输出内容不要啰嗦，冗余，流水，尽量简洁，精炼。
{ABSOLUTE_TIME_RULE_FIRST_PERSON}

【写出来】严格输出以下 JSON，不要任何多余内容。每条体悟都**先落到它扎根的那几件事上**（source_indices：「我回看的素材」里对应的整数编号，每条至少含一个 [event]），**再由这些事写出我想明白了什么**（text，用我的口吻）——认知是从具体的人和事里长出来的，不是先想出一句漂亮话、再回头翻编号给它找脚注；insights 最多 4 条、能少则少，**若确无可写，insights 直接给空数组 []**：
{{"insights": [{{"source_indices": [..], "text": "一条体悟，≤150字", "importance": 0.0到1.0}}]}}"""
        user_prompt = f"""\
{time_header}【这是我】
{self._personality.to_prompt_context(include_goals=False,include_emotion=True)}

【我回看的素材】（每条以 #N 编号，标着 [event] 亲历 或 [insight] 旧念；{MEMORY_ORDER_HINT}；共 {n} 条，source_indices 只能填 1 到 {n} 的整数）：
{listing}

我依上面说定的规矩与 JSON 格式写下此刻的新认知，只输出 JSON、不写任何多余内容。"""
        return system_prompt, user_prompt

    async def _write_insight(
        self,
        *,
        current_step: int,
        text: str,
        source_ids: Sequence[str],
        related_agents: Sequence[str],
        importance: float,
        emotion_valence: float,
        reflection_depth: int = 1,
    ) -> Memory | None:
        """Build an insight Memory and persist it, bypassing write() and its LLM rewrite.
        The caller computes reflection_depth from the sources."""
        memory = self._memory_system.build_insight(
            current_step=current_step,
            text=text,
            related_agents=related_agents,
            importance=importance,
            emotion_valence=emotion_valence,
            source_ids=source_ids,
            reflection_depth=reflection_depth,
        )
        try:
            await self._memory_system._persist(memory)
            return memory
        except Exception as exc:
            logger.warning(
                "reflection_insight_persist_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return None
