"""Unit tests for the agent memory subsystem."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agent.memory import MemorySystem
from agent.memory_ranking import (
    RETRIEVAL_WEIGHTS, VECTOR_WEIGHTS, _minmax, _raw_recency, score_and_rank,
)
from agent.memory_types import (
    Memory, MemoryImportance, MemoryStream, RetrievalResult, coerce_importance, importance_level,
)
from agent.personality import EmotionState, PersonalityLayer, SoulLayer, StateLayer
from core.context import clear_log_context, set_log_context
from core.interfaces.llm import LLMProvider, LLMRouter, LLMScene
from core.interfaces.vector_store import SearchResult
from providers.llm.mock import MockLLMProvider, SequentialMockLLM
from providers.trace import InMemoryTraceSink


def _hit(relevance: float) -> SearchResult:
    """A vector hit. score_and_rank uses the fused score as relevance."""
    return SearchResult(id="", score=relevance, payload={}, dense_score=relevance, sparse_score=0.0)


def _make_personality(
    *,
    name: str = "Test Agent",
    core_traits: list[str] | None = None,
    self_image: str = "",
    emotion: EmotionState | None = None,
) -> PersonalityLayer:
    soul = SoulLayer(
        name=name,
        agent_id="agent-1",
        core_traits=core_traits or ["observant"],
        self_image=self_image,
    )
    state = StateLayer(emotion=emotion or EmotionState(primary="neutral", intensity=0.2, valence=0.0))
    return PersonalityLayer(soul=soul, state=state)


def _router(provider: LLMProvider) -> LLMRouter:
    return LLMRouter({scene: provider for scene in LLMScene})


def _make_memory_system(
    container: object,
    *,
    provider: LLMProvider | None = None,
    world_id: str = "world-1",
    agent_id: str = "agent-1",
    is_main_character: bool = False,
    time_label_for: "Callable[[int], str] | None" = None,
    retrieval_score_floor: float = -1.0,
) -> MemorySystem:
    return MemorySystem(
        _router(provider or MockLLMProvider()),
        container.embedding,  # type: ignore[attr-defined]
        container.vector_store,  # type: ignore[attr-defined]
        world_id=world_id,
        agent_id=agent_id,
        is_main_character=is_main_character,
        time_label_for=time_label_for,
        # Default -1.0 sits below any cosine. Tests use the in_memory hash embedding, whose scores
        # are random and can be negative, so any floor would drop "relevant" memories at random.
        # Floor tests pass a non-negative threshold and hand-built vectors.
        retrieval_score_floor=retrieval_score_floor,
    )


@pytest.mark.asyncio
async def test_write_factual_stores_raw_content_verbatim(container: object) -> None:
    """FACTUAL memories skip the LLM: stored_content == raw_content."""
    provider = SequentialMockLLM(["unused"])
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality(core_traits=["suspicious"])

    memory = await system.write(
        "The gate was left open.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=1,
    )

    assert memory is not None
    assert memory.raw_content == "The gate was left open."
    assert memory.stored_content == "The gate was left open."
    assert memory.emotion_label == "objective"


@pytest.mark.asyncio
async def test_write_experiential_uses_llm_coloring(container: object) -> None:
    provider = SequentialMockLLM(["This felt like another warning that my place was fragile."])
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality(
        self_image="I am the rightful heir.",
        emotion=EmotionState(primary="jealousy", intensity=0.8, valence=-0.7),
    )

    memory = await system.write(
        "Li Shimin was praised in court.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        importance=MemoryImportance.HIGH,
        current_step=2,
    )

    assert memory is not None
    assert memory.stored_content != memory.raw_content
    assert "fragile" in memory.stored_content
    assert memory.emotion_valence == -0.7


@pytest.mark.asyncio
async def test_sample_recent_events_pairs_streams_and_includes_factual_only(container: object) -> None:
    """sample_recent_events feeds short-term goals from both streams: it pairs factual with
    experiential by event_group_id, includes factual-only mundane events as objective anchors,
    drops events outside the window, and sorts by importance."""
    provider = SequentialMockLLM(["我把它读作一次当众的羞辱。"])  # experiential write LLM
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()
    # Strong event with an interpretation: factual + experiential in one group.
    await system.write("陛下当众斥责我。", MemoryStream.FACTUAL, personality=personality,
                       importance=MemoryImportance.HIGH, current_step=5, event_group_id="g1")
    await system.write("陛下当众斥责我。", MemoryStream.EXPERIENTIAL, personality=personality,
                       importance=MemoryImportance.HIGH, current_step=5, event_group_id="g1")
    # Mundane event: factual only, no interpretation.
    await system.write("我用了早膳。", MemoryStream.FACTUAL, personality=personality,
                       importance=MemoryImportance.LOW, current_step=5, event_group_id="g2")
    await system.write("旧事一桩。", MemoryStream.FACTUAL, personality=personality,
                       importance=MemoryImportance.HIGH, current_step=0, event_group_id="g0")

    events = system.sample_recent_events(current_step=6, lookback=3, top_k=10)
    by_group = {(f or e).event_group_id: (f, e) for f, e in events}
    assert "g0" not in by_group
    assert by_group["g1"][0] is not None and by_group["g1"][1] is not None
    assert by_group["g2"][0] is not None and by_group["g2"][1] is None      # mundane events are kept as anchors
    order = [(f or e).event_group_id for f, e in events]
    assert order.index("g1") < order.index("g2")


@pytest.mark.asyncio
async def test_record_event_creates_dual_streams_with_event_group_id(container: object) -> None:
    """The factual and experiential entries of one event share event_group_id."""
    provider = SequentialMockLLM(["A messenger arrived.", "I felt alarmed by the messenger."])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality(
        emotion=EmotionState(primary="fear", intensity=0.6, valence=-0.4),
    )

    entries = await system.record_event(
        current_step=2,
        raw_content="A messenger arrived with urgent news.",
        experiential_content="The urgent news unsettled me.",
        personality=personality,
        importance=MemoryImportance.HIGH,
    )

    streams = {memory.stream for memory in entries}
    assert streams == {MemoryStream.FACTUAL, MemoryStream.EXPERIENTIAL}
    group_ids = {memory.event_group_id for memory in entries}
    assert len(group_ids) == 1 and group_ids != {None}


@pytest.mark.asyncio
async def test_record_event_skips_experiential_when_rewrite_fails(container: object) -> None:
    """If the rewrite LLM fails, keep only factual. Don't store the raw text as the inner stream,
    or the same event takes two slots in recall."""
    system = _make_memory_system(container, is_main_character=True)  # default mock reply is unusable
    personality = _make_personality(
        emotion=EmotionState(primary="fear", intensity=0.6, valence=-0.4),
    )

    entries = await system.record_event(
        current_step=2,
        raw_content="A messenger arrived with urgent news.",
        personality=personality,
        importance=MemoryImportance.HIGH,
    )
    await system.drain_writes()

    by_stream = {m.stream: m for m in entries}
    assert set(by_stream) == {MemoryStream.FACTUAL, MemoryStream.EXPERIENTIAL}
    assert by_stream[MemoryStream.FACTUAL].id in system._entries
    assert by_stream[MemoryStream.EXPERIENTIAL].id not in system._entries


@pytest.mark.asyncio
async def test_record_event_resolves_importance_when_caller_omits(container: object) -> None:
    """With no importance passed, record_event must have the LLM score it up front for the async
    write decision. Treating None as 0.0 would fail the importance gate and wrongly skip
    experiential; here it resolves to 0.7 and both gates pass.
    """
    importance_json = '{"score": 0.7, "reasoning": "subjectively meaningful"}'
    # Call order: importance resolves once at the top of record_event. Factual and experiential
    # then each run one conflict check + content generation, reusing that importance:
    #   importance (top level) → factual conflict → factual content
    #                     → experiential content (since the decision is True)
    provider = SequentialMockLLM([
        importance_json,
        '{"conflict": false, "reaction": "pass"}',
        "A factual record.",
        "An experiential record under the surface.",
    ])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality(
        emotion=EmotionState(primary="neutral", intensity=0.5, valence=0.0),
    )

    entries = await system.record_event(
        current_step=3,
        raw_content="An apparently quiet but subjectively meaningful moment.",
        personality=personality,
    )

    streams = {memory.stream for memory in entries}
    assert MemoryStream.EXPERIENTIAL in streams, (
        "experiential 必须基于 LLM 评估的真实 score 写入；早期版本会因 None→0.0 误判而跳过"
    )
    for memory in entries:
        assert 0.69 <= memory.importance <= 0.71


@pytest.mark.asyncio
async def test_importance_prompt_uses_names_not_ids(container: object) -> None:
    """The MEMORY_IMPORTANCE prompt names related people and never leaks ids. Someone with no
    name is rendered as a descriptive referent instead."""
    provider = MockLLMProvider(fixed_response='{"score": 0.5, "reasoning": "x"}')
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()

    await system._importance.evaluate(
        "朝堂上发生了争执。",
        personality=personality,
        related_agents=["a_id", "unknown_id"],
        related_directory={"a_id": "李四", "unknown_id": ""},
        related_relations_text="",
        dominant_need_label="",
    )

    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "李四" in prompt
    assert "a_id" not in prompt
    assert "unknown_id" not in prompt
    assert "某位相关人物" in prompt


@pytest.mark.asyncio
async def test_importance_prompt_third_person_multidim_with_relations(container: object) -> None:
    """Importance is a functional, third-person judgment over several dimensions, with relation
    context injected and the JSON reason before the score (§5). No first person, and the emotion
    appears only once."""
    provider = MockLLMProvider(fixed_response='{"reason": "x", "score": 0.5}')
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.8, valence=-0.7),
    )

    await system._importance.evaluate(
        "他又一次背着我行事。",
        personality=personality,
        related_agents=["x"],
        related_directory={"x": "李四"},
        related_relations_text="  - 李四：关系[政敌:对手]，信任度0.10、好感度-0.60",
        dominant_need_label="渴望重获父亲的认可",
    )

    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "与相关人的关系" in prompt and "政敌:对手" in prompt
    assert "客观" in prompt and "评判维度" in prompt
    assert "对我此时此地" not in prompt
    assert prompt.count("anger") <= 1                            # emotion is not repeated
    assert "渴望重获父亲的认可" in prompt
    assert "Belonging" not in prompt


@pytest.mark.asyncio
async def test_experiential_prompt_has_negative_constraint_and_relations(container: object) -> None:
    """The experiential prompt asks for two things: the objective facts read through the agent's
    personality and emotion, and the feeling that follows. It also guards against overstating,
    abstract moralizing and restating the facts, and injects relation context."""
    provider = SequentialMockLLM(["一段内心独白。"])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()

    await system._write_experiential(
        raw_content="他在朝堂当众反对我。",
        personality=personality,
        related_agent_directory={"x": "李四"},
        related_relations_text="  - 李四：关系[政敌:对手]，信任度0.10、好感度-0.60",
    )

    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "我怎么理解刚发生的事" in prompt and "此刻这般情绪" in prompt
    assert "我的感受" in prompt
    assert "不要借题发挥" in prompt or "玄虚" in prompt
    # Not inventing things outranks not repeating factual: better to restate than make it up.
    assert "胡编乱造" in prompt
    # An empty result is preferred when nothing meaningful happened.
    assert "什么都不写" in prompt
    assert "宁可" in prompt and "无中生有" in prompt
    assert "【我是谁】" in prompt
    assert "我与他们的关系" in prompt and "政敌:对手" in prompt
    # Related people are a closed set, so the model can't invent anyone.
    assert "李四" in prompt


@pytest.mark.asyncio
async def test_experiential_prompt_includes_situation_header(container: object) -> None:
    """The experiential prompt opens with the first-person situation line "我此刻在...,时间为...".
    Agent passes location_view/world_time_label from _last_perception_spatial into record_event."""
    from core.interfaces.perception import LocationView, Situation

    provider = SequentialMockLLM(["一段内心独白。"])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()

    await system._write_experiential(
        raw_content="他在朝堂当众反对我。",
        personality=personality,
        related_agent_directory={"x": "李四"},
        related_relations_text="  - 李四：关系[政敌:对手]，信任度0.10、好感度-0.60",
        situation=Situation(location_view=LocationView(name="朝堂", description="百官议事的正殿"), time_label="辰时"),
    )

    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    # The situation header comes before the persona so it is prominent.
    assert "我此刻在朝堂" in prompt and "时间为辰时" in prompt
    assert prompt.index("我此刻在朝堂") < prompt.index("【我是谁】")


@pytest.mark.asyncio
async def test_experiential_prompt_without_situation_header_still_valid(container: object) -> None:
    """With no situation (default empty Situation()), the helper returns "" and the header is
    left out. The prompt is still valid, nothing raises, and no time or place is invented."""
    provider = SequentialMockLLM(["一段内心独白。"])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()

    await system._write_experiential(
        raw_content="一件事。", personality=personality,
    )
    # The call sends [system, user] (prefix-cache split); join both for the assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    # No header, but the rest of the prompt is intact and no "某处" time/place is invented.
    assert "【我是谁】" in prompt
    assert "我此刻在" not in prompt
    assert "时间为" not in prompt
    assert "【我是谁】" in prompt


def _user_message(messages: list) -> str:
    """Return the user-role message text. The LLM call is [system, user]; the continuity block and
    【刚发生的事】 both live in user, so assertions check only user and can't be fooled by mentions
    in the system constraints."""
    for m in messages:
        if getattr(m, "role", None) == "user":
            return m.content
    raise AssertionError("no user message in call history")


@pytest.mark.asyncio
async def test_experiential_prompt_injects_recent_factual_continuity_block(container: object) -> None:
    """Given recent_factual_block, _write_experiential renders it as a 【此前近来发生的事】 section
    before 【刚发生的事】, with a soft "don't restate" constraint, so the monologue follows on from
    what came before. An empty block leaves the section out.
    """
    provider = SequentialMockLLM(["一段内心独白。"])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()

    await system._write_experiential(
        raw_content="他冲我拂袖而去。",
        personality=personality,
        recent_factual_block="- （刚刚）他在朝堂上驳了我。\n- （几小时前）我上书自请辞去参政之职。",
    )

    user_prompt = _user_message(provider.call_history[0])
    system_prompt = provider.call_history[0][0].content
    assert "【此前近来发生的事" in user_prompt
    assert "他在朝堂上驳了我" in user_prompt
    assert "上书自请辞去参政之职" in user_prompt
    # The earlier block must come before 【刚发生的事】, since generation happens at the end.
    assert user_prompt.index("【此前近来发生的事") < user_prompt.index("【刚发生的事")
    assert "不要复述" in user_prompt and "不要当作刚发生" in user_prompt
    # The system prompt has the matching hard rule ("if 【此前近来发生的事】 is given ... only
    # 【刚发生的事】 is ...").
    assert "此前近来发生的事" in system_prompt and "刚发生" in system_prompt
    from core.prompts import MEMORY_ORDER_HINT
    assert MEMORY_ORDER_HINT in user_prompt


@pytest.mark.asyncio
async def test_experiential_prompt_omits_continuity_header_when_empty(container: object) -> None:
    """An empty recent_factual_block (cold start or nothing recent) omits the 【此前近来发生的事】
    header from user entirely. The system rule ("if ... is given") stays and only matters when the
    block is present."""
    provider = SequentialMockLLM(["一段内心独白。"])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()

    await system._write_experiential(
        raw_content="一件事发生。",
        personality=personality,
        recent_factual_block="",
    )
    user_prompt = _user_message(provider.call_history[0])
    assert "【此前近来发生的事" not in user_prompt
    assert "【刚发生的事" in user_prompt and "【我是谁】" in user_prompt


@pytest.mark.asyncio
async def test_record_event_builds_recent_factual_block_and_excludes_current_event(container: object) -> None:
    """record_event renders the previous few factual memories into the continuity block for
    _write_experiential. The event being written (same event_group_id) is excluded so it doesn't
    duplicate 【刚发生的事】."""
    provider = MockLLMProvider(fixed_response="我把这一次读作压垮我的最后一根稻草。")
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.6),
    )

    # Two earlier factual memories. write() puts them straight into _entries and the store;
    # passing importance skips the LLM.
    await system.write(
        "陛下当众驳斥了我的奏议。",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.HIGH,
        current_step=3,
    )
    await system.write(
        "东宫属官私下劝我隐忍。",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=4,
    )

    # Main character + emotion 0.7 + importance HIGH: both gates pass, so experiential is written.
    provider.call_history.clear()
    await system.record_event(
        current_step=5,
        raw_content="他终于当着百官骂我狼子野心。",
        personality=personality,
        importance=MemoryImportance.HIGH,
    )
    # The experiential rewrite runs in the background, so the prompt is only sent after drain.
    await system.drain_writes()

    # The experiential rewrite call is the one whose user message has the 【我是谁】 format block.
    monologue_calls = [
        messages for messages in provider.call_history
        if any(getattr(m, "role", None) == "user" and "【我是谁】" in m.content for m in messages)
    ]
    assert monologue_calls, "expected an experiential monologue LLM call"
    user_prompt = _user_message(monologue_calls[-1])

    assert "【此前近来发生的事" in user_prompt
    assert "陛下当众驳斥了我的奏议" in user_prompt
    assert "东宫属官私下劝我隐忍" in user_prompt
    # The current event appears only under 【刚发生的事】. The continuity block must not contain it
    # (same event_group_id is excluded).
    continuity_start = user_prompt.index("【此前近来发生的事")
    continuity_end = user_prompt.index("【刚发生的事")
    continuity_section = user_prompt[continuity_start:continuity_end]
    assert "狼子野心" not in continuity_section
    # Oldest first: step 3 before step 4.
    assert continuity_section.index("陛下当众驳斥了我的奏议") < continuity_section.index("东宫属官私下劝我隐忍")
    # The continuity block must not leak ids or steps (render_memory uses only stored_content plus
    # a coarse relative-recency bucket).
    assert "agent-1-factual-" not in continuity_section
    assert "step=" not in continuity_section


@pytest.mark.asyncio
async def test_record_event_asymmetric_write_skips_experiential(container: object) -> None:
    """Background agent + low emotion + low importance: experiential is skipped."""
    provider = SequentialMockLLM(["A faint breeze passed by."])
    system = _make_memory_system(container, provider=provider, is_main_character=False)
    personality = _make_personality(
        emotion=EmotionState(primary="neutral", intensity=0.1, valence=0.0),
    )

    entries = await system.record_event(
        current_step=2,
        raw_content="A faint breeze stirred the grass.",
        personality=personality,
        importance=MemoryImportance.LOW,
    )

    assert {memory.stream for memory in entries} == {MemoryStream.FACTUAL}


@pytest.mark.asyncio
async def test_retrieve_empty_query_returns_empty_without_search(container: object) -> None:
    """An empty query (possible when nothing was perceived) returns empty without embed/search.
    embed("") would recall arbitrary memories that happen to sit closest to the empty vector."""
    system = _make_memory_system(container)
    personality = _make_personality()
    await system.write(
        "Something happened near the gate.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=1,
    )

    for q in ("", "   ", "\n\t"):
        assert await system.retrieve(q, current_step=10, top_k=3) == []


class _KeywordEmbedding:
    """Deterministic stub embedding: one-hot vectors by keyword, so query/memory cosine is exact.
    Lets the relevance-floor tests avoid the hash embedding's random scores."""

    _KEYS = ("alpha", "beta", "gamma")

    async def embed(self, text: str):
        from core.interfaces.embedding import EmbeddingResult
        return EmbeddingResult(dense=[1.0 if k in text else 0.0 for k in self._KEYS], sparse=None)

    @property
    def dimension(self) -> int:
        return len(self._KEYS)


def _make_floor_system(floor: float) -> MemorySystem:
    from providers.vector_store.in_memory import InMemoryVectorStore
    return MemorySystem(
        _router(MockLLMProvider()),
        _KeywordEmbedding(),
        InMemoryVectorStore(),
        world_id="w", agent_id="agent-1",
        retrieval_score_floor=floor,
    )


@pytest.mark.asyncio
async def test_retrieval_score_floor_drops_low_relevance() -> None:
    """Candidates orthogonal to the query (cosine 0) are dropped by the floor instead of filling
    the top N."""
    system = _make_floor_system(floor=0.2)
    await system.ensure_collections()
    p = _make_personality()
    await system.write("alpha 相关事件", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)
    await system.write("beta 无关事件", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)

    # query "alpha" → [1,0,0]: the alpha memory is a perfect hit and stays; the beta memory
    # (cosine 0) falls below the 0.2 floor and is dropped.
    results = await system.retrieve("alpha", current_step=5, top_k=10, stream=MemoryStream.FACTUAL)
    contents = " ".join(m.stored_content for m in results)
    assert "alpha" in contents and "beta" not in contents


@pytest.mark.asyncio
async def test_normalization_is_global_not_per_stream() -> None:
    """Min-max normalization runs over all candidates together, not per stream before merging.

    Per-stream normalization assumes each stream's best hit is equally relevant, so the best junk in
    one stream gets relevance 1.0, ties with a perfect hit in the other, and recency breaks the tie.
    Here an older perfect factual hit faces newer off-topic experiential memories: per stream they
    score 0.65 vs the hit's 0.60; globally the hit wins 0.60 vs 0.40.
    """
    system = _make_floor_system(floor=-1.0)   # floor off, to test normalization alone
    await system.ensure_collections()
    p = _make_personality()
    await system.write("alpha 完美命中", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)
    await system.write("gamma 同流陪衬", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=2)
    # The other stream: orthogonal to the query (dense=0) but newer. This is noise that ranks
    # up on recency alone.
    await system.write("beta 离题近事一", MemoryStream.EXPERIENTIAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=8)
    await system.write("beta 离题近事二", MemoryStream.EXPERIENTIAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=9)

    results = await system.retrieve("alpha", current_step=10, top_k=4)
    assert results[0].stored_content == "alpha 完美命中"


@pytest.mark.asyncio
async def test_retrieval_floor_compares_fused_score() -> None:
    """The floor compares fused, a weighted average in [0,1] (both streams cosine-normalized,
    weights summing to 1).

    The stub embedding emits no sparse vector, so a perfect hit has fused = VECTOR_WEIGHTS[0]. The
    assertions derive from the weights instead of hard-coding them, since the weights get retuned."""
    dense_w = VECTOR_WEIGHTS[0]
    p = _make_personality()
    for floor, expected in ((dense_w - 0.1, ["alpha 事件"]), (dense_w + 0.1, [])):
        system = _make_floor_system(floor=floor)
        await system.ensure_collections()
        await system.write("alpha 事件", MemoryStream.FACTUAL, personality=p,
                           importance=MemoryImportance.MEDIUM, current_step=1)
        results = await system.retrieve("alpha", current_step=5, top_k=10, stream=MemoryStream.FACTUAL)
        assert [m.stored_content for m in results] == expected


@pytest.mark.asyncio
async def test_retrieval_score_floor_all_irrelevant_returns_empty() -> None:
    """If every candidate is irrelevant, return nothing rather than a top N of unrelated memories."""
    system = _make_floor_system(floor=0.2)
    await system.ensure_collections()
    p = _make_personality()
    await system.write("alpha 事件", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)
    await system.write("beta 事件", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)

    # query "gamma" → [0,0,1]. Both memories have cosine 0, below 0.2, so the result is empty.
    results = await system.retrieve("gamma", current_step=5, top_k=10, stream=MemoryStream.FACTUAL)
    assert results == []


@pytest.mark.asyncio
async def test_recall_about_agent_scopes_to_identity_index() -> None:
    """"What I know about someone" restricts candidates by related_agents instead of searching for
    the name as a vector.

    All three memories share the topic (alpha), but only one involves rival, and only that one may
    come back."""
    system = _make_floor_system(floor=-1.0)
    await system.ensure_collections()
    p = _make_personality()
    await system.write("alpha 我与对手交手", MemoryStream.FACTUAL, personality=p,
                       related_agents=["agent-rival"], importance=MemoryImportance.MEDIUM, current_step=1)
    await system.write("alpha 我与盟友议事", MemoryStream.FACTUAL, personality=p,
                       related_agents=["agent-ally"], importance=MemoryImportance.MEDIUM, current_step=2)
    await system.write("alpha 我独自巡视", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=3)

    results = await system.recall_about_agent(
        "agent-rival", "alpha", current_step=5, top_k=10, stream=MemoryStream.FACTUAL
    )
    assert [m.stored_content for m in results] == ["alpha 我与对手交手"]


@pytest.mark.asyncio
async def test_recall_about_agent_unknown_person_returns_empty_not_nearest_neighbours() -> None:
    """If I have never dealt with this person, the identity scope is empty and so is the result.

    Don't fall back to global semantic neighbors. In a single-theme world every memory looks like
    every other, so the neighbors are reliably memories about someone else on the same topic, and
    the downstream LLM takes them as real history with this person. "I don't know him" is the
    right answer."""
    system = _make_floor_system(floor=-1.0)
    await system.ensure_collections()
    p = _make_personality()
    # Lots of on-topic memories that are all about a third party: exactly what a neighbor search
    # would wrongly return.
    for step in range(1, 5):
        await system.write(f"alpha 我与常何确认伏兵部署 {step}", MemoryStream.FACTUAL, personality=p,
                           related_agents=["agent-changhe"], importance=MemoryImportance.HIGH,
                           current_step=step)

    results = await system.recall_about_agent(
        "agent-yuanji", "alpha 伏兵部署", current_step=5, top_k=5, stream=MemoryStream.FACTUAL
    )
    assert results == []


@pytest.mark.asyncio
async def test_identity_scope_survives_top_k_truncation() -> None:
    """The identity scope is pushed down to the store, restricting candidates before the search,
    rather than filtered afterwards.

    This test tells the two apart. A memory about the target with the worst similarity would be
    cut by the store's top_k before a post-filter ever saw it, giving an empty result. Pushed down,
    it is the only candidate in scope and must come back."""
    system = _make_floor_system(floor=-1.0)
    await system.ensure_collections()
    p = _make_personality()
    # Orthogonal to the query ("alpha"), so it has the worst relevance, but it is the only memory
    # involving rival.
    await system.write("gamma 我与对手的旧怨", MemoryStream.FACTUAL, personality=p,
                       related_agents=["agent-rival"], importance=MemoryImportance.MEDIUM, current_step=1)
    # Enough highly relevant memories unrelated to rival to fill any top_k.
    for step in range(2, 12):
        await system.write(f"alpha 高相关无关人事 {step}", MemoryStream.FACTUAL, personality=p,
                           importance=MemoryImportance.HIGH, current_step=step)

    results = await system.recall_about_agent(
        "agent-rival", "alpha", current_step=20, top_k=3, stream=MemoryStream.FACTUAL
    )
    assert [m.stored_content for m in results] == ["gamma 我与对手的旧怨"]


@pytest.mark.asyncio
async def test_retrieve_trace_exposes_full_funnel_without_touching() -> None:
    """The diagnostic trace records every candidate, including those below the floor, and what
    happened to it. touch=False leaves the world unchanged.

    If a diagnostic read touched memories, looking at recall would bump last_accessed_step and
    recency, so the dev tool would change the recall it is observing."""
    system = _make_floor_system(floor=0.2)
    await system.ensure_collections()
    p = _make_personality()
    await system.write("alpha 命中", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)
    await system.write("beta 离题", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)

    trace: list = []
    results = await system.retrieve(
        "alpha", current_step=9, top_k=10, stream=MemoryStream.FACTUAL, trace=trace, touch=False,
    )
    assert [m.stored_content for m in results] == ["alpha 命中"]

    by_content = {r.memory.stored_content: r for r in trace}
    assert by_content["beta 离题"].dropped == "floor"      # below the floor, but still in the funnel
    assert by_content["alpha 命中"].dropped is None
    assert by_content["alpha 命中"].dense == pytest.approx(1.0)
    # touch=False: no access step written back and retrieval_count unchanged.
    assert by_content["alpha 命中"].memory.last_accessed_step == -1
    assert by_content["alpha 命中"].memory.retrieval_count == 0


@pytest.mark.asyncio
async def test_retrieval_score_floor_zero_keeps_all() -> None:
    """With the floor at -1 (the fixture default) nothing is dropped, which shows the floor is what
    did the dropping above."""
    system = _make_floor_system(floor=-1.0)
    await system.ensure_collections()
    p = _make_personality()
    await system.write("beta 事件", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)
    results = await system.retrieve("alpha", current_step=5, top_k=10, stream=MemoryStream.FACTUAL)
    assert len(results) == 1  # cosine 0, but a -1 floor keeps it


@pytest.mark.asyncio
async def test_retrieve_both_empty_query_returns_empty_result(container: object) -> None:
    """retrieve_both with both queries empty returns an empty RetrievalResult and skips embed/search."""
    system = _make_memory_system(container)
    personality = _make_personality()
    await system.write(
        "Something happened near the gate.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=1,
    )

    result = await system.retrieve_both("   ", current_step=10, top_k_each=3)
    assert result.events == []
    assert result.insights == []
    assert result.period_summaries == []


@pytest.mark.asyncio
async def test_retrieve_updates_stats_and_uses_last_accessed_recency(container: object) -> None:
    system = _make_memory_system(container)
    personality = _make_personality()
    memory = await system.write(
        "Something happened near the gate.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=1,
    )
    assert memory is not None

    results = await system.retrieve(
        "gate",
        current_step=10,
        top_k=3,
        stream=MemoryStream.FACTUAL,
    )

    assert any(item.id == memory.id for item in results)
    stored = system._entries[memory.id]
    assert stored.retrieval_count == 1
    assert stored.last_accessed_step == 10


@pytest.mark.asyncio
async def test_retrieve_excludes_before_truncating_and_touches_only_what_it_returns() -> None:
    """exclude_ids are removed before top_k truncation, so the result is still full. Excluded
    memories aren't touched because they weren't used."""
    system = _make_floor_system(floor=-1.0)
    await system.ensure_collections()
    p = _make_personality()
    written = [
        await system.write(f"alpha 第{i}件", MemoryStream.FACTUAL, personality=p,
                           importance=MemoryImportance.MEDIUM, current_step=i)
        for i in range(1, 5)
    ]
    excluded = {written[0].id, written[1].id}

    results = await system.retrieve("alpha", current_step=10, top_k=2,
                                    stream=MemoryStream.FACTUAL, exclude_ids=excluded)

    assert len(results) == 2
    assert not ({m.id for m in results} & excluded)
    for mid in excluded:
        assert system._entries[mid].retrieval_count == 0


def test_retrieval_weights_are_fixed_rir() -> None:
    """Retrieval uses one fixed R/I/R weight triple, not weights that switch with cognitive context."""
    assert set(RETRIEVAL_WEIGHTS) == {"relevance", "recency", "importance"}
    assert abs(sum(RETRIEVAL_WEIGHTS.values()) - 1.0) < 1e-9


@pytest.mark.asyncio
async def test_critical_memory_survives_decay(container: object) -> None:
    system = _make_memory_system(container)
    personality = _make_personality()
    critical = await system.write(
        "A betrayal changed the succession.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.CRITICAL,
        current_step=1,
    )
    assert critical is not None

    await system.apply_decay(current_step=100)

    stored = system._entries[critical.id]
    assert stored.decay_score == 1.0


@pytest.mark.asyncio
async def test_a_vector_store_hiccup_during_decay_does_not_kill_the_step(
    container: object, caplog
) -> None:
    """A vector store read failure costs only this round of decay and must not propagate:
    runtime.run_step doesn't wrap maintenance in a try.

    Decay reads the whole collection first, so it isn't purely local. If a hiccup raised, the rest of
    the step (perception, actions, snapshot) would not run. Rule 6: a failed read returns empty with
    a warning and the simulation continues.
    """
    system = _make_memory_system(container)
    personality = _make_personality()
    await system.write(
        "The gate was left unbarred.", MemoryStream.FACTUAL,
        personality=personality, current_step=1,
    )

    async def _boom(collection: str):
        raise RuntimeError("vector store unavailable")

    system._vector_store.list_all = _boom  # type: ignore[assignment]

    with caplog.at_level("WARNING", logger="agent.memory"):
        await system.apply_decay(current_step=10)

    failed = [r for r in caplog.records if r.message == "memory_decay_list_failed"]
    assert failed, "读失败必须留下可查的记录,不能静默跳过"
    assert all(r.agent_id for r in failed)


@pytest.mark.asyncio
async def test_decay_never_deletes_memory(container: object) -> None:
    """apply_decay never deletes memories; it only updates decay_score.
    Only Compression and the Forget event may delete."""
    system = _make_memory_system(container)
    personality = _make_personality()
    memory = await system.write(
        "A small errand was completed.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.LOW,
        current_step=1,
        decay_score=0.101,
    )
    assert memory is not None

    for _ in range(20):
        await system.apply_decay(current_step=10)

    assert memory.id in system._entries
    stored = system._entries[memory.id]
    assert stored.decay_score < 0.05


@pytest.mark.asyncio
async def test_compression_with_strict_filter_and_source_ids(container: object) -> None:
    """Compression filters strictly, writes a summary that keeps source_ids, and deletes the sources.

    Candidates must have kind=event, importance<0.4, decay<0.2, |valence|<0.5, and be at least
    MIN_AGE_STEPS steps past created_step.
    """
    # Compression always calls the LLM. Every call returns this valid points-only JSON summary.
    provider = MockLLMProvider(fixed_response='{"points": ["市集如常的一段时期。"]}')
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality()
    # Old, low-activity memories in one window with the same people, enough to reach
    # COMPRESSION_TRIGGER_COUNT=30.
    base_step = 5
    for index in range(35):
        memory = await system.write(
            f"Routine market observation #{index}.",
            MemoryStream.FACTUAL,
            personality=personality,
            related_agents=["bystander"],
            importance=0.2,
            current_step=base_step + index % 15,
            decay_score=0.15,
        )
        assert memory is not None

    produced = await system.compress(MemoryStream.FACTUAL, current_step=200)
    assert produced >= 1

    collection = "world-1:agent-1:memory:factual"
    # The summary is written and the cluster deleted, so the total drops below 35.
    assert len(await container.vector_store.list_all(collection)) < 35
    summaries = [m for m in system._entries.values() if m.kind == "summary"]
    assert any(s.source_ids for s in summaries)


def test_summary_prompt_renders_time_range_not_step(container: object) -> None:
    """The compression prompt shows the time span in the world calendar ("<time> 至 <time>") and
    doesn't leak code-layer steps."""
    shichen = ["子时", "丑时", "寅时", "卯时", "辰时"]
    system = _make_memory_system(
        container, time_label_for=lambda s: f"三月，{shichen[s % len(shichen)]}"
    )
    cluster = [
        Memory(
            id=f"m{i}",
            stream=MemoryStream.FACTUAL,
            agent_id="agent-1",
            stored_content=f"market note {i}",
            created_step=10 + i,
        )
        for i in range(4)
    ]
    time_range = system._span_time_range(cluster)
    # rendered as the calendar span "<start> 至 <end>"
    assert "至" in time_range
    assert system._time_label_for(10) in time_range
    assert system._time_label_for(13) in time_range
    # compress only handles factual memories (see MemorySystem.compress). It returns
    # (system, user) for prefix caching; join them before asserting (the span is in user).
    prompt = "\n".join(system._build_summary_prompt(MemoryStream.FACTUAL, time_range=time_range, contents="- a\n- b"))
    assert "step" not in prompt.lower()
    assert "第" not in prompt or "第N步" not in prompt
    assert time_range in prompt


def test_factual_summary_prompt_is_objective(container: object) -> None:
    """compress only handles factual memories: the summary prompt is objective third person and
    asks for a functional reason before the points."""
    system = _make_memory_system(container)
    fac = "\n".join(system._build_summary_prompt(MemoryStream.FACTUAL, time_range="", contents="- a"))
    assert "reason" in fac
    assert "客观" in fac


@pytest.mark.asyncio
async def test_main_compression_reason_not_persisted(container: object) -> None:
    """Main-character compression returns LLM JSON. reason only anchors the model's thinking and is
    never written to stored_content, so it doesn't end up in the embedding."""
    reason_text = "REASON_SHOULD_NOT_LEAK_INTO_STORAGE"
    payload = (
        '{"reason": "' + reason_text + '", '
        '"points": ["市集如常运转的一段平淡时期", "无异常事件"]}'
    )
    provider = MockLLMProvider(fixed_response=payload)
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    personality = _make_personality()
    for index in range(35):
        memory = await system.write(
            f"Routine observation #{index}.",
            MemoryStream.FACTUAL,
            personality=personality,
            related_agents=["bystander"],
            importance=0.2,
            current_step=5 + index % 15,
            decay_score=0.15,
        )
        assert memory is not None

    produced = await system.compress(MemoryStream.FACTUAL, current_step=200)
    assert produced >= 1
    summaries = [m for m in system._entries.values() if m.kind == "summary"]
    assert summaries
    for s in summaries:
        assert reason_text not in s.stored_content
        assert "市集如常运转" in s.stored_content
        assert "无异常事件" in s.stored_content


@pytest.mark.asyncio
async def test_reflection_candidates_survive_cold_entries(container: object) -> None:
    """After a snapshot restore _entries is empty, but candidates_for_reflection still loads
    candidates from the vector store, so a cold restore doesn't skip reflection."""
    from agent.reflection import MAX_REFLECTION_DEPTH

    system = _make_memory_system(container, is_main_character=True)
    event = Memory(
        id="agent-1-cold-event",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        stored_content="A grounding event.",
        importance=0.7,
        created_step=3,
        kind="event",
        emotion_valence=-0.5,
    )
    await system._persist(event)

    # Simulated restart: a fresh MemorySystem with empty _entries, sharing the same vector store.
    fresh = MemorySystem(
        system._llm_router,
        system._embedding,
        system._vector_store,
        world_id="world-1",
        agent_id="agent-1",
        is_main_character=True,
    )
    assert fresh._entries == {}
    candidates = await fresh.candidates_for_reflection(
        current_step=10,
        lookback_steps=30,
        max_candidates=10,
        max_depth=MAX_REFLECTION_DEPTH,
    )
    assert event.id in {c.id for c in candidates}


def test_coerce_importance_returns_float() -> None:
    """coerce_importance always returns a float in [0, 1]. Enum inputs map to the bucket's center
    score; floats are clamped to [0, 1]."""
    assert coerce_importance(MemoryImportance.CRITICAL) == 0.95
    assert coerce_importance(MemoryImportance.HIGH) == 0.75
    assert coerce_importance(MemoryImportance.MEDIUM) == 0.5
    assert coerce_importance(MemoryImportance.LOW) == 0.25

    assert coerce_importance(0.95) == 0.95
    assert coerce_importance(0.0) == 0.0
    assert coerce_importance(1.0) == 1.0
    assert coerce_importance(1.5) == 1.0
    assert coerce_importance(-0.3) == 0.0

    assert importance_level(0.95) == MemoryImportance.CRITICAL
    assert importance_level(0.70) == MemoryImportance.HIGH
    assert importance_level(0.50) == MemoryImportance.MEDIUM
    assert importance_level(0.20) == MemoryImportance.LOW


def test_memory_new_fields_defaults() -> None:
    """The kind / event_group_id / source_ids fields have the right defaults."""
    from agent.memory_types import Memory, MemoryStream

    m = Memory(id="t", stream=MemoryStream.FACTUAL)
    assert m.kind == "event"
    assert m.event_group_id is None
    assert m.source_ids == []
    assert isinstance(m.importance, float)
    assert m.importance == 0.5


@pytest.mark.asyncio
async def test_compression_critical_memory_survives(container: object) -> None:
    system = _make_memory_system(container)
    personality = _make_personality()

    critical = await system.write(
        "A critical betrayal changed everything.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.CRITICAL,
        current_step=0,
    )
    assert critical is not None

    for index in range(4):
        await system.write(
            f"Routine observation {index}.",
            MemoryStream.FACTUAL,
            personality=personality,
            importance=MemoryImportance.LOW,
            current_step=index + 1,
        )

    await system.compress(MemoryStream.FACTUAL, personality=personality)

    # Critical memory must not be deleted
    assert critical.id in system._entries


@pytest.mark.asyncio
async def test_decay_consolidation_bonus_applied_when_memory_retrieved(container: object) -> None:
    """Retrieving a memory multiple times increases its retrieval_count,
    which then leads to a higher decay_rate (slower decay) in apply_decay."""
    system = _make_memory_system(container)
    personality = _make_personality()

    memory = await system.write(
        "An event worth remembering.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=1,
    )
    assert memory is not None

    # Retrieve the memory multiple times
    for step in range(2, 8):
        await system.retrieve("worth remembering", current_step=step, top_k=5)

    stored = system._entries[memory.id]
    # retrieval_count should have increased, which gives a consolidation bonus
    assert stored.retrieval_count > 0

    # Apply decay — memory with consolidation bonus should survive longer
    # (It may still decay, but it should still exist at moderate step count)
    await system.apply_decay(current_step=20)

    # Memory still survives — consolidation bonus keeps it above the deletion threshold
    assert memory.id in system._entries


# ---------------------------------------------------------------------------
# warm_recent_memories: cross-session bias recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_recent_memories_loads_recent_experiential_into_entries(container: object) -> None:
    """After warm-up, related_memory_bias returns non-zero for recently tagged agent."""
    provider = SequentialMockLLM(["I felt uneasy after the encounter."] * 10)
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality(
        emotion=__import__("agent.personality", fromlist=["EmotionState"]).EmotionState(
            primary=__import__("agent.personality", fromlist=["EmotionType"]).EmotionType.FEAR,
            intensity=0.8,
            valence=-0.7,
        )
    )

    await system.ensure_collections()
    await system.write(
        "A tense exchange with agent-99.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        related_agents=["agent-99"],
        current_step=5,
    )

    # Simulate restart: create a fresh MemorySystem sharing the same vector store.
    fresh = MemorySystem(
        system._llm_router,
        system._embedding,
        system._vector_store,
        world_id="world-1",
        agent_id="agent-1",
    )
    assert fresh.related_memory_bias(target_agent_id="agent-99", current_step=10) == 0.0

    await fresh.warm_recent_memories(current_step=10, lookback_steps=8)

    bias = fresh.related_memory_bias(target_agent_id="agent-99", current_step=10)
    assert bias < 0.0, f"expected negative valence bias, got {bias}"


@pytest.mark.asyncio
async def test_warm_recent_memories_ignores_entries_outside_lookback(container: object) -> None:
    """Memories older than lookback_steps must not be loaded."""
    provider = SequentialMockLLM(["Old memory text."] * 10)
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality()

    await system.ensure_collections()
    await system.write(
        "Ancient interaction with agent-99.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        related_agents=["agent-99"],
        current_step=1,  # very old
    )

    fresh = MemorySystem(
        system._llm_router,
        system._embedding,
        system._vector_store,
        world_id="world-1",
        agent_id="agent-1",
    )
    await fresh.warm_recent_memories(current_step=20, lookback_steps=8)
    # step 1 is 19 steps ago, outside the 8-step window
    assert fresh.related_memory_bias(target_agent_id="agent-99", current_step=20) == 0.0


@pytest.mark.asyncio
async def test_sample_recent_includes_both_streams(container: object) -> None:
    """sample_recent returns both factual and experiential; long-term goal revision reads both."""
    provider = SequentialMockLLM(["Colored experiential text."] * 10)
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality()

    await system.write(
        "皇帝驾崩。", MemoryStream.FACTUAL,
        personality=personality, importance=MemoryImportance.CRITICAL, current_step=5,
    )
    await system.write(
        "我意识到效忠已无对象。", MemoryStream.EXPERIENTIAL,
        personality=personality, importance=MemoryImportance.HIGH, current_step=5,
    )

    results = await system.sample_recent(current_step=7, lookback=3, top_k=5)

    streams = {m.stream for m in results}
    assert MemoryStream.FACTUAL in streams and MemoryStream.EXPERIENTIAL in streams


@pytest.mark.asyncio
async def test_sample_recent_robust_after_cold_entries(container: object) -> None:
    """sample_recent still reads factual memories from the vector store when _entries is empty
    (restored but not yet warmed up)."""
    provider = SequentialMockLLM(["Colored experiential text."] * 10)
    system = _make_memory_system(container, provider=provider)
    personality = _make_personality()

    await system.write(
        "皇帝驾崩。", MemoryStream.FACTUAL,
        personality=personality, importance=MemoryImportance.CRITICAL, current_step=5,
    )
    system._entries.clear()  # cold restore: cache empty, data only in the vector store

    results = await system.sample_recent(current_step=7, lookback=3, top_k=5)

    assert any(m.stream == MemoryStream.FACTUAL and "皇帝驾崩" in m.stored_content for m in results)


# ---------------------------------------------------------------------------
# Contracts: RetrievalResult / kind-aware / decay floor
# ---------------------------------------------------------------------------


def test_minmax_normalization() -> None:
    """Per-query min-max maps linearly to [0,1]; if all values are equal they become 0.5."""
    assert _minmax([0.2, 0.4, 0.7]) == [0.0, pytest.approx(0.4), 1.0]
    assert _minmax([0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5]
    assert _minmax([]) == []


def test_kind_aware_ranking_summary_recency_dampened() -> None:
    """A summary's raw recency is scaled by 0.7 so that touch() rumination doesn't make it look
    fresh. Events and insights share the same recency formula; an insight's confidence is already
    handled by importance, decay and dedup, so scoring adds no extra bias."""
    event = Memory(id="e", stream=MemoryStream.FACTUAL, importance=0.5, created_step=10, kind="event")
    insight = Memory(id="i", stream=MemoryStream.EXPERIENTIAL, importance=0.5, created_step=10, kind="insight")
    summary = Memory(id="s", stream=MemoryStream.FACTUAL, importance=0.5, created_step=10, kind="summary")

    assert _raw_recency(insight, 11) == _raw_recency(event, 11)
    assert _raw_recency(summary, 11) == pytest.approx(_raw_recency(event, 11) * 0.7)


def test_per_query_normalization_makes_relevance_discriminate() -> None:
    """After normalization, relevance actually separates candidates: a highly relevant,
    low-importance memory beats a barely relevant, high-importance one, because relevance has the
    higher weight.

    The assertion depends only on "relevance weight > importance weight", not on the values, which
    get retuned (see RETRIEVAL_WEIGHTS)."""
    assert RETRIEVAL_WEIGHTS["relevance"] > RETRIEVAL_WEIGHTS["importance"]  # the test depends on this
    hi_rel_lo_imp = Memory(id="r", stream=MemoryStream.FACTUAL, importance=0.2, created_step=10)
    lo_rel_hi_imp = Memory(id="i", stream=MemoryStream.FACTUAL, importance=0.9, created_step=10)
    # Same recency; vector scores 0.9 vs 0.3. After normalization relevance dominates.
    ranked = score_and_rank(
        [(hi_rel_lo_imp, _hit(0.9)), (lo_rel_hi_imp, _hit(0.3))], 10, RETRIEVAL_WEIGHTS
    )
    top = max(ranked, key=lambda r: r.score).memory
    assert top.id == "r"


@pytest.mark.asyncio
async def test_retrieve_both_returns_retrieval_result(container: object) -> None:
    """retrieve_both returns a RetrievalResult with each field correctly classified."""
    system = _make_memory_system(container)
    personality = _make_personality()
    await system.write(
        "Court news arrived.",
        MemoryStream.FACTUAL,
        personality=personality,
        importance=MemoryImportance.MEDIUM,
        current_step=1,
    )

    result = await system.retrieve_both(
        "court news",
        current_step=5,
        top_k_each=3,
    )

    assert isinstance(result, RetrievalResult)
    assert isinstance(result.events, list)
    assert isinstance(result.insights, list)
    assert isinstance(result.period_summaries, list)
    assert isinstance(result.stream_links, dict)
    assert isinstance(result.insight_sources, dict)


def test_decay_floor_continuous_dampens_by_decay_score() -> None:
    """When decay_score < 0.2, score_and_rank multiplies the final score by decay_score: a deeply
    buried memory (decay 0.02) is very hard to surface but not impossible. decay >= 0.2 has no
    effect. The three entries differ only in decay, so after normalization every factor is 0.5 and
    only the decay penalty separates them."""
    active = Memory(id="a", stream=MemoryStream.FACTUAL, importance=0.5, created_step=10, decay_score=0.5)
    fading = Memory(id="f", stream=MemoryStream.FACTUAL, importance=0.5, created_step=10, decay_score=0.1)
    buried = Memory(id="b", stream=MemoryStream.FACTUAL, importance=0.5, created_step=10, decay_score=0.02)

    ranked = {r.memory.id: r.score for r in score_and_rank(
        [(active, _hit(0.5)), (fading, _hit(0.5)), (buried, _hit(0.5))], 11, RETRIEVAL_WEIGHTS)}
    # After normalization every factor is 0.5, so the base score is 0.5. active keeps it;
    # fading/buried are multiplied by their decay.
    assert ranked["a"] == pytest.approx(0.5)
    assert ranked["f"] == pytest.approx(0.5 * 0.1)
    assert ranked["b"] == pytest.approx(0.5 * 0.02)
    assert ranked["a"] > ranked["f"] > ranked["b"]


# ---------------------------------------------------------------------------
# Contracts: emotional anchor protection / lifecycle order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emotional_anchor_protected_from_decay(container: object) -> None:
    """Experiential memories with |valence| >= 0.7 don't decay."""
    system = _make_memory_system(container, provider=MockLLMProvider(fixed_response="一段内心独白。"))
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.9, valence=-0.9),
    )
    memory = await system.write(
        "A devastating betrayal.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        importance=MemoryImportance.HIGH,
        current_step=1,
    )
    assert memory is not None
    initial_decay = memory.decay_score

    for _ in range(20):
        await system.apply_decay(current_step=10)

    stored = system._entries[memory.id]
    assert stored.decay_score == initial_decay


# ---------------------------------------------------------------------------
# Contracts: Reflection / source_ids / Compression fixup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflection_produces_insight_with_source_ids(container: object) -> None:
    """ReflectionEngine writes an insight memory whose source_ids point at real memories."""
    from agent.reflection import ReflectionEngine

    system = _make_memory_system(
        container, provider=MockLLMProvider(fixed_response="一段内心独白。"), is_main_character=True,
    )
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.6),
    )

    # experiential events as reflection material
    source_ids: list[str] = []
    for index in range(3):
        memory = await system.write(
            f"I again caught X turning away from me, day {index}.",
            MemoryStream.EXPERIENTIAL,
            personality=personality,
            related_agents=["X"],
            importance=MemoryImportance.HIGH,
            current_step=index + 1,
        )
        assert memory is not None
        source_ids.append(memory.id)

    # The mock LLM returns one insight citing the first two (1-based source_indices).
    insight_payload = (
        '{"insights": [{"text": "X is consistently avoiding me — trust may be gone.", '
        '"importance": 0.7, "source_indices": [1, 2]}]}'
    )
    provider = SequentialMockLLM([insight_payload])
    engine = ReflectionEngine(
        _router(provider), system, personality, agent_id="agent-1"
    )

    result = await engine.reflect(current_step=10)

    assert len(result.new_insights) == 1
    insight = result.new_insights[0]
    assert insight.kind == "insight"
    assert set(insight.source_ids) == {source_ids[0], source_ids[1]}
    # confidence bonus: base 0.7 + 0.05 × 2 = 0.8
    assert 0.79 <= insight.importance <= 0.81


@pytest.mark.asyncio
async def test_reflection_filters_hallucinated_source_ids(container: object) -> None:
    """Source indices from the LLM that aren't in the candidate set are dropped; an insight left
    with no sources is dropped entirely."""
    from agent.reflection import ReflectionEngine

    system = _make_memory_system(
        container, provider=MockLLMProvider(fixed_response="一段内心独白。"), is_main_character=True,
    )
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.6),
    )
    real = await system.write(
        "A real experience.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        importance=MemoryImportance.HIGH,
        current_step=1,
    )
    assert real is not None

    # source_indices: 1 is real (the only candidate); 99 and 42 are out of range, i.e. hallucinated.
    insight_payload = (
        '{"insights": ['
        '{"text": "Insight A grounded in real source.", "importance": 0.6, '
        '"source_indices": [1, 99]}, '
        '{"text": "Insight B entirely fabricated.", "importance": 0.6, '
        '"source_indices": [42]}'
        ']}'
    )
    provider = SequentialMockLLM([insight_payload])
    engine = ReflectionEngine(
        _router(provider), system, personality, agent_id="agent-1"
    )

    result = await engine.reflect(current_step=5)

    # The first insight survives with the bad indices removed; the second is dropped.
    assert len(result.new_insights) == 1
    assert result.new_insights[0].source_ids == [real.id]


@pytest.mark.asyncio
async def test_a_malformed_importance_costs_no_insight(container: object) -> None:
    """A null or non-numeric importance falls back to the default; raising there would drop every
    insight after it in the batch."""
    from agent.reflection import ReflectionEngine

    system = _make_memory_system(
        container, provider=MockLLMProvider(fixed_response="一段内心独白。"), is_main_character=True,
    )
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.6),
    )
    real = await system.write(
        "A real experience.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        importance=MemoryImportance.HIGH,
        current_step=1,
    )
    assert real is not None
    insight_payload = (
        '{"insights": ['
        '{"text": "Insight A.", "importance": null, "source_indices": [1]}, '
        '{"text": "Insight B.", "importance": "high", "source_indices": [1]}, '
        '{"text": "Insight C.", "importance": 0.6, "source_indices": [1]}'
        ']}'
    )
    engine = ReflectionEngine(
        _router(SequentialMockLLM([insight_payload])), system, personality, agent_id="agent-1"
    )

    result = await engine.reflect(current_step=5)

    assert len(result.new_insights) == 3


@pytest.mark.asyncio
async def test_reflection_candidates_include_events_and_insights(container: object) -> None:
    """Candidates include events and insights (so beliefs can evolve) but not summaries, which are
    already an abstraction and give reflection nothing new."""

    system = _make_memory_system(container, is_main_character=True)

    event_mem = Memory(
        id="agent-1-event",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="A real event.",
        stored_content="A real event.",
        importance=0.6,
        created_step=1,
        kind="event",
        reflection_depth=0,
        emotion_valence=-0.5,
    )
    insight_v1 = Memory(
        id="agent-1-insight-v1",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="An older insight (depth 1).",
        stored_content="An older insight (depth 1).",
        importance=0.8,
        created_step=2,
        kind="insight",
        reflection_depth=1,
        emotion_valence=-0.5,
    )
    summary_mem = Memory(
        id="agent-1-summary",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="A period summary.",
        stored_content="A period summary.",
        importance=0.5,
        created_step=3,
        kind="summary",
        emotion_valence=-0.2,
    )
    await system._persist(event_mem)
    await system._persist(insight_v1)
    await system._persist(summary_mem)

    from agent.reflection import MAX_REFLECTION_DEPTH

    candidates = await system.candidates_for_reflection(
        current_step=10,
        lookback_steps=30,
        max_candidates=10,
        max_depth=MAX_REFLECTION_DEPTH,
    )

    candidate_ids = {c.id for c in candidates}
    assert event_mem.id in candidate_ids
    assert insight_v1.id in candidate_ids
    assert summary_mem.id not in candidate_ids


@pytest.mark.asyncio
async def test_reflection_candidates_exclude_depth_capped_insights(container: object) -> None:
    """Insights with reflection_depth >= MAX_REFLECTION_DEPTH aren't candidates, so reflection
    can't keep abstracting its own abstractions."""
    from agent.reflection import MAX_REFLECTION_DEPTH

    system = _make_memory_system(container, is_main_character=True)

    deep_insight = Memory(
        id="agent-1-deep-insight",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="An insight at max depth.",
        stored_content="An insight at max depth.",
        importance=0.85,
        created_step=2,
        kind="insight",
        reflection_depth=MAX_REFLECTION_DEPTH,  # already at the cap
        emotion_valence=-0.5,
    )
    await system._persist(deep_insight)

    candidates = await system.candidates_for_reflection(
        current_step=10,
        lookback_steps=30,
        max_candidates=10,
        max_depth=MAX_REFLECTION_DEPTH,
    )

    assert deep_insight.id not in {c.id for c in candidates}


@pytest.mark.asyncio
async def test_reflection_evolution_promotes_depth_and_keeps_old_insight(container: object) -> None:
    """A new insight can cite an old insight plus a new event to produce v2, with
    v2.reflection_depth = max(sources.depth) + 1. The old insight is kept."""
    from agent.reflection import ReflectionEngine

    system = _make_memory_system(
        container, provider=MockLLMProvider(fixed_response="一段内心独白。"), is_main_character=True,
    )
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.5),
    )

    insight_v1 = Memory(
        id="agent-1-insight-v1",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="X 看不起我",
        stored_content="X 看不起我",
        importance=0.7,
        created_step=20,
        kind="insight",
        reflection_depth=1,
        emotion_valence=-0.6,
    )
    await system._persist(insight_v1)

    new_event = await system.write(
        "X 不顾安危救了我。",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        related_agents=["X"],
        importance=MemoryImportance.HIGH,
        current_step=50,
    )
    assert new_event is not None

    # The mock LLM returns v2 citing v1 and new_event (source_indices 1=v1, 2=new_event).
    insight_payload = (
        '{"insights": [{"text": "X 的傲慢只是面具，关键时刻他在乎。", '
        '"importance": 0.7, '
        '"source_indices": [1, 2]}]}'
    )
    provider = SequentialMockLLM([insight_payload])
    engine = ReflectionEngine(
        _router(provider), system, personality, agent_id="agent-1"
    )

    result = await engine.reflect(current_step=55)

    assert len(result.new_insights) == 1
    v2 = result.new_insights[0]
    assert v2.kind == "insight"
    assert set(v2.source_ids) == {insight_v1.id, new_event.id}
    # depth = max(1, 0) + 1 = 2
    assert v2.reflection_depth == 2
    assert insight_v1.id in system._entries


@pytest.mark.asyncio
async def test_reflection_drops_insight_without_event_grounding(container: object) -> None:
    """An insight citing only old insights, with no event behind it, is dropped as ungrounded."""
    from agent.reflection import ReflectionEngine

    system = _make_memory_system(container, is_main_character=True)
    personality = _make_personality(
        emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.5),
    )

    insight_v1 = Memory(
        id="agent-1-insight-v1",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="旧 insight 一",
        stored_content="旧 insight 一",
        importance=0.7,
        created_step=10,
        kind="insight",
        reflection_depth=1,
        emotion_valence=-0.4,
    )
    insight_v1_b = Memory(
        id="agent-1-insight-v1b",
        stream=MemoryStream.EXPERIENTIAL,
        agent_id="agent-1",
        raw_content="旧 insight 二",
        stored_content="旧 insight 二",
        importance=0.7,
        created_step=11,
        kind="insight",
        reflection_depth=1,
        emotion_valence=-0.4,
    )
    await system._persist(insight_v1)
    await system._persist(insight_v1_b)

    # The LLM returns a "pure abstraction" citing two old insights and no event
    # (source_indices 1=insight_v1, 2=insight_v1_b).
    payload = (
        '{"insights": [{"text": "完全脱离经验的纯抽象推演。", "importance": 0.7, '
        '"source_indices": [1, 2]}]}'
    )
    provider = SequentialMockLLM([payload])
    engine = ReflectionEngine(
        _router(provider), system, personality, agent_id="agent-1"
    )

    result = await engine.reflect(current_step=30)

    assert result.new_insights == []


# ---------------------------------------------------------------------------
# Rule 1: a transient provider failure must not propagate up the feedback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_survives_embedding_failure(container) -> None:
    """_persist is shared by feedback writes, retrieval touches and decay. A single embedding 429
    loses only this entry's vector write (the memory stays in _entries) and must not propagate up
    and kill the caller."""

    class _FailingEmbedding:
        async def embed(self, text: str):
            raise RuntimeError("HTTP 429")

    system = _make_memory_system(container)
    system._embedding = _FailingEmbedding()  # noqa: SLF001

    memory = Memory(id="m-fail", stream=MemoryStream.FACTUAL, raw_content="有人路过", importance=0.5)
    await system._persist(memory)  # noqa: SLF001 — must not raise

    assert "m-fail" in system._entries  # noqa: SLF001
    assert len(await container.vector_store.list_all(
        system._namespace(MemoryStream.FACTUAL)  # noqa: SLF001
    )) == 0


@pytest.mark.asyncio
async def test_touch_writeback_does_not_reembed(container) -> None:
    """A recall touch only writes back last_accessed_step. The text hasn't changed, so it must not
    be embedded again; this path runs for every hit on every recall."""
    system = _make_memory_system(container)
    await system.ensure_collections()
    p = _make_personality()
    await system.write("alpha 命中", MemoryStream.FACTUAL, personality=p,
                       importance=MemoryImportance.MEDIUM, current_step=1)

    calls: list[str] = []
    inner = system._embedding  # noqa: SLF001

    class _CountingEmbedding:
        dimension = inner.dimension

        async def embed(self, text: str):
            calls.append(text)
            return await inner.embed(text)

    system._embedding = _CountingEmbedding()  # noqa: SLF001
    results = await system.retrieve("alpha", current_step=9, top_k=5, stream=MemoryStream.FACTUAL)

    assert [m.stored_content for m in results] == ["alpha 命中"]
    assert calls == ["alpha"]                     # only the query is embedded, not the memory text
    assert results[0].last_accessed_step == 9
    (payload,) = [
        r.payload for r in await container.vector_store.list_all(
            system._namespace(MemoryStream.FACTUAL)  # noqa: SLF001
        ) if r.id == results[0].id
    ]
    assert payload["last_accessed_step"] == 9     # persisted to the vector store too


@pytest.mark.asyncio
async def test_touch_writeback_falls_back_when_record_absent(container) -> None:
    """If the record isn't in the vector store (an earlier write failed), update_payload returns
    False and we fall back to a full embed + upsert. Otherwise the memory would never be recalled."""
    system = _make_memory_system(container)
    await system.ensure_collections()
    memory = Memory(id="m-absent", stream=MemoryStream.FACTUAL, raw_content="有人路过",
                    stored_content="有人路过", importance=0.5)
    system._entries[memory.id] = memory  # noqa: SLF001 — only in the cache, not in the vector store

    await system._persist_payload(memory)  # noqa: SLF001

    assert len(await container.vector_store.list_all(
        system._namespace(MemoryStream.FACTUAL)  # noqa: SLF001
    )) == 1


@pytest.mark.asyncio
async def test_record_event_factual_survives_background_write_failure(container) -> None:
    """Memory writes happen in the background, and a failed embed/upsert there must not propagate or
    stop the run. record_event has already put the factual memory in _entries, so it is usable for
    the session; only embedding recall misses it."""
    system = _make_memory_system(container)

    async def _boom(*args, **kwargs):
        raise RuntimeError("provider down")

    system._persist_vector = _boom  # type: ignore[method-assign]  # background write fails

    entries = await system.record_event(
        current_step=3,
        raw_content="有人路过",
        personality=_make_personality(),
        importance=0.2,  # low importance: no experiential, so a single stream
    )
    # record_event returns the provisional factual memory right away; the background failure
    # doesn't affect it.
    assert [m.stream for m in entries] == [MemoryStream.FACTUAL]
    factual_id = entries[0].id
    # The worker absorbs the failure with a warning, so drain doesn't raise.
    await system.drain_writes()
    # The factual memory is still cached and usable for the session.
    assert factual_id in system._entries


@pytest.mark.asyncio
async def test_record_event_provisional_visible_in_cache_before_vector_upsert(container) -> None:
    """With background writes, a factual memory is in the cache as soon as record_event returns
    (sample_recent_events sees it), but vector search only finds it after the background upsert
    (drain_writes)."""
    system = _make_memory_system(container, is_main_character=False)
    await system.ensure_collections()
    p = _make_personality()
    entries = await system.record_event(
        current_step=5, raw_content="有人在殿前布防", personality=p,
        importance=MemoryImportance.MEDIUM,  # passing importance skips the importance LLM call
    )
    factual_id = entries[0].id
    # visible in the cache right away (sample_recent_events scans _entries)
    events = system.sample_recent_events(current_step=5, lookback=3, top_k=10)
    assert any(f is not None and f.id == factual_id for f, _ in events)
    # not yet visible to vector search (not upserted)
    before = await system.retrieve("布防", current_step=5, top_k=10, stream=MemoryStream.FACTUAL)
    assert factual_id not in {m.id for m in before}
    # after drain the upsert is done and it can be recalled
    await system.drain_writes()
    after = await system.retrieve("布防", current_step=5, top_k=10, stream=MemoryStream.FACTUAL)
    assert factual_id in {m.id for m in after}


@pytest.mark.asyncio
async def test_record_event_reserves_paired_ids_synchronously(container) -> None:
    """ids are reserved synchronously: both streams share event_group_id, and the factual seq is
    lower than the experiential seq."""
    provider = SequentialMockLLM(["我心头一紧。"])  # experiential rewrite (background)
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    await system.ensure_collections()
    p = _make_personality(emotion=EmotionState(primary="fear", intensity=0.7, valence=-0.5))
    entries = await system.record_event(
        current_step=4, raw_content="有人闯入", personality=p, importance=MemoryImportance.HIGH,
    )
    assert {m.stream for m in entries} == {MemoryStream.FACTUAL, MemoryStream.EXPERIENTIAL}
    assert len({m.event_group_id for m in entries}) == 1
    factual = next(m for m in entries if m.stream == MemoryStream.FACTUAL)
    experiential = next(m for m in entries if m.stream == MemoryStream.EXPERIENTIAL)
    assert int(factual.id.rsplit("-", 1)[1]) < int(experiential.id.rsplit("-", 1)[1])


@pytest.mark.asyncio
async def test_drain_writes_completes_experiential_rewrite(container) -> None:
    """A provisional experiential memory stores the raw text as a placeholder; after drain the
    background first-person rewrite replaces it in place."""
    provider = SequentialMockLLM(["我把它读作对我的挑衅。"])
    system = _make_memory_system(container, provider=provider, is_main_character=True)
    await system.ensure_collections()
    p = _make_personality(emotion=EmotionState(primary="anger", intensity=0.7, valence=-0.5))
    entries = await system.record_event(
        current_step=2, raw_content="对方拔剑", personality=p, importance=MemoryImportance.HIGH,
    )
    experiential = next(m for m in entries if m.stream == MemoryStream.EXPERIENTIAL)
    assert experiential.stored_content == "对方拔剑"
    # Until it is rewritten, experiential stays out of _entries: its content is still raw and
    # identical to factual, so caching it adds nothing.
    assert experiential.id not in system._entries
    await system.drain_writes()
    assert experiential.stored_content == "我把它读作对我的挑衅。"  # background rewrite
    assert experiential.id in system._entries  # added to _entries only once rewritten


def test_foiled_attempt_is_transient_never_persisted(container) -> None:
    """A not_executed foiled attempt goes only to the transient ring buffer: not to _entries, not
    to the vector store, and it doesn't consume a _seq.

    This keeps foiled attempts unpersisted and out of reach of the deeper consumers
    (relation_evolution/reflection/summary/goal-eval), which read only _entries and the vectors.
    """
    system = _make_memory_system(container)
    system.note_foiled_attempt(step=5, text="我本想与父亲相谈，却因对方不在场未能如愿。")
    assert system._entries == {}
    assert system._seq == 0
    # The transient buffer still has it for decide. It renders like a normal memory, with a
    # relative-recency prefix (step=now gives "刚刚").
    got = system.recent_foiled_attempts(5)
    assert got == ["（刚刚）我本想与父亲相谈，却因对方不在场未能如愿。"]


def test_recent_foiled_attempts_windowed_and_chrono(container) -> None:
    """decide gets only the foiled attempts inside the window, oldest first. Older ones no longer
    bear on the current situation."""
    system = _make_memory_system(container)
    system.note_foiled_attempt(step=1, text="旧一")
    system.note_foiled_attempt(step=2, text="旧二")
    system.note_foiled_attempt(step=9, text="新一")
    system.note_foiled_attempt(step=10, text="新二")
    # Default window=_FOILED_LOOKBACK_STEPS (5) keeps step>=10-5=5, so only steps 9 and 10 remain.
    # Entries carry a relative-recency prefix like normal memories, so strip the "（…）" prefix and
    # compare only the content order.
    got = system.recent_foiled_attempts(10)
    assert all(t.startswith("（") for t in got)
    assert [t.split("）", 1)[-1] for t in got] == ["新一", "新二"]
    # Empty text is dropped and takes no slot.
    system.note_foiled_attempt(step=10, text="   ")
    assert [t.split("）", 1)[-1] for t in system.recent_foiled_attempts(10)] == ["新一", "新二"]


# ---------------------------------------------------------------------------
# Merging foiled attempts: one entry per thing, with a count
#
# One entry per step would repeat the same sentence in the decision prompt, and repetition makes
# the agent likelier to try again and go in circles. Merging shows it once with the count.
# ---------------------------------------------------------------------------


def _bodies(lines: list[str]) -> list[str]:
    """Strip the recency prefix and keep only the content, as the other foiled-attempt tests do."""
    return [t.split("）", 1)[-1] for t in lines]


def test_foiled_same_key_merges_into_one_line_with_count(container) -> None:
    """Repeated foiled attempts with the same key merge into one entry with a count, using the
    newest entry's text, not the oldest."""
    system = _make_memory_system(container)
    key = ("talk", "agent-shimin")
    # Three wordings of the same thing, all literally different (the shape of a real loop).
    system.note_foiled_attempt(step=1, text="我想快步追上李世民，逼他今夜定策。",
                               key=key, gist="快步追上李世民，逼他今夜定策")
    system.note_foiled_attempt(step=2, text="我想拦住李世民，逼他今夜就定策。",
                               key=key, gist="拦住李世民，逼他今夜就定策")
    system.note_foiled_attempt(step=3, text="我想走到李世民面前，逼他今夜必须定策。",
                               key=key, gist="走到李世民面前，逼他今夜必须定策")
    got = system.recent_foiled_attempts(3)
    assert len(got) == 1
    assert "我想走到李世民面前，逼他今夜必须定策。" in got[0]   # the newest entry represents the group
    assert "快步追上" not in got[0]
    assert "近来同一件事我已试过3次。" in got[0]


def test_foiled_single_attempt_carries_no_count(container) -> None:
    """A single attempt gets no count, so one entry doesn't add noise to the prompt."""
    system = _make_memory_system(container)
    system.note_foiled_attempt(step=1, text="我想追上他。", key=("talk", "agent-x"), gist="追上他")
    got = system.recent_foiled_attempts(1)
    assert got == ["（刚刚）我想追上他。"]


def test_foiled_different_keys_never_merge(container) -> None:
    """Different keys get one line each. Merging only happens within the same actor and action."""
    system = _make_memory_system(container)
    system.note_foiled_attempt(step=1, text="我想找甲说话。", key=("talk", "agent-a"), gist="找甲说话")
    system.note_foiled_attempt(step=2, text="我想找乙说话。", key=("talk", "agent-b"), gist="找乙说话")
    system.note_foiled_attempt(step=3, text="我想去城门。", key=("move", "gate"), gist="去城门")
    got = system.recent_foiled_attempts(3)
    assert _bodies(got) == ["我想找甲说话。", "我想找乙说话。", "我想去城门。"]
    assert not any("试过" in t for t in got)


def test_foiled_keyless_entries_never_merge(container) -> None:
    """Actions without an object (WORK/REST etc.) have key None, so each entry is its own group.

    Two different WORKs (drafting an edict vs. reading files) shown as "tried 2 times" would be
    wrong, so a missing key opts out of merging.
    """
    system = _make_memory_system(container)
    system.note_foiled_attempt(step=1, text="我想拟一份诏书。")
    system.note_foiled_attempt(step=2, text="我想细读那卷案宗。")
    got = system.recent_foiled_attempts(2)
    assert _bodies(got) == ["我想拟一份诏书。", "我想细读那卷案宗。"]
    assert not any("试过" in t for t in got)


def test_foiled_same_key_but_unrelated_content_stays_split(container) -> None:
    """Similarity guard: the same key isn't enough to merge; clearly different content is two
    attempts (same person, both TALK: "explain why I stopped talking" vs. "push him to decide
    tonight").
    """
    system = _make_memory_system(container)
    key = ("talk", "agent-shimin")
    system.note_foiled_attempt(
        step=1, text="我想解释方才为何收住话头。", key=key, gist="解释方才为何收住话头",
    )
    system.note_foiled_attempt(
        step=2, text="我想逼他今夜就定下决断。", key=key, gist="逼他今夜就定下决断",
    )
    got = system.recent_foiled_attempts(2)
    assert len(got) == 2
    assert not any("试过" in t for t in got)


def test_foiled_merge_compares_gist_not_display_text(container) -> None:
    """The guard compares gist, not text.

    Every text shares the "…却因…未能如愿" wrapper, which inflates similarity between any two
    entries enough to defeat the guard. Here the two texts have almost the same wrapper but very
    different gists, and they must not merge.
    """
    system = _make_memory_system(container)
    key = ("talk", "agent-shimin")
    shell = "，却因李世民正忙于他事未能如愿。"
    system.note_foiled_attempt(step=1, text=f"我本想解释方才为何收住话头{shell}",
                               key=key, gist="解释方才为何收住话头")
    system.note_foiled_attempt(step=2, text=f"我本想逼他今夜就定下决断{shell}",
                               key=key, gist="逼他今夜就定下决断")
    assert len(system.recent_foiled_attempts(2)) == 2


def test_foiled_no_object_merges_only_on_high_similarity(container) -> None:
    """Actions without an object have only half a key (type, no object); a higher similarity
    threshold makes up the other half.

    The same WORK reworded must merge into one counted entry. Two different WORKs must stay
    separate: decide reads the count to judge whether to change strategy, so a wrong merge produces
    a false count.
    """
    system = _make_memory_system(container)
    work = ("work", "")
    system.note_foiled_attempt(step=1, text="我想去密室取那卷帛书细看。", key=work,
                               gist="我独自前往秦王府密室，取出无忌那卷帛书细看。")
    system.note_foiled_attempt(step=2, text="我想去密室把那卷帛书摊开细看。", key=work,
                               gist="我走向密室，取出无忌那卷帛书，摊开细看上面的每一字。")
    got = system.recent_foiled_attempts(2)
    assert len(got) == 1 and "试过2次" in got[0]

    other = _make_memory_system(container)
    other.note_foiled_attempt(step=1, text="我想拟一份诏书。", key=work, gist="独自拟一份诏书")
    other.note_foiled_attempt(step=2, text="我想清点府中甲仗。", key=work, gist="清点府中甲仗数目")
    assert len(other.recent_foiled_attempts(2)) == 2


def test_foiled_no_object_keeps_action_types_apart(container) -> None:
    """The type half of the key is still required: a WORK and a REST with similar wording are not
    the same thing."""
    system = _make_memory_system(container)
    gist = "我独自坐在案前，闭目理清明日朝会如何应对。"
    system.note_foiled_attempt(step=1, text="我想坐下理清明日的事。", key=("work", ""), gist=gist)
    system.note_foiled_attempt(step=2, text="我想坐下歇一歇。", key=("rest", ""), gist=gist)
    assert len(system.recent_foiled_attempts(2)) == 2


def test_foiled_count_only_covers_the_window(container) -> None:
    """Only attempts inside the window are counted; older ones neither appear nor count."""
    system = _make_memory_system(container)
    key = ("talk", "agent-x")
    for step in (1, 2, 20, 21, 22):
        system.note_foiled_attempt(step=step, text=f"我想找他谈第{step}回。", key=key, gist="找他谈")
    got = system.recent_foiled_attempts(22)      # window=5 leaves steps 20/21/22
    assert len(got) == 1
    assert "近来同一件事我已试过3次。" in got[0]


def test_foiled_merged_lines_keep_chrono_order(container) -> None:
    """Merged groups stay in ascending order of their representative's step, so merging keeps the
    MEMORY_ORDER_HINT promise."""
    system = _make_memory_system(container)
    system.note_foiled_attempt(step=1, text="我想去城门。", key=("move", "gate"), gist="去城门")
    system.note_foiled_attempt(step=2, text="我想找甲说话。", key=("talk", "a"), gist="找甲说话")
    system.note_foiled_attempt(step=3, text="我想再去城门看看。", key=("move", "gate"), gist="去城门看看")
    got = system.recent_foiled_attempts(3)
    # The move group's representative is step 3 (its newest), so it comes after the step-2 talk.
    bodies = _bodies(got)
    assert len(bodies) == 2
    assert bodies[0] == "我想找甲说话。"
    assert bodies[1].startswith("我想再去城门看看。")
    assert "试过2次" in bodies[1]


def test_foiled_merged_line_leaks_no_id_or_step(container) -> None:
    """Merging and counting must not leak the key's id or step into the text."""
    system = _make_memory_system(container)
    key = ("talk", "agent-fae10bdcbc")
    system.note_foiled_attempt(step=7, text="我想找他谈。", key=key, gist="找他谈")
    system.note_foiled_attempt(step=8, text="我想再找他谈。", key=key, gist="再找他谈")
    line = system.recent_foiled_attempts(8)[0]
    assert "agent-fae10bdcbc" not in line
    assert "talk" not in line
    assert "第7步" not in line and "第8步" not in line


def test_foiled_buffer_bounded(container) -> None:
    """The ring buffer is bounded (maxlen) and keeps only the newest N entries."""
    from agent.memory import _FOILED_BUFFER_MAXLEN

    system = _make_memory_system(container)
    for i in range(_FOILED_BUFFER_MAXLEN + 4):
        system.note_foiled_attempt(step=100 + i, text=f"扑空{i}")
    # All within the window (consecutive steps), but only the last maxlen are kept.
    kept = system.recent_foiled_attempts(200, window=1000)
    assert len(kept) == _FOILED_BUFFER_MAXLEN
    contents = [t.split("）", 1)[-1] for t in kept]  # strip the recency prefix to compare content
    assert contents[-1] == f"扑空{_FOILED_BUFFER_MAXLEN + 3}"
    assert contents[0] == f"扑空{4}"


@pytest.mark.asyncio
async def test_deferred_experiential_write_traced_at_event_step(container: object) -> None:
    """The refine worker is a long-lived ``create_task`` whose contextvars freeze at creation,
    so without ``observe_step`` every background experiential rewrite is mis-attributed to that
    one frozen step. An event at step 13 must trace its rewrite at step 13, even when the
    ambient (frozen worker) log-context step says 2."""
    sink = InMemoryTraceSink()
    provider = SequentialMockLLM(["下令诛杀元吉，我心中只有沉甸甸的悲凉。"])  # experiential rewrite
    system = MemorySystem(
        LLMRouter({scene: provider for scene in LLMScene}, trace_sink=sink),
        container.embedding,  # type: ignore[attr-defined]
        container.vector_store,  # type: ignore[attr-defined]
        world_id="w", agent_id="a1", is_main_character=True, retrieval_score_floor=-1.0,
    )
    personality = _make_personality(
        emotion=EmotionState(primary="sadness", intensity=0.7, valence=-0.6),
    )
    # Ambient context = the frozen worker-creation step (2), NOT the event step. The write
    # queue's worker is created here (first enqueue) and freezes this context.
    clear_log_context()
    set_log_context(world_id="w", agent_id="a1", step="2")
    await system.record_event(
        current_step=13,
        raw_content="在玄武门，我下令诛杀元吉。",
        experiential_content="在玄武门，我下令诛杀元吉。",
        personality=personality,
        importance=MemoryImportance.HIGH,  # skip the importance LLM; go straight to the rewrite
    )
    await system.drain_writes()

    exp = [c for c in sink.llm_calls if c.scene == LLMScene.MEMORY_SUMMARIZATION.value]
    assert exp, "the experiential rewrite should have been traced"
    assert all(c.step == 13 for c in exp), (
        f"deferred write must be traced at its event step 13, not the frozen worker step; "
        f"got {[c.step for c in exp]}"
    )


@pytest.mark.asyncio
async def test_persist_failure_names_the_exception_even_when_its_message_is_empty(
    container: object, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    system = _make_memory_system(container)

    async def _timeout(**_kwargs: object) -> None:
        raise TimeoutError()

    monkeypatch.setattr(container.vector_store, "upsert", _timeout)  # type: ignore[attr-defined]
    memory = Memory(
        id="agent-1-timeout", stream=MemoryStream.FACTUAL, agent_id="agent-1",
        stored_content="A fact.", importance=0.5, created_step=1, kind="event",
    )
    with caplog.at_level("WARNING"):
        assert await system._persist(memory) is False
    record = next(r for r in caplog.records if r.getMessage() == "memory_persist_failed")
    assert "TimeoutError" in record.error

