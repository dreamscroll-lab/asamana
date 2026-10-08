"""Unit tests for RelationEvolution — periodic main-character relation evaluator.

A functional, third-person objective judge that produces label + summary; ids stay out of the
prompt (names + indices instead), and evolution is the sole owner of main characters' summaries."""

from __future__ import annotations

import pytest

from agent.memory import MemorySystem
from agent.memory_types import MemoryImportance, MemoryStream
from agent.personality import EmotionState, PersonalityLayer, SoulLayer, StateLayer
from agent.relation import RelationSystem
from agent.relation_evolution import RelationEvolution
from core.interfaces.agent_store import AgentRelation, relation_has_substance
from core.interfaces.llm import LLMProvider, LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider, SequentialMockLLM


def _router(provider: LLMProvider) -> LLMRouter:
    return LLMRouter({scene: provider for scene in LLMScene})


def _personality(*, name: str = "主角", role: str = "太子") -> PersonalityLayer:
    soul = SoulLayer(
        name=name,
        role=role,
        agent_id="agent-self",
        core_traits=["谨慎"],
        core_values=["忠义"],
    )
    state = StateLayer(emotion=EmotionState(primary="neutral", intensity=0.2, valence=0.0))
    return PersonalityLayer(soul=soul, state=state)


def _make_memory_system(container: object, *, agent_id: str = "agent-self") -> MemorySystem:
    return MemorySystem(
        _router(MockLLMProvider(fixed_response="一段内心独白。")),
        container.embedding,  # type: ignore[attr-defined]
        container.vector_store,  # type: ignore[attr-defined]
        world_id="w",
        agent_id=agent_id,
        is_main_character=True,
    )


def _make_relation_system(container: object, *, agent_id: str = "agent-self") -> RelationSystem:
    return RelationSystem(
        container.agent_store,  # type: ignore[attr-defined]
        world_id="w",
        agent_id=agent_id,
    )


async def _seed_event_about(
    memory: MemorySystem,
    *,
    target_id: str,
    step: int,
    content: str,
) -> None:
    """Write a factual memory mentioning target_id at the given step."""
    personality = _personality()
    await memory.write(
        content,
        MemoryStream.FACTUAL,
        personality=personality,
        related_agents=[target_id],
        importance=MemoryImportance.MEDIUM,
        current_step=step,
    )


async def _seed_existing_relation(
    relation: RelationSystem,
    *,
    target_id: str,
    labels: list[str] | None = None,
    trust: float = 0.5,
    affection: float = 0.0,
    to_name: str = "",
    to_gender: str = "",
) -> AgentRelation:
    rel = await relation.get_or_create(target_id)
    rel.trust_objective = trust
    rel.affection_objective = affection
    rel.labels = list(labels or [])
    rel.to_name = to_name
    rel.to_gender = to_gender
    await relation._store.save_relation(rel)
    return rel


# ---------------------------------------------------------------------------
# (1) Happy path — labels overwritten, returns (id, labels, summary)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_overwrites_labels(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-foe", step=18, content="他在朝堂当众反对我")
    await _seed_existing_relation(
        relation, target_id="agent-foe", labels=["父子:儿子"], trust=0.6, to_name="张三"
    )

    provider = SequentialMockLLM([
        '{"updates": [{"target_index": 1, "rationale": "他朝堂决裂", '
        '"labels": ["父子:儿子", "敌人"]}]}',
    ])
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == [("agent-foe", ["父子:儿子", "敌人"], "")]
    stored = await container.agent_store.load_relation("w", "agent-self", "agent-foe")  # type: ignore[attr-defined]
    assert stored is not None
    assert stored.labels == ["父子:儿子", "敌人"]


# ---------------------------------------------------------------------------
# (2) Empty updates → no writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_updates_writes_nothing(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-x", step=18, content="平淡的一次见面")
    await _seed_existing_relation(relation, target_id="agent-x", labels=["同事"], to_name="某乙")

    provider = SequentialMockLLM(['{"updates": []}'])
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == []
    stored = await container.agent_store.load_relation("w", "agent-self", "agent-x")  # type: ignore[attr-defined]
    assert stored is not None
    assert stored.labels == ["同事"]


# ---------------------------------------------------------------------------
# (3) Invalid target_index → entry dropped, others still applied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_target_index_skipped_others_applied(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-a", step=18, content="A 帮过我")
    await _seed_event_about(memory, target_id="agent-b", step=18, content="B 背叛了我")
    await _seed_existing_relation(relation, target_id="agent-a", labels=["陌生"], to_name="甲")
    await _seed_existing_relation(relation, target_id="agent-b", labels=["陌生"], to_name="乙")

    # Candidates sorted by id: #1=agent-a, #2=agent-b
    # target_index=99 is out of range → dropped; target_index=2 → applied to agent-b
    provider = SequentialMockLLM([
        '{"updates": ['
        '{"target_index": 99, "rationale": "幻觉", "labels": ["叛徒"]},'
        '{"target_index": 2, "rationale": "B 背叛", "labels": ["敌人"]}'
        ']}'
    ])
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == [("agent-b", ["敌人"], "")]
    a_stored = await container.agent_store.load_relation("w", "agent-self", "agent-a")  # type: ignore[attr-defined]
    b_stored = await container.agent_store.load_relation("w", "agent-self", "agent-b")  # type: ignore[attr-defined]
    assert a_stored is not None and a_stored.labels == ["陌生"]
    assert b_stored is not None and b_stored.labels == ["敌人"]


# ---------------------------------------------------------------------------
# (4) Malformed labels + no summary → skipped, others ok
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_labels_skipped(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-a", step=18, content="...")
    await _seed_event_about(memory, target_id="agent-b", step=18, content="...")
    await _seed_existing_relation(relation, target_id="agent-a", labels=["原状"], to_name="甲")
    await _seed_existing_relation(relation, target_id="agent-b", labels=["原状"], to_name="乙")

    # Non-array labels with no summary → skipped; all-empty labels with no summary → skipped.
    provider = SequentialMockLLM([
        '{"updates": ['
        '{"target_index": 1, "rationale": "non-list", "labels": "字符串非数组"},'
        '{"target_index": 2, "rationale": "all empty", "labels": ["   ", ""]}'
        ']}'
    ])
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == []
    for tid in ("agent-a", "agent-b"):
        rel = await container.agent_store.load_relation("w", "agent-self", tid)  # type: ignore[attr-defined]
        assert rel is not None and rel.labels == ["原状"]


# ---------------------------------------------------------------------------
# (5) LLM exception → Rule 1 fallback returns empty list, doesn't bubble
# ---------------------------------------------------------------------------


class _ExplodingLLM(LLMProvider):
    async def complete(self, messages, temperature: float = 0.7, max_tokens: int = 1000,
        **kwargs,
    ):  # type: ignore[override]
        raise RuntimeError("LLM provider is down")

    async def stream(self, messages, temperature: float = 0.7):  # type: ignore[override]
        if False:
            yield ""


@pytest.mark.asyncio
async def test_llm_exception_swallowed_returns_empty(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-x", step=18, content="something")
    await _seed_existing_relation(relation, target_id="agent-x", labels=["原状"], to_name="某")

    evolver = RelationEvolution(
        _router(_ExplodingLLM()), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == []
    stored = await container.agent_store.load_relation("w", "agent-self", "agent-x")  # type: ignore[attr-defined]
    assert stored is not None and stored.labels == ["原状"]


# ---------------------------------------------------------------------------
# (6) No candidates in window → no LLM call, returns empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_candidates_no_llm_call(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)

    provider = MockLLMProvider(fixed_response='{"updates": []}')
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == []
    assert provider.call_history == []


# ---------------------------------------------------------------------------
# (7) Self id excluded from candidate set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_id_excluded_from_candidates(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    personality = _personality()
    await memory.write(
        "I reflected on my own choices.",
        MemoryStream.EXPERIENTIAL,
        personality=personality,
        related_agents=["agent-self"],
        importance=MemoryImportance.MEDIUM,
        current_step=18,
    )

    provider = MockLLMProvider(fixed_response='{"updates": []}')
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == []
    assert provider.call_history == []


# ---------------------------------------------------------------------------
# (8) MemorySystem helpers — collect/list correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_helpers_collect_and_list(container: object) -> None:
    memory = _make_memory_system(container)
    personality = _personality()

    await memory.write(
        "old event", MemoryStream.FACTUAL, personality=personality,
        related_agents=["agent-old"], importance=MemoryImportance.LOW, current_step=5,
    )
    await memory.write(
        "alice helped me", MemoryStream.FACTUAL, personality=personality,
        related_agents=["agent-alice"], importance=MemoryImportance.MEDIUM, current_step=20,
    )
    await memory.write(
        "bob ignored me", MemoryStream.EXPERIENTIAL, personality=personality,
        related_agents=["agent-bob"], importance=MemoryImportance.MEDIUM, current_step=30,
    )
    await memory.write(
        "alice tried to help again", MemoryStream.FACTUAL, personality=personality,
        related_agents=["agent-alice"], importance=MemoryImportance.MEDIUM, current_step=35,
    )

    collected = memory.collect_recent_related_agents(current_step=40, lookback_steps=25)
    assert collected == {"agent-alice", "agent-bob"}

    alice_mentions = memory.list_recent_memories_mentioning(
        "agent-alice", current_step=40, lookback_steps=25
    )
    assert [m.created_step for m in alice_mentions] == [20, 35]

    old_mentions = memory.list_recent_memories_mentioning(
        "agent-old", current_step=40, lookback_steps=25
    )
    assert old_mentions == []


# ---------------------------------------------------------------------------
# (9) Prompt uses name + #N, never leaks id (CLAUDE.md: ids don't cross the LLM boundary)
# ---------------------------------------------------------------------------


def _rel(**kw: object) -> AgentRelation:
    base = dict(world_id="w", from_id="a", to_id="b", trust_objective=0.5,
                affection_objective=0.0, updated_step=0)
    base.update(kw)
    return AgentRelation(**base)  # type: ignore[arg-type]


def test_has_substance_predicate() -> None:
    """The shared 'is this a real relationship' rule: labels OR summary OR interaction.
    interaction_count alone is deliberately NOT sufficient as the sole gate — a seeded
    labeled bond with ic==0 must count."""
    # Bare baseline record (ambient co-occurrence) — no substance.
    assert not _rel().has_substance()
    assert not relation_has_substance(labels=[], history_summary="", interaction_count=0)
    # Seeded bond: labels, but never interacted (ic==0) — MUST count.
    assert _rel(labels=["兄弟:弟弟"], trust_objective=0.1).has_substance()
    # Runtime engagement.
    assert _rel(interaction_count=1).has_substance()
    # Narrative summary alone.
    assert _rel(history_summary="近来渐生嫌隙").has_substance()
    # Off-baseline trust/affection alone isn't checked: float-fragile, and unreachable at runtime
    # without ic>0.


@pytest.mark.asyncio
async def test_insubstantial_ambient_third_party_excluded(container: object) -> None:
    """collect_recent_related_agents sweeps up third parties the agent only OVERHEARD in
    ambient (zero interaction, baseline values, no labels) — those are not relationships and
    must not reach the evolution judge. A seeded bond (labels, interaction_count==0) MUST
    still be evaluated: interaction count alone would wrongly drop it."""
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)

    # Ambient-only third party: a memory names them, but the relation stays a bare baseline
    # record (get_or_create) — no labels, no interaction, no summary. Must be filtered out.
    await _seed_event_about(
        memory, target_id="agent-overheard", step=18, content="在东宫，甲与乙在交谈。"
    )
    # Seeded bond: labels present but interaction_count == 0 (never engaged at runtime).
    # Must survive the filter — the ic-only test would drop the protagonist's own brother.
    await _seed_event_about(memory, target_id="agent-brother", step=18, content="兄长的身影")
    await _seed_existing_relation(
        relation, target_id="agent-brother", labels=["兄弟:弟弟", "政敌"], trust=0.1, to_name="李建成"
    )

    provider = MockLLMProvider(fixed_response='{"updates": []}')
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )
    await evolver.evaluate(20)

    # One LLM call was made, and its candidate list holds the seeded bond only.
    assert len(provider.call_history) == 1
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "李建成" in prompt                       # seeded bond kept (via labels, ic==0)
    assert "某位与你有往来的人" not in prompt        # the overheard third party is gone
    # And the baseline record itself was never labeled/evolved.
    overheard = await container.agent_store.load_relation("w", "agent-self", "agent-overheard")  # type: ignore[attr-defined]
    assert overheard is None or not overheard.has_substance()


@pytest.mark.asyncio
async def test_all_candidates_insubstantial_no_llm_call(container: object) -> None:
    """If every recent-memory agent is an insubstantial baseline record, no LLM fires —
    same short-circuit as having no candidates at all."""
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-a", step=18, content="旁观甲路过")
    await _seed_event_about(memory, target_id="agent-b", step=18, content="旁观乙路过")

    provider = MockLLMProvider(fixed_response='{"updates": []}')
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == []
    assert provider.call_history == []


@pytest.mark.asyncio
async def test_prompt_uses_name_not_id(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-foe", step=18, content="他当众反对我")
    await _seed_existing_relation(relation, target_id="agent-foe", labels=["对手"], to_name="张三")

    provider = MockLLMProvider(fixed_response='{"updates": []}')
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )
    await evolver.evaluate(20)

    # Join the [system, user] split for substring/order assertions.
    prompt = "\n".join(m.content for m in provider.call_history[0])
    assert "agent-foe" not in prompt   # no ids in the prompt
    assert "id:" not in prompt
    assert "张三" in prompt             # uses to_name


# ---------------------------------------------------------------------------
# (10) summary written; summary-only update (labels unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_only_update_keeps_labels(container: object) -> None:
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-x", step=18, content="一次试探性的交谈")
    await _seed_existing_relation(relation, target_id="agent-x", labels=["盟友"], to_name="某甲")

    # Only summary, no labels: labels stay unchanged, summary is written.
    provider = SequentialMockLLM([
        '{"updates": [{"target_index": 1, "rationale": "数次试探后略有疏离", '
        '"summary": "近期数次试探，盟谊未变但多了几分提防。"}]}'
    ])
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )

    applied = await evolver.evaluate(20)

    assert applied == [("agent-x", [], "近期数次试探，盟谊未变但多了几分提防。")]
    stored = await container.agent_store.load_relation("w", "agent-self", "agent-x")  # type: ignore[attr-defined]
    assert stored is not None
    assert stored.labels == ["盟友"]   # unchanged
    assert stored.history_summary == "近期数次试探，盟谊未变但多了几分提防。"


# ---------------------------------------------------------------------------
# (11) The only summary write path: apply_interaction doesn't write, update_history_summary does
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_only_written_by_llm_path(container: object) -> None:
    # For any agent, apply_interaction doesn't write history_summary (mechanical concatenation isn't
    # a valid write path).
    rel_system = _make_relation_system(container, agent_id="any")
    await rel_system.apply_interaction("t", trust_delta=0.1, affection_delta=0.0, step=1)
    rel = await rel_system.get_or_create("t")
    assert rel.history_summary == ""   # not written: left empty

    # Only the LLM path (update_history_summary) writes.
    await rel_system.update_history_summary("t", "近期一次交锋后略有提防。", step=1)
    rel = await rel_system.get_or_create("t")
    assert rel.history_summary == "近期一次交锋后略有提防。"


@pytest.mark.asyncio
async def test_weighing_a_stranger_does_not_bring_a_relation_into_being(container: object) -> None:
    """A third party only observed in ambient: having considered them doesn't mean having a relation
    with them.

    If building the context used ``get_or_create``, that row would be persisted before "is this a
    relation" is decided, then picked up by ``significant_relations`` into the prompt as a noisy
    "关系：某人：[未明确]" line.
    """
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    # Mentioned in memory (observed in ambient) but never interacted with: no relation is created.
    await _seed_event_about(memory, target_id="agent-stranger", step=18, content="远远看见一个人经过")

    evolver = RelationEvolution(
        _router(SequentialMockLLM([])), memory, relation, _personality(), agent_id="agent-self"
    )
    applied = await evolver.evaluate(20)

    assert applied == []
    stored = await container.agent_store.load_relation("w", "agent-self", "agent-stranger")  # type: ignore[attr-defined]
    assert stored is None, "权衡了一遍就凭空长出一段关系"


@pytest.mark.asyncio
async def test_target_block_carries_gender_from_the_stored_relation(container: object) -> None:
    """The gender cached on the relation must reach the judge prompt; gendered kinship labels
    (father-daughter, siblings…) depend on it.

    This is also why `AgentRelation.to_gender` is seeded at build time: people in seed relations
    have mostly never met face to face, so `perceive()` never backfills it, and this record is the
    only source of gender.
    """
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_existing_relation(
        relation, target_id="agent-x", labels=["血亲"], to_name="某甲", to_gender="女",
    )
    evo = RelationEvolution(
        _router(SequentialMockLLM([])), memory, relation, _personality(), agent_id="agent-self"
    )
    ctx = await evo._build_target_context(target_id="agent-x", current_step=20)
    assert ctx.target_gender == "女"
    _system, user = evo._build_prompt([ctx])
    assert "#1 某甲（女）" in user


@pytest.mark.asyncio
async def test_closed_world_rule_reaches_the_judge(container: object) -> None:
    """The judge must carry the closed-world rules: the summary it writes lands in history_summary
    and is reread later.

    Without them, a relation that stops on the eve of a coup gets summarized as if the coup had
    happened: the model knows the history and supplies an ending that never happened. Assert after
    rendering, so the constant isn't sent out as a literal.
    """
    memory = _make_memory_system(container)
    relation = _make_relation_system(container)
    await _seed_event_about(memory, target_id="agent-x", step=18, content="一次密谈")
    await _seed_existing_relation(relation, target_id="agent-x", labels=["同僚"], to_name="某甲")

    provider = MockLLMProvider(fixed_response='{"updates": []}')
    evolver = RelationEvolution(
        _router(provider), memory, relation, _personality(), agent_id="agent-self"
    )
    await evolver.evaluate(20)

    from core.prompts import CLOSED_WORLD_FACT_RULE

    system = provider.call_history[0][0].content
    assert CLOSED_WORLD_FACT_RULE in system
