"""Unit tests for the agent relation subsystem."""

from __future__ import annotations

import pytest

from agent.personality import EmotionState
from agent.relation import (
    NEUTRAL_AFFECTION,
    NEUTRAL_TRUST,
    PerceivedRelation,
    RelationDirection,
    RelationSystem,
    describe_relations,
    parse_relation_direction,
    render_relation_context,
)
from core.interfaces.agent_store import AgentRelation


def _rel(**kw) -> AgentRelation:
    base = dict(
        world_id="w", from_id="a", to_id="b", trust_objective=0.5,
        affection_objective=0.0, updated_step=0,
    )
    base.update(kw)
    return AgentRelation(**base)  # type: ignore[arg-type]


def test_render_relation_context_objective_values_and_labels() -> None:
    text = render_relation_context([
        ("张三", _rel(trust_objective=0.1, affection_objective=-0.6,
                     labels=["政敌:对手"], history_summary="近期数次交锋后信任下降")),
    ])
    assert "张三" in text
    assert "政敌:对手" in text
    assert "0.10" in text and "-0.60" in text   # objective values
    assert "近期数次交锋后信任下降" in text
    assert "信任度 trust" in text                 # includes the scale legend


def test_render_relation_context_empty_returns_blank() -> None:
    assert render_relation_context([]) == ""


@pytest.mark.asyncio
async def test_load_existing_does_not_create(container: object) -> None:
    system = _make_relation_system(container)
    # Missing → None, and nothing is persisted (storage stays clean)
    assert await system.load_existing("ghost") is None
    assert await container.agent_store.load_relation("world-1", "agent-1", "ghost") is None  # type: ignore[attr-defined]


def _make_relation_system(container: object, *, world_id: str = "world-1", agent_id: str = "agent-1") -> RelationSystem:
    return RelationSystem(
        container.agent_store,  # type: ignore[attr-defined]
        world_id=world_id,
        agent_id=agent_id,
    )


def test_perceived_relation_creation_with_defaults() -> None:
    relation = PerceivedRelation(trust=0.5, affection=0.0, target_agent_id="agent-2")

    assert relation.trust == 0.5
    assert relation.affection == 0.0
    assert relation.target_agent_id == "agent-2"
    assert relation.labels == []


@pytest.mark.asyncio
async def test_relation_system_get_or_create_creates_default_relation(container: object) -> None:
    system = _make_relation_system(container)

    relation = await system.get_or_create("agent-2")

    assert relation.world_id == "world-1"
    assert relation.from_id == "agent-1"
    assert relation.to_id == "agent-2"
    assert relation.trust_objective == 0.5
    assert relation.affection_objective == 0.0


@pytest.mark.asyncio
async def test_relation_system_get_or_create_is_idempotent(container: object) -> None:
    system = _make_relation_system(container)

    first = await system.get_or_create("agent-2")
    second = await system.get_or_create("agent-2")

    assert first.trust_objective == second.trust_objective
    assert first.affection_objective == second.affection_objective


@pytest.mark.asyncio
async def test_relation_system_perceive_returns_perceived_relation(container: object) -> None:
    system = _make_relation_system(container)
    emotion = EmotionState(primary="neutral", intensity=0.2, valence=0.0)

    perceived = await system.perceive("agent-2", emotion=emotion)

    assert isinstance(perceived, PerceivedRelation)
    assert perceived.target_agent_id == "agent-2"
    assert 0.0 <= perceived.trust <= 1.0


@pytest.mark.asyncio
async def test_relation_system_apply_interaction_increases_trust(container: object) -> None:
    system = _make_relation_system(container)

    before = await system.get_or_create("agent-2")
    trust_before = before.trust_objective
    assert before.updated_step == 0  # freshly created, never substantively changed

    await system.apply_interaction("agent-2", trust_delta=0.2, affection_delta=0.1, step=7)
    after = await system.get_or_create("agent-2")

    assert after.trust_objective > trust_before
    assert after.updated_step == 7  # trust change stamps the step it happened at


@pytest.mark.asyncio
async def test_relation_system_apply_interaction_negative_trust_penalized_triple(container: object) -> None:
    """Negative trust deltas are multiplied by 3 to model asymmetric trust loss."""
    system = _make_relation_system(container)

    before = await system.get_or_create("agent-3")
    trust_before = before.trust_objective

    await system.apply_interaction("agent-3", trust_delta=-0.1, affection_delta=0.0, step=1)
    after = await system.get_or_create("agent-3")

    # delta is -0.1 * 3 = -0.3, so trust should drop by 0.3
    assert abs((trust_before - after.trust_objective) - 0.3) < 1e-9


@pytest.mark.asyncio
async def test_relation_system_trust_clamped_at_zero(container: object) -> None:
    system = _make_relation_system(container)

    # Drive trust to zero via repeated negative interactions
    for _ in range(10):
        await system.apply_interaction("agent-4", trust_delta=-0.5, affection_delta=0.0, step=1)

    relation = await system.get_or_create("agent-4")
    assert relation.trust_objective == 0.0


@pytest.mark.asyncio
async def test_relation_system_perceive_many_returns_one_per_target(container: object) -> None:
    system = _make_relation_system(container)
    emotion = EmotionState(primary="neutral", intensity=0.2, valence=0.1)

    results = await system.perceive_many(["agent-2", "agent-3", "agent-4"], emotion=emotion)

    assert len(results) == 3
    target_ids = {r.target_agent_id for r in results}
    assert target_ids == {"agent-2", "agent-3", "agent-4"}


@pytest.mark.asyncio
async def test_relation_system_describe_formats_all_relations(container: object) -> None:
    system = _make_relation_system(container)
    emotion = EmotionState(primary="neutral", intensity=0.2, valence=0.0)
    await system.apply_interaction("agent-2", trust_delta=0.2, affection_delta=0.3, step=1)

    relations = await system.perceive_many(["agent-2"], emotion=emotion)
    text = describe_relations(relations)

    # Raw trust/affection numbers with the inline scale legend from core.prompts, not a derived
    # posture string.
    from core.prompts import RELATION_SCALE_LEGEND

    # No to_name set → renders a descriptive reference, never the raw id (the id
    # must not leak into displayed/embedded relation text).
    assert "agent-2" not in text
    assert "某人" in text
    assert RELATION_SCALE_LEGEND in text
    assert "信任度" in text and "好感度" in text


@pytest.mark.asyncio
async def test_relation_system_describe_includes_history_summary(container: object) -> None:
    system = _make_relation_system(container)
    emotion = EmotionState(primary="neutral", intensity=0.2, valence=0.0)
    # history_summary is written only by the LLM path (update_history_summary); apply_interaction
    # doesn't write it.
    await system.apply_interaction("agent-2", trust_delta=0.1, affection_delta=0.0, step=1)
    await system.update_history_summary("agent-2", "Shared a secret.", step=2)

    relations = await system.perceive_many(["agent-2"], emotion=emotion)
    text = describe_relations(relations)

    assert "Shared a secret." in text


@pytest.mark.asyncio
async def test_relation_system_interaction_count_increments(container: object) -> None:
    system = _make_relation_system(container)

    await system.apply_interaction("agent-2", trust_delta=0.1, affection_delta=0.0, step=1)
    await system.apply_interaction("agent-2", trust_delta=0.1, affection_delta=0.0, step=1)

    relation = await system.get_or_create("agent-2")
    assert relation.interaction_count == 2


def test_describe_relations_empty_returns_placeholder() -> None:
    text = describe_relations([])

    assert text  # non-empty placeholder returned


def test_describe_relations_marks_deceased() -> None:
    """Relations with the dead are kept as remembered bonds but render "（已死亡）"; relations with the
    living don't."""
    from agent.relation import PerceivedRelation
    live = PerceivedRelation(trust=0.7, affection=0.5, target_agent_id="a", target_agent_name="活人", labels=["友"])
    dead = PerceivedRelation(
        trust=0.3, affection=-0.2, target_agent_id="b", target_agent_name="故人",
        labels=["父子:子"], deceased=True,
    )
    text = describe_relations([live, dead])
    assert "故人（已死亡）" in text          # the dead are marked deceased; the bond remains
    assert "活人（已死亡）" not in text       # the living carry no mark
    assert "活人" in text


def test_render_relation_lines_marks_deceased_consistently() -> None:
    """The death mark must agree between render_relation_lines (shared by emotion appraisal /
    short-term goal generation) and describe_relations; otherwise the decision block marks someone
    dead while emotion/goal generation treats them as alive (the same perceived_relations feeds all
    three)."""
    from agent.relation import PerceivedRelation, render_relation_lines
    rels = [
        PerceivedRelation(trust=0.7, affection=0.5, target_agent_id="a", target_agent_name="活人"),
        PerceivedRelation(trust=0.3, affection=-0.2, target_agent_id="b", target_agent_name="故人", deceased=True),
    ]
    joined = "\n".join(render_relation_lines(rels))
    assert "故人（已死亡）" in joined
    assert "活人（已死亡）" not in joined and "活人" in joined


@pytest.mark.asyncio
async def test_apply_interaction_never_writes_summary(container: object) -> None:
    """Contract: apply_interaction only touches trust/affection and never writes
    history_summary/labels. The summary has only two write paths: initialization and the LLM
    (update_history_summary)."""
    system = _make_relation_system(container)

    for _ in range(4):
        await system.apply_interaction("agent-2", trust_delta=0.01, affection_delta=0.0, step=1)

    relation = await system.get_or_create("agent-2")
    assert relation.history_summary == ""   # not written: empty, not mechanically concatenated
    assert relation.labels == []

    # Only the LLM path writes it:
    await system.update_history_summary("agent-2", "近期数次往来后渐生信任。", step=1)
    relation = await system.get_or_create("agent-2")
    assert relation.history_summary == "近期数次往来后渐生信任。"


@pytest.mark.asyncio
async def test_updated_step_stamped_by_every_substantive_write(container: object) -> None:
    """updated_step tracks the relation's last substantive change; all three write paths
    (trust/affection, labels, summary) must bump it. A missed bump leaves it stuck at 0 (a dead
    field), and restore/observers think the relation never evolved. perceive's to_name backfill
    isn't a substantive change and doesn't bump."""
    system = _make_relation_system(container)

    # 0 on creation
    rel = await system.get_or_create("agent-2")
    assert rel.updated_step == 0

    # trust/affection change → bump
    await system.apply_interaction("agent-2", trust_delta=0.1, affection_delta=0.0, step=3)
    assert (await system.get_or_create("agent-2")).updated_step == 3

    # labels change → bump (RelationEvolution path)
    await system.replace_labels("agent-2", ["父子:儿子", "政敌"], step=8)
    assert (await system.get_or_create("agent-2")).updated_step == 8

    # summary change → bump, even with labels unchanged
    await system.update_history_summary("agent-2", "父子信任崩塌。", step=12)
    assert (await system.get_or_create("agent-2")).updated_step == 12

    # perceive only backfills to_name; not a substantive change, so updated_step must not move
    # either way
    emotion = EmotionState(primary="neutral", intensity=0.2, valence=0.0)
    await system.perceive("agent-2", emotion=emotion, agent_name="李渊")
    assert (await system.get_or_create("agent-2")).updated_step == 12


# ---------------------------------------------------------------------------
# Single source of truth for direction parsing and the neutral baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("positive", RelationDirection.POSITIVE),
    ("NEGATIVE", RelationDirection.NEGATIVE),   # case/whitespace normalized
    (" Neutral ", RelationDirection.NEUTRAL),
    ("garbage", RelationDirection.NEUTRAL),      # non-canonical values fall back to neutral
    ("", RelationDirection.NEUTRAL),
])
def test_parse_relation_direction(raw, expected) -> None:
    assert parse_relation_direction(raw) == expected


def test_relation_direction_value_is_canonical_string() -> None:
    """.value is the canonical str that executors store back as relation_dir (downstream
    compares/looks up by string)."""
    assert parse_relation_direction("POSITIVE").value == "positive"
    assert parse_relation_direction("x").value == "neutral"


def test_neutral_baseline_constants() -> None:
    assert NEUTRAL_TRUST == 0.5
    assert NEUTRAL_AFFECTION == 0.0


@pytest.mark.asyncio
async def test_perceive_backfills_and_renders_target_gender(container: object) -> None:
    """Gender travels with the name: backfilled into to_gender, rendered in the same person_referent
    bracket."""
    from agent.relation import render_relation_lines

    system = RelationSystem(container.agent_store, world_id="w", agent_id="agent-1")
    perceived = await system.perceive(
        "agent-2", emotion=EmotionState(), agent_name="长孙无垢", agent_gender="女",
    )
    assert perceived.target_agent_gender == "女"
    stored = await container.agent_store.load_relation("w", "agent-1", "agent-2")
    assert stored.to_gender == "女"
    assert render_relation_lines([perceived])[0].startswith("长孙无垢（女）：")


@pytest.mark.asyncio
async def test_relation_line_merges_gender_and_deceased_into_one_paren(container: object) -> None:
    """Deceased and gender share one bracket; splitting them renders "某某（女）（已死亡）"."""
    from agent.relation import PerceivedRelation, render_relation_lines

    line = render_relation_lines([PerceivedRelation(
        trust=0.5, affection=0.0, target_agent_id="a2",
        target_agent_name="长孙无垢", target_agent_gender="女", deceased=True,
    )])[0]
    assert line.startswith("长孙无垢（女，已死亡）：")


@pytest.mark.asyncio
async def test_perceive_existing_reads_without_bringing_a_relation_into_being(container: object) -> None:
    system = _make_relation_system(container)

    assert await system.perceive_existing("stranger", emotion="neutral") is None
    assert await system.load_existing("stranger") is None   # nothing was persisted

    await system.apply_interaction("friend", trust_delta=0.2, affection_delta=0.1, step=1)
    perceived = await system.perceive_existing("friend", emotion="neutral")
    assert perceived is not None and perceived.target_agent_id == "friend"
