"""Unit tests for PerceptionMemoryLayer."""

from __future__ import annotations

import pytest

from agent.memory import MemorySystem
from agent.memory_types import (
    IMPORTANCE_HIGH_CUTOFF, IMPORTANCE_MEDIUM_CUTOFF, MemoryImportance, importance_level,
)
from agent.perception_layer import PerceptionMemoryLayer, PerceptionTuning, _strength_to_importance
from agent.personality import PersonalityLayer, SoulLayer, StateLayer, EmotionState
from agent.relation import RelationSystem
from core.interfaces.llm import LLMRouter, LLMScene
from core.interfaces.message import Message
from core.interfaces.perception import PerceivedPresence, AmbientEvent, Broadcast, BroadcastType, LocationView, SpatialPerception
from providers.llm.mock import MockLLMProvider
from core.interfaces.urgency import Urgency


def _make_personality(agent_id: str = "agent-1") -> PersonalityLayer:
    soul = SoulLayer(
        name="Test",
        agent_id=agent_id,
        role="advisor",
        core_traits=["pragmatic"],
        core_values=["loyalty"],
    )
    state = StateLayer(
        step=1,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        current_location="palace",
    )
    return PersonalityLayer(soul=soul, state=state)


def _make_layer(
    container,
    *,
    agent_id: str = "agent-1",
    is_main_character: bool = True,
) -> tuple[PerceptionMemoryLayer, MemorySystem]:
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    mem = MemorySystem(
        router,
        container.embedding,
        container.vector_store,
        world_id="world-1",
        agent_id=agent_id,
    )
    rel = RelationSystem(container.agent_store, world_id="world-1", agent_id=agent_id)
    layer = PerceptionMemoryLayer(
        memory_system=mem,
        relation_system=rel,
        agent_id=agent_id,
        is_main_character=is_main_character,
    )
    return layer, mem


def _make_spatial(
    *,
    ambient_events: list[str] | None = None,
    ambient_event_strengths: list[float | None] | None = None,
    ambient_actor_ids: list[tuple[str, ...]] | None = None,
    visible_agent_ids: list[str] | None = None,
) -> SpatialPerception:
    """Test helper: takes a list of strings plus optional parallel strength/actor lists and zips
    them into AmbientEvent.

    Production code never builds SpatialPerception from list[str]; this only keeps test assertions
    short.
    """
    events = ambient_events or []
    strengths = ambient_event_strengths or [None] * len(events)
    actors = ambient_actor_ids or [()] * len(events)
    return SpatialPerception(
        location_id="palace",
        location_view=LocationView(name="Palace", description="A grand palace."),
        reachable_locations=[],
        visible_agents={aid: PerceivedPresence() for aid in visible_agent_ids or []},
        ambient_events=[
            # Every actor in this helper counts as having cognition; the split between the two body
            # tiers is derived by record_carry_observation and not tested here (see
            # test_npc_is_not_an_agent).
            AmbientEvent(content=text, strength=s, actor_ids=a, agent_actor_ids=a)
            for text, s, a in zip(events, strengths, actors)
        ],
        world_time_label="morning",
        current_step=1,
    )


def _make_broadcast(content: str, severity: str = "low", location_scope: str | None = None) -> Broadcast:
    return Broadcast(
        content=content,
        source="system",
        broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1,
        severity=severity,
        location_scope=location_scope,
    )


@pytest.mark.asyncio
async def test_message_memory_carries_sender_name(container) -> None:
    """Messages written to memory carry the sender in the body (matching "收到来自X的消息" in
    render_perceived_signals); otherwise once a message ages into recallable history it's just "[消息]
    content", losing who said it."""
    personality = _make_personality()
    layer, mem = _make_layer(container, is_main_character=True)
    msg = Message(
        id="m-1", world_id="world-1", sender_id="agent-2", sender_name="李世民",
        content="速来东宫", recipients=["agent-1"], location_scope=None,
        created_step=1, deliver_step=1, urgency=Urgency.HIGH,
    )
    before = set(mem._entries.keys())
    await layer.record(
        spatial=_make_spatial(), inbox=[msg], broadcasts=[],
        personality=personality, step=1,
    )
    written = [
        m for m in mem._entries.values()
        if m.id not in before and m.raw_content.startswith("[消息]")
    ]
    assert len(written) == 1
    assert "来自李世民的消息：速来东宫" in written[0].raw_content


@pytest.mark.asyncio
async def test_narrator_message_memory_has_no_sender_prefix(container) -> None:
    """Pseudo-senders such as narrator/system (narrative) are narration and get no "来自X" sender
    prefix."""
    personality = _make_personality()
    layer, mem = _make_layer(container, is_main_character=True)
    msg = Message(
        id="m-2", world_id="world-1", sender_id="narrator", sender_name="旁白",
        content="风云骤变", recipients=["agent-1"], location_scope=None,
        created_step=1, deliver_step=1, urgency=Urgency.HIGH,
        metadata={"narrative": True},
    )
    before = set(mem._entries.keys())
    await layer.record(
        spatial=_make_spatial(), inbox=[msg], broadcasts=[],
        personality=personality, step=1,
    )
    written = [
        m for m in mem._entries.values()
        if m.id not in before and m.raw_content.startswith("[消息]")
    ]
    assert len(written) == 1
    assert "来自" not in written[0].raw_content
    assert "风云骤变" in written[0].raw_content


@pytest.mark.asyncio
async def test_scoring_ambient_low(container) -> None:
    """Ambient gets strength 0.2: passes main threshold (0.15) but fails background (0.35)."""
    personality = _make_personality()

    main_layer, main_mem = _make_layer(container, is_main_character=True)
    bg_layer, bg_mem = _make_layer(container, agent_id="agent-2", is_main_character=False)

    spatial = _make_spatial(ambient_events=["有人路过庭院"])
    bg_personality = _make_personality(agent_id="agent-2")

    before_main = set(main_mem._entries.keys())
    await main_layer.record(
        spatial=spatial, inbox=[], broadcasts=[],
        personality=personality, step=1,
    )
    main_ambient = [
        m for m in main_mem._entries.values()
        if m.id not in before_main and m.raw_content.startswith("[环境感知]")
    ]
    assert len(main_ambient) == 1, "Main character should write ambient (0.2 >= 0.15)"

    before_bg = set(bg_mem._entries.keys())
    await bg_layer.record(
        spatial=spatial, inbox=[], broadcasts=[],
        personality=bg_personality, step=1,
    )
    bg_ambient = [
        m for m in bg_mem._entries.values()
        if m.id not in before_bg and m.raw_content.startswith("[环境感知]")
    ]
    assert len(bg_ambient) == 0, "Background agent should NOT write ambient (0.2 < 0.35)"


@pytest.mark.asyncio
async def test_narrator_message_is_not_attributed_to_a_sender(container) -> None:
    """A narrator-type message (metadata['narrative']=True) has no sender in the memory it writes.

    Locked contract: PerceptionLayer doesn't recognize the string sender_id == 'narrator'; it
    excludes the pseudo-sender at the related_agents stage via the message.metadata['narrative']
    flag.
    """
    layer, mem = _make_layer(container, is_main_character=True)

    # A message from a real agent
    real_msg = Message(
        id="real-1", world_id="world-1", sender_id="friend",
        content="老兄,午时见。", recipients=["agent-1"], location_scope=None,
        deliver_step=1, created_step=1, urgency=Urgency.NORMAL,
    )
    # An emotional event triggered by the narrator
    narrator_msg = Message(
        id="nar-1", world_id="world-1", sender_id="narrator",
        content="你心头突然涌起一阵不安。", recipients=["agent-1"], location_scope=None,
        deliver_step=1, created_step=1, urgency=Urgency.HIGH,
        metadata={"narrative": True},
    )

    await layer.record(
        spatial=_make_spatial(),
        inbox=[real_msg, narrator_msg],
        broadcasts=[],
        personality=_make_personality(),
        step=1,
    )

    def related_of(text: str) -> list[str]:
        (memory,) = [m for m in mem._entries.values() if text in m.raw_content]
        return memory.related_agents

    assert related_of("午时见") == ["friend"]
    assert related_of("心头突然涌起") == []
    # The real sender's letter counts as contact, scored by nothing; the narrator is nobody.
    relations = {
        r.to_id: r for r in await container.agent_store.load_all_relations("world-1", "agent-1")
    }
    assert set(relations) == {"friend"}
    assert relations["friend"].interaction_count == 1
    assert (relations["friend"].trust_objective, relations["friend"].affection_objective) == (0.5, 0.0)


@pytest.mark.asyncio
async def test_ambient_explicit_strength_overrides_default(container) -> None:
    """When a producer explicitly marks a "strong social signal" via ambient_event_strengths,
    PerceptionLayer uses that strength directly so background agents perceive it too (default 0.2 <
    0.35 threshold).

    Any ambient source that "bystanders must take seriously" (violence/sudden events/COVERT
    exposure) expresses it through the strength field; PerceptionLayer does not recognize types by
    string prefix.
    """
    bg_layer, bg_mem = _make_layer(container, agent_id="agent-2", is_main_character=False)
    bg_personality = _make_personality(agent_id="agent-2")

    # With the default strength (None), the background agent ignores it (0.2 < 0.35)
    weak = _make_spatial(
        ambient_events=["有人路过"],
        ambient_event_strengths=[None],
    )
    before = set(bg_mem._entries.keys())
    await bg_layer.record(
        spatial=weak, inbox=[], broadcasts=[],
        personality=bg_personality, step=1,
    )
    assert not [m for m in bg_mem._entries.values()
                if m.id not in before and m.raw_content.startswith("[环境感知]")]

    # Explicit strength=0.7 (strong social signal): the background agent perceives and writes it
    strong = _make_spatial(
        ambient_events=["有人在阁楼里翻找东西"],
        ambient_event_strengths=[0.7],
    )
    before = set(bg_mem._entries.keys())
    await bg_layer.record(
        spatial=strong, inbox=[], broadcasts=[],
        personality=bg_personality, step=2,
    )
    written = [m for m in bg_mem._entries.values()
               if m.id not in before and m.raw_content.startswith("[环境感知]")]
    assert len(written) == 1, "Explicit strength=0.7 should let background write the ambient"


@pytest.mark.asyncio
async def test_ambient_memory_records_actor_in_identity_index(container) -> None:
    """Ambient memories from observation must carry actor ids (AmbientEvent.actor_ids →
    Memory.related_agents).

    Ambient is "I watch someone doing something", the main channel for knowing others. Dropping
    actor_ids leaves every observed memory ownerless and the identity index useless, so "what I know
    about someone" falls back to semantic neighbors and reliably recalls the wrong person.
    """
    layer, memory = _make_layer(container, agent_id="agent-1", is_main_character=True)
    personality = _make_personality()

    spatial = _make_spatial(
        ambient_events=["李元吉快步穿过宫门"],
        ambient_event_strengths=[0.7],
        ambient_actor_ids=[("agent-yuanji",)],
    )
    await layer.record(spatial=spatial, inbox=[], broadcasts=[], personality=personality, step=1)

    written = [m for m in memory._entries.values() if m.raw_content.startswith("[环境感知]")]
    assert len(written) == 1
    assert written[0].related_agents == ["agent-yuanji"]


@pytest.mark.asyncio
async def test_scoring_broadcast_high(container) -> None:
    """High-severity broadcast (strength 0.8) passes both main and background thresholds."""
    personality = _make_personality()

    main_layer, main_mem = _make_layer(container, is_main_character=True)
    bg_layer, bg_mem = _make_layer(container, agent_id="agent-2", is_main_character=False)
    bg_personality = _make_personality(agent_id="agent-2")

    spatial = _make_spatial()
    bc = _make_broadcast("大事发生！", severity="high")

    before_main = set(main_mem._entries.keys())
    await main_layer.record(
        spatial=spatial, inbox=[], broadcasts=[bc],
        personality=personality, step=1,
    )
    main_bc = [
        m for m in main_mem._entries.values()
        if m.id not in before_main and m.raw_content.startswith("[世界广播]")
    ]
    assert len(main_bc) == 1

    before_bg = set(bg_mem._entries.keys())
    await bg_layer.record(
        spatial=spatial, inbox=[], broadcasts=[bc],
        personality=bg_personality, step=1,
    )
    bg_bc = [
        m for m in bg_mem._entries.values()
        if m.id not in before_bg and m.raw_content.startswith("[世界广播]")
    ]
    assert len(bg_bc) == 1, "Background should write high-severity broadcast (0.8 >= 0.35)"


@pytest.mark.asyncio
async def test_dedup_same_broadcast(container) -> None:
    """Two broadcasts with identical content are written only once (dedup within call)."""
    personality = _make_personality()
    layer, mem = _make_layer(container, is_main_character=True)
    spatial = _make_spatial()
    bc1 = _make_broadcast("重复广播内容", severity="high")
    bc2 = _make_broadcast("重复广播内容", severity="high")

    before = set(mem._entries.keys())
    await layer.record(
        spatial=spatial, inbox=[], broadcasts=[bc1, bc2],
        personality=personality, step=1,
    )
    world_writes = [
        m for m in mem._entries.values()
        if m.id not in before and m.raw_content.startswith("[世界广播]")
    ]
    assert len(world_writes) == 1, f"Dedup should collapse identical broadcasts, got {len(world_writes)}"


@pytest.mark.asyncio
async def test_cap_ambient(container) -> None:
    """Ambient entries beyond the per-step cap are truncated (main cap=2)."""
    personality = _make_personality()
    layer, mem = _make_layer(container, is_main_character=True)
    spatial = _make_spatial(ambient_events=["事件A", "事件B", "事件C", "事件D"])

    before = set(mem._entries.keys())
    await layer.record(
        spatial=spatial, inbox=[], broadcasts=[],
        personality=personality, step=1,
    )
    ambient_writes = [
        m for m in mem._entries.values()
        if m.id not in before and m.raw_content.startswith("[环境感知]")
    ]
    assert len(ambient_writes) <= 2, f"Main ambient cap is 2, got {len(ambient_writes)}"


@pytest.mark.asyncio
async def test_external_goal_not_persisted_to_memory(container) -> None:
    """External pressure is not written to memory: it isn't something that happened, only a live
    signal (emotion/need/decision). record() doesn't accept external_goals, and perceive never
    produces "[外部压力]" memories."""
    bg_layer, bg_mem = _make_layer(container, agent_id="agent-2", is_main_character=False)
    bg_personality = _make_personality(agent_id="agent-2")
    before = set(bg_mem._entries.keys())
    await bg_layer.record(
        spatial=_make_spatial(), inbox=[], broadcasts=[],
        personality=bg_personality, step=1,
    )
    goal_writes = [
        m for m in bg_mem._entries.values()
        if m.id not in before and m.raw_content.startswith("[外部压力")
    ]
    assert goal_writes == [], "外部压力绝不应进入记忆流"


def test_strength_to_importance() -> None:
    """signal_strength mapping: >=0.7 HIGH, 0.4-0.7 MEDIUM, <0.4 LOW."""
    assert _strength_to_importance(0.8) == MemoryImportance.HIGH
    assert _strength_to_importance(0.7) == MemoryImportance.HIGH
    assert _strength_to_importance(0.5) == MemoryImportance.MEDIUM
    assert _strength_to_importance(0.4) == MemoryImportance.MEDIUM
    assert _strength_to_importance(0.3) == MemoryImportance.LOW
    assert _strength_to_importance(0.15) == MemoryImportance.LOW


def test_strength_to_importance_custom_cutoffs() -> None:
    """Tuning the cutoffs reshapes the buckets."""
    assert _strength_to_importance(0.6, high_cutoff=0.5, medium_cutoff=0.2) == MemoryImportance.HIGH
    assert _strength_to_importance(0.3, high_cutoff=0.5, medium_cutoff=0.2) == MemoryImportance.MEDIUM
    assert _strength_to_importance(0.1, high_cutoff=0.5, medium_cutoff=0.2) == MemoryImportance.LOW


@pytest.mark.asyncio
async def test_tuning_override_changes_background_threshold(container) -> None:
    """A background agent drops a 0.2 ambient at default threshold (0.35) but keeps it
    when the knob lowers threshold_bg to 0.15."""
    spatial = _make_spatial(ambient_events=["有人路过庭院"])  # strength → default 0.2

    bg_layer, _ = _make_layer(container, agent_id="agent-2", is_main_character=False)
    selected_default = bg_layer._select_items(
        await bg_layer._collect_items(spatial, [], [])
    )
    assert not [i for i in selected_default if i.source == "ambient"]

    tuned = PerceptionMemoryLayer(
        memory_system=bg_layer._memory_system,
        relation_system=bg_layer._relation_system,
        agent_id="agent-2",
        is_main_character=False,
        tuning=PerceptionTuning(threshold_bg=0.15),
    )
    selected_tuned = tuned._select_items(await tuned._collect_items(spatial, [], []))
    assert [i for i in selected_tuned if i.source == "ambient"], "lowered threshold should keep ambient"


@pytest.mark.asyncio
async def test_collect_and_select_are_read_only(container) -> None:
    """The tuning capture path (collect + select, no record) must not write memory."""
    layer, mem = _make_layer(container, is_main_character=True)
    spatial = _make_spatial(ambient_events=["庭院有动静"], ambient_event_strengths=[0.6])

    before = set(mem._entries.keys())
    collected = await layer._collect_items(spatial, [], [])
    selected = layer._select_items(collected)
    assert mem._entries.keys() == before, "collect/select must not write memory"
    assert any(i.source == "ambient" for i in collected)
    assert any(i.source == "ambient" for i in selected)


def test_importance_cutoffs_single_source() -> None:
    """Perception's strength→bucket cut points must reuse memory's canonical bucket boundaries
    (single source of truth).

    Two places each writing a cut point (e.g. 0.7 vs 0.65) silently diverge."""
    tuning = PerceptionTuning()
    assert tuning.importance_high_cutoff == IMPORTANCE_HIGH_CUTOFF == 0.65
    assert tuning.importance_medium_cutoff == IMPORTANCE_MEDIUM_CUTOFF == 0.4


def test_strength_to_importance_agrees_with_importance_level_at_boundary() -> None:
    """strength→bucket and score→bucket (importance_level) agree at the HIGH boundary: 0.66 is HIGH
    for both."""
    assert _strength_to_importance(0.66) == MemoryImportance.HIGH
    assert importance_level(0.66) == MemoryImportance.HIGH
    # Below the boundary both say MEDIUM
    assert _strength_to_importance(0.5) == MemoryImportance.MEDIUM
    assert importance_level(0.5) == MemoryImportance.MEDIUM
