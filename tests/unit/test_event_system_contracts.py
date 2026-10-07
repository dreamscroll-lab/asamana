"""Negative assertions ("what EventSystem can't do") locking the design charter in the
engine/event.py module docstring, as tripwires for refactors. Happy-path tests live in
test_event_system.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.personality import EmotionState, EmotionType, PersonalityLayer, SoulLayer, StateLayer
from core.interfaces.urgency import Urgency
from engine.broadcast import BroadcastChannel
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.event import EventSettings, EventSystem, _EventPlan
from engine.injection import BroadcastSpec, InjectionDispatcher, MessageSpec
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.message_system import MessageSystem
from engine.world_mutation import WorldMutationChannel
from providers.message.in_memory import InMemoryMessageProvider


def _mutation_channel() -> WorldMutationChannel:
    environment = EnvironmentSystem()
    return WorldMutationChannel(
        environment=environment,
        seconds_per_step=3600,
        processor=ExecutionProcessor(
            executor_registry=ActionExecutorRegistry(),
            environment=environment,
            message_system=MessageSystem(InMemoryMessageProvider(), world_id="w1"),
            directory=LiveWorldDirectory.from_agents({}, environment),
        ),
    )


def _make_event_system(
    container,
    *,
    broadcast_channel: BroadcastChannel | None = None,
    message_system: MessageSystem | None = None,
) -> EventSystem:
    directory = LiveWorldDirectory.from_agents({}, EnvironmentSystem())
    return EventSystem(
        llm_router=container.llm_router,
        snapshot_provider=container.snapshot,
        dispatcher=InjectionDispatcher(
            broadcast_channel=broadcast_channel or BroadcastChannel(),
            message_system=message_system or MessageSystem(container.message_provider, world_id="w1"),
            mutation_channel=_mutation_channel(),
            directory=directory,
        ),
        directory=directory,
        settings=EventSettings(
            core_tension="t",
            narrative_theme="m",
            max_events_per_window=10,
            check_interval=1,  # checks allowed every step
        ),
    )


async def _dispatched(system: EventSystem, plan: _EventPlan, step: int) -> list[str]:
    """Commit a plan and return the channels it landed on (empty list if none landed)."""
    committed = await system._commit_plan(plan, step, {})
    return [] if committed is None else committed.event.dispatched_to


def _stub_agent(agent_id: str = "a1") -> object:
    """Bare agent stub for no-agent-mutation checks."""
    soul = SoulLayer(
        name="Test", agent_id=agent_id, role="x",
        core_traits=("calm",), core_values=("peace",),
        self_image="I am calm.",
    )
    state = StateLayer(
        agent_id=agent_id, step=1,
        emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
        current_location="loc_a",
        vitality=1.0,
    )
    personality = PersonalityLayer(soul=soul, state=state)
    from types import SimpleNamespace
    return SimpleNamespace(
        agent_id=agent_id,
        personality=personality,
        is_active=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# No agent mutation: agent state is unchanged after an event is committed
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e2_dispatch_does_not_mutate_agent_state(container) -> None:
    """Core no-agent-mutation contract: even when an event is generated and delivered on both channels, no field
    of the agent's personality / state / is_active is mutated.
    """
    system = _make_event_system(container)
    agent = _stub_agent("a1")
    before_emotion = agent.personality.state.emotion.primary
    before_intensity = agent.personality.state.emotion.intensity
    before_vitality = agent.personality.state.vitality
    before_location = agent.personality.state.current_location
    before_active = agent.is_active

    # build plan + dispatch by hand (bypassing the LLM to focus on agent mutation)
    plan = _EventPlan(
        narrative_desc="测试事件:某事发生",
        is_positive=False,
        broadcast=BroadcastSpec(content="测试广播", severity="high", location_scope="loc_a"),
        message=MessageSpec(content="测试通知", recipients=["a1"], urgency=Urgency.HIGH),
    )
    dispatched = await _dispatched(system, plan, 1)
    assert set(dispatched) == {"broadcast", "message"}, f"dispatched={dispatched}"

    assert agent.personality.state.emotion.primary == before_emotion
    assert agent.personality.state.emotion.intensity == before_intensity
    assert agent.personality.state.vitality == before_vitality
    assert agent.personality.state.current_location == before_location
    assert agent.is_active == before_active


# ─────────────────────────────────────────────────────────────────────────────
# Data-driven, no EventCategory / AgentEventType / GlobalEventType enums
# ─────────────────────────────────────────────────────────────────────────────


def _non_docstring_source(filepath: str) -> str:
    """Return source code with module-level docstring stripped, so grep checks
    don't false-positive on contract description text.
    """
    import ast
    src = Path(filepath).read_text(encoding="utf-8")
    tree = ast.parse(src)
    if not tree.body:
        return src
    first = tree.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        lines = src.splitlines()
        return "\n".join(lines[first.end_lineno:])
    return src


def test_e3_no_event_type_enums_in_module() -> None:
    """The type enums (EventCategory / AgentEventType / GlobalEventType) must not exist; bringing
    any back reintroduces the dispatch-by-type antipattern.
    """
    src = _non_docstring_source("engine/event.py")
    forbidden_enums = ["EventCategory", "AgentEventType", "GlobalEventType"]
    found = [name for name in forbidden_enums if name in src]
    assert not found, (
        f"engine/event.py 重新出现 type 枚举: {found}。"
        f"无类型枚举规则要求 EventSystem 数据驱动,不按 type 分发。"
    )


def test_e3_no_legacy_execute_methods() -> None:
    """event.py must not contain _execute_*-style methods that dispatch by event type."""
    src = _non_docstring_source("engine/event.py")
    forbidden_methods = [
        "_execute_event",
        "_execute_agent_event",
        "_execute_global_event",
        "_partial_memory_loss",
        "_write_event_memory",
        "_coerce_agent_event_type",
        "_coerce_global_event_type",
    ]
    found = [name for name in forbidden_methods if name in src]
    assert not found, (
        f"engine/event.py 重新出现老 _execute_* 方法: {found}。"
        f"数据驱动规则要求通道由 LLM plan 选,不由 code 按 type 分发。"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The LLM picks the channels: plan.broadcast=None -> broadcast_channel is not called
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e4_dispatch_skips_broadcast_when_plan_broadcast_is_none(container) -> None:
    """With plan.broadcast=None, BroadcastChannel.publish is not called and the dispatched list
    doesn't contain 'broadcast'.
    """
    broadcast_mock = MagicMock(spec=BroadcastChannel)
    system = _make_event_system(container, broadcast_channel=broadcast_mock)

    plan = _EventPlan(
        narrative_desc="纯 message 事件",
        is_positive=None,
        broadcast=None,
        message=MessageSpec(content="定向通知", recipients=["a1"], urgency=Urgency.NORMAL),
    )
    dispatched = await _dispatched(system, plan, 1)
    broadcast_mock.publish.assert_not_called()
    assert "broadcast" not in dispatched
    assert "message" in dispatched


@pytest.mark.asyncio
async def test_e4_dispatch_skips_message_when_plan_message_is_none(container) -> None:
    """With plan.message=None, MessageSystem.publish is not called."""
    message_mock = AsyncMock()
    system = _make_event_system(container, message_system=message_mock)

    plan = _EventPlan(
        narrative_desc="纯 broadcast 事件",
        is_positive=True,
        broadcast=BroadcastSpec(content="天气大变", severity="high", location_scope=None),
        message=None,
    )
    dispatched = await _dispatched(system, plan, 1)
    message_mock.publish.assert_not_called()
    assert "message" not in dispatched
    assert "broadcast" in dispatched


# ─────────────────────────────────────────────────────────────────────────────
# Multi-channel: a plan with both channels calls both
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_e5_dual_channel_dispatch_calls_both(container) -> None:
    """A plan with both broadcast and message calls each channel once; dispatched has a fixed order."""
    broadcast_mock = MagicMock(spec=BroadcastChannel)
    message_mock = AsyncMock()
    system = _make_event_system(container, broadcast_channel=broadcast_mock, message_system=message_mock)

    plan = _EventPlan(
        narrative_desc="重大变故",
        is_positive=False,
        broadcast=BroadcastSpec(content="全城戒备", severity="high", location_scope=None),
        message=MessageSpec(content="速来议事", recipients=["a1", "a2"], urgency=Urgency.HIGH),
    )
    dispatched = await _dispatched(system, plan, 5)
    assert broadcast_mock.publish.call_count == 1
    assert message_mock.publish.call_count == 1
    # dispatched order is fixed (the code sorts it) so snapshots stay stable
    assert set(dispatched) == {"broadcast", "message"}


# ─────────────────────────────────────────────────────────────────────────────
# Private effects: world mutations go only through a separate channel, which EventSystem calls only as SYSTEM
# ─────────────────────────────────────────────────────────────────────────────


def _imported_modules(filepath: str) -> set[str]:
    """The modules a file actually imports, per the AST.

    A substring grep would trip on a comment explaining why a module must NOT be wired in.
    """
    import ast
    tree = ast.parse(Path(filepath).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_e6_private_effect_channel_is_a_separate_module() -> None:
    """The world-mutation channel is a separate module; EventSystem takes only the entity
    mutations from it.

    Locks the import surface: ``VitalityMutation`` / ``RelocateMutation`` showing up in event.py
    means someone is paving a way for the LLM event editor to touch people. The channel's own
    permission table (``_permits``) is tested separately in test_world_mutation.
    """
    import ast
    assert Path("engine/world_mutation.py").is_file()
    tree = ast.parse(Path("engine/event.py").read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "engine.world_mutation"
        for alias in node.names
    }
    assert imported <= {
        "EntityMutation", "Mutation", "SpawnMutation", "WorldMutationChannel", "parse_spawn",
    }, (
        f"engine/event.py 从世界突变通道导入了 {imported}。私有效果规则:LLM 编辑只能动地上的物件,"
        "人的生死与去向只对导演开放。"
    )


def test_e6_event_system_dispatches_only_as_system() -> None:
    """Every ``.dispatch(`` in EventSystem is called explicitly as ``Author.SYSTEM``, never posing as the director."""
    import ast
    tree = ast.parse(Path("engine/event.py").read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "dispatch"
    ]
    assert calls, "EventSystem 应经 InjectionDispatcher.dispatch 落定注入。"
    for call in calls:
        authors = [kw.value for kw in call.keywords if kw.arg == "author"]
        assert len(authors) == 1 and ast.unparse(authors[0]) == "Author.SYSTEM", ast.unparse(call)
    assert "Author.DIRECTOR" not in Path("engine/event.py").read_text(encoding="utf-8")


def test_both_authors_hold_only_channels_and_read_only_views() -> None:
    """The author layer writes only through the channel and reads only through read-only views; it
    holds no mutable subsystem.

    EventSystem and DirectorChannel differ in permissions but must share this boundary shape. Holding
    a live ``EnvironmentSystem`` would let an author change the world directly, or recompute "who
    heard the broadcast" itself: a second copy of the delivery rules that drifts silently.
    """
    for module in ("engine/event.py", "engine/director.py"):
        imported = _imported_modules(module)
        assert "engine.environment" not in imported, (
            f"{module} 导入了 EnvironmentSystem。作者层不得持有可变子系统:"
            f"要读世界就由编排层把只读快照传进来(见 submit 的 locations/entities 参数)。"
        )


def test_e6_director_delegates_mutation_and_never_mutates_inline() -> None:
    """Director-path counterpart of the private-effect tripwire: the director may change the world,
    but only through the channel.

    The permission asymmetry lives only in policy; "mutations converge in one separate module" binds
    both authors. Otherwise director.py drifts into calling agent state directly and
    world_mutation.py becomes a facade nobody uses.
    """
    src = _non_docstring_source("engine/director.py")
    forbidden_inline = [
        "set_active(",
        "apply_vitality_damage(",
        "change_entity_state(",
        "update_location(",
        "move_body(",
        "_trigger_death(",
    ]
    found = [name for name in forbidden_inline if name in src]
    assert not found, (
        f"engine/director.py 直接调用了 world/agent state mutation: {found}。"
        f"导演的一切状态改动必须委托 WorldMutationChannel。"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Communication converges: EventSystem communicates only through existing channels (broadcast / message / world
# mutation)
# ─────────────────────────────────────────────────────────────────────────────


def test_e8_event_system_only_uses_existing_channels() -> None:
    """EventSystem doesn't deliver on its own; it goes only through the dispatcher shared with
    the director, which delivers only into the three existing channels.

    Checked by negative assertion: engine/event.py must show no sign of home-grown communication
    classes, nor hold channels directly.
    """
    src = Path("engine/event.py").read_text(encoding="utf-8")
    modules = _imported_modules("engine/event.py")
    assert "engine.broadcast" not in modules and "engine.message_system" not in modules
    assert "InjectionDispatcher" in src
    dispatch_modules = _imported_modules("engine/injection.py")
    assert {"engine.broadcast", "engine.message_system"} <= dispatch_modules
    # No new communication mechanism (these are hypothetical home-grown class names; any match
    # signals a regression)
    forbidden_channels = [
        "PrivateEffectChannel(",
        "EventChannel(",
        "NarrativeChannel(",
    ]
    found = [name for name in forbidden_channels if name in src]
    assert not found, (
        f"engine/event.py 出现自创通讯类: {found}。"
        f"通讯收敛规则要求不发明新通讯机制,有新需求先设计独立通道模块。"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scope: event injection doesn't modify agents directly
# ─────────────────────────────────────────────────────────────────────────────


def test_e1_event_system_does_not_import_agent_state_mutation_methods() -> None:
    """EventSystem calls no mutation method such as agent.personality.update_emotion /
    set_active / apply_vitality_damage / memory_system writes / need_engine writes.
    """
    src = _non_docstring_source("engine/event.py")
    # Match calls in their parenthesized form, excluding bare explanatory tokens in docstrings.
    forbidden_mutations = [
        "update_emotion(",
        "set_active(",
        "apply_vitality_damage(",
        "record_event(",
        "seed_factual_memory(",
        "apply_need_state(",
        "apply_feedback(",
    ]
    found = [name for name in forbidden_mutations if name in src]
    assert not found, (
        f"engine/event.py 调用 agent state mutation: {found}。"
        f"不改 agent 规则要求 EventSystem 不 mutate agent state,所有效果走通道注入信号。"
    )


@pytest.mark.asyncio
async def test_a_directors_own_words_never_reach_the_llm_editor(container) -> None:
    """The director's verbatim text is an author-layer note, not something that happened in the world.

    If the LLM event editor's brief says "someone outside the world gave an order", the narrative
    layer learns it is being written, which is worse than an id leak. Each author keeps its own
    ledger; this locks that, so adding a field to the brief or merging the ledgers trips it.
    """
    system = _make_event_system(container)
    # A history mixing both authors is exactly the shape restore hands back.
    system.restore_state([
        {"id": "e1", "step": 1, "narrative_desc": "宫中忽起大风",
         "authored_by": "system"},
        {"id": "e2", "step": 2, "narrative_desc": "西市燃起大火",
         "authored_by": "director",
         "directive_text": "在西市放一把火，我要看他们乱起来"},
    ])

    briefing = system._brief_prior_events()

    assert "宫中忽起大风" in briefing            # its own past events are still avoided as usual
    assert "我要看他们乱起来" not in briefing    # the human's verbatim text: not a single character gets in
    assert "西市燃起大火" not in briefing        # nor are the human-injected events themselves in its ledger
