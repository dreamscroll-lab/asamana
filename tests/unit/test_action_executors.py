"""Tests for the ActionExecutor framework: SimpleExecutor, SocialExecutor,
MovementExecutor, WorkExecutor, and the registry."""

from __future__ import annotations

import json

import pytest

from agent.agent import Agent
from agent.decision import ActionType, AgentAction, DecisionEngine
from core.interfaces.action import ActionTarget, Deed, Observed, Ref
from core.interfaces.llm import LLMScene
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import ActionStatus, PersonalityLayer, SoulLayer
from agent.relation import RelationSystem
from engine.directory import LiveWorldDirectory
from engine.environment import IN_TRANSIT, SALIENT_AMBIENT_STRENGTH, EnvironmentSystem
from engine.executors import build_default_registry
from core.interfaces.execution import TickResult
from engine.executors.base import ActionExecutionState
from engine.scene import SceneVisibility
from engine.executors.covert import CovertExecutor
from engine.executors.movement import MovementExecutor, arrival_view, carried_bodies
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.simple import SimpleExecutor
from engine.executors.social import SocialExecutor
from engine.executors.work import WorkExecutor
from engine.message_system import MessageSystem
from world.models import EntityPresence, WorldEntity, WorldEntityType
from core.interfaces.place import Place


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(
    container, *, world_id: str, agent_id: str, name: str, is_main: bool = False,
    core_traits: list[str] | None = None, background: str = "",
) -> Agent:
    """A test Agent. ``core_traits``/``background`` are overridable so a test can tell two
    agents' persona strings apart — with the shared defaults, an assertion that agent B's
    traits stayed out of a prompt would be satisfied by agent A's identical traits."""
    return Agent(
        world_id=world_id,
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(
                name=name,
                role="official",
                agent_id=agent_id,
                core_traits=core_traits or ["determined"],
                core_values=["loyalty"],
                background=background,
                hard_constraints=[],
            )
        ),
        decision_engine=DecisionEngine(container.llm_router),
        llm_router=container.llm_router,
        memory_system=MemorySystem(
            container.llm_router,
            container.embedding,
            container.vector_store,
            world_id=world_id,
            agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id=world_id, agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=is_main,
    )


def _axis_target(
    action_type: ActionType, *, agent_id: str | None = None, location_id: str | None = None
) -> ActionTarget:
    """Put the target on the right relation axis, using the same rules as
    ``DecisionEngine._build_action`` so the fixture doesn't drift into a second set of semantics.

    A move acts on a place and only occupies the people it brings along. A conversation partner is
    both acted on and occupied. Every other action aimed at a person just acts on them.
    """
    if action_type is ActionType.MOVE:
        return ActionTarget(
            acts_on=[Ref.place(location_id)] if location_id else [],
            claims=[Ref.agent(agent_id)] if agent_id else [],
        )
    if agent_id:
        return ActionTarget(
            acts_on=[Ref.agent(agent_id)],
            claims=[Ref.agent(agent_id)] if action_type is ActionType.TALK else [],
        )
    return ActionTarget(acts_on=[Ref.place(location_id)] if location_id else [])


def _make_action(
    *,
    agent_id: str,
    action_type: ActionType,
    description: str = "test action",
    target_agent_id: str | None = None,
    target_location: str | None = None,
    estimated_steps: int = 1,
    expected_outcome: str = "",
) -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        step=1,
        action_type=action_type,
        action_description=description,
        target=_axis_target(action_type, agent_id=target_agent_id, location_id=target_location),
        estimated_steps=estimated_steps,
        expected_outcome=expected_outcome,
    )


def _directory(agents: dict[str, Agent] | None = None) -> LiveWorldDirectory:
    """Directory for executor construction; with no agents, every name lookup falls back to a
    descriptive referent."""
    return LiveWorldDirectory.from_agents(agents or {}, EnvironmentSystem())


def test_a_conscript_gets_a_target_of_his_own() -> None:
    """Target for a joiner's record: the same action as the initiator's, seen from his side.

    TALK's acts_on points at the joiner, so copying the record as-is would make it say the action
    acts on him. From his side the object is the initiator. MOVE's destination has no self-reference
    and carries over unchanged; claims still record that his turn is taken.
    """
    from engine.executors.base import participant_target

    talk = ActionExecutionState.create(
        action_type=ActionType.TALK, initiator_id="a", participant_ids=["a", "b"],
        purpose="谈", started_step=1, estimated_steps=2, opening_outcome="",
        target=ActionTarget(acts_on=[Ref.agent("b")], claims=[Ref.agent("b")]),
    )
    assert participant_target("a", talk) is talk.target
    joined = participant_target("b", talk)
    assert joined.acts_on == [Ref.agent("a")], "从他这边看,交谈的对象是发起者"
    assert joined.claims == [Ref.agent("b")], "他这一回合确实被占着"

    # With three traveling together, each carried person's record claims only that person, not the
    # companions. The landing record built by _landing_target(landed_id, [agent_id]) does the same,
    # and all three beats must agree.
    move = ActionExecutionState.create(
        action_type=ActionType.MOVE, initiator_id="a", participant_ids=["a", "b", "c"],
        purpose="走", started_step=1, estimated_steps=2, opening_outcome="",
        target=ActionTarget(acts_on=[Ref.place("garden")],
                            claims=[Ref.agent("b"), Ref.agent("c")]),
    )
    carried = participant_target("b", move)
    assert carried.acts_on == [Ref.place("garden")], "地点不含自指,原样带过"
    assert carried.claims == [Ref.agent("b")]


# MovementExecutor's default step length; map helpers below take edge costs in these steps.
_STEP_SECONDS = 3600


def _make_environment(move_steps: int = 1) -> EnvironmentSystem:
    loc_hall = Place(
        place_id="hall", name="hall", description="",
        connections={"garden": move_steps * _STEP_SECONDS}, is_public=True, capacity=50,
    )
    loc_garden = Place(
        place_id="garden", name="garden", description="",
        connections={"hall": move_steps * _STEP_SECONDS}, is_public=True, capacity=50,
    )
    env = EnvironmentSystem()
    env.space.register_place(loc_hall)
    env.space.register_place(loc_garden)
    env.place_agent(agent_id="agent-a", location_id="hall")
    env.place_agent(agent_id="agent-b", location_id="hall")
    return env


def _make_chain_env(node_ids: list[str], edge_steps: int = 1) -> EnvironmentSystem:
    """A linear chain node_ids[0]—…—node_ids[-1] (bidirectional, each edge `edge_steps`).

    A move from the first to the last node has real intermediate waypoints, so it
    exercises the multi-hop "appears at each waypoint" path. agent-a starts at node_ids[0].
    """
    env = EnvironmentSystem()
    for i, nid in enumerate(node_ids):
        neighbors: dict[str, int] = {}
        if i > 0:
            neighbors[node_ids[i - 1]] = edge_steps * _STEP_SECONDS
        if i < len(node_ids) - 1:
            neighbors[node_ids[i + 1]] = edge_steps * _STEP_SECONDS
        env.space.register_place(Place(
            place_id=nid, name=nid, description="",
            connections=neighbors, is_public=True, capacity=50,
        ))
    env.place_agent(agent_id="agent-a", location_id=node_ids[0])
    return env


def _make_route_env(legs: list[tuple[str, int]]) -> EnvironmentSystem:
    """A linear route; each ``(node, seconds)`` gives the walk from that node to the next (the
    last node's seconds are ignored). agent-a starts at the first node."""
    env = EnvironmentSystem()
    for i, (nid, _) in enumerate(legs):
        neighbors: dict[str, int] = {}
        if i > 0:
            neighbors[legs[i - 1][0]] = legs[i - 1][1]
        if i < len(legs) - 1:
            neighbors[legs[i + 1][0]] = legs[i][1]
        env.space.register_place(Place(
            place_id=nid, name=nid, description="",
            connections=neighbors, is_public=True, capacity=50,
        ))
    env.place_agent(agent_id="agent-a", location_id=legs[0][0])
    return env


class _RaisingLLM:
    """Router stub whose every completion raises — exercises the executor's
    LLM-*failure* path (the mock provider never raises, only returns garbage)."""

    def __init__(self) -> None:
        self.call_history: list = []

    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.call_history.append((args, kwargs))
        raise RuntimeError("llm down")


def _obs_text(result) -> str:
    """Join a result's bystander views (one per place, see core.interfaces.action.Observed) into
    one string. Tests that care where a view landed read ``.observations`` directly."""
    obs = getattr(result, "observations", None) or getattr(result, "opening_observations", None) or []
    return "\n".join(o.text for o in obs)

def _joined(messages) -> str:
    """Join a call's (system, user) messages so a test can check whether the LLM saw a phrase
    without knowing which half holds it."""
    return "\n".join(getattr(m, "content", "") for m in messages)


# ---------------------------------------------------------------------------
# ActionExecutorRegistry
# ---------------------------------------------------------------------------

class TestActionExecutorRegistry:
    def test_register_and_get(self) -> None:
        registry = ActionExecutorRegistry()
        executor = SimpleExecutor(_directory())
        registry.register(ActionType.REST, executor)
        assert registry.get_executor(ActionType.REST) is executor

    def test_get_unregistered_returns_none(self) -> None:
        registry = ActionExecutorRegistry()
        assert registry.get_executor(ActionType.TALK) is None

    def test_add_and_remove_active(self) -> None:
        registry = ActionExecutorRegistry()
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.REST,
            initiator_id="agent-a",
            participant_ids=["agent-a"],
            purpose="resting",
            started_step=1,
            opening_outcome="开始",
            estimated_steps=8,
        )
        registry.add_active(state)
        assert registry.is_agent_active("agent-a")
        assert registry.get_active_for_agent("agent-a") is state

        registry.remove_active(state.execution_id)
        assert not registry.is_agent_active("agent-a")

    def test_get_active_for_agent_participant(self) -> None:
        registry = ActionExecutorRegistry()
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.TALK,
            initiator_id="agent-a",
            participant_ids=["agent-a", "agent-b"],
            purpose="conversation",
            started_step=1,
            opening_outcome="开始",
            estimated_steps=3,
        )
        registry.add_active(state)
        assert registry.get_active_for_agent("agent-b") is state

    def test_all_active(self) -> None:
        registry = ActionExecutorRegistry()
        s1 = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.REST, initiator_id="a", participant_ids=["a"],
            purpose="x", started_step=1, estimated_steps=2,
            opening_outcome="开始",
        )
        s2 = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.WORK, initiator_id="b", participant_ids=["b"],
            purpose="y", started_step=1, estimated_steps=3,
            opening_outcome="开始",
        )
        registry.add_active(s1)
        registry.add_active(s2)
        active = registry.all_active()
        assert len(active) == 2
        assert s1 in active
        assert s2 in active

    def test_talk_executor_gets_the_world_calendar(self, container) -> None:
        """The dialogue prompt includes both sides' memories, so the day boundary has to come from
        the clock. Without it, yesterday's events read as today's."""
        registry = build_default_registry(
            container.llm_router, _directory(), world_start_second_of_day=4 * 3600)
        social = registry.get_executor(ActionType.TALK)
        assert social._world_start_second_of_day == 4 * 3600  # noqa: SLF001

    def test_build_default_registry_registers_all_types(self, container) -> None:
        registry = build_default_registry(container.llm_router, _directory())
        for action_type in [
            ActionType.REST,
            ActionType.SEND_MESSAGE, ActionType.TALK,
            ActionType.MOVE, ActionType.WORK, ActionType.PHYSICAL, ActionType.COVERT,
        ]:
            assert registry.get_executor(action_type) is not None


# ---------------------------------------------------------------------------
# SimpleExecutor
# ---------------------------------------------------------------------------

class TestSimpleExecutor:
    @pytest.mark.asyncio
    async def test_rest_start_returns_state(self) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=8)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state is not None
        assert state.action_type == ActionType.REST
        assert state.estimated_steps == 8
        assert state.initiator_id == "agent-a"

    @pytest.mark.asyncio
    async def test_send_message_start_returns_immediate_result(self) -> None:
        from core.interfaces.action import ActionResult
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.SEND_MESSAGE, estimated_steps=1)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded

    @pytest.mark.asyncio
    async def test_rest_tick_produces_narrative(self) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=8)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps -= 1
        results = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert len(results) == 1
        assert isinstance(results[0], TickResult)
        assert results[0].agent_id == "agent-a"
        assert "歇" in results[0].outcome          # REST narrative invariant; the description itself may vary

    @pytest.mark.asyncio
    async def test_rest_complete_returns_result(self) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=8)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        results = await executor.complete(state, 9, agents={}, environment=env, message_system=None)
        assert len(results) == 1
        result = results[0]
        assert result.succeeded
        assert result.action.agent_id == "agent-a"
        assert result.action.action_type == ActionType.REST
        assert "恢复" in result.factual_memory        # 1p memory keeps the internal recovery detail
        assert "恢复" not in result.outcome           # the 3p bystander channel doesn't reveal internal vitality recovery

    @pytest.mark.asyncio
    async def test_execution_state_id_is_unique(self) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=8)
        s1 = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        s2 = await executor.start(action, 2, agents={}, environment=env, message_system=None)
        assert s1.execution_id != s2.execution_id

    # --- SEND_MESSAGE memory ---

    @pytest.mark.asyncio
    async def test_send_message_factual_memory_renders_recipient_name_not_id(self, container) -> None:
        """Memory records the recipient by name (narrative layer), not by agent id."""
        from core.interfaces.action import ActionResult
        agent_b = _make_agent(container, world_id="w", agent_id="bob", name="鲍勃", is_main=False)
        executor = SimpleExecutor(_directory({"bob": agent_b}))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.SEND_MESSAGE,
            description="明天见面", target_agent_id="bob",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.factual_memory is not None
        assert "鲍勃" in result.factual_memory
        assert "bob" not in result.factual_memory    # no id leak
        assert "明天见面" in result.factual_memory

    @pytest.mark.asyncio
    async def test_send_message_unknown_recipient_falls_back_to_descriptive(self) -> None:
        """A recipient the directory doesn't know falls back to "某人" (someone), never to an id."""
        from core.interfaces.action import ActionResult
        executor = SimpleExecutor(_directory())  # empty directory, so every lookup misses
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.SEND_MESSAGE,
            description="密报", target_agent_id="ghost-id",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert "某人" in result.factual_memory
        assert "ghost-id" not in result.factual_memory

    @pytest.mark.asyncio
    async def test_broadcast_memory_names_who_actually_heard_it(self, container) -> None:
        """A broadcast's memory names everyone who was present, not a vague "在场众人" (everyone
        present): later it has to answer "who heard this"."""
        from core.interfaces.action import ActionResult
        listener = _make_agent(container, world_id="w", agent_id="agent-b", name="李元吉")
        executor = SimpleExecutor(_directory({"agent-b": listener}))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.SEND_MESSAGE,
            description="敌军来犯", target_agent_id=None,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        assert "李元吉" in result.factual_memory
        assert "在场众人" not in result.factual_memory
        assert "敌军来犯" in result.factual_memory
        assert "hall" in result.factual_memory
        assert result.outcome.count("成功") == 1  # not "成功向…成功宣告" (success repeated)

    @pytest.mark.asyncio
    async def test_a_shout_into_an_empty_room_is_not_a_delivery(self) -> None:
        """Announcing to an empty room is not a successful delivery, and nothing is delivered."""
        from core.interfaces.action import ActionResult
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        env.move_body(body_id="agent-b", location_id="garden")   # he's the only one left
        sent: list = []

        class _Spy:
            async def dispatch_from_action(self, action, **kw):  # noqa: ANN001
                sent.append(action)

        action = _make_action(
            agent_id="agent-a", action_type=ActionType.SEND_MESSAGE,
            description="敌军来犯", target_agent_id=None,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=_Spy())
        results = await executor.complete(
            state, 1, agents={}, environment=env, message_system=_Spy(),
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        assert result.failure_reason == "此处无人"
        assert "没有人听见" in result.factual_memory and "敌军来犯" in result.factual_memory
        assert not sent, "没有接收者的消息不该进队列"
        # Nothing broke; the room really is empty. This must not be filtered out as a null step.
        assert result.adjudication_failed is False

    @pytest.mark.asyncio
    async def test_send_message_directed_delivers_regardless_of_location(self) -> None:
        """A directed message has location_scope=None so it reaches absent recipients; a broadcast
        is scoped to the sender's location."""
        captured: list[dict] = []

        class _CaptureMessageSystem:
            async def dispatch_from_action(self, action, *, current_step, sender_name="", location_scope=None):
                captured.append({"recipients": list(action.target.acted_on_agents), "location_scope": location_scope})

        executor = SimpleExecutor(_directory())
        env = _make_environment()  # agent-a is in hall
        # directed at bob, who is elsewhere, so scope must be None
        directed = _make_action(
            agent_id="agent-a", action_type=ActionType.SEND_MESSAGE,
            description="速归", target_agent_id="bob",
        )
        await executor.start(directed, 1, agents={}, environment=env, message_system=_CaptureMessageSystem())
        assert captured[-1]["location_scope"] is None

        broadcast = _make_action(
            agent_id="agent-a", action_type=ActionType.SEND_MESSAGE,
            description="全体集合", target_agent_id=None,
        )
        await executor.start(broadcast, 1, agents={}, environment=env, message_system=_CaptureMessageSystem())
        assert captured[-1]["location_scope"] == "hall"

    @pytest.mark.asyncio
    async def test_rest_tick_is_narrative_only(self) -> None:
        """Ticks are pure progress narrative — TickResult carries no memory/state fields."""
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=8)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps -= 1
        results = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert len(results) == 1
        assert not hasattr(results[0], "factual_memory")
        assert not hasattr(results[0], "relation_updates")
        assert results[0].outcome

    # --- complete() ---

    @pytest.mark.asyncio
    async def test_rest_complete_uses_template_for_background_agent(self, container) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=False)
        agents = {"agent-a": agent_a}
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.REST,
            initiator_id="agent-a",
            participant_ids=["agent-a"],
            purpose="recover",
            started_step=1,
            opening_outcome="开始",
            estimated_steps=2,
        )
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        assert len(results) == 1
        assert results[0].succeeded
        assert "恢复" in results[0].factual_memory   # 1p memory keeps the recovery detail
        assert "恢复" not in results[0].outcome      # the 3p channel doesn't reveal internal recovery

    @pytest.mark.asyncio
    async def test_rest_tick_narrates_resting(self) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=4)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps -= 1
        results = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert len(results) == 1
        assert "歇" in results[0].outcome

    @pytest.mark.asyncio
    async def test_rest_narratives_use_natural_duration_not_steps(self) -> None:
        """REST tick/complete/interrupt narration uses natural durations, never a step count like
        "N步"."""
        executor = SimpleExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.REST,
            initiator_id="agent-a",
            participant_ids=["agent-a"],
            purpose="recover",
            started_step=1,
            opening_outcome="开始",
            estimated_steps=4,
        )
        # complete: 4 steps becomes about 4 hours. That's in the 1p factual_memory. The 3p outcome
        # only says where he is resting, with no duration: a bystander sees him resting, not how
        # long it took.
        complete_results = await executor.complete(state, 5, agents={}, environment=env, message_system=None)
        assert "步" not in complete_results[0].outcome
        assert "步" not in complete_results[0].factual_memory
        assert "约4小时" in complete_results[0].factual_memory

        # interrupt: 2 steps of rest becomes about 2 hours
        state.remaining_steps = 2  # elapsed = 2
        interrupt_results = await executor.interrupt(
            state, 3, agents={"agent-a": None}, environment=env, thought="想起一件急事",
        )
        assert "步" not in interrupt_results[0].factual_memory
        assert "约2小时" in interrupt_results[0].factual_memory
        assert "想起一件急事" in interrupt_results[0].outcome
        assert "想起一件急事" not in interrupt_results[0].gist

        # Ticks of other multi-step actions also use natural durations.
        work_state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.REST,  # reuses the REST route; action_type picks the non-REST purpose branch
            initiator_id="agent-a", participant_ids=["agent-a"],
            purpose="批阅奏章", started_step=1, estimated_steps=3,
            opening_outcome="开始",
        )
        work_state.action_type = ActionType.WORK  # takes the generic tick branch
        work_state.remaining_steps = 2  # elapsed = 1
        tick_results = await executor.tick(work_state, 2, agents={}, environment=env, message_system=None)
        assert "步" not in tick_results[0].outcome
        assert "约1小时" in tick_results[0].outcome

    @pytest.mark.asyncio
    async def test_rest_intent_reaches_first_person_memory_only(self) -> None:
        """The rest intention and recovered vitality are internal state. They go into 1p memory
        only, never into the 3p outcome, tick or observation.

        In 1p the intention is the object of ``我打算「…」``: quoted first-person words need an explicit
        verb in front.
        """
        executor = SimpleExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        state = ActionExecutionState.create(
            target=ActionTarget(), action_type=ActionType.REST,
            initiator_id="agent-a", participant_ids=["agent-a"],
            purpose="我闭目养神，平复心绪", started_step=1,
            opening_outcome="开始", estimated_steps=3,
        )
        done = (await executor.complete(state, 4, agents={}, environment=env, message_system=None))[0]
        assert "闭目养神" not in done.outcome
        assert done.factual_memory.startswith("我打算「我闭目养神，平复心绪」。")
        assert "休息了约3小时" in done.factual_memory

        state.remaining_steps = 2
        tick = (await executor.tick(state, 2, agents={}, environment=env, message_system=None))[0]
        assert "闭目养神" not in tick.outcome
        assert all("闭目养神" not in o.text for o in tick.observations)
        assert "已歇" in tick.outcome and "打算再歇" in tick.outcome

        ir = (await executor.interrupt(state, 2, agents={"agent-a": None}, environment=env))[0]
        assert "闭目养神" not in ir.outcome
        assert ir.factual_memory.startswith("我打算「我闭目养神，平复心绪」。")

    @pytest.mark.asyncio
    async def test_rest_third_person_lines_never_splice_the_description(self) -> None:
        """``action_description`` is a complete first-person sentence (the decision
        contract asks for 15–40 characters saying what I plan to do). Spliced into a third-person
        sentence it has no verb and reads as if he said it out loud, but resting is silent. The 1p
        path can't splice it in bare either: "在东宫我决定返回太极宫…，前后歇了约8小时" is ungrammatical and claims he
        is both in the Eastern Palace and back at Taiji Palace.
        """
        executor = SimpleExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        sentence = "我决定返回太极宫，在御书房等候太子建成前来回话。"
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.REST,
            action_description=sentence, target=ActionTarget(), estimated_steps=3,
        )
        state = await executor.start(
            action, 1, agents={}, environment=env, message_system=None,
        )
        # The opening narration, which is also the bystander observation, must not contain his
        # words.
        assert "返回太极宫" not in state.opening_outcome
        assert all("返回太极宫" not in o.text for o in state.opening_observations)

        done = (await executor.complete(state, 4, agents={}, environment=env, message_system=None))[0]
        assert "返回太极宫" not in done.outcome
        # 1p: his words go inside 「」 after "我打算", without a doubled final period.
        assert done.factual_memory.startswith(
            "我打算「我决定返回太极宫，在御书房等候太子建成前来回话」。"
        )
        assert "。，" not in done.factual_memory and "」。。" not in done.factual_memory

    @pytest.mark.asyncio
    async def test_send_message_outcome_quotes_the_first_person_intent(self) -> None:
        """SEND_MESSAGE's 3p outcome works the same way: intent_text is the sender's own narration,
        so it is quoted and attributed."""
        class _Sink:
            async def dispatch_from_action(self, action, *, current_step, sender_name="", location_scope=None):
                return None

        executor = SimpleExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.SEND_MESSAGE,
            action_description="我传话给他，约他今夜在府中相见",
            content="今夜府中一见。",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(
            action, 1, agents={}, environment=env, message_system=_Sink(),
        )
        result = state.extra["completed_result"]
        assert "：「我传话给他，约他今夜在府中相见」" in result.outcome
        assert "传讯：我传话给他" not in result.outcome
        assert result.factual_memory.startswith("我成功在")   # 1p channel unchanged

    @pytest.mark.asyncio
    async def test_rest_outcome_without_description_uses_default(self) -> None:
        """A description with no substance (a bare 'rest') falls back to the default "休息"
        wording."""
        executor = SimpleExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.REST,
            initiator_id="agent-a", participant_ids=["agent-a"],
            purpose="rest", started_step=1, estimated_steps=2,
            opening_outcome="开始",
        )
        results = await executor.complete(state, 3, agents={}, environment=env, message_system=None)
        # 3p outcome uses the default wording "歇息" (no specific activity)
        assert "歇息" in results[0].outcome
        assert "rest" not in results[0].outcome
        # 1p factual_memory keeps "休息了", a natural duration and the recovery detail
        assert "休息了" in results[0].factual_memory
        assert "rest" not in results[0].factual_memory

    @pytest.mark.asyncio
    async def test_rest_interrupt_returns_partial_recovery(self) -> None:
        executor = SimpleExecutor(_directory())
        env = _make_environment()
        action = _make_action(agent_id="agent-a", action_type=ActionType.REST, estimated_steps=4)
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps = 2  # elapsed = 4 - 2 = 2
        results = await executor.interrupt(state, 3, agents={"agent-a": None}, environment=env)
        assert len(results) == 1
        result = results[0]
        assert not result.succeeded
        assert result.vitality_damage < 0  # negative = partial recovery
        assert "2" in result.factual_memory


# ---------------------------------------------------------------------------
# SocialExecutor
# ---------------------------------------------------------------------------

class TestSocialExecutor:
    @pytest.mark.asyncio
    async def test_one_step_talk_creates_state(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=1,
        )
        # 1v1 TALK always uses ActionExecutionState so complete() returns results for both sides.
        result = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(result, ActionExecutionState)
        assert result.action_type == ActionType.TALK
        assert "agent-a" in result.participant_ids
        assert "agent-b" in result.participant_ids

    @pytest.mark.asyncio
    async def test_multi_step_talk_creates_state(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert state is not None
        assert state.action_type == ActionType.TALK
        assert "agent-a" in state.participant_ids
        assert "agent-b" in state.participant_ids
        assert state.extra.get("target_id") == "agent-b"

    @pytest.mark.asyncio
    async def test_social_start_fails_if_not_colocated(self, container) -> None:
        from core.interfaces.action import ActionResult
        executor = SocialExecutor(container.llm_router, _directory())
        env = EnvironmentSystem()
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="garden")  # different location
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert not result.succeeded

    @pytest.mark.asyncio
    async def test_social_tick_returns_results_for_both(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert state is not None
        state.remaining_steps -= 1
        results = await executor.tick(state, 2, agents=agents, environment=env, message_system=None)
        agent_ids = {r.agent_id for r in results}
        assert "agent-a" in agent_ids
        assert "agent-b" in agent_ids

    @pytest.mark.asyncio
    async def test_social_complete_returns_results_for_both(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert state is not None
        results = await executor.complete(state, 4, agents=agents, environment=env, message_system=None)
        result_agent_ids = {r.action.agent_id for r in results}
        assert "agent-a" in result_agent_ids
        assert "agent-b" in result_agent_ids
        for result in results:
            assert result.succeeded

    @pytest.mark.asyncio
    async def test_talk_participant_fact_attributed_to_each_participant(self, container, monkeypatch) -> None:
        """In a joint TALK, each participant's first-person fact is that participant's own cognition
        (their personality, their point of view), so its trace belongs to them, even though
        complete() runs under the initiator's observe_stage. The target's fact call carries
        agent_id=target."""
        from core.context import get_log_context, observe_stage
        from core.interfaces.trace import Stage

        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        # Both agents are in `agents`, so _memory_summary takes the LLM path and there are calls to
        # attribute.
        agents = {
            "agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲将", is_main=True),
            "agent-b": _make_agent(container, world_id="w", agent_id="agent-b", name="乙帅", is_main=True),
        }
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)

        captured: list[tuple[str, str]] = []
        orig = container.llm_router.complete

        async def spy(scene, messages, **kw):  # noqa: ANN001, ANN002, ANN003
            prompt = _joined(messages) if messages else ""
            captured.append((get_log_context().get("agent_id", ""), prompt))
            return await orig(scene, messages, **kw)

        monkeypatch.setattr(container.llm_router, "complete", spy)

        # Run complete() under the initiator's (agent-a) observe_stage, as _finalize_execution does.
        with observe_stage(Stage.ACTION, agent_id="agent-a"):
            await executor.complete(state, 1, agents=agents, environment=env, message_system=None)

        # Each participant's memory_summary fact call has "我视角一句话" in its prompt, and its 【我是谁】
        # section names that participant.
        def fact_ctx(persona_name: str) -> "str | None":
            for ctx_aid, prompt in captured:
                head = prompt.split("【刚结束的对话】")[0]
                if "我视角一句话" in prompt and persona_name in head:
                    return ctx_aid
            return None

        assert fact_ctx("乙帅") == "agent-b"   # the target's fact belongs to the target, not the initiator
        assert fact_ctx("甲将") == "agent-a"   # the initiator's fact belongs to the initiator

    @pytest.mark.asyncio
    async def test_background_pair_dialogue_is_llm_generated(self, container) -> None:
        """background×background TALK is LLM-generated, not a fixed template."""
        from core.interfaces.llm import LLMScene
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=False)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B", is_main=False)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"dialogue": [{"speaker": 1, "line": "你终于来了。"}, '
            '{"speaker": 2, "line": "我有要紧事相告。"}]}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        # The transcript is built from the JSON dialogue, with speaker indices mapped to names: no
        # ids and no A:/B: line prefixes.
        assert "你终于来了。" in results[0].outcome
        assert "A:" not in results[0].outcome and "B:" not in results[0].outcome
        # The gist says that they talked; the transcript is outcome's appended detail.
        assert "交谈" in results[0].gist and "你终于来了。" not in results[0].gist
        # Information asymmetry: the observation (the bystander channel) names both people as
        # talking but leaves out the transcript. Bystanders know they talked, not what was said.
        assert "交谈" in _obs_text(results[0])
        assert "你终于来了。" not in _obs_text(results[0])
        assert "我有要紧事相告。" not in _obs_text(results[0])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("estimated_steps", [1, 3, 20])
    async def test_dialogue_turns_decoupled_from_duration(self, container, estimated_steps) -> None:
        """TALK duration is elapsed world time and is independent of the number of turns. Whatever
        estimated_steps is, including 1, turns is always the soft cap _MAX_DIALOGUE_TURNS and the
        LLM uses fewer when the situation calls for it. Don't cap turns with min(estimated_steps,
        MAX): a short TALK would collapse into a one-turn monologue."""
        from engine.executors.social import _MAX_DIALOGUE_TURNS
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}

        captured: dict[str, int] = {}
        original = executor._llm_full_dialogue  # noqa: SLF001

        async def _capture(*args, **kwargs):
            captured["turns"] = kwargs.get("turns", -1)
            return await original(*args, **kwargs)

        executor._llm_full_dialogue = _capture  # type: ignore[method-assign]  # noqa: SLF001

        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=estimated_steps,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        await executor.complete(state, 1 + estimated_steps, agents=agents, environment=env, message_system=None)

        assert captured["turns"] == _MAX_DIALOGUE_TURNS

    @pytest.mark.asyncio
    async def test_complete_initiator_result_has_relation_updates(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        initiator_result = next(r for r in results if r.action.agent_id == "agent-a")
        assert len(initiator_result.relation_updates) == 1
        tid, trust_d, affect_d = initiator_result.relation_updates[0]
        assert tid == "agent-b"
        assert trust_d != 0.0 or affect_d != 0.0

    @pytest.mark.asyncio
    async def test_complete_target_result_has_relation_updates(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        target_result = next(r for r in results if r.action.agent_id == "agent-b")
        assert len(target_result.relation_updates) == 1
        tid, _, _ = target_result.relation_updates[0]
        assert tid == "agent-a"

    @pytest.mark.asyncio
    async def test_early_failure_result_has_empty_relation_updates(self, container) -> None:
        from core.interfaces.action import ActionResult
        executor = SocialExecutor(container.llm_router, _directory())
        env = EnvironmentSystem()
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="garden")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b",
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert not result.succeeded
        assert result.relation_updates == []

    @pytest.mark.asyncio
    async def test_single_step_talk_does_not_publish_message(self, container) -> None:
        # TALK doesn't go through MessageSystem. Bystanders in the same place perceive it through
        # ActionResult.observation (the bystander channel, not the full outcome), which flows
        # through runtime._carry_step_observations into ambient_events. See the comments in
        # social.py.
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        env.place_agent(agent_id="bystander", location_id="hall")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        ms = MessageSystem(container.message_provider, world_id="w")
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=ms)
        assert isinstance(state, ActionExecutionState)
        state.remaining_steps -= 1
        await executor.tick(state, 2, agents=agents, environment=env, message_system=ms)
        await executor.complete(state, 2, agents=agents, environment=env, message_system=ms)
        delivery = await ms.deliver_for_agents(
            step=2, agent_ids=["agent-a", "agent-b", "bystander"], environment=env,
        )
        # the bystander's inbox stays empty because TALK doesn't use the message system
        assert delivery.inbox_for("bystander") == []
        assert await ms.peek_pending() == []

    @pytest.mark.asyncio
    async def test_multi_step_talk_complete_does_not_publish_message(self, container) -> None:
        # Same for a multi-step TALK: complete() doesn't use MessageSystem either.
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        env.place_agent(agent_id="bystander", location_id="hall")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        ms = MessageSystem(container.message_provider, world_id="w")
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=ms)
        state.remaining_steps = 0
        await executor.complete(state, 4, agents=agents, environment=env, message_system=ms)
        delivery = await ms.deliver_for_agents(
            step=4, agent_ids=["agent-a", "agent-b", "bystander"], environment=env,
        )
        assert delivery.inbox_for("bystander") == []
        assert await ms.peek_pending() == []

    @pytest.mark.asyncio
    async def test_a_conscript_is_not_handed_the_initiators_intent(self, container) -> None:
        """The recruited party's result must not carry the initiator's description and expected
        outcome. The feedback layer asks the person about those two fields directly (emotion
        self-appraisal, goal-eval), so copying them over would have him read "I plan to go meet
        myself". This applies to both completion and interruption."""
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(_RaisingLLM(), _directory(agents))  # type: ignore[arg-type]
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK, target_agent_id="agent-b",
            description="我迎向B，问他深夜来此有何要事",
            expected_outcome="探明B此行的目的", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)

        completed = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )
        interrupted = await executor.interrupt(
            state, 2, agents=agents, environment=env,
            interrupted_agent_id="agent-a", thought="我必须立刻走",
        )
        # Null steps write no memory (see _talk_result), so the only thing to check here is that
        # those two fields aren't copied to the recruited party.
        assert next(
            r for r in completed if r.action.agent_id == "agent-b"
        ).factual_memory == ""

        for results in (completed, interrupted):
            initiator = next(r for r in results if r.action.agent_id == "agent-a")
            joiner = next(r for r in results if r.action.agent_id == "agent-b")
            assert initiator.action.action_description == action.action_description
            assert initiator.expected_outcome == action.expected_outcome
            # Attributed to the initiator by name, not presented as the recruited party's own words.
            assert joiner.action.action_description != action.action_description
            assert "A" in joiner.action.action_description
            # The initiator wrote this expectation when deciding; the recruited party never did.
            # Leave it out entirely rather than rephrasing it as his.
            assert joiner.expected_outcome == ""

    @pytest.mark.asyncio
    async def test_social_interrupt_dual_perspective(self, container) -> None:
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        # This checks the interrupt fallback text, where the asymmetry shows up (only the triggering
        # side gets the reason). So it forces the narration LLM to fail: the judge mock would have
        # both sides repeat the same line and the asymmetry would be invisible.
        executor = SocialExecutor(_RaisingLLM(), _directory(agents))  # type: ignore[arg-type]
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        state.remaining_steps = 3  # elapsed = 4 - 3 = 1, progress 25% < 50%
        results = await executor.interrupt(
            # Pass environment, as production always does (interrupt_coordinator). Bystander views
            # need it to resolve where they land; without it there's nowhere to deliver and the
            # declaration is empty.
            state, 2, agents=agents, environment=env,
            interrupted_agent_id="agent-a", thought="我必须立刻走",
        )
        assert len(results) == 2
        result_a = next(r for r in results if r.action.agent_id == "agent-a")
        result_b = next(r for r in results if r.action.agent_id == "agent-b")
        # reason is the interrupter's first-person perception signal, a prompt input. It must never
        # reach narrative text. Spliced into outcome and memory, observers would read
        # "原因：我感受到的来自X的外部压力：…", and that string would sit in embedded memory and be recalled again
        # and again.
        for r in results:
            assert "外部压力" not in r.outcome
            assert "外部压力" not in r.factual_memory
        # The asymmetry still holds, in the right place: the side that broke it off (A) knows it and
        # keeps his own thought; the passive side (B) only sees the other stop.
        assert "我中途撂下" in result_a.factual_memory
        assert "我必须立刻走" in result_a.factual_memory      # thought = his own reason
        assert "对方突然中止" in result_b.factual_memory
        assert "我必须立刻走" not in result_b.factual_memory  # not B's thought, so it must not be attributed to him
        # Says who started it: A sought B out, B was the one approached.
        assert "我主动找" in result_a.factual_memory
        assert "找我" in result_b.factual_memory
        # The outcome is the full 3p record: it names both people, who interrupted, and why, quoting
        # and attributing the interrupter's thought. The "why" goes only on the interrupter's own
        # outcome. B's outcome feeds B's cognition (the goal judge reads it), so putting it there
        # would let B read A's private intent. The frame is third person; only the quote is first
        # person.
        assert "A与B的交谈" in result_a.outcome and "被A打断" in result_a.outcome
        assert "A当时的心思：我必须立刻走" in result_a.outcome   # quote clearly attributed
        assert "我必须立刻走" not in result_b.outcome
        assert result_a.outcome.startswith(result_b.outcome)    # everything else identical
        # The gist is the event sentence without the quoted thought, the same on both records.
        assert result_a.gist == result_b.gist == result_b.outcome
        assert not result_a.outcome.startswith("我")            # the frame isn't first person
        # An interruption produces no bystander line. Onlookers learn about it from whatever cut it
        # off (see ActionExecutor.interrupt).
        assert result_a.observations == [] and result_b.observations == []

    @pytest.mark.asyncio
    async def test_social_interrupt_perspective_keys_on_interrupted_participant(self, container) -> None:
        """is_triggered must follow the interrupted participant — works when the one who
        broke off is the TARGET (b), not the initiator. Mirror of the dual-perspective test
        with the roles flipped: b knows the reason, the initiator a does not."""
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        # Force the narration LLM to fail, as in the test above, so the fallback asymmetry shows.
        executor = SocialExecutor(_RaisingLLM(), _directory(agents))  # type: ignore[arg-type]
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        state.remaining_steps = 3
        # The TARGET (b) is the one who broke off (received the triggering signal).
        results = await executor.interrupt(
            state, 2, agents=agents, interrupted_agent_id="agent-b", thought="宫里等不得",
        )
        result_a = next(r for r in results if r.action.agent_id == "agent-a")
        result_b = next(r for r in results if r.action.agent_id == "agent-b")
        # With the roles swapped the asymmetry still holds, and the raw signal still stays out of
        # every narrative channel.
        assert "我中途撂下" in result_b.factual_memory     # b broke off
        assert "宫里等不得" in result_b.factual_memory     # …and it was HIS reason
        assert "对方突然中止" in result_a.factual_memory   # a only saw the partner leave
        assert "宫里等不得" not in result_a.factual_memory
        for r in results:
            assert "急召入宫" not in r.factual_memory
            assert "急召入宫" not in r.outcome
        assert "被B打断" in result_a.outcome               # the 3p record names the interrupter
        assert "B当时的心思：宫里等不得" in result_b.outcome  # the reason appears only on the interrupter's own record
        assert "宫里等不得" not in result_a.outcome

    @pytest.mark.asyncio
    async def test_social_interrupt_no_delta_below_half_progress(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        state.remaining_steps = 3  # elapsed = 1, progress_ratio = 0.25 < 0.5
        results = await executor.interrupt(state, 2, agents=agents)
        for result in results:
            assert result.relation_updates == []

    @pytest.mark.asyncio
    async def test_social_interrupt_never_gives_relation_delta(self, container) -> None:
        """An interrupted talk has an undetermined effect on the relation, so it produces no delta.
        Relation changes come only from complete."""
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        state.remaining_steps = 1  # elapsed = 3; even high progress yields no delta
        results = await executor.interrupt(state, 3, agents=agents, environment=env)
        for result in results:
            assert result.relation_updates == []

    @pytest.mark.asyncio
    async def test_talk_dialogue_prompt_functional_no_id_no_step(self, container) -> None:
        """Dialogue prompt: functional sections, speakers by index, no agent ids, no steps."""
        from core.interfaces.llm import LLMScene
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"dialogue": [{"speaker": 1, "line": "近来可好。"}, {"speaker": 2, "line": "尚可。"}]}'
        )
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        dialogue_prompt = _joined(provider.call_history[0])
        assert "【任务】" in dialogue_prompt and "中立的叙事者" in dialogue_prompt
        assert "speaker 用 1" in dialogue_prompt          # IndexedRef index contract
        assert "agent-a" not in dialogue_prompt and "agent-b" not in dialogue_prompt
        assert "步" not in dialogue_prompt
        # transcript speakers are mapped to names
        assert "甲：近来可好。" in results[0].outcome

    @pytest.mark.asyncio
    async def test_talk_call_records_each_lane_exactly_as_the_prompt_shows_it(self, container) -> None:
        """The leak audit reads the two trace columns instead of parsing the prompt, so the
        annotation must be exactly the two texts the model saw."""
        from core.context import get_call_annotations
        from core.interfaces.llm import LLMScene
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = '{"dialogue": [{"speaker": 1, "line": "近来可好。"}]}'
        seen: list[dict] = []
        real_complete = provider.complete

        async def _capture(*args, **kwargs):
            seen.append(get_call_annotations())
            return await real_complete(*args, **kwargs)

        provider.complete = _capture
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 2, agents=agents, environment=env, message_system=None)

        lanes = seen[0]["dialogue_lanes"]
        user = provider.call_history[0][1].content
        assert (lanes["a"]["name"], lanes["b"]["name"]) == ("甲", "乙")
        assert f"【A：甲】\n{lanes['a']['text']}\n\n【B：乙】\n{lanes['b']['text']}\n" in user

    @pytest.mark.asyncio
    async def test_observation_instruction_offers_the_bland_option_without_positive_examples(
        self, container,
    ) -> None:
        """The observation instruction must not offer a fixed list of four outcomes
        (投契／话不投机／从容作别／不欢而散) without a "nothing to tell" option. Positive examples collapse diversity
        (§2): nearly every talk observation ends up with an emotional or posture cue, and each one
        gives bystanders a hook for suspecting a secret.

        This test keeps that list out and keeps the instruction that most talks are unremarkable and
        the field can be left empty. It also protects the three existing prohibitions from being
        dropped during edits.
        """
        from core.interfaces.llm import LLMScene
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"dialogue": [{"speaker": 1, "line": "近来可好。"}], "observation": ""}'
        )
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        prompt = _joined(provider.call_history[0])

        # Keep the positive list out, but only check the observation sentence. "话不投机／不欢而散" also
        # appears legitimately in the 【约束】 section above, where it tells the narrator the dialogue
        # itself may go badly; that use must not trip this check.
        assert "据此写一句——谈得投契" not in prompt
        assert "从容作别" not in prompt, "observation 指令又给出了正例清单（§2 正例坍缩多样性）"
        # the "nothing to tell" option must be present
        assert "多数交谈从外面看就是平平无奇的" in prompt
        assert "留空" in prompt
        # the three existing prohibitions stay verbatim
        assert "主语不必你写" in prompt
        assert "绝不可写出任何谈话内容、话题或双方的意图" in prompt
        assert "不要写他们心里怎么想" in prompt

    @pytest.mark.asyncio
    async def test_empty_observation_falls_back_to_the_neutral_line(self, container) -> None:
        """When the LLM returns nothing, bystanders get the neutral fallback "甲与乙在交谈。", not an empty
        string.

        Allowing an empty observation makes this fallback fire much more often. Without it, plain
        conversations would disappear from bystanders' perception altogether, which loses
        information.
        """
        from core.interfaces.llm import LLMScene
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"dialogue": [{"speaker": 1, "line": "近来可好。"}], "observation": ""}'
        )
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(
            state, 3, agents=agents, environment=env, message_system=None,
        )
        seen = [o.text for r in results for o in r.observations]
        assert seen, "空 observation 也必须给旁观者留下一条「他们在交谈」"
        assert any("甲与乙在交谈。" in t for t in seen), seen
        # bystanders still can't read the dialogue; the fallback doesn't break the membrane
        assert all("近来可好" not in t for t in seen)

    @pytest.mark.asyncio
    async def test_a_repairable_dialogue_failure_is_asked_again_not_written_off(
        self, container
    ) -> None:
        """A missing comma or a line with no identifiable speaker is recoverable, so ask again
        instead of wasting a beat.

        complete_with_retry only retries transport failures. An HTTP 200 with broken JSON gets
        through, and a single missing comma would void the whole conversation and leave both sides
        idle for a step. Semantic retries belong to the caller, as in DecisionEngine._llm_select.
        """
        from core.interfaces.llm import LLMResponse

        good = ('{"dialogue": [{"speaker": 1, "line": "近来可好？"}, '
                '{"speaker": 2, "line": "尚可。"}], "observation": ""}')

        class _FlakyOnce:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, scene, messages, **kwargs):
                self.calls += 1
                # The first reply is missing a comma (json_mode doesn't prevent this); the second is
                # fine.
                bad = '{"dialogue": [{"speaker": 1 "line": "…"}]}'
                return LLMResponse(content=bad if self.calls == 1 else good,
                                   input_tokens=0, output_tokens=0, model="test")

            def get(self, scene):  # noqa: D102 — used outside _joined
                return self

        llm = _FlakyOnce()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(llm, _directory(agents))  # type: ignore[arg-type]
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(
            state, 3, agents=agents, environment=env, message_system=None,
        )

        assert llm.calls >= 2, "第一次残缺之后必须再问一次"
        # Recovered: this beat has dialogue as usual and isn't a null step.
        assert all(r.dialogue for r in results)
        assert all(r.adjudication_failed is False for r in results)


    @pytest.mark.asyncio
    async def test_one_unattributable_line_voids_the_whole_transcript(self, container) -> None:
        """If any line has no identifiable speaker, the whole transcript is discarded and the
        memory layer isn't called.

        Models do drop the speaker partway through. Keeping the good lines would hand the memory
        layer a fragment as if it were a finished talk, and it would write a fabricated summary,
        verdict and relation direction for each side. An empty transcript is the same: no
        transcript means no conversation, so this is a null step.
        """
        from core.interfaces.llm import LLMScene
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"dialogue": [{"speaker": 1, "line": "你还要我等到什么时候？"}, '
            '{"speaker": 2, "line": "臣看见了。"}, '
            '{"line": "那你告诉我，什么是真相？"}, '
            '{"line": "臣不敢断言。"}], "observation": ""}'
        )
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(
            state, 3, agents=agents, environment=env, message_system=None,
        )

        # A recoverable failure gets one more attempt, but both calls must be dialogue calls:
        # neither a fragment nor an empty transcript reaches the memory layer. The summary prompt is
        # easy to spot because it is first person and contains 【我是谁】.
        prompts = [_joined(c) for c in provider.call_history]
        assert len(prompts) == 2, "整段作废也该重投一次"
        assert all("【我是谁】" not in p for p in prompts), "没有转录就没有可总结的东西"
        for r in results:
            assert r.adjudication_failed is True
            assert r.relation_updates == [], "关系变化不骑在一场没发生的谈话上"

    @pytest.mark.asyncio
    async def test_talk_memory_summary_first_person_with_initiator(self, container) -> None:
        """memory_summary is written in the first person and says who started the talk.

        The same reply feeds both the dialogue call and the summary call. The summary only runs when
        the talk succeeded, so the dialogue has to parse; otherwise this would be testing a null
        step.
        """
        from core.interfaces.llm import LLMScene
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"dialogue": [{"speaker": 1, "line": "近来可好？"}, '
            '{"speaker": 2, "line": "尚可。"}], "observation": "", '
            '"fact": "我摸清了他的态度", "success": true, "relation": "neutral"}'
        )
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        prompts = [_joined(c) for c in provider.call_history]
        summaries = [p for p in prompts if "【我是谁】" in p]
        assert summaries, "memory_summary 应为第一人称(含【我是谁】)"
        assert any("是我主动找乙谈的" in p for p in summaries)   # initiator's point of view
        assert any("乙来找我谈" in p for p in summaries) or any("甲来找我谈" in p for p in summaries)
        assert all("步" not in p for p in summaries)

    @pytest.mark.asyncio
    async def test_talk_tick_uses_natural_duration(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory(), seconds_per_step=3600)
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        state.remaining_steps = 2  # elapsed = 2
        results = await executor.tick(state, 3, agents=agents, environment=env, message_system=None)
        assert all("步" not in r.outcome for r in results)
        assert all("约2小时" in r.outcome for r in results)

    @pytest.mark.asyncio
    async def test_talk_feasibility_renders_name_and_location_no_leak(self, container) -> None:
        from core.interfaces.action import ActionResult
        env = EnvironmentSystem()
        for eid, name in (("hall-loc", "大殿"), ("garden-loc", "后花园")):
            env.space.register_place(Place(
                place_id=eid, name=name, description="", connections={}, is_public=True, capacity=50,
            ))
        env.place_agent(agent_id="agent-a", location_id="hall-loc")
        env.place_agent(agent_id="agent-b", location_id="garden-loc")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        executor = SocialExecutor(container.llm_router, _directory({"agent-b": agent_b}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents={"agent-b": agent_b}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={"agent-b": agent_b}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert not result.succeeded
        assert "乙" in result.outcome and "大殿" in result.outcome
        assert "agent-b" not in result.outcome
        assert "后花园" not in result.outcome   # doesn't reveal where the other person actually is


# ---------------------------------------------------------------------------
# MovementExecutor
# ---------------------------------------------------------------------------

class TestMovementExecutor:
    @pytest.mark.asyncio
    async def test_one_step_move_returns_immediate_result(self) -> None:
        from core.interfaces.action import ActionResult
        executor = MovementExecutor(_directory())
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        assert env.get_body_location("agent-a") == "garden"

    @pytest.mark.asyncio
    async def test_multi_step_move_creates_state_and_departs(self) -> None:
        executor = MovementExecutor(_directory())
        env = _make_environment(move_steps=3)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state is not None
        assert state.action_type == ActionType.MOVE
        assert state.extra["destination"] == "garden"
        assert state.extra["origin"] == "hall"
        assert env.get_body_location("agent-a") == "__in_transit__"

    @pytest.mark.asyncio
    async def test_movement_tick_narrative(self) -> None:
        executor = MovementExecutor(_directory())
        env = _make_environment(move_steps=3)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state is not None
        state.remaining_steps -= 1
        results = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert len(results) == 1
        assert "garden" in results[0].outcome

    @pytest.mark.asyncio
    async def test_movement_complete_places_agent_at_destination(self) -> None:
        executor = MovementExecutor(_directory())
        env = _make_environment(move_steps=3)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state is not None
        results = await executor.complete(state, 4, agents={}, environment=env, message_system=None)
        assert len(results) == 1
        assert results[0].succeeded
        assert env.get_body_location("agent-a") == "garden"

    @pytest.mark.asyncio
    async def test_move_without_destination_returns_failed_result(self) -> None:
        from core.interfaces.action import ActionResult
        executor = MovementExecutor(_directory())
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            estimated_steps=3,  # no target_location
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert not result.succeeded

    @pytest.mark.asyncio
    async def test_movement_interrupt_returns_to_origin_below_half(self) -> None:
        executor = MovementExecutor(_directory())
        env = _make_environment(move_steps=4)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps = 3  # elapsed = 4 - 3 = 1, progress 25% < 50%
        results = await executor.interrupt(state, 2, agents={"agent-a": None}, environment=env)
        assert len(results) == 1
        assert not results[0].succeeded
        assert env.get_body_location("agent-a") == "hall"

    @pytest.mark.asyncio
    async def test_movement_interrupt_arrives_at_dest_above_half(self) -> None:
        executor = MovementExecutor(_directory())
        env = _make_environment(move_steps=4)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps = 1  # elapsed = 4 - 1 = 3, progress 75% >= 50%
        results = await executor.interrupt(state, 4, agents={"agent-a": None}, environment=env)
        assert len(results) == 1
        assert results[0].succeeded
        assert env.get_body_location("agent-a") == "garden"

    @pytest.mark.asyncio
    async def test_multi_hop_move_stores_path_and_arrivals(self) -> None:
        """A move to a non-adjacent node stores the full waypoint path + per-node arrivals."""
        from engine.executors.movement import transit_view
        executor = MovementExecutor(_directory())
        env = _make_chain_env(["hall", "corridor", "garden"])  # hall—corridor—garden, each 1 step
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state.extra["path"] == ["hall", "corridor", "garden"]
        assert state.extra["arrivals"] == [0, 1, 2]
        assert state.estimated_steps == 2
        # transit_view exposes the logical path and when each waypoint is reached.
        assert transit_view(state)["path"] == ["hall", "corridor", "garden"]
        assert transit_view(state)["arrivals"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_multi_hop_surfaces_at_intermediate_waypoints(self) -> None:
        """En route the mover physically appears at each intermediate room (visible there),
        and is IN_TRANSIT only while mid-edge — never teleporting straight to the end."""
        executor = MovementExecutor(_directory())
        env = _make_chain_env(["hall", "a", "b", "garden"])  # 3 edges, total 3 steps
        env.place_agent(agent_id="watcher", location_id="a")  # bystander at first waypoint
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        # start = beat 1 (elapsed 1): first edge is 1 step → surfaces at "a".
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert env.get_body_location("agent-a") == "a"
        # the bystander at "a" now sees the passing mover — falls out of the location map.
        assert "agent-a" in env.spatial_for(agent_id="watcher").visible_agent_ids
        # tick (elapsed 2): surfaces at the next waypoint "b".
        state.remaining_steps -= 1
        await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert env.get_body_location("agent-a") == "b"
        # complete lands at the final destination.
        results = await executor.complete(state, 4, agents={}, environment=env, message_system=None)
        assert results[0].succeeded
        assert env.get_body_location("agent-a") == "garden"

    @pytest.mark.asyncio
    async def test_a_short_route_in_a_coarse_clock_takes_one_step_not_one_per_hop(self) -> None:
        """Three 10-minute hops under a 6-hour step: one step of travel and a 30-minute walk, not
        three steps; each waypoint crossed still sees him pass."""
        executor = MovementExecutor(_directory(), seconds_per_step=6 * 3600)
        env = _make_route_env([("hall", 600), ("a", 600), ("b", 600), ("garden", 0)])
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE, target_location="garden",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state.estimated_steps == 1
        assert [o.location_id for o in state.opening_observations] == ["hall"]  # only the departure
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        assert "约30分钟" in results[0].outcome
        assert "约30分钟" in results[0].factual_memory
        passed = [o.location_id for o in results[0].observations if "途经此地" in o.text]
        assert passed == ["a", "b"]
        assert env.get_body_location("agent-a") == "garden"

    @pytest.mark.asyncio
    async def test_waypoints_crossed_on_one_step_are_each_seen_and_he_stands_on_the_furthest(
        self,
    ) -> None:
        """Hops of 20, 20 and 60 minutes under a 1-hour step: a and b both fall on step 1, garden
        on step 2. He ends step 1 at b, and a still sees him pass; each waypoint is reported once."""
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_route_env([("hall", 1200), ("a", 1200), ("b", 3600), ("garden", 0)])
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE, target_location="garden",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state.estimated_steps == 2
        assert state.extra["arrivals"] == [0, 1, 1, 2]
        assert env.get_body_location("agent-a") == "b"
        passed = [o.location_id for o in state.opening_observations if "途经此地" in o.text]
        assert passed == ["a", "b"]
        state.remaining_steps -= 1
        results = await executor.complete(state, 2, agents={}, environment=env, message_system=None)
        assert not [o for o in results[0].observations if "途经此地" in o.text]
        assert "约1小时" in results[0].outcome  # 100 minutes of walking

    @pytest.mark.asyncio
    async def test_a_landed_trip_reports_the_route_it_walked(self) -> None:
        """A trip crossed within one step has no transit; its arrival carries the whole route."""
        executor = MovementExecutor(_directory(), seconds_per_step=6 * 3600)
        env = _make_route_env([("hall", 600), ("a", 600), ("b", 600), ("garden", 0)])
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE, target_location="garden",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert arrival_view(state, env) is None  # not landed yet
        await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        view = arrival_view(state, env)
        assert view is not None
        assert view["path"] == ["hall", "a", "b", "garden"]
        assert view["elapsed_steps"] == view["total_steps"] == 1

    @pytest.mark.asyncio
    async def test_no_arrival_while_under_way_or_when_he_never_got_there(self) -> None:
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_route_env([("hall", 3600), ("a", 3600), ("garden", 0)])
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE, target_location="garden",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert arrival_view(state, env) is None  # two steps: still under way
        state.remaining_steps = 0
        assert arrival_view(state, env) is None  # out of steps, but the body isn't at garden

    @pytest.mark.asyncio
    async def test_start_opening_observation_matches_tick_passthrough_at_waypoint(self) -> None:
        """start IS the first execution step: when it surfaces at a waypoint (elapsed 1), its
        bystander observation must read as a "途经此地" pass-through, identical to the tick that
        surfaces at the next waypoint, not be swallowed by the "从A动身" opening. Both come from
        the shared _passthrough_line, so the first step and a mid-journey step read the same."""
        executor = MovementExecutor(_directory())
        env = _make_chain_env(["hall", "a", "b", "garden"])  # 3 edges, 3 steps
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        # start surfaces at "a": opening_observation is the pass-through, DISTINCT from the
        # opening announcement (opening_outcome), so a bystander at "a" perceives the pass.
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert env.get_body_location("agent-a") == "a"
        assert "途经此地" in _obs_text(state)
        assert "从hall动身" in state.opening_outcome  # the opening stays the god-view announcement
        assert _obs_text(state) != state.opening_outcome
        # the tick surfacing at the next waypoint "b" produces the same-shaped pass-through beat.
        state.remaining_steps -= 1
        tick = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert "途经此地" in tick[0].outcome

    @pytest.mark.asyncio
    async def test_start_opening_observation_empty_when_mid_edge(self) -> None:
        """A start that lands mid-edge (IN_TRANSIT, no waypoint reached) has NO observable
        content — no one is in transit to see it — so opening_observation is "" (not carried),
        while the opening announcement still feeds the god-view stream."""
        executor = MovementExecutor(_directory())
        env = _make_chain_env(["hall", "garden"], edge_steps=3)  # one 3-step edge, no waypoint
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert env.get_body_location("agent-a") == IN_TRANSIT
        # Mid-edge there is no waypoint, so no "途经此地" line is produced. The departure line is still
        # there: people left in hall did watch him go, and every leg of a trip leaves a trace where
        # it passes.
        sites = {o.location_id for o in state.opening_observations}
        assert sites == {"hall"}, f"边中段只应留下起点那条,实际:{state.opening_observations}"
        assert "途经此地" not in _obs_text(state)
        assert state.opening_outcome  # the opening announcement is still present (god-view)

    @pytest.mark.asyncio
    async def test_multi_hop_interrupt_lands_at_real_waypoint(self) -> None:
        """Interrupted mid-journey, the mover stops at the real waypoint it reached —
        not snapped all the way back to origin nor forced to the destination."""
        executor = MovementExecutor(_directory())
        env = _make_chain_env(["hall", "a", "b", "garden"])
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps = 1  # elapsed = 3 - 1 = 2 → at waypoint "b"
        results = await executor.interrupt(state, 3, agents={"agent-a": None}, environment=env)
        assert not results[0].succeeded
        assert env.get_body_location("agent-a") == "b"
        assert "b" in results[0].outcome

    @pytest.mark.asyncio
    async def test_multi_step_edge_stays_in_transit_between_nodes(self) -> None:
        """A single edge costing >1 step keeps the mover IN_TRANSIT until it arrives —
        adjacent-but-far handled correctly (no phantom mid-edge waypoint)."""
        executor = MovementExecutor(_directory())
        env = _make_chain_env(["hall", "garden"], edge_steps=3)  # one 3-step edge, no intermediate
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state.extra["arrivals"] == [0, 3]
        assert env.get_body_location("agent-a") == "__in_transit__"  # elapsed 1, mid-edge
        state.remaining_steps -= 1  # elapsed 2, still mid-edge
        await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert env.get_body_location("agent-a") == "__in_transit__"

    @pytest.mark.asyncio
    async def test_movement_narratives_use_natural_duration_not_steps(self) -> None:
        """Narrative text (tick narrative, outcome, factual_memory) uses natural durations, never a
        step count like "N步"."""
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment(move_steps=4)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)

        state.remaining_steps = 3  # elapsed = 1
        tick_results = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert "步" not in tick_results[0].outcome
        assert "约1小时" in tick_results[0].outcome   # elapsed 1 step × 3600s
        assert "约3小时" in tick_results[0].outcome   # remaining 3 steps

        interrupt_results = await executor.interrupt(
            state, 2, agents={"agent-a": None}, environment=env,
        )
        assert "步" not in interrupt_results[0].factual_memory

        state2 = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        complete_results = await executor.complete(state2, 5, agents={}, environment=env, message_system=None)
        assert "步" not in complete_results[0].outcome
        assert "步" not in complete_results[0].factual_memory
        assert "约4小时" in complete_results[0].outcome

    @pytest.mark.asyncio
    async def test_move_memory_carries_the_actor_intent_marked_as_intent(self) -> None:
        """A trip's first-person memory states what he was going for, marked as an intention rather
        than a fact.

        Without it, the memory is just a change of place ("从A前往了B,路上花了…"), and repeated trips made
        for the same reason can't see each other, so an agent can shuttle back and forth toward an
        unreachable goal without noticing. The expectation hasn't happened yet, so it's marked
        "我打算". "我原本期望" is wrong: it sounds like looking back on a failure while the outcome is still
        open.
        """
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=1,
            expected_outcome="见到那位管事，把话带到。",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state.expected_outcome == "见到那位管事，把话带到。"

        result = (await executor.complete(
            state, 1, agents={}, environment=env, message_system=None,
        ))[0]
        assert result.factual_memory.startswith("我打算「见到那位管事，把话带到」。")
        assert "从hall前往了garden" in result.factual_memory
        assert "。。" not in result.factual_memory      # trailing punctuation from his own words is stripped
        assert "原本期望" not in result.factual_memory  # doesn't judge success for him
        # his own expectation is reported as-is, not replaced by a generated "抵达X"
        assert result.expected_outcome == "见到那位管事，把话带到。"

    @pytest.mark.asyncio
    async def test_move_intent_stays_out_of_the_bystander_channels(self) -> None:
        """Intent belongs only in first-person memory: others see him leave and arrive, not why he
        went."""
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment(move_steps=3)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=3,
            expected_outcome="见到那位管事，把话带到。",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        for text in (state.opening_outcome, *(o.text for o in state.opening_observations)):
            assert "管事" not in text

        tick_result = (await executor.tick(
            state, 2, agents={}, environment=env, message_system=None,
        ))[0]
        assert "管事" not in tick_result.outcome
        assert "管事" not in _obs_text(tick_result)

        env2 = _make_environment(move_steps=3)   # the agent above is still traveling; the arrival needs a clean world
        state2 = await executor.start(action, 1, agents={}, environment=env2, message_system=None)
        state2.remaining_steps = 0
        done = (await executor.complete(
            state2, 4, agents={}, environment=env2, message_system=None,
        ))[0]
        assert "管事" not in done.outcome
        assert "管事" not in _obs_text(done)
        assert "管事" in done.factual_memory

    @pytest.mark.asyncio
    async def test_interrupted_move_memory_orders_intent_fact_thought(self) -> None:
        """A trip abandoned midway reads: what he set out for, how far he got, why he stopped."""
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment(move_steps=4)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=4,
            expected_outcome="见到那位管事，把话带到。",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps = 2
        result = (await executor.interrupt(
            state, 3, agents={"agent-a": None}, environment=env,
            thought="我心里忽然没了底。", cause="一阵喧哗",
        ))[0]
        memory = result.factual_memory
        assert memory.startswith("我打算「见到那位管事，把话带到」。")
        assert memory.index("我打算") < memory.index("garden") < memory.index("没了底")
        assert "管事" not in result.outcome          # the 3p record doesn't contain the intent either
        assert "管事" not in _obs_text(result)
        assert "没了底" in result.outcome and "没了底" not in result.gist

    @pytest.mark.asyncio
    async def test_move_without_expected_outcome_keeps_the_bare_memory(self) -> None:
        """With no expectation the clause is dropped rather than inventing an intent; the memory is
        just the change of place."""
        executor = MovementExecutor(_directory(), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.MOVE,
            target_location="garden", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents={}, environment=env, message_system=None,
        ))[0]
        assert result.factual_memory == "从hall前往了garden，路上花了约1小时。"
        assert result.expected_outcome == "抵达garden"


# ---------------------------------------------------------------------------
# WorkExecutor
# ---------------------------------------------------------------------------

class TestWorkExecutor:
    @pytest.mark.asyncio
    async def test_one_step_work_returns_immediate_result(self, container) -> None:
        from core.interfaces.action import ActionResult
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        # The judge writes the outcome (see the outcome field of _generate_outcome); the
        # first-person purpose isn't quoted. When the judge's line names no one ("他把这桩事做完了"),
        # ensure_actor_named adds the actor's name in code.
        assert result.outcome == "在hall，A：他把这桩事做完了。"

    @pytest.mark.asyncio
    async def test_multi_step_work_creates_state(self, container) -> None:
        executor = WorkExecutor(container.llm_router, _directory())
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state is not None
        assert state.action_type == ActionType.WORK
        assert state.estimated_steps == 4

    @pytest.mark.asyncio
    async def test_work_tick_uses_natural_duration(self, container) -> None:
        executor = WorkExecutor(container.llm_router, _directory(), seconds_per_step=3600)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert state is not None
        # start beat is step 1 (remaining = estimated-1 = 3); one tick decrement → step 2.
        state.remaining_steps -= 1  # remaining 3→2, elapsed = 4-2 = 2
        ticks = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert "步" not in ticks[0].outcome
        assert "约2小时" in ticks[0].outcome
        # WORK is public → the tick authors observation EXPLICITLY to the same public text as
        # outcome (materialized here, never derived from outcome at the carry boundary).
        assert _obs_text(ticks[0]) == ticks[0].outcome

    @pytest.mark.asyncio
    async def test_work_complete_returns_result(self, container) -> None:
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert state is not None
        results = await executor.complete(state, 5, agents=agents, environment=env, message_system=None)
        assert len(results) == 1
        assert results[0].succeeded
        assert results[0].outcome == "在hall，A：他把这桩事做完了。"   # the judge named no one, so code adds the name
        # The product appears only in the outcome (the observation leaves it out by design), so
        # watchers have to be granted it; see COVERTABLE_ACTION_TYPES. Otherwise covert watching
        # could never find out what a job produced.
        assert results[0].happening == results[0].outcome

    # ── Products ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_work_product_becomes_a_thing_in_the_world(self, container) -> None:
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="常何")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我把换防部署令写完了。", "success": true,'
            ' "product_name": "换防部署令", "product_desc": "圈定亲信的名单", "product_carried": true,'
            ' "outcome": "常何写完了换防部署令。", "observation": "常何伏案写了半日。", "why": ""}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="拟定换防部署", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert len(result.entity_spawns) == 1
        spawn = result.entity_spawns[0]
        assert spawn.name == "换防部署令"
        assert spawn.description == "圈定亲信的名单"
        assert spawn.holder_id == "agent-a"
        # A product is privileged content (see the observation contract): at first only its author
        # can reach it.
        assert spawn.is_public is False
        # Bystanders don't get "他多了份文书" either; that would leak back what the observation just
        # stripped.
        assert spawn.perception == ""

    @pytest.mark.asyncio
    async def test_a_thing_left_on_the_ground_is_public(self, container) -> None:
        """Something left in place can't be hidden. Whether a product is private depends on where it
        ends up."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="常何")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我把栅栏筑起来了。", "success": true,'
            ' "product_name": "一道栅栏", "product_desc": "拦住了侧门", "product_carried": false,'
            ' "outcome": "常何筑起了一道栅栏。", "observation": "常何忙了半日。", "why": ""}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="筑一道栅栏", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        spawn = result.entity_spawns[0]
        assert spawn.name == "一道栅栏"
        assert spawn.holder_id is None      # not in his pocket
        assert spawn.is_public is True      # standing in the open where anyone can see it
        # Putting down a newly made item gets an ambient line, just as picking one up does.
        # Otherwise only half of the exchange is visible.
        assert "一道栅栏" in spawn.perception

    @pytest.mark.asyncio
    async def test_work_with_nothing_to_show_creates_nothing(self, container) -> None:
        """Thinking a plan through ("闭目默想推演各步变数") can succeed completely and produce nothing. The
        judge names the product explicitly; it isn't inferred from success."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="常何")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我把明日各步在心里走了一遍。", "success": true,'
            ' "product_name": "", "product_desc": "",'
            ' "outcome": "常何默坐推演了半日。", "observation": "常何闭目默坐。", "why": ""}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="默想明日变数", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert result.succeeded is True
        assert result.entity_spawns == []

    @pytest.mark.asyncio
    async def test_a_half_done_thing_lands_even_though_the_work_failed(self, container) -> None:
        """What gets left behind is independent of whether the intent succeeded. A draft missing a
        key section is a failed intent but still a real stack of paper. If products depended on
        success, it would never appear and the next beat would have to write it again from scratch."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="常何")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "终究没能写完。", "success": false,'
            ' "product_name": "换防部署令", "product_desc": "还差一半",'
            ' "outcome": "常何没能写完。", "observation": "常何伏案半日。", "why": "时候不够"}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="拟定换防部署", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert result.succeeded is False
        assert [s.name for s in result.entity_spawns] == ["换防部署令"]

    @pytest.mark.asyncio
    async def test_a_thing_in_hand_is_changed_not_twinned(self, container) -> None:
        """Adding to an item edits that item; it doesn't create a twin."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="李建成")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="draft", name="誊清的说辞", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD, presence_ref="agent-a",
            is_takeable=True, state="已选定一套", is_public=False,
        ))
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我添了几处转圜的余地。", "success": true,'
            ' "updated_index": 1, "updated_state": "已添转圜", "updated_desc": "三页，边角添了小字",'
            ' "product_name": "", "product_desc": "",'
            ' "outcome": "李建成添了几处转圜。", "observation": "李建成伏案半日。", "why": ""}'
        )
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="给说辞添几处转圜",
            target=ActionTarget(acts_on=[Ref.entity("draft")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert result.entity_spawns == []                      # no twin
        assert [c.entity_id for c in result.entity_state_changes] == ["draft"]
        assert result.entity_state_changes[0].new_state == "已添转圜"
        assert result.entity_state_changes[0].new_description == "三页，边角添了小字"
        # He's editing something in his own hands. Others can see he's busy, not what he added.
        assert result.entity_state_changes[0].perception == ""

    @pytest.mark.asyncio
    async def test_work_product_carries_its_content(self, container) -> None:
        """A letter's text belongs to the letter and lands in the world, not just in the author's
        memory."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="常何")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我写好了给秦王的信。", "success": true,'
            ' "product_name": "密信", "product_desc": "火漆封口", "product_content": "明晨卯时，玄武门换防",'
            ' "product_carried": true,'
            ' "outcome": "常何写好了一封信。", "observation": "常何伏案写了半日。", "why": ""}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK, description="给秦王写信", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]
        assert result.entity_spawns[0].content == "明晨卯时，玄武门换防"

    @pytest.mark.asyncio
    async def test_work_rewrites_the_content_in_hand_and_sees_it_first(self, container) -> None:
        """Continuing an item in hand: the judge sees its current text and returns the full revised
        content."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="李建成")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="draft", name="奏章", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD, presence_ref="agent-a", is_takeable=True,
            is_public=False, content="臣请削秦王兵权",
        ))
        llm = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        llm.fixed_response = (
            '{"fact": "我在奏章后头添了一句。", "success": true,'
            ' "updated_index": 1, "updated_state": "", "updated_desc": "",'
            ' "updated_content": "臣请削秦王兵权，并调尉迟恭出京",'
            ' "product_name": "", "product_desc": "", "product_content": "",'
            ' "outcome": "李建成改了奏章。", "observation": "李建成伏案。", "why": ""}'
        )
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="续写奏章", target=ActionTarget(acts_on=[Ref.entity("draft")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert "臣请削秦王兵权" in llm.call_history[-1][-1].content
        assert result.entity_spawns == []
        assert result.entity_state_changes[0].new_content == "臣请削秦王兵权，并调尉迟恭出京"
        env.change_entity_state(result.entity_state_changes[0])
        assert env.get_entity("draft").content == "臣请削秦王兵权，并调尉迟恭出京"

    @pytest.mark.asyncio
    async def test_one_stint_can_both_change_a_thing_and_make_another(self, container) -> None:
        """Carving a chair from a block shortens the block and creates the chair. Both happen; the
        code doesn't make the judge pick one."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="timber", name="一段木料", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD, presence_ref="agent-a",
            is_takeable=True, state="完整",
        ))
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我把木料削成了一把椅子。", "success": true,'
            ' "updated_index": 1, "updated_name": "半截木料", "updated_state": "只剩半截",'
            ' "updated_desc": "", "product_name": "一把矮凳", "product_desc": "四条腿，未上漆",'
            ' "product_carried": true,'
            ' "outcome": "甲削出了一把矮凳。", "observation": "甲忙了半日。", "why": ""}'
        )
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="用这段木料削一把凳子",
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert [c.new_state for c in result.entity_state_changes] == ["只剩半截"]
        assert [c.new_name for c in result.entity_state_changes] == ["半截木料"]
        assert [s.name for s in result.entity_spawns] == ["一把矮凳"]

    @pytest.mark.asyncio
    async def test_merely_handling_a_thing_leaves_it_unchanged(self, container) -> None:
        """Just reading an item through: an empty result means unchanged, with no need to echo the
        original text."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="scroll", name="一卷旧册", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD, presence_ref="agent-a",
            is_takeable=True, state="完好",
        ))
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我翻看了一遍,什么也没改。", "success": true,'
            ' "updated_index": 0, "updated_state": "", "updated_desc": "",'
            ' "product_name": "", "product_desc": "",'
            ' "outcome": "甲翻看了一遍。", "observation": "甲翻看了一遍。", "why": ""}'
        )
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="翻看那卷旧册",
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert result.entity_state_changes == [] and result.entity_spawns == []

    @pytest.mark.asyncio
    async def test_a_thing_in_hand_is_listed_once(self, container, monkeypatch) -> None:
        """The scene list and the roster divide the room between them: the item in my hands is
        listed only in the roster, not again among the scene's items."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="scroll", name="一卷旧册", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD, presence_ref="agent-a",
            is_takeable=True,
        ))
        env.register_entity(WorldEntity(
            entity_id="lamp", name="一盏灯", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True,
        ))
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我读了一阵。", "success": true, "updated_index": 0,'
            ' "outcome": "甲读了一阵。", "observation": "甲读了一阵。", "why": ""}'
        )
        captured: list[str] = []
        orig = container.llm_router.complete

        async def spy(scene, messages, **kw):  # noqa: ANN001, ANN002, ANN003
            captured.append(_joined(messages) if messages else "")
            return await orig(scene, messages, **kw)

        monkeypatch.setattr(container.llm_router, "complete", spy)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK, description="读那卷旧册",
            estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)

        prompt = captured[0]
        assert prompt.count("一卷旧册") == 1, "手上那件只该出现在「我手上的东西」那行里"
        assert "- 我手上的东西：#1 一卷旧册" in prompt
        assert "一盏灯" in prompt, "地上那件仍是现场证据"
        assert "现场物件：一盏灯" in prompt, "地上那件不进名单——动它得先夺过来"

    @pytest.mark.asyncio
    async def test_a_thing_on_the_floor_is_out_of_reach_of_a_solo_stint(self, container) -> None:
        """Editing an item requires holding it. Anyone can use the one on the floor, and one person
        working alone must not quietly change it.

        It isn't on the judge's menu, and if the judge makes up an index for it anyway, out-of-range
        indices are treated as no reference.
        """
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="李建成")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="onfloor", name="案上的旧稿", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True,
        ))
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我另写了一份。", "success": true,'
            ' "updated_index": 1, "updated_state": "已改定", "updated_desc": "",'
            ' "product_name": "新的说辞", "product_desc": "三页", "product_carried": true,'
            ' "outcome": "李建成另写了一份。", "observation": "李建成伏案半日。", "why": ""}'
        )
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.WORK,
            action_description="另写一份",
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert result.entity_state_changes == []
        assert [s.name for s in result.entity_spawns] == ["新的说辞"]
        assert env.get_entity("onfloor").state == "intact"

    @pytest.mark.asyncio
    async def test_work_product_never_reaches_bystanders(self, container) -> None:
        """Membrane guard: the product's name stays out of the bystander channel. Others see him at
        his desk, not what he wrote."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="常何")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我把换防部署令写完了。", "success": true,'
            ' "product_name": "换防部署令", "product_desc": "圈定亲信的名单", "product_carried": true,'
            ' "outcome": "常何写完了换防部署令。", "observation": "常何伏案写了半日。", "why": ""}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="拟定换防部署", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]

        assert "换防部署令" in result.outcome          # the full god's-eye record includes the product
        assert "换防部署令" not in _obs_text(result)   # the bystander channel strips it

    @pytest.mark.asyncio
    async def test_work_main_char_llm_failure_returns_succeeded_false(
        self, container, monkeypatch
    ) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.work import WorkExecutor as WE
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        executor = WE(container.llm_router, _directory({"agent-a": agent_a}))

        async def _fail_outcome(
            self_inner, *, agent, purpose, duration_label, now_step=0, location="",
            situation_header="", scene="", expected_outcome="", in_hand=None,
        ):
            from engine.executors.work import _WorkVerdict
            return _WorkVerdict(
                succeeded=False, fact="账目出了差错，未能理清。",
                outcome="A对着账本忙了半日，终究没理清。",
                observation="A对着账本忙了半日。", failure_reason="账目本身有出入",
            )

        monkeypatch.setattr(WE, "_generate_outcome", _fail_outcome)
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None
        )
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        # The self-appraised failure goes into the 1p factual_memory. The 3p outcome says what can
        # be seen (it didn't get done) but not his private reasons.
        assert result.factual_memory == "账目出了差错，未能理清。"
        # The judge also writes the 3p outcome, in the same call as the fact; the first-person
        # purpose isn't quoted. The judge already named him, so ensure_actor_named does nothing and
        # must not add a second name.
        assert result.outcome == "在hall，A对着账本忙了半日，终究没理清。"
        assert result.failure_reason == "账目本身有出入"   # structured reason the rendering layer uses directly
        assert "账目出了差错" not in result.outcome     # his private appraisal stays out of the 3p channel
        # On failure the outcome must say the job wasn't done, never that it was.
        assert "做完了" not in result.outcome
        assert "没理清" in result.outcome and "A：" not in result.outcome

    @pytest.mark.asyncio
    async def test_work_llm_self_judge_can_fail(self, container) -> None:
        """WORK is self-appraised by the LLM, and a failed verdict propagates. There is no rule
        template that defaults to success."""
        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        executor = WorkExecutor(container.llm_router, _directory())
        agent_main = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"success": false, "fact": "账目出了差错，未能理清。"}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_main}, environment=env, message_system=None
        )
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_main}, environment=env, message_system=None
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        # The LLM's self-appraised failure goes into the 1p factual_memory.
        assert result.factual_memory == "账目出了差错，未能理清。"
        # The judge returned success/fact but no 3p outcome, so this falls back to the binary
        # template in _outcome(). That template never quotes the first-person purpose, which would
        # give the mixed-voice "甲在「我整理账本…」上忙活".
        assert result.outcome == "在hall，某人忙活了一阵，终究没能做成手上的事。"
        assert "整理账本" not in result.outcome
        assert result.failure_reason == ""            # the judge gave no reason, so none is invented
        assert "账目出了差错" not in result.outcome     # his private appraisal stays out of the 3p channel
        # The _generate_outcome prompt includes the location (no id) and evidence from the scene.
        # WORK is done alone but not without context: what's at hand and who else is in the room
        # decide whether it succeeds (see _generate_outcome).
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "hall" in prompt
        assert "在场的其他人" in prompt    # scene evidence is included (agent-b is in hall with the actor)

    @pytest.mark.asyncio
    async def test_work_background_agent_is_llm_judged_and_can_fail(self, container) -> None:
        """A background agent's WORK is also self-appraised by the LLM, and failed verdicts
        propagate.

        Background agents must not fall back to a rule template that defaults to success. If they
        did, every non-main character's work would succeed without friction while the simulation
        still appeared to run fine.
        """
        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        executor = WorkExecutor(container.llm_router, _directory())
        agent_bg = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=False)
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"success": false, "fact": "账目太乱，没能理清。"}'
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_bg}, environment=env, message_system=None
        )
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_bg}, environment=env, message_system=None
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False           # background agents can fail too
        assert result.adjudication_failed is False  # a real event in the world, not an infrastructure failure
        assert result.factual_memory == "账目太乱，没能理清。"
        assert "没能做成" in result.outcome

    @pytest.mark.asyncio
    async def test_work_judge_sees_ordinary_fixtures_beside_an_empty_item_list(
        self, container
    ) -> None:
        """An empty item list must not read as an empty room: the judge also sees what any such
        place ordinarily has, so the work can't fail for want of it."""
        from core.interfaces.llm import LLMScene
        from engine.scene import _ORDINARY_FIXTURES_LINE
        executor = WorkExecutor(container.llm_router, _directory())
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent}, environment=_make_environment(),
            message_system=None,
        )
        await executor.complete(
            state, 1, agents={"agent-a": agent}, environment=_make_environment(),
            message_system=None,
        )
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert _ORDINARY_FIXTURES_LINE in prompt
        assert "所给现场列出的才算数" not in prompt

    # -- Evidence for the judge -------------------------------------------------
    # If the judge rules on WORK from the persona alone, it has nothing to base a success on. It
    # then always rules that nothing concrete came of it, so a WORK goal never advances and gets
    # picked again and again. The tests below check that each kind of evidence reaches the judge,
    # and that each stays within its limits: information asymmetry, the membrane, and a single
    # time/place anchor.

    @pytest.mark.asyncio
    async def test_work_judge_sees_items_at_hand(self, container) -> None:
        """Items at hand go into the judge prompt. Whether the case files are in the room decides
        whether the work gets anywhere."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="dossier", name="毒酒案卷", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True,
            description="记录了详细的证词与证据",
        ))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="翻检案卷，理出头绪", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "毒酒案卷" in prompt
        assert "记录了详细的证词与证据" in prompt   # the description comes with it, not just the name

    @pytest.mark.asyncio
    async def test_work_judge_sees_expected_outcome(self, container) -> None:
        """What he wanted, written at decision time, is the measure of success. Without it there's
        nothing to judge "done" against."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
            expected_outcome="理出这一季的亏空数目",
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "理出这一季的亏空数目" in prompt

    @pytest.mark.asyncio
    async def test_work_judge_never_sees_bystander_background(self, container) -> None:
        """Information asymmetry guard: this is a first-person self-appraisal, so people present get
        only name, gender and identity.

        Background, temperament and current intent are reserved for the functional third-party judge
        (with_background=True). Showing the actor someone else's background and plans would make him
        omniscient, so this checks with_background=False.
        """
        from agent.personality import AgentActivityStatus
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        # Give the other person a temperament and background distinct from the actor's so the test
        # can tell whose persona is in the prompt.
        agent_b = _make_agent(
            container, world_id="w", agent_id="agent-b", name="B",
            core_traits=["多疑善妒"], background="自幼在暗处替人打探消息",
        )
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()   # agent-b is in hall with the actor
        agent_b.personality.begin_action(
            step=1, description="我暗中盯着他翻检案卷", activity_status=AgentActivityStatus.IDLE,
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "B" in prompt                    # that he's present can be perceived
        assert "此刻意图" not in prompt          # what he's planning can't
        assert "我暗中盯着他翻检案卷" not in prompt
        assert "多疑善妒" not in prompt          # another person's temperament is background-only
        assert "自幼在暗处替人打探消息" not in prompt

    @pytest.mark.asyncio
    async def test_work_judge_prompt_leaks_no_ids(self, container) -> None:
        """Membrane guard: scene evidence is rendered through directory / render_entity, so no ids
        reach the LLM."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="ledger-7", name="账册", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True,
        ))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "agent-a" not in prompt and "agent-b" not in prompt
        assert "ledger-7" not in prompt
        assert "乙" in prompt and "账册" in prompt   # names are present, ids aren't

    @pytest.mark.asyncio
    async def test_work_judge_prompt_has_one_situation_anchor(self, container) -> None:
        """Single time/place anchor: the scene block uses include_header=False so it doesn't add a
        second, third-person time and place.

        The judge speaks in character in the first person, so the anchor has to be first person
        ("我此刻在…"). The header from assemble_scene_context is third person ("地点：…"); having both says
        the same thing twice in opposite voices.
        """
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "我此刻在" in prompt      # first-person anchor
        assert "地点：" not in prompt    # no second, third-person anchor

    @pytest.mark.asyncio
    async def test_work_judge_handles_empty_scene(self, container) -> None:
        """An empty scene doesn't crash. Alone in an empty room, the scene block still says
        explicitly that nobody is there, because the absence is evidence too."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.move_body(body_id="agent-b", location_id="garden")   # only the actor is left, and hall has no items
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        assert results[0].succeeded
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "在场的其他人：无" in prompt

    @pytest.mark.asyncio
    async def test_work_unfinished_remainder_rides_in_fact_not_failure_reason(self, container) -> None:
        """Keep "mostly done, one part still missing" on success too, and keep it in the fact.

        failure_reason is shared by all executors (3p, at most 20 characters, failure only), so a
        leftover there would show a failure reason for a success. The residue judge reads the
        leftover from this 1p memory (via recent_factuals) to produce the next short-term goal, so
        it must survive success.
        """
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "success": True,
            "fact": "我圈定了三个可托付的旧部，但第四人的下落仍查不到。",
            "outcome": "A伏案核对名录，圈出了几个名字。",
            "why": "",
        }, ensure_ascii=False)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="核对名录，挑出可托付的旧部", estimated_steps=1,
            expected_outcome="圈定三五个可托付的名字",
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]
        assert result.succeeded
        assert "第四人的下落仍查不到" in result.factual_memory   # the leftover stays in 1p
        assert result.failure_reason == ""                      # on success the reason field is empty
        assert "第四人" not in result.outcome                    # the product stays out of the 3p channel

    @pytest.mark.asyncio
    async def test_work_observation_is_its_own_channel_not_a_copy_of_outcome(self, container) -> None:
        """The judge writes the bystander sentence separately, not as a copy of the outcome: others
        see him at his desk, not what he wrote. Merging the channels would either leak the product
        or strip it from the full record.
        """
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "fact": "我理出了三条可用的线索。", "success": True,
            "outcome": "A理出了三条可用的线索。",      # full record, including the product
            "observation": "A伏案翻检了半日。",         # bystander: sees him busy, not the result
            "why": "",
        }, ensure_ascii=False)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="翻检卷宗", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]
        assert "三条可用的线索" in result.outcome          # the full record keeps the product
        assert "三条可用的线索" not in _obs_text(result)    # the bystander view strips it
        assert "伏案翻检" in _obs_text(result)

    @pytest.mark.asyncio
    async def test_work_missing_observation_falls_back_to_template_never_to_outcome(self, container) -> None:
        """With no observation from the judge, fall back to a template without the product, never to
        the outcome, which would hand the product to bystanders."""
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "fact": "我理出了三条可用的线索。", "success": True,
            "outcome": "A理出了三条可用的线索。", "why": "",   # no observation field at all
        }, ensure_ascii=False)
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="翻检卷宗", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        ))[0]
        assert "三条可用的线索" not in _obs_text(result)
        assert "做完了手上的事" in _obs_text(result)        # binary template, no product

    @pytest.mark.asyncio
    async def test_work_judge_states_a_fact_length_cap(self, container) -> None:
        """The fact field needs an explicit length cap: max_tokens is computed from it, and a bare
        "简洁明了" gives it nothing to go on, so an overlong reply produces unparseable JSON."""
        import re
        from core.interfaces.llm import LLMScene
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agents = {"agent-a": agent_a}
        executor = WorkExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        call = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1]
        schema_line = [ln for ln in _joined(call).splitlines() if '"fact"' in ln][-1]
        assert re.search(r"≤\d+字", schema_line), schema_line   # the fact field has an explicit cap

    @pytest.mark.asyncio
    async def test_work_interrupt_first_person_natural_duration(self, container) -> None:
        """Interrupt narration: first person, natural durations instead of steps.

        Uses the LLM-failure fallback text (_RaisingLLM) so the fallback string's shape is
        deterministic. The interruption is a real, known event and gets recorded even when the
        narration LLM fails.
        """
        executor = WorkExecutor(_RaisingLLM(), _directory(), seconds_per_step=3600)  # type: ignore[arg-type]
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=False)
        agents = {"agent-a": agent_a}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="写报告", estimated_steps=4,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        state.remaining_steps = 2  # elapsed = 2
        results = await executor.interrupt(state, 3, agents=agents, thought="我先去看看")
        assert len(results) == 1
        result = results[0]
        assert not result.succeeded
        assert "我先去看看" in result.outcome and "我先去看看" not in result.gist
        assert "步" not in result.factual_memory
        assert "约2小时" in result.factual_memory
        assert "写报告" in result.factual_memory
        # agent's first-person thought is set off by an em-dash (format_interrupt_thought),
        # not blurred into the objective record with a bare comma.
        assert "——心想：我先去看看" in result.factual_memory


# ---------------------------------------------------------------------------
# Default interrupt behavior
# ---------------------------------------------------------------------------

class TestInterruptDefault:
    @pytest.mark.asyncio
    async def test_interrupt_produces_failed_result_for_all_participants(self, container) -> None:
        executor = SocialExecutor(container.llm_router, _directory())
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert state is not None
        results = await executor.interrupt(state, 2, agents=agents)
        assert len(results) == 2
        for result in results:
            assert not result.succeeded
            # reason is a 1p signal and stays out of the narrative. The outcome says only who,
            # where, what, and that it was interrupted.
            assert "打断" in result.outcome
            assert "sudden event" not in result.outcome


# ---------------------------------------------------------------------------
# PhysicalExecutor
# ---------------------------------------------------------------------------

class TestPhysicalExecutor:
    @pytest.mark.asyncio
    async def test_entity_target_block_shows_owner_id_translated_to_name(self, container) -> None:
        """When the action targets something another person holds, the adjudication material
        includes who holds it, since taking or destroying someone else's property matters to the
        verdict. owner_id is translated to a name through the directory to keep ids out; unowned
        items get no owner line."""
        from engine.executors.physical import PhysicalExecutor
        owner = _make_agent(container, world_id="w", agent_id="owner-1", name="李世民", is_main=True)
        executor = PhysicalExecutor(container.llm_router, _directory({"owner-1": owner}))
        held = WorldEntity(
            entity_id="sword", name="天子剑", entity_type=WorldEntityType.ITEM,
            state="intact", description="一柄宝剑", is_takeable=True,
            presence=EntityPresence.HELD, presence_ref="owner-1",
        )
        block = executor._entity_target_block(held, actor_id="other")
        # In someone else's hands it's marked with the holder only, not as takeable: it has to be
        # taken from him, not picked up.
        assert "天子剑（由李世民持有）" in block
        assert "天子剑" in block and "一柄宝剑" in block
        assert "owner-1" not in block  # membrane: no ids in the prompt
        unowned = WorldEntity(
            entity_id="vase", name="花瓶", entity_type=WorldEntityType.ITEM,
            state="intact", is_takeable=True,
        )
        assert "持有" not in executor._entity_target_block(unowned, actor_id="other")
        # In the actor's own hands, all he can do is hand it over, matching the deed options that
        # offer only relinquish.
        assert "由李世民持有；可交出" in executor._entity_target_block(held, actor_id="owner-1")

    @pytest.mark.asyncio
    async def test_attack_agent_main_char_negative_llm_writes_negative_relation_updates(
        self, container, monkeypatch
    ) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        executor = PhysicalExecutor(container.llm_router, _directory())
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)

        from engine.executors.physical import _Verdict

        async def _hostile_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="某人击中了对方", fact="攻击成功了",
                relation_dir="negative",
                actor_damage=0.0, target_damage=0.3, target_relief=0.0, new_entity_state="", deed=Deed.STRIKE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _hostile_judge)
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="攻击对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        # One notch heavier than TALK (see _REL_DELTA_* in physical.py) — a deed persuades a
        # little more than a word — but still small: no single act swings a relation across the
        # band, the more so since negative trust is tripled downstream (−0.03 → actual −0.09).
        assert result.relation_updates == [("agent-b", -0.03, -0.03)]

    @pytest.mark.asyncio
    async def test_attack_agent_main_char_positive_llm_writes_positive_relation_updates(
        self, container, monkeypatch
    ) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        executor = PhysicalExecutor(container.llm_router, _directory())
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)

        from engine.executors.physical import _Verdict

        async def _helpful_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="某人搀扶了对方", fact="成功搀扶了对方",
                relation_dir="positive",
                actor_damage=0.0, target_damage=0.0, target_relief=0.0, new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _helpful_judge)
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="搀扶对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        assert result.relation_updates == [("agent-b", 0.04, 0.05)]

    @pytest.mark.parametrize(
        ("deed", "success", "damage", "actor_damage", "condition", "actor_condition", "expected"),
        [
            (Deed.STRIKE, True, 0.95, 0.0, "", "", "salient"),         # a fatal blow
            (Deed.STRIKE, True, 0.3, 0.0, "", "", "salient"),          # visibly wounded, at the bottom of the band
            (Deed.STRIKE, True, 0.1, 0.0, "", "", "light"),            # light wound
            (Deed.STRIKE, False, 0.0, 0.0, "", "", "light"),           # a stab that misses is still violence in public
            (Deed.RESTRAIN, True, 0.0, 0.0, "双手被反绑", "", "salient"),  # subduing someone
            (Deed.RESTRAIN, True, 0.0, 0.5, "", "", "salient"),        # breaking up a fight and getting hurt
            (Deed.RESTRAIN, False, 0.0, 0.0, "", "被按倒在地", "salient"),  # the attacker gets subdued instead
            (Deed.RESTRAIN, True, 0.0, 0.1, "", "", None),             # grazed while helping someone up
            (Deed.RESTRAIN, True, 0.0, 0.0, "", "", None),             # helping someone up or treating them is ordinary
        ],
    )
    @pytest.mark.asyncio
    async def test_bystander_strength_tracks_how_bad_the_violence_was(
        self, container, monkeypatch, deed, success, damage, actor_damage, condition,
        actor_condition, expected,
    ) -> None:
        """How strongly bystanders perceive it: severe wounds and restraint are HIGH, light wounds
        and missed stabs MEDIUM, helping someone up is ordinary ambient.

        Expected values come from the perception layer's bands (importance cutoffs, the
        background-agent threshold), not from numbers in the implementation.
        """
        from agent.memory_types import IMPORTANCE_HIGH_CUTOFF, IMPORTANCE_MEDIUM_CUTOFF
        from agent.perception_layer import PerceptionTuning
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="", success=success, outcome="甲对乙动了手", fact="我对乙动了手",
                relation_dir="negative", actor_damage=actor_damage, target_damage=damage, target_relief=0.0,
                new_entity_state="", deed=deed, target_condition_desc=condition,
                actor_condition_desc=actor_condition,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        executor = PhysicalExecutor(container.llm_router, _directory())
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL, action_description="对乙动手",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None)

        [observed] = results[0].observations
        if expected == "salient":
            assert observed.strength == SALIENT_AMBIENT_STRENGTH
            assert observed.strength >= IMPORTANCE_HIGH_CUTOFF
        elif expected == "light":
            assert IMPORTANCE_MEDIUM_CUTOFF <= observed.strength < IMPORTANCE_HIGH_CUTOFF
            assert observed.strength >= PerceptionTuning().threshold_bg     # background agents remember it too
        else:
            assert observed.strength is None

    @pytest.mark.asyncio
    async def test_a_person_held_down_is_not_let_go_by_the_clock(
        self, container, monkeypatch,
    ) -> None:
        """For a person, 0 still means no time limit. A person has two ways out (breaking free, or
        someone freeing him), while the tier without cognition has only one, so only that tier gets
        a fallback duration (see NPC_CONDITION_FALLBACK_SECONDS). An automatic timeout would
        suddenly free a prisoner nobody had dealt with.
        """
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="", success=True, outcome="甲把乙按住了", fact="我把乙按住了",
                relation_dir="negative", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.RESTRAIN, target_condition_desc="双手被反绑",
                target_condition_steps=0,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        executor = PhysicalExecutor(container.llm_router, _directory())
        env = _make_environment()
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="按住乙",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        )

        [effect] = results[0].target_effects
        assert effect.condition_set.description == "双手被反绑"
        assert effect.condition_set.until_step is None

    @pytest.mark.asyncio
    async def test_item_pickup_succeeds_and_declares_entity_change(self, container) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        # This case tests SEIZE (taking changes ownership), so it overrides the verdict's deed
        # explicitly; conftest defaults to operate.
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps(
            {"success": True, "fact": "我拿起了灯笼。", "relation": "neutral",
             "new_entity_state": "在手中", "deed": "seize"},
            ensure_ascii=False,
        )
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        item = WorldEntity(
            entity_id="lantern", name="lantern", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True, state="ready",
        )
        env.register_entity(item)
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="拿起灯笼",
            target=ActionTarget(acts_on=[Ref.entity("lantern", "item")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        assert result.relation_updates == []
        # Executor declares intent; runtime applies via change_entity_state
        assert len(result.entity_state_changes) == 1
        change = result.entity_state_changes[0]
        assert change.entity_id == "lantern"
        assert change.owner_id == "agent-a"

    @pytest.mark.asyncio
    async def test_seizing_from_a_mans_hands_has_a_victim(self, container, monkeypatch) -> None:
        """Taking something from a person involves that person: the victim gets a relation change
        and a first-person reaction. Otherwise robbery costs nothing and the holder's memory
        contradicts the world state.
        """
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _seize_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲夺走了虎符", fact="我夺走了虎符",
                relation_dir="negative", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.SEIZE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _seize_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-b", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我伸手去夺他手中的虎符",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)   # belonging to someone else doesn't block the action
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        assert isinstance(result, ActionResult) and result.succeeded
        assert [aid for aid, _t, _a in result.relation_updates] == ["agent-b"]
        assert [e.agent_id for e in result.target_effects] == ["agent-b"]
        assert result.entity_state_changes[0].owner_id == "agent-a"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("holder, recipient, expected_role", [
        ("agent-b", None, "持有者"),       # in someone else's hands: take it from him
        ("agent-a", "agent-b", "接收人"),  # in the actor's hands, to be given to him
        ("agent-a", None, None),           # in the actor's hands, just being put down
        (None, None, None),                # lying on the ground
    ])
    async def test_the_judge_reads_the_other_party_of_a_handover(
        self, container, monkeypatch, holder, recipient, expected_role,
    ) -> None:
        """Whoever the item leaves or goes to, the judge gets that person's vitality and both
        directions of the relation, the same counterparty evidence as acting on a person."""
        from engine.executors.physical import PhysicalExecutor

        seen: dict[str, str] = {}

        async def _capture(self, agent, **kwargs):
            seen["target_block"] = kwargs["target_block"]
            return None

        monkeypatch.setattr(PhysicalExecutor, "_judge", _capture)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD if holder else EntityPresence.AT_LOCATION,
            presence_ref=holder or env.get_body_location("agent-a"),
            is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我动一动虎符",
            target=ActionTarget(
                acts_on=[Ref.entity("tally", "item")],
                reaches=[Ref.agent(recipient)] if recipient else [],
            ),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        block = seen["target_block"]
        assert block.startswith("【物品】虎符")
        for role in ("持有者", "接收人"):
            if role == expected_role:
                assert f"【{role}】\n乙（人）" in block
                assert f"行动者对{role}的关系：" in block and f"{role}对行动者的关系：" in block
            else:
                assert f"【{role}】" not in block and f"{role}对行动者" not in block

    @pytest.mark.asyncio
    async def test_a_resisted_snatch_still_has_a_victim(self, container, monkeypatch) -> None:
        """A failed seizure still has a victim. The effect is triggered by the deed, not by success.

        Gating on success would make the most tense case, a seizure that meets resistance, the one
        nobody remembers. The world state really is unchanged (no EntityStateChange), and that
        doesn't conflict with the victim remembering it.
        """
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _failed_seize(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="对方早有防备", success=False, outcome="甲没能夺走虎符",
                fact="我没能夺走虎符", relation_dir="negative", actor_damage=0.0,
                target_damage=0.0, target_relief=0.0, new_entity_state="", deed=Deed.SEIZE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _failed_seize)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-b", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我伸手去夺他手中的虎符",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        assert isinstance(result, ActionResult) and not result.succeeded
        assert result.entity_state_changes == []              # the world didn't change
        assert [aid for aid, _t, _a in result.relation_updates] == ["agent-b"]
        assert [e.agent_id for e in result.target_effects] == ["agent-b"]

    @pytest.mark.asyncio
    async def test_target_reaction_reads_the_verdict_not_the_actors_own_words(
        self, container, monkeypatch
    ) -> None:
        """The person acted on reads the judge's 3p verdict sentence, not the actor's first-person
        action_description.

        This prompt addresses him as "你", so the actor's "我…" would be read as himself and give a
        role-reversed memory (the restrained man remembers submitting). The verdict also says what
        actually happened, which may contradict the intent.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _restrain_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲制住了乙，将其双手反剪",
                fact="我制住了乙", relation_dir="negative", actor_damage=0.0,
                target_damage=0.2, target_relief=0.0, new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _restrain_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我不再挣扎，双手微微抬起示意无害",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "甲制住了乙，将其双手反剪" in prompt
        assert "我不再挣扎" not in prompt        # the actor's first-person words never reach the prompt of the person acted on
        assert "结果：做成了" not in prompt      # the verdict sentence already includes the result; nothing extra is appended

    @pytest.mark.asyncio
    async def test_possession_target_reaction_keeps_its_second_person_line(
        self, container, monkeypatch
    ) -> None:
        """The item path keeps the second-person template plus the result, not the outcome.

        "他要从你手里夺走…" conveys whose item it is, which the 3p verdict sentence can't. The template
        quotes no one, so there's no clash of voices to fix. It describes an intent that hasn't
        played out yet, so the result is added as its own sentence.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _seize_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲夺走了虎符", fact="我夺走了虎符",
                relation_dir="negative", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.SEIZE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _seize_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-b", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我伸手去夺他手中的虎符",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "甲要从你手里夺走虎符，结果：做成了" in prompt
        assert "我伸手去夺" not in prompt

    @pytest.mark.asyncio
    async def test_a_namesake_across_the_map_does_not_answer_for_what_is_not_here(
        self, container, monkeypatch
    ) -> None:
        """An item with the same name elsewhere on the map must not stand in for one that isn't
        here.

        The item slot holds the LLM's free text ("城门"). A name lookup not restricted to the
        location finds a namesake elsewhere: the judge rules on an absent item, the mutation finds
        nothing, and outcome and memory claim a seizure the world never saw.
        """
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _seize_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲夺走了城门", fact="我夺走了城门",
                relation_dir="neutral", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.SEIZE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _seize_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()                      # agent-a is in hall
        env.register_entity(WorldEntity(
            entity_id="far_gate", name="城门", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.AT_LOCATION, presence_ref="garden",
            is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我去夺那座城门",
            target=ActionTarget(acts_on=[Ref.entity("城门", "object")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        )

        changes = [c for r in results for c in r.entity_state_changes]
        assert changes == [], f"不在场的东西不该产出世界变更: {changes}"
        far = env.find_item("far_gate")
        assert far is not None and far.owner_id is None, "别处那件同名物不得易主"

    @pytest.mark.asyncio
    async def test_an_entity_here_is_bound_by_its_id_not_by_the_text_that_found_it(
        self, container, monkeypatch
    ) -> None:
        """The item that is actually here can still be seized, and the mutation carries its entity
        id, not the text used to find it.

        With the wrong id nothing is found: ``change_entity_state`` returns False silently (the
        landing code ignores the return value), the world is unchanged, and outcome and memory
        already say it was taken.
        """
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _seize_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲拿起了密信", fact="我拿起了密信",
                relation_dir="neutral", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.SEIZE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _seize_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="letter_7", name="密信", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.AT_LOCATION, presence_ref="hall",
            is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我拿起那封密信",
            target=ActionTarget(acts_on=[Ref.entity("密信", "object")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        )

        changes = [c for r in results for c in r.entity_state_changes]
        assert [c.entity_id for c in changes] == ["letter_7"]
        for change in changes:
            env.change_entity_state(change, acting_agent_id="agent-a")
        assert env.find_item("letter_7").owner_id == "agent-a", "世界必须真的变了"


    @pytest.mark.asyncio
    async def test_judge_outcome_fallback_quotes_the_first_person_intent(self, container) -> None:
        """The fallback used when the judge gives no outcome must also be clean third person.

        Splicing his words in bare renders "甲我伸手去夺…". This fallback is both the bystander
        observation and the sentence the person acted on reads, so one bad render spoils both. The
        words are quoted in 「」 with attribution, and no final "。" is added because the quote usually
        has one.
        """
        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"success": true, "fact": "我按住了他。"}'   # the judge left out the outcome
        )
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我猛地扑上去按住他",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        assert isinstance(result, ActionResult)
        assert "甲着手做「我猛地扑上去按住他」" in result.outcome
        assert "甲我猛地扑上去" not in result.outcome

    @pytest.mark.asyncio
    async def test_seizing_a_thing_off_the_ground_has_no_victim(self, container, monkeypatch) -> None:
        """An unowned item has no counterparty, so no victim is invented."""
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _seize_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲拿起了灯笼", fact="我拿起了灯笼",
                relation_dir="negative", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.SEIZE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _seize_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="lantern", name="灯笼", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我拿起灯笼",
            target=ActionTarget(acts_on=[Ref.entity("lantern", "item")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        assert isinstance(result, ActionResult)
        assert result.relation_updates == [] and result.target_effects == []

    @pytest.mark.asyncio
    async def test_relinquish_hands_the_thing_over_or_sets_it_down(self, container, monkeypatch) -> None:
        """An item must have a way out of someone's hands: handed to another person, or put down
        here.

        If the holder dying were the only way from HELD to AT_LOCATION, none of the handing over
        described in the text ("递/交给/呈上") would ever reach the world state.
        """
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _give_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲把虎符交给了乙", fact="我把虎符交给了乙",
                relation_dir="positive", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="", deed=Deed.RELINQUISH,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _give_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))

        def _env_with_tally() -> EnvironmentSystem:
            env = _make_environment()
            env.register_entity(WorldEntity(
                entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
                presence=EntityPresence.HELD,
                presence_ref="agent-a", is_takeable=True, state="intact",
            ))
            return env

        # 1. Handed to someone: the holder changes and the recipient gets his own first-person
        # reaction.
        env = _env_with_tally()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我把虎符交到乙手上",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")], reaches=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        assert isinstance(result, ActionResult)
        change = result.entity_state_changes[0]
        assert change.owner_id == "agent-b" and change.location_id is None
        assert [e.agent_id for e in result.target_effects] == ["agent-b"]

        # 2. Just put down: it returns to this location and nobody is affected.
        env = _env_with_tally()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我把虎符搁在案上",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        change = result.entity_state_changes[0]
        assert change.owner_id is None and change.location_id == "hall"
        assert result.target_effects == []

    @pytest.mark.asyncio
    async def test_the_deed_decides_who_receives_not_the_intent(self, container, monkeypatch) -> None:
        """A recipient was bound, but the judge ruled that he smashed the item. The deed decides,
        and the recipient gets nothing.

        The recipient reflects the actor's intent; the deed is what the judge says he actually did.
        Compensating the recipient for a destroyed item because the actor meant to hand it over
        would leave the mutation and the compensation disagreeing.
        """
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _smash_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="", success=True, outcome="甲把虎符砸了", fact="我把虎符砸了",
                relation_dir="negative", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="shattered", deed=Deed.DESTROY,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _smash_judge)
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-a", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我要把虎符交给乙",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")], reaches=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = (await executor.complete(state, 1, agents=agents, environment=env, message_system=None))[0]
        assert isinstance(result, ActionResult)
        assert result.entity_state_changes[0].destroyed is True
        assert result.entity_state_changes[0].owner_id is None
        assert result.target_effects == []      # the item was already his, so nobody lost it, and the recipient got nothing
        assert result.deed == Deed.DESTROY.value

    @pytest.mark.asyncio
    async def test_a_recipient_who_walked_away_blocks_the_handover(self, container) -> None:
        """The recipient may leave between planning and execution, the same race the check on person
        targets handles.

        The failure text can say who he is, but not where he went, which would be free scouting.
        """
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        env.place_agent(agent_id="agent-b", location_id="garden")
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-a", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="我把虎符交到乙手上",
            target=ActionTarget(acts_on=[Ref.entity("tally", "item")], reaches=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        result = state.extra.get("completed_result")
        assert isinstance(result, ActionResult) and not result.succeeded
        assert "乙" in result.failure_reason and "garden" not in result.failure_reason

    @pytest.mark.asyncio
    async def test_item_destroy_declares_removal(self, monkeypatch) -> None:
        """A judged destruction declares an EntityStateChange(destroyed=True) —
        terminal removal wins over pick-up/use and carries no owner/location."""
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _destroy_judge(self, *args, **kwargs) -> _Verdict:
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲砸碎了灯笼", fact="我砸碎了灯笼",
                relation_dir="neutral", actor_damage=0.0, target_damage=0.0, target_relief=0.0,
                new_entity_state="shattered", deed=Deed.DESTROY,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _destroy_judge)
        executor = PhysicalExecutor(None, _directory())  # type: ignore[arg-type]
        env = _make_environment()
        env.register_entity(WorldEntity(
            entity_id="lantern", name="lantern", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True, state="intact",
        ))
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="砸碎灯笼",
            target=ActionTarget(acts_on=[Ref.entity("lantern", "item")]),
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult) and result.succeeded
        assert len(result.entity_state_changes) == 1
        change = result.entity_state_changes[0]
        assert change.entity_id == "lantern"
        assert change.destroyed is True
        assert change.owner_id is None and change.location_id is None

    @pytest.mark.asyncio
    async def test_untracked_object_always_feasible_no_mutation(self, container) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="推倒石柱",
            target=ActionTarget(acts_on=[Ref.entity("石柱", "structure")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        assert result.relation_updates == []

    @pytest.mark.asyncio
    async def test_attack_agent_different_location_returns_infeasible(self) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        executor = PhysicalExecutor(None, _directory())  # type: ignore[arg-type]
        env = EnvironmentSystem()
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="garden")
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="攻击",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert not result.succeeded
        # Rendered by the executor with descriptive referents (directory misses don't fall back to
        # ids), without revealing where the other person actually is.
        assert "不在" in result.outcome
        assert "agent-b" not in result.outcome
        assert "garden" not in result.outcome

    @pytest.mark.asyncio
    async def test_judge_prompt_injects_target_scene_and_legend(self, container) -> None:
        """The adjudication prompt contains the target's persona, vitality, a legend for both
        directions of the relation, and the scene, in clear sections with no agent ids."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"success": true, "fact": "x", "relation": "neutral", '
            '"actor_damage": 0.0, "target_damage": 0.0}'
        )
        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="hall-loc", name="大殿", description="梁柱森然，卫士环立",
            connections={}, is_public=True, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="hall-loc")
        env.place_agent(agent_id="agent-b", location_id="hall-loc")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="制服对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        judge_prompt = _joined(provider.call_history[0])
        assert "【裁决任务】" in judge_prompt and "【目标】" in judge_prompt
        assert "乙" in judge_prompt                         # target persona (name)
        assert "体力状况" in judge_prompt                   # both sides' vitality
        assert "目标对行动者的关系" in judge_prompt          # relation in both directions
        assert "关系数值含义" in judge_prompt               # relation_legend is present
        assert "生命力损耗" in judge_prompt                 # VITALITY_DAMAGE_DEFINITION scale
        assert "梁柱森然" in judge_prompt                   # scene (location description)
        assert "agent-b" not in judge_prompt                # no id leak

    def test_same_place_rule_names_no_particular_kind_of_world(self) -> None:
        """The same-place invariant talks only about moving itself, with no genre-specific wording
        (Rule 7).

        Verbs like "押去", "潜行至", "架起" and "差遣" belong to particular settings (escorting prisoners,
        sneaking about), and "差遣" also assumes a master and servant. The adjudication template
        serves every world, so such words don't belong in it. Adjudication is strict output, where
        examples are the most likely to be copied (§2).
        """
        from engine.narration import SAME_PLACE_VERDICT_RULE

        for word in ("押", "潜行", "架起", "差遣", "派人", "刀", "宫", "府"):
            assert word not in SAME_PLACE_VERDICT_RULE, f"规则里混进了题材说法「{word}」"
        # the invariant itself and its exemption must still be there
        assert "不会改变任何人所在的位置" in SAME_PLACE_VERDICT_RULE
        assert "另一个人" in SAME_PLACE_VERDICT_RULE

    def test_idle_bystander_rule_names_no_particular_kind_of_world(self) -> None:
        """The absence rule likewise only says that someone who didn't act didn't act, with no genre
        wording (Rule 7). It must also state that "nothing happened" is a normal verdict; otherwise
        the judge treats it as a non-answer and fills in something."""
        from engine.scene import IDLE_BYSTANDER_VERDICT_RULE

        for word in ("诏", "符", "宫", "殿", "偷", "刀", "潜行"):
            assert word not in IDLE_BYSTANDER_VERDICT_RULE, f"规则里混进了题材说法「{word}」"
        assert "此刻没有任何动作" in IDLE_BYSTANDER_VERDICT_RULE   # matches the scene line word for word
        assert "不是信息缺失" in IDLE_BYSTANDER_VERDICT_RULE
        assert "什么也没发生" in IDLE_BYSTANDER_VERDICT_RULE

    @pytest.mark.asyncio
    async def test_judge_is_told_this_action_moves_nobody(self, container) -> None:
        """The judge must be told this action doesn't change anyone's location
        (SAME_PLACE_VERDICT_RULE).

        Executors never call move_body, so a verdict that someone ended up elsewhere always
        contradicts the world state. "Consider the current location" isn't enough: the judge still
        has people marched off while scene_line prefixes the outcome with their real location.
        """
        from core.interfaces.llm import LLMScene
        from engine.narration import SAME_PLACE_VERDICT_RULE
        from engine.executors.physical import PhysicalExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"success": true, "fact": "x", "relation": "neutral", '
            '"actor_damage": 0.0, "target_damage": 0.0}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="提起被捆的乙，押他去别处关起来",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        judge_prompt = _joined(provider.call_history[0])
        assert SAME_PLACE_VERDICT_RULE in judge_prompt
        assert "不会改变任何人所在的位置" in judge_prompt
        assert "让**另一个人**去某处不属于此列" in judge_prompt   # narratives about sending someone else aren't caught by this

    @pytest.mark.asyncio
    async def test_physical_judge_is_told_an_idle_bystander_did_nothing(self, container) -> None:
        """When someone present is marked "此刻没有任何动作", the judge needs both that line and the rule
        that goes with it.

        The scene line (user message) states the absence; the rule (system message) says it's an
        established fact not to be filled in. Without either, the judge invents a reaction.
        """
        from core.interfaces.llm import LLMScene
        from engine.scene import IDLE_BYSTANDER_VERDICT_RULE
        from engine.executors.physical import PhysicalExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"success": true, "fact": "x", "relation": "neutral", '
            '"actor_damage": 0.0, "target_damage": 0.0}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="伸手去夺乙手里的东西",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        judge_prompt = _joined(provider.call_history[0])
        assert "此刻没有任何动作" in judge_prompt          # scene side: the absence is stated
        assert IDLE_BYSTANDER_VERDICT_RULE in judge_prompt  # rule side: it's a fact, don't fill it in

    @pytest.mark.asyncio
    async def test_target_damage_single_source_for_background_target(self, container) -> None:
        """The verdict's target_damage is the only source of damage; background targets don't use a
        hard-coded value."""
        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor

        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"success": true, "fact": "一掌击中", "relation": "negative", '
            '"actor_damage": 0.05, "target_damage": 0.4}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="攻击对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        effect = result.target_effects[0]
        assert effect.vitality_damage == pytest.approx(0.4)        # the verdict's value, not a fixed 0.10
        assert effect.relation_toward_actor == ("agent-a", -0.03, -0.03)  # direction from the verdict, fixed magnitude
        assert result.vitality_damage == pytest.approx(0.05)       # the actor's own exertion
        assert result.relation_updates == [("agent-b", -0.03, -0.03)]

    @pytest.mark.asyncio
    async def test_a_failed_reaction_writes_no_memory_but_still_lands_the_damage(
        self, container, monkeypatch
    ) -> None:
        """If the reaction call fails, nothing is written to memory, but the damage still lands: it
        came from the judge, which already succeeded, and has nothing to do with this failure.

        A memory like "某某对我做了一件事。" carries nothing useful on recall, yet it takes a retrieval slot
        and crowds out memories with real content (Rule 1, tier 1, like _adjudication_failed_result
        and _listener_memory).
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="", success=True, outcome="甲一掌击中乙", fact="一掌击中",
                relation_dir="negative", actor_damage=0.0, target_damage=0.4,
                target_relief=0.0, new_entity_state="", deed=Deed.STRIKE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _judge)
        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)

        async def _boom(messages, **kwargs):
            raise RuntimeError("reaction endpoint down")

        monkeypatch.setattr(provider, "complete", _boom)

        env = _make_environment()
        agents = {
            "agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True),
            "agent-b": _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True),
        }
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="攻击对方", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)

        effect = results[0].target_effects[0]
        assert effect.vitality_damage == pytest.approx(0.4)   # the judge's part still lands
        assert effect.factual_memory == ""                    # the cognition part writes nothing

    @pytest.mark.asyncio
    async def test_reaction_is_subjective_and_damage_comes_from_judge(
        self, container, monkeypatch
    ) -> None:
        """Reaction stage: emotion goes through parse_emotion_type (synonyms are normalized), and
        damage comes from the verdict, not the self-report."""
        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict, _damage_label

        async def _judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲一掌击中乙", fact="一掌击中",
                relation_dir="negative",
                actor_damage=0.0, target_damage=0.4, target_relief=0.0, new_entity_state="", deed=Deed.STRIKE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _judge)
        # The reaction LLM returns a Chinese synonym for an emotion and tries to report its own
        # vitality_damage, which must be ignored.
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我被击中了，胸口剧痛", "emotion_type": "愤怒", '
            '"emotion_intensity": 0.8, "emotion_valence": -0.7, '
            '"relation": "negative", "vitality_damage": 0.95}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="攻击对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        effect = result.target_effects[0]
        assert effect.emotion_type == "anger"                   # normalized by parse_emotion_type
        assert effect.vitality_damage == pytest.approx(0.4)     # the verdict's value; the self-reported 0.95 is ignored
        assert effect.relation_toward_actor == ("agent-a", -0.03, -0.03)
        assert "胸口剧痛" in effect.factual_memory
        # The reaction prompt is in character: it includes the subjective relation block and a
        # description of the harm
        reaction_prompt = _joined(
            container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1]
        )
        assert "【你是谁】" in reaction_prompt
        assert "你身上的变化：生命力大损" in reaction_prompt   # 0.3 ≤ 0.4 < 0.7; the line states how bad it is, not what caused it
        assert "伤" not in _damage_label(0.4)                # depletion labels don't assume a cause (exhaustion and hunger fall in the same band)
        assert "关系数值含义" in reaction_prompt              # legend present (reaction side)
        assert "情绪强度" in reaction_prompt                  # emotion_legend scale
        assert "情绪基调" in reaction_prompt

    @pytest.mark.asyncio
    async def test_target_reaction_attributed_to_target_not_initiator(self, container, monkeypatch) -> None:
        """The PHYSICAL target's reaction is B's own first-person cognition (B's personality,
        addressed as "you"), so its trace belongs to B, even though complete() runs under the
        initiator's (A) observe_stage."""
        from core.interfaces.llm import LLMScene
        from core.context import get_log_context, observe_stage
        from core.interfaces.trace import Stage
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _judge(self_inner, agent, **kwargs):  # noqa: ANN001, ANN002, ANN003
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲击中乙", fact="击中", relation_dir="negative",
                actor_damage=0.0, target_damage=0.4, target_relief=0.0, new_entity_state="", deed=Deed.STRIKE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我被击中", "emotion_type": "愤怒", "emotion_intensity": 0.8, '
            '"emotion_valence": -0.7, "relation": "negative"}'
        )
        env = _make_environment()
        agents = {
            "agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True),
            "agent-b": _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True),
        }
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="攻击对方", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)

        captured: list[tuple[str, str]] = []
        orig = container.llm_router.complete

        async def spy(scene, messages, **kw):  # noqa: ANN001, ANN002, ANN003
            captured.append((get_log_context().get("agent_id", ""), _joined(messages) if messages else ""))
            return await orig(scene, messages, **kw)

        monkeypatch.setattr(container.llm_router, "complete", spy)

        # Run complete() under the initiator's (agent-a) observe_stage, as _finalize_execution does.
        with observe_stage(Stage.ACTION, agent_id="agent-a"):
            await executor.complete(state, 1, agents=agents, environment=env, message_system=None)

        # The target reaction prompt contains "【你是谁】"; that call's agent_id must be the target
        # (agent-b), not the initiator.
        reaction_ctx = [ctx for ctx, prompt in captured if "【你是谁】" in prompt]
        assert reaction_ctx, "expected a target-reaction LLM call"
        assert all(ctx == "agent-b" for ctx in reaction_ctx)

    @pytest.mark.asyncio
    async def test_reaction_helpful_action_no_harm_framing(self, container, monkeypatch) -> None:
        """A non-violent PHYSICAL action (helping someone up, target_damage=0): the reaction prompt
        doesn't assume any "损伤/伤势"."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _help_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲扶住了乙", fact="扶住了对方",
                relation_dir="positive",
                actor_damage=0.0, target_damage=0.0, target_relief=0.0, new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _help_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "他扶了我一把，我心头一暖", "emotion_type": "joy", '
            '"emotion_intensity": 0.4, "emotion_valence": 0.6, "relation": "positive"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="搀扶踉跄的对方", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        reaction_prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "甲扶住了乙" in reaction_prompt   # the person acted on reads the verdict sentence (see _counterpart_effect)
        assert "损伤" not in reaction_prompt and "伤势" not in reaction_prompt  # a harmless action doesn't mention injury

    @pytest.mark.asyncio
    async def test_relief_lands_as_a_negative_vitality_delta(self, container, monkeypatch) -> None:
        """Treatment goes through target_relief and lands as a negative vitality_damage: harm and
        relief share one axis."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _rescue_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲为乙敷药止住了伤", fact="我为他敷上了药",
                relation_dir="positive",
                actor_damage=0.0, target_damage=0.0, target_relief=0.12,
                new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _rescue_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "他为我敷了药，痛楚缓了下来", "emotion_type": "joy", '
            '"emotion_intensity": 0.5, "emotion_valence": 0.6, "relation": "positive"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agent_b.personality.apply_vitality_damage(0.6)          # badly hurt: 0.4 left
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="为他敷药", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        effect = results[0].target_effects[0]
        assert effect.vitality_damage == pytest.approx(-0.12)   # negative means relief
        assert effect.death_cause is None
        # The landing side splits harm and relief by sign (apply_target_effect checks != 0), so
        # vitality really goes up.
        await agent_b.apply_target_effect(effect, from_agent_id="agent-a", step=1)
        assert agent_b.personality.state.vitality == pytest.approx(0.52)
        assert agent_b.is_active

    @pytest.mark.asyncio
    async def test_death_verdict_reads_the_net_value_not_the_damage(self, container, monkeypatch) -> None:
        """A stab followed by stanching the wound on the spot: judging by damage alone would declare
        a living man dead."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _mixed_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲刺伤乙后又为他止住了血", fact="我刺伤他又为他止了血",
                relation_dir="neutral",
                actor_damage=0.0, target_damage=0.5, target_relief=0.2,
                new_entity_state="", deed=Deed.STRIKE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _mixed_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "他刺了我又为我止血", "emotion_type": "fear", '
            '"emotion_intensity": 0.7, "emotion_valence": -0.5, "relation": "neutral"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agent_b.personality.apply_vitality_damage(0.55)         # 0.45 left: the net loss of 0.3 leaves him alive; damage alone (0.5) would kill him
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="刺他一刀", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        effect = results[0].target_effects[0]
        assert effect.vitality_damage == pytest.approx(0.3)
        assert effect.death_cause is None                       # the net change doesn't kill him
        await agent_b.apply_target_effect(effect, from_agent_id="agent-a", step=1)
        assert agent_b.personality.state.vitality == pytest.approx(0.15)
        assert agent_b.is_active
        # The harm comes first: being hurt still matters most to him, even if it was offset in the
        # same beat.
        reaction_prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "你身上的变化：生命力大损" in reaction_prompt   # 0.3≤0.5<0.7

    @pytest.mark.asyncio
    async def test_reaction_tells_the_rescued_what_changed_on_him(self, container, monkeypatch) -> None:
        """The rescued person's first-person line must describe what changed in his body. The
        damage-only branch would render treatment as "nothing happened"."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _rescue_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲把昏死的乙弄醒了", fact="我把他弄醒了",
                relation_dir="positive",
                actor_damage=0.0, target_damage=0.0, target_relief=0.18,
                new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _rescue_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "我醒了过来", "emotion_type": "joy", '
            '"emotion_intensity": 0.5, "emotion_valence": 0.5, "relation": "positive"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="把他弄醒", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        reaction_prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "你身上的变化" in reaction_prompt
        assert "你受到的损伤" not in reaction_prompt            # no injury mentioned when he wasn't hurt

    @pytest.mark.asyncio
    async def test_experience_line_takes_the_net_direction(self, container, monkeypatch) -> None:
        """When harm and relief happen together, the line reports the direction and size of the net
        change, since only the net says where his body ended up.

        Reporting the harm side would tell someone who ended up better off that he was slightly worn
        down, the opposite of what happened.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import (
            PhysicalExecutor, _Verdict, _damage_label, _relief_label,
        )

        async def _mixed_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲撞开乙又将他扶稳", fact="我撞开他又扶稳了他",
                relation_dir="positive",
                actor_damage=0.0, target_damage=0.05, target_relief=0.1,
                new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _mixed_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "他撞了我一下又把我扶稳", "emotion_type": "joy", '
            '"emotion_intensity": 0.4, "emotion_valence": 0.3, "relation": "positive"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agent_b.personality.apply_vitality_damage(0.5)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="撞开他又扶稳", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        assert results[0].target_effects[0].vitality_damage == pytest.approx(-0.05)   # net relief
        reaction_prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        # relief wording for a net 0.05, not the depletion wording for damage=0.05
        assert f"你身上的变化：{_relief_label(0.05)}" in reaction_prompt
        assert _damage_label(0.05) not in reaction_prompt

    @pytest.mark.asyncio
    async def test_experience_line_is_omitted_when_the_two_cancel(self, container, monkeypatch) -> None:
        """Harm and relief cancel out exactly, so the line is omitted: his body really is back where
        it was. event_line still describes the action."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict

        async def _wash_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲划伤乙随即为他止住", fact="我划伤他随即为他止住",
                relation_dir="neutral",
                actor_damage=0.0, target_damage=0.2, target_relief=0.2,
                new_entity_state="", deed=Deed.STRIKE,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _wash_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "他划了我一下又给我止住", "emotion_type": "confusion", '
            '"emotion_intensity": 0.4, "emotion_valence": 0.0, "relation": "neutral"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="划他一下又止住", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        assert results[0].target_effects[0].vitality_damage == pytest.approx(0.0)
        reaction_prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "你身上的变化" not in reaction_prompt
        assert "甲划伤乙随即为他止住" in reaction_prompt   # the action is still there, carried by the judge's verdict sentence

    @pytest.mark.asyncio
    async def test_relief_is_not_bound_to_wounds(self, container, monkeypatch) -> None:
        """Vitality can be restored no matter what drained it. Tie this axis to healing wounds and
        feeding a starving man scores 0.

        Vitality is lost to more than injury (death_handler lets people die of hunger and
        exhaustion), so the relief scale and the line the recipient reads describe only how much,
        never why.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor, _Verdict, _relief_label

        async def _feed_judge(self_inner, agent, **kwargs):
            return _Verdict(
                failure_reason="",
                success=True, outcome="甲把水囊递到乙唇边", fact="我给他灌了几口水",
                relation_dir="positive",
                actor_damage=0.0, target_damage=0.0, target_relief=0.08,
                new_entity_state="", deed=Deed.RESTRAIN,
            )

        monkeypatch.setattr(PhysicalExecutor, "_judge", _feed_judge)
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"fact": "他喂我喝了水，我缓了过来", "emotion_type": "joy", '
            '"emotion_intensity": 0.4, "emotion_valence": 0.5, "relation": "positive"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agent_b.personality.apply_vitality_damage(0.75)          # 0.25 left: close to dying of thirst, not wounded
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="把水囊递到他唇边", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        effect = results[0].target_effects[0]
        assert effect.vitality_damage == pytest.approx(-0.08)
        await agent_b.apply_target_effect(effect, from_agent_id="agent-a", step=1)
        assert agent_b.personality.state.vitality == pytest.approx(0.33)
        # Neither the line nor the scale may assume a cause: someone dying of thirst shouldn't be
        # told about his wounds.
        assert "伤" not in _relief_label(0.08)
        reaction_prompt = _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])
        assert "你身上的变化：生命力回了一些" in reaction_prompt

    @pytest.mark.asyncio
    async def test_relief_slot_only_exists_in_the_person_shape(self, container) -> None:
        """target_relief and its explanation depend on the form. Item adjudication has no such key,
        and leaving the explanation in only invites the model to emit it."""
        from core.interfaces.llm import LLMScene
        from engine.executors.physical import PhysicalExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"success": true, "fact": "x", "relation": "neutral", '
            '"actor_damage": 0.0, "target_damage": 0.0, "target_relief": 0.2}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))

        person = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="为他裹伤", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(person, 1, agents=agents, environment=env, message_system=None)
        await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        person_prompt = _joined(provider.call_history[0])
        assert "target_relief" in person_prompt
        assert "生命力挽回" in person_prompt                     # VITALITY_RELIEF_DEFINITION scale
        assert "不得凭空挽回" in person_prompt                   # means gate
        # Relief isn't tied to a cause; if the judge only recognized wounds, losses from anything
        # else could never be relieved.
        assert "不论他亏在哪里" in person_prompt
        assert "亏在哪里都算" in person_prompt

        provider.call_history.clear()
        env.register_entity(WorldEntity(
            entity_id="door", name="木门", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.AT_LOCATION, presence_ref="hall",
            is_takeable=False, state="closed",
        ))
        entity = AgentAction(
            agent_id="agent-a", step=2, action_type=ActionType.PHYSICAL,
            action_description="推开门", target=ActionTarget(acts_on=[Ref.entity("door", "item")]),
        )
        state = await executor.start(entity, 2, agents=agents, environment=env, message_system=None)
        results = await executor.complete(state, 2, agents=agents, environment=env, message_system=None)
        entity_prompt = _joined(provider.call_history[0])
        assert "target_relief" not in entity_prompt
        assert "生命力挽回" not in entity_prompt
        # The same goes for harm, in every form: requiring an injury would score pure physical
        # exhaustion as 0, contradicting the next rule that plain exertion may be ≤0.099.
        for prompt in (person_prompt, entity_prompt):
            assert "造成创伤" not in prompt
            assert "耗在哪里都算" in prompt
        # The item branch drops target_relief even if the LLM returns it (same as target_damage).
        assert results[0].target_effects == []

    @pytest.mark.asyncio
    async def test_infeasible_renders_target_name_and_own_location(self, container) -> None:
        """Feasibility failure text includes the target's name and the actor's own location name,
        but no ids and not where the target is."""
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor

        env = EnvironmentSystem()
        for eid, name in (("hall-loc", "大殿"), ("garden-loc", "后花园")):
            env.space.register_place(Place(
                place_id=eid, name=name, description="",
                connections={}, is_public=True, capacity=50,
            ))
        env.place_agent(agent_id="agent-a", location_id="hall-loc")
        env.place_agent(agent_id="agent-b", location_id="garden-loc")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        executor = PhysicalExecutor(container.llm_router, _directory({"agent-b": agent_b}))
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="攻击",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert not result.succeeded
        assert "乙" in result.outcome and "大殿" in result.outcome
        assert "agent-b" not in result.outcome
        assert "后花园" not in result.outcome   # doesn't reveal where the other person actually is

    @pytest.mark.asyncio
    async def test_untracked_object_has_no_target_effects(self) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        executor = PhysicalExecutor(None, _directory())  # type: ignore[arg-type]
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="破坏石门",
            target=ActionTarget(acts_on=[Ref.entity("石门", "structure")]),
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.target_effects == []


    @pytest.mark.asyncio
    async def test_landmark_physical_action_declares_state_change_without_owner(self, container) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent_a}
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        landmark = WorldEntity(
            entity_id="gate_sign",
            name="城门告示",
            entity_type=WorldEntityType.LANDMARK,
            presence_ref="hall",
            is_takeable=False,
            state="intact",
            description="悬挂于城门上的官方告示",
        )
        env.register_entity(landmark)
        action = AgentAction(
            agent_id="agent-a", step=1,
            action_type=ActionType.PHYSICAL,
            action_description="撕下告示",
            target=ActionTarget(acts_on=[Ref.entity("gate_sign", "landmark")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded
        assert len(result.entity_state_changes) == 1
        change = result.entity_state_changes[0]
        assert change.entity_id == "gate_sign"
        assert change.owner_id is None          # LANDMARK: no pick-up semantics
        assert change.new_state                 # state was determined (non-empty)


# ---------------------------------------------------------------------------
# CovertExecutor
# ---------------------------------------------------------------------------

class TestCovertExecutor:
    @pytest.mark.asyncio
    async def test_covert_start_has_no_bystander_observation(self, container) -> None:
        """A covert action's first step carries nothing to bystanders: whether it's perceived is
        decided only by complete()'s detection adjudication. opening_outcome (god view) still names
        the deed."""
        executor = CovertExecutor(container.llm_router, _directory())
        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="vault", name="vault", description="", connections={}, is_public=False, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="vault")
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT,
            description="潜入密室盗取密诏", estimated_steps=2,  # multi-step → has a start beat
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert _obs_text(state) == ""               # nothing observable to bystanders
        assert "潜入密室盗取密诏" in state.opening_outcome   # god-view outcome still names the deed

    @pytest.mark.asyncio
    async def test_covert_tick_has_no_bystander_observation(self, container) -> None:
        """The covert progress beat splits like its start, at the field level regardless of carry
        routing: outcome names the deed (god view), observation is empty."""
        executor = CovertExecutor(container.llm_router, _directory())
        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="vault", name="vault", description="", connections={}, is_public=False, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="vault")
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT,
            description="潜入密室盗取密诏", estimated_steps=3,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        state.remaining_steps -= 1  # elapsed 2 → a mid-action tick
        tick = await executor.tick(state, 2, agents={}, environment=env, message_system=None)
        assert tick[0].observations == []                    # nothing carried to bystanders
        assert "潜入密室盗取密诏" in tick[0].outcome          # god-view progress still names the deed

    @pytest.mark.asyncio
    async def test_covert_interrupt_no_witnesses_undetected(self, container) -> None:
        """Interrupted alone in a sealed room with a garbage LLM reply: nothing to adjudicate, so no
        detection is claimed."""
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        executor = CovertExecutor(container.llm_router, _directory())
        # Force the LLM-failure path: a non-JSON response gives verdict None, so no exposure is
        # invented.
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = "not json"
        # Build isolated environment: agent-a is alone
        loc = Place(
            place_id="vault", name="vault", description="",
            connections={}, is_public=False, capacity=50,
        )
        env = EnvironmentSystem()
        env.space.register_place(loc)
        env.place_agent(agent_id="agent-a", location_id="vault")

        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        agents = {"agent-a": agent_a}
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.COVERT,
            initiator_id="agent-a",
            participant_ids=["agent-a"],
            purpose="偷听机密",
            started_step=1,
            opening_outcome="开始",
            estimated_steps=3,
        )
        state.remaining_steps = 2  # elapsed = 1
        results = await executor.interrupt(state, 2, agents=agents, environment=env, thought="风声紧")
        assert len(results) == 1
        result = results[0]
        assert "风声紧" in result.outcome and "风声紧" not in result.gist
        # Adjudication failed, so no detection is claimed (detected=False); interrupted, so not
        # achieved.
        assert result.detected is False
        assert result.succeeded is False
        assert "步" not in result.factual_memory  # no steps in the narrative layer
        # Information asymmetry is the point of covert actions: when not exposed, observation is
        # empty and bystanders perceive nothing, while the outcome (the full record, carried
        # privately for the actor) still records the action.
        assert _obs_text(result) == ""
        assert "偷听机密" in result.outcome

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("achieved", "detected"),
        [
            (True, False),   # complete success: found something and nobody noticed
            (True, True),    # found something but was exposed
            (False, False),  # found nothing but stayed hidden
            (False, True),   # complete failure
        ],
    )
    async def test_covert_quadrant_succeeded_and_detected(
        self, container, achieved: bool, detected: bool,
    ) -> None:
        """achieved and detected are independent: succeeded follows achieved, and detected has its
        own channel."""
        import json

        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        # The fact says what he found, not what he did: achieved means something was gained, not
        # that the action finished.
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps(
            {"achieved": achieved, "detected": detected, "fact": "我看清密信上写着三日后动手"},
            ensure_ascii=False,
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="翻看密信",
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is achieved
        assert result.detected is detected
        assert result.factual_memory == "我看清密信上写着三日后动手"

    @pytest.mark.asyncio
    async def test_covert_judge_is_told_this_action_moves_nobody(self, container) -> None:
        """Covert actions don't change location either (SAME_PLACE_VERDICT_RULE). Sneaking somewhere
        ("潜行至某地") is the most common way this goes wrong.

        A verdict can have an agent "潜行至另一处附近，暗中观察了4小时" in a place hours away while he never
        moved. It's the same gap as PHYSICAL, so they share one invariant.
        """
        import json

        from core.interfaces.llm import LLMScene
        from engine.narration import SAME_PLACE_VERDICT_RULE
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = json.dumps(
            {"achieved": True, "detected": False, "fact": "看清了队列虚实"}, ensure_ascii=False,
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="潜行至别处，暗中观察",
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        judge_prompt = _joined(provider.call_history[0])
        assert SAME_PLACE_VERDICT_RULE in judge_prompt
        assert "不会改变任何人所在的位置" in judge_prompt

    @pytest.mark.asyncio
    async def test_covert_judge_is_told_the_watched_person_did_nothing(self, container) -> None:
        """Observational COVERT is where this rule matters most. The goal is to see whether someone
        does something, and the judge has to write what actually happened. If an IDLE target's
        absence isn't stated, expected_outcome is all the judge has to go on, and it will have
        someone who never moved fetch and hide things, exchange glances, and make an item disappear."""
        import json

        from core.interfaces.llm import LLMScene
        from engine.scene import IDLE_BYSTANDER_VERDICT_RULE
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = json.dumps(
            {"achieved": True, "detected": False, "fact": "她始终没动"}, ensure_ascii=False,
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = CovertExecutor(container.llm_router, _directory(agents))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT,
            description="假寐留意乙的一举一动，看她会不会对案上那物做什么",
        )
        state = await executor.start(
            action, 1, agents=agents, environment=env, message_system=None,
        )
        await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        )
        judge_prompt = _joined(provider.call_history[0])
        assert "此刻没有任何动作" in judge_prompt
        assert IDLE_BYSTANDER_VERDICT_RULE in judge_prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize("detected", [False, True])
    async def test_covert_never_authorizes_a_happening(self, container, detected: bool) -> None:
        """A covert action never declares the extra layer that only a watcher would catch, whether
        or not it was exposed.

        This is the structural guarantee that another lurker can't read it. When exposed, bystanders
        already get the exposure as ambient; a sighting would only put the outcome into someone
        else's adjudication context. "The outcome only describes visible traces" is a prompt
        convention; this check is the guarantee.
        """
        import json

        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps(
            {"achieved": True, "detected": detected, "outcome": "甲在帘后窥探",
             "fact": "我听见他们议定三日后动手"},
            ensure_ascii=False,
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="盯住此处",
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        [result] = await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        assert isinstance(result, ActionResult)
        assert result.detected is detected
        assert result.happening == "", "秘密行动绝不把自己的 outcome 交给别人的见闻轨"
        # what he found stays in his own memory channel
        assert "三日后动手" in result.factual_memory
        assert "三日后动手" not in result.happening

    @pytest.mark.asyncio
    async def test_covert_judge_gets_this_places_recent_happenings(self, container) -> None:
        """This is COVERT's only source of intelligence.

        Without it the judge has only a static scene and invents findings, which are written to
        memory, embedded and recalled for many steps. Sightings are written at the end of each beat,
        so the window [started_step-1, now] covers exactly the beats he spent watching, up to the
        sightings' own retention window.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"detected": false, "outcome": "o", "fact": "f", "achieved": true}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        loc = env.get_body_location("agent-a")
        for step, text in ((2, "上一拍之前的事"), (3, "他守着的第一拍"), (4, "他守着的第二拍")):
            env.record_happening(
                location_id=loc, outcome=text, step=step, actor_ids=("someone-else",),
            )
        # what he did himself isn't something he found out
        env.record_happening(
            location_id=loc, outcome="他自己弄出的动静", step=3, actor_ids=("agent-a",),
        )

        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="盯住此处",
        )
        state = await executor.start(
            action, 4, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        state.started_step = 4        # a stakeout starting at beat 4 gives a window starting at beat 3
        await executor.complete(
            state, 4, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )

        prompt = _joined(provider.call_history[-1])
        assert "他守着的第一拍" in prompt and "他守着的第二拍" in prompt
        assert "上一拍之前的事" not in prompt, "窗口之外的不该给"
        assert "他自己弄出的动静" not in prompt, "自己做下的事不是自己探来的"
        assert "落到他眼里耳里的事】" in prompt

    @pytest.mark.asyncio
    async def test_covert_happenings_carry_time_prefixes_and_order_hint(self, container) -> None:
        """Sightings are memory-shaped input and follow the shared memory-injection contract:
        ordering, MEMORY_ORDER_HINT and a time prefix on each entry. Without the prefix the judge
        knows the order but not the spacing, so the fact it writes has no sense of time, and that
        fact goes into memory. Not being persisted doesn't exempt input from the rendering rules.
        """
        from core.interfaces.llm import LLMScene
        from core.prompts import MEMORY_ORDER_HINT, recency_prefix
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"detected": false, "outcome": "o", "fact": "f", "achieved": true}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        loc = env.get_body_location("agent-a")
        env.record_happening(
            location_id=loc, outcome="早些时候的那件事", step=1, actor_ids=("other",),
        )
        env.record_happening(
            location_id=loc, outcome="刚才的那件事", step=5, actor_ids=("other",),
        )

        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="盯住此处",
        )
        state = await executor.start(
            action, 6, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        state.started_step = 2
        await executor.complete(
            state, 6, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )

        prompt = _joined(provider.call_history[-1])
        assert MEMORY_ORDER_HINT in prompt, "定序规则要照统一契约告知,别手搓"
        # The prefix must come from the shared renderer. Don't assert that the two entries render
        # differently: a prompt uses a single wording for past time (see recency_prefix), entries in
        # a short window can share a band, and the ordering carries the spacing.
        for at_step, text in ((1, "早些时候的那件事"), (5, "刚才的那件事")):
            want = recency_prefix(
                now_step=6, ref_step=at_step, seconds_per_step=3600,
                world_start_second_of_day=0,
            )
            assert f"- {want}{text}" in prompt, f"{text} 没走共享的时间渲染器"
        assert "步" not in prompt, "拍号是代码层坐标,绝不进 prompt"

    @pytest.mark.asyncio
    async def test_covert_judge_never_sees_what_a_bystander_intends(self, container) -> None:
        """Covert adjudication can see whether the people present are busy, but not what they
        intend.

        An intent hasn't happened, and this judge's fact goes straight into the actor's memory:
        given intent text, the judge reports it as overheard ("我听见他说…"). Detection only needs
        busy or idle.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"detected": false, "outcome": "o", "fact": "f", "achieved": false}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        env.place_agent(agent_id="agent-b", location_id=env.get_body_location("agent-a"))
        agent_b.personality.update_action_status(
            status=ActionStatus.IN_PROGRESS,
            current_action="我去把密信烧了，免得落在旁人手里",
            remaining_steps=2,
        )

        executor = CovertExecutor(
            container.llm_router, _directory({"agent-a": agent_a, "agent-b": agent_b}),
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="盯住此处",
        )
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        state = await executor.start(
            action, 1, agents=agents, environment=env, message_system=None,
        )
        await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        )

        prompt = _joined(provider.call_history[-1])
        assert "密信" not in prompt, "在场者打算做什么,秘密行动的裁决不该看见"
        assert "此刻意图" not in prompt
        assert "此刻正忙着手上的事" in prompt, "忙/闲这一维要留着——判 detected 靠它"

    @pytest.mark.asyncio
    async def test_covert_judge_is_told_when_nothing_happened_here(self, container) -> None:
        """When nothing happened here, the prompt must say so instead of leaving the section out.

        Coming back empty-handed is normal for covert actions, and "nothing happened" is the key
        fact for judging achieved. Leave it out and the judge, with nothing to go on, invents
        findings that match expected_outcome.
        """
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = (
            '{"detected": false, "outcome": "o", "fact": "f", "achieved": false}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="盯住此处",
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        prompt = _joined(provider.call_history[-1])
        assert "没有任何事发生" in prompt
        assert "探到了什么" in prompt, "裁决任务问的是所得,不是动作完成度"

    @pytest.mark.asyncio
    async def test_covert_missing_field_claims_nothing_gained(self, container) -> None:
        """achieved defaults to False. Assuming he found something when the field is missing would
        grant a success out of nothing, advancing goals and feeding need feedback while he has
        nothing (Rule 1 fallback: pick the option that claims the least)."""
        from core.interfaces.action import ActionResult
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"detected": false, "outcome": "甲在帘后站了一阵。"}'
        )
        env = _make_environment()
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="盯住此处",
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        assert "什么也没探到" in result.factual_memory
        assert result.adjudication_failed is False, "裁决发生了,只是没探到 —— 不是空步"

    @pytest.mark.asyncio
    async def test_covert_judge_prompt_has_scene_no_ids_no_steps(self, container) -> None:
        """The adjudication prompt includes the scene (location description, names of those present,
        items), with no ids or steps."""
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
        provider.fixed_response = '{"achieved": true, "detected": false, "fact": "x"}'

        loc = Place(
            place_id="garden-loc", name="后花园", description="草木深密，假山环绕",
            connections={}, is_public=True, capacity=50,
        )
        dagger = WorldEntity(
            entity_id="dagger-1", name="短匕", entity_type=WorldEntityType.ITEM,
            state="intact", description="",
            presence_ref="garden-loc", is_takeable=True,
        )
        env = EnvironmentSystem()
        env.space.register_place(loc)
        env.register_entity(dagger)
        env.place_agent(agent_id="agent-a", location_id="garden-loc")
        env.place_agent(agent_id="agent-b", location_id="garden-loc")

        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        executor = CovertExecutor(
            container.llm_router,
            LiveWorldDirectory.from_agents({"agent-a": agent_a, "agent-b": agent_b}, env),
        )
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="藏起一件信物",
        )
        state = await executor.start(
            action, 1,
            agents={"agent-a": agent_a, "agent-b": agent_b},
            environment=env, message_system=None,
        )
        await executor.complete(
            state, 1,
            agents={"agent-a": agent_a, "agent-b": agent_b},
            environment=env, message_system=None,
        )
        prompt = _joined(provider.call_history[-1])
        assert "后花园" in prompt and "草木深密" in prompt  # location name and description
        assert "乙" in prompt                                # names of those present
        assert "短匕" in prompt                              # items in the scene
        assert "agent-b" not in prompt and "garden-loc" not in prompt  # no id leak
        assert "步" not in prompt                            # no steps (natural durations)
        assert "【裁决任务】" in prompt and "【现场】" in prompt  # functional sections

    @pytest.mark.asyncio
    async def test_scene_injects_bystander_concurrent_intent_for_functional_judge(self, container) -> None:
        """The scene block gives the functional judge each person's current intent ("此刻意图：「…」"),
        so evidence rests on actual intent, not guesses about temperament. The "我…" is quoted and
        attributed so it isn't confused with the judge's third-person scene; idle people get an
        explicit "此刻没有任何动作". The first-person path (with_background=False) never shows
        intents. begin runs before adjudication, so current_action covers actions starting this
        beat."""
        from agent.personality import AgentActivityStatus
        from engine.scene import assemble_scene_context

        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="hall", name="大殿", description="", connections={}, is_public=True, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="hall")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        directory = LiveWorldDirectory.from_agents({"agent-a": agent_a, "agent-b": agent_b}, env)
        agents = {"agent-a": agent_a, "agent-b": agent_b}

        # IDLE bystander: no intent to quote — the line stays, saying the absence out loud.
        idle_scene = assemble_scene_context(
            "agent-a", environment=env, directory=directory, agents=agents, with_background=True,
            visibility=SceneVisibility.GOD,
        ).text
        assert "乙" in idle_scene and "此刻意图" not in idle_scene
        assert "此刻没有任何动作" in idle_scene

        # 乙 is acting this step (begin sets current_action + IN_PROGRESS), so its intent shows,
        # quoted.
        agent_b.personality.begin_action(
            step=1, description="按剑戒备、环视四下", activity_status=AgentActivityStatus.IDLE,
        )
        judge_scene = assemble_scene_context(
            "agent-a", environment=env, directory=directory, agents=agents, with_background=True,
            visibility=SceneVisibility.GOD,
        ).text
        assert "此刻意图：「按剑戒备、环视四下」" in judge_scene

        # First-person intent is carried verbatim inside the attribution quote. The "我…" reads as
        # 乙's own, so it doesn't need rewriting into third person.
        agent_b.personality.begin_action(
            step=2, description="我即刻拔剑扑向甲", activity_status=AgentActivityStatus.IDLE,
        )
        first_person_intent_scene = assemble_scene_context(
            "agent-a", environment=env, directory=directory, agents=agents, with_background=True,
            visibility=SceneVisibility.GOD,
        ).text
        assert "此刻意图：「我即刻拔剑扑向甲」" in first_person_intent_scene

        # First-person path (with_background=False) never leaks it — information asymmetry.
        first_person_scene = assemble_scene_context(
            "agent-a", environment=env, directory=directory, agents=agents, with_background=False,
            visibility=SceneVisibility.OWN_EYES,
        ).text
        assert "此刻意图" not in first_person_scene
        # The absence line is judge-only too; the in-character path shouldn't learn even that he
        # isn't doing anything.
        assert "此刻没有任何动作" not in first_person_scene

    def test_scene_labels_a_long_background_and_does_not_cut_it(self, container) -> None:
        """The full background goes into the adjudication scene, always under a "背景：" header.
        Without the header it runs straight on from the values and the judge reads them as one
        field. The length cap is set only in the build-time prompt, so nothing is truncated here."""
        from engine.scene import assemble_scene_context

        long_bg = "少年时随父从军，" * 30  # 240 characters
        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="hall", name="大殿", description="", connections={}, is_public=True, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="hall")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", background=long_bg)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        scene = assemble_scene_context(
            "agent-a", environment=env, directory=LiveWorldDirectory.from_agents(agents, env),
            agents=agents, with_background=True, visibility=SceneVisibility.GOD,
        ).text

        assert f"背景：{long_bg}" in scene
        assert "…" not in scene

    def test_scene_lists_what_is_in_hands_and_says_so_when_there_is_nothing(self, container) -> None:
        """The adjudication scene lists what the people present are holding, with the holder, and
        says "无" when there's nothing.

        Without the first, the work judge's scene omits the sword he's holding while the prompt says
        only listed items count. Without the second, an absence becomes a silent blank, the same
        reasoning as "在场的其他人：无".
        """
        from engine.scene import assemble_scene_context

        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="hall", name="大殿", description="", connections={}, is_public=True, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="hall")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        directory = LiveWorldDirectory.from_agents(agents, env)

        assert "- 现场物件：无" in assemble_scene_context(
            "agent-a", environment=env, directory=directory, agents=agents,
            visibility=SceneVisibility.GOD,
        ).text

        env.register_entity(WorldEntity(
            entity_id="lamp", name="灯笼", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True, state="intact",
        ))
        env.register_entity(WorldEntity(
            entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-b", is_takeable=True, state="intact",
        ))
        scene = assemble_scene_context(
            "agent-a", environment=env, directory=directory, agents=agents,
            visibility=SceneVisibility.GOD,
        ).text
        assert "灯笼" in scene
        assert "虎符（由乙持有" in scene
        assert "agent-b" not in scene and "tally" not in scene     # membrane: no ids in the scene text

    def test_ordinary_fixtures_are_one_unnumbered_line_whatever_the_items(self, container) -> None:
        """Every scene states what the place ordinarily has, through any eyes, with or without
        tracked items; it is never numbered or listed as something in hand."""
        from engine.scene import _ORDINARY_FIXTURES_LINE, SceneContext, assemble_scene_context

        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="hall", name="大殿", description="", connections={}, is_public=True, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="hall")
        agents = {"agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲")}
        directory = LiveWorldDirectory.from_agents(agents, env)

        def render(**kw) -> SceneContext:
            return assemble_scene_context(
                "agent-a", environment=env, directory=directory, agents=agents, **kw,
            )

        empty = render(visibility=SceneVisibility.OWN_EYES, split_own_entities=True)
        assert empty.text.count(_ORDINARY_FIXTURES_LINE) == 1
        assert "- 现场物件：无" in empty.text                  # the tracked list stays truthful
        assert empty.own_entities == ()

        env.register_entity(WorldEntity(
            entity_id="lamp", name="灯笼", entity_type=WorldEntityType.ITEM,
            presence_ref="hall", is_takeable=True, state="intact",
        ))
        stocked = render(visibility=SceneVisibility.GOD).text
        assert "灯笼" in stocked
        assert stocked.count(_ORDINARY_FIXTURES_LINE) == 1
        assert "#" not in _ORDINARY_FIXTURES_LINE

    def test_who_tells_it_and_whose_eyes_it_is_seen_through_are_two_axes(
        self, container
    ) -> None:
        """Voice and visibility are separate questions; one combined parameter can't express the
        third combination.

        Using the actor's own name when addressing him breaks the first-person voice (§1). Showing
        him what others have hidden leaks past what the perception layer already blocks. The first
        is voice, the second is visibility. Someone sent on an errand reports back with exactly
        THIRD × OWN_EYES: an onlooker's voice, but only what he could see himself.
        """
        from core.prompts import SituationVoice
        from engine.scene import SceneVisibility, assemble_scene_context

        env = EnvironmentSystem()
        env.space.register_place(Place(
            place_id="hall", name="大殿", description="", connections={}, is_public=True, capacity=50,
        ))
        env.place_agent(agent_id="agent-a", location_id="hall")
        env.place_agent(agent_id="agent-b", location_id="hall")
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        directory = LiveWorldDirectory.from_agents(agents, env)

        mine = WorldEntity(
            entity_id="blade", name="短刀", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-a", is_takeable=True, state="intact",
        )
        hidden = WorldEntity(
            entity_id="token", name="暗记", entity_type=WorldEntityType.ITEM,
            presence=EntityPresence.HELD,
            presence_ref="agent-b", is_takeable=True, state="intact",
        )
        hidden.is_public = False
        env.register_entity(mine)
        env.register_entity(hidden)

        def _scene(**kw) -> str:
            return assemble_scene_context(
                "agent-a", environment=env, directory=directory, agents=agents, **kw,
            ).text

        judge = _scene(visibility=SceneVisibility.GOD)       # functional judge
        assert "短刀（由甲持有" in judge
        assert "暗记" in judge                                # the judge must know what's actually in the room

        actor = _scene(                                      # WORK's self-appraisal scene
            voice=SituationVoice.FIRST, visibility=SceneVisibility.OWN_EYES,
        )
        assert "短刀（由我持有" in actor                       # doesn't address him by his own name
        assert "由甲持有" not in actor
        assert "暗记" not in actor                            # what others have hidden, he shouldn't know

        # The third combination: someone sent out reports back, written in the third person but
        # limited to what he could see himself.
        borne = _scene(voice=SituationVoice.THIRD, visibility=SceneVisibility.OWN_EYES)
        assert "短刀（由甲持有" in borne                       # onlooker's voice: his name, not "我"
        assert "暗记" not in borne                            # not god's-eye visibility

    @pytest.mark.asyncio
    async def test_covert_interrupt_llm_detected(self, container) -> None:
        """The LLM rules that the interrupted action was exposed: succeeded=False, gap=0.8, notes
        carry True, and the verdict's fact is used."""
        from core.interfaces.llm import LLMScene
        from engine.executors.covert import CovertExecutor

        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
            '{"achieved": true, "detected": true, "fact": "仓促间撞翻烛台，被人看见了"}'
        )
        env = _make_environment()  # agent-a and agent-b are both in hall
        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        executor = CovertExecutor(container.llm_router, _directory({"agent-a": agent_a}))
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.COVERT,
            initiator_id="agent-a",
            participant_ids=["agent-a"],
            purpose="窃取兵符",
            started_step=1,
            opening_outcome="开始",
            estimated_steps=4,
        )
        state.remaining_steps = 2  # elapsed = 2
        results = await executor.interrupt(state, 3, agents={"agent-a": agent_a}, environment=env)
        result = results[0]
        # Interrupted means unfinished, so achieved=true from the LLM is ignored.
        assert result.succeeded is False
        assert result.detected is True
        assert "撞翻烛台" in result.factual_memory


# ---------------------------------------------------------------------------
# LLM-failure fallback (CLAUDE.md Rule 1 fallback tiers)
#
# On an LLM failure an executor records an honest "outcome indeterminate" result:
# succeeded=False, no fabricated success/fact, no relation/entity/detection/target effects.
# _RaisingLLM drives the `except` branch (the mock provider never raises).
# ---------------------------------------------------------------------------

class TestExecutorLLMFailureFallback:
    @pytest.mark.asyncio
    async def test_work_outcome_llm_exception_is_honest_indeterminate(self, container) -> None:
        from core.interfaces.action import ActionResult

        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        executor = WorkExecutor(_RaisingLLM(), _directory())  # type: ignore[arg-type]
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        # The adjudication LLM raises, giving a null step: it says honestly that the outcome is
        # unknown, invents no result, and gives no failure_reason.
        assert "一时未能确知是否做成" in result.outcome
        assert "完成了" not in result.outcome   # doesn't invent "完成了" (done)
        assert result.failure_reason == ""      # can't judge doesn't mean there's a reason

    @pytest.mark.asyncio
    async def test_covert_judge_llm_exception_no_fabricated_detection(self, container) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.covert import CovertExecutor

        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        executor = CovertExecutor(_RaisingLLM(), _directory({"agent-a": agent_a}))  # type: ignore[arg-type]
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="窃取兵符",
        )
        state = await executor.start(
            action, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(
            state, 1, agents={"agent-a": agent_a}, environment=env, message_system=None,
        )
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        assert result.detected is False   # a failure doesn't invent an exposure
        assert "未能确知" in result.outcome
        assert "完成" not in result.outcome

    @pytest.mark.asyncio
    async def test_physical_judge_llm_exception_no_fabricated_effects(self, container) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor

        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = PhysicalExecutor(_RaisingLLM(), _directory(agents))  # type: ignore[arg-type]
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="攻击对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.succeeded is False
        assert result.relation_updates == []        # no invented relation change
        assert result.target_effects == []          # no invented memory or damage for B
        assert result.entity_state_changes == []    # no invented world-state change
        assert result.vitality_damage == pytest.approx(0.0)
        assert "攻击对方" in result.outcome and "未能确知" in result.outcome

    @pytest.mark.asyncio
    async def test_talk_summary_llm_exception_no_fabricated_relation(self, container) -> None:
        from core.interfaces.action import ActionResult
        from engine.executors.social import SocialExecutor

        agent_a = _make_agent(container, world_id="w", agent_id="agent-a", name="A", is_main=True)
        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B", is_main=True)
        agents = {"agent-a": agent_a, "agent-b": agent_b}
        executor = SocialExecutor(_RaisingLLM(), _directory(agents))  # type: ignore[arg-type]
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, ActionResult)
            assert result.succeeded is False
            assert result.relation_updates == []        # no relation change based on an exchange that never happened
            assert result.adjudication_failed is True
            # Null steps write no memory: adjudication_failed makes the feedback layer return before
            # reading it, so a line here would only suggest null steps get recorded (same as
            # physical._adjudication_failed_result).
            assert result.factual_memory == ""


# ---------------------------------------------------------------------------
# A missing actor object (agent is None) is an infrastructure fault, not a rule path. It must end
# in an `adjudication_failed=True` null step, never a default success with a made-up
# factual_memory (CLAUDE.md Rule 1).
#
# These cases check that nothing is written, not just that nothing crashes: a reintroduced
# "default to success" would still run, with polluted narrative data.
# ---------------------------------------------------------------------------

class TestExecutorMissingActorAdjudicationFailed:
    @pytest.mark.asyncio
    async def test_physical_missing_actor_is_adjudication_failed(self, container) -> None:
        """PhysicalExecutor: actor missing from agents gives a null step, with no made-up effect on
        the target."""
        from core.interfaces.action import ActionResult
        from engine.executors.physical import PhysicalExecutor

        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙", is_main=False)
        agents = {"agent-b": agent_b}  # actor agent-a is missing
        executor = PhysicalExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL,
            action_description="攻击对方",
            target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.adjudication_failed is True
        assert result.succeeded is False
        assert result.factual_memory == ""          # no made-up first-person memory
        assert result.target_effects == []          # no made-up memory, damage or relation change for B
        assert result.relation_updates == []
        assert result.entity_state_changes == []
        assert result.vitality_damage == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_covert_missing_actor_is_adjudication_failed(self, container) -> None:
        """CovertExecutor: actor missing gives a null step that claims neither success nor
        exposure."""
        from core.interfaces.action import ActionResult
        from engine.executors.covert import CovertExecutor

        executor = CovertExecutor(container.llm_router, _directory())
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.COVERT, description="窃取兵符",
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.adjudication_failed is True
        assert result.succeeded is False
        assert result.factual_memory == ""
        assert result.detected is False   # no invented exposure

    @pytest.mark.asyncio
    async def test_work_missing_actor_is_adjudication_failed(self, container) -> None:
        """WorkExecutor: actor missing gives a null step, never a template success."""
        from core.interfaces.action import ActionResult

        executor = WorkExecutor(container.llm_router, _directory())
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK,
            description="整理账本", estimated_steps=1,
        )
        state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 1, agents={}, environment=env, message_system=None)
        result = results[0]
        assert isinstance(result, ActionResult)
        assert result.adjudication_failed is True
        assert result.succeeded is False
        assert result.factual_memory == ""
        assert "完成了" not in result.outcome   # must not claim the job was done

    @pytest.mark.asyncio
    async def test_talk_missing_participant_is_adjudication_failed(self, container) -> None:
        """SocialExecutor: a missing participant makes both results null steps, with no made-up
        relation change.

        This uses a missing initiator, since production code produces no result for a missing
        target (`if target is not None`).

        Both sides get a null step: without the initiator no dialogue was generated, so giving the
        other party a memory, a verdict and a relation shift is the fabrication Rule 1 forbids.
        """
        from core.interfaces.action import ActionResult
        from engine.executors.social import SocialExecutor

        agent_b = _make_agent(container, world_id="w", agent_id="agent-b", name="B", is_main=True)
        agents = {"agent-b": agent_b}  # initiator agent-a is missing
        executor = SocialExecutor(container.llm_router, _directory(agents))
        env = _make_environment()
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            target_agent_id="agent-b", estimated_steps=2,
        )
        state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
        assert isinstance(state, ActionExecutionState)
        results = await executor.complete(state, 3, agents=agents, environment=env, message_system=None)
        by_agent = {r.action.agent_id: r for r in results}

        absent = by_agent["agent-a"]           # missing initiator: null step
        assert isinstance(absent, ActionResult)
        assert absent.adjudication_failed is True
        assert absent.succeeded is False
        assert absent.relation_updates == []   # no relation change based on an exchange that never happened

        present = by_agent["agent-b"]          # the party who was there also gets a null step: nobody came, so there was no talk
        assert present.adjudication_failed is True
        assert present.relation_updates == []


# ---------------------------------------------------------------------------
# Every tick and terminal outcome says who, where and what; WORK keeps private appraisal out of
# shared channels.
# ---------------------------------------------------------------------------

class TestNarrativePrinciple:
    """The location resolver, name resolver and scene_line together make sure each line says who,
    where and what."""

    def test_narrative_location_name_in_transit_and_miss(self) -> None:
        env = _make_environment()
        assert env.narrative_location_name(IN_TRANSIT) == "途中"   # the in-transit pseudo-location renders as a natural phrase
        assert env.narrative_location_name("nowhere") == "此处"     # a miss renders descriptively, never as a bare id
        assert env.narrative_location_name("hall") == "hall"

    def _state(self, action_type: ActionType, purpose: str, *, steps: int = 4) -> ActionExecutionState:
        return ActionExecutionState.create(
            target=ActionTarget(),
            action_type=action_type, initiator_id="agent-a", participant_ids=["agent-a"],
            purpose=purpose, started_step=1, estimated_steps=steps, opening_outcome="x",
        )

    @pytest.mark.asyncio
    async def test_rest_tick_states_who_where(self, container) -> None:
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = SimpleExecutor(_directory({"agent-a": agent}))
        [tr] = await ex.tick(self._state(ActionType.REST, "闭目养神"), 2,
                             agents={}, environment=env, message_system=None)
        assert tr.outcome.startswith("在hall，")   # where (scene_line prefix)
        assert "甲" in tr.outcome                   # who

    @pytest.mark.asyncio
    async def test_work_tick_states_who_where(self, container) -> None:
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = WorkExecutor(container.llm_router, _directory({"agent-a": agent}))
        [tr] = await ex.tick(self._state(ActionType.WORK, "批阅文书"), 2,
                             agents={}, environment=env, message_system=None)
        assert "甲" in tr.outcome and "在hall，" in tr.outcome

    @pytest.mark.asyncio
    async def test_work_third_person_lines_never_carry_the_description(self, container) -> None:
        """WORK's three 3p strings refer to the job only in general terms and never quote
        ``action_description``.

        That description is a whole first-person sentence, and these strings are also what
        bystanders in the same place observe. They can see him start working, not what he's
        planning. Quoting it would also render "甲着手做「我回到书房…」", as if he said it aloud. ``_outcome``
        and the judge prompt follow the same rule.
        """
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = WorkExecutor(container.llm_router, _directory({"agent-a": agent}))
        desc = "我回到书房，摊开卷宗逐页细读。"
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.WORK, description=desc, estimated_steps=3,
        )
        state = await ex.start(action, 1, agents={}, environment=env, message_system=None)
        assert state.opening_outcome == "在hall，甲着手做手上的事。"
        assert "书房" not in state.opening_outcome
        # bystanders get the same sentence (the work is public, so both channels share the text)
        assert _obs_text(state) == state.opening_outcome

        state.remaining_steps -= 1
        tick = (await ex.tick(state, 2, agents={}, environment=env, message_system=None))[0]
        assert "书房" not in tick.outcome and "手上的事仍在进行" in tick.outcome
        assert _obs_text(tick) == tick.outcome

        ir = (await ex.interrupt(
            state, 2, agents={"agent-a": agent}, environment=env, thought="有更急的事",
        ))[0]
        assert "书房" not in ir.outcome
        assert "有更急的事" in ir.outcome and "有更急的事" not in ir.gist

    @pytest.mark.asyncio
    async def test_work_complete_observable_is_membrane_safe(self, container) -> None:
        """WORK's terminal outcome describes only the visible result (the job is done); his private
        judgment of its quality goes only into the factual memory."""
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = WorkExecutor(container.llm_router, _directory({"agent-a": agent}))
        [res] = await ex.complete(self._state(ActionType.WORK, "批阅文书"), 5,
                                  agents={"agent-a": agent}, environment=env, message_system=None)
        assert "在hall，" in res.outcome                          # scene_line still adds the location prefix
        assert "做完了" in res.outcome                            # visible result
        assert res.outcome != res.factual_memory                 # private judgment stays in the 1p channel
        assert "我" not in res.outcome                            # the 3p channel has no first person
        assert "批阅文书" not in res.outcome                       # doesn't quote the 1p purpose

    @pytest.mark.asyncio
    async def test_move_tick_states_who_in_transit_and_dest(self, container) -> None:
        env = _make_environment(move_steps=3)
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        env.move_body(body_id="agent-a", location_id=IN_TRANSIT)
        ex = MovementExecutor(_directory({"agent-a": agent}))
        state = self._state(ActionType.MOVE, "前往garden", steps=3)
        state.extra["destination"] = "garden"
        state.extra["origin"] = "hall"
        [tr] = await ex.tick(state, 2, agents={}, environment=env, message_system=None)
        assert "甲" in tr.outcome and "途中" in tr.outcome and "garden" in tr.outcome

    @pytest.mark.asyncio
    async def test_covert_tick_states_who_where(self, container) -> None:
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = CovertExecutor(container.llm_router, _directory({"agent-a": agent}))
        [tr] = await ex.tick(self._state(ActionType.COVERT, "潜行查探"), 2,
                             agents={}, environment=env, message_system=None)
        assert "甲" in tr.outcome and "在hall，" in tr.outcome

    @pytest.mark.asyncio
    async def test_social_tick_states_who_where_for_both(self, container) -> None:
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        ex = SocialExecutor(container.llm_router, _directory({"agent-a": a, "agent-b": b}))
        state = ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.TALK, initiator_id="agent-a",
            participant_ids=["agent-a", "agent-b"], purpose="议事",
            started_step=1, estimated_steps=3, opening_outcome="x",
        )
        state.extra["target_id"] = "agent-b"
        results = await ex.tick(state, 2, agents={"agent-a": a, "agent-b": b},
                                environment=env, message_system=None)
        assert len(results) == 2
        for tr in results:
            assert "甲" in tr.outcome and "乙" in tr.outcome and "在hall，" in tr.outcome


class TestStartArbitratedMembrane:
    """The start step is a real first execution step, so its ActionResult must keep the three
    channels on the right side of the membrane:
      - outcome (3p god-view) → the opening marker, always.
      - observation (3p bystander) → the executor's opening_observation, VERBATIM. It is never
        derived from the outcome/marker: a public start authors it equal to the outcome, COVERT/
        mid-edge MOVE leave it "", and an unset field defaults "" (fail-safe — nothing leaks).
      - factual_memory (1p) → "" — start writes NO memory, so no 3p string may sit in the 1p field.
    """

    def _arbiter(self):
        from engine.arbiter import ExecutionArbiter
        from engine.clock import GlobalClock, WorldTimeConfig
        # _start_arbitrated with a non-empty opening_outcome touches none of these deps
        # (directory/clock are used only by the empty-narrative fallback), so a minimal wiring is
        # enough to exercise the three-channel mapping.
        from engine.execution_processor import ExecutionProcessor

        registry = ActionExecutorRegistry()
        env = EnvironmentSystem()
        return ExecutionArbiter(
            executor_registry=registry,
            environment=env,
            message_system=None,
            directory=_directory(),
            clock=GlobalClock(WorldTimeConfig()),
            processor=ExecutionProcessor(
                executor_registry=registry, environment=env,
                message_system=None, directory=_directory(),
            ),
        )

    def _exec_state(self, *, opening_outcome: str, opening_observation: str = ""):
        obs = [Observed(location_id="corridor", text=opening_observation)] if opening_observation else []
        return ActionExecutionState.create(
            action_type=ActionType.MOVE, initiator_id="agent-a", participant_ids=["agent-a"],
            purpose="从hall前往garden", started_step=1, estimated_steps=3,
            opening_outcome=opening_outcome, opening_observations=obs,
            target=ActionTarget(),
        )

    def _plan(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            agent_id="agent-a", step=1,
            spatial=SimpleNamespace(visible_agent_ids=[]), inbox=[],
        )

    def test_distinct_observation_and_no_first_person_memory(self) -> None:
        """opening_observation set → observation uses it (DISTINCT from the outcome opening);
        factual_memory is "" (① — never a 3p marker in the 1p channel)."""
        arbiter = self._arbiter()
        state = self._exec_state(
            opening_outcome="甲从hall动身前往garden，路程约3小时。",
            opening_observation="在corridor，甲途经此地，正赶往garden。",
        )
        aa = arbiter._start_arbitrated(self._plan(), state, location_id="corridor")
        assert aa.action_result.outcome == "甲从hall动身前往garden，路程约3小时。"
        assert _obs_text(aa.action_result) == "在corridor，甲途经此地，正赶往garden。"
        assert _obs_text(aa.action_result) != aa.action_result.outcome
        assert aa.action_result.factual_memory == ""      # ① no 3p in the 1p channel

    def test_explicit_empty_observation_stays_empty(self) -> None:
        """opening_observations="" (② covert, or MOVE mid-edge) → observation stays "", NOT the
        marker: nothing is observable this step, so nothing carries to bystanders."""
        arbiter = self._arbiter()
        state = self._exec_state(
            opening_outcome="甲悄悄着手做「潜入密室」。", opening_observation="",
        )
        aa = arbiter._start_arbitrated(self._plan(), state, location_id="vault")
        assert _obs_text(aa.action_result) == ""
        assert aa.action_result.outcome == "甲悄悄着手做「潜入密室」。"  # god-view still sees it

    def test_unset_observation_defaults_empty_never_falls_back(self) -> None:
        """An unset opening_observation defaults "" and is NOT back-filled from the outcome/marker
        — the fail-safe rule: the carry channel reads observation verbatim, so a beat whose author
        forgot to authorize a bystander view carries NOTHING rather than leaking the god-view text.
        (Public starts author opening_observations=opening_outcome explicitly — see the executors.)"""
        arbiter = self._arbiter()
        state = self._exec_state(opening_outcome="甲着手清点府库。")  # opening_observation unset
        aa = arbiter._start_arbitrated(self._plan(), state, location_id="hall")
        assert _obs_text(aa.action_result) == ""              # NOT the marker — no fallback
        assert aa.action_result.outcome == "甲着手清点府库。"   # god-view still authored


# ---------------------------------------------------------------------------
# Failure reasons: succeeded=False must come with a failure_reason, which stays in the full record
# and out of the bystander channel.
# ---------------------------------------------------------------------------

class TestFailureReasonContract:
    """`failure_reason` is the rendering layer's only structured source for why an action failed
    (see the ActionResult viewpoint contract).

    These cases pin down how it divides the work with the other three channels. Without a field of
    its own, each call site would write the reason into its own template and append a 3p template to
    the 1p memory text.
    """

    @staticmethod
    def _work_state() -> ActionExecutionState:
        return ActionExecutionState.create(
            target=ActionTarget(),
            action_type=ActionType.WORK, initiator_id="agent-a", participant_ids=["agent-a"],
            purpose="整理账本", started_step=1, estimated_steps=1, opening_outcome="x",
        )

    @pytest.mark.asyncio
    async def test_infeasible_talk_carries_structural_reason(self, container) -> None:
        """Unmet precondition: the reason is a structured fact (the target isn't here) and goes into
        failure_reason as-is, without asking the LLM."""
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = SocialExecutor(container.llm_router, _directory({"agent-a": a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK, description="与乙相谈",
        )
        action.target = ActionTarget(acts_on=[Ref.agent("ghost")], claims=[Ref.agent("ghost")])   # not here
        state = await ex.start(action, 1, agents={"agent-a": a}, environment=env, message_system=None)
        [res] = await ex.complete(state, 1, agents={"agent-a": a}, environment=env, message_system=None)

        assert res.succeeded is False and res.not_executed is True
        assert res.failure_reason                       # must be able to say why
        assert _obs_text(res) == ""                    # the reason never leaks into the bystander channel

    @pytest.mark.asyncio
    async def test_infeasible_talk_quotes_the_first_person_intent(self, container) -> None:
        """The actor's first-person words inside this third-person outcome are quoted in 「」 and
        attributed.

        Spliced in bare, they render as "甲本想我去劝他回心转意", where the actor's "我" belongs to no one.
        """
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = SocialExecutor(container.llm_router, _directory({"agent-a": a}))
        action = _make_action(
            agent_id="agent-a", action_type=ActionType.TALK,
            description="我去劝他回心转意",
        )
        action.target = ActionTarget(acts_on=[Ref.agent("ghost")], claims=[Ref.agent("ghost")])   # not here
        state = await ex.start(action, 1, agents={"agent-a": a}, environment=env, message_system=None)
        [res] = await ex.complete(state, 1, agents={"agent-a": a}, environment=env, message_system=None)
        assert "甲本想做「我去劝他回心转意」" in res.outcome
        assert "甲本想我去劝他" not in res.outcome
        # 1p memory is in the same voice as his words, so they stay unquoted.
        assert res.factual_memory.startswith("想要做：我去劝他回心转意")

    @pytest.mark.asyncio
    async def test_adjudication_failure_asserts_no_reason(self, container) -> None:
        """If adjudication never happened, there's no reason to give: a null step invents nothing
        (Rule 1, tier 1)."""
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = "not json"
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = WorkExecutor(container.llm_router, _directory({"agent-a": a}))
        [res] = await ex.complete(
            self._work_state(), 1, agents={"agent-a": a}, environment=env, message_system=None,
        )
        assert res.adjudication_failed is True
        assert res.failure_reason == ""

    @pytest.mark.asyncio
    async def test_judged_failure_reason_never_reaches_observation(self, container) -> None:
        """The judge's reason belongs to the full record: it goes into an outcome-side field, never
        into the bystander's perception string."""
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "success": False, "fact": "我没能理清账目。",
            "outcome": "甲对着账本忙了半日，终究没理清。", "why": "账册本身有出入",
        }, ensure_ascii=False)
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = WorkExecutor(container.llm_router, _directory({"agent-a": a}))
        [res] = await ex.complete(
            self._work_state(), 1, agents={"agent-a": a}, environment=env, message_system=None,
        )
        assert res.failure_reason == "账册本身有出入"
        assert res.failure_reason not in _obs_text(res)
        assert "我" not in res.outcome          # the 3p channel has no first person


class TestTalkObservationNamesParticipants:
    """The code, not the judge, guarantees that the bystander line says who: left to the judge it
    often says only "二人" ("在东宫，二人低语匆匆…") and bystanders can't tell who is talking.
    """

    @pytest.mark.asyncio
    async def test_judge_view_is_prefixed_with_both_names(self, container) -> None:
        """The judge supplies only the demeanor and the code adds the subject, even when the judge
        names nobody."""
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "dialogue": [{"speaker": 1, "line": "近来可好？"}, {"speaker": 2, "line": "尚可。"}],
            "observation": "二人低语匆匆，神色凝重，随即分头离去。",   # a line from the judge with no names (an LLM response, not an ActionResult)
            "fact": "我与他谈了一场。", "success": True, "relation": "neutral",
        }, ensure_ascii=False)
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="李建成")
        b = _make_agent(container, world_id="w", agent_id="agent-b", name="魏徵")
        agents = {"agent-a": a, "agent-b": b}
        ex = SocialExecutor(container.llm_router, _directory(agents))
        action = _make_action(agent_id="agent-a", action_type=ActionType.TALK, description="议事")
        action.target = ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")])
        state = await ex.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await ex.complete(state, 1, agents=agents, environment=env, message_system=None)

        obs = _obs_text(results[0])
        assert "李建成" in obs and "魏徵" in obs, f"旁观者必须知道是谁在谈,实际:{obs}"
        assert "二人低语匆匆" in obs, "判官的举止描述应当保留"
        assert obs.startswith("在"), "地点前缀仍由 scene_line 统一加"
        # The membrane still holds: bystanders don't get the dialogue.
        assert "近来可好" not in obs and "尚可" not in obs

    @pytest.mark.asyncio
    async def test_names_survive_a_judge_that_gave_no_view(self, container) -> None:
        """Without an observation from the judge, the fallback is a neutral line that still names
        them."""
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "dialogue": [{"speaker": 1, "line": "近来可好？"}],
            "fact": "我与他谈了一场。", "success": True, "relation": "neutral",
        }, ensure_ascii=False)
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="李建成")
        b = _make_agent(container, world_id="w", agent_id="agent-b", name="魏徵")
        agents = {"agent-a": a, "agent-b": b}
        ex = SocialExecutor(container.llm_router, _directory(agents))
        action = _make_action(agent_id="agent-a", action_type=ActionType.TALK, description="议事")
        action.target = ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")])
        state = await ex.start(action, 1, agents=agents, environment=env, message_system=None)
        results = await ex.complete(state, 1, agents=agents, environment=env, message_system=None)

        obs = _obs_text(results[0])
        assert "李建成" in obs and "魏徵" in obs, f"兜底路径同样必须点名,实际:{obs}"


class TestActorAlwaysNamedIn3p:
    """Code guarantees the "who": all three executors whose judge writes a 3p outcome go through
    ensure_actor_named.

    A prompt instruction to name the actor is only a convention. This checks both directions: a
    missing name is added, an existing one is left alone.
    """

    def test_helper_adds_only_when_missing(self) -> None:
        from engine.narration import ensure_actor_named
        # missing name: added
        assert ensure_actor_named("一拳打中对方肩头。", "李世民") == "李世民：一拳打中对方肩头。"
        # name present: unchanged (the judge's sentence is usually livelier than the template)
        assert ensure_actor_named("李世民一拳打中李元吉。", "李世民") == "李世民一拳打中李元吉。"
        # when the body has its own subject, join with a colon instead of concatenating, which would
        # give the broken "李世民对方早有防备"
        assert ensure_actor_named("对方早有防备。", "李世民") == "李世民：对方早有防备。"
        # no empty prefix when there's no name
        assert ensure_actor_named("有人动了手。", "") == "有人动了手。"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action_type", [ActionType.WORK, ActionType.COVERT])
    async def test_anonymous_judge_outcome_gets_the_name_back(self, container, action_type) -> None:
        """An outcome where the judge named nobody must name the actor by the time it's recorded."""
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
            "success": True, "achieved": True, "detected": True,
            "outcome": "那桩事悄没声地办妥了。",     # no subject
            "fact": "我把事办妥了。", "why": "",
        }, ensure_ascii=False)
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="李世民")
        agents = {"agent-a": agent}
        ex = (WorkExecutor if action_type == ActionType.WORK else CovertExecutor)(
            container.llm_router, _directory(agents),
        )
        state = ActionExecutionState.create(
            target=ActionTarget(), action_type=action_type, initiator_id="agent-a",
            participant_ids=["agent-a"], purpose="办那桩事", started_step=1,
            estimated_steps=1, opening_outcome="x",
        )
        [res] = await ex.complete(state, 1, agents=agents, environment=env, message_system=None)

        assert "李世民" in res.outcome, f"3p 通道必须点名行动者,实际:{res.outcome}"
        assert "那桩事悄没声地办妥了" in res.outcome, "判官的叙述本身应当保留"
        if _obs_text(res):             # a COVERT exposure shows bystanders the same string, so adding the name once fixes both channels
            assert "李世民" in _obs_text(res)


class TestClosedWorldRuleReachesEveryFactAuthor:
    """Every prompt that asserts what happened in the world must actually include the closed-world
    rule.

    Without it, dialogue can have a character say "……已伏诛" while both men are alive in the
    snapshot: the model knows the historical record and nothing in the context says otherwise.

    These check the rendered prompt, not the source: the rule is inserted via both f-strings and
    concatenation, and only the rendered text proves it isn't sent as a literal
    `{CLOSED_WORLD_FACT_RULE}`.
    """

    @staticmethod
    def _prompt_of(container) -> str:
        return _joined(container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history[-1])

    @staticmethod
    def _assert_carries(prompt: str, *, first_person: bool) -> None:
        """Check that the whole constant appears verbatim: a hard-coded sentence from it would
        break on rewording and wouldn't prove the whole rule reached the model."""
        from core.prompts import (
            CLOSED_WORLD_FACT_RULE, CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
        )
        want = CLOSED_WORLD_FACT_RULE_FIRST_PERSON if first_person else CLOSED_WORLD_FACT_RULE
        assert want in prompt, "闭世界守则没进 prompt"
        assert "{CLOSED_WORLD" not in prompt, "占位符原样发给了模型（没被渲染）"

    @pytest.mark.asyncio
    async def test_talk_dialogue_and_self_summary(self, container) -> None:
        """Dialogue generation (functional) and per-person self-appraisal (in character) each carry
        their own version."""
        env = _make_environment()
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        b = _make_agent(container, world_id="w", agent_id="agent-b", name="乙")
        agents = {"agent-a": a, "agent-b": b}
        ex = SocialExecutor(container.llm_router, _directory(agents))
        action = _make_action(agent_id="agent-a", action_type=ActionType.TALK, description="议事")
        action.target = ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")])
        state = await ex.start(action, 1, agents=agents, environment=env, message_system=None)
        await ex.complete(state, 1, agents=agents, environment=env, message_system=None)

        joined = "\n".join(
            _joined(m) for m in container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).call_history
        )
        from core.prompts import (
            CLOSED_WORLD_FACT_RULE, CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
        )
        assert CLOSED_WORLD_FACT_RULE in joined, "对白生成(中立叙事者)缺 functional 版"
        assert CLOSED_WORLD_FACT_RULE_FIRST_PERSON in joined, "逐人自评(第一人称)缺 in-character 版"
        assert "{CLOSED_WORLD" not in joined

    @pytest.mark.asyncio
    async def test_work_self_judge_carries_first_person_rule(self, container) -> None:
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent}
        ex = WorkExecutor(container.llm_router, _directory(agents))
        state = ActionExecutionState.create(
            target=ActionTarget(), action_type=ActionType.WORK, initiator_id="agent-a",
            participant_ids=["agent-a"], purpose="整理账本", started_step=1,
            estimated_steps=1, opening_outcome="x",
        )
        await ex.complete(state, 1, agents=agents, environment=env, message_system=None)
        self._assert_carries(self._prompt_of(container), first_person=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["covert", "physical"])
    async def test_third_party_judges_carry_functional_rule(self, container, kind) -> None:
        env = _make_environment()
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        agents = {"agent-a": agent}
        if kind == "covert":
            ex = CovertExecutor(container.llm_router, _directory(agents))
            state = ActionExecutionState.create(
                target=ActionTarget(), action_type=ActionType.COVERT, initiator_id="agent-a",
                participant_ids=["agent-a"], purpose="窥探", started_step=1,
                estimated_steps=1, opening_outcome="x",
            )
        else:
            from engine.executors.physical import PhysicalExecutor
            ex = PhysicalExecutor(container.llm_router, _directory(agents))
            state = ActionExecutionState.create(
                target=ActionTarget(acts_on=[Ref.entity("door", "object")]),
                action_type=ActionType.PHYSICAL, initiator_id="agent-a",
                participant_ids=["agent-a"], purpose="推门", started_step=1,
                estimated_steps=1, opening_outcome="x",
            )
        await ex.complete(state, 1, agents=agents, environment=env, message_system=None)
        self._assert_carries(self._prompt_of(container), first_person=False)


class TestDepartureIsDeclaredAtTheOrigin:
    """Every leg of a trip leaves a trace where it passes: the origin, each waypoint and the
    destination.

    The origin is easiest to miss: ``start()`` calls move_body before building the record, so its
    location_id is already the new place. Vanishing from visible_agent_ids isn't enough, since
    SpatialPerception has no diff between steps and the disappearance is silent.

    The executor only declares it (opening_observations); runtime delivers it (see the routing case
    in tests/unit/test_runtime_smoke.py). This only checks what is declared and when.
    """

    @staticmethod
    async def _start(container, *, move_steps: int):
        env = _make_environment(move_steps=move_steps)
        a = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
        ex = MovementExecutor(_directory({"agent-a": a}))
        action = _make_action(agent_id="agent-a", action_type=ActionType.MOVE, description="前往garden")
        action.target = ActionTarget(acts_on=[Ref.place("garden")])
        return await ex.start(action, 1, agents={"agent-a": a}, environment=env, message_system=None)

    @staticmethod
    def _declared(state) -> tuple[str, str]:
        origin = [o for o in state.opening_observations if o.location_id == "hall"]
        assert len(origin) == 1, f"应恰好在出发地声明一条:{state.opening_observations}"
        return origin[0].location_id, origin[0].text

    @pytest.mark.asyncio
    async def test_departure_is_declared_against_the_origin(self, container) -> None:
        place, text = self._declared(await self._start(container, move_steps=3))
        assert place == "hall", "必须落在**出发地**,而不是记录自己的新位置"
        assert "甲" in text and "离开" in text

    @pytest.mark.asyncio
    async def test_declared_even_when_the_trip_takes_one_step(self, container) -> None:
        """With duration=1, opening_outcome is empty because complete finishes the trip in the same
        step, but people at the origin still see him leave.

        Hanging the departure on opening_outcome would miss this case, so it gets its own test.
        """
        state = await self._start(container, move_steps=1)
        assert state.opening_outcome == ""
        _, text = self._declared(state)
        assert "离开" in text                        # the declaration is still there

    @pytest.mark.asyncio
    async def test_only_the_next_hop_is_revealed_never_the_far_destination(self, container) -> None:
        """Watching someone leave, you see which road he takes, not where he's ultimately headed.

        On a single-hop trip the next hop is the destination, and that is genuinely visible. On a
        multi-hop trip the destination can't be seen from the origin, so only the next hop is
        reported. The wording must not announce a destination, as "正赶往" does.
        """
        _, text = self._declared(await self._start(container, move_steps=1))
        assert "garden" in text          # the next hop is a direction, so it's visible
        assert "正赶往" not in text       # no wording that announces a destination


class TestMovementCarry:
    """MOVE can force others along; the people taken along are participants, not a separate list.

    That's what makes the design work: their bodies are occupied by the trip, so one action per
    body, conscription conflicts, teardown, self-filtering and turn consumption are all handled by
    existing machinery. MOVE implements none of it itself (see Conscription).
    """

    def _carry_action(self, carried: list[str], *, dest: str = "garden", steps: int = 1) -> AgentAction:
        return AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.MOVE,
            action_description="拽着他离开", estimated_steps=steps,
            target=ActionTarget(acts_on=[Ref.place(dest)], claims=[Ref.agent(a) for a in carried]),
        )

    @pytest.mark.asyncio
    async def test_admission_is_alive_and_co_located_only(self, container) -> None:
        """Taking someone along by force has only two physical preconditions. It doesn't ask whether
        he's willing or what he's doing.

        Being busy isn't grounds for refusal; that's the difference between COMPEL and INVITE, and
        arbitration enforces it, not this code. This only answers whether he can be reached.
        """
        env = _make_environment()
        free = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        dead = _make_agent(container, world_id="w", agent_id="agent-c", name="C")
        dead.is_active = False
        away = _make_agent(container, world_id="w", agent_id="agent-e", name="E")
        env.place_agent(agent_id="agent-c", location_id="hall")
        env.place_agent(agent_id="agent-e", location_id="garden")
        agents = {"agent-b": free, "agent-c": dead, "agent-e": away}

        kept = carried_bodies("agent-a", ["agent-b", "agent-c", "agent-e"], agents, env)
        assert kept == ["agent-b"]

    @pytest.mark.asyncio
    async def test_the_carried_body_is_a_participant_not_a_side_list(self, container) -> None:
        env = _make_environment()
        taken = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-b": taken}
        executor = MovementExecutor(_directory())

        state = await executor.start(
            self._carry_action(["agent-b"]), 1,
            agents=agents, environment=env, message_system=None,
        )

        assert state.participant_ids == ["agent-a", "agent-b"]
        assert env.get_body_location("agent-b") == env.get_body_location("agent-a")

    @pytest.mark.asyncio
    async def test_each_participant_gets_its_own_result_on_arrival(self, container) -> None:
        """The person taken along gets a participant result, not a target effect.

        His own location is then written by the MOVE branch of ``Agent._apply_feedback`` (reading
        ``target.acted_on_place``), the same path as when the actor walks, so no separate
        displacement field is needed.
        """
        env = _make_environment()
        taken = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-b": taken}
        executor = MovementExecutor(_directory())
        state = await executor.start(
            self._carry_action(["agent-b"]), 1,
            agents=agents, environment=env, message_system=None,
        )

        results = await executor.complete(
            state, 1, agents=agents, environment=env, message_system=None,
        )

        assert [r.action.agent_id for r in results] == ["agent-a", "agent-b"]
        carried = results[1]
        assert carried.action.target.acted_on_place == "garden"
        assert env.get_body_location("agent-b") == "garden"
        # First person, from the passive side, with no agent ids (ids don't exist in the narrative
        # layer).
        assert carried.factual_memory.startswith("我被")
        assert "agent-" not in carried.factual_memory
        # One event happens once: both results share the same narration and the same bystander
        # delivery.
        assert carried.outcome == results[0].outcome
        assert carried.observations == results[0].observations

    @pytest.mark.asyncio
    async def test_a_journey_cut_short_still_lands_and_tells_the_carried_body(self, container) -> None:
        """An unfinished trip still moved him, so the interrupt path also gives him his result, at
        his real location."""
        env = _make_chain_env(["n0", "n1", "n2", "n3"], edge_steps=2)
        taken = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        env.place_agent(agent_id="agent-b", location_id="n0")
        agents = {"agent-b": taken}
        executor = MovementExecutor(_directory())
        state = await executor.start(
            self._carry_action(["agent-b"], dest="n3", steps=6), 1,
            agents=agents, environment=env, message_system=None,
        )
        state.remaining_steps -= 2

        results = await executor.interrupt(
            state, 3, agents=agents, environment=env, cause="有人拦路",
        )

        assert [r.action.agent_id for r in results] == ["agent-a", "agent-b"]
        landed = results[1].action.target.acted_on_place
        assert landed == env.get_body_location("agent-b") == env.get_body_location("agent-a")
        assert landed != IN_TRANSIT, "挣脱/中止的代价不该是从世界上消失"

    @pytest.mark.asyncio
    async def test_start_touches_the_world_only_as_its_last_act(self, container) -> None:
        """If ``start()`` raises partway through, the world must be untouched, so the world
        mutation is its last step. Otherwise a crash leaves someone IN_TRANSIT forever: no execution
        is left to move him, and arbitration's executor_start_failed only logs.
        """
        import engine.executors.movement as movement_mod

        env = _make_environment(move_steps=3)
        executor = MovementExecutor(_directory())
        # The failure has to land in the window where the ordering matters: building the state,
        # which comes after all the narrative computation and before the world mutation. An earlier
        # failure (say, an empty directory) comes before the move under either ordering, so the test
        # would prove nothing.
        original_create = movement_mod.ActionExecutionState.create
        movement_mod.ActionExecutionState.create = staticmethod(
            lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        try:
            with pytest.raises(RuntimeError):
                await executor.start(
                    self._carry_action([], steps=3), 1,
                    agents={}, environment=env, message_system=None,
                )
        finally:
            movement_mod.ActionExecutionState.create = original_create

        assert env.get_body_location("agent-a") == "hall", "崩在中途,人不该已经动身"

    @pytest.mark.asyncio
    async def test_a_journey_that_cannot_be_made_claims_nobody(self, container) -> None:
        """A trip that can't happen can't take anyone along. claim_bodies must agree with start(),
        or a trip that never departs would first cancel whatever the companions were doing."""
        env = _make_environment(move_steps=3)
        taken = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-b": taken}
        executor = MovementExecutor(_directory())
        action = self._carry_action(["agent-b"], steps=3)
        assert executor.claim_bodies(action, agents=agents, environment=env) == ["agent-b"]

        nowhere = AgentAction(
            agent_id="agent-a", step=1, action_type=ActionType.MOVE,
            action_description="前往不存在的地方",
            target=ActionTarget(acts_on=[Ref.place("nowhere")], claims=[Ref.agent("agent-b")]),
            estimated_steps=3,
        )
        assert executor.claim_bodies(nowhere, agents=agents, environment=env) == []
        state = await executor.start(
            nowhere, 1, agents=agents, environment=env, message_system=None,
        )
        assert state.participant_ids == ["agent-a"]

    @pytest.mark.asyncio
    async def test_the_thought_belongs_to_whoever_broke_off(self, container) -> None:
        """When the person being taken along calls a halt, the thought is his. It must not be
        attributed to the one taking him, let alone written into that person's memory.

        Interrupts pick candidates by ``participant_ids``, so a person taken along can halt the
        whole trip. On a solo MOVE the one who halts and the initiator are always the same, so
        mixing them up wouldn't show; once others can be taken along, it's a real bug.
        """
        env = _make_chain_env(["n0", "n1", "n2", "n3"], edge_steps=2)
        taken = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        env.place_agent(agent_id="agent-b", location_id="n0")
        agents = {"agent-b": taken}
        executor = MovementExecutor(_directory({
            "agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲"),
            "agent-b": taken,
        }))
        state = await executor.start(
            self._carry_action(["agent-b"], dest="n3", steps=6), 1,
            agents=agents, environment=env, message_system=None,
        )
        state.remaining_steps -= 2

        results = await executor.interrupt(
            state, 3, agents=agents, environment=env,
            interrupted_agent_id="agent-b", thought="我非挣开不可",
        )

        actor_result, carried_result = results
        assert "我非挣开不可" not in actor_result.factual_memory, "别人的念头不进他的记忆"
        assert "我非挣开不可" in carried_result.factual_memory, "挣脱的人得知道自己为什么挣脱"
        # The god view quotes the thought and attributes it to whoever actually halted the trip, on
        # that person's own record only. The taker's outcome feeds his own cognition (the goal judge
        # reads it), so putting it there would let him read that the other wants to get away.
        assert "B当时的心思：我非挣开不可" in carried_result.outcome
        assert "我非挣开不可" not in actor_result.outcome
        assert carried_result.outcome.startswith(actor_result.outcome)   # everything else identical
        assert carried_result.gist == actor_result.gist == actor_result.outcome

    @pytest.mark.asyncio
    async def test_two_movers_cannot_carry_off_the_same_body(self, container) -> None:
        """Two people try to take the same person in one step. Whoever acts first gets him; the
        second can't reach him.

        No extra conflict ledger is needed: the first start() has already moved the body, so the
        second's same-place check fails right away. Initiative order decides the winner, which is
        deterministic and reproducible.
        """
        env = _make_environment(move_steps=2)
        env.place_agent(agent_id="agent-x", location_id="hall")
        taken = _make_agent(container, world_id="w", agent_id="agent-b", name="B")
        agents = {"agent-b": taken}
        executor = MovementExecutor(_directory())

        first = await executor.start(
            self._carry_action(["agent-b"], steps=2), 1,
            agents=agents, environment=env, message_system=None,
        )
        second = await executor.start(
            AgentAction(
                agent_id="agent-x", step=1, action_type=ActionType.MOVE,
                action_description="我也要带走他", estimated_steps=2,
                target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-b")]),
            ),
            1, agents=agents, environment=env, message_system=None,
        )

        assert first.participant_ids == ["agent-a", "agent-b"]
        assert second.participant_ids == ["agent-x"]


def test_rest_activity_blocks_the_code_layer_placeholder() -> None:
    """``_rest_activity`` only decides whether the description can go into the narrative.

    With an empty description ``start`` sets purpose to ``ActionType.REST.value``, a code-layer enum
    value; rendering "我打算「rest」" would leak it into the narrative. ``format_intent_clause`` handles
    trailing punctuation, so it isn't stripped twice here.
    """
    from engine.executors.simple import SimpleExecutor

    assert SimpleExecutor._rest_activity("闭目养神，平复心绪。") == "闭目养神，平复心绪。"
    assert SimpleExecutor._rest_activity("闭目养神") == "闭目养神"
    assert SimpleExecutor._rest_activity("") is None
    assert SimpleExecutor._rest_activity(ActionType.REST.value) is None


def test_talk_column_carries_the_owners_recent_past_in_order() -> None:
    """The recent-events column: its title says whose it is, it carries the shared ordering hint,
    and entries are in the order they happened (the same contract as other cognition steps)."""
    from agent.memory_types import Memory, MemoryStream
    from core.prompts import MEMORY_ORDER_HINT
    from engine.executors.social import _format_context

    def fact(mid: str, text: str, step: int) -> Memory:
        return Memory(id=mid, stream=MemoryStream.FACTUAL, agent_id="agent-b",
                      stored_content=text, importance=0.5, created_step=step, kind="event")

    text = _format_context(
        {"relation": None, "about_other": [], "about_topic": [],
         "recent": [fact("m2", "我在玄武门被拦下盘问。", 4), fact("m1", "我出了东宫。", 2)]},
        "长孙无忌", "李建成", now_step=5, seconds_per_step=3600,
    )
    assert f"长孙无忌近来的经历（{MEMORY_ORDER_HINT}）" in text
    assert text.index("我出了东宫") < text.index("被拦下盘问")


@pytest.mark.asyncio
async def test_talk_recent_past_comes_first_and_the_other_blocks_give_way() -> None:
    """Recent events are selected first and take priority: factual memories only, at most 5. The
    other two columns skip what it already took during retrieval (the topic column also skips "what
    I know of him"). Doing it the other way round would drop the most important part of the
    timeline."""
    from types import SimpleNamespace

    from agent.memory_types import Memory, MemoryStream
    from engine.executors.social import SocialExecutor, _TALK_RECENT_FACTUAL_K

    def mem(mid: str, stream: MemoryStream = MemoryStream.FACTUAL) -> Memory:
        return Memory(id=mid, stream=stream, agent_id="a", stored_content=mid,
                      importance=0.5, created_step=1, kind="event")

    pairs = [(mem("quarrel"), None), (None, mem("exp-only", MemoryStream.EXPERIENTIAL))] + [
        (mem(f"r{i}"), None) for i in range(8)
    ]
    excluded: dict[str, set] = {}

    async def _recall_about(*_a, exclude_ids=(), **_k):
        excluded["about_other"] = set(exclude_ids)
        return [mem("old-friendship")]

    async def _retrieve(*_a, exclude_ids=(), **_k):
        excluded["about_topic"] = set(exclude_ids)
        return []

    async def _perceive(**_k):
        return None

    agent = SimpleNamespace(
        agent_id="a",
        personality=SimpleNamespace(state=SimpleNamespace(emotion=None)),
        relation_system=SimpleNamespace(perceive=_perceive),
        memory_system=SimpleNamespace(
            recall_about_agent=_recall_about, retrieve=_retrieve,
            sample_recent_events=lambda step, top_k: pairs[:top_k],
        ),
    )
    ctx = await SocialExecutor(None, _directory())._gather_context(agent, "b", "乙", "议事", 5)  # type: ignore[arg-type]

    recent_ids = [m.id for m in ctx["recent"]]
    assert recent_ids[0] == "quarrel" and "exp-only" not in recent_ids
    assert len(recent_ids) == _TALK_RECENT_FACTUAL_K
    assert excluded["about_other"] == set(recent_ids)
    assert excluded["about_topic"] == set(recent_ids) | {"old-friendship"}


@pytest.mark.asyncio
async def test_talk_one_failed_recall_does_not_blank_the_other_blocks() -> None:
    """If "what I know of him" fails to load, only that column is dropped; the topic column and
    recent events stay."""
    from types import SimpleNamespace

    from agent.memory_types import Memory, MemoryStream
    from engine.executors.social import SocialExecutor

    fact = Memory(id="r0", stream=MemoryStream.FACTUAL, agent_id="a", stored_content="r0",
                  importance=0.5, created_step=1, kind="event")

    async def _boom(*_a, **_k):
        raise RuntimeError("store down")

    async def _retrieve(*_a, **_k):
        return [Memory(id="t0", stream=MemoryStream.FACTUAL, agent_id="a", stored_content="t0",
                       importance=0.5, created_step=1, kind="event")]

    async def _perceive(**_k):
        return None

    agent = SimpleNamespace(
        agent_id="a",
        personality=SimpleNamespace(state=SimpleNamespace(emotion=None)),
        relation_system=SimpleNamespace(perceive=_perceive),
        memory_system=SimpleNamespace(
            recall_about_agent=_boom, retrieve=_retrieve,
            sample_recent_events=lambda step, top_k: [(fact, None)],
        ),
    )
    ctx = await SocialExecutor(None, _directory())._gather_context(agent, "b", "乙", "议事", 5)  # type: ignore[arg-type]

    assert ctx["about_other"] == []
    assert [m.id for m in ctx["about_topic"]] == ["t0"]
    assert [m.id for m in ctx["recent"]] == ["r0"]


def test_talk_topic_block_never_carries_the_initiators_purpose() -> None:
    """The purpose of the talk is the initiator's private plan and must not appear in either side's
    memory column. Each side can only say what's in his own column, so putting it in the other's
    column amounts to telling him. Titles also must not claim these memories are about the matter at
    hand."""
    from agent.memory_types import Memory, MemoryStream
    from engine.executors.social import _format_context

    purpose = "我召见长孙无忌，当面盘问他凌晨突至东宫的来意，观其言辞虚实。"
    unrelated = Memory(
        id="m1", stream=MemoryStream.FACTUAL, agent_id="agent-b",
        stored_content="魏征来报：秦王今晨登崇仁坊他的门探其来意。",
        importance=0.7, created_step=3, kind="event",
    )
    # This renders the invitee's (Zhangsun Wuji's) column.
    text = _format_context(
        {"relation": None, "about_other": [], "about_topic": [unrelated]}, "长孙无忌", "李建成",
        now_step=5, seconds_per_step=3600,
    )
    assert purpose not in text
    assert "观其言辞虚实" not in text
    assert "其他事" in text and "魏征来报" in text
    # The generator is a third-party narrator. With a "我" in each column it can't tell them apart,
    # so the title names whose column it is.
    assert "长孙无忌记得的其他事" in text


@pytest.mark.asyncio
async def test_default_interrupt_gist_leaves_out_the_thought(container) -> None:
    """The base interrupt quotes the thought on outcome only; the gist is the bare event."""
    from engine.executors.errand import ErrandExecutor

    agent = _make_agent(container, world_id="w", agent_id="agent-a", name="A")
    state = ActionExecutionState.create(
        target=ActionTarget(), action_type=ActionType.ERRAND, initiator_id="agent-a",
        participant_ids=["agent-a"], purpose="差人送信", started_step=1,
        opening_outcome="开始", estimated_steps=3,
    )
    (result,) = await ErrandExecutor(_directory()).interrupt(
        state, 2, agents={"agent-a": agent}, environment=_make_environment(), thought="变了主意",
    )
    assert "变了主意" in result.outcome and "变了主意" not in result.gist
    assert "差人送信" in result.gist
