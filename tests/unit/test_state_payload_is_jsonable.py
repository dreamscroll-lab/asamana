"""Nothing that leaves the sim as *state* or as an *action record* may be a live Python object.

Enums such as ``EmotionType`` / ``ActionType`` are ``str`` subclasses, so on the disk path
``json.dumps`` writes "frustration" / "talk". The live path hands the same dict to the read model,
whose ``str(...)`` gives "EmotionType.FRUSTRATION" / "ActionType.TALK": the observer shows a
different label live than after a refresh, and every renderer keyed on the action type (map
effects, icons, feed card style) falls through to a generic chip live while replay is correct.

Two guards, for any dict that leaves the sim:
  1. No enum may appear anywhere in a serialized payload (the boundary invariant).
  2. The read model, fed an in-memory payload (never round-tripped through JSON), must produce
     clean labels.

Don't add a JSON round-trip to these tests: it would hide the defect the same way the disk path
does.
"""

from __future__ import annotations

import enum
from typing import Any

from agent.agent import snapshot_agent_state
from agent.need import NeedType
from agent.personality import AgentActivityStatus, EmotionType
from core.interfaces.action import ActionResult, ActionTarget, ActionType, AgentAction, Observed, Ref
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.directory import LiveWorldDirectory
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.base import ActionExecutionState
from interaction.models import _action_from_record, _agent_states_from_snapshot
from core.interfaces.snapshot import WorldSnapshot


def _enum_leaks(obj: Any, path: str = "payload") -> list[str]:
    """Every live Enum reachable in a payload, with the path that reaches it."""
    if isinstance(obj, enum.Enum):
        return [f"{path} = {obj!r}"]
    if isinstance(obj, dict):
        return [leak for k, v in obj.items() for leak in _enum_leaks(v, f"{path}.{k}")]
    if isinstance(obj, (list, tuple)):
        return [leak for i, v in enumerate(obj) for leak in _enum_leaks(v, f"{path}[{i}]")]
    return []


def test_the_detector_can_actually_see_an_enum() -> None:
    # A guard that cannot fail is not a guard. Prove it catches the exact shape of the defect.
    assert _enum_leaks({"emotion": {"primary": EmotionType.FRUSTRATION}}) == [
        "payload.emotion.primary = <EmotionType.FRUSTRATION: 'frustration'>"
    ]
    assert _enum_leaks({"emotion": {"primary": "frustration"}}) == []


def test_snapshot_agent_state_carries_no_live_enums(container) -> None:
    from tests.unit.test_action_executors import _make_agent

    agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
    agent.personality.update_emotion(
        primary=EmotionType.FRUSTRATION, intensity=0.8, valence=-0.6
    )

    leaks = _enum_leaks(snapshot_agent_state(agent))
    assert not leaks, f"serialized agent state must be JSON-native, found: {leaks}"


def _talk_completion_record() -> dict[str, Any]:
    """One completed TALK, as the runtime hands it to BOTH the snapshot and the live push."""
    env = EnvironmentSystem()
    processor = ExecutionProcessor(
        # completion_record only reads the directory + environment; an empty registry
        # keeps this a pure serialization assertion with no LLM anywhere near it.
        executor_registry=ActionExecutorRegistry(),
        environment=env,
        message_system=None,  # type: ignore[arg-type]  # completion_record touches neither
        directory=LiveWorldDirectory.from_agents({}, env),
    )
    action = AgentAction(
        agent_id="agent-a",
        step=1,
        action_type=ActionType.TALK,
        action_description="我走向他，试探口风。",
        target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]),
        estimated_steps=1,
    )
    state = ActionExecutionState.create(
        action_type=ActionType.TALK,
        initiator_id="agent-a",
        participant_ids=["agent-a", "agent-b"],
        purpose="试探口风",
        started_step=1,
        estimated_steps=1,
        opening_outcome="",
        target=action.target,
    )
    result = ActionResult(
        action=action,
        expected_outcome="他松口",
        outcome="在门下，甲与乙的交谈。",
        observations=[Observed(location_id="gate", text="在门下，甲与乙在交谈。")],
    )
    return processor.completion_record(state, result, {})


def test_completion_record_carries_no_live_enums() -> None:
    leaks = _enum_leaks(_talk_completion_record(), path="record")
    assert not leaks, f"an action record must be JSON-native, found: {leaks}"


def test_live_path_action_type_is_the_value_not_the_enum_name() -> None:
    """The read model, fed the record straight from memory — as the live WebSocket push does.

    NO json round-trip (see the module docstring): the enum NAME matches none of the renderers'
    "talk"/"move"/"physical"… comparisons.
    """
    summary = _action_from_record(_talk_completion_record())

    assert summary.action_type == "talk"
    assert "ActionType" not in summary.action_type


def test_live_path_emotion_label_is_the_value_not_the_enum_name(container) -> None:
    """The read model, fed the payload straight from memory — as the live WebSocket push does.

    NO json round-trip: that is the whole point (see the module docstring).
    """
    from tests.unit.test_action_executors import _make_agent

    agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
    agent.personality.update_emotion(
        primary=EmotionType.FRUSTRATION, intensity=0.8, valence=-0.6
    )
    snapshot = WorldSnapshot(
        world_id="w",
        step=1,
        timestamp="2026-07-13 00:00:00",
        agent_states={"agent-a": snapshot_agent_state(agent)},
    )

    states = _agent_states_from_snapshot(snapshot, actions=[])

    assert states["agent-a"].emotion == "frustration"
    assert "EmotionType" not in states["agent-a"].emotion
    # The narrative-layer names come from the backend's own tables, not the client's.
    assert states["agent-a"].emotion_label == EmotionType.FRUSTRATION.label
    assert states["agent-a"].activity_label == AgentActivityStatus.IDLE.label


def test_every_emotion_activity_and_need_has_a_label() -> None:
    """A member without a label would raise in the read model's label lookup."""
    assert all(e.label for e in EmotionType)
    assert all(a.label for a in AgentActivityStatus)
    assert all(n.label for n in NeedType)


def test_an_unknown_value_reads_as_no_label() -> None:
    snapshot = WorldSnapshot(
        world_id="w", step=1, timestamp="2026-07-13 00:00:00",
        agent_states={"agent-a": {
            "agent_id": "agent-a", "emotion": "elation", "activity_status": "", "dominant_need": "fame",
        }},
    )

    state = _agent_states_from_snapshot(snapshot, actions=[])["agent-a"]

    assert (state.emotion_label, state.activity_label, state.dominant_need_label) == ("", "", "")


def test_the_dominant_need_reads_with_its_label() -> None:
    snapshot = WorldSnapshot(
        world_id="w", step=1, timestamp="2026-07-13 00:00:00",
        agent_states={"agent-a": {"agent_id": "agent-a", "dominant_need": "safety"}},
    )

    state = _agent_states_from_snapshot(snapshot, actions=[])["agent-a"]

    assert state.dominant_need_label == NeedType.SAFETY.label
