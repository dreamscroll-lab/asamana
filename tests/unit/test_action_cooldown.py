"""Action cooldown: firing, escalation, its hold on the AI decision, and the actor left free."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.action_cooldown import COOLDOWN_POLICIES, ActionCooldown, CooldownPolicy
from agent.agent import AgentStepPlan
from agent.decision import _ACTION_SPACE, DecisionEngine, DecisionResult
from core.interfaces.action import ActionTarget, ActionType, AgentAction
from tests.unit.test_agent import _make_agent, _make_need_evaluation, _make_spatial
from tests.unit.test_decision import _decision_prompt, _make_engine, _make_packet, _make_personality

TALK = ActionType.TALK
_POLICY = COOLDOWN_POLICIES[TALK]
COOLDOWN_THRESHOLD = _POLICY.threshold
COOLDOWN_WINDOW_DECISIONS = _POLICY.window_decisions
COOLDOWN_MAX_DECISIONS = _POLICY.max_decisions


def _start(cooldown: ActionCooldown, count: int, action_type: ActionType = TALK, *, gap: int = 1) -> None:
    """``count`` decisions ``gap`` apart, each starting ``action_type``."""
    for _ in range(count):
        _idle(cooldown, gap - 1)
        cooldown.note_decided()
        cooldown.record_start(action_type, step=0, agent_id="agent-1")


def _idle(cooldown: ActionCooldown, decisions: int) -> None:
    """Decisions that start nothing listed."""
    for _ in range(decisions):
        cooldown.note_decided()


def _decisions_held(cooldown: ActionCooldown) -> int:
    """Decides, starting nothing, until TALK is free; returns how many decisions were held."""
    held = 0
    while TALK in cooldown.held_back():
        held += 1
        cooldown.note_decided()
    return held


def test_starts_below_the_threshold_hold_nothing_back() -> None:
    cooldown = ActionCooldown()
    _start(cooldown, COOLDOWN_THRESHOLD - 1)
    assert cooldown.held_back() == frozenset()


def test_starts_spread_beyond_the_window_hold_nothing_back() -> None:
    cooldown = ActionCooldown()
    _start(cooldown, COOLDOWN_THRESHOLD + 2, gap=COOLDOWN_WINDOW_DECISIONS)
    assert cooldown.held_back() == frozenset()


def test_starts_count_in_decisions_however_long_each_act_runs() -> None:
    """Steps spent inside a multi-step act aren't decisions, so they don't stretch the window."""
    cooldown = ActionCooldown()
    for step in (1, 3, 5):
        cooldown.note_decided()
        cooldown.record_start(TALK, step=step, agent_id="agent-1")
    assert cooldown.held_back() == frozenset({TALK})


def test_unlisted_types_are_never_held_back() -> None:
    cooldown = ActionCooldown()
    _start(cooldown, COOLDOWN_THRESHOLD + 2, ActionType.WORK)
    assert cooldown.held_back() == frozenset()


def test_each_type_follows_its_own_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        COOLDOWN_POLICIES, ActionType.WORK, CooldownPolicy(threshold=2, window_decisions=2, max_decisions=1))
    cooldown = ActionCooldown()
    _start(cooldown, 2, ActionType.WORK)
    assert cooldown.held_back() == frozenset({ActionType.WORK})
    _start(cooldown, 2)
    assert TALK not in cooldown.held_back()


def test_reaching_the_threshold_holds_the_next_decision() -> None:
    cooldown = ActionCooldown()
    _start(cooldown, COOLDOWN_THRESHOLD)
    assert cooldown.held_back() == frozenset({TALK})
    assert _decisions_held(cooldown) == 1


def test_firing_soon_again_escalates_up_to_the_cap() -> None:
    cooldown = ActionCooldown()
    lengths = []
    for _ in range(COOLDOWN_MAX_DECISIONS + 2):
        _start(cooldown, COOLDOWN_THRESHOLD)
        lengths.append(_decisions_held(cooldown))
    assert lengths == [*range(1, COOLDOWN_MAX_DECISIONS + 1), COOLDOWN_MAX_DECISIONS, COOLDOWN_MAX_DECISIONS]


def test_a_quiet_spell_resets_the_escalation() -> None:
    cooldown = ActionCooldown()
    _start(cooldown, COOLDOWN_THRESHOLD)
    _decisions_held(cooldown)
    _start(cooldown, COOLDOWN_THRESHOLD)
    assert _decisions_held(cooldown) == 2
    _idle(cooldown, COOLDOWN_WINDOW_DECISIONS)
    _start(cooldown, COOLDOWN_THRESHOLD)
    assert _decisions_held(cooldown) == 1


def test_a_held_back_type_is_ruled_out_in_the_prompt(container: object) -> None:
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    free = _decision_prompt(engine, _make_personality(), packet, _ACTION_SPACE)
    held = _decision_prompt(engine, _make_personality(), packet, _ACTION_SPACE, held_back=frozenset({TALK}))
    assert "此刻不可选 TALK，换一种做法" not in free
    assert "此刻不可选 TALK，换一种做法" in held
    assert "【此刻够不着的行动】" in held


def test_a_held_back_type_chosen_anyway_is_rejected(container: object) -> None:
    engine = _make_engine(container)
    packet = _make_packet(visible_agent_ids=["agent-2"])
    talk = next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is TALK)
    content = json.dumps(
        {"selected_index": talk, "person_indices": [1], "talk_topic": "旧事",
         "action_description": "找他再谈。", "inner_monologue": "还得说。", "estimated_hours": 1},
        ensure_ascii=False,
    )
    assert engine._parse_llm_selection(content, _ACTION_SPACE, packet) is not None  # noqa: SLF001
    assert engine._parse_llm_selection(  # noqa: SLF001
        content, _ACTION_SPACE, packet, held_back=frozenset({TALK})) is None


def _talk_plan(step: int) -> AgentStepPlan:
    action = AgentAction(
        action_type=TALK, action_description="与他交谈。", agent_id="agent-1",
        target=ActionTarget(), step=step, estimated_steps=1, remaining_steps=1,
    )
    return AgentStepPlan(
        agent_id="agent-1", step=step, spatial=_make_spatial(), inbox=[], broadcasts=[],
        need_evaluation=_make_need_evaluation(), action=action,
    )


@pytest.mark.asyncio
async def test_the_ai_decision_is_held_back_after_repeated_talks(
    container: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting TALK THRESHOLD times holds it back from the next AI decision, and only that one."""
    agent = _make_agent(container)
    seen: list[frozenset[ActionType]] = []

    async def _decide(self: DecisionEngine, **kwargs: Any) -> DecisionResult:
        seen.append(kwargs["held_back"])
        return DecisionResult.no_action()

    async def _context(**_kwargs: Any) -> Any:
        return _make_packet().internal_context

    monkeypatch.setattr(DecisionEngine, "decide", _decide)
    monkeypatch.setattr(agent, "_build_internal_context", _context)
    for step in range(1, COOLDOWN_THRESHOLD + 1):
        await agent.begin_ongoing_step(plan=_talk_plan(step), executed=True)
    next_step = COOLDOWN_THRESHOLD + 1
    for step in (next_step, next_step + 1):
        await agent.plan_step(step=step, spatial=_make_spatial(), inbox=[], broadcasts=[])
    assert seen == [frozenset({TALK}), frozenset()]


@pytest.mark.asyncio
async def test_an_act_that_never_began_is_not_a_start(container: object) -> None:
    """A denied or refused TALK lands as a failure through the same begin path, but no talk took
    place: counting it would cool down the one who couldn't get a word in."""
    agent = _make_agent(container)
    for step in range(1, COOLDOWN_THRESHOLD):
        await agent.begin_ongoing_step(plan=_talk_plan(step), executed=False)
    await agent.begin_ongoing_step(plan=_talk_plan(COOLDOWN_THRESHOLD), executed=True)
    assert agent._cooldown.held_back() == frozenset()  # noqa: SLF001


def test_runtime_tells_a_begun_act_from_one_denied_or_refused(container: object) -> None:
    """``_executed`` reads the same marks as the decision verdict: a refusal, or a
    ``create_failed`` execution (arbitration veto, failed precondition), never began."""
    from engine.arbiter import ArbitratedAction
    from engine.executors.base import ActionExecutionState
    from core.interfaces.action import ActionResult
    from tests.unit.test_runtime_smoke import _build_runtime

    runtime, _env, _msg = _build_runtime(container, world_id="world-cooldown")
    registry = runtime._executor_registry  # noqa: SLF001
    action = AgentAction(action_type=TALK, action_description="与他交谈。", agent_id="agent-1")

    def verdict(state: ActionExecutionState, *, refused: bool = False) -> bool:
        registry.add_active(state)
        resolved = ArbitratedAction(
            agent_id="agent-1", action_result=ActionResult(action=action, expected_outcome="", outcome=""),
            location_id="palace", ongoing_execution_id=state.execution_id, refused=refused,
        )
        return runtime._executed(resolved)  # noqa: SLF001

    def failed(*, not_executed: bool) -> ActionExecutionState:
        return ActionExecutionState.create_failed(
            action_type=TALK, initiator_id="agent-1", started_step=1, not_executed=not_executed,
            failure_result=ActionResult(action=action, expected_outcome="", outcome="", succeeded=False),
        )

    begun = ActionExecutionState.create(
        target=ActionTarget(), action_type=TALK, initiator_id="agent-1",
        participant_ids=["agent-1", "agent-2"], purpose="x", started_step=1, estimated_steps=2,
        opening_outcome="",
    )
    assert verdict(begun) is True
    assert verdict(failed(not_executed=True)) is False
    assert verdict(failed(not_executed=False), refused=True) is False
