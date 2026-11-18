"""An errand report reaching a character who isn't deciding that step is shown at his next
decision, once, and survives a snapshot until then. Memory keeps only what the runner said, so
without this the scene he brought back would never reach a decision."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.agent import snapshot_agent_state
from agent.decision import DecisionEngine, DecisionResult
from agent.state_codec import agent_state_from_record
from core.interfaces.message import ERRAND_REPORT_SOURCE, Message
from core.serialization import dump_json
from tests.unit.test_agent import _make_agent, _make_spatial
from tests.unit.test_decision import _make_packet


def _report(msg_id: str = "rep-1", *, step: int = 3) -> Message:
    return Message(
        id=msg_id, world_id="world-1", sender_id="npc-1", sender_name="宫人",
        content="任务报告：我去了城门执行任务，回来了。\n 到达时所见：城门下只有一名卫士。",
        recipients=["agent-1"], location_scope=None, deliver_step=step, created_step=step - 1,
        sender_is_agent=False,
        metadata={"source": ERRAND_REPORT_SOURCE, "spoken": "任务报告：我去了城门执行任务，回来了。"},
    )


def _letter(msg_id: str = "letter-1", *, step: int = 3) -> Message:
    return Message(
        id=msg_id, world_id="world-1", sender_id="agent-2", sender_name="乙",
        content="今夜子时见。", recipients=["agent-1"], location_scope=None,
        deliver_step=step, created_step=step - 1,
    )


def _capture_decisions(agent: Any, monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> list[list[str]]:
    """Each decision's inbox, as message ids."""
    seen: list[list[str]] = []

    async def _decide(self: DecisionEngine, **kwargs: Any) -> DecisionResult:
        seen.append([m.id for m in kwargs["packet"].inbox])
        return DecisionResult.failed() if fail else DecisionResult.no_action()

    async def _context(**_kwargs: Any) -> Any:
        return _make_packet().internal_context

    monkeypatch.setattr(DecisionEngine, "decide", _decide)
    monkeypatch.setattr(agent, "_build_internal_context", _context)
    return seen


async def _perceive(agent: Any, inbox: list[Message], step: int) -> None:
    await agent.perceive_step(spatial=_make_spatial(), inbox=inbox, broadcasts=[], step=step)


async def _plan(agent: Any, inbox: list[Message], step: int) -> Any:
    return await agent.plan_step(step=step, spatial=_make_spatial(), inbox=inbox, broadcasts=[])


@pytest.mark.asyncio
async def test_a_report_reaching_a_busy_character_is_shown_at_his_next_decision_once(
    container: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(container)
    seen = _capture_decisions(agent, monkeypatch)

    await _perceive(agent, [_report(step=3), _letter(step=3)], step=3)  # busy: no decision
    letter = _letter("letter-2", step=5)
    await _perceive(agent, [letter], step=5)
    plan = await _plan(agent, [letter], step=5)
    await _perceive(agent, [], step=6)
    await _plan(agent, [], step=6)

    # Only the report is carried: a letter's whole text is already in his memory.
    assert seen == [["rep-1", "letter-2"], []]
    # The plan keeps this step's real inbox: arbitration records it as the step's messages.
    assert [m.id for m in plan.inbox] == ["letter-2"]


@pytest.mark.asyncio
async def test_a_report_arriving_on_a_decision_step_is_shown_once(
    container: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(container)
    seen = _capture_decisions(agent, monkeypatch)
    report = _report(step=3)

    await _perceive(agent, [report], step=3)
    await _plan(agent, [report], step=3)
    await _perceive(agent, [], step=4)
    await _plan(agent, [], step=4)

    assert seen == [["rep-1"], []]


@pytest.mark.asyncio
async def test_a_failed_decision_keeps_the_report_for_the_next_one(
    container: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _make_agent(container)
    seen = _capture_decisions(agent, monkeypatch, fail=True)

    await _perceive(agent, [_report(step=3)], step=3)
    await _plan(agent, [], step=4)

    assert seen == [["rep-1"]]
    assert [m.id for m in agent.pending_errand_reports] == ["rep-1"]


@pytest.mark.asyncio
async def test_a_performed_character_drops_carried_reports_at_its_decision(
    container: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actor sees every message in its thread as it arrives, so its decision leaves nothing
    carried for a later AI one."""

    class _Actor:
        def perceived(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    agent = _make_agent(container)
    agent.actor = _Actor()  # type: ignore[assignment]

    async def _actor_plan(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(agent, "_plan_with_actor", _actor_plan)
    await _perceive(agent, [_report(step=3)], step=3)
    await _plan(agent, [], step=4)
    assert agent.pending_errand_reports == []


@pytest.mark.asyncio
async def test_an_unseen_report_survives_a_snapshot(container: object) -> None:
    agent = _make_agent(container)
    await _perceive(agent, [_report(step=3)], step=3)

    record = json.loads(dump_json(snapshot_agent_state(agent)))
    fresh = _make_agent(container)
    fresh.apply_stored_state(agent_state_from_record(record), in_resumed_execution=False)

    [restored] = fresh.pending_errand_reports
    assert restored == _report(step=3)
