"""Trigger contract for long-term goal revision, independent of reflection.

- A separate maintenance sub-step, not tied to reflection. Each agent is due on its own:
  `long_term_goal_revision_interval` steps after its last revision, or once
  FAILED_GOALS_PER_REVISION short-term goals have failed since then (Agent.long_term_revision_due).
- Both conditions count from the same last revision, so a failure-driven revision also restarts
  the period and the two never run back to back.
- It fires for everyone who is due: background agents revise long-term goals too (no cognitive
  tiering gate).
- Skipping when there are no recent memories happens inside Agent (see test_agent.py); this file
  only covers when maintenance calls it.

A stub agent carries the real ``Agent.long_term_revision_due`` over a bare state, with
revise_long_term_goals as an AsyncMock spy; reflection is off to isolate the revision sub-step.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.agent import Agent
from agent.goals import FAILED_GOALS_PER_REVISION
from engine.cognition_maintenance import CognitionMaintenance


def _build_maintenance(
    *, long_term_goal_revision_interval: int = 30, enabled: bool = True,
) -> CognitionMaintenance:
    return CognitionMaintenance(
        memory_decay_interval=0,
        memory_compression_interval=0,
        reflection_interval=0,  # reflection off: shows revision runs independently
        label_evolution_interval=0,
        long_term_goal_revision_interval=long_term_goal_revision_interval,
        long_term_goal_revision_enabled=enabled,
    )


def _stub_agent(
    *, is_main: bool = True, agent_id: str = "a1", revised_step: int = 0, failed: int = 0,
):
    agent = SimpleNamespace(
        agent_id=agent_id,
        memory_system=AsyncMock(),
        reflection_engine=None,
        relation_evolution=None,
        is_main_character=is_main,
        is_active=True,
        personality=SimpleNamespace(state=SimpleNamespace(
            long_term_revised_step=revised_step, failed_goals_since_revision=failed,
        )),
    )
    agent.revise_long_term_goals = AsyncMock()
    agent.long_term_revision_due = MethodType(Agent.long_term_revision_due, agent)
    return agent


@pytest.mark.asyncio
async def test_revision_fires_on_own_interval() -> None:
    """A full period after the last revision (never revised = step 0): revision fires, even with
    reflection off."""
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    agent = _stub_agent()
    await maintenance.run([agent], step=30)
    agent.revise_long_term_goals.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_skipped_before_period_without_enough_failures() -> None:
    """Short of the period and of the failure count: no revision."""
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    agent = _stub_agent(failed=FAILED_GOALS_PER_REVISION - 1)
    await maintenance.run([agent], step=20)
    agent.revise_long_term_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_revision_fires_for_background_agent_too() -> None:
    """Background agents revise long-term goals too. `is_main_character` sets narrative tier, not
    which cognition runs.

    If background agents silently dropped out of the revision set, their long-term goals would stay
    at their initial values with no error.
    """
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    main = _stub_agent(is_main=True, agent_id="main")
    bg = _stub_agent(is_main=False, agent_id="bg")
    await maintenance.run([main, bg], step=30)
    main.revise_long_term_goals.assert_awaited_once()
    bg.revise_long_term_goals.assert_awaited_once()


@pytest.mark.asyncio
async def test_enough_failed_goals_bring_revision_forward() -> None:
    """Before the period, only the agent with enough failed short-term goals revises: the long-term
    goal behind a blocked pursuit must not keep spawning it until the period comes round."""
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    failing = _stub_agent(agent_id="failing", failed=FAILED_GOALS_PER_REVISION)
    other = _stub_agent(agent_id="other")
    await maintenance.run([failing, other], step=20)
    failing.revise_long_term_goals.assert_awaited_once()
    other.revise_long_term_goals.assert_not_awaited()


@pytest.mark.asyncio
async def test_period_counts_from_a_failure_driven_revision() -> None:
    """A revision brought forward at step 28 restarts the period: no second revision at step 30."""
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    agent = _stub_agent(revised_step=28)
    await maintenance.run([agent], step=30)
    agent.revise_long_term_goals.assert_not_awaited()
    await maintenance.run([agent], step=58)
    agent.revise_long_term_goals.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_goals_do_not_revise_when_revision_disabled() -> None:
    """The ablation switch turns the failure trigger off with the period."""
    maintenance = _build_maintenance(enabled=False)
    agent = _stub_agent(failed=FAILED_GOALS_PER_REVISION)
    await maintenance.run([agent], step=20)
    agent.revise_long_term_goals.assert_not_awaited()


def test_revision_due_state_survives_the_snapshot() -> None:
    """Both halves of the due time are committed: lost on restore, the count would restart and the
    period would fall due for everyone at once."""
    from agent.personality import PersonalityLayer, SoulLayer
    from agent.state_codec import decode_agent_state, encode_agent_state

    personality = PersonalityLayer(soul=SoulLayer(name="李世民", role="prince", agent_id="a"))
    personality.mark_long_term_revised(17)
    personality.count_failed_goals(3)

    stored = encode_agent_state("w", "a", personality.state)
    restored = decode_agent_state("a", stored, orphaned=False)

    assert restored.long_term_revised_step == 17
    assert restored.failed_goals_since_revision == 3
