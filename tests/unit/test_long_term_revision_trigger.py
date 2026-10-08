"""Trigger contract for long-term goal revision, independent of reflection.

- A separate maintenance sub-step with its own period, `long_term_goal_revision_interval`, not
  tied to reflection.
- Fires for everyone: background agents revise long-term goals too; the period is the only gate
  (no cognitive tiering gate).
- Skipping when there are no recent memories happens inside Agent (see test_agent.py); this file
  only covers the maintenance scheduler's period gate.

A stub agent (revise_long_term_goals as an AsyncMock spy) pins the gate; reflection is off to
isolate the revision sub-step.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.cognition_maintenance import CognitionMaintenance


def _build_maintenance(*, long_term_goal_revision_interval: int = 30) -> CognitionMaintenance:
    return CognitionMaintenance(
        memory_decay_interval=0,
        memory_compression_interval=0,
        reflection_interval=0,  # reflection off: shows revision runs independently
        label_evolution_interval=0,
        long_term_goal_revision_interval=long_term_goal_revision_interval,
    )


def _stub_agent(*, is_main: bool = True, agent_id: str = "a1"):
    agent = SimpleNamespace(
        agent_id=agent_id,
        memory_system=AsyncMock(),
        reflection_engine=None,
        relation_evolution=None,
        is_main_character=is_main,
        is_active=True,
    )
    agent.revise_long_term_goals = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_revision_fires_on_own_interval() -> None:
    """On-period (30%30==0): revision fires, even with reflection off."""
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    agent = _stub_agent()
    await maintenance.run([agent], step=30)
    agent.revise_long_term_goals.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_skipped_off_interval() -> None:
    """Off-period (20%30!=0): no revision. The period is the only gate."""
    maintenance = _build_maintenance(long_term_goal_revision_interval=30)
    agent = _stub_agent()
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
