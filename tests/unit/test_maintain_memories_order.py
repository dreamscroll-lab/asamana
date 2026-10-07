"""Ordering contract for the three memory phases in `CognitionMaintenance.run`.

Design contract (see the CognitionMaintenance.run docstring in engine/cognition_maintenance.py):
- Fixed order: decay -> compression -> reflection
- Only Compression deletes memories (Decay and Reflection don't)
- Reflection skips agents without a reflection_engine (here only the main-character stub has one)
- Failure tolerance (CLAUDE.md Rule 1): if Compression/Reflection fails for one agent, it logs a
  warning and doesn't raise

These tests record the call sequence through mocks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.cognition_maintenance import CognitionMaintenance


def _build_maintenance(*,
                       decay_interval: int = 1,
                       compression_interval: int = 1,
                       reflection_interval: int = 1,
                       label_evolution_interval: int = 0,
                       long_term_goal_revision_interval: int = 0,
                       compression_enabled: bool = True,
                       reflection_enabled: bool = True,
                       relation_evolution_enabled: bool = True,
                       long_term_goal_revision_enabled: bool = True) -> CognitionMaintenance:
    return CognitionMaintenance(
        memory_decay_interval=decay_interval,
        memory_compression_interval=compression_interval,
        reflection_interval=reflection_interval,
        label_evolution_interval=label_evolution_interval,
        long_term_goal_revision_interval=long_term_goal_revision_interval,
        compression_enabled=compression_enabled,
        reflection_enabled=reflection_enabled,
        relation_evolution_enabled=relation_evolution_enabled,
        long_term_goal_revision_enabled=long_term_goal_revision_enabled,
    )


def _stub_agent(*, agent_id: str, is_main: bool, call_log: list[tuple[str, str]], is_active: bool = True):
    """Stub agent that logs (agent_id, phase) for every maintenance call."""
    mem = AsyncMock()
    mem.apply_decay = AsyncMock(side_effect=lambda step: call_log.append((agent_id, "decay")))

    async def _compress(stream, current_step):
        call_log.append((agent_id, f"compress:{stream.value}"))

    mem.compress = AsyncMock(side_effect=_compress)

    reflection = None
    relation_evolution = None
    if is_main:
        reflection = AsyncMock()
        async def _reflect(step):
            call_log.append((agent_id, "reflect"))
        reflection.reflect = AsyncMock(side_effect=_reflect)

        relation_evolution = AsyncMock()
        async def _evolve(step):
            call_log.append((agent_id, "evolve"))
        relation_evolution.evaluate = AsyncMock(side_effect=_evolve)

    agent = SimpleNamespace(
        agent_id=agent_id,
        memory_system=mem,
        reflection_engine=reflection,
        relation_evolution=relation_evolution,
        is_main_character=is_main,
        is_active=is_active,
    )

    async def _revise(*, step):
        call_log.append((agent_id, "revise"))
    agent.revise_long_term_goals = AsyncMock(side_effect=_revise)
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Fixed order: decay -> compression -> reflection
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_maintain_memories_strict_phase_order() -> None:
    """The three phases must run strictly in decay -> compression -> reflection order."""
    maintenance = _build_maintenance()
    call_log: list[tuple[str, str]] = []
    main = _stub_agent(agent_id="a1", is_main=True, call_log=call_log)
    bg = _stub_agent(agent_id="a2", is_main=False, call_log=call_log)
    agents = [main, bg]

    await maintenance.run(agents, step=1)

    phases = []
    for _, phase in call_log:
        if phase == "decay":
            phases.append("decay")
        elif phase.startswith("compress:"):
            phases.append("compress")
        elif phase == "reflect":
            phases.append("reflect")

    first_decay = phases.index("decay") if "decay" in phases else -1
    first_compress = phases.index("compress") if "compress" in phases else -1
    first_reflect = phases.index("reflect") if "reflect" in phases else -1

    assert first_decay >= 0, f"decay 未触发: {phases}"
    assert first_compress >= 0, f"compress 未触发: {phases}"
    assert first_reflect >= 0, f"reflect 未触发: {phases}"

    assert first_decay < first_compress < first_reflect, (
        f"阶段顺序错乱: decay@{first_decay}, compress@{first_compress}, reflect@{first_reflect};"
        f"phases={phases}"
    )


@pytest.mark.asyncio
async def test_decay_runs_on_all_agents_main_and_background() -> None:
    """Decay runs for every agent, main and background alike."""
    maintenance = _build_maintenance()
    call_log: list[tuple[str, str]] = []
    main = _stub_agent(agent_id="main", is_main=True, call_log=call_log)
    bg1 = _stub_agent(agent_id="bg1", is_main=False, call_log=call_log)
    bg2 = _stub_agent(agent_id="bg2", is_main=False, call_log=call_log)

    await maintenance.run([main, bg1, bg2], step=1)

    decay_agents = {aid for aid, phase in call_log if phase == "decay"}
    assert decay_agents == {"main", "bg1", "bg2"}, (
        f"Decay 应对所有 agent 触发,实际只: {decay_agents}"
    )


@pytest.mark.asyncio
async def test_compression_runs_on_all_agents_main_and_background() -> None:
    """Compression runs for every agent (background agents need memory capacity managed too)."""
    maintenance = _build_maintenance()
    call_log: list[tuple[str, str]] = []
    main = _stub_agent(agent_id="main", is_main=True, call_log=call_log)
    bg = _stub_agent(agent_id="bg", is_main=False, call_log=call_log)

    await maintenance.run([main, bg], step=1)

    compress_agents = {aid for aid, phase in call_log if phase.startswith("compress:")}
    assert compress_agents == {"main", "bg"}, (
        f"Compression 应对所有 agent 触发: {compress_agents}"
    )


@pytest.mark.asyncio
async def test_reflection_runs_only_for_main_characters() -> None:
    """Reflection skips an agent whose reflection_engine is None (the background stub here)."""
    maintenance = _build_maintenance()
    call_log: list[tuple[str, str]] = []
    main = _stub_agent(agent_id="main", is_main=True, call_log=call_log)
    bg = _stub_agent(agent_id="bg", is_main=False, call_log=call_log)

    await maintenance.run([main, bg], step=1)

    reflect_agents = {aid for aid, phase in call_log if phase == "reflect"}
    assert reflect_agents == {"main"}, (
        f"Reflection 应只对主角触发,实际: {reflect_agents}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interval isolation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decay_skipped_when_interval_not_aligned() -> None:
    """With decay_interval=5, step=3 must not trigger decay."""
    maintenance = _build_maintenance(decay_interval=5, compression_interval=0, reflection_interval=0)
    call_log: list[tuple[str, str]] = []
    agent = _stub_agent(agent_id="a1", is_main=True, call_log=call_log)
    await maintenance.run([agent], step=3)  # 3 % 5 != 0
    assert all(phase != "decay" for _, phase in call_log)


@pytest.mark.asyncio
async def test_zero_interval_disables_phase() -> None:
    """interval=0 disables the phase entirely (config switch semantics)."""
    maintenance = _build_maintenance(decay_interval=0, compression_interval=0, reflection_interval=0)
    call_log: list[tuple[str, str]] = []
    agent = _stub_agent(agent_id="a1", is_main=True, call_log=call_log)
    await maintenance.run([agent], step=1)
    assert call_log == [], f"interval=0 时不应触发任何阶段: {call_log}"


# ─────────────────────────────────────────────────────────────────────────────
# Ablation switches: enabled=False skips the mechanism entirely. It is ANDed with the interval, so
# a mechanism can be turned off without changing its cadence.
# ─────────────────────────────────────────────────────────────────────────────


async def _phases_at_step1(**maintenance_kwargs) -> set[str]:
    """Run one maintenance step and return the phases that fired (main-character stub, every
    interval=1 so all are due)."""
    maintenance = _build_maintenance(**maintenance_kwargs)
    call_log: list[tuple[str, str]] = []
    agent = _stub_agent(agent_id="a1", is_main=True, call_log=call_log)
    await maintenance.run([agent], step=1)
    return {phase for _, phase in call_log}


@pytest.mark.asyncio
async def test_compression_switch_gates_phase() -> None:
    """compression_enabled=False skips compression; True runs it on the same cadence, so the switch
    is what gates it, not the interval."""
    on = await _phases_at_step1(compression_interval=1, compression_enabled=True)
    off = await _phases_at_step1(compression_interval=1, compression_enabled=False)
    assert any(p.startswith("compress") for p in on)
    assert not any(p.startswith("compress") for p in off)


@pytest.mark.asyncio
async def test_reflection_switch_gates_phase() -> None:
    on = await _phases_at_step1(reflection_interval=1, reflection_enabled=True)
    off = await _phases_at_step1(reflection_interval=1, reflection_enabled=False)
    assert "reflect" in on and "reflect" not in off


@pytest.mark.asyncio
async def test_relation_evolution_switch_gates_phase() -> None:
    on = await _phases_at_step1(label_evolution_interval=1, relation_evolution_enabled=True)
    off = await _phases_at_step1(label_evolution_interval=1, relation_evolution_enabled=False)
    assert "evolve" in on and "evolve" not in off


@pytest.mark.asyncio
async def test_long_term_goal_revision_switch_gates_phase() -> None:
    on = await _phases_at_step1(long_term_goal_revision_interval=1, long_term_goal_revision_enabled=True)
    off = await _phases_at_step1(long_term_goal_revision_interval=1, long_term_goal_revision_enabled=False)
    assert "revise" in on and "revise" not in off


# ─────────────────────────────────────────────────────────────────────────────
# Failure tolerance (CLAUDE.md Rule 1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compression_failure_does_not_propagate() -> None:
    """A compression failure for one agent logs a warning, doesn't raise, and doesn't stop other
    phases."""
    maintenance = _build_maintenance()
    call_log: list[tuple[str, str]] = []
    failing = _stub_agent(agent_id="failer", is_main=True, call_log=call_log)
    failing.memory_system.compress = AsyncMock(side_effect=RuntimeError("simulated"))

    await maintenance.run([failing], step=1)
    decay_count = sum(1 for _, p in call_log if p == "decay")
    reflect_count = sum(1 for _, p in call_log if p == "reflect")
    assert decay_count == 1, "Decay 应仍执行"
    assert reflect_count == 1, "Reflection 应仍执行"


@pytest.mark.asyncio
async def test_dead_agents_skip_all_maintenance() -> None:
    """No cognition after death: an agent with is_active=False must trigger no maintenance phase.

    A main character who dies on a reflection/compression interval step (after
    DeathHandler.process_new_deaths sets is_active=False) must be skipped, or posthumous insights
    and memories get written."""
    # Every interval=1 so all five phases (decay/compress/reflect/evolve/revise) are due. That shows
    # a dead agent skips evolve/revise too, rather than those being off by default.
    maintenance = _build_maintenance(label_evolution_interval=1, long_term_goal_revision_interval=1)
    call_log: list[tuple[str, str]] = []
    alive = _stub_agent(agent_id="alive", is_main=True, call_log=call_log)
    dead = _stub_agent(agent_id="dead", is_main=True, call_log=call_log, is_active=False)

    await maintenance.run([alive, dead], step=1)

    touched = {aid for aid, _ in call_log}
    assert "dead" not in touched, (
        f"死亡 agent 不应触发任何维护阶段(decay/compress/reflect/evolve/revise),实际: "
        f"{[e for e in call_log if e[0] == 'dead']}"
    )
    # the living main character still runs every phase (positive control)
    alive_phases = {p for aid, p in call_log if aid == "alive"}
    assert "decay" in alive_phases and "reflect" in alive_phases and "evolve" in alive_phases and "revise" in alive_phases
