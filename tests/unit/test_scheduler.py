"""AgentScheduler ordering: tier batching + seeded intra-batch shuffle."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.personality import StateLayer
from engine.scheduler import AgentScheduler


class _StubPersonality:
    def __init__(self, state: StateLayer) -> None:
        self._state = state

    @property
    def state(self) -> StateLayer:
        return self._state


@dataclass
class _StubAgent:
    """An agent the cadence gate always admits.

    ``StateLayer.last_decision_step`` defaults to 0 = "never decided", which the gate reads as
    starved. These tests lock ordering only, so a gate regression can't pass for an ordering one;
    the gate has its own suite, tests/unit/test_decision_cadence.py.
    """

    agent_id: str
    is_main_character: bool = False
    is_active: bool = True
    pending_external_goals: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.personality = _StubPersonality(StateLayer(agent_id=self.agent_id))


def _ids(plan, phase: str) -> list[str]:
    for batch in plan.batches:
        if batch.phase == phase:
            return batch.agent_ids()
    return []


def test_unseeded_keeps_deterministic_agent_id_sort() -> None:
    """No seed → stable agent_id sort (unseeded callers / tests stay deterministic)."""
    agents = [_StubAgent("c"), _StubAgent("a"), _StubAgent("b")]
    scheduler = AgentScheduler()
    plan = scheduler.plan(agents, step=1, in_progress_at_step_start=set())
    assert _ids(plan, "background") == ["a", "b", "c"]


def test_seeded_shuffle_is_reproducible_for_same_seed_and_step() -> None:
    """Same (seed, step) → identical order, regardless of input arrival order."""
    seed = "world-xyz"
    a1 = [_StubAgent(x) for x in ("a", "b", "c", "d", "e")]
    a2 = [_StubAgent(x) for x in ("e", "d", "c", "b", "a")]  # different arrival order
    p1 = AgentScheduler(seed=seed).plan(a1, step=7, in_progress_at_step_start=set())
    p2 = AgentScheduler(seed=seed).plan(a2, step=7, in_progress_at_step_start=set())
    assert _ids(p1, "background") == _ids(p2, "background")


def test_seeded_shuffle_varies_across_steps() -> None:
    """Order changes step to step → no fixed initiative winner."""
    agents = [_StubAgent(x) for x in ("a", "b", "c", "d", "e", "f")]
    scheduler = AgentScheduler(seed="world-xyz")
    orders = {
        tuple(_ids(scheduler.plan(agents, step=s, in_progress_at_step_start=set()), "background"))
        for s in range(1, 12)
    }
    # Across 11 steps the shuffle yields more than one distinct ordering.
    assert len(orders) > 1


def test_shuffle_never_crosses_tier_boundary() -> None:
    """Main always precedes background; shuffle is strictly intra-batch."""
    agents = [
        _StubAgent("bg1"),
        _StubAgent("hero_a", is_main_character=True),
        _StubAgent("bg2"),
        _StubAgent("hero_b", is_main_character=True),
    ]
    scheduler = AgentScheduler(seed="world-xyz")
    for s in range(1, 8):
        plan = scheduler.plan(agents, step=s, in_progress_at_step_start=set())
        main = _ids(plan, "main")
        background = _ids(plan, "background")
        assert sorted(main) == ["hero_a", "hero_b"]
        assert sorted(background) == ["bg1", "bg2"]
        # ordered_agents concatenates batches: every main precedes every background.
        ordered = [a.agent_id for a in plan.ordered_agents()]
        assert ordered.index("hero_a") < ordered.index("bg1")
        assert ordered.index("hero_b") < ordered.index("bg1")
