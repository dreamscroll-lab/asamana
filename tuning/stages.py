"""Tuning stage registry: the single source of truth for which stage suites exist.

Each stage suite is defined by four enumerable facts: key, display name, semantic criteria and
scenario file. Every runner has the same signature
``(container, config, world_id, *, scenarios_path, trace_dir, judge_*)``, so callers (the CLI,
the developer tools' stage routes) are driven from this table without per-stage if branches.

Adding a stage takes one line here plus `validation_<stage>.py` + `judge_<stage>.py` +
`run_<stage>` in `phase_harness/<stage>.py` + `scenarios/<stage>.json`. The CLI and developer
tools pick the stage up automatically; nothing else needs editing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from tuning import scenario_store

SCENARIO_DIR = Path(__file__).parent / "scenarios"


@dataclass(frozen=True)
class StageSpec:
    """One stage suite. Empty ``criteria`` = a purely deterministic suite (no LLM judge, e.g. perception)."""

    key: str
    label: str
    criteria: tuple[str, ...]
    runner: Callable[..., Awaitable[dict[str, Any]]]

    @property
    def scenarios_path(self) -> Path:
        return SCENARIO_DIR / f"{self.key}.json"

    @property
    def judged(self) -> bool:
        return bool(self.criteria)

    def as_dict(self) -> dict[str, Any]:
        """The form handed to the developer tools: immutable stage identity only, no runner."""
        return {
            "key": self.key,
            "label": self.label,
            "criteria": list(self.criteria),
            "judged": self.judged,
            "scenarios": scenario_names(self),
        }


def _specs() -> tuple[StageSpec, ...]:
    """Import the runner lazily: each validation module drags in half the world, and fetching one by key shouldn't pay for all of them."""
    from tuning.validation import validate_world_pressure
    from tuning.validation_action import validate_action
    from tuning.validation_decision import validate_decision
    from tuning.validation_event import validate_event
    from tuning.validation_feedback import validate_feedback
    from tuning.validation_interrupt import validate_interrupt
    from tuning.validation_long_term_goal import validate_long_term_goal
    from tuning.validation_memory_maintain import validate_memory_maintain
    from tuning.validation_memory_retrieve import validate_memory_retrieve
    from tuning.validation_memory_write import validate_memory_write
    from tuning.validation_need import validate_need
    from tuning.validation_perception import validate_perception
    from tuning.validation_perception_emotion import validate_perception_emotion
    from tuning.validation_relation import validate_relation

    return (
        # Order follows the cognition loop; CLI help and the developer tools list stages in this order.
        StageSpec("world_pressure", "World pressure", ("boundary", "reasonableness", "fields"), validate_world_pressure),
        StageSpec("perception", "Perception", (), validate_perception),
        StageSpec("perception_emotion", "Perception emotion", ("fields", "semantic", "functional"), validate_perception_emotion),
        StageSpec("need", "Motivation", ("fields", "ranking", "goals"), validate_need),
        StageSpec("decision", "Decision", ("fields", "action_fit", "coherence"), validate_decision),
        StageSpec("action", "Action", ("fields", "outcome_fidelity", "consequence_realism"), validate_action),
        StageSpec("feedback", "Feedback", ("fields", "appraisal_fidelity", "landing_consistency"), validate_feedback),
        StageSpec("memory_write", "Memory write", ("fields", "experiential", "importance"), validate_memory_write),
        StageSpec("memory_maintain", "Memory maintain", ("fidelity", "voice"), validate_memory_maintain),
        StageSpec("memory_retrieve", "Memory retrieve", ("relevance", "coverage"), validate_memory_retrieve),
        StageSpec("relation", "Relation", ("fidelity", "objectivity"), validate_relation),
        StageSpec("event", "Event", ("pacing", "groundedness", "concreteness", "craft", "discipline"), validate_event),
        StageSpec("interrupt", "Interrupt", ("decision_fit", "voice"), validate_interrupt),
        StageSpec("long_term_goal", "Long-term goal", ("revision_fit", "goal_quality"), validate_long_term_goal),
    )


_CACHE: tuple[StageSpec, ...] | None = None


def all_stages() -> tuple[StageSpec, ...]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _specs()
    return _CACHE


def stage(key: str) -> StageSpec | None:
    return next((s for s in all_stages() if s.key == key), None)


def stage_keys() -> list[str]:
    return [s.key for s in all_stages()]


# ---------------------------------------------------------------------------
# Scenario library — thin wrappers over the stdlib-only tuning.scenario_store
# ---------------------------------------------------------------------------
# The implementation lives there so the web process can edit scenarios without importing this
# registry (see that module's docstring). These are conveniences for the CLI, which holds a StageSpec.


def scenario_names(spec: StageSpec) -> list[str]:
    return scenario_store.names(spec.scenarios_path)


def write_scenario_subset(spec: StageSpec, picked: Sequence[str], dest: Path) -> None:
    scenario_store.write_subset(spec.scenarios_path, picked, dest, stage=spec.key)
