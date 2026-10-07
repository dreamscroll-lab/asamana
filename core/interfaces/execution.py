"""Execution contracts shared between the engine and agent cognition layer."""

from __future__ import annotations

from dataclasses import dataclass, field


from core.interfaces.action import Observed  # noqa: E402  (same contract layer, no import cycle)


@dataclass
class TickResult:
    """Intermediate output of one executor tick for one agent.

    A pure progress-narrative emitter for the observer/event stream: it never touches agent
    memory or state (that belongs to finalize_ongoing_action), and has no ``factual_memory``
    (progress is not an experience).

    ``outcome`` (god view) and ``observations`` (bystanders) are independent, both authored by
    the executor, as in ``ActionResult``. Consumers never fall back from ``observations`` to
    ``outcome``: that would leak privileged text. Empty by default is fail-safe; a public tick
    (WORK/MOVE/REST) repeats its outcome, a COVERT tick leaves it empty.
    """

    agent_id: str
    outcome: str
    observations: list["Observed"] = field(default_factory=list)
