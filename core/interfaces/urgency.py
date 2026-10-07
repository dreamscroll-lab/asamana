"""Urgency — the unified urgency ladder.

Single source of truth for "urgency": the urgency signals on ``Message``, ``AgentAction`` and
``ExternalGoal`` all use this 4-level enum. LLM output picks one of 4 strings (a continuous float
is false precision, and LLM outputs cluster at the mid-high end); thresholds use ordered
comparison (``__ge__``).

Anchors, matching ``core.prompts.URGENCY_SCALE_DESCRIPTION``:
    LOW       ↔ 0.20  worth noting (background signal)
    NORMAL    ↔ 0.45  worth responding to (needs planning)   ← Message default
    HIGH      ↔ 0.75  should take priority (drop other things) ← interrupt threshold
    CRITICAL  ↔ 0.90  must respond now (life or death)
"""

from __future__ import annotations

from enum import Enum

from core.coerce import coerce_enum


class Urgency(str, Enum):
    """Unified urgency ladder, 4 levels, ordered (``msg.urgency >= Urgency.HIGH`` works)."""

    LOW      = "low"
    NORMAL   = "normal"
    HIGH     = "high"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        return _URGENCY_LEVEL[self]

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Urgency):
            return self.level >= other.level
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Urgency):
            return self.level > other.level
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Urgency):
            return self.level <= other.level
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Urgency):
            return self.level < other.level
        return NotImplemented


_URGENCY_LEVEL: dict[Urgency, int] = {
    Urgency.LOW:      1,
    Urgency.NORMAL:   2,
    Urgency.HIGH:     3,
    Urgency.CRITICAL: 4,
}


URGENCY_INTERRUPT_THRESHOLD: Urgency = Urgency.HIGH
"""Urgency at or above this interrupts an ongoing action.

It only decides whether to interrupt. Which need dominates is a separate question, answered by
MotivationBlender's continuous additive boost (external_pressure_boost), which doesn't use
this threshold."""


URGENCY_PREEMPT_THRESHOLD: Urgency = Urgency.CRITICAL
"""External pressure at or above this keeps an agent's own action from yielding to another
agent's TALK recruitment during arbitration.

It is one level stricter than ``URGENCY_INTERRUPT_THRESHOLD`` because someone else pays: an
interrupt costs the agent its own action, while a preemption wastes the recruiter's step.
Don't lower it to HIGH: a large share of recruitments would then miss and TALK would
routinely fail to land."""


def parse_urgency(raw: object, default: Urgency = Urgency.NORMAL) -> Urgency:
    """Lenient parse: any input → Urgency; unrecognized input returns ``default``.

    Every LLM output value goes through this.
    """
    return coerce_enum(Urgency, raw, default=default)
