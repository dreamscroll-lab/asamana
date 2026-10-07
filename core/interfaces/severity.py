"""Severity — the unified event-scale ladder.

Motivation
==========

``severity`` is a Broadcast's objective, producer-side event scale; ``PerceptionMemoryLayer``
maps it to the consumer-side ``Urgency``. Separate types: if a Broadcast carried Urgency, the
producer would decide the consumer's response pressure. Used like ``Urgency``: ``parse_severity``
at LLM boundaries, ``prompt_choices()`` in prompts, ordered comparison for thresholds.

Subclasses ``str`` → JSON/snapshots serialize directly as "high" etc.
"""

from __future__ import annotations

from enum import Enum

from core.coerce import coerce_enum
from core.interfaces.urgency import Urgency


class Severity(str, Enum):
    """Unified event-scale ladder, 3 levels (ordered: HIGH > MEDIUM > LOW)."""

    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

    @property
    def level(self) -> int:
        return _SEVERITY_LEVEL[self]

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.level >= other.level
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.level > other.level
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.level <= other.level
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.level < other.level
        return NotImplemented

    @classmethod
    def prompt_choices(cls) -> str:
        """Prompt-side choices, largest first ("high|medium|low"), generated from the enum."""
        return "|".join(s.value for s in sorted(cls, key=lambda s: s.level, reverse=True))


_SEVERITY_LEVEL: dict[Severity, int] = {
    Severity.LOW:    1,
    Severity.MEDIUM: 2,
    Severity.HIGH:   3,
}


# Translation membrane: Broadcast objective scale → perceived urgency (consumed by perception).
SEVERITY_TO_URGENCY: dict[Severity, Urgency] = {
    Severity.LOW:    Urgency.LOW,
    Severity.MEDIUM: Urgency.NORMAL,
    Severity.HIGH:   Urgency.HIGH,
}


def parse_severity(raw: object, default: Severity = Severity.LOW) -> Severity:
    """Lenient parse: any input → Severity; unrecognized input returns ``default``.

    Every LLM output value goes through this.
    """
    return coerce_enum(Severity, raw, default=default)


def severity_to_urgency(severity: Severity) -> Urgency:
    """Event scale → urgency. Unmapped levels fall back to LOW so a KeyError can't break perception."""
    return SEVERITY_TO_URGENCY.get(severity, Urgency.LOW)
