"""Severity enum contract — the event-scale counterpart of Urgency.

severity is the core.interfaces.severity.Severity enum, not loose strings (Literal + a validation
set + hand-written prompt choices + == "high" comparisons). This file locks: a single source of
truth for valid values, the LLM boundary always going through parse_severity, prompt choices
generated from the enum, a complete severity→urgency bridge, and ordered comparison.
"""

from __future__ import annotations

from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.severity import (
    SEVERITY_TO_URGENCY,
    Severity,
    parse_severity,
    severity_to_urgency,
)
from core.interfaces.urgency import Urgency


def test_severity_values_and_order() -> None:
    assert {s.value for s in Severity} == {"low", "medium", "high"}
    assert Severity.LOW < Severity.MEDIUM < Severity.HIGH
    assert Severity.HIGH >= Severity.MEDIUM
    assert Severity.LOW != Severity.HIGH


def test_severity_is_str_enum_serializes_as_value() -> None:
    # Subclasses str → zero migration for JSON/snapshots; .value is the canonical string.
    assert Severity.HIGH == "high"
    assert Severity.HIGH.value == "high"


def test_prompt_choices_generated_from_enum_descending() -> None:
    # Prompt choices are generated from the enum, so they stay in sync with the valid values.
    assert Severity.prompt_choices() == "high|medium|low"


def test_severity_to_urgency_bridge_covers_all_three_levels() -> None:
    assert SEVERITY_TO_URGENCY[Severity.HIGH] == Urgency.HIGH
    assert SEVERITY_TO_URGENCY[Severity.MEDIUM] == Urgency.NORMAL
    assert SEVERITY_TO_URGENCY[Severity.LOW] == Urgency.LOW
    assert len(SEVERITY_TO_URGENCY) == 3
    for s in Severity:
        assert severity_to_urgency(s) == SEVERITY_TO_URGENCY[s]


def test_parse_severity_accepts_enum_and_strings() -> None:
    assert parse_severity(Severity.HIGH) is Severity.HIGH
    assert parse_severity("high") == Severity.HIGH
    assert parse_severity("HIGH") == Severity.HIGH
    assert parse_severity("  medium ") == Severity.MEDIUM


def test_parse_severity_invalid_falls_back_to_low() -> None:
    assert parse_severity("nonsense") == Severity.LOW
    assert parse_severity("") == Severity.LOW
    assert parse_severity(None) == Severity.LOW
    assert parse_severity(0.75) == Severity.LOW
    assert parse_severity("nonsense", default=Severity.MEDIUM) == Severity.MEDIUM


def test_broadcast_defaults_to_low_severity() -> None:
    bc = Broadcast(content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1)
    assert bc.severity == Severity.LOW
