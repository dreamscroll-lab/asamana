"""Unit tests for the shared coercion primitives.

Pins the canonical behavior of each family, including two edge cases:
optional-int bool rejection and optional-str whitespace → None.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from core.coerce import (
    coerce_datetime,
    coerce_dict,
    coerce_enum,
    coerce_float,
    coerce_int,
    coerce_list,
    coerce_mapping_list,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_str,
    coerce_str,
    coerce_str_list,
)


# --- total numeric ---------------------------------------------------------

def test_coerce_int_default_and_clamp() -> None:
    assert coerce_int(7, default=1) == 7
    assert coerce_int("12", default=1) == 12
    assert coerce_int(None, default=5) == 5
    assert coerce_int(0, default=1, minimum=1, maximum=12) == 1
    assert coerce_int(99, default=1, minimum=1, maximum=12) == 12


def test_coerce_float_default_and_clamp() -> None:
    assert coerce_float("0.5", default=0.0) == 0.5
    assert coerce_float("x", default=0.3) == 0.3
    assert coerce_float(9.0, default=0.0, minimum=0.0, maximum=1.0) == 1.0


# --- total string / container ----------------------------------------------

def test_coerce_str_strips_and_falls_back() -> None:
    assert coerce_str("  hi ") == "hi"
    assert coerce_str(None, "x") == "x"
    assert coerce_str("   ", "x") == "x"
    assert coerce_str(5) == "5"


def test_coerce_str_list() -> None:
    assert coerce_str_list(["a", " b ", "", None]) == ["a", "b"]
    assert coerce_str_list("solo") == ["solo"]
    assert coerce_str_list(None) == []


def test_coerce_mapping_list() -> None:
    assert coerce_mapping_list([{"a": 1}, "skip", 3, {"b": 2}]) == [{"a": 1}, {"b": 2}]
    assert coerce_mapping_list("not-a-seq") == []


def test_coerce_dict_and_list() -> None:
    assert coerce_dict({"a": 1}) == {"a": 1}
    assert coerce_dict("x") == {}
    assert coerce_list([1, 2]) == [1, 2]
    assert coerce_list((1, 2)) == [1, 2]
    assert coerce_list("x") == []


# --- optional --------------------------------------------------------------

def test_coerce_optional_int_rejects_bool_but_parses_numbers() -> None:
    assert coerce_optional_int(3) == 3
    assert coerce_optional_int("3") == 3
    assert coerce_optional_int(3.7) == 3  # truncates floats
    assert coerce_optional_int(None) is None
    assert coerce_optional_int("nope") is None
    # A bool where an int is expected is a type error, not 1/0.
    assert coerce_optional_int(True) is None
    assert coerce_optional_int(False) is None


def test_coerce_optional_float() -> None:
    assert coerce_optional_float("1.5") == 1.5
    assert coerce_optional_float(None) is None
    assert coerce_optional_float("x") is None
    assert coerce_optional_float(9.0, minimum=0.0, maximum=1.0) == 1.0


def test_coerce_optional_str_empty_becomes_none() -> None:
    assert coerce_optional_str("hi") == "hi"
    assert coerce_optional_str(None) is None
    # Empty / whitespace-only collapses to None (absence semantics).
    assert coerce_optional_str("") is None
    assert coerce_optional_str("   ") is None


def test_coerce_optional_bool() -> None:
    assert coerce_optional_bool(True) is True
    assert coerce_optional_bool(False) is False
    assert coerce_optional_bool(1) is None
    assert coerce_optional_bool("true") is None


def test_coerce_datetime() -> None:
    now = datetime(2026, 6, 19, 12, 0, 0)
    assert coerce_datetime(now) is now
    assert coerce_datetime("2026-06-19T12:00:00") == now
    assert coerce_datetime("not-a-date") is None
    assert coerce_datetime(123) is None


class _Tone(str, Enum):
    LOW = "low"
    HIGH = "high"


def test_coerce_enum_reads_labels_case_insensitively() -> None:
    assert coerce_enum(_Tone, " HIGH ", default=_Tone.LOW) is _Tone.HIGH
    assert coerce_enum(_Tone, _Tone.HIGH, default=_Tone.LOW) is _Tone.HIGH


def test_coerce_enum_falls_back_on_unknown_or_missing() -> None:
    assert coerce_enum(_Tone, "loud", default=_Tone.LOW) is _Tone.LOW
    assert coerce_enum(_Tone, None, default=_Tone.HIGH) is _Tone.HIGH
    assert coerce_enum(_Tone, 3, default=_Tone.LOW) is _Tone.LOW
