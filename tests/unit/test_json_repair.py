"""A bare quote inside a string value must not void the whole payload.

Chinese output routinely uses ASCII quotes as title marks (``我筹谋数日的"共响一声"之策``); under
Rule 1 such a payload falls back to a null step and the whole cognition call is wasted.

`extract_json` parses normally first, retries once with escapes added on failure, and raises if
that fails too; a valid payload never reaches the repair.
"""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import extract_json, extract_json_object


def test_raw_inner_quotes_are_recovered_verbatim() -> None:
    data = extract_json(
        '{"reason": "我筹谋数日的"共响一声"之策，竟成旁立暗处之局。", '
        '"emotion": "frustration", "intensity": 0.7, "valence": -0.6}'
    )
    assert data["emotion"] == "frustration"
    assert data["intensity"] == 0.7
    # Add escapes, never drop characters: content stays exactly as written.
    assert data["reason"] == '我筹谋数日的"共响一声"之策，竟成旁立暗处之局。'


def test_a_closing_quote_is_not_swallowed_into_the_value() -> None:
    # Position decides: only a quote followed by `,}]:` closes the string. One misread merges
    # every later field into the previous value.
    data = extract_json('{"a": "他说"好"，真的", "b": "x", "c": [1, 2]}')
    assert data == {"a": '他说"好"，真的', "b": "x", "c": [1, 2]}


def test_raw_control_chars_inside_a_string_are_escaped() -> None:
    assert extract_json('{"line": "上句\n下句"}')["line"] == "上句\n下句"


def test_valid_payloads_are_untouched() -> None:
    assert extract_json('{"a": "已转义的\\"引号\\""}') == {"a": '已转义的"引号"'}
    assert extract_json('```json\n{"a": {"b": [1, 2]}}\n```') == {"a": {"b": [1, 2]}}
    assert extract_json('闲话 {"a": "x"} 闲话') == {"a": "x"}


def test_structural_damage_still_raises() -> None:
    # Escapes can be repaired, structure can't. Truncation still raises, and the caller falls back
    # per Rule 1.
    with pytest.raises(json.JSONDecodeError):
        extract_json('{"reason": "话说到一半就断了')
    with pytest.raises(json.JSONDecodeError):
        extract_json("not json at all")


def test_extract_json_object_repairs_too() -> None:
    assert extract_json_object('{"reason": "他的"共响一声"之策", "emotion": "joy"}')["emotion"] == "joy"
    assert extract_json_object("garbage") is None
