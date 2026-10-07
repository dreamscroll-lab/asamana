"""Phenomenon enum contract: the classification of visible phenomena of world change.

An LLM-boundary enum shaped like Severity, but unordered ("is fire bigger than rain" is
meaningless); intensity is carried by the same Broadcast's severity.

This file locks: a single source of valid values, prompt choices generated from the enum, the LLM
boundary going through parse_phenomenon, unrecognized values falling back to NONE (the render
layer would rather draw nothing), Broadcast defaulting to no phenomenon, and a sited phenomenon
being invalid without a location.

That last rule is held from both sides: the prompt asks the author for a location (both authors
share one definition), and parsing filters out, with a log, what can't stand. The filter can only
drop the fire while the author usually wants the location added, so both are needed. Only parsing
filters; the Broadcast constructor doesn't check again (see the boundary test below).
"""

from __future__ import annotations

import logging

from core.interfaces.llm import IndexedRef
from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.phenomenon import Phenomenon, parse_phenomenon
from core.interfaces.severity import Severity
from engine.injection import parse_broadcast_spec


def test_phenomenon_values_are_the_single_source_of_truth() -> None:
    assert {p.value for p in Phenomenon} == {
        "none", "rain", "snow", "wind", "fire", "smoke", "quake", "dark",
    }


def test_phenomenon_is_str_enum_serializes_as_value() -> None:
    # Subclasses str → zero migration for JSON/snapshots; .value is the canonical string.
    assert Phenomenon.FIRE == "fire"
    assert Phenomenon.FIRE.value == "fire"


def test_prompt_choices_generated_from_enum_with_none_first() -> None:
    # Prompt choices come from the enum, in sync with the valid values. none comes first: it's a
    # valid, common answer, and giving the LLM "nothing to see here" beats forcing it to pick a
    # phenomenon.
    choices = Phenomenon.prompt_choices()
    assert choices.startswith("none|")
    assert set(choices.split("|")) == {p.value for p in Phenomenon}


def test_phenomenon_is_categorical_not_a_scale() -> None:
    """Unordered classification: no level / ordering; intensity belongs to severity, don't build a
    second ruler here."""
    assert not hasattr(Phenomenon.FIRE, "level")


def test_parse_phenomenon_accepts_enum_and_strings() -> None:
    assert parse_phenomenon(Phenomenon.RAIN) is Phenomenon.RAIN
    assert parse_phenomenon("rain") == Phenomenon.RAIN
    assert parse_phenomenon("RAIN") == Phenomenon.RAIN
    assert parse_phenomenon("  fire ") == Phenomenon.FIRE


def test_parse_phenomenon_invalid_falls_back_to_none() -> None:
    # Fall back to NONE rather than guess a close phenomenon: drawing weather that didn't happen is
    # worse than nothing.
    assert parse_phenomenon("下雨") == Phenomenon.NONE
    assert parse_phenomenon("thunderstorm") == Phenomenon.NONE
    assert parse_phenomenon("") == Phenomenon.NONE
    assert parse_phenomenon(None) == Phenomenon.NONE
    assert parse_phenomenon(0.75) == Phenomenon.NONE
    assert parse_phenomenon("nope", default=Phenomenon.SMOKE) == Phenomenon.SMOKE


def test_broadcast_defaults_to_no_phenomenon() -> None:
    bc = Broadcast(
        content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1,
    )
    assert bc.phenomenon == Phenomenon.NONE


def test_broadcast_normalizes_a_string_phenomenon() -> None:
    """Normalized at the boundary: snapshots/old callers pass strings and get an enum after
    construction (as with severity)."""
    bc = Broadcast(
        content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1,
        severity="high", phenomenon="fire", location_scope="loc_market",
    )
    assert bc.phenomenon is Phenomenon.FIRE
    assert bc.severity is Severity.HIGH


def test_broadcast_normalizes_an_unknown_phenomenon_to_none() -> None:
    bc = Broadcast(
        content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1,
        phenomenon="volcano",
    )
    assert bc.phenomenon is Phenomenon.NONE


def test_sited_phenomena_are_the_ones_that_must_happen_somewhere() -> None:
    """Sited: always happens somewhere. Ambient: covers the whole sky and is valid without a
    location."""
    assert {p for p in Phenomenon if p.is_sited} == {
        Phenomenon.FIRE, Phenomenon.SMOKE, Phenomenon.QUAKE,
    }
    # "Nothing happened" has no place it happened.
    assert Phenomenon.NONE.is_sited is False


def test_the_broadcast_constructor_only_normalises_it_never_judges() -> None:
    """The constructor does not check "sited phenomena need a location"; that rule lives only at
    parse time (next test).

    Don't add a constructor check as a "last line of defense": every construction path gets an
    already-filtered BroadcastSpec or no phenomenon at all, so it never fires and only becomes a
    second source of truth. The constructor is a data contract: it normalizes and makes no policy.
    """
    bc = Broadcast(
        content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1, phenomenon="fire", location_scope=None,
    )
    assert bc.phenomenon is Phenomenon.FIRE      # never silently rewritten


def test_an_ambient_phenomenon_may_cover_the_whole_world() -> None:
    """Rain, snow, wind and darkness cover the whole sky; having no location is their normal
    state."""
    for value in ("rain", "snow", "wind", "dark"):
        bc = Broadcast(
            content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
            deliver_step=1, phenomenon=value, location_scope=None,
        )
        assert bc.phenomenon.value == value


def test_a_sited_phenomenon_survives_when_it_has_a_place() -> None:
    bc = Broadcast(
        content="x", source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1,
        phenomenon="fire", location_scope="loc_market",
    )
    assert bc.phenomenon is Phenomenon.FIRE


def test_the_same_filter_runs_where_the_llm_output_is_parsed() -> None:
    """The fallback filter applies at parse time: the plan itself shouldn't carry a fire that
    can't burn.

    The spec is what gets enqueued, previewed and laid out by describe_plan; let it through and
    upstream sees a plan different from what's actually sent. This is the only filter: every
    downstream construction path gets an already-filtered spec.
    """
    ref = IndexedRef(["loc_market"])
    nowhere = parse_broadcast_spec(
        {"content": "火起", "severity": "high", "location_scope": None, "phenomenon": "fire"}, ref,
    )
    assert nowhere is not None
    assert nowhere.phenomenon is Phenomenon.NONE
    assert nowhere.content == "火起"        # only the scene is dropped; the text still goes out

    somewhere = parse_broadcast_spec(
        {"content": "西市火起", "severity": "high", "location_scope": 1, "phenomenon": "fire"}, ref,
    )
    assert somewhere is not None
    assert somewhere.phenomenon is Phenomenon.FIRE

    # Ambient phenomena normally have no location; don't extend the sited rule to them.
    ambient = parse_broadcast_spec(
        {"content": "大雨倾盆", "severity": "medium", "location_scope": None, "phenomenon": "rain"},
        ref,
    )
    assert ambient is not None
    assert ambient.phenomenon is Phenomenon.RAIN


# ---------------------------------------------------------------------------
# Prompt side: both authors share one definition; phenomenon names come from the enum
# ---------------------------------------------------------------------------


def test_sited_and_ambient_choices_are_generated_from_the_enum() -> None:
    """A hand-written list won't follow when a new sited phenomenon is added, and nobody would
    notice."""
    from core.interfaces.phenomenon import Phenomenon

    sited = Phenomenon.sited_choices().split("|")
    ambient = Phenomenon.ambient_choices().split("|")

    assert set(sited) == {p.value for p in Phenomenon if p.is_sited}
    assert set(ambient) == {p.value for p in Phenomenon if not p.is_sited} - {"none"}
    assert not set(sited) & set(ambient)          # a phenomenon can only be on one side
    assert "none" not in ambient                  # "nothing" can't cover the world


def test_the_shared_definition_states_that_a_sited_phenomenon_needs_a_place() -> None:
    """The fallback can only drop the fire; adding a location is the other direction, which only a
    model reading the instruction can do.

    This definition is shared by the world's two authors (asserted here for DirectorChannel, and in
    ``test_event_system.py`` for EventSystem); separate copies would mean fire on one side and
    silently none on the other.
    """
    from core.prompts import PHENOMENON_DEFINITION
    from engine.director import _SYSTEM_PROMPT as director_prompt

    assert "必须同时给出 location_scope" in PHENOMENON_DEFINITION
    assert Phenomenon.sited_choices() in PHENOMENON_DEFINITION
    assert PHENOMENON_DEFINITION in director_prompt


# ---------------------------------------------------------------------------
# This fallback must leave a trace
# ---------------------------------------------------------------------------


def test_dropping_a_fire_is_logged_at_the_parse_seam(caplog) -> None:
    """A silent fallback is the worst kind: the broadcast goes out, the author's scene is gone, and
    nothing looks wrong.

    The log carries the raw location_scope, to tell whether the model left it empty (fix the prompt)
    or gave an index that doesn't resolve (fix the parsing).
    """
    ref = IndexedRef(["loc_market"])
    with caplog.at_level(logging.WARNING, logger="engine.injection"):
        parse_broadcast_spec(
            {"content": "火起", "severity": "high", "location_scope": 99, "phenomenon": "fire"}, ref,
        )

    assert "sited_phenomenon_without_a_place" in caplog.text
    record = next(r for r in caplog.records if r.message == "sited_phenomenon_without_a_place")
    assert record.phenomenon == "fire"
    assert record.raw_location_scope == 99      # out of range: pointed, but wrongly


def test_a_phenomenon_that_keeps_its_place_stays_quiet(caplog) -> None:
    """No fallback, no log: a warning that fires every time is no warning."""
    with caplog.at_level(logging.WARNING):
        Broadcast(
            content="火起", source="system", broadcast_type=BroadcastType.WORLD_EVENT,
            deliver_step=1, phenomenon="fire", location_scope="loc_market",
        )
        parse_broadcast_spec(
            {"content": "雨", "severity": "low", "location_scope": None, "phenomenon": "rain"},
            IndexedRef(["loc_market"]),
        )

    assert "sited_phenomenon_without_a_place" not in caplog.text
    assert "broadcast_dropped_a_sited_phenomenon" not in caplog.text
