"""Unit tests for tuning ambient injection (scenario → per-agent spatials)."""

from __future__ import annotations

from core.interfaces.perception import LocationView, SpatialPerception
from tuning.phase_harness.scenario import apply_ambient_from_scenario


def _spatial(location_id: str) -> SpatialPerception:
    return SpatialPerception(
        location_id=location_id,
        location_view=LocationView(name=location_id, description=""),
        world_time_label="t",
        current_step=1,
    )


def test_location_scoped_ambient_excludes_actor() -> None:
    """A location-scoped ambient reaches every agent at that location EXCEPT the
    excluded actor — the basis of the bystander-reaction test."""
    spatials = {
        "a1": _spatial("loc1"),  # victim, at the scene
        "a2": _spatial("loc1"),  # bystander, at the scene
        "a3": _spatial("loc1"),  # the actor — excluded
        "a4": _spatial("loc2"),  # elsewhere
    }
    name_by_id = {"a1": "Victim", "a2": "Bystander", "a3": "Actor", "a4": "Elsewhere"}
    scenario = {
        "ambient": [
            {"content": "持刃逼近", "strength": 0.8, "location": "loc1", "exclude": "Actor"}
        ]
    }

    apply_ambient_from_scenario(scenario, spatials, set(name_by_id), name_by_id)

    assert [e.content for e in spatials["a1"].ambient_events] == ["持刃逼近"]
    assert [e.content for e in spatials["a2"].ambient_events] == ["持刃逼近"]
    assert spatials["a3"].ambient_events == []  # actor excluded
    assert spatials["a4"].ambient_events == []  # different location


def test_global_ambient_reaches_all() -> None:
    spatials = {"a1": _spatial("loc1"), "a2": _spatial("loc2")}
    name_by_id = {"a1": "A", "a2": "B"}
    apply_ambient_from_scenario(
        {"ambient": [{"content": "血腥味", "strength": 0.5}]}, spatials, set(name_by_id), name_by_id
    )
    assert all(s.ambient_events[0].content == "血腥味" for s in spatials.values())
