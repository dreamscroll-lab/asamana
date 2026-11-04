"""Read a StepView across its scopes, for tests that ask about one body or one kind of event
without caring which scope holds it.

Scope keys: a location id, ``TRANSIT`` for the way between waypoints, ``WORLD`` for the world.
"""

from __future__ import annotations

from typing import Any

from core.interfaces.snapshot import WorldSnapshot, WorldState
from engine.step_record import action_events
from interaction.models import ActionSummary, AgentView, NpcView, ScopeEvent, StepView

TRANSIT = "<transit>"
WORLD = "<world>"


def agents(step: StepView) -> dict[str, tuple[str, AgentView]]:
    out = {a.agent_id: (TRANSIT, a) for a in step.transit.agents}
    for place_id, scope in step.locations.items():
        out.update({a.agent_id: (place_id, a) for a in scope.agents})
    return out


def agent(step: StepView, agent_id: str) -> AgentView:
    return agents(step)[agent_id][1]


def npcs(step: StepView) -> dict[str, tuple[str, NpcView]]:
    out = {n.npc_id: (TRANSIT, n) for n in step.transit.npcs}
    for place_id, scope in step.locations.items():
        out.update({n.npc_id: (place_id, n) for n in scope.npcs})
    return out


def events(step: StepView, kind: str | None = None) -> list[tuple[str, ScopeEvent]]:
    """Every event of ``kind`` (all kinds when None), with its scope key, in seq order."""
    found: list[tuple[str, ScopeEvent]] = [(WORLD, e) for e in step.world.events]
    found += [(TRANSIT, e) for e in step.transit.events]
    for place_id, scope in step.locations.items():
        found += [(place_id, e) for e in scope.events]
    return sorted(
        (pair for pair in found if kind is None or pair[1].kind == kind), key=lambda p: p[1].seq,
    )


def snapshot_of(
    *,
    step: int = 1,
    world_time: dict[str, Any] | None = None,
    agents: dict[str, dict[str, Any]] | None = None,
    npcs: dict[str, dict[str, Any]] | None = None,
    entities: dict[str, dict[str, Any]] | None = None,
    relations: dict[str, dict[str, Any]] | None = None,
    records: list[dict[str, Any]] = (),  # type: ignore[assignment]
    routes: dict[str, dict[str, Any]] | None = None,
    events: list[dict[str, Any]] = (),  # type: ignore[assignment]
) -> WorldSnapshot:
    """A snapshot as the runtime writes it. ``records`` are raw per-participant action records,
    folded the way the step record folds them; each event keeps the initiator record's ``seq``
    (or gets the next free one)."""
    actions = action_events(list(records), routes=routes or {})
    for event in actions:
        initiator = event["initiator_id"]
        seqs = [
            r.get("seq") for r in records
            if str(r.get("execution_id") or "") == event["execution_id"]
            and str(r.get("agent_id") or "") == initiator and r.get("seq") is not None
        ]
        if seqs:
            event["seq"] = seqs[0]
    taken = [e.get("seq") for e in [*events, *actions] if e.get("seq") is not None]
    nxt = max(taken, default=0) + 1
    for event in [*events, *actions]:
        if event.get("seq") is None:
            event["seq"] = nxt
            nxt += 1
    return WorldSnapshot(
        world_id="w", step=step, timestamp=None, world_time=world_time or {},  # type: ignore[arg-type]
        state=WorldState(
            agents=agents or {}, npcs=npcs or {}, entities=entities or {},
            relations=relations or {},
        ),
        events=[*events, *actions],
    )


def step_of_records(*records: dict) -> StepView:
    """A step holding nothing but these action records, as the runtime emits them."""
    return StepView.from_snapshot(snapshot_of(records=list(records)))


def the_action(*records: dict) -> ActionSummary:
    """The one action event these records fold into."""
    [(_, action)] = events(step_of_records(*records), "action")
    assert isinstance(action, ActionSummary)
    return action


def beats(snapshot: WorldSnapshot) -> list[dict[str, Any]]:
    """Each participant's beat, read off the step's action events: the deed's fields with that
    participant's own part (description, outcome, thought) laid over them."""
    return [
        {**{k: v for k, v in event.items() if k != "parts"}, **part}
        for event in snapshot.events_of("action")
        for part in event.get("parts") or []
    ]
