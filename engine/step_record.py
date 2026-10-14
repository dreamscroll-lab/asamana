"""What happened in one step, as the snapshot records it: each event once.

The runtime works with one action record per participant (each writes its own memory, carry
filters on each); the record of the step tells each execution's beat once, with every
participant's own part inside it. This is the only place those records are folded, so no
reader of a snapshot folds them again.

Events are JSON-native dicts, one ``kind`` each:

  action     one beat of one execution: the deed facts, where it was recorded
             (``location_id``), what each place observed (``observations``) and ``parts``,
             one per participant (their own description, outcome and thought).
  message    one delivered message, with where each receiver stood when it reached them.
  broadcast  one delivered broadcast; a place's broadcast lists who heard it (``receivers``).
  injection  one injection's record (``engine.injection.serialize_injection``), plus what
             onlookers at each place perceived of its world changes (``observations``).

``seq`` is stamped afterwards by ``EventSequencer``, in emission order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping

from engine.broadcast import BroadcastChannel

if TYPE_CHECKING:
    from core.interfaces.perception import Broadcast
    from engine.message_system import MessageDelivery

# What each participant's record says in its own words; everything else is the deed's.
_PART_FIELDS = ("action_description", "outcome", "gist", "inner_monologue")


def _unique(items: Iterable[Any]) -> list[str]:
    return [str(i) for i in dict.fromkeys(items) if i]


def action_events(
    records: list[Mapping[str, Any]],
    *,
    routes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fold each execution's per-participant records into one event per beat.

    Records join on (execution_id, phase): an agent who finishes one act and starts the next in a
    step has two executions, and one execution closes at most one beat per step. A record
    without an execution_id stands alone. The initiator's record speaks for
    the deed; a conscript's only restates the intent back at him.

    ``routes``: by initiator, the way walked on a MOVE that ended this step (``arrival_view``).
    """
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, record in enumerate(records):
        execution_id = str(record.get("execution_id") or "")
        key = (execution_id, str(record.get("phase") or "")) if execution_id else (f"#{index}", "")
        groups.setdefault(key, []).append(record)

    events: list[dict[str, Any]] = []
    for group in groups.values():
        initiator_id = str(group[0].get("initiator_id") or group[0].get("agent_id") or "")
        lead = next((r for r in group if str(r.get("agent_id") or "") == initiator_id), group[0])
        # Where each place saw it: participants' records repeat the same observations.
        observations: list[dict[str, Any]] = []
        for record in group:
            for o in record.get("observations") or []:
                if o not in observations:
                    observations.append(dict(o))
        action_type = str(lead.get("action_type") or "")
        phase = str(lead.get("phase") or "")
        events.append({
            "kind": "action",
            "execution_id": str(lead.get("execution_id") or ""),
            "initiator_id": initiator_id,
            "participant_ids": _unique(
                lead.get("participant_ids") or [r.get("agent_id") for r in group]
            ),
            "action_type": action_type,
            "phase": phase,
            "deed": str(lead.get("deed") or ""),
            "target": lead.get("target") or {"acts_on": [], "claims": [], "reaches": []},
            "affected_entity_ids": _unique(e for r in group for e in r.get("affected_entity_ids") or []),
            "acted_upon": _unique(a for r in group for a in r.get("acted_upon") or []),
            "overheard_by": _unique(a for r in group for a in r.get("overheard_by") or []),
            "interrupted_by": str(lead.get("interrupted_by") or ""),
            "succeeded": bool(lead.get("succeeded", True)),
            "failure_reason": str(lead.get("failure_reason") or ""),
            "detected": bool(lead.get("detected", False)),
            "not_executed": bool(lead.get("not_executed", False)),
            "location_id": str(lead.get("location_id") or ""),
            "observations": observations,
            "happening": str(lead.get("happening") or ""),
            "dialogue": list(next((r.get("dialogue") for r in group if r.get("dialogue")), [])),
            "duration_label": str(lead.get("duration_label") or ""),
            "elapsed_steps": int(lead.get("elapsed_steps") or 0),
            "total_steps": int(lead.get("total_steps") or 0),
            "route": (
                dict(routes[initiator_id])
                if action_type == "move" and phase in ("ongoing_complete", "settled")
                and initiator_id in routes
                else None
            ),
            "parts": [
                {"agent_id": str(r.get("agent_id") or ""),
                 **{f: str(r.get(f) or "") for f in _PART_FIELDS}}
                for r in sorted(group, key=lambda r: str(r.get("agent_id") or "") != initiator_id)
            ],
        })
    return events


def message_events(
    deliveries: "MessageDelivery",
    *,
    where: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Each delivered message, with every receiver and where they stood when it reached them
    (``where``: agent id → the location they perceived from this step)."""
    receivers_of: dict[str, list[str]] = {}
    for agent_id, inbox in sorted(deliveries.inboxes.items()):
        for message in inbox:
            receivers_of.setdefault(message.id, []).append(agent_id)
    return [
        {
            "kind": "message",
            "message_id": message.id,
            "sender_id": message.sender_id,
            "sender_name": message.sender_name,
            "recipients": list(message.recipients) if message.recipients is not None else None,
            "location_scope": message.location_scope,
            "content": message.content,
            "intent": message.intent,
            "urgency": message.urgency.value,
            "metadata": dict(message.metadata),
            "receivers": [
                {"agent_id": agent_id, "location_id": where.get(agent_id, "")}
                for agent_id in receivers_of.get(message.id, [])
            ],
        }
        for message in deliveries.delivered_messages
    ]


def broadcast_events(
    broadcasts: Iterable["Broadcast"], *, where: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Each delivered broadcast. A place's broadcast is heard by whoever stood there when it
    reached them (``where``: each living agent → the location they perceived from this step), and
    may have left by the step's end, so its record names them. A world broadcast reaches
    everyone and names nobody (``receivers`` None)."""
    return [
        {
            "kind": "broadcast",
            "content": b.content,
            "source": b.source,
            # JSON-native: a live str-Enum reaches the live observer as "BroadcastType.WORLD_EVENT".
            "broadcast_type": str(getattr(b.broadcast_type, "value", b.broadcast_type or "")),
            "location_scope": b.location_scope,
            "severity": b.severity.value,
            # What the change looks like (fire/rain/…), orthogonal to severity.
            "phenomenon": b.phenomenon.value,
            "receivers": BroadcastChannel.audience(b, where) if b.location_scope else None,
        }
        for b in broadcasts
    ]


def injection_events(serialized: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"kind": "injection", **record} for record in serialized]
