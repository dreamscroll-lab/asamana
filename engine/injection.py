"""Shared substrate for the world's **injection authors**.

The world has two authors with completely asymmetric permissions:

- ``EventSystem`` (``engine/event.py``): the automated LLM event editor, bound by its design charter.
  It has a pacing gate and a quota, can only grow events out of the narrative brief, only adds
  pressure and never resolves anything, and through ``WorldMutationChannel`` can only touch
  unheld things lying on the ground.
- ``DirectorChannel`` (``engine/director.py``): the human director, bound by none of those, who
  can change any world state of people and things through the same ``WorldMutationChannel``.

Those constraints exist because an automated LLM shouldn't have god powers, so the two policy
layers stay separate: plan shape, prompt, pacing gate and quota each belong to one author.

This module holds what the two genuinely share:

1. ``WorldEvent`` + ``serialize_world_event``: the injection record, one schema for observers,
   snapshots, replay and the frontend. ``authored_by`` keeps the two authors apart in the data.
2. ``BroadcastSpec`` / ``MessageSpec`` and their parsers: the payloads both authors send to the
   same two existing channels.
3. ``InjectionDispatcher``: sends a validated injection into the broadcast / message / world
   mutation channels and reports where it landed and whom it reached. Permissions live in
   ``WorldMutationChannel._permits``, so dispatch doesn't branch by author. Don't keep two
   copies: a fix would land in one and not the other.

There is deliberately no shared ledger (see ``InjectionLedger``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import IndexedRef
from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.phenomenon import Phenomenon, parse_phenomenon
from core.interfaces.severity import Severity, parse_severity
from core.interfaces.urgency import Urgency, parse_urgency
from core.logging import get_logger
from engine.broadcast import BroadcastChannel
from engine.message_system import MessageSystem

if TYPE_CHECKING:
    # world_mutation imports this module's Author: a runtime import would be circular.
    from agent.agent import Agent
    from engine.world_mutation import Mutation, WorldMutationChannel

logger = get_logger(__name__)


class Author(str, Enum):
    """Who injected this event; how observers tell "the world did this" from "you did this"."""

    SYSTEM   = "system"      # the automated LLM event editor
    DIRECTOR = "director"    # the human director


@dataclass
class WorldEvent:
    """The record of one injection: who injected what this tick and through which channels."""

    id: str
    triggered_step: int
    narrative_desc: str                              # the editor's overview
    is_positive: bool | None                         # display only; not used for severity
    dispatched_to: list[str]                         # ["broadcast"] / ["message"] / ..., sorted
    # Names, not ids: these go into LLM brief text, where an id would be a layer leak.
    affected_names: list[str] = field(default_factory=list)
    location_label: str | None = None
    authored_by: Author = Author.SYSTEM
    # The director's original text; always empty for SYSTEM.
    #
    # Author layer only; never put it in any prompt: it comes from outside the story, and the LLM
    # editor reading it would know the narrative is being written. Separate ledgers prevent this
    # today; keep it in mind when adding new readers.
    directive_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def serialize_world_event(event: WorldEvent) -> dict[str, Any]:
    """WorldEvent → serialized dict (display form shared by snapshots, event bus and replay).

    One field, one meaning, no aliases: before adding a field, check it isn't an alias of an
    existing one.

    Record the injection itself, not its consequences. ``seq`` and ``receipt`` don't exist yet
    when the injection lands, so the Runtime stamps them in place when assembling the step.
    """
    return {
        "id": event.id,
        "step": event.triggered_step,
        "narrative_desc": event.narrative_desc,
        "is_positive": event.is_positive,
        "dispatched_to": list(event.dispatched_to),
        "affected_names": list(event.affected_names),   # names, for replay / rebuilding briefs
        "location_label": event.location_label,         # narrative place name
        "authored_by": event.authored_by.value,
        "directive_text": event.directive_text,          # author-layer text; never in a prompt
        "metadata": dict(event.metadata),
    }


class InjectionLedger:
    """One author's injections, queryable by step, serializable and restorable.

    Each author has its own ledger, so "the quota only constrains the LLM editor" is a
    structural fact rather than a filter to maintain. Both authors' ``CommittedInjection`` are
    merged by the Runtime on the read side.

    ``author`` is only used by ``restore`` to sort mixed history into the right ledger.
    """

    def __init__(self, author: Author) -> None:
        self._author = author
        self._events: list[WorldEvent] = []

    def __len__(self) -> int:
        return len(self._events)

    def record(self, event: WorldEvent) -> None:
        self._events.append(event)

    def all(self) -> list[WorldEvent]:
        return list(self._events)

    def count_since(self, step: int) -> int:
        """Number of injections after ``step`` (exclusive), for the sliding-window quota count."""
        return sum(1 for ev in self._events if ev.triggered_step > step)

    def recent(self, limit: int) -> list[WorldEvent]:
        """The latest ``limit`` injections, in ascending step order."""
        return sorted(self._events, key=lambda e: e.triggered_step)[-limit:]

    def restore(self, fired_events: Iterable[Mapping[str, Any]]) -> None:
        """Recover only this author's events from the mixed history in the snapshot; otherwise
        the LLM editor would count the human's injections against its own quota."""
        self._events = [
            ev for ev in restore_world_events(fired_events) if ev.authored_by is self._author
        ]


def restore_world_events(fired_events: Iterable[Mapping[str, Any]]) -> list[WorldEvent]:
    """Serialized events → WorldEvent list (inverse of ``serialize_world_event``), by step.

    Unparseable entries are silently skipped: they're only replay material. An unrecognized
    author falls back to ``SYSTEM``, the conservative side (it can only cost the LLM editor quota,
    never grant extra).
    """
    restored: list[WorldEvent] = []
    for payload in fired_events:
        if not isinstance(payload, Mapping):
            continue
        step = payload.get("step")
        if not isinstance(step, int):
            continue
        try:
            authored_by = Author(str(payload.get("authored_by", "")))
        except ValueError:
            authored_by = Author.SYSTEM
        restored.append(
            WorldEvent(
                id=str(payload.get("id") or uuid.uuid4().hex),
                triggered_step=step,
                narrative_desc=str(payload.get("narrative_desc") or ""),
                is_positive=payload.get("is_positive"),
                dispatched_to=list(payload.get("dispatched_to") or []),
                affected_names=list(payload.get("affected_names") or []),
                location_label=payload.get("location_label"),
                authored_by=authored_by,
                directive_text=str(payload.get("directive_text") or ""),
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    restored.sort(key=lambda ev: ev.triggered_step)
    return restored


# ─────────────────────────────────────────────────────────────────────────────
# Channel specs: the payloads both authors send to the same two existing channels, plus parsing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BroadcastSpec:
    """A pending world broadcast (a public change of state with no sender)."""

    content: str
    severity: Severity                               # via parse_severity; invalid → LOW
    location_scope: str | None                       # a known location, or None (world-wide)
    # The visible phenomenon accompanying the change (fire/rain/…), used by the renderer to decide
    # what to draw; NONE = draw nothing.
    phenomenon: Phenomenon = Phenomenon.NONE


@dataclass(frozen=True)
class MessageSpec:
    """A pending directed message."""

    content: str
    recipients: list[str]                            # agent_ids from IndexedRef.resolve
    urgency: Urgency                                 # 4-level enum; invalid → NORMAL


def normalize_is_positive(raw: Any) -> bool | None:
    """The event's own positive/negative tone; unrelated to relation direction, so
    parse_relation_direction isn't reused."""
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _POSITIVE_ALIASES:
            return True
        if s in _NEGATIVE_ALIASES:
            return False
    return None


def parse_broadcast_spec(raw: Any, location_ref: IndexedRef) -> BroadcastSpec | None:
    """LLM output → BroadcastSpec. Empty content means "no such channel"; returns None."""
    if not isinstance(raw, dict):
        return None
    content = str(raw.get("content", "")).strip()
    if not content:
        return None
    severity = parse_severity(raw.get("severity"))
    # Invalid index → world-wide None.
    resolved = location_ref.resolve([raw.get("location_scope")])
    location_scope: str | None = resolved[0] if resolved else None
    # A sited phenomenon (fire/smoke/earthquake) needs a place; drop it here, at parse time, so
    # the spec queued and previewed by describe_plan matches what is actually sent. The prompt
    # carries the other half ("fire needs a location"). See core/interfaces/phenomenon.py.
    phenomenon = parse_phenomenon(raw.get("phenomenon"))
    if location_scope is None and phenomenon.is_sited:
        # The only trace of this fallback. The raw location_scope tells a blank (prompt issue)
        # from an unparseable index (parser issue).
        logger.warning(
            "sited_phenomenon_without_a_place",
            extra={
                "phenomenon": phenomenon.value,
                "raw_location_scope": raw.get("location_scope"),
                "content": content[:60],
            },
        )
        phenomenon = Phenomenon.NONE
    return BroadcastSpec(
        content=content,
        severity=severity,
        location_scope=location_scope,
        phenomenon=phenomenon,
    )


def parse_message_spec(raw: Any, agent_id_list: list[str]) -> MessageSpec | None:
    """LLM output → MessageSpec. None if content is empty or no recipient is valid."""
    if not isinstance(raw, dict):
        return None
    content = str(raw.get("content", "")).strip()
    if not content:
        return None
    indices = raw.get("recipients")
    if not isinstance(indices, list):
        return None
    agent_ids = IndexedRef(agent_id_list).resolve(indices)
    if not agent_ids:
        return None
    urgency = parse_urgency(raw.get("urgency"), default=Urgency.NORMAL)
    return MessageSpec(content=content, recipients=agent_ids, urgency=urgency)


_POSITIVE_ALIASES = {"正面", "positive", "好", "good"}
_NEGATIVE_ALIASES = {"负面", "negative", "坏", "bad"}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch: the one piece of code both authors use to send a validated injection into the world
# ─────────────────────────────────────────────────────────────────────────────

# An injected message has no sender in the story: the recipient reads words of unknown origin.
# The author is a code-layer fact, recorded in ``metadata["source"]`` and not in sender;
# otherwise the out-of-story "director" identity would appear in the narrative layer.
_UNSIGNED_SENDER_ID = "narrator"
_UNSIGNED_SENDER_NAME = "不知来源"


@dataclass(frozen=True)
class CommittedInjection:
    """An injection that has landed in the world; what both authors hand to the Runtime.

    ``target_ids`` / ``displaced_ids`` are code-layer coordinates and never go on the wire:
    the first feeds the end-of-step receipt, the second lets the observation read model mark
    moves the person didn't make themselves.
    """

    event: WorldEvent
    target_ids: tuple[str, ...]
    displaced_ids: tuple[str, ...] = ()


class InjectionDispatcher:
    """Send an injection into broadcast / message / world mutation and produce the record.

    No bookkeeping: the ledger belongs to the author.
    """

    def __init__(
        self,
        *,
        broadcast_channel: BroadcastChannel,
        message_system: MessageSystem,
        mutation_channel: "WorldMutationChannel",
        directory: WorldDirectory,
    ) -> None:
        self._broadcast_channel = broadcast_channel
        self._message_system = message_system
        self._mutations = mutation_channel
        self._directory = directory

    async def dispatch(
        self,
        *,
        author: Author,
        step: int,
        agents: Mapping[str, "Agent"],
        narrative_desc: str,
        broadcast: BroadcastSpec | None,
        message: MessageSpec | None,
        mutations: Sequence["Mutation"] = (),
        is_positive: bool | None = None,
        directive_text: str = "",
    ) -> CommittedInjection | None:
        """Dispatch and return the record; returns ``None`` if nothing went out (don't record
        something that didn't happen).

        Order: mutation → broadcast → message, since broadcasts and messages often talk about the
        change. A failing channel fails alone: a partial result beats losing it all.
        """
        dispatched: set[str] = set()
        touched: list[str] = []
        displaced: list[str] = []
        live = dict(agents)

        for mutation in mutations:
            # The channel reports whom it reached. Don't guess via getattr("body_id"):
            # EntityMutation has no such field, so a torn letter would report "reached nobody".
            outcome = await self._mutations.apply(mutation, step=step, agents=live, author=author)
            if outcome is None:
                continue
            dispatched.add("mutation")
            touched.extend(outcome.reached)
            displaced.extend(outcome.displaced)

        if broadcast is not None:
            published = Broadcast(
                content=broadcast.content,
                source=author.value,
                broadcast_type=BroadcastType.WORLD_EVENT,
                # Published before perception, so perceived the same step.
                deliver_step=step,
                location_scope=broadcast.location_scope,
                severity=broadcast.severity,
                phenomenon=broadcast.phenomenon,
            )
            self._broadcast_channel.publish(published)
            dispatched.add("broadcast")
            # Count the audience, or a pure broadcast reports "affected nobody". Reachability is
            # the channel's call; the dead don't count.
            touched.extend(BroadcastChannel.audience(published, {
                aid: agent.personality.state.current_location
                for aid, agent in live.items() if agent.is_active
            }))

        if message is not None and message.recipients:
            try:
                await self._message_system.publish(Message(
                    id=str(uuid.uuid4()),
                    world_id=self._message_system.world_id,
                    sender_id=_UNSIGNED_SENDER_ID,
                    sender_name=_UNSIGNED_SENDER_NAME,
                    content=message.content,
                    recipients=list(message.recipients),
                    location_scope=None,
                    created_step=step,
                    deliver_step=step,
                    urgency=message.urgency,
                    # Otherwise the recipient would try to form a relation with it and reply.
                    sender_is_agent=False,
                    metadata={"source": author.value, "narrative": True},
                ))
                dispatched.add("message")
                touched.extend(message.recipients)
            except Exception as exc:  # noqa: BLE001 — a partial result beats losing it all
                logger.warning(
                    "injection_message_publish_failed",
                    extra={"author": author.value, "step": step, "error": str(exc)},
                )

        if not dispatched:
            logger.warning("injection_dispatched_nothing", extra={"author": author.value, "step": step})
            return None

        target_ids = tuple(dict.fromkeys(touched))
        event = WorldEvent(
            id=str(uuid.uuid4()),
            triggered_step=step,
            narrative_desc=narrative_desc,
            is_positive=is_positive,
            dispatched_to=sorted(dispatched),
            affected_names=[self._directory.agent_name(aid) for aid in target_ids],
            location_label=self._location_label(broadcast, mutations),
            authored_by=author,
            directive_text=directive_text,
        )
        logger.info(
            "injection_committed",
            extra={
                "author": author.value,
                "step": step,
                "event_id": event.id,
                "channels": event.dispatched_to,
            },
        )
        return CommittedInjection(
            event=event,
            target_ids=target_ids,
            displaced_ids=tuple(dict.fromkeys(displaced)),
        )

    def _location_label(
        self, broadcast: BroadcastSpec | None, mutations: Sequence["Mutation"],
    ) -> str | None:
        """Where this tick lands: the broadcast's location, else the first new thing's place."""
        from engine.world_mutation import SpawnMutation  # circular at module level


        if broadcast is not None and broadcast.location_scope:
            return self._directory.location_name(broadcast.location_scope)
        spawn = next((m for m in mutations if isinstance(m, SpawnMutation)), None)
        return self._directory.location_name(spawn.location_id) if spawn is not None else None
