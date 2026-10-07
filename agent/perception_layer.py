"""Unified perception-to-memory layer.

Perception is memorized when the agent is present or is the target, regardless of whether it
is acting. perceive_step() runs for every active agent each step, whatever its action_status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

from agent.memory_types import (
    IMPORTANCE_HIGH_CUTOFF, IMPORTANCE_MEDIUM_CUTOFF, MemoryImportance, MemoryStream,
)
from core.interfaces.perception import Broadcast, SpatialPerception
from core.interfaces.urgency import Urgency
from core.logging import get_logger
from core.interfaces.severity import severity_to_urgency
from core.prompts import URGENCY_TO_STRENGTH

if TYPE_CHECKING:
    from agent.memory import MemorySystem
    from agent.relation import RelationSystem
    from agent.personality import PersonalityLayer
    from core.interfaces.message import Message

logger = get_logger(__name__)

# External pressure (ExternalGoal) is never written to memory: it is a motive, not an event, and in
# the FACTUAL stream it would later be recalled as real experience. It is consumed live elsewhere.
_SOURCE_TYPE = Literal["ambient", "broadcast", "message"]


# Max writes per source per step (structural, not tuning knobs)
_CAPS_MAIN: dict[str, int] = {
    "ambient": 2, "broadcast": 3, "message": 5,
}
_CAPS_BG: dict[str, int] = {
    "ambient": 1, "broadcast": 2, "message": 3,
}


@dataclass(frozen=True)
class PerceptionTuning:
    """Tunable scalar knobs of perception→memory selection; the tuning harness injects variants.
    Structural maps (per-source caps, urgency/severity→strength) are fixed constants, not knobs."""

    threshold_main: float = 0.15
    threshold_bg: float = 0.35
    # Perception-internal salience, not an urgency concept
    ambient_default_strength: float = 0.2
    # Memory's bucket edges, so the HIGH bar can't diverge from importance_level.
    importance_high_cutoff: float = IMPORTANCE_HIGH_CUTOFF
    importance_medium_cutoff: float = IMPORTANCE_MEDIUM_CUTOFF


_PREFIX: dict[str, str] = {
    "ambient": "[环境感知]",
    "broadcast": "[世界广播]",
    "message": "[消息]",
}


@dataclass
class PerceptionItem:
    content: str
    source: _SOURCE_TYPE
    related_agents: List[str]
    signal_strength: float  # 0.0–1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _strength_to_importance(
    s: float,
    high_cutoff: float = IMPORTANCE_HIGH_CUTOFF,
    medium_cutoff: float = IMPORTANCE_MEDIUM_CUTOFF,
) -> MemoryImportance:
    if s >= high_cutoff:
        return MemoryImportance.HIGH
    if s >= medium_cutoff:
        return MemoryImportance.MEDIUM
    return MemoryImportance.LOW


class PerceptionMemoryLayer:
    """Perception-to-memory subsystem. This step's perception reaches cognition two ways:
      direct:  raw PerceptionPacket fields, visible to this step's decision
      history: memories written by PerceptionMemoryLayer, retrievable from the next step on
    They don't overlap: retrieve_both(before_step=current_step) keeps this step's writes out of
    this step's retrieval.
    """

    def __init__(
        self,
        memory_system: "MemorySystem",
        relation_system: "RelationSystem",
        agent_id: str,
        is_main_character: bool,
        tuning: "PerceptionTuning | None" = None,
    ) -> None:
        self._memory_system = memory_system
        self._relation_system = relation_system
        self._agent_id = agent_id
        self._is_main_character = is_main_character
        self._tuning = tuning or PerceptionTuning()

    async def record(
        self,
        *,
        spatial: SpatialPerception,
        inbox: "List[Message]",
        broadcasts: List[Broadcast],
        personality: "PersonalityLayer",
        step: int,
    ) -> None:
        items = await self._collect_items(spatial, inbox, broadcasts)
        selected = self._select_items(items)
        for item in selected:
            await self._write_item(item, personality, step)

    async def _collect_items(
        self,
        spatial: SpatialPerception,
        inbox: "List[Message]",
        broadcasts: List[Broadcast],
    ) -> List[PerceptionItem]:
        items: List[PerceptionItem] = []

        # Driven purely by AmbientEvent.strength, set by the producer; unset means a weak
        # background signal. A strong signal (exposure, violence) is declared by the producer,
        # never recognized here by source type.
        for ev in spatial.ambient_events:
            if not ev.content:
                continue
            items.append(PerceptionItem(
                content=ev.content,
                source="ambient",
                # agent_actor_ids, not actor_ids: cognition-less bodies must not enter the identity
                # index or they become relation-evolution candidates. Left empty, observed memories
                # have no owner and identity-scoped retrieval misses them.
                related_agents=list(ev.agent_actor_ids),
                signal_strength=ev.strength if ev.strength is not None else self._tuning.ambient_default_strength,
                metadata={"source": "ambient", "explicit_strength": ev.strength is not None},
            ))

        # The caller has already filtered broadcasts by location_scope.
        for bc in broadcasts:
            if not bc.content:
                continue
            items.append(PerceptionItem(
                content=bc.content,
                source="broadcast",
                related_agents=[],
                signal_strength=URGENCY_TO_STRENGTH[severity_to_urgency(bc.severity)],
                metadata={"source": "broadcast", "severity": bc.severity.value},
            ))

        # related_agents holds only senders with cognition; _write_item reads it to decide whether
        # a message is a relation signal. Excluded: narrator pseudo-senders (narration is not a
        # person) and cognition-less bodies (people, but they don't form relations; they still
        # keep sender_name so memory knows who said it).
        for msg in inbox:
            if not msg.content:
                continue
            urgency = getattr(msg, "urgency", Urgency.NORMAL)
            is_narrative = bool(msg.metadata.get("narrative"))
            related = [msg.sender_id] if (
                msg.sender_id and not is_narrative and msg.sender_is_agent
            ) else []
            # Weave the sender into the text, or a recalled memory loses who said it.
            sender_name = (getattr(msg, "sender_name", "") or "某人") if not is_narrative else ""
            # Memory keeps only the speech (metadata "spoken"), not attached material: an
            # errand-runner's scene report is long, stale next step, and would crowd out recall.
            # With no mark, the whole text is speech (a letter).
            body = str(msg.metadata.get("spoken") or "") or msg.content
            content = f"来自{sender_name}的消息：{body}" if sender_name else body
            items.append(PerceptionItem(
                content=content,
                source="message",
                related_agents=related,
                signal_strength=URGENCY_TO_STRENGTH[urgency],
                metadata={
                    "source": "message",
                    "sender_id": msg.sender_id,
                    "message_id": msg.id,
                    "narrative": is_narrative,
                },
            ))

        # Mere co-presence is not memorized: episodic memory records events. Presence reaches
        # the decision through the direct channel (render_perceived_signals).

        return items

    def _select_items(
        self,
        items: List[PerceptionItem],
    ) -> List[PerceptionItem]:
        t = self._tuning
        threshold = t.threshold_main if self._is_main_character else t.threshold_bg
        caps = _CAPS_MAIN if self._is_main_character else _CAPS_BG

        passing = [i for i in items if i.signal_strength >= threshold]

        passing.sort(key=lambda i: i.signal_strength, reverse=True)

        counts: dict[str, int] = {}
        selected: List[PerceptionItem] = []
        for item in passing:
            cap = caps.get(item.source, 0)
            if cap == 0:
                continue
            cnt = counts.get(item.source, 0)
            if cnt >= cap:
                continue
            if any(s.source == item.source and s.content == item.content for s in selected):
                continue
            selected.append(item)
            counts[item.source] = cnt + 1

        return selected

    async def _write_item(
        self,
        item: PerceptionItem,
        personality: "PersonalityLayer",
        step: int,
    ) -> None:
        prefix = _PREFIX.get(item.source, "[感知]")
        content = f"{prefix} {item.content}"
        importance = _strength_to_importance(
            item.signal_strength,
            self._tuning.importance_high_cutoff,
            self._tuning.importance_medium_cutoff,
        )
        triggered_by = f"perception_{item.source}_{step}"
        md = {**item.metadata, "perception_layer": True}
        try:
            await self._memory_system.write(
                content,
                MemoryStream.FACTUAL,
                personality=personality,
                related_agents=item.related_agents,
                triggered_by=triggered_by,
                importance=importance,
                current_step=step,
                metadata=md,
            )
        except Exception as exc:
            logger.warning(
                "perception_layer_write_failed",
                extra={"agent_id": self._agent_id, "source": item.source, "error": str(exc)},
            )

        # A letter that made it into memory is contact with its sender. Count it, don't score it:
        # which way it moves the relation depends on what it says, and that is RelationEvolution's
        # (LLM) call. Without the count a correspondence never becomes a relation at all.
        if item.source == "message":
            for sender_id in item.related_agents:
                if not sender_id:
                    continue
                try:
                    await self._relation_system.record_contact(sender_id, step=step)
                except Exception as exc:
                    logger.warning(
                        "perception_contact_record_failed",
                        extra={"agent_id": self._agent_id, "sender_id": sender_id, "error": str(exc)},
                    )
