"""Memory records and their scales: streams, kinds, importance buckets, and the stored entry itself.

Data only; the memory system that writes, retrieves and maintains them is ``agent.memory``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List

from agent.personality import EmotionType


# Importance bucket cutoffs. Perception's strength buckets reuse them, so "HIGH/MEDIUM" has one
# global boundary.
IMPORTANCE_CRITICAL_CUTOFF = 0.9
IMPORTANCE_HIGH_CUTOFF = 0.65
IMPORTANCE_MEDIUM_CUTOFF = 0.4


class MemoryStream(str, Enum):
    """Memory stream: the dual-stream contract at the core of Asamana's emergent narrative.

    "Knowing the world" and "shaping the self" can't share one kind of record: the first must be
    faithful, citable and emotion-free, the second bent by personality, private and emotional.
    Merging them makes event reference and self-construction interfere.

    FACTUAL — the agent's private mirror of world state (a log)
      - Shareable: two agents' factual memories of one event may differ in granularity but
        shouldn't contradict; they can be cited, retold and reconciled.
      - Faithful: viewpoint bias is allowed ("I didn't see that part"), distortion is not.
      - Cold: emotion_label is fixed at "objective".
      - Used for: reference anchors, causal tracing for planning and reflection, dialogue
        grounding, falsifiable goal progress, and checking one's view against one's feelings
        (the basis of reflection, deception and self-deception).

    EXPERIENTIAL — how events affected this personality (inner monologue)
      - Not shareable: witnesses of the same event don't get it; the root of information asymmetry.
      - Bent by self_image / core_traits / current emotion: two personalities should record one
        event very differently.
      - Hot: bound to emotion_valence / emotion_label; the event is only the trigger.
      - Used for: emotional continuity, identity, relation coloring (see related_memory_bias),
        motivation behind long-term goals, narrative material, and reflection's insights.

    The streams are bound by a shared event anchor, so retrieval never pairs the facts of one event
    with the feelings of another. Writes are asymmetric: intensity and self-relevance decide
    whether an experiential memory is written; an observation may get only a factual one, and
    dreams or reflection only experiential ones. FACTUAL keeps a character's world consistent;
    EXPERIENTIAL makes their inner life genuinely their own.
    """

    FACTUAL = "factual"
    EXPERIENTIAL = "experiential"


class MemoryImportance(IntEnum):
    """A memory's intrinsic weight, the Importance factor of R/I/R, as buckets.

    Memory.importance is stored as a float in [0, 1]; this enum is only the bucket mapping
    (``importance_level``) for lifecycle rules. Unlike relevance (per query) and recency (over
    time), importance is fixed at write time and subjective: different agents weigh the same
    event differently. It filters noise ("ate noodles" never outranks "saw my father killed"),
    keeps foundational events recallable hundreds of steps later, and is the only factor in the
    lifecycle: CRITICAL memories never decay and are never compressed.

    Every agent is judged by the LLM; ImportanceEvaluator's rule is only the failure floor.
    No theme-specific keyword lists (Rule 7).
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


_IMPORTANCE_ENUM_TO_SCORE: Dict[MemoryImportance, float] = {
    MemoryImportance.LOW: 0.25,
    MemoryImportance.MEDIUM: 0.5,
    MemoryImportance.HIGH: 0.75,
    MemoryImportance.CRITICAL: 0.95,
}


def importance_level(score: float) -> MemoryImportance:
    """Map a continuous importance score to its bucket, for lifecycle decisions only.
    Retrieval ranking uses the continuous score directly."""
    if score >= IMPORTANCE_CRITICAL_CUTOFF:
        return MemoryImportance.CRITICAL
    if score >= IMPORTANCE_HIGH_CUTOFF:
        return MemoryImportance.HIGH
    if score >= IMPORTANCE_MEDIUM_CUTOFF:
        return MemoryImportance.MEDIUM
    return MemoryImportance.LOW


class MemoryKind(str, Enum):
    EVENT = "event"
    INSIGHT = "insight"
    SUMMARY = "summary"


@dataclass
class RetrievalResult:
    """Structured retrieval result.

    Memories are split by kind, plus link info, so callers can present them in separate prompt
    sections (recalled experiences, judgments already formed, the overall impression of a
    period) and the LLM can tell the three kinds of recollection apart.
    """

    events: List[Memory] = field(default_factory=list)
    insights: List[Memory] = field(default_factory=list)
    period_summaries: List[Memory] = field(default_factory=list)
    # event_group_id → the other stream's memory id (dual-stream binding only)
    stream_links: Dict[str, str] = field(default_factory=dict)
    # insight.id → some of its source memories (≤ 3)
    insight_sources: Dict[str, List[Memory]] = field(default_factory=dict)


@dataclass
class Memory:
    """Stored memory entry.

    Dual-stream binding, insight provenance and kind:
      - kind: "event" ordinary event / "insight" reflection output / "summary" compression output
      - event_group_id: both streams (factual + experiential) of one event share a group_id;
        meaningful only for kind="event", always None otherwise.
      - source_ids: for kind="insight", the source experiential ids reflected on;
        for kind="summary", the ids of the compressed cluster members; always empty for "event".
    """

    id: str
    stream: MemoryStream
    agent_id: str = ""
    raw_content: str = ""
    stored_content: str = ""
    importance: float = 0.5  # continuous [0, 1]; bucket via importance_level()
    created_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    emotion_valence: float = 0.0
    # EmotionType canonical value, or meta-label "objective" (factual stream) / "summary" (compressed)
    emotion_label: str = EmotionType.NEUTRAL.value
    related_agents: List[str] = field(default_factory=list)
    triggered_by: str | None = None
    last_accessed_step: int = -1
    decay_score: float = 1.0
    retrieval_count: int = 0
    kind: MemoryKind = MemoryKind.EVENT
    event_group_id: str | None = None
    source_ids: List[str] = field(default_factory=list)
    # insight evolution: event=0; an insight's depth = max(source.depth) + 1.
    # Caps the layering when Reflection selects candidates (see the ReflectionEngine contract).
    reflection_depth: int = 0

    def __post_init__(self) -> None:
        if not self.stored_content:
            self.stored_content = self.raw_content
        self.importance = coerce_importance(self.importance)
        self.kind = MemoryKind(self.kind)

    def touch(self, current_step: int) -> None:
        self.last_accessed_step = current_step
        self.retrieval_count += 1


def coerce_importance(value: MemoryImportance | float | int) -> float:
    """Normalize to a score in [0, 1]. A MemoryImportance or an int 1–4 (a serialized bucket)
    maps to its bucket's center; a float (including 0 and 1) is clamped."""
    if isinstance(value, MemoryImportance):
        return _IMPORTANCE_ENUM_TO_SCORE[value]
    if isinstance(value, int) and value in {1, 2, 3, 4}:
        return _IMPORTANCE_ENUM_TO_SCORE[MemoryImportance(value)]
    score = float(value)
    return max(0.0, min(1.0, score))
