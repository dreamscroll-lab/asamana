"""Relation tracking for agents.

Single source of truth for the trust/affection numeric semantics and for the
prompt-side formatting of perceived relations. Other subsystems (executors,
world builder, observer) must import the constants and helpers below rather
than re-stating the scale anchors or label format on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence

from agent.personality import EmotionState, EmotionType, parse_emotion_type
from core.interfaces.agent_store import AgentRelation, AgentStoreProvider
from core.interfaces.perception import PerceivedIdentity
from core.numeric import clamp


# Perceived-relation distortion weights (see `_compute_perceived` + test_compute_perceived).
# Fit against scale-anchored targets: K_EMOTION_PULL caps emotion displacement at one trust band
# (0.20) so a passing mood bends but cannot flip a settled bond; K_MEMORY_PULL is set so the
# betrayal-memory anchor lands on its target.
K_EMOTION_PULL: float = 0.20
K_MEMORY_PULL: float = 0.30


# Neutral relation baseline for new/unknown relations; use these, never literal 0.5 / 0.0.
NEUTRAL_TRUST: float = 0.5       # trust ∈ [0,1]; 0.5 = neutral (neither trusting nor suspicious)
NEUTRAL_AFFECTION: float = 0.0   # affection ∈ [-1,1]; 0.0 = indifferent


class RelationDirection(str, Enum):
    """Direction of the actor→target relation in an executor adjudication (the LLM's value for
    the relation field)."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


def parse_relation_direction(raw: str) -> RelationDirection:
    """Normalize the LLM's relation field; non-canonical values (no aliases) become NEUTRAL."""
    s = (raw or "").lower().strip()
    for direction in RelationDirection:
        if s == direction.value:
            return direction
    return RelationDirection.NEUTRAL


from core.prompts import (  # noqa: E402
    AFFECTION_RANGE,
    DECEASED_MARK,
    TRUST_RANGE,
    person_referent,
    relation_legend,
)


@dataclass
class PerceivedRelation:
    """Relation as perceived by an agent in the current moment.

    ``labels`` are from the holder's viewpoint, each ``"kind:role"`` (``"父子:儿子"``) or bare
    ``"kind"`` (``"见过"``); several coexist.
    """

    trust: float
    affection: float
    target_agent_id: str = ""
    target_agent_name: str = ""
    # From the same perception channel as name. Kinship labels ("父子"/"父女") depend on it,
    # especially in relation evolution.
    target_agent_gender: str = ""
    labels: List[str] = field(default_factory=list)
    history_summary: str = ""
    # Transient, derived from dead_ids each step. True renders "（已死亡）" and excludes them as a
    # contactable target.
    deceased: bool = False


def _relation_line(
    *,
    name: str,
    labels: Sequence[str],
    trust: float,
    affection: float,
    history_summary: str = "",
) -> str:
    """Canonical single-relation render: ``名字：[标签 | 标签]，信任度0.70、好感度0.30（history）``.

    Empty ``name`` drops the prefix. Callers pass perceived values for cognition paths and
    objective ones for memory paths.
    """
    labels_text = " | ".join(labels) if labels else "未明确"
    prefix = f"{name}：" if name else ""
    line = f"{prefix}[{labels_text}]，信任度{trust:.2f}、好感度{affection:.2f}"
    if history_summary:
        line += f"（{history_summary}）"
    return line


def render_relation_lines(relations: Sequence[PerceivedRelation], *, limit: int = 5) -> List[str]:
    """One line per perceived relation (no leading ``- ``); inject with ``relation_legend()``.

    Labels give the kind of relation, trust/affection its current strength (two brothers can be
    close or bitter, which labels alone lose). People go through ``person_referent`` so gender and
    "（已死亡）" share one parenthesis.
    """
    return [
        _relation_line(
            name=person_referent(
                rel.target_agent_name,
                rel.target_agent_gender,
                *((DECEASED_MARK,) if rel.deceased else ()),
            ),
            labels=rel.labels,
            trust=rel.trust,
            affection=rel.affection,
            history_summary=rel.history_summary,
        )
        for rel in list(relations)[:limit]
    ]


def format_relation_block(
    *,
    labels: Sequence[str],
    trust: float,
    affection: float,
    include_legend: bool = True,
) -> str:
    """Single-target relation line for executor prompts (no name prefix: the caller names the
    subject in its prose), plus an optional legend."""

    head = _relation_line(name="", labels=labels, trust=trust, affection=affection)
    if include_legend:
        return f"关系数值含义：{relation_legend()}\n{head}"
    return head


def render_relation_context(named_relations: Sequence[tuple[str, AgentRelation]]) -> str:
    """The "my relations with the people involved" block for memory-write paths; "" if empty.

    Objective values, not perceived: the prompt injects current emotion separately, so perceived
    values would count emotion twice. Callers pass only substantive relations, with names.
    """
    if not named_relations:
        return ""
    lines = [relation_legend()]
    for name, rel in named_relations:
        lines.append(
            "  - "
            + _relation_line(
                name=name,
                labels=rel.labels,
                trust=rel.trust_objective,
                affection=rel.affection_objective,
                history_summary=rel.history_summary,
            )
        )
    return "\n".join(lines)


def describe_relations(relations: Sequence[PerceivedRelation]) -> str:
    """Perceived relations as a prompt block: header + legend + one bullet each.

    Module-level so any holder of the perceived list renders identically without a
    ``RelationSystem``. No truncation here; relevant_relations is capped upstream.
    """
    rels = list(relations)
    if not rels:
        return "附近没有相关人物。"
    lines = ["相关人物的关系：", relation_legend()]
    lines.extend("- " + line for line in render_relation_lines(rels, limit=len(rels)))
    return "\n".join(lines)


def _perceived(
    relation: AgentRelation,
    *,
    emotion: str | EmotionState,
    recent_memory_bias: float = 0.0,
    agent_name: str = "",
    agent_gender: str = "",
) -> PerceivedRelation:
    """The perceived view of an objective relation: trust/affection tinted by mood and memory."""
    emotion_state = _coerce_emotion(emotion)
    trust = _compute_perceived(
        objective=relation.trust_objective,
        recent_memory_bias=recent_memory_bias,
        emotion=emotion_state,
        value_range=TRUST_RANGE,
    )
    affection = _compute_perceived(
        objective=relation.affection_objective,
        recent_memory_bias=recent_memory_bias,
        emotion=emotion_state,
        value_range=AFFECTION_RANGE,
    )
    return PerceivedRelation(
        trust=trust,
        affection=affection,
        target_agent_id=relation.to_id,
        target_agent_name=agent_name or relation.to_name or "某人",
        target_agent_gender=agent_gender or relation.to_gender,
        labels=list(relation.labels),
        history_summary=relation.history_summary,
    )


class RelationSystem:
    """Persist objective relations and derive perceived ones."""

    def __init__(self, store: AgentStoreProvider, *, world_id: str, agent_id: str) -> None:
        self._store = store
        self._world_id = world_id
        self._agent_id = agent_id

    async def load_existing(self, target_agent_id: str) -> AgentRelation | None:
        """Load a persisted relation read-only; None if absent. For context reads, where
        get_or_create would persist an empty default relation."""
        return await self._store.load_relation(
            self._world_id, self._agent_id, target_agent_id
        )

    async def get_or_create(self, target_agent_id: str) -> AgentRelation:
        relation = await self._store.load_relation(
            self._world_id,
            self._agent_id,
            target_agent_id,
        )
        if relation is not None:
            relation.trust_objective = _clamp_trust(relation.trust_objective)
            relation.affection_objective = _clamp_affection(relation.affection_objective)
            return relation

        relation = AgentRelation(
            from_id=self._agent_id,
            world_id=self._world_id,
            to_id=target_agent_id,
            trust_objective=NEUTRAL_TRUST,
            affection_objective=NEUTRAL_AFFECTION,
            updated_step=0,
        )
        await self._store.save_relation(relation)
        return relation

    async def perceive(
        self,
        target_agent_id: str,
        emotion: str | EmotionState,
        recent_memory_bias: float = 0.0,
        agent_name: str = "",
        agent_gender: str = "",
    ) -> PerceivedRelation:
        relation = await self.get_or_create(target_agent_id)
        # Cache the referent once learned, so an absent-but-known relation renders by name, not id.
        dirty = False
        if agent_name and relation.to_name != agent_name:
            relation.to_name = agent_name
            dirty = True
        if agent_gender and relation.to_gender != agent_gender:
            relation.to_gender = agent_gender
            dirty = True
        if dirty:
            await self._store.save_relation(relation)
        return _perceived(
            relation, emotion=emotion, recent_memory_bias=recent_memory_bias,
            agent_name=agent_name, agent_gender=agent_gender,
        )

    async def perceive_existing(
        self, target_agent_id: str, emotion: str | EmotionState,
    ) -> PerceivedRelation | None:
        """``perceive`` read-only: None when no relation exists, and nothing is persisted. For
        reads that must not bring a relation into being (an executor's ``complete``)."""
        relation = await self.load_existing(target_agent_id)
        if relation is None:
            return None
        relation.trust_objective = _clamp_trust(relation.trust_objective)
        relation.affection_objective = _clamp_affection(relation.affection_objective)
        return _perceived(relation, emotion=emotion)

    async def significant_relations(self, *, limit: int) -> List[AgentRelation]:
        """Top-N relations by distance from neutral, to bring absent people with a real relation
        into this step's perception (an ally or rival matters even when absent)."""
        relations = await self._store.load_all_relations(self._world_id, self._agent_id)
        relations.sort(
            key=lambda r: abs(r.trust_objective - NEUTRAL_TRUST) + abs(r.affection_objective),
            reverse=True,
        )
        return relations[:limit]

    async def perceive_many(
        self,
        target_agent_ids: Sequence[str],
        *,
        emotion: EmotionState,
        memory_biases: Dict[str, float] | None = None,
        identities: "Dict[str, PerceivedIdentity] | None" = None,
    ) -> List[PerceivedRelation]:
        """Build a perceived snapshot for multiple agents. ``identities`` holds name and gender
        together; two separate maps would drift apart."""

        biases = memory_biases or {}
        known = identities or {}
        relations: list[PerceivedRelation] = []
        for target_agent_id in target_agent_ids:
            who = known.get(target_agent_id)
            relations.append(
                await self.perceive(
                    target_agent_id,
                    emotion=emotion,
                    recent_memory_bias=biases.get(target_agent_id, 0.0),
                    agent_name=who.name if who else "",
                    agent_gender=who.gender if who else "",
                )
            )
        return relations

    async def apply_interaction(
        self,
        target_agent_id: str,
        *,
        trust_delta: float,
        affection_delta: float,
        step: int,
    ) -> AgentRelation:
        """Update trust/affection only.

        Labels and history_summary are written only at world init and by RelationEvolution;
        without an LLM summary they stay empty, which beats stitched low-signal noise.
        ``step`` goes into ``updated_step``, the last substantive change (the to_name backfill in
        ``perceive`` doesn't count).
        """
        relation = await self.get_or_create(target_agent_id)
        adjusted_trust = trust_delta if trust_delta >= 0 else trust_delta * 3.0
        relation.trust_objective = _clamp_trust(relation.trust_objective + adjusted_trust)
        relation.affection_objective = _clamp_affection(
            relation.affection_objective + affection_delta
        )
        relation.interaction_count += 1
        relation.updated_step = step
        await self._store.save_relation(relation)
        return relation

    async def record_contact(self, target_agent_id: str, *, step: int) -> None:
        """Count one exchange without judging it: trust/affection stay as they are. The count is
        what makes a relation one ``RelationEvolution`` weighs (``relation_has_substance``)."""
        relation = await self.get_or_create(target_agent_id)
        relation.interaction_count += 1
        relation.updated_step = step
        await self._store.save_relation(relation)

    async def update_history_summary(
        self, target_agent_id: str, summary: str, *, step: int
    ) -> AgentRelation:
        """Overwrite history_summary (RelationEvolution); independent of labels."""
        relation = await self.get_or_create(target_agent_id)
        relation.history_summary = summary.strip()
        relation.updated_step = step
        await self._store.save_relation(relation)
        return relation

    async def replace_labels(
        self,
        target_agent_id: str,
        labels: Sequence[str],
        *,
        step: int,
    ) -> AgentRelation:
        """Overwrite ``labels`` with a new full set (RelationEvolution). No code-side guardrails:
        structural bonds (blood / spouse / master-disciple / liege) are preserved only by the
        RelationEvolution prompt."""
        cleaned = [str(label).strip() for label in labels if str(label).strip()]
        relation = await self.get_or_create(target_agent_id)
        relation.labels = cleaned
        relation.updated_step = step
        await self._store.save_relation(relation)
        return relation


def _compute_perceived(
    *,
    objective: float,
    recent_memory_bias: float,
    emotion: EmotionState,
    value_range: tuple[float, float],
) -> float:
    """Distort the objective baseline by current emotion + recent-memory bias.

    ``valence`` carries sign and magnitude (mild displeasure bends less than rage); valence 0
    leaves the baseline untouched. See K_EMOTION_PULL for the bound.
    """
    emotion_pull = emotion.intensity * emotion.valence * K_EMOTION_PULL
    memory_pull = recent_memory_bias * K_MEMORY_PULL
    low, high = value_range
    return max(low, min(high, objective + memory_pull + emotion_pull))


def _coerce_emotion(emotion: str | EmotionState) -> EmotionState:
    if isinstance(emotion, EmotionState):
        return emotion
    emotion_type = parse_emotion_type(str(emotion))
    if emotion_type in {EmotionType.FRUSTRATION, EmotionType.ANGER, EmotionType.FEAR, EmotionType.DISGUST, EmotionType.SHAME, EmotionType.CONTEMPT, EmotionType.SADNESS, EmotionType.JEALOUSY}:
        return EmotionState(primary=emotion_type, intensity=0.6, valence=-0.5)
    if emotion_type in {EmotionType.JOY, EmotionType.ANTICIPATION, EmotionType.TRUST, EmotionType.PRIDE}:
        return EmotionState(primary=emotion_type, intensity=0.5, valence=0.3)
    return EmotionState(primary=emotion_type, intensity=0.2, valence=0.0)


def _clamp_trust(value: float) -> float:
    return clamp(value, *TRUST_RANGE)


def _clamp_affection(value: float) -> float:
    return clamp(value, *AFFECTION_RANGE)
