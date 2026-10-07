"""Agent-layer perception: RetrievalQuery, InternalContext, and PerceptionPacket."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from core.interfaces.message import Message
from core.interfaces.perception import Broadcast, SpatialPerception

if TYPE_CHECKING:
    from agent.memory_types import Memory
    from agent.need import NeedEvaluation, NeedState, NeedType
    from agent.personality import EmotionState
    from agent.relation import PerceivedRelation


@dataclass
class RetrievalQuery:
    """Machine-recall projection of the current perception: embedded for ANN search and
    keyword-scanned for need relevance.

    Keep the strings flat, content-dense natural text: no ``" | "`` delimiters, ``key:`` tags,
    ids or timestamps, which dilute the embedding and never appear in memory prose. Never reuse
    them as prompt context; the LLM gets ``render_perceived_signals`` and labeled memory sections.

    ``as_primary`` (factual) uses spatial + direct messages; ``as_combined`` (experiential) adds
    broadcasts. The agent's own goals are intentions, not perceived narrative, so they stay out:
    matching intentions against memory prose is noise.
    """

    spatial_context: str    # location, visible agents, items, ambient events
    message_context: str    # direct inbox content
    broadcast_context: str  # world broadcast content

    def as_primary(self) -> str:
        """High-signal query for factual memory: spatial + direct messages."""
        return "，".join(p for p in [self.spatial_context, self.message_context] if p)

    def as_combined(self) -> str:
        """Full query for experiential memory and need keyword relevance."""
        return "，".join(
            p
            for p in [self.spatial_context, self.message_context, self.broadcast_context]
            if p
        )


@dataclass
class InternalContext:
    """Internal state for one cognition step, built by Agent._build_internal_context() and passed
    to DecisionEngine.decide() alongside external perception.

    "Overlap" is the set of cues fed into in-character prompts to keep behavior consistent:
      P persona            — identity / personality / values / self-image
      E emotion            — current mood (a backdrop, not the answer)
      D direction          — dominant need + short/long-term goals
      R relations+situation — perceived relations, time/place header, people present / signals
      M recent past        — recent factual/experiential memories + recalled memories/insights
    InternalContext is decide's full overlap; other prompts carry only the kinds their output is a
    function of, not all five. A functional judge carries only the signals its judgment depends
    on: first-person overlap would bias it, break the prefix cache and break information asymmetry.
    Perception emotion omits D: short-term goals are plans with preset moves and would echo into
    the output as if perceived.

    Memories are bucketed by kind (as in RetrievalResult) so the decision prompt can tell apart
    events (factual vs experiential), insights ("judgments you have already formed", shown with up
    to 3 insight_sources as grounds) and period_summaries ("the overall impression of that period").
    """

    emotion: "EmotionState"
    dominant_need: "NeedType | None"
    active_needs: "List[NeedState]"
    short_term_goals: List[str]
    long_term_goals: List[str]
    factual_memories: "List[Memory]"
    experiential_memories: "List[Memory]"
    # Not pre-rendered: rendering is the prompt layer's job, via describe_relations().
    relevant_relations: "List[PerceivedRelation]"
    need_evaluation: "NeedEvaluation"
    insights: "List[Memory]" = field(default_factory=list)
    period_summaries: "List[Memory]" = field(default_factory=list)
    insight_sources: "Dict[str, List[Memory]]" = field(default_factory=dict)
    # Transient (from MemorySystem.recent_foiled_attempts, not the persistent substrate); only
    # decide sees them, to back "change tactics after repeated failure".
    recent_foiled_attempts: List[str] = field(default_factory=list)


@dataclass
class PerceptionPacket:
    """Cognition loop's unified perception input, assembled by Agent.plan_step().

    Design contract: perception affects an agent through six channels (see each subsystem for
    implementation status):
      1. Memory formation — weighted direct message > ambient narrative > world broadcast.
      2. Emotion trigger — emotion sits between perception and decision.
      3. Need activation — food appears, hunger rises; under attack, safety spikes.
      4. Goal adjustment — "the palace gate is sealed" invalidates "fetch the letter at the gate".
      5. Decision input — all three external streams enter the decision prompt.
      6. Interrupt trigger — urgent perception can interrupt an ongoing multi-step action.
    """

    agent_id: str
    step: int
    spatial: SpatialPerception
    inbox: List[Message]
    broadcasts: List[Broadcast]
    internal_context: InternalContext
