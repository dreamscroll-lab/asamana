"""External drive evaluation and motivation blending."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List

from core.interfaces.urgency import Urgency

if TYPE_CHECKING:
    from agent.need import NeedType


class ExternalDriveType(str, Enum):
    AUTHORITY  = "authority"   # order or instruction from a higher-ranked agent
    THREAT     = "threat"      # perceived danger to safety or esteem
    OBLIGATION = "obligation"  # social duty — debt, promise, loyalty
    EVENT      = "event"       # world event creating external urgency


@dataclass(frozen=True)
class ExternalGoal:
    """A single external pressure surface produced by WorldPressureEvaluator."""

    text: str                              # action imperative shown to the LLM
    source_id: str                         # agent_id or event_id that produced this goal
    urgency: Urgency
    drive_type: ExternalDriveType
    related_need: "NeedType | None" = None


class MotivationBlender:
    """Arbitrate external drives against internal needs — additively.

    An external pressure boosts its related_need's score on the same scale as a perceived
    situational signal (``agent.need.LLM_RELEVANCE_WEIGHT``), scaled by urgency. NeedEngine.run
    folds the boosts in before ranking, so ``dominant_need == argmax(scores)`` always holds.
    Not a hard override that forces dominant_need = related_need.
    """

    def external_pressure_boost(
        self,
        external_goals: "List[ExternalGoal]",
    ) -> "Dict[NeedType, float]":
        """Per-need additive score boost, summed over goals.

        Urgency LOW..CRITICAL maps linearly to [0,1] times the shared situational weight:
        LOW→0, NORMAL≈0.67, HIGH≈1.33, CRITICAL→2.0 (the full I×W range). Goals without a
        related_need don't score; they still surface as prompt-visible pressure via NeedEngine.run.
        """

        from agent.need import LLM_RELEVANCE_WEIGHT  # local import: avoid import cycle

        span = Urgency.CRITICAL.level - Urgency.LOW.level
        boosts: "Dict[NeedType, float]" = {}
        for goal in external_goals:
            if goal.related_need is None:
                continue
            strength = (goal.urgency.level - Urgency.LOW.level) / span
            boosts[goal.related_need] = boosts.get(goal.related_need, 0.0) + LLM_RELEVANCE_WEIGHT * strength
        return boosts
