"""Action cooldown: a backstop against an AI character looping on one action type.

A character that starts the same type too often within a short window may not choose it in its
next decision, so the AI has to move the story some other way. The cooldown escalates when it
fires again soon after the last one was used up, up to a cap. Each type has its own
``CooldownPolicy``; the mechanism is the same for all.

This is an objective threshold over the character's own action starts, so it is a rule, not a
cognition gate. It binds only the AI decision: an actor's choices are the user's and are never
held back. Whether a repetition is meaningful stays the decision LLM's call; the cooldown only
breaks a loop that the prompt failed to break.

Both the window and the cooldown are counted in the character's decisions, not in steps. A
character busy with a multi-step act doesn't decide: a step-counted cooldown would run out
unused, and a step window would let a two-step talk never reach the threshold however often the
character chose it.

Only an act that really began counts. One denied in arbitration, refused by the invitee or
ruled infeasible doesn't: repeated failure is the foiled-attempt channel's to handle, and
counting it here would cool down the character arbitration is about to let go first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.interfaces.action import ActionType
from core.logging import get_logger
from core.validate import require_at_least

logger = get_logger(__name__)


@dataclass(frozen=True)
class CooldownPolicy:
    """When one action type cools down, and for how long.

    ``threshold`` of the character's last ``window_decisions`` decisions starting the type fires
    a cooldown. A cooldown that fires within ``window_decisions`` decisions of the last one being
    used up holds for one decision more than it did, up to ``max_decisions``; otherwise it holds
    for one.
    """

    threshold: int
    window_decisions: int
    max_decisions: int

    def __post_init__(self) -> None:
        require_at_least(self.threshold, 1, "threshold")
        require_at_least(self.window_decisions, 1, "window_decisions")
        require_at_least(self.max_decisions, 1, "max_decisions")


# Types under cooldown, each with its own policy; a type not listed never cools down.
COOLDOWN_POLICIES: dict[ActionType, CooldownPolicy] = {
    ActionType.TALK: CooldownPolicy(threshold=3, window_decisions=4, max_decisions=3),
    # Looser than TALK: watching a scene a few times running is often how a lurker works.
    ActionType.COVERT: CooldownPolicy(threshold=4, window_decisions=5, max_decisions=2),
}


@dataclass
class _TypeCooldown:
    # Ordinals of the decisions that started the type.
    starts: list[int] = field(default_factory=list)
    length: int = 0
    remaining: int = 0
    # Ordinal of the decision that used up the latest cooldown; -1 when none has been.
    ended_decision: int = -1


class ActionCooldown:
    """One character's cooldown bookkeeping per listed action type.

    Not committed to the snapshot: a restore restarts it, which at worst lets a loop run a few
    more steps before the cooldown fires again.
    """

    def __init__(self) -> None:
        self._types: dict[ActionType, _TypeCooldown] = {}
        # AI decisions made so far; the n-th decision has ordinal n, and an act that begins
        # belongs to the latest one.
        self._decisions = 0

    def record_start(self, action_type: ActionType, *, step: int, agent_id: str) -> None:
        """The character's own act of ``action_type``, from its latest decision, really began at
        ``step`` (logged only)."""
        policy = COOLDOWN_POLICIES.get(action_type)
        if policy is None:
            return
        state = self._types.setdefault(action_type, _TypeCooldown())
        decision = self._decisions
        state.starts = [d for d in state.starts if decision - d < policy.window_decisions] + [decision]
        if len(state.starts) < policy.threshold:
            return
        soon_again = state.ended_decision >= 0 and decision - state.ended_decision <= policy.window_decisions
        state.length = min(state.length + 1, policy.max_decisions) if soon_again else 1
        state.remaining = state.length
        # Counting restarts: the next cooldown needs ``threshold`` fresh starts.
        state.starts = []
        logger.info(
            "action_cooldown_started",
            extra={
                "agent_id": agent_id,
                "step": step,
                "action_type": action_type.value,
                "cooldown_decisions": state.length,
            },
        )

    def held_back(self) -> frozenset[ActionType]:
        """Types the AI decision may not choose now."""
        return frozenset(t for t, state in self._types.items() if state.remaining > 0)

    def note_decided(self) -> None:
        """The AI decided: every running cooldown has held for one decision."""
        self._decisions += 1
        for state in self._types.values():
            if state.remaining > 0:
                state.remaining -= 1
                if state.remaining == 0:
                    state.ended_decision = self._decisions
