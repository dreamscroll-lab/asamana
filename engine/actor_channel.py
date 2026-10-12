"""The runtime's link to the actor playing one character.

The step suspends on ``request`` until the actor answers, so the world waits at the point where
the character would otherwise think. Nothing here is persisted: a suspended step is never
committed, and after a restore the character is simply asked again.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from core.interfaces.actor import (
    ActorDecision,
    ActorInterruptAnswer,
    ActorTurn,
    ActorWait,
    TurnKind,
    TurnTrigger,
)
from core.logging import get_logger
from engine.run_control import StepWithdrawn

logger = get_logger(__name__)

ActorAnswer = ActorDecision | ActorInterruptAnswer
# Checks an answer against the step it answers, raising to refuse it (``ActorAnswerRejected``).
AnswerCheck = Callable[[ActorAnswer], None]
# Told the open turn whenever it changes: the turn when one opens, None when it closes.
TurnListener = Callable[[ActorTurn | None], None]

_ANSWER_FOR: dict[TurnKind, type] = {
    TurnKind.DECIDE: ActorDecision,
    TurnKind.INTERRUPT: ActorInterruptAnswer,
}


class ActorChannel:
    def __init__(self, on_turn: TurnListener | None = None) -> None:
        self._on_turn = on_turn
        self._agent_id: str | None = None
        self._pending: tuple[ActorTurn, asyncio.Future[ActorAnswer], AnswerCheck | None] | None = None
        # What the actor said to do next along with a break-off, and the step it belongs to. Read
        # at that step's decision stage and never carried into another step.
        self._next_words: tuple[int, str] | None = None
        self._waiting = False
        # None while waiting means "until something happens".
        self._wake_step: int | None = None
        self._targeted = False
        # The step of the actor's last decision and the memories listed to them then, so the
        # next turn lists only feedback they haven't seen.
        self._decided_step: int | None = None
        self._listed_memory_ids: frozenset[str] = frozenset()

    @property
    def agent_id(self) -> str | None:
        return self._agent_id

    @property
    def pending_turn(self) -> ActorTurn | None:
        return self._pending[0] if self._pending is not None else None

    def follow(self, agent_id: str | None) -> None:
        """Track the character the actor performs (``None``: nobody). A change of character
        clears the turn state."""
        if agent_id == self._agent_id:
            return
        if self._pending is not None:
            raise ValueError("Cannot change character while a turn is waiting for the actor")
        self._agent_id = agent_id
        self._next_words = None
        self._waiting = False
        self._wake_step = None
        self._targeted = False
        self._decided_step = None
        self._listed_memory_ids = frozenset()

    async def request(self, turn: ActorTurn, *, check: AnswerCheck | None = None) -> ActorAnswer:
        """Suspend until the actor answers ``turn`` with an answer ``check`` accepts."""
        if turn.agent_id != self._agent_id:
            raise ValueError("Turn is not for the followed character")
        if self._pending is not None:
            raise ValueError("A turn is already waiting for the actor")
        future: asyncio.Future[ActorAnswer] = asyncio.get_running_loop().create_future()
        self._pending = (turn, future, check)
        logger.info(
            "actor_turn_opened",
            extra={
                "agent_id": turn.agent_id, "step": turn.step, "kind": turn.kind.value,
                "turn_id": turn.turn_id,
            },
        )
        self._announce(turn)
        try:
            return await future
        finally:
            if self._pending is not None and self._pending[1] is future:
                self._pending = None
                self._announce(None)

    def submit(self, answer: ActorAnswer) -> None:
        """Answer the waiting turn. A refused answer (wrong kind, or one ``check`` rejects)
        raises and leaves the turn open for another."""
        if self._pending is None:
            raise ValueError("No turn is waiting for the actor")
        turn, future, check = self._pending
        if not isinstance(answer, _ANSWER_FOR[turn.kind]):
            raise ValueError(f"A {turn.kind.value} turn takes a {_ANSWER_FOR[turn.kind].__name__}")
        if check is not None:
            check(answer)
        # Cleared now, not when the step resumes: until then the answered turn would still look
        # open to anyone asking.
        self._pending = None
        future.set_result(answer)
        self._announce(None)

    def abandon(self) -> None:
        """Withdraw the waiting turn; the suspended step raises ``StepWithdrawn``."""
        if self._pending is not None:
            _turn, future, _check = self._pending
            self._pending = None
            future.set_exception(StepWithdrawn())
            self._announce(None)

    def _announce(self, turn: ActorTurn | None) -> None:
        if self._on_turn is None:
            return
        # A listener is an observer: its failure must not fail the step or lose the answer.
        try:
            self._on_turn(turn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("actor_turn_announce_failed", extra={"error": str(exc)})

    def hold_next_words(self, words: str, *, step: int) -> None:
        self._next_words = (step, words)

    def take_next_words(self, *, step: int) -> str:
        """The words said for ``step``'s decision; any held for another step have lapsed."""
        held, self._next_words = self._next_words, None
        if held is None or held[0] != step:
            return ""
        return held[1]

    def feedback_window(self, *, step: int) -> tuple[int, frozenset[str]]:
        """Since which step to list feedback, and the memories already listed.

        From the last decision's step on: feedback written later in that step (the action
        settling) is new. Without one in this session, the previous step's.
        """
        if self._decided_step is None:
            return max(step - 1, 1), frozenset()
        return self._decided_step, self._listed_memory_ids

    def note_decided(self, *, step: int, listed_memory_ids: frozenset[str]) -> None:
        self._decided_step = step
        self._listed_memory_ids = listed_memory_ids

    def start_wait(self, wait: ActorWait, *, step: int) -> None:
        self._waiting = True
        self._wake_step = step + wait.steps if wait.steps is not None else None

    def note_targeted(self) -> None:
        self._targeted = True

    def take_targeted(self) -> bool:
        targeted, self._targeted = self._targeted, False
        return targeted

    def wants_decision(self, *, step: int, triggers: tuple[TurnTrigger, ...]) -> bool:
        """Whether the idle character is asked to decide this step.

        Always, unless it chose to wait; a wait ends on any trigger or when its time is up.
        """
        if not self._waiting:
            return True
        if triggers or (self._wake_step is not None and step >= self._wake_step):
            self._waiting = False
            self._wake_step = None
            return True
        return False
