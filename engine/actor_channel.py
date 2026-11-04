"""The runtime's link to the actor playing one character.

The step suspends on ``request`` until the actor answers, so the world waits at the point where
the character would otherwise think. Between turns it holds the actor's request to break off the
act in progress until the interrupt stage that takes it. Nothing here is persisted: a suspended
step is never committed, and after a restore the character is simply asked again.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from core.interfaces.actor import (
    ActorAnswer, ActorDecision, ActorInterruptAnswer, ActorInvitationAnswer, ActorDialogueTurn, ActorTurn,
    TurnKind,
)
from core.logging import get_logger
from engine.run_control import StepWithdrawn

logger = get_logger(__name__)

# Checks an answer against the step it answers, raising to refuse it (``ActorAnswerRejected``).
AnswerCheck = Callable[[ActorAnswer], None]
# Told the open turn whenever it changes: the turn when one opens, None when it closes.
TurnListener = Callable[[ActorTurn | None], None]

_ANSWER_FOR: dict[TurnKind, type] = {
    TurnKind.DECIDE: ActorDecision,
    TurnKind.INTERRUPT: ActorInterruptAnswer,
    TurnKind.INVITE: ActorInvitationAnswer,
    TurnKind.SPEAK: ActorDialogueTurn,
}


class ActorChannel:
    def __init__(self, on_turn: TurnListener | None = None) -> None:
        self._on_turn = on_turn
        self._agent_id: str | None = None
        self._pending: tuple[ActorTurn, asyncio.Future[ActorAnswer], AnswerCheck | None] | None = None
        # The actor asked to break off what the character is doing, at the next interrupt stage.
        # Given up at the decision stage: by then the character is idle.
        self._break_off = False
        # Set by ``abandon`` until the next run: a stop lands at any point of the step, and a turn
        # that opens after it would wait for an answer nobody gives. Withdrawal is a state, not a
        # one-shot on the turn waiting right now.
        self._withdrawn = False

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
        self._break_off = False

    async def request(self, turn: ActorTurn, *, check: AnswerCheck | None = None) -> ActorAnswer:
        """Suspend until the actor answers ``turn`` with an answer ``check`` accepts."""
        if turn.agent_id != self._agent_id:
            raise ValueError("Turn is not for the followed character")
        if self._pending is not None:
            raise ValueError("A turn is already waiting for the actor")
        if self._withdrawn:
            raise StepWithdrawn()
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

    def submit(self, answer: ActorAnswer, *, on_taken: Callable[[ActorTurn], None] | None = None) -> None:
        """Answer the waiting turn. A refused answer (wrong kind, or one ``check`` rejects)
        raises and leaves the turn open for another. ``on_taken`` is told the turn once the answer
        is taken, before the turn closes."""
        if self._pending is None:
            raise ValueError("No turn is waiting for the actor")
        turn, future, check = self._pending
        if not isinstance(answer, _ANSWER_FOR[turn.kind]):
            raise ValueError(f"A {turn.kind.value} turn takes a {_ANSWER_FOR[turn.kind].__name__}")
        if check is not None:
            check(answer)
        if on_taken is not None:
            on_taken(turn)
        # Cleared now, not when the step resumes: until then the answered turn would still look
        # open to anyone asking.
        self._pending = None
        future.set_result(answer)
        self._announce(None)

    def abandon(self) -> None:
        """Withdraw the waiting turn and every turn the run would still open; the suspended step
        raises ``StepWithdrawn``. Holds until ``reopen``."""
        self._withdrawn = True
        if self._pending is not None:
            _turn, future, _check = self._pending
            self._pending = None
            future.set_exception(StepWithdrawn())
            self._announce(None)

    def reopen(self) -> None:
        """Let turns open again: a new run starts."""
        self._withdrawn = False

    def _announce(self, turn: ActorTurn | None) -> None:
        if self._on_turn is None:
            return
        # A listener is an observer: its failure must not fail the step or lose the answer.
        try:
            self._on_turn(turn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("actor_turn_announce_failed", extra={"error": str(exc)})

    @property
    def break_off_requested(self) -> bool:
        return self._break_off

    def request_break_off(self) -> None:
        self._break_off = True

    def take_break_off(self) -> bool:
        requested, self._break_off = self._break_off, False
        return requested
