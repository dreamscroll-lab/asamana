"""Actor routes: how the actor performing one character is asked and answers.

- ``GET /actor/candidates``: the cast as anyone could know them at a glance, to choose whom to
  perform. Casting itself happens at confirmation (``POST /confirm`` with ``character_id``).
- ``GET /actor/turn``: the turn waiting for the actor, if any. The same turn is pushed over the
  WebSocket as an ``actor_turn`` frame whenever one opens or closes.
- ``POST /actor/interpret``: read the actor's own words against the open turn. Answers nothing:
  the actor confirms the reading (or edits it) and submits it through ``/actor/answer``.
- ``POST /actor/answer``: answer the open turn, naming it by ``turn_id``.
- ``POST /actor/exit``: the actor steps out for good.

A refusal (an action the character can't take, an amendment to something the turn didn't list,
words with nothing to read) is a normal answer, carried by ``accepted`` and a ``reason`` slug,
the same way a refused directive is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from core.interfaces.action import ActionType
from core.interfaces.actor import (
    ActorAction,
    ActorAmendments,
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorWait,
    EmotionAmendment,
    MemoryAmendment,
    RelationAmendment,
    TurnKind,
)
from core.interfaces.urgency import Urgency
from core.logging import get_logger

from interaction.api.app import ApiServices
from interaction.api.serialization import to_jsonable
from interaction.models import CastCandidate

logger = get_logger(__name__)

# Same quality limit as a directive: a few thousand characters is not one answer.
_MAX_ANSWER_CHARS = 500

# The reason given when the words name nothing the turn can take.
_UNREADABLE = "unreadable"


def build_actor_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(prefix="/api/worlds/{world_id}/actor", tags=["actor"])
    application = services.application
    manager = services.manager

    @router.get("/candidates")
    async def list_candidates(world_id: str) -> list[dict[str, Any]]:
        """Whom the actor could perform: identity visible at a glance, nothing of anyone's
        inner life, goals or relations, so choosing tells the actor nothing about the others."""
        if await manager.get_world(world_id) is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        profiles = await manager.character_profiles(world_id)
        return [
            to_jsonable(CastCandidate(
                agent_id=agent_id,
                name=entry.get("name", ""),
                gender=entry.get("gender", ""),
                age=entry.get("age"),
                role=entry.get("role", ""),
                color=entry.get("color", ""),
                is_main_character=bool(entry.get("is_main_character", False)),
            ))
            for agent_id, entry in sorted(profiles.items())
        ]

    @router.get("/turn")
    async def get_turn(world_id: str) -> dict[str, Any]:
        return {"turn": to_jsonable(application.actor_turn(world_id))}

    @router.post("/interpret")
    async def interpret(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """The open turn's reading of the actor's words; ``reading`` is the answer to confirm,
        in the shape ``/actor/answer`` takes."""
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        if len(text) > _MAX_ANSWER_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"答复过长（{len(text)} 字），请压缩到 {_MAX_ANSWER_CHARS} 字以内。",
            )
        services.require_model_keys()
        turn = application.actor_turn(world_id)
        if turn is None:
            raise HTTPException(status_code=409, detail="No turn is waiting for the actor")
        try:
            turn, reading = await application.interpret_actor_answer(world_id, text)
        except ActorAnswerRejected as exc:
            return _reading(turn.turn_id, None, exc.reason)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _reading(turn.turn_id, reading, None if reading is not None else _UNREADABLE)

    @router.post("/answer")
    async def answer(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        turn_id = payload.get("turn_id")
        turn = application.actor_turn(world_id)
        if turn is None or turn.turn_id != turn_id:
            raise HTTPException(status_code=409, detail="That turn is no longer waiting for the actor")
        try:
            decoded = _decode_answer(turn.kind, payload["answer"])
        # AttributeError: a text member that isn't a string.
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail=f"Malformed answer: {exc}") from exc
        try:
            application.answer_actor_turn(world_id, turn_id=turn.turn_id, answer=decoded)
        except ActorAnswerRejected as exc:
            return {"accepted": False, "reason": exc.reason}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            "actor_answered",
            extra={"world_id": world_id, "step": turn.step, "kind": turn.kind.value, "turn_id": turn.turn_id},
        )
        return {"accepted": True, "reason": None}

    @router.post("/exit")
    async def exit_actor(world_id: str) -> dict[str, Any]:
        """The actor steps out for good: the character returns to AI cognition and this world
        takes no actor again. Stop the run first."""
        # Releasing rebuilds the session, which runs on the models.
        services.require_model_keys()
        if await manager.get_world(world_id) is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        try:
            await application.release_actor(world_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        meta = await manager.get_world(world_id)
        return {**to_jsonable(meta), "run_state": services.run_state(world_id)}

    return router


def _reading(turn_id: str, reading: Any, reason: str | None) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "accepted": reading is not None,
        "reading": to_jsonable(reading),
        "reason": reason,
    }


# ----- answer decoding -------------------------------------------------------
# The wire shape is the contract's own (what ``to_jsonable`` makes of it), so a reading from
# ``/actor/interpret`` goes back unchanged. Unknown keys and wrong types are refused, not
# dropped: an answer that loses a field on the way in is not the answer the actor confirmed.

_T = TypeVar("_T")


def _decode_answer(kind: TurnKind, data: Any) -> ActorDecision | ActorInterruptAnswer:
    if kind is TurnKind.INTERRUPT:
        return _build(ActorInterruptAnswer, data)
    return _build(ActorDecision, data, {
        "action": _optional(lambda v: _build(ActorAction, v, _ACTION_FIELDS)),
        "wait": _optional(lambda v: _build(ActorWait, v)),
        "long_term_goals": _optional(_strings),
        "short_term_goals": _optional(_strings),
        "amendments": _optional(lambda v: _build(ActorAmendments, v, {
            "emotion": _optional(lambda e: _build(EmotionAmendment, e)),
            "memories": lambda ms: tuple(_build(MemoryAmendment, m) for m in _items(ms)),
            "relations": lambda rs: tuple(_build(RelationAmendment, r) for r in _items(rs)),
        })),
    })


def _build(cls: Callable[..., _T], data: Any, fields: Mapping[str, Callable[[Any], Any]] | None = None) -> _T:
    """``cls`` from a JSON object; ``fields`` decode the members that aren't plain JSON."""
    if not isinstance(data, Mapping):
        raise TypeError(f"{getattr(cls, '__name__', cls)} must be an object")
    decode = fields or {}
    return cls(**{key: decode[key](value) if key in decode else value for key, value in data.items()})


def _optional(decode: Callable[[Any], _T]) -> Callable[[Any], _T | None]:
    """A member where null means "not given"."""
    return lambda value: None if value is None else decode(value)


def _items(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return value


def _strings(value: Any) -> tuple[str, ...]:
    items = _items(value)
    if not all(isinstance(item, str) for item in items):
        raise TypeError("expected a list of strings")
    return tuple(items)


_ACTION_FIELDS: dict[str, Callable[[Any], Any]] = {
    "action_type": ActionType,
    "urgency": Urgency,
    "person_ids": _strings,
    "move_carry_ids": _strings,
}
