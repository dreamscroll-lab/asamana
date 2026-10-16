"""Actor routes: how the actor performing one character is asked and answers.

- ``GET /actor/candidates``: the cast as anyone could know them at a glance, to choose whom to
  perform. Casting itself happens at confirmation (``POST /confirm`` with ``character_id``).
- ``GET /actor/emotions``: the vocabulary a character's feeling is named in, which an emotion
  amendment picks from.
- ``GET /actor/thread``: the actor's thread over the latest steps (``ActorEntry``s as JSON, in
  order). Each new entry is pushed over the WebSocket as an ``actor_entry`` frame.
- ``GET /actor/turn``: the turn waiting for the actor, if any. The same turn is pushed over the
  WebSocket as an ``actor_turn`` frame whenever one opens or closes.
- ``POST /actor/say``: the actor's own words, read against the open DECIDE turn. Answers
  nothing: the reading comes as the thread's ``said`` entry, and the actor confirms it (or says
  it again) by submitting it through ``/actor/answer``.
- ``POST /actor/answer``: answer the open turn, naming it by ``turn_id``. A SPEAK turn is
  answered with the character's dialogue turn (``ActorDialogueTurn``); one longer than the
  conversation's ``max_turn_chars`` is refused (``turn_too_long``).
- ``POST /actor/break-off``: break off what the character is doing, outside any turn.
- ``PUT /actor/pace``: when an idle character's actor is asked to decide (``ActorPace``).
- ``GET /actor/mind``: the character's mind as the actor sees it, at any time: feeling, goals,
  relations, and any amendments saved but not landed yet.
- ``POST /actor/amend``: amend the character's mind, or add an insight to it. ``landed`` says
  whether it took effect now (a turn is open) or waits for the start of the next step.
- ``POST /actor/exit``: the actor steps out for good.

A refusal (an action the character can't take, an amendment to something the mind doesn't list,
words with nothing to read, a break-off with nothing to break off) is a normal answer, carried
by ``accepted`` and a ``reason`` slug, the same way a refused directive is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from agent.personality import EmotionType
from core.interfaces.action import ActionType
from core.interfaces.actor import (
    ActorAction,
    ActorAmendments,
    ActorAnswer,
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorInvitationAnswer,
    ActorDialogueTurn,
    ActorPace,
    EmotionAmendment,
    GoalAmendment,
    RelationAmendment,
)
from core.interfaces.urgency import Urgency
from core.logging import get_logger
from core.validate import require_in_range

from interaction.api.app import ApiServices
from interaction.api.serialization import to_jsonable

logger = get_logger(__name__)

# Same quality limit as a directive: a few thousand characters is not one answer, nor one insight.
_MAX_ANSWER_CHARS = 500

# How many of the latest steps the thread covers unless asked otherwise, and at most: each step
# read is a snapshot loaded.
_THREAD_STEPS = 60
_THREAD_STEPS_RANGE = (1, 500)


def build_actor_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, Body, HTTPException

    router = APIRouter(prefix="/api/worlds/{world_id}/actor", tags=["actor"])
    application = services.application
    manager = services.manager

    @router.get("/emotions")
    async def list_emotions(world_id: str) -> list[dict[str, str]]:
        return [{"value": e.value, "label": e.label} for e in EmotionType]

    @router.get("/turn")
    async def get_turn(world_id: str) -> dict[str, Any]:
        return {"turn": to_jsonable(application.actor_turn(world_id))}

    @router.get("/thread")
    async def get_thread(world_id: str, steps: int = _THREAD_STEPS) -> dict[str, Any]:
        try:
            require_in_range(steps, _THREAD_STEPS_RANGE, "steps")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if await manager.get_world(world_id) is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        return {"entries": await application.actor_thread(world_id, steps=steps)}

    @router.post("/say")
    async def say(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        """Whether the words have a reading, or the ``reason`` they have none. The reading itself
        reaches the actor as the thread's ``said`` entry, the one place it is told."""
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        if len(text) > _MAX_ANSWER_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"Words too long ({len(text)} chars); keep them within {_MAX_ANSWER_CHARS}.",
            )
        services.require_model_keys()
        await services.ensure_session(world_id)
        try:
            said = await application.actor_say(world_id, text)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"accepted": said.reading is not None, "reason": said.reason}

    @router.post("/break-off")
    async def break_off(world_id: str) -> dict[str, Any]:
        await services.ensure_session(world_id)
        try:
            application.actor_break_off(world_id)
        except ActorAnswerRejected as exc:
            return {"accepted": False, "reason": exc.reason}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info("actor_break_off_requested", extra={"world_id": world_id})
        return {"accepted": True, "reason": None}

    @router.post("/answer")
    async def answer(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        turn_id = payload.get("turn_id")
        if not isinstance(turn_id, str):
            raise HTTPException(status_code=400, detail="turn_id is required")
        try:
            decoded = _decode_answer(payload["answer"])
        # AttributeError: a text member that isn't a string.
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail=f"Malformed answer: {exc}") from exc
        try:
            application.answer_actor_turn(world_id, turn_id=turn_id, answer=decoded)
        except ActorAnswerRejected as exc:
            return {"accepted": False, "reason": exc.reason}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info("actor_answered", extra={"world_id": world_id, "turn_id": turn_id})
        return {"accepted": True, "reason": None}

    @router.put("/pace")
    async def set_pace(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        try:
            pace = ActorPace(payload.get("pace"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown pace: {payload.get('pace')!r}") from exc
        try:
            await application.set_actor_pace(world_id, pace)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info("actor_pace_set", extra={"world_id": world_id, "pace": pace.value})
        return {"pace": pace.value}

    @router.get("/mind")
    async def get_mind(world_id: str) -> dict[str, Any]:
        await services.ensure_session(world_id)
        try:
            mind = await application.actor_mind(world_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return to_jsonable(mind)

    @router.post("/amend")
    async def amend(world_id: str, payload: dict = Body(...)) -> dict[str, Any]:
        await services.ensure_session(world_id)
        try:
            amendments = _decode_amendments(payload)
        # AttributeError: a text member that isn't a string.
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail=f"Malformed amendments: {exc}") from exc
        if any(len(text) > _MAX_ANSWER_CHARS for text in amendments.insights):
            raise HTTPException(
                status_code=400, detail=f"Insight too long; keep it within {_MAX_ANSWER_CHARS} chars.",
            )
        try:
            landed = await application.amend_actor_mind(world_id, amendments)
        except ActorAnswerRejected as exc:
            return {"accepted": False, "reason": exc.reason, "landed": False}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info("actor_amended", extra={"world_id": world_id, "landed": landed})
        return {"accepted": True, "reason": None, "landed": landed}

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


# ----- answer decoding -------------------------------------------------------
# The wire shape is the contract's own (what ``to_jsonable`` makes of it), so a reading from
# ``/actor/say`` goes back unchanged. Unknown keys and wrong types are refused, not
# dropped: an answer that loses a field on the way in is not the answer the actor confirmed.

_T = TypeVar("_T")


def _decode_answer(data: Any) -> ActorAnswer:
    """By its own shape: an INTERRUPT turn's answer is the one with ``interrupt``, an INVITE
    turn's the one with ``accept``, a SPEAK turn's the one with ``end``. Whether it fits the open
    turn is the turn's to say (``ActorChannel.submit``)."""
    if isinstance(data, Mapping) and "end" in data:
        return _build(ActorDialogueTurn, data, {"line": _string, "end": _boolean})
    if isinstance(data, Mapping) and "interrupt" in data:
        return _build(ActorInterruptAnswer, data, {"interrupt": _boolean})
    if isinstance(data, Mapping) and "accept" in data:
        return _build(ActorInvitationAnswer, data, {"accept": _boolean})
    return _build(ActorDecision, data, {
        "action": _optional(lambda v: _build(ActorAction, v, _ACTION_FIELDS)),
        "idle": _boolean,
    })


def _decode_amendments(data: Any) -> ActorAmendments:
    return _build(ActorAmendments, data, {
        "emotion": _optional(lambda e: _build(EmotionAmendment, e)),
        "relations": lambda rs: tuple(_build(RelationAmendment, r) for r in _items(rs)),
        "long_term_goals": _optional(_strings),
        "short_term_goals": _optional(lambda gs: tuple(
            _build(GoalAmendment, g, {"text": _string, "id": _optional(_string)}) for g in _items(gs)
        )),
        "removed_short_term_goal_ids": _strings,
        "insights": _strings,
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


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("expected true or false")
    return value


def _string(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
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
