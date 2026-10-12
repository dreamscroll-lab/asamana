"""ActorChannel, the submission contract and the actor's goal list, without a runtime."""

from __future__ import annotations

import asyncio

import pytest

from agent.goals import GoalEntity, GoalStatus, replace_live_goals
from core.interfaces.action import ActionType
from core.interfaces.actor import (
    ActorAction,
    ActorAnswerRejected,
    ActorDecision,
    ActorInterruptAnswer,
    ActorTurn,
    ActorWait,
    TriggerKind,
    TurnKind,
    TurnTrigger,
)
from engine.run_control import StepWithdrawn
from engine.actor_channel import ActorChannel

_MESSAGE = (TurnTrigger(TriggerKind.MESSAGE, "速来"),)


def _work() -> ActorAction:
    return ActorAction(action_type=ActionType.WORK, action_description="批阅文书")


def test_each_answer_type_holds_its_own_stage() -> None:
    ActorDecision(action=_work())
    ActorDecision(wait=ActorWait())
    with pytest.raises(ValueError):
        ActorDecision()
    with pytest.raises(ValueError):
        ActorDecision(action=_work(), wait=ActorWait())
    with pytest.raises(ValueError):
        ActorDecision(wait=ActorWait(), short_term_goals=("  ",))
    with pytest.raises(ValueError):
        ActorWait(steps=0)
    with pytest.raises(ValueError):
        ActorAction(action_type=ActionType.WORK, action_description="  ")
    with pytest.raises(ValueError):
        ActorAction(action_type=ActionType.WORK, action_description="批阅文书", estimated_steps=0)

    ActorInterruptAnswer(interrupt=False)
    ActorInterruptAnswer(interrupt=True, then="去玄武门")
    with pytest.raises(ValueError):   # carrying on leads into no decision
        ActorInterruptAnswer(interrupt=False, then="去玄武门")


@pytest.mark.asyncio
async def test_request_suspends_until_the_actor_answers() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    turn = ActorTurn(agent_id="hero", step=3, kind=TurnKind.DECIDE)
    waiting = asyncio.create_task(channel.request(turn))
    await asyncio.sleep(0)
    assert channel.pending_turn == turn
    assert not waiting.done()

    decision = ActorDecision(action=_work())
    channel.submit(decision)

    assert channel.pending_turn is None   # closed at once, before the step resumes
    assert await waiting is decision


@pytest.mark.asyncio
async def test_submit_rejects_answers_that_do_not_fit_the_turn() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    with pytest.raises(ValueError):
        channel.submit(ActorDecision(wait=ActorWait()))   # nothing is waiting

    waiting = asyncio.create_task(channel.request(ActorTurn("hero", 1, TurnKind.DECIDE)))
    await asyncio.sleep(0)
    with pytest.raises(ValueError):
        channel.submit(ActorInterruptAnswer(interrupt=False))   # wrong stage
    channel.submit(ActorDecision(wait=ActorWait()))
    await waiting

    waiting = asyncio.create_task(channel.request(ActorTurn("hero", 2, TurnKind.INTERRUPT)))
    await asyncio.sleep(0)
    with pytest.raises(ValueError):
        channel.submit(ActorDecision(wait=ActorWait()))   # wrong stage
    channel.submit(ActorInterruptAnswer(interrupt=False))
    await waiting


@pytest.mark.asyncio
async def test_a_refused_answer_leaves_the_turn_open() -> None:
    def refuse_actions(answer) -> None:
        if answer.action is not None:
            raise ActorAnswerRejected("talk_target_not_present")

    channel = ActorChannel()
    channel.follow("hero")
    waiting = asyncio.create_task(
        channel.request(ActorTurn("hero", 1, TurnKind.DECIDE), check=refuse_actions),
    )
    await asyncio.sleep(0)

    with pytest.raises(ActorAnswerRejected) as refused:
        channel.submit(ActorDecision(action=_work()))
    assert refused.value.reason == "talk_target_not_present"
    assert channel.pending_turn is not None and not waiting.done()

    channel.submit(ActorDecision(wait=ActorWait()))
    assert (await waiting).wait == ActorWait()


@pytest.mark.asyncio
async def test_abandon_raises_into_the_suspended_step() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    waiting = asyncio.create_task(channel.request(ActorTurn("hero", 1, TurnKind.DECIDE)))
    await asyncio.sleep(0)

    channel.abandon()

    assert channel.pending_turn is None
    with pytest.raises(StepWithdrawn):
        await waiting


def test_a_turn_asked_again_is_another_turn() -> None:
    # A step replayed after a restore asks the same step and kind again.
    first, again = ActorTurn("hero", 5, TurnKind.DECIDE), ActorTurn("hero", 5, TurnKind.DECIDE)
    assert first.turn_id != again.turn_id


@pytest.mark.asyncio
async def test_every_opening_and_closing_is_announced() -> None:
    def broken_listener(_turn) -> None:
        raise RuntimeError("observer down")

    heard: list[ActorTurn | None] = []
    channel = ActorChannel(heard.append)
    channel.follow("hero")
    first, second, third = (ActorTurn("hero", step, TurnKind.DECIDE) for step in (1, 2, 3))

    waiting = asyncio.create_task(channel.request(first, check=lambda _a: None))
    await asyncio.sleep(0)
    with pytest.raises(ValueError):
        channel.submit(ActorInterruptAnswer(interrupt=False))   # refused: still open, nothing said
    channel.submit(ActorDecision(wait=ActorWait()))
    await waiting

    waiting = asyncio.create_task(channel.request(second))
    await asyncio.sleep(0)
    channel.abandon()
    with pytest.raises(StepWithdrawn):
        await waiting

    waiting = asyncio.create_task(channel.request(third))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert heard == [first, None, second, None, third, None]

    # A failing observer neither fails the step nor loses the answer.
    channel = ActorChannel(broken_listener)
    channel.follow("hero")
    waiting = asyncio.create_task(channel.request(first))
    await asyncio.sleep(0)
    channel.submit(ActorDecision(wait=ActorWait()))
    assert (await waiting).wait == ActorWait()


def test_an_idle_character_decides_every_step_unless_waiting() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    assert channel.wants_decision(step=1, triggers=())

    channel.start_wait(ActorWait(), step=1)
    assert not channel.wants_decision(step=2, triggers=())
    assert not channel.wants_decision(step=50, triggers=())
    assert channel.wants_decision(step=51, triggers=_MESSAGE)
    # The wait ended with the trigger.
    assert channel.wants_decision(step=52, triggers=())


def test_a_timed_wait_ends_on_time_or_on_a_trigger() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    channel.start_wait(ActorWait(steps=3), step=1)
    assert not channel.wants_decision(step=2, triggers=())
    assert not channel.wants_decision(step=3, triggers=())
    assert channel.wants_decision(step=4, triggers=())

    channel.start_wait(ActorWait(steps=10), step=4)
    assert channel.wants_decision(step=5, triggers=_MESSAGE)


def test_words_for_the_next_step_belong_to_their_step_only() -> None:
    """Said with a break-off that then didn't happen, they must not surface in a later step."""
    channel = ActorChannel()
    channel.follow("hero")

    channel.hold_next_words("去玄武门", step=4)
    assert channel.take_next_words(step=4) == "去玄武门"
    assert channel.take_next_words(step=4) == ""

    channel.hold_next_words("去玄武门", step=4)
    assert channel.take_next_words(step=5) == ""


def test_following_the_same_character_keeps_turn_state() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    channel.start_wait(ActorWait(), step=1)
    channel.note_targeted()
    channel.follow("hero")
    assert not channel.wants_decision(step=2, triggers=())
    assert channel.take_targeted()


def test_a_change_of_character_clears_turn_state() -> None:
    channel = ActorChannel()
    channel.follow("hero")
    channel.start_wait(ActorWait(), step=1)
    channel.note_targeted()
    channel.hold_next_words("去玄武门", step=1)
    channel.follow("rival")
    assert channel.wants_decision(step=2, triggers=())
    assert not channel.take_targeted()
    assert channel.take_next_words(step=1) == ""


def test_the_actors_goal_list_becomes_the_live_queue() -> None:
    kept = GoalEntity(id="stg-1-1", text="查清账目", goal_type="short_term", progress_summary="已查一半")
    dropped = GoalEntity(id="stg-1-2", text="拜访旧友", goal_type="short_term")
    done = GoalEntity(id="stg-0-1", text="早起", goal_type="short_term", status=GoalStatus.COMPLETED)

    result = replace_live_goals([done, kept, dropped], ["查清账目", "去城门打听消息"], current_step=7)

    by_text = {g.text: g for g in result}
    assert by_text["查清账目"] is kept                       # progress survives
    assert by_text["拜访旧友"].status is GoalStatus.FAILED   # let go, kept in history
    assert by_text["早起"].status is GoalStatus.COMPLETED
    added = by_text["去城门打听消息"]
    assert added.status is GoalStatus.ACTIVE and added.created_step == 7
    assert len({g.id for g in result}) == len(result)
