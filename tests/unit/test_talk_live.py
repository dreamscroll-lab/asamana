"""A conversation with the performed character, spoken turn by turn.

The performed side's actor speaks its own dialogue turns; the AI side speaks in character from its
own column only. What these tests pin: who speaks when, how the conversation ends (a side ends it,
the turns run out, an AI turn can't be had), what the AI side is shown, and that the result is the
same transcript a narrated conversation produces.
"""

from __future__ import annotations

import json

import pytest

from agent.decision import ActionType, AgentAction
from core.interfaces.action import ActionTarget, DialogueTurn, Ref
from core.interfaces.actor import ActorDialogueTurn, Conversation
from core.interfaces.llm import LLMScene
from engine.directory import LiveWorldDirectory
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.social import (
    _LIVE_ACTOR_TURN_CHARS, _LIVE_TURN_SYSTEM, _LIVE_MAX_TURNS, SocialExecutor,
)
from engine.message_system import MessageSystem
from engine.run_control import StepWithdrawn

from tests.unit.test_action_executors import _make_agent
from tests.unit.test_talk_listeners import _env

HERO, OTHER = "agent-a", "agent-b"


class _ScriptedActor:
    """An actor that speaks the given lines in order, then ends; records what it was shown."""

    def __init__(self, lines: list[ActorDialogueTurn] | None = None, *, withdraw: bool = False) -> None:
        self._lines = list(lines or [])
        self._withdraw = withdraw
        self.asked: list[Conversation] = []
        self.told: list[tuple[DialogueTurn, bool]] = []

    async def speak(self, agent, *, step: int, conversation: Conversation) -> ActorDialogueTurn:
        self.asked.append(conversation)
        if self._withdraw:
            raise StepWithdrawn()
        return self._lines.pop(0) if self._lines else ActorDialogueTurn(end=True)

    def spoke(self, *, step: int, dialogue_turn: DialogueTurn, end: bool) -> None:
        self.told.append((dialogue_turn, end))


def _ai_says(container, line: str, *, end: bool = False) -> None:
    """One canned reply serves the AI's lines and both memory summaries."""
    container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps({
        "thought": "先听他说。", "line": line, "end": end,
        "fact": "我和他谈了一场。", "success": True, "relation": "neutral",
    }, ensure_ascii=False)


def _talk(initiator: str, target: str, steps: int = 1) -> AgentAction:
    return AgentAction(
        agent_id=initiator, step=1, action_type=ActionType.TALK, action_description="密议兵符之事",
        target=ActionTarget(acts_on=[Ref.agent(target)], claims=[Ref.agent(target)]),
        estimated_steps=steps, expected_outcome="让他交出兵符",
    )


def _setup(container, actor: _ScriptedActor | None, *, performed: str = HERO):
    agents = {
        HERO: _make_agent(container, world_id="w", agent_id=HERO, name="李世民", background="兵符藏在东宫夹墙"),
        OTHER: _make_agent(container, world_id="w", agent_id=OTHER, name="李建成", background="昨夜调了禁军"),
    }
    agents[performed].actor = actor
    env = _env()
    return SocialExecutor(container.llm_router, LiveWorldDirectory.from_agents(agents, env)), env, agents


def _line_calls(container) -> list[str]:
    """The user message of every AI dialogue-turn call, in order."""
    provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
    return [m[1].content for m in provider.call_history if m[0].content == _LIVE_TURN_SYSTEM]


async def _converse(executor, env, agents, action: AgentAction):
    state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
    return state, await executor.complete(state, 1, agents=agents, environment=env, message_system=None)


@pytest.mark.asyncio
async def test_a_live_conversation_takes_one_step_whatever_was_estimated(container) -> None:
    executor, env, agents = _setup(container, _ScriptedActor())
    state = await executor.start(_talk(HERO, OTHER, steps=3), 1, agents=agents, environment=env, message_system=None)
    assert (state.estimated_steps, state.extra["live"]) == (1, True)

    agents[HERO].actor = None
    narrated = await executor.start(_talk(HERO, OTHER, steps=3), 1, agents=agents, environment=env, message_system=None)
    assert (narrated.estimated_steps, narrated.extra["live"]) == (3, False)


@pytest.mark.asyncio
async def test_the_initiator_speaks_first_and_either_side_may_end_it(container) -> None:
    _ai_says(container, "何事？")
    actor = _ScriptedActor([ActorDialogueTurn(line="大哥留步"), ActorDialogueTurn(line="改日再说", end=True)])
    executor, env, agents = _setup(container, actor)

    _state, results = await _converse(executor, env, agents, _talk(HERO, OTHER))

    assert [(t["speaker_id"], t["line"]) for t in results[0].dialogue] == [
        (HERO, "大哥留步"), (OTHER, "何事？"), (HERO, "改日再说"),
    ]
    first, second = actor.asked
    assert (first.other.agent_id, first.other.name, first.dialogue) == (OTHER, "李建成", ())
    assert (first.turns_left, first.last, first.max_turn_chars) == (
        _LIVE_MAX_TURNS, False, _LIVE_ACTOR_TURN_CHARS,
    )
    assert [t.line for t in second.dialogue] == ["大哥留步", "何事？"]
    # Every line is told to the actor as it is spoken, either side's, the last with the end.
    assert [(t.speaker_id, t.line, end) for t, end in actor.told] == [
        (HERO, "大哥留步", False), (OTHER, "何事？", False), (HERO, "改日再说", True),
    ]
    # The same transcript downstream reads: each side remembers it and moves its relation.
    assert all(r.factual_memory == "我和他谈了一场。" and not r.adjudication_failed for r in results)


@pytest.mark.asyncio
async def test_the_ai_side_ends_it_with_its_line(container) -> None:
    _ai_says(container, "不必多言。", end=True)
    actor = _ScriptedActor([ActorDialogueTurn(line="大哥留步"), ActorDialogueTurn(line="还有一事")])
    executor, env, agents = _setup(container, actor)

    _state, results = await _converse(executor, env, agents, _talk(HERO, OTHER))

    assert [t["line"] for t in results[0].dialogue] == ["大哥留步", "不必多言。"]
    assert len(actor.asked) == 1
    assert actor.told[-1][1] is True


@pytest.mark.asyncio
async def test_an_ai_initiator_speaks_first(container) -> None:
    _ai_says(container, "二郎，坐。")
    actor = _ScriptedActor([ActorDialogueTurn(line="", end=True)])
    executor, env, agents = _setup(container, actor, performed=OTHER)

    _state, results = await _converse(executor, env, agents, _talk(HERO, OTHER))

    assert [(t["speaker_id"], t["line"]) for t in results[0].dialogue] == [(HERO, "二郎，坐。")]
    assert [t.line for t in actor.asked[0].dialogue] == ["二郎，坐。"]


@pytest.mark.asyncio
async def test_the_turns_run_out(container) -> None:
    _ai_says(container, "再说说。")
    actor = _ScriptedActor([ActorDialogueTurn(line=f"第{i}轮") for i in range(_LIVE_MAX_TURNS)])
    executor, env, agents = _setup(container, actor)

    _state, results = await _converse(executor, env, agents, _talk(HERO, OTHER))

    assert len(results[0].dialogue) == _LIVE_MAX_TURNS
    # The actor speaks every other turn, the first of them; each counts what is left.
    assert [c.turns_left for c in actor.asked] == list(range(_LIVE_MAX_TURNS, 0, -2))
    assert not any(c.last for c in actor.asked)   # with an even cap the AI speaks the last
    # The last turn, the AI's, is the conversation's end.
    assert [end for _t, end in actor.told] == [False] * (_LIVE_MAX_TURNS - 1) + [True]


@pytest.mark.asyncio
async def test_a_failed_ai_turn_ends_it_with_what_was_said(container) -> None:
    container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = "not json"
    actor = _ScriptedActor([ActorDialogueTurn(line="大哥留步")])
    executor, env, agents = _setup(container, actor)

    _state, results = await _converse(executor, env, agents, _talk(HERO, OTHER))

    # Nothing is filled in for the side that couldn't speak.
    assert [t["line"] for t in results[0].dialogue] == ["大哥留步"]
    assert [(t.line, end) for t, end in actor.told] == [("大哥留步", False), ("", True)]


@pytest.mark.asyncio
async def test_the_ai_side_is_shown_only_its_own_column(container) -> None:
    _ai_says(container, "何事？")
    executor, env, agents = _setup(container, _ScriptedActor([ActorDialogueTurn(line="大哥留步")]))

    await _converse(executor, env, agents, _talk(HERO, OTHER))

    first = _line_calls(container)[0]
    assert "昨夜调了禁军" in first and "大哥留步" in first
    # Neither the other side's private history nor the purpose he came with.
    assert "兵符藏在东宫夹墙" not in first
    assert "密议兵符之事" not in first and "让他交出兵符" not in first
    assert HERO not in first and OTHER not in first   # no ids across the LLM boundary
    # The turn cap is a backstop, not the AI's to pace itself by.
    assert "还剩" not in first and "最后一轮" not in first


@pytest.mark.asyncio
async def test_an_ai_initiator_is_shown_its_own_purpose(container) -> None:
    _ai_says(container, "二郎，坐。")
    executor, env, agents = _setup(container, _ScriptedActor(), performed=OTHER)

    await _converse(executor, env, agents, _talk(HERO, OTHER))

    first = _line_calls(container)[0]
    assert "密议兵符之事" in first and "让他交出兵符" in first


@pytest.mark.asyncio
async def test_a_withdrawn_step_is_not_adjudicated_as_a_failure(container) -> None:
    """The run stopped while the actor's line was awaited: the step is discarded, so the
    withdrawal must pass through the executor's failure net."""
    executor, env, agents = _setup(container, _ScriptedActor(withdraw=True))
    registry = ActionExecutorRegistry()
    registry.register(ActionType.TALK, executor)
    processor = ExecutionProcessor(
        executor_registry=registry, environment=env,
        message_system=MessageSystem(container.message_provider, world_id="w"),
        directory=LiveWorldDirectory.from_agents(agents, env),
    )
    state = await executor.start(_talk(HERO, OTHER), 1, agents=agents, environment=env, message_system=None)

    with pytest.raises(StepWithdrawn):
        await processor.finalize_executions_batch([state], agents, 1)
