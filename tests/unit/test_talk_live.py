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


def _spy_recall(agent, recalled: list | None = None) -> list[tuple[str, str, dict]]:
    """Record every recall ``agent`` makes as ``(kind, query, kwargs)``; ``retrieve`` returns
    ``recalled`` (default nothing), ``recall_about_agent`` nothing."""
    calls: list[tuple[str, str, dict]] = []

    async def _about(other_id, query, **kwargs):
        calls.append(("about", query, kwargs))
        return []

    async def _retrieve(query, **kwargs):
        calls.append(("retrieve", query, kwargs))
        return list(recalled or [])

    agent.memory_system.recall_about_agent = _about
    agent.memory_system.retrieve = _retrieve
    return calls


def _memory(mid: str, text: str):
    from agent.memory_types import Memory, MemoryStream

    return Memory(id=mid, stream=MemoryStream.FACTUAL, agent_id=OTHER, stored_content=text,
                  importance=0.5, created_step=1, kind="event")


@pytest.mark.asyncio
async def test_the_addressee_recalls_by_the_stated_topic_never_the_initiators_purpose(container) -> None:
    _ai_says(container, "何事？", end=True)
    executor, env, agents = _setup(container, _ScriptedActor([ActorDialogueTurn(line="大哥")]))
    addressee = _spy_recall(agents[OTHER])
    performed = _spy_recall(agents[HERO])
    action = _talk(HERO, OTHER)
    action.talk_topic = "商议秋猎"

    await _converse(executor, env, agents, action)

    upfront = [(kind, query) for kind, query, kw in addressee if "score_floor" not in kw]
    # What he knows of the other is ranked by the other's name, the subject by the stated topic.
    assert upfront == [("about", "李世民"), ("retrieve", "商议秋猎")]
    # The performed side's actor remembers for itself: nothing is recalled for it.
    assert performed == []


@pytest.mark.asyncio
async def test_with_no_stated_topic_the_addressee_recalls_no_subject_up_front(container) -> None:
    _ai_says(container, "何事？", end=True)
    executor, env, agents = _setup(container, _ScriptedActor([ActorDialogueTurn(line="大哥")]))
    addressee = _spy_recall(agents[OTHER])

    await _converse(executor, env, agents, _talk(HERO, OTHER))

    upfront = [(kind, query) for kind, query, kw in addressee if "score_floor" not in kw]
    assert upfront == [("about", "李世民")]


@pytest.mark.asyncio
async def test_a_live_ai_initiator_recalls_by_its_own_purpose(container) -> None:
    _ai_says(container, "二郎，坐。", end=True)
    executor, env, agents = _setup(container, _ScriptedActor(), performed=OTHER)
    initiator = _spy_recall(agents[HERO])
    action = _talk(HERO, OTHER)
    action.talk_topic = "商议秋猎"

    await _converse(executor, env, agents, action)

    upfront = [(kind, query) for kind, query, kw in initiator if "score_floor" not in kw]
    assert upfront == [("about", "李建成"), ("retrieve", "密议兵符之事")]


@pytest.mark.asyncio
async def test_a_narrated_talk_recalls_both_sides_by_the_purpose(container) -> None:
    """The narrator writes both sides and recalls only up front: the addressee needs what bears on
    the purpose, or the talk can't get anywhere."""
    executor, env, agents = _setup(container, None)
    agents[HERO].actor = None
    initiator, addressee = _spy_recall(agents[HERO]), _spy_recall(agents[OTHER])
    action = _talk(HERO, OTHER)
    action.talk_topic = "商议秋猎"

    await _converse(executor, env, agents, action)

    assert [(k, q) for k, q, _kw in initiator] == [("about", "李建成"), ("retrieve", "密议兵符之事")]
    assert [(k, q) for k, q, _kw in addressee] == [("about", "李世民"), ("retrieve", "密议兵符之事")]


@pytest.mark.asyncio
async def test_the_ai_side_recalls_by_the_other_sides_last_line_as_the_talk_goes_on(container) -> None:
    from engine.executors.social import (
        _LIVE_RECALL_PER_TURN, _LIVE_RECALL_SCORE_FLOOR, _LIVE_RECALL_TOTAL,
    )

    _ai_says(container, "确有此事。")
    lines = ["嗯", "你可还记得三年前那场大火？", "那夜东宫是谁当值？", "火是从哪里烧起来的？", "后来查出了什么？"]
    actor = _ScriptedActor([ActorDialogueTurn(line=t) for t in lines])
    executor, env, agents = _setup(container, actor)
    pool = [_memory(f"m{i}", f"三年前东宫大火的第{i}件事") for i in range(10)]
    calls: list[tuple[str, dict]] = []

    async def _retrieve(query, *, exclude_ids=(), top_k=10, **kwargs):
        calls.append((query, {"exclude_ids": set(exclude_ids), "top_k": top_k, **kwargs}))
        return [m for m in pool if m.id not in exclude_ids][:top_k]

    async def _about(*_a, **_k):
        return []

    agents[OTHER].memory_system.retrieve = _retrieve
    agents[OTHER].memory_system.recall_about_agent = _about

    await _converse(executor, env, agents, _talk(HERO, OTHER))

    mid_talk = [(q, kw) for q, kw in calls if "score_floor" in kw]
    # "嗯" carries no subject; every longer line is recalled by, at the stricter floor.
    assert [q for q, _kw in mid_talk] == lines[1:1 + _LIVE_RECALL_TOTAL // _LIVE_RECALL_PER_TURN]
    assert all(kw["score_floor"] == _LIVE_RECALL_SCORE_FLOOR for _q, kw in mid_talk)
    # What was recalled already is not recalled again.
    assert mid_talk[1][1]["exclude_ids"] >= {"m0", "m1"}
    prompts = _line_calls(container)
    assert "话说到这里，我想起的" not in prompts[0]
    assert "三年前东宫大火的第0件事" in prompts[1]
    # The quota is per side for the whole conversation.
    assert sum(text.count("三年前东宫大火的第") for text in prompts[-1:]) == _LIVE_RECALL_TOTAL


@pytest.mark.asyncio
async def test_the_ai_lines_are_tagged_actor_and_each_memory_call_carries_the_transcript(container) -> None:
    from core.context import clear_log_context, set_log_context
    from providers.trace.in_memory import InMemoryTraceSink

    sink = InMemoryTraceSink()
    container.llm_router._trace_sink = sink  # noqa: SLF001
    clear_log_context()
    set_log_context(world_id="w", step="1")
    _ai_says(container, "何事？")
    actor = _ScriptedActor([ActorDialogueTurn(line="大哥留步"), ActorDialogueTurn(line="改日再说", end=True)])
    executor, env, agents = _setup(container, actor)
    try:
        await _converse(executor, env, agents, _talk(HERO, OTHER))
    finally:
        clear_log_context()

    lines = [c for c in sink.llm_calls if c.prompt_messages[0]["content"] == _LIVE_TURN_SYSTEM]
    memories = [c for c in sink.llm_calls if c.extra.get("talk_role")]
    # Spoken turn by turn only because a side is performed; the line is still the AI side's own.
    assert lines and all(c.actor and c.agent_id == OTHER for c in lines)
    assert {c.agent_id for c in memories} == {HERO, OTHER}
    assert all(not c.actor for c in memories)
    assert all(c.extra["dialogue"] == [
        {"speaker": "李世民", "line": "大哥留步"}, {"speaker": "李建成", "line": "何事？"},
        {"speaker": "李世民", "line": "改日再说"},
    ] for c in memories)
