"""TALK listeners: the people a decision names beyond the one interlocutor.

The dialogue stays 1v1. What these tests pin is where the extras go: they reach the world
as overheard feedback, and they never cost anybody a turn.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent.agent import Agent, _foiled_key
from agent.decision import ActionCandidate, ActionType, AgentAction
from agent.memory_types import MemoryImportance
from core.interfaces.action import ActionTarget, Ref, TargetAgentEffect
from core.interfaces.llm import LLMScene
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.executors.social import SocialExecutor, _present_listeners
from core.interfaces.place import Place

from tests.unit.test_action_executors import _make_agent


def _env() -> EnvironmentSystem:
    env = EnvironmentSystem()
    for lid in ("hall", "garden"):
        env.space.register_place(Place(
            place_id=lid, name=lid, description="",
            connections={"garden" if lid == "hall" else "hall": 1},
            is_public=True, capacity=50,
        ))
    for aid in ("agent-a", "agent-b", "agent-c"):
        env.place_agent(agent_id=aid, location_id="hall")
    return env


def _talk(agent_id: str, target_id: str, listeners: list[str], steps: int = 1) -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        step=1,
        action_type=ActionType.TALK,
        action_description="密议",
        target=ActionTarget(acts_on=[Ref.agent(target_id)], claims=[Ref.agent(target_id)], reaches=[Ref.agent(a) for a in list(listeners)]),
        estimated_steps=steps,
    )


# ---------------------------------------------------------------------------
# Decision layer: extras become listeners instead of vanishing
# ---------------------------------------------------------------------------

def test_extra_person_indices_become_listeners_not_the_interlocutor(container: object) -> None:
    from tests.unit.test_decision import _make_engine, _make_packet

    engine = _make_engine(container)
    candidates = [ActionCandidate(action_type=ActionType.TALK, description="面对面交谈")]
    packet = _make_packet(visible_agent_ids=["agent-2", "agent-3"])
    payload = json.dumps(
        {
            "selected_index": 0,
            "person_indices": [1, 2],
            "action_description": "我与他们二人密议。",
            "inner_monologue": "此事须当面说定。",
            "estimated_steps": 1,
        },
        ensure_ascii=False,
    )

    selection = engine._parse_llm_selection(payload, candidates, packet)  # noqa: SLF001

    assert selection is not None
    # First named is the one actually talked to; the rest merely hear it.
    assert selection.target.acted_on_agents == ["agent-2"]
    assert selection.target.reached_agents == ["agent-3"]


def test_listeners_do_not_move_the_repeated_intent_key(container: object) -> None:
    """_foiled_key must stay keyed on the interlocutor alone.

    Folding listeners into ``acts_on`` would swap TALK's retry-merge key from one person to
    a sorted set, so the same intent named with and without a bystander would stop merging.
    """
    alone = _talk("agent-a", "agent-b", [])
    with_listener = _talk("agent-a", "agent-b", ["agent-c"])

    assert _foiled_key(alone) == _foiled_key(with_listener)


def test_listeners_are_not_reported_as_targets_of_the_action() -> None:
    action = _talk("agent-a", "agent-b", ["agent-c"])
    # "who was this done to" — a listener was not done to.
    assert action.target.acted_on_agents == ["agent-b"]


# ---------------------------------------------------------------------------
# Presence filter
# ---------------------------------------------------------------------------

class TestPresentListeners:
    def _agents(self, container) -> dict[str, Agent]:
        return {
            aid: _make_agent(container, world_id="w", agent_id=aid, name=aid.upper())
            for aid in ("agent-a", "agent-b", "agent-c")
        }

    def test_keeps_a_colocated_live_listener(self, container) -> None:
        agents = self._agents(container)
        assert _present_listeners(
            ["agent-c"], environment=_env(), agents=agents,
            anchor_id="agent-a", exclude={"agent-a", "agent-b"},
        ) == ["agent-c"]

    def test_drops_the_actor_and_the_interlocutor(self, container) -> None:
        agents = self._agents(container)
        assert _present_listeners(
            ["agent-a", "agent-b", "agent-c"], environment=_env(), agents=agents,
            anchor_id="agent-a", exclude={"agent-a", "agent-b"},
        ) == ["agent-c"]

    def test_drops_someone_who_walked_off(self, container) -> None:
        env = _env()
        env.place_agent(agent_id="agent-c", location_id="garden")
        assert _present_listeners(
            ["agent-c"], environment=env, agents=self._agents(container),
            anchor_id="agent-a", exclude={"agent-a", "agent-b"},
        ) == []

    def test_drops_the_dead_and_the_unknown(self, container) -> None:
        # is_active is what death flips (_trigger_death) and what the arbiter reads to
        # decide a body is unreachable — the listener filter must agree with it.
        agents = self._agents(container)
        agents["agent-c"].personality.apply_vitality_damage(1.0)
        agents["agent-c"].set_active(False)
        assert _present_listeners(
            ["agent-c", "agent-ghost"], environment=_env(), agents=agents,
            anchor_id="agent-a", exclude={"agent-a", "agent-b"},
        ) == []

    def test_deduplicates(self, container) -> None:
        assert _present_listeners(
            ["agent-c", "agent-c"], environment=_env(), agents=self._agents(container),
            anchor_id="agent-a", exclude={"agent-a", "agent-b"},
        ) == ["agent-c"]


# ---------------------------------------------------------------------------
# Executor: listeners are carried, never enrolled, and land as overheard feedback
# ---------------------------------------------------------------------------

class TestSocialExecutorListeners:
    def _setup(self, container, *, listener_fact: str | None = "我听见他们议定了明早入宫的说辞。"):
        # One canned response serves both calls on this scene: "dialogue" feeds the
        # transcript, "fact" feeds the per-agent first-person memory. Dropping "fact"
        # models the listener-memory LLM coming back with nothing usable.
        payload = {
            "deed": "operate",
            "dialogue": [
                {"speaker": 1, "line": "近来可好？"},
                {"speaker": 2, "line": "尚可，你呢？"},
            ],
        }
        if listener_fact is not None:
            payload["fact"] = listener_fact
        container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = json.dumps(
            payload, ensure_ascii=False,
        )
        agents = {
            aid: _make_agent(container, world_id="w", agent_id=aid, name=aid.upper())
            for aid in ("agent-a", "agent-b", "agent-c")
        }
        return SocialExecutor(container.llm_router, LiveWorldDirectory.from_agents(agents, _env())), _env(), agents

    @pytest.mark.asyncio
    async def test_a_listener_never_becomes_a_participant(self, container) -> None:
        """The whole point: arbitration reads participant_ids, so a listener keeps its turn."""
        executor, env, agents = self._setup(container)
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        assert set(state.participant_ids) == {"agent-a", "agent-b"}
        assert state.listener_ids == ["agent-c"]

    @pytest.mark.asyncio
    async def test_complete_lands_the_listener_as_overheard(self, container) -> None:
        executor, env, agents = self._setup(container)
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )

        effects = [e for r in results for e in r.target_effects]
        assert [e.agent_id for e in effects] == ["agent-c"]
        effect = effects[0]
        assert effect.overheard is True
        # Opt-outs: a listener said nothing, so no relation moves and nothing was done to it.
        assert effect.relation_toward_actor is None
        assert effect.vitality_damage == 0
        # The listener's own first-person line — not the raw transcript dumped into memory.
        assert effect.factual_memory == "我听见他们议定了明早入宫的说辞。"
        assert "近来可好" not in effect.factual_memory
        assert "agent-" not in effect.factual_memory

    @pytest.mark.asyncio
    async def test_talk_authorizes_its_transcript_as_a_happening(self, container) -> None:
        """A TALK declares the layer only a watcher can catch, and its content is the
        transcript.

        This is the entire source of what a covert action can learn: bystanders only see "the
        two talked", while someone hiding nearby hears what was said. Without this declaration the
        whole observation track is empty, and then lurking has no intel to hand over except what
        the judge invents (exactly what this mechanism exists to eliminate).
        """
        executor, env, agents = self._setup(container)
        state = await executor.start(
            _talk("agent-a", "agent-b", []), 1,
            agents=agents, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )

        assert all(r.happening for r in results), "交谈必须声明这一层"
        for r in results:
            assert r.happening == r.outcome
            # The gap vs. what bystanders see: they only get "they talked"; this side has the words
            assert all(r.happening != o.text for o in r.observations)

    @pytest.mark.asyncio
    async def test_listener_memory_flies_with_the_participant_summaries(self, container) -> None:
        """One batch, not two.

        The three first-person memories (initiator, addressee, listener) each depend only
        on the transcript, so they belong in one gather. Batching the listener separately
        costs a whole extra LLM round-trip on the critical path. With one listener a merged
        batch peaks at 3 calls in flight; two batches peak at 2.
        """
        executor, env, agents = self._setup(container)
        # Structural, not timed: assert the listener coroutine is already running while both
        # summaries are still outstanding. Measuring wall-clock or peak in-flight instead
        # would measure the router's concurrency cap (2 under the test config, 16 in
        # production), not how these calls were batched.
        summaries_done: list[int] = []
        outstanding_when_listener_began: list[int] = []
        orig_summary = executor._memory_summary  # noqa: SLF001
        orig_listener = executor._listener_memory  # noqa: SLF001

        async def _summary(**kwargs):
            await asyncio.sleep(0)  # suspend, so every sibling task gets to start
            result = await orig_summary(**kwargs)
            summaries_done.append(1)
            return result

        async def _listener(**kwargs):
            outstanding_when_listener_began.append(len(summaries_done))
            return await orig_listener(**kwargs)

        executor._memory_summary = _summary  # noqa: SLF001
        executor._listener_memory = _listener  # noqa: SLF001
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        await executor.complete(state, 2, agents=agents, environment=env, message_system=None)

        # 0 = it started alongside them. 2 = it waited for both, i.e. a second batch.
        assert outstanding_when_listener_began == [0]

    @pytest.mark.asyncio
    async def test_the_record_names_whoever_actually_overheard_it(self, container) -> None:
        """Otherwise the memory a listener walks away with traces back to no event at all.

        Sourced from the landed effects, not the named list, so the record cannot claim
        someone overheard a conversation they were never given a memory of.
        """
        from engine.execution_processor import ExecutionProcessor
        from engine.executors.registry import ActionExecutorRegistry

        executor, env, agents = self._setup(container)
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )
        initiator_result = next(r for r in results if r.action.agent_id == "agent-a")

        processor = ExecutionProcessor(
            executor_registry=ActionExecutorRegistry(),
            environment=env,
            message_system=None,  # type: ignore[arg-type]
            directory=LiveWorldDirectory.from_agents(agents, env),
        )
        record = processor.completion_record(state, initiator_result, agents)

        assert record["overheard_by"] == ["agent-c"]
        # A listener spent no turn, so it must not be folded in as a speaker.
        assert "agent-c" not in record["participant_ids"]

    @pytest.mark.asyncio
    async def test_an_unusable_listener_memory_writes_nothing(self, container) -> None:
        """No conservative filler: a line stripped of what was heard carries no information."""
        executor, env, agents = self._setup(container, listener_fact=None)
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )
        assert [e for r in results for e in r.target_effects] == []

    @pytest.mark.asyncio
    async def test_the_listener_effect_rides_on_exactly_one_result(self, container) -> None:
        """Both sides get an ActionResult; attaching to both would apply the effect twice."""
        executor, env, agents = self._setup(container)
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )
        assert sum(1 for r in results if r.target_effects) == 1

    @pytest.mark.asyncio
    async def test_a_listener_who_left_mid_conversation_hears_nothing(self, container) -> None:
        executor, env, agents = self._setup(container)
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"], steps=3), 1,
            agents=agents, environment=env, message_system=None,
        )
        assert state.listener_ids == ["agent-c"]
        env.place_agent(agent_id="agent-c", location_id="garden")

        results = await executor.complete(
            state, 4, agents=agents, environment=env, message_system=None,
        )
        assert [e for r in results for e in r.target_effects] == []

    @pytest.mark.asyncio
    async def test_no_dialogue_means_no_fabricated_overhearing(self, container) -> None:
        """A talk that never produced words leaves the listener with nothing to remember."""
        executor, env, agents = self._setup(container)
        env.place_agent(agent_id="agent-b", location_id="garden")  # feasibility fails
        state = await executor.start(
            _talk("agent-a", "agent-b", ["agent-c"]), 1,
            agents=agents, environment=env, message_system=None,
        )
        results = await executor.complete(
            state, 2, agents=agents, environment=env, message_system=None,
        )
        assert [e for r in results for e in r.target_effects] == []


# ---------------------------------------------------------------------------
# Feedback layer: overheard content is weighed, not asserted
# ---------------------------------------------------------------------------

class TestOverheardImportance:
    async def _apply(self, container, monkeypatch, *, overheard: bool):
        agent = _make_agent(container, world_id="w", agent_id="agent-c", name="C")
        seen: dict = {}

        async def _record_event(**kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(agent.memory_system, "record_event", _record_event)
        await agent.apply_target_effect(
            TargetAgentEffect(
                agent_id="agent-c", factual_memory="我听见他们说了些什么。", overheard=overheard,
            ),
            from_agent_id="agent-a",
            step=2,
        )
        return seen

    @pytest.mark.asyncio
    async def test_overheard_content_is_left_to_the_importance_evaluator(
        self, container, monkeypatch
    ) -> None:
        seen = await self._apply(container, monkeypatch, overheard=True)
        # None → record_event runs the evaluator; overhearing small talk and overhearing a
        # murder plot must not land on the same shelf.
        assert seen["importance"] is None

    @pytest.mark.asyncio
    async def test_something_done_to_you_keeps_its_asserted_weight(
        self, container, monkeypatch
    ) -> None:
        seen = await self._apply(container, monkeypatch, overheard=False)
        assert seen["importance"] is MemoryImportance.HIGH
