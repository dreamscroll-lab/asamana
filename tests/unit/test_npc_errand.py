"""An errand from handoff to report-back: the requester spends one step, someone else does the
running.

This file tests two things: the single ``ErrandExecutor`` ruling (accept or not) and the
deterministic progress ``NpcRunner`` makes afterwards (walk, do, look, return, report). Nothing is
discretionary after the ruling, so apart from that one LLM call everything can be asserted exactly.
"""

from __future__ import annotations

import pytest

from core.interfaces.action import (
    ActionTarget, ActionType, AgentAction, ErrandOrder, Ref,
)
from core.interfaces.condition import BodyCondition
from core.interfaces.llm import LLMResponse
from engine.clock import WorldTime, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.executors.base import action_semantics
from engine.executors.errand import ErrandExecutor
from engine.message_system import MessageSystem
from agent.personality import SoulLayer
from engine.npc_runner import NpcRunner
from providers.message.in_memory import InMemoryMessageProvider
from world.models import EntityPresence, NpcSeed, WorldEntity, WorldEntityType
from core.interfaces.place import Place


class _FixedLLM:
    """A judge that always says the same thing; used only for PHYSICAL.

    ERRAND has no judge: sending an NPC leaves nothing to adjudicate (see ErrandExecutor). Acting on
    a person is different and genuinely discretionary, so this one stays.
    """

    def __init__(self, content: str) -> None:
        self.content = content

    async def complete(self, scene, messages, **kw) -> LLMResponse:
        return LLMResponse(content=self.content, input_tokens=0, output_tokens=0, model="test")

    async def complete_with_retry(self, scene, messages, **kw) -> LLMResponse:
        return await self.complete(scene, messages, **kw)


def _location(entity_id: str, name: str, **connections: int) -> Place:
    return Place(
        place_id=entity_id, name=name, connections=dict(connections),
    )


def _world() -> EnvironmentSystem:
    env = EnvironmentSystem()
    # hall — yard — gate: two hops, to test one tile per step.
    env.space.register_place(_location("hall", "大殿", yard=1))
    env.space.register_place(_location("yard", "庭中", hall=1, gate=1))
    env.space.register_place(_location("gate", "宫门", yard=1))
    return env


def _runner(
    env: EnvironmentSystem, *, souls: dict | None = None,
) -> tuple[NpcRunner, MessageSystem]:
    """Only pass souls when checking that he says names, not ids; elsewhere names don't matter."""
    messages = MessageSystem(InMemoryMessageProvider(), world_id="w")
    return NpcRunner(
        environment=env, message_system=messages,
        directory=LiveWorldDirectory(souls=souls or {}, environment=env), world_id="w",
        # One second per step: the maps here write edge walking time in steps.
        seconds_per_step=1,
    ), messages


def _bearer(env: EnvironmentSystem, at: str = "hall") -> str:
    env.spawn_npc(
        NpcSeed(name="王二", gender="男", age=34, description="跑得快、认得路"),
        location_id=at,
    )
    return env.all_npcs()[-1].npc_id


def _world_time(step: int) -> WorldTime:
    return WorldTime.from_step(step, WorldTimeConfig(start_hour=6, seconds_per_step=3600))


async def _run_full_errand(runner: NpcRunner, steps: int = 2) -> None:
    """Run a hall→gate errand to completion: two steps after the step it was assigned.

    Both edges of hall—yard—gate weigh one step and he walks ``NPC_PACE`` (=2) steps per step, so
    step 1 reaches gate and does the task, step 2 walks back to hall and reports. He doesn't move
    again on a leg's last step, so the task happens where he stands (the same-place invariant in
    ``NpcRunner._advance_one``).

    These tests skip ``begin_step`` and record the errand on step 0; for the full schedule including
    the assignment step, see ``test_the_legs_he_walks_are_the_lines_the_reader_is_told``.
    """
    for step in range(1, steps + 1):
        await runner.advance(step)


# ---------------------------------------------------------------------------
# Assignment: told to go, he goes; there's no third party to adjudicate


@pytest.mark.asyncio
async def test_a_tool_body_is_told_and_it_goes() -> None:
    """Sending an NPC needs no adjudication: it has no will and no skill to judge.

    This path makes no LLM calls. All three viewpoint channels come from templates (they derive
    entirely from the four axes and the names), and for observation a template is safer than an LLM:
    the instructions are privileged, and a fixed template can't leak them.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")

    order = ErrandOrder(npc_id, "gate", recipient_id="a2", message="今夜勿出")
    result = (await _run_errand(env, order, actor_id="a1"))[0]

    assert result.succeeded is True
    assert result.errand_orders == [order]
    assert result.adjudication_failed is False
    # The observer's deed comes from this action record: ERRAND is its own action.
    assert action_semantics(result.action, result)["deed"] == "errand"
    assert "宫门" in result.outcome and "今夜勿出" in result.outcome    # full authoritative view
    assert result.factual_memory.startswith("我派")                    # first person
    # Bystanders see him call the man over, say a few words, and the man leave, but not what he was
    # told.
    assert result.observations
    seen = result.observations[0].text
    assert "今夜勿出" not in seen and "宫门" not in seen
    # That gap is the layer only a close watcher catches: what bystanders miss, someone hiding
    # nearby can hear, so it's explicitly granted (see COVERTABLE_ACTION_TYPES).
    assert result.happening == result.outcome


@pytest.mark.asyncio
async def test_a_body_already_out_on_one_cannot_be_sent_again() -> None:
    """Whether he's free is a rule, not a judgment call: an objective check on structured facts,
    rejected on the spot.

    The rejection is a real world event (remembered as usual), not an ``adjudication_failed`` null
    step.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    # Already sent by someone else. assign_errand requires the requester to stand on a real
    # location, so place a2 first.
    assert env.assign_errand(ErrandOrder(npc_id, "gate"), requester_id="a2")

    result = (await _run_errand(env, ErrandOrder(npc_id, "gate"), actor_id="a1"))[0]
    assert result.succeeded is False
    assert result.adjudication_failed is False        # a real event, not a null step
    assert result.errand_orders == []                 # no second errand appears in the world
    assert "抽不开身" in result.failure_reason
    assert result.factual_memory                      # he still remembers being foiled this step


@pytest.mark.asyncio
async def test_two_people_cannot_send_the_same_body_on_the_same_beat() -> None:
    """Two requesters for the same body in one step: the later one is foiled immediately, not
    rejected afterwards.

    Check and claim must be one operation; otherwise both see him free, the later one is rejected
    only at commit, after his success outcome and first-person memory are already written.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    executor = ErrandExecutor(LiveWorldDirectory.from_agents({}, env))

    # _enact awaits start sequentially; follow that order here.
    states = [
        await executor.start(
            _action(aid, npc_id, ErrandOrder(npc_id, "gate")), 1,
            agents={}, environment=env, message_system=None,
        )
        for aid in ("a1", "a2")
    ]
    results = [
        (await executor.complete(st, 1, agents={}, environment=env, message_system=None))[0]
        for st in states
    ]
    first, second = results

    assert first.succeeded is True and first.errand_orders
    assert second.succeeded is False and second.errand_orders == []
    assert "抽不开身" in second.failure_reason
    assert second.adjudication_failed is False        # a real event, not a null step
    # The later requester's narrative must say it didn't happen.
    assert "未能成行" in second.outcome and second.not_executed is True
    # The world holds only one errand, owned by whoever came first.
    assert env.get_npc(npc_id).errand.requester_id == "a1"


@pytest.mark.asyncio
async def test_a_dispatch_already_made_can_be_frozen_but_not_cancelled() -> None:
    """The errand is settled at ``start``, so knocking him out in the same step can only freeze it,
    not cancel it.

    Two facts each pin half of this: the commit phase runs after start (a condition can't get to him
    first), and a restrained NPC still owes the errand without moving (see why ``NpcEffect`` has no
    "abort errand").
    """
    from core.interfaces.action import NpcEffect

    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    executor = ErrandExecutor(LiveWorldDirectory.from_agents({}, env))
    action = _action("a1", npc_id, ErrandOrder(npc_id, "gate"))

    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
    assert env.get_npc(npc_id).errand is not None      # the errand is already in the world

    # Someone else pins him in the same step; npc_effect runs in the commit phase, after the line
    # above.
    env.apply_npc_effect(NpcEffect(
        npc_id=npc_id, condition_set=BodyCondition(description="被按在地上", since_step=1),
    ))
    result = (await executor.complete(
        state, 1, agents={}, environment=env, message_system=None,
    ))[0]

    assert result.succeeded is True
    errand = env.get_npc(npc_id).errand
    assert errand is not None and errand.requester_id == "a1"   # not cancelled
    assert env.get_npc(npc_id).condition is not None            # only frozen

    runner, _ = _runner(env)
    await runner.advance(1)
    assert env.get_body_location(npc_id) == "hall"              # frozen = doesn't set off


@pytest.mark.asyncio
async def test_a_body_that_cannot_move_cannot_be_sent() -> None:
    """A restrained NPC is handled by rule too: it isn't unwilling, it can't move."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.get_npc(npc_id).condition = BodyCondition(description="被按在地上", since_step=1)

    result = (await _run_errand(env, ErrandOrder(npc_id, "gate"), actor_id="a1"))[0]
    assert result.succeeded is False and "动弹不得" in result.failure_reason
    assert result.errand_orders == []


@pytest.mark.asyncio
async def test_a_bearer_that_vanished_between_the_beats_fabricates_nothing() -> None:
    """He vanished between start and complete: a structurally valid null step, with no invented "he
    set off"."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    executor = ErrandExecutor(LiveWorldDirectory.from_agents({}, env))
    action = _action("a1", npc_id, ErrandOrder(npc_id, "gate"))
    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
    del env._npcs[npc_id]                                    # noqa: SLF001

    result = (await executor.complete(
        state, 1, agents={}, environment=env, message_system=None,
    ))[0]
    assert result.adjudication_failed is True
    assert result.succeeded is False and result.errand_orders == []
    assert result.factual_memory == ""


@pytest.mark.asyncio
async def test_telling_someone_costs_the_teller_exactly_one_beat() -> None:
    """The point of sending someone is not having to wait: the actor's step estimate is ignored and
    fixed at 1."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    executor = ErrandExecutor(LiveWorldDirectory.from_agents({}, env))
    action = _action("a1", npc_id, ErrandOrder(npc_id, "gate"), estimated_steps=9)
    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)

    assert state.estimated_steps == 1
    assert state.remaining_steps == 0            # born at zero: completes the same step
    assert state.participant_ids == ["a1"]       # takes nobody else's turn


# ---------------------------------------------------------------------------
# Progress: walk, do, look, return, report


@pytest.mark.asyncio
async def test_going_to_look_and_coming_back_to_say_so() -> None:
    """The minimal errand: destination only. Looking and reporting are the fixed start and end, not
    optional deeds."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")

    await runner.advance(1)
    # Two steps of walking per step gets him straight there, and he does the task on arrival: looks,
    # finishes, turns back, still standing where he did it.
    assert env.get_body_location(npc_id) == "gate"
    assert env.get_npc(npc_id).errand.outbound is False
    assert env.get_npc(npc_id).errand.seen
    await runner.advance(2)

    assert env.get_body_location(npc_id) == "hall"      # walks back and reports the same step
    assert env.get_npc(npc_id).errand is None
    delivered = await messages.collect(step=3)
    assert len(delivered) == 1
    assert delivered[0].sender_id == npc_id
    assert delivered[0].recipients == ["a1"]
    assert "宫门" in delivered[0].content


@pytest.mark.asyncio
async def test_a_bare_look_brings_back_both_what_is_there_and_what_just_happened() -> None:
    """A "go take a look" errand brings back both what's there and what just happened there.

    The second is what moves things along ("two guards just argued at the palace gate" vs "two
    people are standing at the palace gate"). He excludes himself, or it reads "I saw myself".
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    env.register_entity(WorldEntity(
        entity_id="banner", name="旌旗", entity_type=WorldEntityType.ITEM,
        presence_ref="gate", is_takeable=True,
    ))
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate"), requester_id="a1")
    # He looks on the step he reaches the gate, and ambient carries over to the next step, so this
    # event is recorded on the previous step, same as anyone present would read it: he brings back
    # what just happened.
    env.record_carry_observation(location_id="gate", observation="在宫门，两名守卒起了争执。")
    for step in range(1, 3):
        env.begin_step(step=step, world_time=_world_time(step))
        await runner.advance(step)

    report = (await messages.collect(step=3))[0].content
    assert "宫门" in report and "旌旗" in report          # what's there
    assert "两名守卒起了争执" in report                   # what just happened there
    assert "王二" not in report                          # he isn't news there himself
    # Say so even if nothing happened; a blank would read as "he didn't look".
    assert "刚发生的事" in report


@pytest.mark.asyncio
async def test_it_brings_back_what_it_could_see_and_nothing_more() -> None:
    """He isn't omniscient: what others hide can't be brought back; what's public can."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    env.register_entity(WorldEntity(
        entity_id="banner", name="旌旗", entity_type=WorldEntityType.ITEM,
        presence_ref="gate", is_takeable=True,
    ))
    hidden = WorldEntity(
        entity_id="token", name="暗记", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref="a2", is_takeable=True,
    )
    hidden.is_public = False
    env.register_entity(hidden)

    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")
    await _run_full_errand(runner)

    report = (await messages.collect(step=4))[0].content
    assert "旌旗" in report
    assert "暗记" not in report
    # The narrative layer has no ids and no step numbers.
    assert npc_id not in report and "a2" not in report and "gate" not in report


@pytest.mark.asyncio
async def test_one_addressing_axis_carries_both_the_thing_and_the_word() -> None:
    """With a person named, both the item and the message go to that same person: "who" is one
    addressing axis, not two optional objects."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    env.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
    ))
    runner, messages = _runner(env)
    env.assign_errand(
        ErrandOrder(npc_id, "gate", item_id="letter", recipient_id="a2", message="今夜勿出"),
        requester_id="a1",
    )
    await _run_full_errand(runner)

    assert env.get_entity("letter").owner_id == "a2"
    posted = await messages.collect(step=4)
    said = [m for m in posted if m.recipients == ["a2"]]
    assert said and said[0].content == "今夜勿出"          # verbatim
    report = [m for m in posted if m.recipients == ["a1"]][0].content
    assert "密信已交到" in report and "话已带到" in report


@pytest.mark.asyncio
async def test_with_nobody_named_both_fall_to_the_place() -> None:
    """With no person named, the same axis puts both on the location: the item is left there,
    the message announced there.

    Announcing uses the ``recipients=None + location_scope`` cell of MessageSystem's delivery matrix
    (a location broadcast), not a new mechanism; whether the addressing axis is filled decides the
    cell.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    env.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
    ))
    runner, messages = _runner(env, souls={"a2": SoulLayer(name="阿石", gender="男", age=27)})
    env.assign_errand(
        ErrandOrder(npc_id, "gate", item_id="letter", message="城门今夜不开"),
        requester_id="a1",
    )
    await _run_full_errand(runner)

    letter = env.get_entity("letter")
    assert (letter.owner_id, letter.location_id) == (None, "gate")   # left there
    posted = await messages.collect(step=4)
    shouted = [m for m in posted if m.recipients is None]
    assert shouted and shouted[0].location_scope == "gate"           # the location-broadcast cell
    assert shouted[0].content == "城门今夜不开"
    report = [m for m in posted if m.recipients == ["a1"]][0].content
    assert "已放在" in report and "当着阿石的面说了" in report


@pytest.mark.asyncio
async def test_a_shout_into_an_empty_place_comes_back_as_one() -> None:
    """An errand that shouts into an empty place must not report the same as one heard by people.

    The report is what the requester keeps; claiming delivery would leave him waiting for a reply
    that never comes. The presence check shares its source with SEND_MESSAGE's broadcast.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")          # nobody at all at the gate
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", message="城门今夜不开"), requester_id="a1")
    await _run_full_errand(runner)

    posted = await messages.collect(step=4)
    assert not [m for m in posted if m.recipients is None]       # no broadcast into the void
    report = [m for m in posted if m.recipients == ["a1"]][0].content
    assert "没有人听见" in report


@pytest.mark.asyncio
async def test_a_shout_that_landed_names_who_heard_it() -> None:
    """Name who heard it; that's exactly what this report has to answer later ("who heard this?").

    A vague "announced it publicly" loses the answer, and the requester would assume the person he
    was after was among them.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    runner, messages = _runner(env, souls={"a2": SoulLayer(name="阿石", gender="男", age=27)})
    env.assign_errand(ErrandOrder(npc_id, "gate", message="城门今夜不开"), requester_id="a1")
    await _run_full_errand(runner)

    report = [
        m for m in await messages.collect(step=4) if m.recipients == ["a1"]
    ][0].content
    assert "当着阿石的面说了" in report


@pytest.mark.asyncio
async def test_the_one_who_dictated_it_does_not_hear_it_shouted_back() -> None:
    """The one who dictated a public announcement doesn't receive it, even if he's standing there.

    The mouth and the author are different people: this body speaks, but the words are his.
    Excluding only the sender, he'd receive his own order where he sent the NPC and read the echo as
    someone falsely relaying his words (self-exclusion, see ``Message.actor_ids``).
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a3", location_id="gate")
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", message="城门今夜不开"), requester_id="a1")
    await runner.advance(1)                              # reaches gate and announces the same step
    env.move_body(body_id="a1", location_id="gate")      # he has arrived there too

    delivery = await messages.deliver_for_agents(
        step=2, agent_ids=["a1", "a3"], environment=env,
    )
    assert [m.content for m in delivery.inbox_for("a3")] == ["城门今夜不开"]
    assert delivery.inbox_for("a1") == []


@pytest.mark.asyncio
async def test_naming_someone_with_nothing_to_give_is_going_to_see_if_they_are_there() -> None:
    """A person named but nothing to give or say: what's wanted is whether that person is there.

    This cell must not silently collapse into "go take a look": that would drop exactly what the
    requester wanted to know.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", recipient_id="a2"), requester_id="a1")
    await _run_full_errand(runner)
    assert "不在" in (await messages.collect(step=4))[0].content

    env.place_agent(agent_id="a2", location_id="gate")
    env.assign_errand(ErrandOrder(npc_id, "gate", recipient_id="a2"), requester_id="a1")
    for step in range(4, 7):
        await runner.advance(step)
    report = [m for m in await messages.collect(step=7) if m.recipients == ["a1"]][0].content
    assert "在宫门。" in report


@pytest.mark.asyncio
async def test_a_thing_meant_for_someone_absent_comes_back_rather_than_being_left() -> None:
    """Addressed to someone who isn't there: bring the item back as is; never redirect it to others
    or leave it behind.

    Redirecting would tell a roomful of people what was meant for him alone; leaving it would make a
    decision the requester never made.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a3", location_id="gate")   # a2 isn't at the gate
    env.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
    ))
    runner, messages = _runner(env)
    env.assign_errand(
        ErrandOrder(npc_id, "gate", item_id="letter", recipient_id="a2", message="今夜勿出"),
        requester_id="a1",
    )
    await _run_full_errand(runner)

    assert env.get_entity("letter").owner_id == npc_id   # still in his hands
    posted = await messages.collect(step=4)
    assert not [m for m in posted if m.recipients is None]   # not redirected into a public one
    report = [m for m in posted if m.recipients == ["a1"]][0].content
    assert "没能交出去" in report and "话没能带到" in report


@pytest.mark.asyncio
async def test_it_walks_footpace_not_one_node_per_beat() -> None:
    """He walks by distance, not by node: a heavy edge he can't finish in one step takes several.

    Edge weight is walking time; one hop per step would give him a different physics from agents
    on the same map.
    """
    env = EnvironmentSystem()
    env.space.register_place(_location("hall", "大殿", far=6))     # a long road, six steps of walking
    env.space.register_place(_location("far", "远处", hall=6))
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, _ = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "far"), requester_id="a1")

    # Two steps of walking per step → a six-step edge takes three steps, during which he stays at
    # the origin (never lands on a pseudo-location midway).
    for expected_left in (4, 2):
        await runner.advance(1)
        assert env.get_body_location(npc_id) == "hall"
        assert env.get_npc(npc_id).errand.leg_remaining == expected_left
    await runner.advance(1)
    assert env.get_body_location(npc_id) == "far"        # arrives on the third step
    assert env.get_npc(npc_id).errand.leg_remaining == 0


@pytest.mark.asyncio
async def test_a_half_walked_edge_survives_a_restore() -> None:
    """Saved midway: the remaining distance must be restored, or he'd start the walk over once he
    gets there."""
    env = EnvironmentSystem()
    env.space.register_place(_location("hall", "大殿", far=6))
    env.space.register_place(_location("far", "远处", hall=6))
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, _ = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "far"), requester_id="a1")
    await runner.advance(1)
    assert env.get_npc(npc_id).errand.leg_remaining == 4

    fresh = EnvironmentSystem()
    fresh.space.register_place(_location("hall", "大殿", far=6))
    fresh.space.register_place(_location("far", "远处", hall=6))
    fresh.restore_state(env.snapshot_state())
    assert fresh.get_npc(npc_id).errand.leg_remaining == 4


@pytest.mark.asyncio
async def test_it_does_not_report_seeing_itself() -> None:
    """What he sees may include himself; a scene that counts himself as present reads back as "I saw
    myself"."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")
    await _run_full_errand(runner)

    report = (await messages.collect(step=4))[0].content
    assert "王二" not in report


@pytest.mark.asyncio
async def test_the_thing_changes_hands_when_he_is_sent_off_with_it() -> None:
    """The item to carry changes hands on the spot: it must reach his hands to be delivered.

    This bridges the decision side ("can only choose what I hold") and the runner ("must be in his
    hands"). A broken bridge doesn't error: every carrying errand just reports "couldn't deliver".
    Tests that spawn the item directly in his hands skip it.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    env.register_entity(WorldEntity(
        entity_id="box", name="木匣", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref="a1", is_takeable=True,   # requester holds it
    ))
    order = ErrandOrder(npc_id, "gate", item_id="box", recipient_id="a2")
    result = (await _run_errand(env, order, actor_id="a1"))[0]

    # The executor only declares; applying it belongs to the feedback layer, done by hand here.
    assert [c.entity_id for c in result.entity_state_changes] == ["box"]
    for change in result.entity_state_changes:
        env.change_entity_state(change, acting_agent_id="a1")
    assert env.get_entity("box").presence_ref == npc_id      # now in his hands
    assert "木匣" in result.observations[0].text             # handed over in sight of bystanders

    # So it can actually be delivered at the other end instead of "couldn't deliver".
    env.assign_errand(order, requester_id="a1")
    runner, messages = _runner(env)
    await _run_full_errand(runner)
    report = [m for m in await messages.collect(step=4) if m.recipients == ["a1"]][0].content
    assert "已交到" in report, report
    assert env.get_entity("box").presence_ref == "a2"


@pytest.mark.asyncio
async def test_he_cannot_send_off_a_thing_that_is_not_in_his_own_hands() -> None:
    """You can't send what you don't hold, or it would jump into the errand from someone's hands
    across town.

    The handoff rewrites the holder unconditionally (``change_entity_state`` doesn't ask where the
    item was, which is right for a PHYSICAL seizure), so this gate must be held at commit, not only
    at the decision-side binding.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    env.register_entity(WorldEntity(
        entity_id="box", name="木匣", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref="a2",   # held by someone at the gate
        is_takeable=True,
    ))
    results = await _run_errand(
        env, ErrandOrder(npc_id, "gate", item_id="box"), actor_id="a1",
    )

    assert results[0].succeeded is False
    assert "木匣" in (results[0].failure_reason or "")
    assert env.get_entity("box").presence_ref == "a2"     # didn't move
    assert not results[0].entity_state_changes
    assert env.get_npc(npc_id).errand is None            # and he wasn't sent either


@pytest.mark.asyncio
async def test_walking_through_makes_no_news_but_it_is_still_standing_there() -> None:
    """Passing through is traffic, not an event, so walking produces no ambient.

    Ambient gets only 2 slots per step (main character) / 1 (background), and footsteps would crowd
    out real happenings. Being seen is handled by the presence lists: whoever he stands in front of
    can see him and stop him.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="watch", location_id="gate")
    runner, _ = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")

    await runner.advance(1)                              # hall → yard → gate
    env.begin_step(step=2, world_time=_world_time(2))    # carry → next step's ambient
    for who, where in (("watch", "gate"), ("a1", "hall")):
        heard = [e.content for e in env.spatial_for(agent_id=who).ambient_events]
        assert not any("王二" in text for text in heard), where

    # But he's standing at the gate and can be seen; that's the channel that lets someone stop him.
    assert npc_id in env.spatial_for(agent_id="watch").visible_npcs


@pytest.mark.asyncio
async def test_a_held_body_stops_where_it_stands_and_keeps_the_errand() -> None:
    """A restrained NPC can't do the task, but the errand isn't void: once untied he carries on."""
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, _ = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")
    env.get_npc(npc_id).condition = BodyCondition(description="被按在地上", since_step=1)

    await runner.advance(1)
    assert env.get_body_location(npc_id) == "hall"      # didn't move
    assert env.get_npc(npc_id).errand is not None        # errand still there

    env.get_npc(npc_id).condition = None
    await runner.advance(2)
    assert env.get_body_location(npc_id) == "gate"      # released, he carries on


@pytest.mark.asyncio
async def test_a_place_it_cannot_reach_still_gets_an_answer() -> None:
    """Even when he can't get there there's a report: the requester has already recorded sending
    him, so the world can't act as if nothing happened."""
    env = _world()
    env.space.register_place(_location("island", "孤岛"))       # connected to nothing
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "island", ()), requester_id="a1")

    await runner.advance(1)                               # no route → turns back
    await runner.advance(2)                               # already home → reports
    assert env.get_npc(npc_id).errand is None
    report = (await messages.collect(step=3))[0].content
    assert "到不了" in report


@pytest.mark.asyncio
async def test_every_cell_of_the_space_reads_as_its_own_errand() -> None:
    """Run through all eight cells of the errand, each reading as a different task.

    The capability set isn't enumerated; it's the combination of four axes (see ``ErrandOrder``), so
    the number of errand kinds is 2³, not the length of a table. This test makes that executable:
    no cell may silently collapse into another. Collapse is the quietest kind of loss: the
    requester's intent is gone and nothing in the world shows it.
    """
    cells = [
        # (carries item, names person, carries message) → the line that must appear in the report
        ((False, False, False), "刚发生的事"),   # "go look": what he saw comes back
        ((False, False, True),  "当着"),
        ((False, True,  False), "在宫门。"),
        ((False, True,  True),  "话已带到"),
        ((True,  False, False), "已放在宫门"),
        ((True,  False, True),  "当着"),
        ((True,  True,  False), "已交到"),
        ((True,  True,  True),  "已交到"),
    ]
    for i, ((carry, name, speak), expected) in enumerate(cells):
        env = _world()
        npc_id = _bearer(env)
        env.place_agent(agent_id="a1", location_id="hall")
        env.place_agent(agent_id="a2", location_id="gate")
        if carry:
            env.register_entity(WorldEntity(
                entity_id="thing", name="木匣", entity_type=WorldEntityType.ITEM,
                presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
            ))
        runner, messages = _runner(env)
        env.assign_errand(ErrandOrder(
            npc_id, "gate",
            item_id="thing" if carry else "",
            recipient_id="a2" if name else "",
            message="传一句话" if speak else "",
        ), requester_id="a1")
        await _run_full_errand(runner)

        report = [
            m for m in await messages.collect(step=4) if m.recipients == ["a1"]
        ][0].content
        assert expected in report, f"第{i + 1}格({carry},{name},{speak}) 读不出「{expected}」：\n{report}"


def test_the_judge_and_the_runner_read_the_same_errand_the_same_way() -> None:
    """What the judge reads must be exactly what the runner will do.

    The judge uses the order to decide acceptance, the runner to carry it out; interpreted
    separately, "just go look" gets approved and something else gets done. Compared cell by cell.
    """
    from engine.executors.errand import _render_errand

    env = _world()
    npc_id = _bearer(env)
    env.register_entity(WorldEntity(
        entity_id="thing", name="木匣", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
    ))
    directory = LiveWorldDirectory.from_agents({}, env)

    # The phrase the judge side must name in each cell, matching the runner-side test cell by cell.
    for (carry, name, speak), must_say in [
        ((False, False, False), "去宫门走一趟"),   # destination-only is a task too
        ((False, False, True),  "宫门当众"),
        ((False, True,  False), "在不在宫门"),
        ((False, True,  True),  "带一句话"),
        ((True,  False, False), "放在宫门"),
        ((True,  False, True),  "宫门当众"),
        ((True,  True,  False), "交到"),
        ((True,  True,  True),  "交到"),
    ]:
        rendered = " / ".join(_render_errand(
            ErrandOrder(
                npc_id, "gate",
                item_id="thing" if carry else "",
                recipient_id="a2" if name else "",
                message="传一句话" if speak else "",
            ),
            environment=env, directory=directory,
        ))
        assert must_say in rendered, f"({carry},{name},{speak}) 判官读到「{rendered}」，缺「{must_say}」"


# ---------------------------------------------------------------------------
# Counterweight: he can be stopped


@pytest.mark.asyncio
async def test_a_body_can_be_held_down_without_losing_what_it_was_sent_to_do() -> None:
    """Otherwise sending someone is a risk-free remote arm; but stopping him doesn't cancel the
    errand on the requester's behalf.

    Consequences go only through ``npc_effects``: no ``TargetAgentEffect``, no relation_updates. He
    has no emotion, relations, memory or vitality; writing them would just be dropped by the
    feedback layer, and the relation delta would put an id outside the relation namespace into the
    actor's relation table.
    """
    from core.interfaces.action import ActionType as _AT
    from engine.executors.physical import PhysicalExecutor

    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    # Someone else sent him (a1 is the one stopping him). The requester must really be in the
    # world, or assign_errand silently returns False and this test checks "is the errand still
    # there" for an errand never sent.
    env.place_agent(agent_id="a2", location_id="hall")
    assert env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a2")

    llm = _FixedLLM(
        '{"reason":"他手无寸铁，被当场按住","success":true,"deed":"restrain",'
        '"outcome":"甲一把按住王二","fact":"我把他按在了地上",'
        '"target_condition":"被死死按住","target_condition_steps":0}'
    )
    directory = LiveWorldDirectory.from_agents({}, env)
    executor = PhysicalExecutor(llm, directory)
    action = AgentAction(
        agent_id="a1", step=1, action_type=_AT.PHYSICAL,
        action_description="按住那个正要出门的人",
        target=ActionTarget(acts_on=[Ref.npc(npc_id)]),
        estimated_steps=1,
    )
    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
    results = await executor.complete(
        state, 1, agents={"a1": _StubAgent("a1")}, environment=env, message_system=None,
    )

    result = results[0]
    assert result.target_effects == []          # he has no inner state to write
    assert result.relation_updates == []        # and isn't in the relation namespace
    assert len(result.npc_effects) == 1
    effect = result.npc_effects[0]
    assert effect.npc_id == npc_id
    assert effect.condition_set.description == "被死死按住"

    # After commit: he stays put, can't take new errands, and what he holds can be seized (via the
    # existing item channel). But the errand is still there: one interception isn't a permanent
    # cancel.
    env.apply_npc_effect(effect)
    assert env.get_npc(npc_id).errand is not None
    assert env.get_body_location(npc_id) == "hall"
    assert not env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")


@pytest.mark.asyncio
async def test_a_body_that_cannot_struggle_is_not_held_for_good() -> None:
    """The judge's 0 means "someone else must free him or he breaks free", and a body with no
    cognition never breaks free.

    So for him 0 can't mean indefinite, or one interception would erase a body and its errand for
    good. The fallback is half a day, converted at one hour per step.
    """
    from core.interfaces.action import ActionType as _AT
    from engine.executors.physical import PhysicalExecutor

    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    llm = _FixedLLM(
        '{"reason":"他手无寸铁，被当场按住","success":true,"deed":"restrain",'
        '"outcome":"甲一把按住王二","fact":"我把他按在了地上",'
        '"target_condition":"被死死按住","target_condition_steps":0}'
    )
    executor = PhysicalExecutor(
        llm, LiveWorldDirectory.from_agents({}, env), seconds_per_step=3600,
    )
    action = AgentAction(
        agent_id="a1", step=1, action_type=_AT.PHYSICAL,
        action_description="按住那个正要出门的人",
        target=ActionTarget(acts_on=[Ref.npc(npc_id)]),
        estimated_steps=1,
    )
    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
    results = await executor.complete(
        state, 1, agents={"a1": _StubAgent("a1")}, environment=env, message_system=None,
    )

    condition = results[0].npc_effects[0].condition_set
    assert condition.until_step == 1 + 12          # half a day ÷ one hour per step

    # The world actually releases him; the fallback isn't a write-only number.
    env.apply_npc_effect(results[0].npc_effects[0])
    assert not env.expire_npc_condition(npc_id, 12)
    assert env.expire_npc_condition(npc_id, 13)
    assert env.get_npc(npc_id).condition is None


# ---------------------------------------------------------------------------


def _action(actor_id: str, npc_id: str, order: ErrandOrder, *, estimated_steps: int = 1):
    from agent.decision import ActionIntent

    return AgentAction(
        agent_id=actor_id,
        step=1,
        action_type=ActionType.ERRAND,
        action_description="请他去宫门看看守备",
        target=ActionTarget(acts_on=[Ref.npc(npc_id)]),
        estimated_steps=estimated_steps,
        intent=ActionIntent(purpose="请他去宫门看看守备", errand=order),
    )


async def _run_errand(env, order: ErrandOrder, *, actor_id: str):
    """Run one assignment through the whole executor lifecycle (start → complete). No LLM, so no
    stubs needed."""
    executor = ErrandExecutor(LiveWorldDirectory.from_agents({}, env))
    action = _action(actor_id, order.npc_id, order)
    state = await executor.start(action, 1, agents={}, environment=env, message_system=None)
    return await executor.complete(
        state, 1, agents={}, environment=env, message_system=None,
    )


class _StubAgent:
    """The PHYSICAL judge reads the actor's ``personality`` and nothing else."""

    def __init__(self, agent_id: str) -> None:
        from agent.personality import PersonalityLayer, SoulLayer

        self.agent_id = agent_id
        self.personality = PersonalityLayer(soul=SoulLayer(
            name="甲", role="将军", agent_id=agent_id,
            core_traits=["果决"], core_values=["秩序"], hard_constraints=[],
        ))


@pytest.mark.asyncio
async def test_the_report_says_where_the_words_end_and_the_scene_begins() -> None:
    """A report-back has two parts: the line spoken and the scene attached. The split is made
    here and travels with the message.

    Downstream uses differ: memory keeps only the first (the scene is stale by next step, and
    hundreds of characters would push out what matters), and the speech bubble only fits the first.
    Making each consumer guess the cut by length means recomputing a fact already known here.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, messages = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")
    for step in (1, 2, 3):
        await runner.advance(step)

    report = (await messages.collect(step=4))[0]
    spoken = report.metadata["spoken"]
    assert spoken and spoken != report.content
    assert report.content.startswith(spoken)     # spoken line first, then attachment
    assert "那里的情形是这样" in report.content   # the scene is in the full text
    assert "那里的情形是这样" not in spoken       # but not in the spoken line


def test_the_wire_carries_both_halves_and_an_ordinary_letter_has_only_one() -> None:
    """The read model exposes the split as fields: ``spoken`` and ``perceived_summary``.

    The display layer wants the short one and the full one, and gets both directly. An ordinary
    letter is all words with no attachment, so the two are equal; every view just reads its own
    field without first checking whether this is a report-back.
    """
    from core.interfaces.snapshot import WorldSnapshot
    from interaction.models import StepEvent

    def _wire(mid: str, content: str, metadata: dict) -> dict:
        return {
            "id": mid, "sender_id": "s", "sender_name": "某人", "content": content,
            "recipients": ["a1"], "location_scope": None, "metadata": metadata,
        }

    report = "我去了宫门一趟，回来了。\n那里的情形是这样：\n- 在场的其他人：无"
    snapshot = WorldSnapshot(
        world_id="w", step=1, timestamp=None, world_time={}, agent_states={},
        metadata={"messages": {
            "delivered": [
                _wire("m1", report, {"spoken": "我去了宫门一趟，回来了。"}),
                _wire("m2", "你我明日午时相见。", {}),
            ],
            "inboxes": {"a1": ["m1", "m2"]},
        }},
    )
    seen = StepEvent.from_snapshot(snapshot).messages
    assert seen[0].spoken == "我去了宫门一趟，回来了。"
    assert seen[0].perceived_summary == report
    # An ordinary letter: nothing declared, so both are equal.
    assert seen[1].spoken == seen[1].perceived_summary == "你我明日午时相见。"


@pytest.mark.asyncio
async def test_the_judge_is_told_the_body_it_judges_has_no_will() -> None:
    """The judge's target section must state what this tier is, not just a person-shaped header.

    Given only name/gender/age/blurb, the judge gives a body with no cognition a stance ("the
    soldier is unmoved"), and that ruling is written into both sides' memories. It states the tier,
    not "therefore it always succeeds": whether he can be held down is still judged physically.
    """
    from engine.executors.base import ActionExecutionState
    from engine.executors.physical import PhysicalExecutor

    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    state = ActionExecutionState.create(
        action_type=ActionType.PHYSICAL, initiator_id="a1", participant_ids=["a1"],
        purpose="按住他", started_step=1, estimated_steps=1, opening_outcome="",
        target=ActionTarget(acts_on=[Ref.npc(npc_id)]),
    )
    executor = PhysicalExecutor(_FixedLLM("{}"), LiveWorldDirectory.from_agents({}, env))
    subject = await executor._resolve_subject(  # noqa: SLF001
        state, agents={}, environment=env, step=1,
    )
    assert "王二" in subject.target_block             # still rendered as a person
    assert "没有自己的认知" in subject.target_block     # but the tier is stated
    assert "一概不会发生" in subject.target_block       # the part about understanding/agreeing/complying
    assert "就是没成" in subject.target_block           # and if that's what the act was after, it fails
    # Don't create a separate ruling for him: actions on him are judged as usual. Listing examples
    # of what can be judged would push the unlisted ones (supporting him, shielding him, standing in
    # front of him) out of the judge's view.
    assert "照常裁" in subject.target_block


@pytest.mark.asyncio
async def test_each_step_says_what_happened_to_him_where_he_stands() -> None:
    """The observer side needs to know what happened to him this step: one line per step,
    written by the code that did it.

    The line has no location because it always happens where he stands when the step ends; that's
    what the same-place invariant buys (the next test pins it step by step). The observer joins the
    body location table to get the place, with no second copy.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    runner, _ = _runner(env, souls={
        "a1": SoulLayer(name="阿墨", gender="男", age=24),
        "a2": SoulLayer(name="阿石", gender="男", age=27),
    })
    env.assign_errand(
        ErrandOrder(npc_id, "gate", (), recipient_id="a2", message="今夜勿出"),
        requester_id="a1",
    )

    seen = []
    for step in (1, 2):
        env.begin_step(step=step, world_time=_world_time(step))
        await runner.advance(step)
        seen.append((
            env.snapshot_state()["npc_outcomes"].get(npc_id),
            env.get_body_location(npc_id),
        ))

    # He reaches the gate and delivers the message that step, so he's at the gate: what's said and
    # where he stands match.
    assert seen[0] == ({"text": "给阿石带了句话：今夜勿出", "ongoing": False}, "gate")
    # He walks back to the hall and reports that step, so he's in the hall.
    assert seen[1] == ({"text": "回来向阿墨回了话", "ongoing": False}, "hall")

    # Each line belongs to its own step: begin_step clears the previous one.
    env.begin_step(step=3, world_time=_world_time(3))
    assert env.snapshot_state()["npc_outcomes"] == {}


@pytest.mark.asyncio
async def test_a_trip_with_nothing_to_deliver_still_says_he_got_there() -> None:
    """The destination-only errand: nothing to deliver, but that step must still have a line.

    He did get there and did look (looking is a fixed part of every errand). An empty line here
    would remove both the feed row and the floating text on the map, and viewers would see someone
    walk over and back with nothing in between.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, _ = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")
    env.begin_step(step=1, world_time=_world_time(1))
    await runner.advance(1)

    assert env.snapshot_state()["npc_outcomes"][npc_id] == {"text": "看了一眼", "ongoing": False}


@pytest.mark.asyncio
async def test_one_visit_can_get_two_things_done() -> None:
    """The last cell of the table: hand over the item and deliver the message; both must be
    said.

    The combined line is written by the code that actually does the task, since only it knows these
    are two halves of one visit, and that the recipient is one person, so the name appears once;
    twice reads like two people.
    """
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="gate")
    letter = WorldEntity(
        entity_id="e1", name="书信", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
    )
    env.register_entity(letter)
    runner, _ = _runner(env, souls={
        "a1": SoulLayer(name="阿墨", gender="男", age=24),
        "a2": SoulLayer(name="阿石", gender="男", age=27),
    })
    env.assign_errand(
        ErrandOrder(npc_id, "gate", item_id="e1", recipient_id="a2", message="今夜勿出"),
        requester_id="a1",
    )
    env.begin_step(step=1, world_time=_world_time(1))
    await runner.advance(1)

    assert env.snapshot_state()["npc_outcomes"][npc_id] == {
        "text": "把书信交给了阿石，又带了句话：今夜勿出", "ongoing": False,
    }


def test_a_runner_s_float_is_raised_when_his_own_walk_lands() -> None:
    """The floating text has two strict ordering constraints, each avoiding a silent failure.

    After movement: it floats above his head, so his position must include this step's walk, or it
    hangs over last step's room. After ``clearEphemeral``: floating text goes through
    ``pushEphemeral``, whose pool that call clears; ordered before it, every line is cleared in the
    same frame and never shown. The track runs as a whole after the clear, so this one is pinned on
    the ``draw`` side.
    """
    from pathlib import Path

    src = Path("frontend/src/phaser/TiledWorldScene.ts").read_text()
    track = src[src.index("private async playTrack("):]
    assert track.index("this.noteNpcOutcomes(") > track.index("this.moveAgent("), (
        "浮在走位之前,会挂到他上一拍站的那间屋子上"
    )
    draw = src[src.index("private async draw("):]
    assert draw.index("this.partition(") > draw.index("this.clearEphemeral()"), (
        "轨道排在清扫之前,浮字会被当场扫掉"
    )


def test_a_float_wraps_instead_of_running_across_the_map() -> None:
    """Floating text must wrap: its content has no length limit (the whole message he's carrying is
    in it).

    Unwrapped, a 54-character order at 12px is 650px wide, stretched across half the city. Width is
    fixed and height grows, with the bottom edge aligned to the anchor (centered, extra lines would
    grow downward over the person it labels).
    """
    from pathlib import Path

    src = Path("frontend/src/phaser/hud.ts").read_text()
    body = src[src.index("floatNote(x: number"):]
    body = body[:body.index("\n  }")]
    assert "wordWrap" in body, "浮字不换行,长内容会横穿地图"
    assert "setOrigin(0.5, 1)" in body, "浮字不是底边对齐,多出来的行会盖住被标注的人"


@pytest.mark.asyncio
async def test_the_legs_he_walks_are_the_lines_the_reader_is_told() -> None:
    """Run two real errands (one hop, two hops) and pin step by step: timing, the same-place
    invariant, and when the map speaks.

    Timing: the assignment takes one step (he doesn't move), then each leg's task is done on the
    step that leg ends: 1 + T + T steps. The extra step comes first, when only the bystanders who
    heard the order have a reason to stop him.

    Same-place: whenever a step has a line, he stands where it happened; otherwise the feed and the
    map disagree by a whole leg on the same step.
    """
    for hops, legs in ((1, ["gate"]), (2, ["far"])):
        env = _world()
        if hops == 2:
            # hall—yard—gate—far: two steps of walking per step, so the outbound trip takes two
            # steps.
            env.space.register_place(_location("far", "远处", gate=1))
            env.space.register_place(_location("gate", "宫门", yard=1, far=1))
        npc_id = _bearer(env)
        env.place_agent(agent_id="a1", location_id="hall")
        runner, _ = _runner(env, souls={"a1": SoulLayer(name="阿墨", gender="男", age=24)})
        dest = legs[0]

        env.begin_step(step=1, world_time=_world_time(1))
        env.assign_errand(ErrandOrder(npc_id, dest, message="今夜勿出"), requester_id="a1")
        await runner.advance(1)
        # The assignment step: he doesn't move or speak; this step's event is in the requester's
        # line.
        assert env.get_body_location(npc_id) == "hall"
        assert env.snapshot_state()["npc_outcomes"] == {}

        seen: list[tuple[str, str, bool]] = []
        for step in range(2, 2 + 2 * hops):
            env.begin_step(step=step, world_time=_world_time(step))
            await runner.advance(step)
            line = env.snapshot_state()["npc_outcomes"].get(npc_id, {})
            seen.append((
                env.get_body_location(npc_id),
                str(line.get("text", "")),
                bool(line.get("ongoing", False)),
            ))

        going, home = "正往{}去当众传句话", "回来向阿墨回了话"
        empty = "传了句话，但此处无人，没有人听见：今夜勿出"
        if hops == 1:
            assert seen == [
                ("gate", empty, False),                  # arrives and does it that step
                ("hall", home, False),               # walks back + reports that step
            ]
        else:
            assert seen == [
                ("gate", going.format("远处"), True),    # still on the road: no result yet
                ("far", empty, False),
                ("yard", "正往回走", True),
                ("hall", home, False),
            ]
        assert env.get_npc(npc_id).errand is None        # reported, errand cleared


@pytest.mark.asyncio
async def test_a_trip_that_covered_no_ground_still_announces_what_he_did() -> None:
    """He speaks because he got something done, not because he walked, so he speaks even without
    taking a step.

    In degenerate trips "he's at the destination" holds without a single step (the destination is
    this room, or there's no route and he turns back at once). What he did still happened where he
    stands, so floating text above him is right: it labels the deed, not the trip.
    """
    # (1) The destination is this room: he delivered the message without taking a step.
    env = _world()
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")
    runner, _ = _runner(env, souls={"a2": SoulLayer(name="阿石", gender="男", age=27)})
    env.assign_errand(
        ErrandOrder(npc_id, "hall", recipient_id="a2", message="今夜勿出"), requester_id="a1",
    )
    env.begin_step(step=1, world_time=_world_time(1))
    await runner.advance(1)
    assert env.snapshot_state()["npc_outcomes"][npc_id] == {
        "text": "给阿石带了句话：今夜勿出", "ongoing": False,
    }
    assert env.get_body_location(npc_id) == "hall"

    # (2) No route: he never left. The wasted trip must be said; otherwise it looks identical to a
    #     success on the observer side, while the requester has already recorded sending him.
    env = EnvironmentSystem()
    env.space.register_place(_location("hall", "大殿"))
    env.space.register_place(_location("gate", "宫门"))
    npc_id = _bearer(env)
    env.place_agent(agent_id="a1", location_id="hall")
    runner, _ = _runner(env)
    env.assign_errand(ErrandOrder(npc_id, "gate", message="今夜勿出"), requester_id="a1")
    env.begin_step(step=1, world_time=_world_time(1))
    await runner.advance(1)
    assert env.snapshot_state()["npc_outcomes"][npc_id] == {"text": "去不了宫门", "ongoing": False}


def test_a_restrained_person_is_not_reported_as_free() -> None:
    """In the scene an NPC brings back, the condition of those present must be included; it goes
    into the report verbatim and into the requester's memory.

    ``condition`` exists only on a live ``Agent`` (see the agents contract of
    assemble_scene_context). Without passing agents down, someone tied up is rendered as free: a
    persistent narrative that contradicts the world, with no error.
    """
    from types import SimpleNamespace

    env = _world()
    npc_id = _bearer(env, at="gate")
    env.place_agent(agent_id="a1", location_id="gate")
    soul = SoulLayer(name="李建成", role="太子", agent_id="a1", age=37, gender="男")
    bound = SimpleNamespace(personality=SimpleNamespace(
        soul=soul,
        state=SimpleNamespace(condition=BodyCondition(description="双手被反绑", since_step=1)),
    ))

    runner, _ = _runner(env, souls={"a1": soul})
    seen_without = runner._look(env.get_npc(npc_id))              # noqa: SLF001
    seen_with = runner._look(env.get_npc(npc_id), {"a1": bound})  # noqa: SLF001

    assert "李建成" in seen_with
    assert "双手被反绑" in seen_with, "带上 agents 才看得见处境"
    assert "双手被反绑" not in seen_without, "这条测试若失去区分力,说明处境换了来源"


def test_the_report_carries_no_judge_premise() -> None:
    """The ordinary-fixtures line tells a judge what's at hand; it isn't something the bearer saw,
    so it stays out of what he relays."""
    from engine.scene import _ORDINARY_FIXTURES_LINE, SceneVisibility, assemble_scene_context

    env = _world()
    npc_id = _bearer(env, at="gate")
    runner, _ = _runner(env)

    report = runner._look(env.get_npc(npc_id))  # noqa: SLF001
    assert "- 现场物件：" in report
    assert _ORDINARY_FIXTURES_LINE not in report
    judged = assemble_scene_context(
        npc_id, environment=env, directory=runner._directory,  # noqa: SLF001
        visibility=SceneVisibility.OWN_EYES,
    ).text
    assert _ORDINARY_FIXTURES_LINE in judged
