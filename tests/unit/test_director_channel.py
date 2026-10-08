"""DirectorChannel contract tests: the entry point through which a human author reaches into the
world.

Two main lines:
- Validate on submit: if nothing actionable can be parsed, reject on the spot and inject nothing
  (doing nothing beats doing it badly).
- Commit is pure dispatch: the queue holds validated plans, and drain never touches the LLM.

Plus the asymmetry between the director and the LLM event editor: no quota, no pacing gate, and it
can change world state, but it never decides an agent's actions for them.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import (
    EmotionState,
    EmotionType,
    PersonalityLayer,
    SoulLayer,
    StateLayer,
)
from agent.relation import RelationSystem
from core.context import set_active_call
from core.interfaces.action import ErrandOrder
from core.interfaces.llm import LLMMessage, LLMResponse, LLMRouter, LLMScene
from core.interfaces.phenomenon import Phenomenon
from engine.broadcast import BroadcastChannel
from engine.clock import WorldTime, WorldTimeConfig
from engine.directory import LiveWorldDirectory
from engine.director import DirectorChannel, NpcOnMenu
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.injection import Author, InjectionDispatcher, serialize_world_event
from engine.message_system import MessageSystem
from engine.world_mutation import (
    ConditionMutation, EntityMutation, RelocateMutation, SpawnMutation, WorldMutationChannel,
)
from providers.llm.mock import MockLLMProvider
from world.models import EntityPresence, NpcSeed, WorldEntity, WorldEntityType
from core.interfaces.place import Place

WORLD_ID = "world-1"


class _Router:
    """Router stub: returns preset responses in call order and records the prompts it receives for
    assertions."""

    def __init__(self, *responses: str) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[LLMScene, list[LLMMessage]]] = []

    async def complete(self, scene, messages, temperature=0.7, max_tokens=1000, **kwargs):
        self.calls.append((scene, list(messages)))
        return LLMResponse(
            content=next(self._responses, "{}"), input_tokens=0, output_tokens=0, model="t",
        )

    def joined(self, index: int = 0) -> str:
        return "\n".join(m.content for m in self.calls[index][1])


def _plan_json(**overrides) -> str:
    payload = {
        "reason": "导演要在西市放一把火",
        "feasible": True,
        "refusal": "",
        "broadcast": None,
        "message": None,
        "mutations": [],
        "narrative_desc": "西市起火",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def _make_agent(
    container, *, agent_id: str, name: str, location: str,
    role: str = "臣", age: int = 28, gender: str = "男", background: str = "",
) -> Agent:
    router = LLMRouter({scene: MockLLMProvider() for scene in LLMScene})
    return Agent(
        world_id=WORLD_ID,
        agent_id=agent_id,
        personality=PersonalityLayer(
            soul=SoulLayer(name=name, agent_id=agent_id, role=role,
                           age=age, gender=gender, background=background,
                           core_traits=("谨慎",), core_values=("家族",)),
            state=StateLayer(
                agent_id=agent_id, step=1,
                emotion=EmotionState(primary=EmotionType.NEUTRAL, intensity=0.3, valence=0.0),
                current_location=location, vitality=1.0,
            ),
        ),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router, container.embedding, container.vector_store,
            world_id=WORLD_ID, agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(container.agent_store, world_id=WORLD_ID, agent_id=agent_id),
        agent_store=container.agent_store,
        is_main_character=False,
    )


def _world(container, router):
    environment = EnvironmentSystem()
    for eid, name, desc in (
        ("palace", "太极宫", "宫城正殿，天子听政之所"),
        ("market", "西市", "胡商云集的西面市集"),
    ):
        environment.space.register_place(Place(
            place_id=eid, name=name, description=desc,
        ))
    environment.register_entity(WorldEntity(
        entity_id="gate", name="玄武门", entity_type=WorldEntityType.LANDMARK,
        state="intact", description="宫城北门，禁军屯守之要冲",
        presence=EntityPresence.AT_LOCATION, presence_ref="palace",
    ))
    agents = {
        "a1": _make_agent(container, agent_id="a1", name="李世民", location="palace",
                          role="秦王", age=28, gender="男",
                          background="李渊次子，自幼随父起兵，少年统军平定四方。"),
        "a2": _make_agent(container, agent_id="a2", name="李建成", location="market",
                          role="太子", age=37, gender="男",
                          background="李渊长子，久居长安留守理政。"),
    }
    for aid, agent in agents.items():
        environment.place_agent(agent_id=aid, location_id=agent.personality.state.current_location)

    registry = ActionExecutorRegistry()
    message_system = MessageSystem(container.message_provider, world_id=WORLD_ID)
    broadcast_channel = BroadcastChannel()
    directory = LiveWorldDirectory.from_agents(agents, environment)
    mutation_channel = WorldMutationChannel(
        environment=environment,
        seconds_per_step=3600,
        processor=ExecutionProcessor(
            executor_registry=registry, environment=environment,
            message_system=message_system, directory=directory,
        ),
    )
    channel = DirectorChannel(
        llm_router=router,
        dispatcher=InjectionDispatcher(
            broadcast_channel=broadcast_channel,
            message_system=message_system,
            mutation_channel=mutation_channel,
            directory=directory,
        ),
        directory=directory,
    )
    return channel, agents, environment, broadcast_channel, message_system


def _watch_adoption():
    """Put a fake "current LLM call" into the contextvar. In production LLMRouter sets it; _Router
    is a stub."""
    call = SimpleNamespace(adopted=None, reject_reason="")
    set_active_call(call)
    return call


def _time(step: int = 1) -> WorldTime:
    return WorldTime.from_step(step, WorldTimeConfig(start_hour=6, seconds_per_step=60))


def _npc_menu(environment):
    """Does what ``_director_inputs`` does in production: puts the errand-runners and their current
    locations into the menu."""
    return [
        NpcOnMenu(npc=npc, location_id=environment.get_body_location(npc.npc_id))
        for npc in environment.all_npcs()
    ]


async def _submit(channel, text, agents, environment):
    """The channel doesn't hold EnvironmentSystem. The selection menu is passed in by the caller
    (the orchestration layer in production)."""
    return await channel.submit(
        text,
        all_agents=agents,
        npcs=_npc_menu(environment),
        locations=environment.space.all_places(),
        entities=environment.all_live_entities(),
        world_time=_time(),
    )


# ---------------------------------------------------------------------------
# Submit: rejection is first-class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blank_directive_is_refused_without_touching_the_llm(container) -> None:
    router = _Router()
    channel, agents, environment, *_ = _world(container, router)
    result = await _submit(channel, "   ", agents, environment)
    assert result.accepted is False
    assert router.calls == []          # an empty directive shouldn't cost a single call


@pytest.mark.asyncio
async def test_llm_refusal_is_relayed_verbatim_and_nothing_is_queued(container) -> None:
    """"Doing nothing beats doing it badly" needs a place to say no. This is it, and the reason goes
    back to the director verbatim."""
    router = _Router(_plan_json(
        feasible=False, refusal="没说清是谁要去哪,请指名道姓。",
        broadcast={"content": "含糊的东西", "severity": "low", "location_scope": None},
    ))
    channel, agents, environment, bc, _ms = _world(container, router)

    result = await _submit(channel, "让他去那边", agents, environment)

    assert result.accepted is False
    assert result.reason == "没说清是谁要去哪,请指名道姓。"
    assert channel.pending_count() == 0
    # When judged not actionable, even a broadcast it wrote along the way must not be queued.
    assert bc.peek_pending() == []


@pytest.mark.asyncio
async def test_unparseable_llm_output_is_refused_not_guessed(container) -> None:
    router = _Router("这不是 JSON")
    channel, agents, environment, *_ = _world(container, router)
    result = await _submit(channel, "放一把火", agents, environment)
    assert result.accepted is False
    assert channel.pending_count() == 0


@pytest.mark.asyncio
async def test_index_ref_in_narrative_text_is_refused(container) -> None:
    """Indices leaking into human-readable text: still dispatched, but the trace records it.

    The model can carry "refer by index" into the summary and write something like
    "将 #1、#2 传送至 #20".
    """
    router = _Router(_plan_json(
        narrative_desc="将 #1、#2 直接传送至 #20。",
        mutations=[{
            "kind": "relocate", "person": 2, "location": 2,
            "observation": "李建成出现在太极宫。",
        }],
    ))
    channel, agents, environment, *_ = _world(container, router)
    active_call = _watch_adoption()

    result = await _submit(channel, "把他们弄到玄武门", agents, environment)

    assert result.accepted is True
    assert channel.pending_count() == 1
    assert active_call.adopted is False
    assert active_call.reject_reason == "index_ref_in_narrative"


@pytest.mark.asyncio
async def test_index_ref_in_a_perceived_channel_is_flagged_too(container) -> None:
    """Every human-readable field is checked: besides the summary, broadcasts, targeted messages,
    and observation."""
    router = _Router(_plan_json(
        narrative_desc="西市燃起大火",
        broadcast={"content": "#18 火起,浓烟蔽日。", "severity": "high",
                   "location_scope": 2, "phenomenon": "fire"},
    ))
    channel, agents, environment, bc, _ms = _world(container, router)
    active_call = _watch_adoption()

    result = await _submit(channel, "西市起火", agents, environment)

    assert result.accepted is True
    assert active_call.reject_reason == "index_ref_in_narrative"   # not only narrative_desc is checked


@pytest.mark.asyncio
async def test_feasible_but_empty_plan_is_refused(container) -> None:
    """Judged actionable but nothing dispatchable was parsed: treat it as a rejection instead of
    injecting an empty event."""
    router = _Router(_plan_json(feasible=True))
    channel, agents, environment, *_ = _world(container, router)
    result = await _submit(channel, "做点什么", agents, environment)
    assert result.accepted is False
    assert channel.pending_count() == 0


# ---------------------------------------------------------------------------
# Submit: acceptance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_directive_is_queued_with_a_preview(container) -> None:
    router = _Router(_plan_json(
        narrative_desc="西市燃起大火",
        broadcast={"content": "西市火起,浓烟蔽日。", "severity": "high",
                   "location_scope": 2, "phenomenon": "fire"},
    ))
    channel, agents, environment, *_ = _world(container, router)

    result = await _submit(channel, "西市起火", agents, environment)

    assert result.accepted is True
    assert result.preview == "西市燃起大火"
    assert result.queued == 1
    assert channel.pending_count() == 1


@pytest.mark.asyncio
async def test_parse_prompt_is_a_faithful_translator_not_an_author(container) -> None:
    """Director parsing and event authoring are opposites: one translates faithfully, the other
    invents freely. Low temperature, "don't embellish", and explicit permission to refuse are what
    set this path apart from EventSystem."""
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)

    await _submit(channel, "西市起火", agents, environment)

    prompt = router.joined()
    assert "不加戏" in prompt
    # Both halves are needed: "reject when it can't be carried out" alone makes the model reject
    # whatever it can't make someone do directly (Rule 2 already turns that into a message);
    # "carry it out indirectly" alone makes it reject directives that give no messenger or wording.
    assert "两条路都走不通才拒绝" in prompt
    assert "那句话由你来写" in prompt
    assert "不替人做决定" in prompt          # pressure is fine, manipulation isn't
    assert "西市起火" in prompt              # the director's words reach the prompt verbatim
    # References use indices; the lists never contain ids.
    assert "#1 李世民" in prompt
    assert "a1" not in prompt


@pytest.mark.asyncio
async def test_menu_carries_what_it_takes_to_recognize_someone(container) -> None:
    """The director rarely uses names. He says "the crown prince", "the one guarding the gate",
    "the south gate". The lists have to answer "is this the person / place / thing he means", so
    identity, condition, and description are all required."""
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)

    await _submit(channel, "让那个太子去西市", agents, environment)

    prompt = router.joined()
    # The identity header always goes through SoulLayer.identity_text() (name, age, gender); gender
    # and age are often the only way to tell brothers apart
    assert "#2 李建成，37岁，男（太子，现于西市）" in prompt
    assert "李渊长子，久居长安留守理政。" in prompt            # background: the main clue for recognizing someone
    assert "宫城正殿，天子听政之所" in prompt                  # location description: the only way to tell the six gates apart
    assert "宫城北门，禁军屯守之要冲" in prompt                # items: besides the name, say what it is
    # Item location plays the role of a person's "现于西市": with two letters on the list, where
    # they are is the only thing that tells them apart
    assert "#1 玄武门（此刻状态：intact，现于太极宫）" in prompt


@pytest.mark.asyncio
async def test_prompt_states_rules_without_explaining_the_engine(container) -> None:
    """The prompt states rules, not consequences.

    Reasons meant for maintainers ("this gets written to memory, shown in replay") don't improve the
    output and dilute attention on the rules. Menu headings likewise don't mention schema paths.
    """
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)

    await _submit(channel, "放个消息", agents, environment)

    prompt = router.joined()
    # The blocklist holds whole-sentence explanations of downstream mechanisms, not nouns: nouns
    # would hit legitimate world language ("记忆" in "不能直接改一个人的情绪、记忆或关系").
    for leaked in ("写进角色记忆", "被反复回想", "进回放", "被召回", "与事实脱节",
                   "vitality.agent", "relocate.location", "broadcast.location_scope"):
        assert leaked not in prompt, f"prompt 泄漏了引擎细节:{leaked}"
    # The rules themselves stay (what to do, not what downstream does with it).
    assert "observation" in prompt
    assert "#序号" in prompt


@pytest.mark.asyncio
async def test_prompt_tells_the_model_a_fire_needs_somewhere_to_burn(container) -> None:
    """A sourced phenomenon without a location is dropped by code, but the director wants the fire.

    Only a model that understands the directive can add the location, so the prompt says so, with
    the drop kept as the floor (see core/interfaces/phenomenon.py).
    """
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)

    await _submit(channel, "西市放一把火", agents, environment)

    prompt = router.joined()
    assert "必须同时给出 location_scope" in prompt
    assert "笼罩全域" in prompt      # pervasive phenomena (rain, snow, wind) are stated the other way: they need no location
    # Indices go only in reference fields; "always refer by index" alone leaks them into narrative
    # text ("将 #1、#2 传送至 #20"). Both sentences must be there.
    assert "**序号(#N)只填在引用位置**" in prompt
    assert "出现 #序号、id、「第N步」都是错的" in prompt


# ---------------------------------------------------------------------------
# Commit: pure dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_dispatches_broadcast_with_its_phenomenon(container) -> None:
    router = _Router(_plan_json(
        narrative_desc="西市燃起大火",
        # Location indices bind to the list sorted by id (market, palace), not registration
        # order. The prompt's numbering and the parser use the same list, so as long as both come
        # from one source it's deterministic.
        broadcast={"content": "西市火起。", "severity": "high",
                   "location_scope": 1, "phenomenon": "fire"},
    ))
    channel, agents, environment, bc, _ms = _world(container, router)
    await _submit(channel, "西市起火", agents, environment)

    committed = await channel.drain(step=4, agents=agents)

    assert len(committed) == 1
    assert serialize_world_event(committed[0].event)["authored_by"] == "director"   # the frontend uses this for the "导演" tag
    assert serialize_world_event(committed[0].event)["step"] == 4
    pending = bc.peek_pending()
    assert len(pending) == 1
    assert pending[0].location_scope == "market"
    assert pending[0].phenomenon is Phenomenon.FIRE    # the renderer uses this to play the fire
    assert pending[0].deliver_step == 4                # injected this step → perceived this step


@pytest.mark.asyncio
async def test_drain_needs_no_llm(container) -> None:
    """The queue stores validated plans, so committing in the step loop never touches the LLM or
    adds latency."""
    router = _Router(_plan_json(
        broadcast={"content": "x", "severity": "low", "location_scope": None},
    ))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "放个消息", agents, environment)
    calls_after_submit = len(router.calls)

    await channel.drain(step=2, agents=agents)

    assert len(router.calls) == calls_after_submit


@pytest.mark.asyncio
async def test_drain_empties_the_whole_backlog_in_one_step(container) -> None:
    """Three queued directives mean the director wants all three to happen. Spreading them over
    three steps would just make the system feel slow to him."""
    router = _Router(*[
        _plan_json(narrative_desc=f"第{i}件事",
                   broadcast={"content": f"事{i}", "severity": "low", "location_scope": None})
        for i in range(1, 4)
    ])
    channel, agents, environment, *_ = _world(container, router)
    for i in range(3):
        await _submit(channel, f"指令{i}", agents, environment)
    assert channel.pending_count() == 3

    events = await channel.drain(step=9, agents=agents)

    assert len(events) == 3
    assert channel.pending_count() == 0


@pytest.mark.asyncio
async def test_drain_applies_mutations_and_records_who_was_touched(container) -> None:
    router = _Router(_plan_json(
        narrative_desc="李建成被带到太极宫",
        mutations=[{
            "kind": "relocate", "person": 2, "location": 2,   # #2 = palace (sorted by id)
            "observation": "李建成出现在殿前。",
        }],
    ))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "把李建成带到太极宫", agents, environment)

    committed = await channel.drain(step=5, agents=agents)

    assert environment.get_body_location("a2") == "palace"
    assert agents["a2"].personality.state.current_location == "palace"
    assert serialize_world_event(committed[0].event)["dispatched_to"] == ["mutation"]
    # The channel returns who was reached: the person moved plus whoever sees it at either end. A
    # caller guessing via getattr("agent_id") on the mutation only counts the subject; see the next
    # test.
    assert committed[0].target_ids == ("a2", "a1")   # code-layer coordinates, not sent over the wire
    assert serialize_world_event(committed[0].event)["affected_names"] == ["李建成", "李世民"]   # narrative-layer referents
    # He was moved, not walked: from location_id alone the map would draw the teleport as a walk.
    assert committed[0].displaced_ids == ("a2",)


@pytest.mark.asyncio
async def test_an_entity_mutation_counts_the_people_who_saw_it(container) -> None:
    """Destroying a thing also reaches the people who see it.

    Pitfall: guessing the audience with ``getattr("agent_id")`` on the mutation finds nothing
    (``EntityMutation`` has no such field), so tearing up a letter in front of two people would
    report reaching no one. ``InjectionDispatcher.dispatch`` follows the same rule for broadcasts.
    """
    router = _Router(_plan_json(
        narrative_desc="玄武门被撞开",
        mutations=[{
            "kind": "entity", "entity": 1, "state": "broken", "destroyed": False,
            "observation": "玄武门轰然洞开。",
        }],
    ))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "把玄武门撞开", agents, environment)

    committed = await channel.drain(step=5, agents=agents)

    assert environment.get_entity("gate").state == "broken"
    assert serialize_world_event(committed[0].event)["dispatched_to"] == ["mutation"]
    # Li Shimin is in Taiji Palace, right in front of the gate; Li Jiancheng is in the West Market
    # and can't see it.
    assert committed[0].target_ids == ("a1",)
    assert serialize_world_event(committed[0].event)["affected_names"] == ["李世民"]


@pytest.mark.asyncio
async def test_a_mutation_nobody_witnessed_still_counts_as_dispatched(container) -> None:
    """A change nobody saw still happened; an empty audience is not a failure.

    ``apply`` returns ``()`` for "applied, nobody in the room". A caller testing ``if reached:``
    would keep the intervention out of the event stream, and the world would change silently.
    """
    router = _Router(_plan_json(
        narrative_desc="西市起火",
        mutations=[{
            "kind": "entity", "entity": 1, "state": "burning", "destroyed": False,
            "observation": "玄武门腾起大火。",
        }],
    ))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "点了玄武门", agents, environment)
    # Move away the only person who can see it: the gate is still in Taiji Palace, which is now
    # empty.
    environment.move_body(body_id="a1", location_id="market")

    committed = await channel.drain(step=5, agents=agents)

    assert environment.get_entity("gate").state == "burning"
    assert len(committed) == 1                                  # ← still enters the event stream
    assert serialize_world_event(committed[0].event)["dispatched_to"] == ["mutation"]
    assert committed[0].target_ids == ()                        # really nobody saw it
    assert serialize_world_event(committed[0].event)["affected_names"] == []


@pytest.mark.asyncio
async def test_mutation_without_an_observation_is_dropped(container) -> None:
    """Invariant: when the world changes, someone must be able to see it. A mutation without a
    description is dropped at parse time instead of landing in the world with an empty string."""
    router = _Router(_plan_json(
        narrative_desc="有事发生",
        mutations=[{"kind": "relocate", "person": 2, "location": 1, "observation": "  "}],
    ))
    channel, agents, environment, *_ = _world(container, router)

    result = await _submit(channel, "把他挪走", agents, environment)

    # The only mutation is dropped → the plan is empty → the whole directive is rejected.
    assert result.accepted is False
    assert environment.get_body_location("a2") == "market"


@pytest.mark.asyncio
async def test_directive_events_are_authored_by_the_director(container) -> None:
    router = _Router(_plan_json(
        broadcast={"content": "x", "severity": "low", "location_scope": None},
    ))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "放个消息", agents, environment)
    committed = await channel.drain(step=2, agents=agents)

    assert [c.event.authored_by.value for c in committed] == [Author.DIRECTOR.value]


# ---------------------------------------------------------------------------
# The debug console's two seams: assemble (without sending) and validate (without queuing).
#
# Both must share their source with submit (a separate prompt or parser would test something other
# than production), and neither changes the world.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_prompt_is_the_very_prompt_submit_sends(container) -> None:
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)

    prompt = channel.build_prompt(
        "西市起火",
        all_agents=agents,
        npcs=_npc_menu(environment),
        locations=environment.space.all_places(),
        entities=environment.all_live_entities(),
        world_time=_time(),
    )
    await _submit(channel, "西市起火", agents, environment)

    scene, messages = router.calls[0]
    assert [m.content for m in messages] == [prompt.system, prompt.user]
    assert scene is prompt.scene
    # Index → name: the LLM only outputs indices, and the console relies on this table to check
    # who it pointed at, so it must be the one in the prompt, not a recount.
    assert prompt.menus["cast"] == {"1": "李世民", "2": "李建成"}
    assert "#1 李世民" in prompt.user


def test_interpret_validates_without_queueing_anything(container) -> None:
    """Validation is separate from queuing, so the console can run a response through production
    validation while the world stays untouched."""
    channel, agents, environment, bc, _ms = _world(container, _Router())
    raw = _plan_json(
        narrative_desc="西市燃起大火",
        broadcast={"content": "西市火起。", "severity": "high",
                   "location_scope": 1, "phenomenon": "fire"},
    )

    result, plan = channel.interpret(
        raw,
        all_agents=agents,
        npcs=_npc_menu(environment),
        locations=environment.space.all_places(),
        entities=environment.all_live_entities(),
    )

    assert result.accepted is True
    assert plan is not None
    assert channel.pending_count() == 0     # validating is not submitting
    assert bc.peek_pending() == []          # let alone injecting
    described = channel.describe_plan(plan)
    assert described["channels"] == ["broadcast"]
    assert described["broadcast"]["location"] == "西市"   # id already replaced with a name


def test_interpret_reports_the_same_refusal_the_director_would_hear(container) -> None:
    """An out-of-range index resolves to no target. The rejection the console sees is the same
    sentence the director will hear."""
    channel, agents, environment, *_ = _world(container, _Router())

    result, plan = channel.interpret(
        _plan_json(narrative_desc="有事发生",
                   message={"recipients": [99], "content": "急讯", "urgency": "high"}),
        all_agents=agents,
        npcs=_npc_menu(environment),
        locations=environment.space.all_places(),
        entities=environment.all_live_entities(),
    )

    assert plan is None
    assert result.accepted is False
    assert "没有落到具体的人" in result.reason


@pytest.mark.asyncio
async def test_the_original_sentence_survives_serialization_and_restore(container) -> None:
    """The director's original words round-trip with the record, so another process opening the
    world still has them. They live on ``WorldEvent`` so live and replay share one serialization
    path, with no second source of truth.
    """
    from engine.injection import restore_world_events

    router = _Router(_plan_json(
        narrative_desc="西市燃起大火",
        broadcast={"content": "西市火起。", "severity": "high", "location_scope": 1},
    ))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "在西市放一把火，要让全城都看见", agents, environment)

    committed = await channel.drain(step=7, agents=agents)

    payload = serialize_world_event(committed[0].event)
    assert payload["directive_text"] == "在西市放一把火，要让全城都看见"
    restored = restore_world_events([payload])
    assert restored[0].directive_text == "在西市放一把火，要让全城都看见"
    assert restored[0].authored_by is Author.DIRECTOR


@pytest.mark.asyncio
async def test_menu_says_who_is_holding_a_thing(container) -> None:
    """Ownership is a branch of placement and type, not a separate field. A thing is either in
    someone's hands or at some place, and the menu says one or the other. A held thing has no
    location_id, so missing this branch would say nothing at all about such items."""
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)
    held = WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM,
        state="sealed", description="火漆未启的一封信",
        presence=EntityPresence.HELD, presence_ref="a2", is_takeable=True,
    )

    prompt = channel.build_prompt(
        "把消息放出去",
        all_agents=agents,
        npcs=_npc_menu(environment),
        locations=environment.space.all_places(),
        entities=[held],
        world_time=_time(),
    )

    assert "#1 密信（此刻状态：sealed，在李建成手上）" in prompt.user
    assert "a2" not in prompt.user   # ownership is translated to a name via directory; never leak an id


# ---------------------------------------------------------------------------
# Errand-runners: share one list and one index space with agents
# ---------------------------------------------------------------------------


def _add_npc(environment: EnvironmentSystem, at: str = "market") -> str:
    environment.spawn_npc(
        NpcSeed(name="王二", gender="男", age=34, description="东宫的跑腿小厮"), location_id=at,
    )
    return environment.all_npcs()[-1].npc_id


@pytest.mark.asyncio
async def test_npcs_follow_the_agents_on_the_one_cast_menu(container) -> None:
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)
    npc_id = _add_npc(environment)
    environment.assign_errand(ErrandOrder(npc_id, "palace"), requester_id="a2")

    prompt = channel.build_prompt(
        "截住那个跑腿的", all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
        world_time=_time(),
    )

    assert prompt.menus["cast"] == {"1": "李世民", "2": "李建成", "3": "王二"}
    line = next(ln for ln in prompt.user.splitlines() if ln.startswith("#3 "))
    assert "听吩咐办事" in line and "现于西市" in line and "正替李建成办一趟差事" in line


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation, expected", [
    ({"kind": "relocate", "person": 3, "location": 2, "observation": "王二被人拖进了太极宫。"},
     RelocateMutation),
    ({"kind": "condition", "person": 3, "condition": "被捆在柱上", "observation": "王二被捆在了柱上。"},
     ConditionMutation),
], ids=["relocate", "condition"])
async def test_an_npc_index_resolves_to_the_npc(container, mutation, expected) -> None:
    raw = _plan_json(narrative_desc="王二出了事", mutations=[mutation])
    channel, agents, environment, *_ = _world(container, _Router(raw))
    npc_id = _add_npc(environment)

    result, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert result.accepted is True and plan is not None
    [parsed] = plan.mutations
    assert isinstance(parsed, expected)
    assert parsed.body_id == npc_id


@pytest.mark.asyncio
async def test_a_message_is_never_addressed_to_an_npc(container) -> None:
    raw = _plan_json(narrative_desc="有人递来急讯",
                     message={"recipients": [1, 3], "content": "速回宫。", "urgency": "high"})
    channel, agents, environment, *_ = _world(container, _Router(raw))
    _add_npc(environment)

    _, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert plan is not None and plan.message is not None
    assert plan.message.recipients == ["a1"]


@pytest.mark.asyncio
async def test_a_message_only_to_an_npc_is_no_message_at_all(container) -> None:
    raw = _plan_json(narrative_desc="有人递来急讯",
                     message={"recipients": [3], "content": "速回宫。", "urgency": "high"})
    channel, agents, environment, *_ = _world(container, _Router(raw))
    _add_npc(environment)

    result, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert plan is None and result.accepted is False


@pytest.mark.asyncio
async def test_a_thing_an_npc_holds_is_named_in_its_hands(container) -> None:
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)
    npc_id = _add_npc(environment)
    environment.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM, is_takeable=True,
        presence=EntityPresence.HELD, presence_ref=npc_id,
    ))

    prompt = channel.build_prompt(
        "烧了那封信", all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
        world_time=_time(),
    )

    assert "在王二手上" in prompt.user
    assert "在某人手上" not in prompt.user


def test_the_director_prompt_never_names_the_machine_tier() -> None:
    """"NPC" / "工具人" are facts about the machine, not words from the story. The director prompt
    doesn't use them either."""
    from engine.director import _SYSTEM_PROMPT

    for word in ("NPC", "Npc", "npc", "工具人"):
        assert word not in _SYSTEM_PROMPT, word


@pytest.mark.asyncio
async def test_life_and_death_on_an_npc_is_refused_at_submit_not_dropped_at_drain(container) -> None:
    """Submission is the only rejection point: accepting an intervention that can never land would
    make the director think it happened."""
    raw = _plan_json(narrative_desc="王二倒下了", mutations=[
        {"kind": "vitality", "person": 3, "effect": "kill", "observation": "王二倒在了街心。"},
    ])
    channel, agents, environment, *_ = _world(container, _Router(raw))
    _add_npc(environment)

    result, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert plan is None and result.accepted is False


# ---------------------------------------------------------------------------
# Items: placing new things / changing their appearance and text
# ---------------------------------------------------------------------------


_SPAWN = {
    "kind": "spawn", "location": 1, "entity_type": "item", "name": "密信",
    "description": "火漆封口的信", "content": "明日辰时玄武门见。",
    "observation": "西市的石阶上多了一封密信。",
}


@pytest.mark.asyncio
async def test_director_can_put_a_new_thing_down_somewhere(container) -> None:
    raw = _plan_json(narrative_desc="西市多了一封密信", mutations=[_SPAWN])
    channel, agents, environment, *_ = _world(container, _Router(raw))

    _, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert plan is not None
    [spawn] = plan.mutations
    assert isinstance(spawn, SpawnMutation)
    # The location list is sorted by id (market, palace), so #1 is the West Market.
    assert (spawn.location_id, spawn.name, spawn.entity_type, spawn.content) == (
        "market", "密信", "item", "明日辰时玄武门见。",
    )


@pytest.mark.asyncio
async def test_a_spawned_thing_lands_in_the_world_at_drain(container) -> None:
    router = _Router(_plan_json(narrative_desc="西市多了一封密信", mutations=[_SPAWN]))
    channel, agents, environment, *_ = _world(container, router)
    await _submit(channel, "西市出现一封密信", agents, environment)

    committed = await channel.drain(step=4, agents=agents)

    [letter] = [e for e in environment.all_live_entities() if e.name == "密信"]
    assert (letter.location_id, letter.owner_id, letter.content) == ("market", None, "明日辰时玄武门见。")
    # Li Jiancheng is in the West Market: he sees it appear, and the receipt records him.
    assert committed[0].target_ids == ("a2",)


@pytest.mark.asyncio
async def test_director_can_rewrite_how_a_thing_looks_and_what_it_says(container) -> None:
    raw = _plan_json(narrative_desc="玄武门的门匾被人改了字", mutations=[{
        "kind": "entity", "entity": 1, "destroyed": False, "state": "",
        "description": "门匾上的漆被刮花", "content": "此门不通",
        "observation": "玄武门的门匾上多了四个字。",
    }])
    channel, agents, environment, *_ = _world(container, _Router(raw))

    _, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert plan is not None
    [alter] = plan.mutations
    assert isinstance(alter, EntityMutation)
    assert (alter.entity_id, alter.new_state, alter.new_description, alter.new_content) == (
        "gate", "", "门匾上的漆被刮花", "此门不通",
    )


@pytest.mark.asyncio
async def test_the_menu_shows_what_every_thing_says_held_or_not(container) -> None:
    """Changing text takes the full text: without seeing the original, it could only be rewritten
    from nothing. The director has a god's-eye view, so items carried on someone are included."""
    router = _Router(_plan_json())
    channel, agents, environment, *_ = _world(container, router)
    environment.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM, is_takeable=True,
        presence=EntityPresence.HELD, presence_ref="a1", content="明日辰时玄武门见。",
    ))

    prompt = channel.build_prompt(
        "改那封信", all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
        world_time=_time(),
    )

    assert "上面写着：明日辰时玄武门见。" in prompt.user


@pytest.mark.asyncio
async def test_a_new_thing_of_no_known_kind_is_refused_at_submit(container) -> None:
    """An unrecognized category means ``spawn_entity`` will reject it at commit, so reject it at
    submit instead of accepting it and failing later."""
    raw = _plan_json(narrative_desc="西市多了一封密信", mutations=[{**_SPAWN, "entity_type": "物品"}])
    channel, agents, environment, *_ = _world(container, _Router(raw))

    result, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    assert plan is None and result.accepted is False


@pytest.mark.asyncio
async def test_a_new_thing_need_not_say_anything(container) -> None:
    """content is optional: a knife or a well has no writing, and placing one needs no text."""
    knife = {k: v for k, v in _SPAWN.items() if k != "content"} | {
        "name": "短刀", "description": "一把沾血的短刀", "observation": "西市的石阶上多了一把短刀。",
    }
    raw = _plan_json(narrative_desc="西市多了一把短刀", mutations=[knife])
    channel, agents, environment, *_ = _world(container, _Router(raw))
    await _submit(channel, "西市出现一把刀", agents, environment)

    await channel.drain(step=4, agents=agents)

    [placed] = [e for e in environment.all_live_entities() if e.name == "短刀"]
    assert placed.content == ""


@pytest.mark.asyncio
async def test_the_console_reads_every_mutation_kind_by_name(container) -> None:
    """Read side of the dev debug console: every change is rendered as names and what exactly
    changed, without dropping kind or exposing ids."""
    raw = _plan_json(narrative_desc="王二被绑、门匾改字、西市多了一封信", mutations=[
        {"kind": "condition", "person": 3, "condition": "被捆在柱上", "observation": "王二被捆在了柱上。"},
        {"kind": "entity", "entity": 1, "destroyed": False, "state": "", "description": "漆被刮花",
         "content": "此门不通", "observation": "门匾上多了四个字。"},
        _SPAWN,
    ])
    channel, agents, environment, *_ = _world(container, _Router(raw))
    _add_npc(environment)

    _, plan = channel.interpret(
        raw, all_agents=agents, npcs=_npc_menu(environment),
        locations=environment.space.all_places(), entities=environment.all_live_entities(),
    )

    described = channel.describe_plan(plan)["mutations"]
    assert [(m["kind"], m["target"], m["detail"]) for m in described] == [
        ("ConditionMutation", "王二", "被捆在柱上"),
        ("EntityMutation", "玄武门", "样子：漆被刮花；写着：此门不通"),
        ("SpawnMutation", "密信", "放在西市"),
    ]
