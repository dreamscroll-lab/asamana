"""The render-neutral membrane: what an observer is told an action WAS.

Untested, it can tell observers that picking a letter up and smashing it are the same
event. It is the only source of truth a renderer
(or any future observer — audio, a text narrator) has for *what the body did*, so its
derivation is pinned here.
"""

from core.interfaces.action import (
    KIND_ITEM, KIND_OBJECT,
    ActionResult,
    ActionTarget,
    ActionType,
    Ref,
    Deed,
    EntityStateChange,
    AgentAction,
    parse_deed,
)
from engine.executors.base import action_semantics


def _act(action_type: ActionType, target: ActionTarget | None = None) -> AgentAction:
    return AgentAction(
        agent_id="agent-a",
        step=1,
        action_type=action_type,
        action_description="做点什么",
        target=target or ActionTarget(),
    )


def _result(action: AgentAction, **kw) -> ActionResult:
    return ActionResult(action=action, expected_outcome="", outcome="", **kw)


class TestDeedFromPhysical:
    """PHYSICAL's deed comes from its adjudicator — it is the only one who knows the act."""

    def test_seize_and_destroy_are_not_the_same_event(self):
        """Picking a thing up and smashing it must not read as one event.

        Both change an entity's state, so a field keyed on that change alone calls both
        "alter", and a renderer keyed on it draws a pick-up as a smash. The deed keeps them apart.
        """
        target = ActionTarget(acts_on=[Ref.entity("e1", "item")])
        action = _act(ActionType.PHYSICAL, target)
        change = [EntityStateChange(entity_id="e1", new_state="held")]

        seized = action_semantics(action, _result(action, deed=Deed.SEIZE.value, entity_state_changes=change))
        destroyed = action_semantics(action, _result(action, deed=Deed.DESTROY.value, entity_state_changes=change))

        assert seized["deed"] == "seize"
        assert destroyed["deed"] == "destroy"
        assert seized["deed"] != destroyed["deed"]
        # …while everything else about them still looks identical, which is the whole point.
        assert seized["target"] == destroyed["target"]
        assert seized["affected_entity_ids"] == destroyed["affected_entity_ids"] == ["e1"]

    def test_deed_survives_failure(self):
        """A snatch that missed is still a snatch, not a punch that missed.

        The deed is the ACT, not its outcome, so it must reach the observer even when the
        action failed and left no trace in the world to infer it from.
        """
        action = _act(ActionType.PHYSICAL, ActionTarget(acts_on=[Ref.entity("e1", "item")]))
        result = _result(action, succeeded=False, deed=Deed.SEIZE.value)  # nothing changed
        assert action_semantics(action, result)["deed"] == "seize"

    def test_no_adjudication_means_no_deed(self):
        """not_executed / adjudication_failed: nothing was done, so nothing is claimed.

        Empty must NOT fall back to "physical" — that is a category, not an act, and an
        observer that drew it would be showing a deed that never happened.
        """
        action = _act(ActionType.PHYSICAL, ActionTarget(acts_on=[Ref.agent("agent-b")]))
        result = _result(action, succeeded=False, not_executed=True)  # executor set no deed
        assert action_semantics(action, result)["deed"] == ""


class TestDeedFromOtherTypes:
    """Every other type IS its own deed — one axis is enough for them."""

    def test_each_type_is_its_own_deed(self):
        for action_type in (
            ActionType.TALK, ActionType.SEND_MESSAGE, ActionType.MOVE,
            ActionType.WORK, ActionType.REST, ActionType.COVERT, ActionType.ERRAND,
        ):
            action = _act(action_type)
            assert action_semantics(action, _result(action))["deed"] == action_type.value


class TestTargetOnTheWire:
    def test_the_wire_carries_the_three_relations_verbatim(self):
        """One storage for every kind of object, and the kind rides with the id — a blow
        aimed at a PERSON reports the person, a journey reports the place it is aimed at.
        Nothing is flattened into a separate "which list do I read" question."""
        cases = [
            (ActionTarget(acts_on=[Ref.agent("agent-b")]), [{"kind": "agent", "id": "agent-b"}]),
            (ActionTarget(acts_on=[Ref.entity("e1", "item")]), [{"kind": "item", "id": "e1"}]),
            (ActionTarget(acts_on=[Ref.place("hall")]), [{"kind": "location", "id": "hall"}]),
            (ActionTarget(), []),
        ]
        for target, expected in cases:
            action = _act(ActionType.PHYSICAL, target)
            assert action_semantics(action, _result(action))["target"]["acts_on"] == expected

    def test_a_journeys_destination_survives_onto_the_wire(self):
        """The place is a first-class aim, not a kind with nothing behind it. Reporting only
        "this is aimed at somewhere" and dropping WHERE leaves the observer unable to say what
        a journey was ever about."""
        action = _act(
            ActionType.MOVE,
            ActionTarget(acts_on=[Ref.place("buzheng")], claims=[Ref.agent("weichi")]),
        )
        target = action_semantics(action, _result(action))["target"]
        assert target["acts_on"] == [{"kind": "location", "id": "buzheng"}]
        # …and whose turn it spends, which is how a carried companion is known at all.
        assert target["claims"] == [{"kind": "agent", "id": "weichi"}]

    def test_a_handover_keeps_the_object_as_the_aim_and_names_the_recipient(self):
        """A hand-over's aim is two-dimensional: the thing acted on stays the thing; the receiver
        goes in a separate slot.

        In acts_on, the receiver would be read as the aim and the item would vanish, so a gift would
        be drawn as a shove. He belongs in reaches. The read model is the persistent record: if it
        doesn't record the receiver, replay loses him for good.
        """
        target = ActionTarget(acts_on=[Ref.entity("e1", "item")], reaches=[Ref.agent("agent-b")])
        action = _act(ActionType.PHYSICAL, target)
        sem = action_semantics(action, _result(action))["target"]
        assert sem["acts_on"] == [{"kind": "item", "id": "e1"}]
        assert sem["reaches"] == [{"kind": "agent", "id": "agent-b"}]
        assert target.acted_on_agents == []
        # No receiver means this slot is empty.
        assert action_semantics(
            _act(ActionType.PHYSICAL, ActionTarget(acts_on=[Ref.entity("e1", "item")])),
            _result(action),
        )["target"]["reaches"] == []

    def test_relinquish_is_a_physical_deed(self):
        """Handing over is PHYSICAL's seventh verb and survives a parse_deed round-trip intact."""
        from core.interfaces.action import PHYSICAL_DEEDS, parse_deed
        assert Deed.RELINQUISH in PHYSICAL_DEEDS
        assert parse_deed("relinquish", default=Deed.OPERATE) is Deed.RELINQUISH

    def test_a_physical_target_reaches_the_observer_as_a_directed_id(self):
        """A kind alone is not enough: the observer needs WHICH agent.

        Both fields feed the same question, and PHYSICAL is where they are easiest to get
        wrong — the people in `acts_on` are what the map turns a figure toward, walks him to, and aims
        his blow at, so an empty list is a fight with no one in it.
        """
        assert ActionTarget(acts_on=[Ref.agent("agent-b")]).acted_on_agents == ["agent-b"]
        # A thing is not a person, and the same field carries both.
        assert ActionTarget(acts_on=[Ref.entity("e1", "item")]).acted_on_agents == []
        # A letter to two people is aimed at two — one aim, several referents.
        assert ActionTarget(
            acts_on=[Ref.agent("a"), Ref.agent("b")]
        ).acted_on_agents == ["a", "b"]


class TestAimedAtVersusChanged:
    """The THING half of the same question people already had answered for them.

    `affected_entity_ids` reports what CHANGED, which is empty exactly when a man heaved at
    a gate and the gate held — the one moment the map most needs to know where he was
    pushing. Aim belongs to the intent and survives failure, like the deed does.
    """

    def test_a_failed_act_still_says_what_it_was_aimed_at(self):
        action = _act(ActionType.PHYSICAL, ActionTarget(acts_on=[Ref.entity("gate", "landmark")]))
        sem = action_semantics(action, _result(action, succeeded=False, deed=Deed.DESTROY.value))
        assert sem["affected_entity_ids"] == []      # nothing gave
        assert sem["target"]["acts_on"] == [{"kind": "landmark", "id": "gate"}]  # …but we know where he was pushing

    def test_a_person_target_aims_at_no_entity(self):
        """A blow at a man is not aimed at a thing — that channel stays empty for him."""
        action = _act(ActionType.PHYSICAL, ActionTarget(acts_on=[Ref.agent("agent-b")]))
        sem = action_semantics(action, _result(action, deed=Deed.STRIKE.value))
        assert sem["target"]["acts_on"] == [{"kind": "agent", "id": "agent-b"}]

    def test_the_two_channels_agree_when_the_act_lands(self):
        action = _act(ActionType.PHYSICAL, ActionTarget(acts_on=[Ref.entity("e1", "item")]))
        sem = action_semantics(action, _result(
            action, deed=Deed.SEIZE.value,
            entity_state_changes=[EntityStateChange(entity_id="e1", new_state="held")],
        ))
        assert [r["id"] for r in sem["target"]["acts_on"]] == sem["affected_entity_ids"] == ["e1"]


class TestParseDeed:
    def test_unknown_label_falls_to_the_conservative_default(self):
        # An LLM label is a label, not a promise. Anything unrecognised must not become
        # violence by accident.
        assert parse_deed("擒抱", default=Deed.RESTRAIN) is Deed.RESTRAIN
        assert parse_deed("", default=Deed.OPERATE) is Deed.OPERATE
        assert parse_deed("  STRIKE ", default=Deed.RESTRAIN) is Deed.STRIKE


class TestResolveDeed:
    """The executor reconciles the judge's LABEL with the FACTS the judge also reported.

    The label crosses an LLM boundary; the facts are the world. Where they disagree, the
    world wins — otherwise the picture and the simulation would tell different stories.
    """

    @staticmethod
    def _resolve(label, **kw):
        from engine.executors.physical import PhysicalExecutor
        base = dict(is_person=False, has_entity=False, seize_available=False,
                    actor_holds=False, target_damage=0.0)
        return PhysicalExecutor._resolve_deed(label, **{**base, **kw})

    def test_blood_settles_it(self):
        # The judge called it a restraint but reported a wound. A body that bleeds was struck.
        assert self._resolve("restrain", is_person=True, target_damage=0.4) is Deed.STRIKE

    def test_a_seize_that_cannot_be_honoured_is_an_operate(self):
        # The executor will not move an untakeable thing, so claiming a seize would leave
        # the picture (a man walking off with the city gate) disagreeing with the world.
        assert self._resolve("seize", has_entity=True, seize_available=False) is Deed.OPERATE
        assert self._resolve("seize", has_entity=True, seize_available=True) is Deed.SEIZE
        # Same guard, mirrored twice more: taking what is already in his own hand is not a
        # taking, and letting go of what he is not holding is not a letting-go.
        assert self._resolve("seize", has_entity=True, seize_available=False,
                             actor_holds=True) is Deed.OPERATE
        assert self._resolve("relinquish", has_entity=True, actor_holds=False) is Deed.OPERATE
        assert self._resolve("relinquish", has_entity=True, actor_holds=True) is Deed.RELINQUISH

    def test_a_state_word_saying_it_is_gone_settles_it(self):
        # Both fields say "it's destroyed"; the judge just picked the wrong channel. Without
        # honoring that, the item stays in the world as "状态：destroyed；可取" while the outcome
        # has already announced it no longer exists.
        assert self._resolve("operate", has_entity=True,
                             new_entity_state="destroyed") is Deed.DESTROY
        assert self._resolve("seize", has_entity=True, seize_available=True,
                             new_entity_state="已焚毁") is Deed.DESTROY

    def test_merely_damaged_is_not_gone(self):
        # This net only catches "explicitly says it no longer exists". A broken-but-present thing
        # judged destroyed gets deleted from the world — that error is worse than a miss, so when
        # unsure it stays operate.
        for still_there in ("shattered", "torn", "open", "破损", "裂开", "被动过"):
            assert self._resolve("operate", has_entity=True,
                                 new_entity_state=still_there) is Deed.OPERATE

    def test_the_state_word_never_destroys_a_person(self):
        # People don't take the entity branch: a ruling that kills someone is STRIKE, not DESTROY.
        assert self._resolve("strike", is_person=True, target_damage=1.0,
                             new_entity_state="destroyed") is Deed.STRIKE

    def test_defaults_are_conservative(self):
        # Garbage in must never manufacture violence or destruction.
        assert self._resolve("???", is_person=True) is Deed.RESTRAIN
        assert self._resolve("???", has_entity=True, seize_available=True) is Deed.OPERATE

    def test_a_deed_from_the_wrong_shape_is_rejected(self):
        # You cannot restrain a gate, nor destroy a man (that is a STRIKE that killed him).
        assert self._resolve("restrain", has_entity=True) is Deed.OPERATE
        assert self._resolve("destroy", is_person=True) is Deed.RESTRAIN

    def test_no_target_is_exert(self):
        assert self._resolve("strike") is Deed.EXERT


# ---------------------------------------------------------------------------
# The judge's PARSE path. The executor's own tests all monkeypatch _judge away,
# so these drive the real _judge with a scripted router to cover the
# prompt→JSON→_Verdict chain, where the deed actually enters the system.
# ---------------------------------------------------------------------------

import pytest


class _ScriptedLLM:
    """Router stub that returns one canned JSON body for any completion."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.messages = []

    async def complete(self, scene, messages, **kwargs):
        self.messages = messages

        class _R:
            content = self.payload

        return _R()


async def _judge_with(payload: str, *, agent, is_person: bool, has_entity: bool, takeable: bool,
                      holds: bool = False, needs_relation: bool = False, recipient_line: str = ""):
    from engine.executors.physical import PhysicalExecutor
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    llm = _ScriptedLLM(payload)
    ex = PhysicalExecutor(llm, LiveWorldDirectory.from_agents({}, EnvironmentSystem()))  # type: ignore[arg-type]
    verdict = await ex._judge(
        agent, description="做点什么", expected_outcome="", scene="",
        target_block="目标", is_person=is_person, has_entity=has_entity,
        seize_available=takeable, actor_holds=holds, needs_relation=needs_relation,
        recipient_line=recipient_line, step=1,
    )
    return verdict, llm


class TestJudgeParsesDeed:
    @pytest.mark.asyncio
    async def test_seize_reaches_the_verdict(self, container) -> None:
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        verdict, llm = await _judge_with(
            '{"reason":"他弯腰把它收进袖中","deed":"seize","success":true,'
            '"outcome":"甲拿起了令牌","fact":"我拿起了令牌","new_entity_state":"intact","actor_damage":0.0}',
            agent=agent, is_person=False, has_entity=True, takeable=True,
        )
        assert verdict is not None and verdict.deed is Deed.SEIZE
        # …and the prompt actually offered the choice (a schema that never asks gets no answer).
        prompt = "".join(m.content for m in llm.messages)
        assert '"deed"' in prompt and "seize" in prompt

    @pytest.mark.asyncio
    async def test_the_judge_is_told_that_taking_from_a_holder_is_not_picking_up(self, container) -> None:
        # In the materials, things in someone else's hands are only marked "held by X"; the judge
        # must know it has to be taken from him, and that holding is authoritative over description.
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        _, llm = await _judge_with(
            '{"reason":"r","deed":"seize","success":false,"outcome":"甲没拿到","fact":"我没拿到",'
            '"why":"对方攥得紧","relation":"negative","new_entity_state":"","actor_damage":0.0}',
            agent=agent, is_person=False, has_entity=True, takeable=True, needs_relation=True,
        )
        system = llm.messages[0].content
        assert "由某人持有" in system and "不是捡起一件无主之物" in system
        assert "以「由某人持有」为准" in system

    @pytest.mark.asyncio
    @pytest.mark.parametrize("is_person, has_entity, expected", [
        (True, False, True),     # at a person (incl. bodies without cognition): this path can't change ownership
        (False, True, False),    # at a thing: transfer goes through this path
        (False, False, False),   # no identifiable target
    ])
    async def test_only_a_judge_of_a_body_is_told_nothing_changes_hands(
        self, container, is_person, has_entity, expected,
    ) -> None:
        # A ruling on a person doesn't change item ownership, so the judge must not write a
        # transfer — otherwise both sides' memories disagree with world state.
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        _, llm = await _judge_with(
            '{"reason":"r","success":true,"outcome":"甲动了手","fact":"我动了手","actor_damage":0.0}',
            agent=agent, is_person=is_person, has_entity=has_entity, takeable=has_entity,
        )
        assert ("不会让任何东西易手" in llm.messages[0].content) is expected

    @pytest.mark.asyncio
    async def test_an_overreaching_seize_is_reined_in_at_parse_time(self, container) -> None:
        # The gate is not takeable. The judge says seize anyway; the executor will not move
        # it, so the deed must land on OPERATE or the picture would show a man walking off
        # with a city gate.
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        verdict, llm = await _judge_with(
            '{"reason":"他推开了它","deed":"seize","success":true,'
            '"outcome":"甲推开了门","fact":"我推开了门","new_entity_state":"open","actor_damage":0.0}',
            agent=agent, is_person=False, has_entity=True, takeable=False,
        )
        assert verdict is not None and verdict.deed is Deed.OPERATE
        assert "seize" not in "".join(m.content for m in llm.messages)  # never even offered

    @pytest.mark.asyncio
    async def test_the_real_payload_that_left_a_destroyed_thing_in_the_world(self, container) -> None:
        """A real production payload: both judge fields say "destroyed" but it reported operate.

        Without honoring that, the secret silk letter would stay on the scene menu as
        "状态：destroyed；可取" for many steps, while the outcome has already told the two people
        in the room that it no longer exists.
        """
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        verdict, _ = await _judge_with(
            '{"reason":"帛书在场且标注可取","deed":"operate","success":true,'
            '"outcome":"李建成取出密报帛书细读后亲手销毁，帛书不再存在。",'
            '"fact":"我取出密报帛书细读，暗记线索后亲手销毁。","why":"",'
            '"new_entity_state":"destroyed","actor_damage":0.05,"frees_actor":false}',
            agent=agent, is_person=False, has_entity=True, takeable=True,
        )
        assert verdict is not None and verdict.deed is Deed.DESTROY

    @pytest.mark.asyncio
    async def test_a_person_deed_is_reconciled_with_the_blood(self, container) -> None:
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        verdict, _ = await _judge_with(
            '{"reason":"他只想拦住对方","deed":"restrain","success":true,'
            '"outcome":"甲拦住了乙","fact":"我拦住了乙","relation":"negative",'
            '"target_damage":0.5,"actor_damage":0.0}',
            agent=agent, is_person=True, has_entity=False, takeable=False,
        )
        assert verdict is not None and verdict.deed is Deed.STRIKE  # blood settles it

    @pytest.mark.asyncio
    async def test_no_target_is_exert_even_though_the_schema_never_asked(self, container) -> None:
        from tests.unit.test_action_executors import _make_agent
        agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        verdict, llm = await _judge_with(
            '{"reason":"他用力一推","success":true,"outcome":"甲使了把力","fact":"我使了把力","actor_damage":0.1}',
            agent=agent, is_person=False, has_entity=False, takeable=False,
        )
        assert verdict is not None and verdict.deed is Deed.EXERT
        assert '"deed"' not in "".join(m.content for m in llm.messages)


class TestTargetSurvivesTheExecutor:
    """An executor rebuilds its ActionResult from a stub AgentAction, and the stub must NOT
    drop the target: an empty `acts_on` means the sim acted on a named recipient while the
    observer is told the letter was addressed to nobody — drawn as a notice posted on a wall.

    ActionExecutionState carries the target for the life of the execution, which is what
    lets a stub restore it.
    """

    @pytest.mark.asyncio
    async def test_send_message_result_keeps_its_recipients(self, container) -> None:
        from tests.unit.test_action_executors import _make_agent, _make_environment, _directory
        from engine.executors.simple import SimpleExecutor
        from core.interfaces.action import AgentAction as AA

        sender = _make_agent(container, world_id="w", agent_id="agent-a", name="甲", is_main=True)
        env = _make_environment()
        ex = SimpleExecutor(_directory({"agent-a": sender}))
        action = AA(
            agent_id="agent-a", step=1, action_type=ActionType.SEND_MESSAGE,
            action_description="传讯给乙", target=ActionTarget(acts_on=[Ref.agent("agent-b")]),
        )
        # message_system=None: the executor guards on it, so the dispatch is skipped. This
        # test is about the RECORD channel, not delivery — delivery working regardless is
        # what makes losing the target so quiet.
        state = await ex.start(action, 1, agents={"agent-a": sender}, environment=env,
                               message_system=None)
        results = await ex.complete(state, 1, agents={"agent-a": sender}, environment=env,
                                    message_system=None)

        assert results, "send_message must produce a result"
        # THE ASSERTION THAT MATTERS: the recipient survives into the result the record is
        # built from. Without it, action_semantics reports an unaddressed letter.
        assert results[0].action.target.acted_on_agents == ["agent-b"]
        assert state.target.acted_on_agents == ["agent-b"]


class TestSpawnAttribution:
    """Which beat created a thing — only the backend can answer that.

    The observer can diff out "it appeared this step" (the entity table only grows; tombstones
    stay), but when several people act in one step it can't infer who made it. The renderer uses
    this to hold the entrance until the causing beat; otherwise the thing pops up at step start,
    before the narration says he finished.
    """

    def test_a_made_thing_is_reported_as_touched(self) -> None:
        from core.interfaces.action import ActionResult, EntitySpawn
        from engine.executors.base import action_semantics

        action = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                             action_description="拟定换防部署")
        result = ActionResult(
            action=action, expected_outcome="", outcome="常何写完了换防部署令。",
            entity_spawns=[EntitySpawn(name="换防部署令", entity_id="made_x")],
        )

        assert action_semantics(action, result)["affected_entity_ids"] == ["made_x"]

    def test_a_refused_spawn_touched_nothing(self) -> None:
        """World is full → spawn_entity returns no id → it really touched nothing, so don't report."""
        from core.interfaces.action import ActionResult, EntitySpawn
        from engine.executors.base import action_semantics

        action = AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK,
                             action_description="拟定换防部署")
        result = ActionResult(
            action=action, expected_outcome="", outcome="常何写完了换防部署令。",
            entity_spawns=[EntitySpawn(name="换防部署令")],   # entity_id stays empty
        )

        assert action_semantics(action, result)["affected_entity_ids"] == []


class TestAKindIsCarriedThrough:
    """Category has one authority (``WorldEntityType``); nobody along the way may hand-copy a
    whitelist and rule with it."""

    def test_a_kind_is_never_laundered_into_something_else(self) -> None:
        """``core/`` can't validate categories (it can't import ``world/``), so it changes nothing.

        Don't keep a whitelist that launders unrecognized categories into ``object`` ("a thing on
        no list"): when a new category is added that becomes a lie — it is on the list, yet turns
        into something else in traces and ruling text.
        """
        assert Ref.entity("car", "a-kind-that-does-not-exist-yet").kind == "a-kind-that-does-not-exist-yet"
        assert Ref.entity("e1", "item").kind == "item"
        assert Ref.entity("e1").kind == "item"           # default is still item
        assert Ref.entity("e1", "").kind == "item"       # empty string falls back to default; never produces a category-less Ref

    def test_a_free_text_target_keeps_saying_it_is_off_roster(self) -> None:
        """``object`` is a real category (named by free text in decisions), not a bin for unknown
        categories."""
        assert Ref.entity("地上", "object").kind == "object"


class TestWhatTheJudgeIsToldAboutATarget:
    """The name the judge reads depends on two things: whether the world can resolve it, and
    whether it is verbatim text or a menu item."""

    @staticmethod
    def _executor():
        from engine.executors.physical import PhysicalExecutor

        class _Dir:
            def entity_name(self, entity_id: str) -> str:
                return "某物"                      # descriptive fallback on directory miss

        executor = PhysicalExecutor.__new__(PhysicalExecutor)
        executor._directory = _Dir()
        return executor

    def test_free_text_reaches_the_judge_as_the_words_that_were_said(self) -> None:
        """On the free-text branch, ``Ref.id`` is the verbatim phrase, not an id.

        Looking it up as an id in the directory always misses and falls back to "某物" — which
        throws away the only real information on this path: he named "the bloodstain on the
        floor", yet the judge rules on an empty target.
        """
        block = self._executor()._unresolved_target_block("地上的血迹", KIND_OBJECT)
        assert block == "地上的血迹"

    def test_a_menu_bound_target_that_vanished_never_leaks_its_id(self) -> None:
        """On the menu branch, ``Ref.id`` is a real id; it can only miss because the thing was
        destroyed this beat. Ids don't cross the LLM boundary, so fall back to a descriptive
        referent."""
        block = self._executor()._unresolved_target_block("made_blade-1", KIND_ITEM)
        assert "made_blade-1" not in block and block == "某物"

    def test_a_routing_key_is_never_shown_to_the_judge(self) -> None:
        """``object`` / ``item`` are code-layer coordinates; printing them for the judge is
        useless and a layer leak."""
        executor = self._executor()
        for target_id, kind in (("地上的血迹", KIND_OBJECT), ("made_blade-1", KIND_ITEM)):
            block = executor._unresolved_target_block(target_id, kind)
            assert kind not in block, (target_id, kind)
