"""Actor operations preserve perception, parsing constraints and turn authority."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from agent.actor_operations import actor_operations, build_actor_decision_summary, build_actor_operation, requires_actor_interpretation
from agent.decision import DecisionEngine, candidate_lists
from agent.personality import ActionStatus
from core.interfaces.action import ActionType, Ref
from core.interfaces.actor import (
    ActorAction, ActorAnswerRejected, ActorConfirmation, ActorDecision, ActorIntent, ActorIdleAnswer,
    ActorInterruptAnswer,
    ActorOperation, ActorOperationParameters,
)
from core.interfaces.llm import LLMScene
from core.interfaces.perception import NpcIdentity, PerceivedNpc
from engine.run_control import StepWithdrawn
from interaction.api.actor import _decode_answer, _decode_intent
from tests.unit.test_actor_action import AWAY, HERE, _index_of, _packet
from tests.unit.test_actor_runtime import _next_turn, _setup, _work
from tests.unit.test_decision_m4_gaps import _make_personality
from tests.unit.world_setups import make_npc


def _intent(operation: str, target: Ref, *, description: str = "", **parameters) -> ActorIntent:
    return ActorIntent("turn", operation=ActorOperation(operation), target=target,
                       parameters=ActorOperationParameters(**parameters), description=description)


@pytest.mark.parametrize("intent,expected", [
    (ActorIntent("turn", text="我整理账册"), True),
    (_intent("broadcast", Ref.place("palace"), message="请留步"), False),
    (_intent("move", Ref.place("garden")), False),
    (_intent("move", Ref.place("garden"), description="带上他"), True),
    (_intent("rest", Ref.agent("self")), False),
    (_intent("rest", Ref.agent("self"), description="休息一会儿"), True),
    (_intent("work", Ref.agent("self"), description="整理账册"), True),
    (_intent("covert", Ref.agent("self"), description="观察此处的谈话"), True),
    (_intent("send_message", Ref.agent("self"), description="写信告诉他今晚别出门"), True),
])
def test_interpretation_policy_follows_operation_semantics(intent: ActorIntent, expected: bool) -> None:
    assert requires_actor_interpretation(intent) is expected


def test_broadcast_intent_rejects_description() -> None:
    with pytest.raises(ValueError, match="broadcast intent takes no description"):
        _intent("broadcast", Ref.place("palace"), message="请留步", description="悄悄告诉他")


def test_an_intent_target_outside_the_intent_kinds_is_rejected() -> None:
    with pytest.raises(ValueError, match="target.kind"):
        _intent("move", Ref("object", "the ground"))
    with pytest.raises(ValueError, match="target.kind"):
        _decode_intent({
            "turn_id": "turn", "operation": "move", "target": {"kind": "spaceship", "id": "x"},
            "parameters": {},
        })


def test_an_idle_answer_requires_a_true_bool() -> None:
    with pytest.raises(ValueError, match="idle to be true"):
        ActorIdleAnswer(idle=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("operation", ["attack", "restrain"])
def test_specific_physical_shortcuts_are_not_public_operations(operation: str) -> None:
    with pytest.raises(ValueError):
        ActorOperation(operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("destination_index,reason", [(1, None), (None, "errand_without_destination"), (99, "errand_without_destination")])
async def test_description_only_errand_binds_parsed_destination(container, destination_index: int | None, reason: str | None) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    intent = _intent("errand", Ref.npc("runner"), description="去门口看看后回来")
    capability = actor_operations("turn", intent.target, packet).operations[0]
    assert capability.required_parameters == ("description",)
    base = build_actor_operation(intent, packet)
    assert base.errand_npc_id == "runner" and base.errand_destination_id is None
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({
        "decision": "act", "action_description": "我让信使去门口看看", "errand_destination_index": destination_index,
    })
    engine = DecisionEngine(container.llm_router)
    decision, _reason = await engine.interpret_actor_decision(intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    if reason is not None:
        with pytest.raises(ActorAnswerRejected, match=reason):
            engine.bind_actor_action(decision.action, personality=_make_personality(), packet=packet)
    else:
        bound = engine.bind_actor_action(decision.action, personality=_make_personality(), packet=packet)
        assert bound.intent.errand.npc_id == "runner"
        assert bound.intent.errand.destination_id == "xuanwu_gate"


@pytest.mark.asyncio
@pytest.mark.parametrize("destination_index", [1, None])
async def test_description_only_errand_reaches_confirmation_after_full_binding(container, destination_index: int | None) -> None:

    world_id = "operations-errand-description"
    runtime, _messages, agents = _setup(container, world_id)
    environment = runtime._environment
    location_id = environment.space.all_place_ids()[0]
    for agent in agents:
        environment.place_agent(agent_id=agent.agent_id, location_id=location_id)
    assert environment.spawn_npc(make_npc("Runner", gender="male", age=30, description="Runs errands"), location_id=location_id)
    npc_id = environment.all_npcs()[0].npc_id
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    assert turn.briefing.places
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({
        "decision": "act", "action_description": "我请他去看看后回来", "errand_destination_index": destination_index,
    })
    result = await runtime.actor_intent(ActorIntent(
        turn.turn_id, operation=ActorOperation.ERRAND, target=Ref.npc(npc_id),
        description=f"去{turn.briefing.places[0].name}看看后回来",
    ))
    if destination_index is None:
        assert result.status == "rejected" and result.reason == "errand_without_destination"
        assert runtime.actor.pending_turn is turn
        runtime.answer_actor(ActorIdleAnswer())
    else:
        assert result.status == "needs_confirmation" and not task.done()
        assert result.decision_summary.target == Ref.npc(npc_id)
        assert result.decision_summary.parameters["destination_id"] == turn.briefing.places[0].location_id
        runtime.answer_actor(ActorConfirmation(result.decision_summary.decision_summary_id))
    await task
    thread = (await container.snapshot.load(world_id, 1)).actor_thread
    said = next(entry for entry in thread if entry["kind"] == "said")
    assert said["intent"]["parameters"]["destination_id"] is None
    assert said["reason"] == result.reason


@pytest.mark.parametrize("action_type,expected_steps", [
    (ActionType.PHYSICAL, 1), (ActionType.SEND_MESSAGE, 1), (ActionType.ERRAND, 1),
    (ActionType.TALK, 1), (ActionType.WORK, 7), (ActionType.REST, 7), (ActionType.COVERT, 7),
])
@pytest.mark.parametrize("seconds_per_step", [1800, 7200])
def test_summary_duration_uses_execution_authority(action_type: ActionType, expected_steps: int, seconds_per_step: int) -> None:
    action = ActorAction(action_type, "我按计划行动", estimated_steps=7)
    summary = build_actor_decision_summary(ActorDecision(action=action), _packet(), seconds_per_step=seconds_per_step)
    assert summary.estimated_duration_seconds == expected_steps * seconds_per_step


@pytest.mark.parametrize("travel_seconds,expected_steps", [(600, 1), (18000, 2), (21600, 3)])
def test_move_summary_duration_uses_scheduled_travel_time(travel_seconds: int, expected_steps: int) -> None:
    packet = _packet()
    packet.spatial.reachable_locations[0] = replace(packet.spatial.reachable_locations[0], travel_seconds=travel_seconds)
    action = ActorAction(ActionType.MOVE, "我前往门口", estimated_steps=99, destination_id="xuanwu_gate")
    summary = build_actor_decision_summary(ActorDecision(action=action), packet, seconds_per_step=7200)
    assert summary.estimated_duration_seconds == expected_steps * 7200


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["work", "covert", "send_message"])
@pytest.mark.parametrize("description", ["", " \n "])
async def test_self_operations_reject_missing_description_without_parsing(container, operation: str, description: str) -> None:
    runtime, _messages, agents = _setup(container, "operations-description-required")
    agents[0].interpret_actor_decision = AsyncMock(side_effect=AssertionError("An incomplete action must not parse"))
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    result = await runtime.actor_intent(ActorIntent(
        turn.turn_id, operation=ActorOperation(operation), target=Ref.agent(agents[0].agent_id), description=description,
    ))
    assert result.status == "rejected" and result.reason == "description_required"
    assert runtime.actor.pending_turn is turn
    agents[0].interpret_actor_decision.assert_not_awaited()
    runtime.answer_actor(ActorIdleAnswer())
    await task


def test_capabilities_are_grounded_in_perception_and_entity_state() -> None:
    packet = _packet()
    away = actor_operations("turn", Ref.agent(AWAY), packet)
    assert away.operations and all(not c.available and c.reason == "target_not_perceived" for c in away.operations)
    current = actor_operations("turn", Ref.place(packet.spatial.location_id), packet)
    assert [(c.operation.value, c.available) for c in current.operations] == [("move", False), ("broadcast", True)]
    remote = actor_operations("turn", Ref.place("xuanwu_gate"), packet)
    assert remote.operations[0].available and remote.operations[1].reason == "broadcast_requires_current_location"
    sword = packet.spatial.visible_entities[0]
    sword.is_takeable = True
    owned = actor_operations("turn", Ref.entity("sword"), packet)
    assert owned.operations[0].reason == "already_held"
    assert all(not c.available for c in actor_operations("turn", Ref.entity("sword", "landmark"), packet).operations)
    sword.holder_id = None
    assert actor_operations("turn", Ref.entity("sword"), packet).operations[0].available
    sword.is_takeable = False
    assert actor_operations("turn", Ref.entity("sword"), packet).operations[0].reason == "item_not_takeable"


@pytest.mark.parametrize("operation,target,parameters,expected", [
    ("move", Ref.place("xuanwu_gate"), {}, ActionType.MOVE),
    ("talk", Ref.agent(HERE), {}, ActionType.TALK),
    ("physical", Ref.agent(HERE), {}, ActionType.PHYSICAL),
    ("use", Ref.entity("sword"), {}, ActionType.PHYSICAL),
    ("destroy", Ref.entity("sword"), {}, ActionType.PHYSICAL),
])
def test_explicit_operations_bind_through_existing_action_rules(container, operation, target, parameters, expected) -> None:
    packet = _packet()
    intent = _intent(operation, target, description="轻轻推开他" if operation == "physical" else "", **parameters)
    action = build_actor_operation(intent, packet)
    bound = DecisionEngine(container.llm_router).bind_actor_action(action, personality=_make_personality(), packet=packet)
    assert bound.action_type is expected
    assert bound.target.acts_on == [target]
    decision_summary = build_actor_decision_summary(ActorDecision(action=action), packet, seconds_per_step=3600, intent=intent)
    assert decision_summary.operation == operation and decision_summary.target == target
    assert "action" not in vars(decision_summary)


def test_broadcast_and_errand_parameters_are_authoritative(container) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    engine = DecisionEngine(container.llm_router)
    broadcast = build_actor_operation(_intent("broadcast", Ref.place(packet.spatial.location_id), message="请来这里"), packet)
    assert broadcast.message_content == "请来这里" and broadcast.message_announce
    errand = build_actor_operation(_intent("errand", Ref.npc("runner"), destination_id="xuanwu_gate",
                                           item_id="sword", recipient_id=AWAY, message="请来这里", description="请他带话"), packet)
    bound = engine.bind_actor_action(errand, personality=_make_personality(), packet=packet)
    assert bound.intent.errand.message == "请来这里" and bound.intent.errand.recipient_id == AWAY
    assert "男" in errand.action_description
    with pytest.raises(ActorAnswerRejected, match="description_required"):
        build_actor_operation(_intent("errand", Ref.npc("runner"), destination_id="xuanwu_gate", message="你好"), packet)
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使"), busy=True)
    assert actor_operations("turn", Ref.npc("runner"), packet).operations[0].reason == "npc_busy"


@pytest.mark.asyncio
async def test_errand_address_conflicts_are_refused_by_backend(container) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    parameters = {"destination_id": "xuanwu_gate", "message": "请来这里", "announce": True}
    with pytest.raises(ActorAnswerRejected, match="errand_address_conflict"):
        build_actor_operation(_intent("errand", Ref.npc("runner"), recipient_id=HERE, description="请他带话", **parameters), packet)
    intent = _intent("errand", Ref.npc("runner"), description="交代清楚", **parameters)
    base = build_actor_operation(intent, packet)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "action_description": "我让他传话", "errand_recipient_index": 1,
    })
    with pytest.raises(ActorAnswerRejected, match="errand_address_conflict"):
        await DecisionEngine(container.llm_router).interpret_actor_decision(
            intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base,
        )


@pytest.mark.parametrize("payload", [
    {"turn_id": "t", "text": 3},
    {"turn_id": "t", "text": None},
    {"turn_id": "t", "text": "去", "operation": "move"},
    {"turn_id": "t", "text": "去", "description": "小心"},
    {"turn_id": "t", "operation": "move", "target": {"kind": "location", "id": "p"}, "parameters": {"message": "hello"}},
    {"turn_id": "t", "operation": "move", "target": {"kind": "location", "id": "p"}, "parameters": {}, "skip_parse": True},
    {"turn_id": "t", "operation": "move", "target": {"kind": "location", "id": 2}, "parameters": {}},
    {"turn_id": "t", "operation": "move", "target": {"kind": "entity", "id": "e"}, "parameters": {}},
    {"turn_id": "t", "operation": "broadcast", "target": {"kind": "location", "id": "p"}, "parameters": {"message": True}},
    {"turn_id": "t", "operation": "errand", "target": {"kind": "npc", "id": "n"}, "parameters": {"announce": "true"}},
    {"turn_id": "t", "operation": "move", "target": {"kind": "location", "id": "p"}, "parameters": [], "description": "小心"},
    {"turn_id": "t", "operation": "move", "target": {"kind": "location", "id": "p"}, "parameters": {}, "description": None},
])
def test_intent_decoder_rejects_wrong_types_and_extra_fields(payload) -> None:
    with pytest.raises((TypeError, ValueError)):
        _decode_intent(payload)


def test_intent_description_matches_capability_requirements() -> None:
    packet = _packet()
    target = Ref.agent(HERE)
    capability = next(c for c in actor_operations("turn", target, packet).operations
                      if c.operation is ActorOperation.PHYSICAL)
    assert capability.required_parameters == ("description",)
    payload = {"turn_id": "turn", "operation": "physical",
               "target": {"kind": "agent", "id": HERE}, "parameters": {}}
    with pytest.raises(ActorAnswerRejected, match="description_required"):
        build_actor_operation(_decode_intent(payload), packet)
    intent = _decode_intent({**payload, "description": "轻轻推开他"})
    assert intent.description == "轻轻推开他"
    assert build_actor_operation(intent, packet).action_type is ActionType.PHYSICAL
    with pytest.raises(TypeError, match="unexpected keyword argument 'details'"):
        _decode_intent({**payload, "details": "轻轻推开他"})


def test_confirmation_decoder_never_accepts_execution_slots() -> None:
    assert _decode_answer({"decision_summary_id": "r"}) == ActorConfirmation("r")
    assert _decode_answer({"idle": True}) == ActorIdleAnswer()
    for payload in ({"decision_summary_id": 3}, {"decision_summary_id": " "}, {"decision_summary_id": "r", "idle": True},
                    {"reading_id": "r"}, {"action_id": "r"},
                    {"idle": False}, {"idle": 1}, {"idle": "true"}, {"idle": True, "extra": True},
                    {"action": {"action_type": "rest", "action_description": "休息"}}):
        with pytest.raises((TypeError, ValueError)):
            _decode_answer(payload)


def test_interrupt_answer_decodes_optional_thought() -> None:
    assert _decode_answer({"interrupt": True}) == ActorInterruptAnswer(True)
    assert _decode_answer({"interrupt": True, "thought": "我得先处理来信"}) == ActorInterruptAnswer(True, "我得先处理来信")
    assert _decode_answer({"interrupt": False, "thought": "先把手上的事做完"}) == ActorInterruptAnswer(False, "先把手上的事做完")
    for thought in (None, 3, False, []):
        with pytest.raises(TypeError):
            _decode_answer({"interrupt": True, "thought": thought})


def test_move_summary_preserves_the_carried_people_concept() -> None:
    action = ActorAction(ActionType.MOVE, "我带他离开", destination_id="xuanwu_gate", move_carry_ids=(HERE,))
    summary = build_actor_decision_summary(ActorDecision(action=action), _packet(), seconds_per_step=3600)
    assert summary.target == Ref.place("xuanwu_gate")
    assert summary.parameters == {"move_carry_ids": [HERE]}


def test_summary_preserves_the_item_name_concept() -> None:
    action = ActorAction(ActionType.PHYSICAL, "我检查门闩", item_name="门闩")
    summary = build_actor_decision_summary(ActorDecision(action=action), _packet(), seconds_per_step=3600)
    assert summary.target is None
    assert summary.parameters == {"item_name": "门闩"}


@pytest.mark.parametrize("idle", [False, 0, 1, "true", None])
def test_idle_answer_requires_explicit_true(idle) -> None:
    with pytest.raises(ValueError):
        ActorIdleAnswer(idle=idle)


def _fixed_part(engine: DecisionEngine, intent, packet) -> tuple[str, dict]:
    """The 【已定下的】 section of the reading prompt for ``intent``, and the fixed slots."""
    from agent.decision import _fixed_intent_payload

    base = build_actor_operation(intent, packet)
    fixed = _fixed_intent_payload(packet, intent, base)
    user = engine._intent_user_prompt(  # noqa: SLF001
        intent.description, _make_personality(), packet, recent=(), recalled=(), fixed=(base, fixed),
    )
    return user[user.index("【已定下的】"):user.index("【意图】")], fixed


@pytest.mark.parametrize("operation,fixed_keys", [
    ("move", {"selected_index", "destination_index"}),
    ("talk", {"selected_index", "person_indices"}),
    ("errand", {"selected_index", "errand_npc_index", "errand_destination_index", "errand_message"}),
])
def test_the_fixed_part_of_an_operation_is_shown_as_slots_the_reading_keeps(container, operation, fixed_keys) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    targets = {"move": Ref.place("xuanwu_gate"),
               "talk": Ref.agent(HERE), "errand": Ref.npc("runner")}
    parameters = {"message": "请来这里"} if operation == "errand" else {}
    if operation == "errand":
        parameters["destination_id"] = "xuanwu_gate"
    intent = _intent(operation, targets[operation], description="平静地做", **parameters)

    section, fixed = _fixed_part(DecisionEngine(container.llm_router), intent, packet)

    shown = json.loads(section.split("已定下的栏（序号沿用上面的名单）：", 1)[1].splitlines()[0])
    assert fixed_keys <= shown.keys()
    # What the reading fills is never shown as fixed.
    assert not {"action_description", "estimated_steps", "expected_outcome", "urgency"} & shown.keys()
    assert fixed_keys <= fixed.keys()


def test_operation_prompt_preserves_explicit_false_announce(container) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    intent = _intent("errand", Ref.npc("runner"), description="顺便看看那边的情况",
                     destination_id="xuanwu_gate", announce=False)
    section, fixed = _fixed_part(DecisionEngine(container.llm_router), intent, packet)
    assert fixed["errand_announce"] is False
    assert '"errand_announce": false' in section


@pytest.mark.asyncio
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("response", [None, "not JSON", '{"decision": "unclear"}', '{"decision": "act", "selected_index": -1}'])
async def test_interpretation_paths_share_failure_handling(container, constrained: bool, response: str | None) -> None:
    packet = _packet()
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    if response is None:
        provider.complete = AsyncMock(side_effect=RuntimeError("Provider unavailable"))
    else:
        provider.fixed_response = response
    intent = _intent("talk", Ref.agent(HERE), description="平静地说") if constrained else None
    base = build_actor_operation(intent, packet) if intent is not None else None
    result = await DecisionEngine(container.llm_router).interpret_actor_decision(
        "平静地说", personality=_make_personality(), packet=packet, intent=intent, base=base,
    )
    # Only an unclear reading has reasoning to keep; a failed call or an unbuildable act has none.
    assert result == ((None, "") if response == '{"decision": "unclear"}' else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["type", "target", "message"])
async def test_constrained_interpretation_cannot_change_authority(container, change: str) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    intent = _intent("errand", Ref.npc("runner"), destination_id="xuanwu_gate",
                     message="请来这里", recipient_id=AWAY, description="说得温和")
    base = build_actor_operation(intent, packet)
    payload = {"decision": "act", "selected_index": _index_of(ActionType.ERRAND),
               "errand_message": "请来这里", "action_description": "我请信使温和地传话"}
    if change == "type":
        payload["selected_index"] = _index_of(ActionType.MOVE)
    elif change == "target":
        payload["errand_recipient_index"] = 1
    else:
        payload["errand_message"] = "别来这里"
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps(payload)
    with pytest.raises(ActorAnswerRejected, match="operation_constraint_conflict"):
        await DecisionEngine(container.llm_router).interpret_actor_decision(
            intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)


@pytest.mark.asyncio
async def test_send_reading_without_description_describes_what_it_bound(container) -> None:
    """The operation's placeholder ("我打算发送消息") names no one; a reading that writes no
    description is described from the recipients and words it bound."""
    packet = _packet()
    intent = _intent("send_message", Ref.agent(packet.agent_id), description="告诉李元吉明早入宫")
    base = build_actor_operation(intent, packet)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "person_indices": [2], "message_content": "明早入宫。",
    })
    decision, _reason = await DecisionEngine(container.llm_router).interpret_actor_decision(
        intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    assert decision.action.action_description == "我打算发消息给李元吉说：「明早入宫。」"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,extra", [
    ("move", {"person_indices": [1], "physical_entity_index": 1}),
    ("talk", {"physical_person_index": 2, "destination_index": 1}),
    ("physical", {"person_indices": [1], "message_content": "多余话语", "move_carry_indices": [1]}),
    ("work", {"person_indices": [1], "physical_person_index": 1}),
    ("rest", {"destination_index": 1, "errand_announce": True}),
    ("covert", {"physical_entity_index": 1, "message_announce": True}),
    ("errand", {"person_indices": [1], "destination_index": 1}),
])
async def test_selected_operation_ignores_unrelated_output_fields(container, operation: str, extra: dict) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    targets = {"move": Ref.place("xuanwu_gate"), "talk": Ref.agent(HERE),
               "physical": Ref.agent(HERE), "errand": Ref.npc("runner")}
    target = targets.get(operation, Ref.agent(packet.agent_id))
    parameters = {"destination_id": "xuanwu_gate"} if operation == "errand" else {}
    intent = _intent(operation, target, description="具体行动", **parameters)
    base = build_actor_operation(intent, packet)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "action_description": "我照此行动", "unknown_field": {"noise": [1]},
        "inner_monologue": "不属于意图解读的独白", "expected_outcome": "让事情有个着落", **extra,
    })
    engine = DecisionEngine(container.llm_router)
    decision, _reason = await engine.interpret_actor_decision(intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    # The reading fills the decision around the fixed part: its description and expectation.
    assert decision.action == replace(base, action_description="我照此行动", expected_outcome="让事情有个着落")
    bound = engine.bind_actor_action(decision.action, personality=_make_personality(), packet=packet)
    assert bound.target == engine.bind_actor_action(base, personality=_make_personality(), packet=packet).target


@pytest.mark.asyncio
@pytest.mark.parametrize("target_field,target_index", [
    ("physical_person_index", 2), ("physical_npc_index", 1), ("physical_entity_index", 1),
])
async def test_selected_physical_operation_rejects_rewritten_targets(container, target_field: str, target_index: int) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    intent = _intent("physical", Ref.agent(HERE), description="搀扶他坐下")
    base = build_actor_operation(intent, packet)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "action_description": "我扶他坐下", target_field: target_index,
    })
    with pytest.raises(ActorAnswerRejected, match="operation_constraint_conflict"):
        await DecisionEngine(container.llm_router).interpret_actor_decision(
            intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)


@pytest.mark.asyncio
async def test_helping_a_person_with_redundant_person_indices_reaches_confirmation(container) -> None:
    world_id = "operations-help-person"
    runtime, _messages, agents = _setup(container, world_id, other_at="palace")
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act",
        "reason": "搀扶眼前的人坐下", "person_indices": [1],
        "action_description": "我搀住他的手臂，扶他坐下。", "estimated_steps": 1,
    })
    result = await runtime.actor_intent(ActorIntent(
        turn.turn_id, operation=ActorOperation.PHYSICAL, target=Ref.agent(agents[1].agent_id), description="搀扶他坐下",
    ))
    assert result.status == "needs_confirmation"
    assert result.decision_summary.target == Ref.agent(agents[1].agent_id)
    assert result.decision_summary.operation == "physical" and result.decision_summary.parameters == {}
    runtime.answer_actor(ActorIdleAnswer())
    await task


@pytest.mark.asyncio
async def test_a_fixed_talk_takes_its_topic_from_the_description(container) -> None:
    packet = _packet()
    intent = _intent("talk", Ref.agent(HERE), description="借这个机会问问秋猎的事")
    base = build_actor_operation(intent, packet)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "reason": "当面问他秋猎的事", "decision": "act", "selected_index": _index_of(ActionType.TALK),
        "person_indices": [1], "action_description": "我借机问他秋猎的事", "talk_topic": "秋猎的事",
    })
    decision, _reason = await DecisionEngine(container.llm_router).interpret_actor_decision(
        intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    assert (decision.action.person_ids[0], decision.action.talk_topic) == (HERE, "秋猎的事")


@pytest.mark.asyncio
async def test_an_intent_that_conflicts_with_the_fixed_operation_reads_as_nothing(container) -> None:
    packet = _packet()
    packet.spatial.visible_entities[1].is_takeable = True
    intent = _intent("take", Ref.entity("memorial"), description="把它烧掉")
    base = build_actor_operation(intent, packet)
    for reading, expected in (({"reason": "拿起不是烧毁", "decision": "unclear"}, (None, "拿起不是烧毁")),
                              # Doing nothing is no reading of an intent that fixed an act.
                              ({"reason": "先不动", "decision": "idle"}, None)):
        container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps(reading)
        assert await DecisionEngine(container.llm_router).interpret_actor_decision(
            intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base) == expected


@pytest.mark.asyncio
async def test_constrained_interpretation_preserves_parameters_and_requires_confirmation(container) -> None:
    runtime, _messages, agents = _setup(container, "operations-constraint")
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({
        "decision": "act", "action_description": "我整理账册",
    })
    result = await runtime.actor_intent(ActorIntent(turn.turn_id, operation=ActorOperation.WORK,
        target=Ref.agent(agents[0].agent_id), description="整理账册"))
    assert result.status == "needs_confirmation" and not task.done()
    assert result.decision_summary.operation == "work"
    assert result.decision_summary.target == Ref.agent(agents[0].agent_id)
    runtime.answer_actor(ActorConfirmation(result.decision_summary.decision_summary_id))
    with pytest.raises(ActorAnswerRejected, match="decision_summary_not_pending"):
        runtime.answer_actor(ActorConfirmation(result.decision_summary.decision_summary_id))
    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("description", ["", " \n "])
async def test_direct_submission_bypasses_parser_and_records_only_public_data(container, description: str) -> None:
    world_id = "operations-direct"
    runtime, _messages, agents = _setup(container, world_id)
    agents[0].interpret_actor_decision = AsyncMock(side_effect=AssertionError("must not parse"))
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    result = await runtime.actor_intent(ActorIntent(turn.turn_id, operation=ActorOperation.BROADCAST,
        target=Ref.place("palace"), parameters=ActorOperationParameters(message="请来这里"), description=description))
    assert result.status == "submitted" and runtime.actor.pending_turn is None
    agents[0].interpret_actor_decision.assert_not_awaited()
    await task
    thread = (await container.snapshot.load(world_id, 1)).actor_thread
    said = next(e for e in thread if e["kind"] == "said")
    answered = next(e for e in thread if e["kind"] == "answered")
    assert said["decision_summary"]["operation"] == "broadcast"
    assert said["intent"]["description"] == description
    assert said["decision_summary"]["description"] == "我打算向在场众人发通知说：「请来这里」"
    assert said["decision_summary"]["parameters"]["message"] == "请来这里"
    assert answered["answer"] == {"decision_summary_id": said["decision_summary"]["decision_summary_id"]}
    assert "action_type" not in json.dumps(thread)


@pytest.mark.asyncio
async def test_unclear_words_keep_where_the_reading_got_stuck(container) -> None:
    world_id = "operations-unclear-reading"
    runtime, _messages, agents = _setup(container, world_id)
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    agents[0].interpret_actor_decision = AsyncMock(return_value=(None, "只是一声叹息，没说要做什么"))
    assert (await runtime.actor_intent(ActorIntent(turn.turn_id, text="唉"))).status == "rejected"
    agents[0].interpret_actor_decision = AsyncMock(return_value=(ActorDecision(idle=True), "说的是先等等"))
    summary = (await runtime.actor_intent(ActorIntent(turn.turn_id, text="先等等看"))).decision_summary
    runtime.answer_actor(ActorConfirmation(summary.decision_summary_id))
    await task

    said = [e for e in (await container.snapshot.load(world_id, 1)).actor_thread if e["kind"] == "said"]
    assert [(e["reason"], e["reading"]) for e in said] == [("unreadable", "只是一声叹息，没说要做什么"), (None, "")]


@pytest.mark.asyncio
async def test_new_failed_intent_invalidates_old_confirmation(container) -> None:
    runtime, _messages, agents = _setup(container, "operations-invalidation")
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    agents[0].interpret_actor_decision = AsyncMock(return_value=(ActorDecision(action=_work()), ""))
    first = await runtime.actor_intent(ActorIntent(turn.turn_id, text="工作"))
    agents[0].interpret_actor_decision.return_value = None
    assert (await runtime.actor_intent(ActorIntent(turn.turn_id, text="唉"))).status == "rejected"
    with pytest.raises(ActorAnswerRejected, match="decision_summary_not_pending"):
        runtime.answer_actor(ActorConfirmation(first.decision_summary.decision_summary_id))
    runtime.answer_actor(ActorIdleAnswer())
    await task


@pytest.mark.asyncio
async def test_words_read_as_idling_are_confirmed_like_any_reading(container) -> None:
    from core.interfaces.actor import IDLE_OPERATION

    runtime, _messages, agents = _setup(container, "operations-idle-reading")
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    agents[0].interpret_actor_decision = AsyncMock(return_value=(ActorDecision(idle=True), "说的是先等等"))

    result = await runtime.actor_intent(ActorIntent(turn.turn_id, text="先等等看"))

    assert result.status == "needs_confirmation" and not task.done()
    summary = result.decision_summary
    assert (summary.operation, summary.description, summary.target) == (IDLE_OPERATION, "", None)
    runtime.answer_actor(ActorConfirmation(summary.decision_summary_id))
    await task
    assert agents[0].personality.state.action_status != ActionStatus.IN_PROGRESS


@pytest.mark.asyncio
@pytest.mark.parametrize("newer", ["failure", "direct", "withdraw"])
async def test_inflight_old_intent_cannot_publish_after_newer_request_or_closed_turn(container, newer: str) -> None:
    runtime, _messages, agents = _setup(container, f"operations-stale-{newer}")
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    entered, release = asyncio.Event(), asyncio.Event()

    async def interpret(text, packet):
        if text == "first":
            entered.set()
            await release.wait()
            return ActorDecision(action=_work()), ""
        return None

    agents[0].interpret_actor_decision = interpret
    old = asyncio.create_task(runtime.actor_intent(ActorIntent(turn.turn_id, text="first")))
    await entered.wait()
    if newer == "failure":
        await runtime.actor_intent(ActorIntent(turn.turn_id, text="second"))
    elif newer == "direct":
        await runtime.actor_intent(ActorIntent(turn.turn_id, operation=ActorOperation.BROADCAST,
            target=Ref.place("palace"), parameters=ActorOperationParameters(message="请来这里")))
    else:
        runtime.actor.abandon()
    release.set()
    result = await old
    assert result.status == "rejected" and result.reason == "stale_intent"
    assert all(e.get("words") != "first" for e in runtime.actor_live_thread)
    if newer == "failure":
        runtime.answer_actor(ActorIdleAnswer())
    if newer == "withdraw":
        with pytest.raises(StepWithdrawn):
            await task
    else:
        await task


@pytest.mark.asyncio
async def test_constrained_long_verbatim_message_is_preserved_without_model_echo(container) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    message = "请暂时留在原处，等我回来再说。" * 25
    operation = "errand"
    intent = _intent(operation, Ref.npc("runner"), description="平静地说",
                     message=message, destination_id="xuanwu_gate", recipient_id=AWAY)
    base = build_actor_operation(intent, packet)
    provider = container.llm_router.get(LLMScene.ACTOR_INTENT)
    provider.fixed_response = json.dumps({
        "decision": "act", "action_description": "我平静地交代这番话",
    })
    provider.complete = AsyncMock(wraps=provider.complete)
    result = await DecisionEngine(container.llm_router).interpret_actor_decision(
        intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    action = result[0].action
    assert action.errand_message == message
    system = provider.complete.call_args.args[0][0].content
    assert message not in system


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["move", "physical", "talk", "errand"])
async def test_constrained_details_can_fill_unspecified_business_parameters(container, operation: str) -> None:
    packet = _packet()
    packet.spatial.visible_npcs["runner"] = PerceivedNpc(NpcIdentity("信使", "男"))
    targets = {"move": Ref.place("xuanwu_gate"), "physical": Ref.entity("sword"),
               "talk": Ref.agent(HERE), "errand": Ref.npc("runner")}
    parameters = {"destination_id": "xuanwu_gate", "message": "请来这里"} if operation == "errand" else {}
    intent = _intent(operation, targets[operation], description="也照顾到在场的那个人", **parameters)
    base = build_actor_operation(intent, packet)
    payload = {"decision": "act", "action_description": "我照此行动"}
    if operation == "move":
        payload["move_carry_indices"] = [1]
    elif operation == "physical":
        payload["physical_recipient_index"] = 1
    elif operation == "talk":
        payload["person_indices"] = [1]
        payload["message_content"] = "我们说说话"
    else:
        payload["errand_recipient_index"] = 1
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps(payload)
    engine = DecisionEngine(container.llm_router)
    result = await engine.interpret_actor_decision(
        intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    bound = engine.bind_actor_action(result[0].action, personality=_make_personality(), packet=packet)
    if operation == "move":
        assert bound.target.claims == [Ref.agent(HERE)]
    elif operation == "physical":
        assert bound.target.reaches == [Ref.agent(HERE)]
    elif operation == "talk":
        assert bound.target.acts_on[0] == Ref.agent(HERE)
    else:
        assert bound.intent.errand.recipient_id == HERE and bound.intent.errand.message == "请来这里"


@pytest.mark.asyncio
@pytest.mark.parametrize("intent_factory,reason", [
    (lambda t: ActorIntent(t, operation=ActorOperation.MOVE, target=Ref.place("unseen")), "target_not_perceived"),
    (lambda t: ActorIntent(t, operation=ActorOperation.BROADCAST, target=Ref.place("palace")), "message_required"),
    (lambda t: ActorIntent(t, operation=ActorOperation.TAKE, target=Ref.place("palace")), "operation_not_applicable"),
])
async def test_invalid_explicit_operation_is_rejected_without_parsing_and_keeps_turn(container, intent_factory, reason) -> None:
    runtime, _messages, agents = _setup(container, f"operations-reject-{reason}")
    agents[0].interpret_actor_decision = AsyncMock(side_effect=AssertionError("must not parse"))
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    result = await runtime.actor_intent(intent_factory(turn.turn_id))
    assert result.status == "rejected" and result.reason == reason
    assert runtime.actor.pending_turn == turn
    agents[0].interpret_actor_decision.assert_not_awaited()
    runtime.answer_actor(ActorIdleAnswer())
    await task


@pytest.mark.asyncio
async def test_shortcut_broadcast_records_the_spoken_message_in_sender_feedback(container) -> None:
    from agent.memory_types import MemoryStream

    world_id = "operations-broadcast-feedback"
    runtime, _messages, agents = _setup(container, world_id, other_at="palace")
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    message = "大家准备好了吗？请留在这里！"
    result = await runtime.actor_intent(ActorIntent(
        turn.turn_id, operation=ActorOperation.BROADCAST, target=Ref.place("palace"),
        parameters=ActorOperationParameters(message=message),
    ))
    assert result.status == "submitted"
    await task
    snapshot = await container.snapshot.load(world_id, 1)
    ended = next(entry for entry in snapshot.actor_thread if entry["kind"] == "ended")
    assert message in ended["fact"]
    assert "宣告：「我向此处的人广播」" not in ended["fact"]
    memories = agents[0].memory_system._entries.values()  # noqa: SLF001
    assert any(memory.stream is MemoryStream.FACTUAL and message in memory.raw_content for memory in memories)


def test_self_capabilities_are_distinct_from_other_people() -> None:
    packet = _packet()
    capabilities = actor_operations("turn", Ref.agent(packet.agent_id), packet)
    assert [(c.operation.value, c.available) for c in capabilities.operations] == [
        ("work", True), ("rest", True), ("covert", True), ("send_message", True),
    ]
    assert not {"work", "rest", "covert", "send_message"}.intersection(
        c.operation.value for c in actor_operations("turn", Ref.agent(HERE), packet).operations
    )
    assert [c.operation.value for c in actor_operations("turn", Ref.agent(HERE), packet).operations] == ["talk", "physical"]
    for operation in ("talk", "physical"):
        with pytest.raises(ActorAnswerRejected, match="operation_not_applicable"):
            build_actor_operation(_intent(operation, Ref.agent(packet.agent_id), description="照此做"), packet)


@pytest.mark.parametrize("operation,action_type", [
    ("work", ActionType.WORK), ("rest", ActionType.REST), ("covert", ActionType.COVERT),
])
def test_self_operations_preserve_targetless_execution_and_public_identity(container, operation, action_type) -> None:
    packet = _packet()
    intent = _intent(operation, Ref.agent(packet.agent_id), description="具体行动")
    action = build_actor_operation(intent, packet)
    bound = DecisionEngine(container.llm_router).bind_actor_action(action, personality=_make_personality(), packet=packet)
    assert bound.action_type is action_type
    assert not bound.target.acts_on and not bound.target.claims and not bound.target.reaches
    decision_summary = build_actor_decision_summary(ActorDecision(action=action), packet, seconds_per_step=3600, intent=intent)
    assert decision_summary.target == Ref.agent(packet.agent_id) and decision_summary.operation == operation
    assert "某人" not in decision_summary.description
    if operation == "covert":
        assert decision_summary.description == "我打算暗中探查"
        assert "前往" not in decision_summary.description and "潜行" not in decision_summary.description


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["rest"])
async def test_direct_self_operations_bypass_parser_and_persist_request(container, operation: str) -> None:
    world_id = f"operations-self-{operation}"
    runtime, _messages, agents = _setup(container, world_id)
    agents[0].interpret_actor_decision = AsyncMock(side_effect=AssertionError("must not parse"))
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    assert turn.agent_id == agents[0].agent_id
    assert "character" not in vars(turn.briefing)
    assert agents[0].agent_id not in {person.agent_id for person in turn.briefing.people}
    target = Ref.agent(agents[0].agent_id)
    result = await runtime.actor_intent(ActorIntent(turn.turn_id, operation=ActorOperation(operation), target=target))
    assert result.status == "submitted"
    agents[0].interpret_actor_decision.assert_not_awaited()
    await task
    thread = (await container.snapshot.load(world_id, 1)).actor_thread
    said = next(e for e in thread if e["kind"] == "said")
    answered = next(e for e in thread if e["kind"] == "answered")
    assert said["intent"]["operation"] == said["decision_summary"]["operation"] == operation
    assert said["intent"]["target"] == said["decision_summary"]["target"] == {"kind": "agent", "id": target.id}
    assert answered["answer"] == {"decision_summary_id": said["decision_summary"]["decision_summary_id"]}
    assert "action_type" not in json.dumps(thread)


@pytest.mark.asyncio
async def test_rejected_explicit_intent_persists_action_target_parameters_and_description(container) -> None:
    world_id = "operations-rejected-record"
    runtime, _messages, agents = _setup(container, world_id)
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    result = await runtime.actor_intent(ActorIntent(
        turn.turn_id, operation=ActorOperation.ERRAND, target=Ref.npc("unseen"),
        parameters=ActorOperationParameters(destination_id="palace", message="请来这里"), description="语气温和",
    ))
    assert result.status == "rejected"
    runtime.answer_actor(ActorIdleAnswer())
    await task
    thread = (await container.snapshot.load(world_id, 1)).actor_thread
    said = next(e for e in thread if e["kind"] == "said")
    assert said["intent"]["operation"] == "errand"
    assert said["intent"]["target"] == {"kind": "npc", "id": "unseen"}
    assert said["intent"]["parameters"]["message"] == "请来这里"
    assert said["intent"]["description"] == "语气温和"
    assert said["decision_summary"] is None and said["reason"] == result.reason
    assert next(e for e in thread if e["kind"] == "answered")["answer"] == {"idle": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["work", "rest", "covert"])
async def test_self_operation_ignores_unrelated_target_slots(container, operation: str) -> None:
    packet = _packet()
    intent = _intent(operation, Ref.agent(packet.agent_id), description="照此做")
    base = build_actor_operation(intent, packet)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "action_description": "我照此做", "physical_person_index": 1,
    })
    engine = DecisionEngine(container.llm_router)
    decision, _reason = await engine.interpret_actor_decision(
        intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    bound = engine.bind_actor_action(decision.action, personality=_make_personality(), packet=packet)
    assert not bound.target.acts_on and not bound.target.claims and not bound.target.reaches


async def _read_self_message(container, packet, *, reading_anonymous: bool, **parameters):
    intent = _intent("send_message", Ref.agent(packet.agent_id), description="告诉他今晚别出门", **parameters)
    base = build_actor_operation(intent, packet)
    recipient = candidate_lists(packet).roster_ids.index(AWAY) + 1
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "action_description": "我给他递话", "person_indices": [recipient],
        "message_content": "今晚别出门", "anonymous": reading_anonymous,
    })
    engine = DecisionEngine(container.llm_router)
    decision, _reason = await engine.interpret_actor_decision(
        intent.description, personality=_make_personality(), packet=packet, intent=intent, base=base)
    return intent, decision, engine


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_self_send_message_reads_recipients_words_and_an_unset_signature(container, anonymous: bool) -> None:
    packet = _packet()
    intent, decision, engine = await _read_self_message(container, packet, reading_anonymous=anonymous)
    bound = engine.bind_actor_action(decision.action, personality=_make_personality(), packet=packet)
    assert bound.action_type is ActionType.SEND_MESSAGE
    assert bound.target.acts_on == [Ref.agent(AWAY)]
    assert bound.content == "今晚别出门" and bound.anonymous is anonymous
    summary = build_actor_decision_summary(decision, packet, seconds_per_step=3600, intent=intent)
    assert summary.operation == "send_message" and summary.target == Ref.agent(packet.agent_id)
    assert summary.parameters["person_ids"] == [AWAY] and summary.parameters["anonymous"] is anonymous


@pytest.mark.asyncio
@pytest.mark.parametrize("anonymous", [False, True])
async def test_self_send_message_keeps_a_set_signature(container, anonymous: bool) -> None:
    packet = _packet()
    _, decision, _ = await _read_self_message(
        container, packet, reading_anonymous=anonymous, anonymous=anonymous)
    assert decision.action.anonymous is anonymous
    with pytest.raises(ActorAnswerRejected, match="operation_constraint_conflict"):
        await _read_self_message(container, packet, reading_anonymous=not anonymous, anonymous=anonymous)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["work", "covert"])
async def test_parsed_self_operation_records_selected_operation_and_description(container, operation: str) -> None:
    world_id = "operations-self-decision_summary-record"
    runtime, _messages, agents = _setup(container, world_id)
    task = asyncio.create_task(runtime.run_step(agents))
    turn = await _next_turn(runtime, task)
    container.llm_router.get(LLMScene.ACTOR_INTENT).fixed_response = json.dumps({
        "decision": "act", "action_description": "我整理手头的账册",
    })
    result = await runtime.actor_intent(ActorIntent(
        turn.turn_id, operation=ActorOperation(operation), target=Ref.agent(agents[0].agent_id), description="整理账册",
    ))
    assert result.status == "needs_confirmation" and not task.done()
    runtime.answer_actor(ActorConfirmation(result.decision_summary.decision_summary_id))
    await task
    thread = (await container.snapshot.load(world_id, 1)).actor_thread
    said = next(e for e in thread if e["kind"] == "said")
    assert said["intent"]["operation"] == operation and said["intent"]["description"] == "整理账册"
    assert said["intent"]["target"] == {"kind": "agent", "id": agents[0].agent_id}
    assert said["decision_summary"]["description"] == "我整理手头的账册"


@pytest.mark.parametrize("place_id,place_name", [("harbor-7", "海港"), ("deck-a", "星舰甲板")])
def test_operation_targets_use_runtime_world_facts_not_theme_presets(container, place_id: str, place_name: str) -> None:
    from dataclasses import replace
    from core.interfaces.perception import LocationView

    packet = _packet()
    packet.spatial.reachable_locations = [replace(packet.spatial.reachable_locations[0], location_id=place_id,
                                                 view=LocationView(name=place_name))]
    action = build_actor_operation(_intent("move", Ref.place(place_id)), packet)
    bound = DecisionEngine(container.llm_router).bind_actor_action(action, personality=_make_personality(), packet=packet)
    assert bound.target.acts_on == [Ref.place(place_id)]
    assert place_name in action.action_description
