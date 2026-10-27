"""Explicit actor operations, grounded in the same perception as ordinary decisions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from core.interfaces.action import ActionType, Ref
from core.action_duration import action_duration_steps, travel_duration_steps
from core.prompts import person_referent
from core.interfaces.actor import (
    IDLE_OPERATION, ActorAction, ActorAnswerRejected, ActorDecision, ActorDecisionSummary, ActorIntent,
    ActorOperation, ActorOperationCapability, ActorOperations,
)

if TYPE_CHECKING:
    from agent.perception import PerceptionPacket


@dataclass(frozen=True)
class _OperationDefinition:
    operation: ActorOperation
    action_type: ActionType
    target_kinds: tuple[str, ...]
    description: str
    required_parameters: tuple[str, ...] = ()
    self_only: bool = False


_OPERATIONS = (
    _OperationDefinition(ActorOperation.MOVE, ActionType.MOVE, ("location",), "我前往{target}"),
    _OperationDefinition(ActorOperation.BROADCAST, ActionType.SEND_MESSAGE, ("location",),
                         "我向这里的人广播", ("message",)),
    _OperationDefinition(ActorOperation.TALK, ActionType.TALK, ("agent",), "我和{target}交谈"),
    _OperationDefinition(ActorOperation.ERRAND, ActionType.ERRAND, ("npc",), "我向{target}指派任务", ("description",)),
    _OperationDefinition(ActorOperation.TAKE, ActionType.PHYSICAL, ("item",), "我拿起{target}"),
    _OperationDefinition(ActorOperation.USE, ActionType.PHYSICAL, ("item", "landmark"), "我使用{target}"),
    _OperationDefinition(ActorOperation.DESTROY, ActionType.PHYSICAL, ("item", "landmark"), "我毁坏{target}"),
    _OperationDefinition(ActorOperation.PHYSICAL, ActionType.PHYSICAL, ("agent", "item", "landmark"),
                         "我处理{target}", ("description",)),
    _OperationDefinition(ActorOperation.WORK, ActionType.WORK, ("agent",), "我处理手头的事务", ("description",), self_only=True),
    _OperationDefinition(ActorOperation.REST, ActionType.REST, ("agent",), "我休息片刻", self_only=True),
    _OperationDefinition(ActorOperation.COVERT, ActionType.COVERT, ("agent",), "我暗中探查", ("description",), self_only=True),
    _OperationDefinition(ActorOperation.SEND_MESSAGE, ActionType.SEND_MESSAGE, ("agent",), "我发送消息",
                         ("description",), self_only=True),
)


def requires_actor_interpretation(intent: ActorIntent) -> bool:
    return intent.text is not None or bool(intent.description.strip())


def _target_name(target: Ref, packet: "PerceptionPacket") -> str:
    spatial = packet.spatial
    if target == Ref.agent(packet.agent_id):
        return "我"
    if target.kind == "location":
        if target.id == spatial.location_id:
            return (spatial.location_view.name if spatial.location_view else "") or "此处"
        for place in spatial.reachable_locations:
            if place.location_id == target.id:
                return place.view.name or "某处"
    elif target.kind == "agent":
        person = spatial.visible_agents.get(target.id)
        if person is not None:
            return person_referent(person.identity.name or "某人", person.identity.gender)
    elif target.kind == "npc":
        npc = spatial.visible_npcs.get(target.id)
        if npc is not None:
            return person_referent(npc.identity.name or "某人", npc.identity.gender)
    else:
        for entity in spatial.visible_entities:
            if entity.entity_id == target.id and entity.entity_type == target.kind:
                return entity.name or "某物"
    raise ActorAnswerRejected("target_not_perceived")


def actor_decision_beat(decision: ActorDecision, packet: "PerceptionPacket") -> dict[str, Any]:
    """The decision as a trace beat records it: the act, with each target slot holding names
    (as the character knows them) where an AI decision's holds indices. Code-layer only."""
    from agent.decision import ACTOR_TARGET_SLOTS, candidate_lists

    action = decision.action
    if action is None:
        return {"idle": True}
    spatial = packet.spatial
    names: dict[str, dict[str, str]] = {
        "person": {aid: who.name for aid, who in candidate_lists(packet).roster},
        "place": {rl.location_id: rl.view.name or "某处" for rl in spatial.reachable_locations},
        "npc": {nid: npc.identity.name or "某人" for nid, npc in spatial.visible_npcs.items()},
        "entity": {e.entity_id: e.name or "某物" for e in spatial.visible_entities},
    }
    fallback = {"person": "某人", "place": "某处", "npc": "某人", "entity": "某物"}
    targets: dict[str, list[str]] = {}
    for field_name, slot, kind in ACTOR_TARGET_SLOTS:
        value = getattr(action, field_name)
        ids = list(value) if isinstance(value, tuple) else ([value] if value else [])
        if ids:
            targets[slot] = [names[kind].get(i, fallback[kind]) for i in ids]
    beat: dict[str, Any] = {
        "action_type": action.action_type.value.upper(),
        "action_description": action.action_description,
        "expected_outcome": action.expected_outcome,
        "targets": targets,
    }
    if words := action.message_content or action.errand_message:
        beat["message_content"] = words
    if action.action_type is ActionType.SEND_MESSAGE:
        beat["anonymous"] = action.anonymous
    if action.item_name:
        beat["item_name"] = action.item_name
    return beat


def actor_operations(turn_id: str, target: Ref, packet: "PerceptionPacket") -> ActorOperations:
    try:
        _target_name(target, packet)
        perceived = True
    except ActorAnswerRejected:
        perceived = False
    capabilities: list[ActorOperationCapability] = []
    self_target = target == Ref.agent(packet.agent_id)
    for definition in _OPERATIONS:
        if definition.self_only != self_target or target.kind not in definition.target_kinds:
            continue
        operation = definition.operation
        reason = None if perceived else "target_not_perceived"
        required = definition.required_parameters
        if operation is ActorOperation.MOVE and target.id == packet.spatial.location_id:
            reason = "already_here"
        elif operation is ActorOperation.BROADCAST:
            if target.id != packet.spatial.location_id:
                reason = reason or "broadcast_requires_current_location"
        elif operation is ActorOperation.ERRAND:
            npc = packet.spatial.visible_npcs.get(target.id)
            if npc is not None and npc.busy:
                reason = "npc_busy"
        elif operation is ActorOperation.TAKE and perceived:
            entity = next(e for e in packet.spatial.visible_entities if e.entity_id == target.id)
            if not entity.is_takeable:
                reason = "item_not_takeable"
            elif entity.holder_id == packet.agent_id:
                reason = "already_held"
        capabilities.append(ActorOperationCapability(operation, reason is None, reason, required))
    return ActorOperations(turn_id, target, tuple(capabilities))


def build_actor_operation(intent: ActorIntent, packet: "PerceptionPacket") -> ActorAction:
    assert intent.operation is not None and intent.target is not None
    operation, target, params = intent.operation, intent.target, intent.parameters
    capability = next((c for c in actor_operations(intent.turn_id, target, packet).operations
                       if c.operation is operation), None)
    if capability is None:
        raise ActorAnswerRejected("operation_not_applicable")
    if not capability.available:
        raise ActorAnswerRejected(capability.reason or "operation_unavailable")
    for required in capability.required_parameters:
        value = intent.description if required == "description" else getattr(params, required)
        if value is None or not str(value).strip():
            raise ActorAnswerRejected(f"{required}_required")
    definition = next(d for d in _OPERATIONS if d.operation is operation)
    name = _target_name(target, packet)
    description = definition.description.format(target=name)
    if operation is ActorOperation.SEND_MESSAGE:
        return ActorAction(definition.action_type, description, anonymous=params.anonymous is True)
    if definition.self_only:
        return ActorAction(definition.action_type, description)
    if operation is ActorOperation.MOVE:
        return ActorAction(definition.action_type, description, destination_id=target.id)
    if operation is ActorOperation.BROADCAST:
        return ActorAction(definition.action_type, description, message_announce=True,
                           message_content=(params.message or "").strip(), anonymous=params.anonymous is True)
    if operation is ActorOperation.TALK:
        return ActorAction(definition.action_type, description, person_ids=(target.id,))
    if operation is ActorOperation.ERRAND:
        if params.recipient_id is not None and params.announce is True:
            raise ActorAnswerRejected("errand_address_conflict")
        if params.destination_id is not None:
            destination = _target_name(Ref.place(params.destination_id), packet)
            description = f"我向{name}指派任务：前往{destination}"
        return ActorAction(
            definition.action_type, description, errand_npc_id=target.id,
            errand_destination_id=params.destination_id, errand_item_id=params.item_id,
            errand_recipient_id=params.recipient_id, errand_message=(params.message or "").strip(),
            errand_announce=params.announce is True,
        )
    targets = {
        "agent": {"physical_person_id": target.id},
        "item": {"physical_entity_id": target.id},
        "landmark": {"physical_entity_id": target.id},
    }[target.kind]
    return ActorAction(definition.action_type, description, **targets)


def check_actor_operation(action: ActorAction, base: ActorAction, intent: ActorIntent) -> None:
    """Protect fixed operation parameters after unrelated model fields are removed."""
    mutable = {"action_description", "estimated_steps", "expected_outcome", "urgency"}
    if intent.operation is ActorOperation.MOVE:
        mutable.add("move_carry_ids")
    elif intent.operation is ActorOperation.TALK:
        # No operation parameter states a topic: the description is the only place one can come from.
        mutable.update(("person_ids", "message_content", "talk_topic"))
        if not action.person_ids or action.person_ids[0] != base.person_ids[0]:
            raise ActorAnswerRejected("operation_constraint_conflict")
    elif intent.operation is ActorOperation.SEND_MESSAGE:
        # Recipients and words come from the description; so does the signature unless it was set.
        mutable.update(("person_ids", "message_announce", "message_content"))
        if intent.parameters.anonymous is None:
            mutable.add("anonymous")
    elif base.action_type is ActionType.PHYSICAL:
        mutable.add("physical_recipient_id")
    elif intent.operation is ActorOperation.ERRAND:
        for parameter, slot in {
            "destination_id": "errand_destination_id",
            "item_id": "errand_item_id", "recipient_id": "errand_recipient_id",
            "message": "errand_message", "announce": "errand_announce",
        }.items():
            if getattr(intent.parameters, parameter) is None:
                mutable.add(slot)
    if action.action_type is ActionType.ERRAND:
        if action.errand_recipient_id and action.errand_announce:
            raise ActorAnswerRejected("errand_address_conflict")
        if action.errand_message and not action.errand_recipient_id and not action.errand_announce:
            raise ActorAnswerRejected("message_requires_recipient_or_announce")
    for field in fields(ActorAction):
        if field.name not in mutable and getattr(action, field.name) != getattr(base, field.name):
            raise ActorAnswerRejected("operation_constraint_conflict")


def build_actor_decision_summary(
    decision: ActorDecision, packet: "PerceptionPacket", *, seconds_per_step: int,
    intent: ActorIntent | None = None,
) -> ActorDecisionSummary:
    action = decision.action
    if action is None:
        return ActorDecisionSummary(uuid.uuid4().hex, "", IDLE_OPERATION, None, 0)
    target: Ref | None = None
    params: dict[str, str | bool | list[str]] = {}
    if action.destination_id:
        target = Ref.place(action.destination_id)
        if action.move_carry_ids:
            params["move_carry_ids"] = list(action.move_carry_ids)
    elif action.errand_npc_id:
        target = Ref.npc(action.errand_npc_id)
        params["destination_id"] = action.errand_destination_id or ""
        for name, value in (("item_id", action.errand_item_id), ("recipient_id", action.errand_recipient_id),
                            ("message", action.errand_message)):
            if value:
                params[name] = value
        params["announce"] = action.errand_announce
    elif action.physical_person_id:
        target = Ref.agent(action.physical_person_id)
    elif action.physical_npc_id:
        target = Ref.npc(action.physical_npc_id)
    elif action.physical_entity_id:
        entity = next(e for e in packet.spatial.visible_entities if e.entity_id == action.physical_entity_id)
        target = Ref.entity(entity.entity_id, entity.entity_type)
    elif action.person_ids:
        target = Ref.agent(action.person_ids[0])
        if len(action.person_ids) > 1:
            params["person_ids"] = list(action.person_ids)
    if action.item_name:
        params["item_name"] = action.item_name
    if action.physical_recipient_id:
        params["recipient_id"] = action.physical_recipient_id
    if action.talk_topic:
        params["talk_topic"] = action.talk_topic
    if action.expected_outcome:
        params["expected_outcome"] = action.expected_outcome
    if action.action_type is ActionType.SEND_MESSAGE:
        params["message"] = action.message_content
        params["anonymous"] = action.anonymous
        params["announce"] = action.message_announce
        params["urgency"] = action.urgency.value
        if action.message_announce and not action.person_ids:
            target = Ref.place(packet.spatial.location_id)
    if intent is not None and intent.target == Ref.agent(packet.agent_id):
        target = intent.target
        # Recipients are no longer the target then: every one of them is listed.
        if action.person_ids:
            params["person_ids"] = list(action.person_ids)
    operation = intent.operation.value if intent is not None and intent.operation else action.action_type.value
    if action.action_type is ActionType.MOVE:
        destination = next(place for place in packet.spatial.reachable_locations
                           if place.location_id == action.destination_id)
        steps = travel_duration_steps(destination.travel_seconds, seconds_per_step)
    else:
        steps = action_duration_steps(action.action_type, action.estimated_steps, live_talk=True)
    return ActorDecisionSummary(
        uuid.uuid4().hex, action.action_description, operation, target,
        steps * seconds_per_step, params,
    )
