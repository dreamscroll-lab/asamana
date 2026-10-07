"""Runtime message delivery: a cross-location / cross-time transport pipe.

MessageSystem design rules:

Scope
    Carries only cross-location / cross-time information. Same-place, same-time information
    flows through EnvironmentSystem.record_carry_observation → SpatialPerception.ambient_events
    → PerceptionMemoryLayer, not through this pipe.

Lossless transport
    No rendering, distortion or subjective interpretation. Message.content leaves the pipe
    byte-identical to how it entered. Distortion arises naturally in the memory / agent
    layers; subjective interpretation is written into EXPERIENTIAL by the receiving agent.

Data-driven, no type enum
    There is no PropagationType. Delivery is decided by the combination of three fields
    (recipients × location_scope × deliver_step). New scenarios come from field combinations,
    not new types.

No side effects
    No relation updates, no memory writes, no interrupts, no LLM calls: the receiving
    PerceptionLayer and the agent's decision own those.

Actions and messages are decoupled
    Only cross-location actions (currently just SEND_MESSAGE) enter via dispatch_from_action.
    Same-place actions (TALK / COVERT / WORK / etc.) propagate naturally via
    ActionResult.outcome → _carry_step_observations and never enter the message queue.

Rumors spread through agents
    There is no gossip propagation mechanism: a rumor spreads only through agents' own social
    decisions.

The recipient decides
    urgency is a sender → recipient meta-signal, not a command. The recipient may ignore it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Sequence

from core.interfaces.message import Message, MessageProvider
from core.interfaces.urgency import Urgency

if TYPE_CHECKING:
    from core.interfaces.action import AgentAction
    from engine.environment import EnvironmentSystem


@dataclass(frozen=True)
class MessageDelivery:
    """Messages delivered for a single runtime step.

    Lossless: the Message objects in inbox are the same references as in
    delivered_messages; content is never rewritten.
    """

    step: int
    delivered_messages: List[Message] = field(default_factory=list)
    inboxes: Dict[str, List[Message]] = field(default_factory=dict)
    undelivered_messages: List[Message] = field(default_factory=list)

    def inbox_for(self, agent_id: str) -> List[Message]:
        return list(self.inboxes.get(agent_id, []))

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "delivered": [
                {
                    "id": message.id,
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name,
                    "recipients": list(message.recipients) if message.recipients is not None else None,
                    "location_scope": message.location_scope,
                    "content": message.content,
                    "intent": message.intent,
                    "urgency": message.urgency.value,
                    "metadata": dict(message.metadata),
                }
                for message in self.delivered_messages
            ],
            "inboxes": {
                agent_id: [message.id for message in messages]
                for agent_id, messages in sorted(self.inboxes.items())
            },
            "undelivered": [message.id for message in self.undelivered_messages],
        }


class MessageSystem:
    """Transport pipe for cross-location / cross-time information; see the design rules in the
    module docstring."""

    def __init__(
        self,
        provider: MessageProvider,
        *,
        world_id: str,
    ) -> None:
        self._provider = provider
        self._world_id = world_id

    @property
    def world_id(self) -> str:
        return self._world_id

    async def publish(self, message: Message) -> None:
        """Messages with deliver_step <= current_step come out on the next collect."""
        await self._provider.enqueue(message)

    async def collect(self, *, step: int) -> List[Message]:
        """Pop every message with deliver_step <= step; the provider queue enforces the delay."""
        return await self._provider.dequeue_ready(self._world_id, step)

    async def peek_pending(self) -> List[Message]:
        """For snapshots/observers: read the current pending messages without dequeuing."""
        return await self._provider.peek_pending(self._world_id)

    async def deliver_for_agents(
        self,
        *,
        step: int,
        agent_ids: Sequence[str],
        environment: "EnvironmentSystem | None" = None,
    ) -> MessageDelivery:
        """Collect provider messages once and route the same Message references into per-agent
        inboxes."""
        inboxes: Dict[str, List[Message]] = {agent_id: [] for agent_id in agent_ids}
        undelivered: List[Message] = []
        delivered_messages = await self.collect(step=step)

        for message in delivered_messages:
            receivers = self._resolve_receivers(message, agent_ids, environment)
            delivered_any = False
            for receiver_id in receivers:
                if receiver_id not in inboxes:
                    continue
                inboxes[receiver_id].append(message)
                delivered_any = True
            if not delivered_any:
                undelivered.append(message)

        return MessageDelivery(
            step=step,
            delivered_messages=delivered_messages,
            inboxes=inboxes,
            undelivered_messages=undelivered,
        )

    async def dispatch_from_action(
        self,
        action: "AgentAction",
        *,
        current_step: int,
        sender_name: str = "",
        location_scope: str | None = None,
    ) -> Message:
        """Translate a cross-location action into a Message and enqueue it.

        Only cross-location actions (currently just SEND_MESSAGE) may use this path; same-place
        actions must not enter the queue (see the module docstring).
        """
        agent_ids = action.target.acted_on_agents
        recipients: list[str] | None = agent_ids or None  # None = broadcast
        urgency = getattr(action, "urgency", Urgency.NORMAL)
        message = Message(
            id=str(uuid.uuid4()),
            world_id=self._world_id,
            sender_id=action.agent_id,
            sender_name=sender_name,
            # Deliver the exact words addressed to the recipient; action_description is the
            # sender's own narration of the act, used only when content is missing.
            content=action.content or action.action_description,
            intent=(action.inner_monologue or action.reason or ""),
            recipients=recipients,
            location_scope=location_scope,
            deliver_step=current_step,
            created_step=current_step,
            urgency=urgency,
            metadata={"source": "action", "action_type": str(action.action_type.value)},
        )
        await self.publish(message)
        return message

    def _resolve_receivers(
        self,
        message: Message,
        agent_ids: Sequence[str],
        environment: "EnvironmentSystem | None",
    ) -> list[str]:
        """Resolve actual recipients from (recipients × location_scope).

        The four-way matrix:

            recipients   | location_scope | result
            ─────────────┼────────────────┼────────────────────────────────────────
            list[str]    | None           | the agents in the list
            None         | str (L)        | every agent at location L
            None         | None           | all agent_ids = world-wide broadcast
            list[str]    | str (L)        | list ∩ (agents at location L)

        All four cells then subtract ``{sender_id} ∪ actor_ids``: nobody receives, over an
        external channel, what they did themselves. The exclusion is applied once outside the
        matrix, since it is orthogonal to addressing and four copies drift apart.
        """
        excluded = {message.sender_id, *message.actor_ids}
        if message.recipients is not None:
            candidates = [aid for aid in message.recipients if aid not in excluded]
            if message.location_scope is None or environment is None:
                return candidates
            location_agents = set(environment.agents_at(message.location_scope))
            return [aid for aid in candidates if aid in location_agents]

        if message.location_scope is not None and environment is not None:
            return [
                aid for aid in environment.agents_at(message.location_scope)
                if aid not in excluded
            ]
        return [aid for aid in agent_ids if aid not in excluded]
