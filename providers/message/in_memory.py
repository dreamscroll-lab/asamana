"""In-memory message provider."""

from __future__ import annotations

from typing import Dict, List

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.message import Message, MessageProvider


@ProviderFactory.register("in_memory", kind=ComponentKind.MESSAGE)
class InMemoryMessageProvider(MessageProvider):
    """Queue messages in-process, one queue per world."""

    def __init__(self) -> None:
        self._queues: Dict[str, List[Message]] = {}

    async def enqueue(self, message: Message) -> None:
        self._queues.setdefault(message.world_id, []).append(message)

    async def dequeue_ready(self, world_id: str, current_step: int) -> List[Message]:
        queue = self._queues.get(world_id, [])
        available = [message for message in queue if message.deliver_step <= current_step]
        if available:
            self._queues[world_id] = [
                message for message in queue if message.deliver_step > current_step
            ]
        return available

    async def peek_pending(self, world_id: str) -> List[Message]:
        return list(self._queues.get(world_id, []))

    async def clear(self, world_id: str | None = None) -> None:
        if world_id is None:
            self._queues.clear()
            return
        self._queues.pop(world_id, None)
