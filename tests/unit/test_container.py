from __future__ import annotations

import pytest

from core.container import Container
from core.interfaces.llm import LLMMessage, LLMScene
from core.interfaces.message import Message
from tests.unit.bus_tap import tap


def test_container_builds_from_test_config(container) -> None:
    description = container.describe()

    # No world_config in the container: each world picks its own map; it's not a deployment-level
    # dependency.
    assert not hasattr(container, "world_config")
    assert type(container.embedding).__name__ == "InMemoryEmbeddingProvider"
    assert type(container.vector_store).__name__ == "InMemoryVectorStore"
    assert type(container.agent_store).__name__ == "InMemoryAgentStore"
    assert type(container.message_provider).__name__ == "InMemoryMessageProvider"
    assert type(container.snapshot).__name__ == "InMemorySnapshotProvider"
    assert description["message_provider"] == "InMemoryMessageProvider"


@pytest.mark.parametrize("scene", list(LLMScene))
@pytest.mark.asyncio
async def test_container_llm_router_routes_all_scenes_to_mock_provider(container, scene: LLMScene) -> None:
    prompt = f"hello asamana from {scene.value}"
    response = await container.llm_router.complete(
        scene,
        [LLMMessage(role="user", content=prompt)],
    )

    assert response.model == "mock"
    # Each scene's reply must come from its own mock provider (that's what "routed correctly"
    # means). Don't assert the literal "mock response": conftest swaps the adjudication scene's
    # (AGENT_ACTION_NARRATION) provider for a mock returning valid ruling JSON — all four executors
    # adjudicate via LLM, and a non-JSON reply turns every action into an adjudication_failed null
    # step.
    assert response.content == container.llm_router.get(scene).fixed_response


@pytest.mark.asyncio
async def test_containers_do_not_share_provider_state(test_config) -> None:
    first = Container.from_config(test_config)
    second = Container.from_config(test_config)

    await first.message_provider.enqueue(
        Message(
            id="msg-1",
            world_id="world-1",
            sender_id="agent-1",
            content="Only first container should see this",
            recipients=["agent-2"],
            location_scope=None,
            created_step=1,
            deliver_step=1,
        )
    )
    first_published = tap(first.event_bus)
    second_published = tap(second.event_bus)
    first.event_bus.publish({"type": "step", "step": 1})

    assert await first.message_provider.dequeue_ready("world-1", 1) != []
    assert await second.message_provider.dequeue_ready("world-1", 1) == []
    assert first_published() == [{"type": "step", "step": 1}]
    assert second_published() == []
