from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from core.interfaces.action import ActionResult, ActionTarget, ActionType, AgentAction, EntityStateChange, ErrandOrder, Ref
from core.interfaces.entity import EntityPresence, WorldEntity, WorldEntityType
from engine.executors.base import ActionExecutionState, ActionExecutor
from tests.unit.test_runtime_smoke import _build_runtime


@pytest.mark.asyncio
async def test_concurrent_step_is_rejected_without_releasing_first_step(container) -> None:
    runtime, _, _ = _build_runtime(container, world_id="step-reentry")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_step(agents: list) -> object:
        entered.set()
        await release.wait()
        return object()

    with patch.object(runtime, "_run_step", side_effect=blocked_step) as run_step:
        first = asyncio.create_task(runtime.run_step([]))
        await entered.wait()
        try:
            with pytest.raises(ValueError, match="step in progress"):
                await runtime.run_step([])
            assert runtime.step_in_progress
            assert run_step.call_count == 1
        finally:
            release.set()
            await first
        assert not runtime.step_in_progress
        await runtime.run_step([])
        assert run_step.call_count == 2


@pytest.mark.asyncio
async def test_failed_step_releases_reentry_guard(container) -> None:
    runtime, _, _ = _build_runtime(container, world_id="step-error")
    with patch.object(runtime, "_run_step", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            await runtime.run_step([])
    assert not runtime.step_in_progress


class _ItemExecutor(ActionExecutor):
    def __init__(self, reads: list[tuple[str, str | None]], *, destroy: bool = False) -> None:
        self.reads = reads
        self.destroy = destroy

    async def start(self, *args: object, **kwargs: object) -> ActionExecutionState:
        raise NotImplementedError

    async def tick(self, *args: object, **kwargs: object) -> list:
        return []

    async def complete(self, state: ActionExecutionState, step: int, **kwargs: object) -> list[ActionResult]:
        environment = kwargs["environment"]
        item = environment.get_entity("letter")
        self.reads.append((state.initiator_id, item.owner_id))
        await asyncio.sleep(0)
        changes = []
        if state.initiator_id == "first":
            changes = [EntityStateChange(
                entity_id="letter", destroyed=self.destroy,
                owner_id=None if self.destroy else "first",
            )]
        elif not item.is_destroyed and state.action_type is ActionType.PHYSICAL:
            changes = [EntityStateChange(entity_id="letter", owner_id="second")]
        return [ActionResult(
            action=AgentAction(action_type=state.action_type, agent_id=state.initiator_id, step=step),
            expected_outcome="", outcome="done", entity_state_changes=changes,
        )]


@pytest.mark.asyncio
@pytest.mark.parametrize("second_type", [ActionType.PHYSICAL, ActionType.WORK, ActionType.ERRAND])
async def test_shared_item_adjudication_reads_previous_landing(container, second_type: ActionType) -> None:
    runtime, environment, _ = _build_runtime(container, world_id="shared-item")
    item = WorldEntity(
        entity_id="letter", name="Letter", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref="second",
    )
    environment.register_entity(item)
    reads: list[tuple[str, str | None]] = []
    executor = _ItemExecutor(reads)
    for kind in {ActionType.PHYSICAL, second_type}:
        runtime._executor_registry.register(kind, executor)
    states = [ActionExecutionState.create(
        action_type=kind, initiator_id=doer, participant_ids=[doer], purpose="handle letter",
        started_step=1, estimated_steps=1, opening_outcome="",
        target=ActionTarget(acts_on=[Ref.entity("letter")]) if kind is ActionType.PHYSICAL else ActionTarget(),
    ) for doer, kind in [("first", ActionType.PHYSICAL), ("second", second_type)]]
    if second_type is ActionType.ERRAND:
        states[1].extra["errand_order"] = ErrandOrder(npc_id="bearer", destination_id="hall", item_id="letter")

    await runtime._processor.finalize_executions_batch(states, {}, step=1)

    assert reads == [("first", "second"), ("second", "first")]


@pytest.mark.asyncio
async def test_destroyed_item_cannot_be_revived_by_later_seize(container) -> None:
    runtime, environment, _ = _build_runtime(container, world_id="destroy-seize")
    item = WorldEntity(entity_id="letter", name="Letter", entity_type=WorldEntityType.ITEM, presence_ref="hall")
    environment.register_entity(item)
    reads: list[tuple[str, str | None]] = []
    runtime._executor_registry.register(ActionType.PHYSICAL, _ItemExecutor(reads, destroy=True))
    states = [ActionExecutionState.create(
        action_type=ActionType.PHYSICAL, initiator_id=doer, participant_ids=[doer], purpose="handle letter",
        started_step=1, estimated_steps=1, opening_outcome="", target=ActionTarget(acts_on=[Ref.entity("letter")]),
    ) for doer in ["first", "second"]]

    results = await runtime._processor.finalize_executions_batch(states, {}, step=1)

    assert results[states[1].execution_id][0].entity_state_changes == []
    assert not environment.change_entity_state(EntityStateChange(entity_id="letter", owner_id="second"))
    assert item.is_destroyed
    assert environment.get_items_of("second") == []


@pytest.mark.asyncio
async def test_character_feedback_lands_after_every_wave_is_judged(container) -> None:
    """Waves order only the items: a later judge must not see an earlier wave's feedback."""
    runtime, environment, _ = _build_runtime(container, world_id="feedback-after-waves")
    environment.register_entity(WorldEntity(
        entity_id="letter", name="Letter", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref="second",
    ))
    reads: list[tuple[str, str | None]] = []
    runtime._executor_registry.register(ActionType.PHYSICAL, _ItemExecutor(reads))
    landed: list[str] = []

    async def record_feedback(result: ActionResult, agents_dict: dict, step: int) -> None:
        landed.append(result.action.agent_id)
        assert len(reads) == 2, "feedback landed before the second wave was judged"

    runtime._processor.land_agent_feedback = record_feedback  # type: ignore[method-assign]
    states = [ActionExecutionState.create(
        action_type=ActionType.PHYSICAL, initiator_id=doer, participant_ids=[doer], purpose="handle letter",
        started_step=1, estimated_steps=1, opening_outcome="", target=ActionTarget(acts_on=[Ref.entity("letter")]),
    ) for doer in ["first", "second"]]

    await runtime._processor.finalize_executions_batch(states, {}, step=1)

    assert reads == [("first", "second"), ("second", "first")]
    assert sorted(landed) == ["first", "second"]


class _BarrierExecutor(ActionExecutor):
    def __init__(self) -> None:
        self.inflight = 0
        self.peak = 0
        self.both = asyncio.Event()

    async def start(self, *args: object, **kwargs: object) -> ActionExecutionState:
        raise NotImplementedError

    async def tick(self, *args: object, **kwargs: object) -> list:
        return []

    async def complete(self, state: ActionExecutionState, step: int, **kwargs: object) -> list[ActionResult]:
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        if self.inflight == 2:
            self.both.set()
        try:
            await asyncio.wait_for(self.both.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            pass
        self.inflight -= 1
        return []


@pytest.mark.asyncio
async def test_acts_on_the_same_person_are_judged_concurrently(container) -> None:
    """Only items order the waves; two acts on one person never contend for an item."""
    runtime, _, _ = _build_runtime(container, world_id="same-person")
    executor = _BarrierExecutor()
    runtime._executor_registry.register(ActionType.PHYSICAL, executor)
    states = [ActionExecutionState.create(
        action_type=ActionType.PHYSICAL, initiator_id=doer, participant_ids=[doer], purpose="shove",
        started_step=1, estimated_steps=1, opening_outcome="", target=ActionTarget(acts_on=[Ref.agent("victim")]),
    ) for doer in ["first", "second"]]

    await runtime._processor.finalize_executions_batch(states, {}, step=1)

    assert executor.peak == 2
