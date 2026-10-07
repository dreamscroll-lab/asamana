"""Per-step execution lifecycle: ticking ongoing executions, adjudicating completions, and
landing results (the sole agent-state writeback path, per the Executor/Feedback boundary).

The shared completion layer for the runtime's main sweep, death teardown and the interrupt
path. Holds no per-step runtime state and never calls back into the runtime.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List

from agent.personality import ActionStatus
from core.context import annotate_call, observe_stage
from core.interfaces.action import ActionResult
from core.interfaces.directory import WorldDirectory
from core.interfaces.trace import Stage
from core.logging import get_logger
from engine.environment import EnvironmentSystem
from engine.executors.base import (
    ActionExecutionState, action_semantics, execution_annotations, participant_action_desc,
    target_semantics,
)
from engine.executors.registry import ActionExecutorRegistry
from engine.message_system import MessageSystem

if TYPE_CHECKING:
    from agent.agent import Agent


logger = get_logger(__name__)


class ExecutionProcessor:
    """Advances and lands this step's executions."""

    def __init__(
        self,
        *,
        executor_registry: ActionExecutorRegistry,
        environment: EnvironmentSystem,
        message_system: MessageSystem,
        directory: WorldDirectory,
    ) -> None:
        self._executor_registry = executor_registry
        self._environment = environment
        self._message_system = message_system
        self._directory = directory
        # Records of forced teardowns, held for the step's display stream: they happen deep in
        # death handling and director moves, which have no channel back to the runtime.
        self._forced_records: List[Dict[str, Any]] = []

    def take_forced_records(self) -> List[Dict[str, Any]]:
        """The step's forced-teardown records so far, handed over once."""
        records, self._forced_records = self._forced_records, []
        return records

    async def tick_ongoing_executions(
        self,
        agents_dict: Dict[str, "Agent"],
        step: int,
    ) -> List[Dict[str, Any]]:
        """Advance all active multi-step executions by one tick before new decisions."""
        tick_records: List[Dict[str, Any]] = []
        completing: List[ActionExecutionState] = []
        completing_pos: List[int] = []

        for exec_state in self._executor_registry.all_active():
            executor = self._executor_registry.get_executor(exec_state.action_type)
            if executor is None:
                continue

            exec_state.remaining_steps -= 1

            for pid in exec_state.participant_ids:
                agent = agents_dict.get(pid)
                if agent is not None:
                    remaining = max(0, agent.personality.state.action_remaining_steps - 1)
                    agent.personality.update_action_status(
                        status=agent.personality.state.action_status,
                        current_action=agent.personality.state.current_action,
                        remaining_steps=remaining,
                    )

            # On the final step emit only the completion, never a tick too: one record per agent
            # per step. Completions are finished as a batch after the loop and spliced back at
            # completing_pos to keep order.
            if exec_state.remaining_steps <= 0:
                completing.append(exec_state)
                completing_pos.append(len(tick_records))
                continue

            # Compare positions, not action types, so any executor that moves people is covered.
            was_at = {
                pid: self._environment.get_body_location(pid)
                for pid in exec_state.participant_ids
            }
            try:
                with observe_stage(Stage.ACTION, agent_id=exec_state.initiator_id), \
                        annotate_call(**execution_annotations(self._directory, exec_state)):
                    tick_results = await executor.tick(
                        exec_state,
                        step,
                        agents=agents_dict,
                        environment=self._environment,
                        message_system=self._message_system,
                    )
            except Exception as exc:  # noqa: BLE001 — Rule 5: an executor exception must not kill the run
                # A lost tick costs nothing; the remaining-step count has already advanced.
                logger.warning(
                    "executor_tick_failed",
                    extra={
                        "step": step,
                        "action_type": exec_state.action_type.value,
                        "initiator_id": exec_state.initiator_id,
                        "error": str(exc),
                    },
                )
                continue

            # Perception ran before tick; without this, later cognition this step would place
            # him at the previous crossroads.
            for pid, before in was_at.items():
                agent = agents_dict.get(pid)
                if agent is None or self._environment.get_body_location(pid) == before:
                    continue
                agent.refresh_situation(self._environment.spatial_for(agent_id=pid, step=step))

            for tick_result in tick_results:
                # A full observer record, or the middle beat is unrenderable. Never written to
                # memory: progress is not an experience.
                tick_records.append({
                    "agent_id": tick_result.agent_id,
                    "agent_name": self._directory.agent_name(tick_result.agent_id),
                    "is_main_character": (
                        agents_dict[tick_result.agent_id].is_main_character
                        if tick_result.agent_id in agents_dict else False
                    ),
                    "action_type": exec_state.action_type.value,  # JSON-native; see runtime.py
                    # Same deed as the act. Taking the type verbatim is safe: PHYSICAL, whose
                    # deed differs from its type, never ticks.
                    "deed": str(getattr(exec_state.action_type, "value", exec_state.action_type)),
                    # The renderer re-aims every tick; an empty target silently turns a figure
                    # mid-conversation to face nobody.
                    **target_semantics(exec_state.target),
                    "location_id": self._environment.get_body_location(tick_result.agent_id),
                    "phase": "ongoing_tick",
                    "outcome": tick_result.outcome,
                    "gist": tick_result.outcome,
                    # Never fall back to outcome: COVERT ticks leave observations empty so their
                    # outcome stays god-view only. Missing a perception beats leaking one.
                    "observations": [o.__dict__ for o in tick_result.observations],
                    "participant_ids": list(exec_state.participant_ids),  # every member self-filters
                    # The client folds per-participant ticks into one beat by these.
                    "execution_id": exec_state.execution_id,
                    "initiator_id": exec_state.initiator_id,
                    # A conscripted participant's own framing, not the initiator's.
                    "action_description": participant_action_desc(
                        self._directory, tick_result.agent_id, exec_state,
                    ),
                    # Progress for a renderer. Step counts are fine in structured fields; never
                    # 「第N步」 in narrative text.
                    "elapsed_steps": exec_state.estimated_steps - exec_state.remaining_steps,
                    "total_steps": exec_state.estimated_steps,
                })

        if completing:
            results_by_exec = await self.finalize_executions_batch(completing, agents_dict, step)
            # Reverse order keeps earlier insertion points valid.
            for exec_state, pos in reversed(list(zip(completing, completing_pos))):
                completion_recs = [
                    self.completion_record(exec_state, result, agents_dict, "ongoing_complete")
                    for result in results_by_exec.get(exec_state.execution_id, [])
                ]
                tick_records[pos:pos] = completion_recs
                self._executor_registry.remove_active(exec_state.execution_id)

        return tick_records

    async def _adjudicate_execution(
        self,
        exec_state: ActionExecutionState,
        agents_dict: Dict[str, "Agent"],
        step: int,
    ) -> List[ActionResult]:
        """Adjudicate one execution's completion without landing it. complete() only reads
        shared state, so a batch can be adjudicated concurrently. Executor failure is a null
        step: no results, participants reset to idle (Rule 5).
        """
        executor = self._executor_registry.get_executor(exec_state.action_type)
        if executor is None:
            # Null step. Release participants, or they stay IN_PROGRESS forever once the caller
            # removes the execution.
            logger.warning(
                "executor_missing",
                extra={"step": step, "action_type": exec_state.action_type.value,
                       "initiator_id": exec_state.initiator_id},
            )
            self.release_stuck_participants(exec_state, agents_dict)
            return []
        try:
            with observe_stage(Stage.ACTION, agent_id=exec_state.initiator_id), \
                    annotate_call(**execution_annotations(self._directory, exec_state)):
                final_results = await executor.complete(
                    exec_state,
                    step,
                    agents=agents_dict,
                    environment=self._environment,
                    message_system=self._message_system,
                )
        except Exception as exc:  # noqa: BLE001 — Rule 5: an executor exception must not kill the run
            # Null step; release participants as above.
            logger.warning(
                "executor_complete_failed",
                extra={
                    "step": step,
                    "action_type": exec_state.action_type.value,
                    "initiator_id": exec_state.initiator_id,
                    "error": str(exc),
                },
            )
            self.release_stuck_participants(exec_state, agents_dict)
            return []
        logger.info(
            "action_completed",
            extra={
                "step": step,
                "action_type": exec_state.action_type.value,
                "initiator_id": exec_state.initiator_id,
                "participant_count": len(exec_state.participant_ids),
                "result_count": len(final_results),
                "succeeded": [r.succeeded for r in final_results],
                "adjudication_failed": any(r.adjudication_failed for r in final_results),
            },
        )
        return final_results

    async def finalize_executions_batch(
        self,
        exec_states: List[ActionExecutionState],
        agents_dict: Dict[str, "Agent"],
        step: int,
    ) -> Dict[str, List[ActionResult]]:
        """Complete a batch of executions in one beat; returns {execution_id: results}.

        Read-only executors adjudicate concurrently against start-of-step state; executors that
        mutate the world in complete() (MOVE) run serially afterwards, so judges see pre-move
        positions. Then results land grouped by conflict key (_land_results_grouped).
        """
        readonly: List[ActionExecutionState] = []
        world_infra: List[ActionExecutionState] = []
        for es in exec_states:
            ex = self._executor_registry.get_executor(es.action_type)
            mutates = getattr(ex, "mutates_world_during_complete", False)
            (world_infra if mutates else readonly).append(es)

        results_by_exec: Dict[str, List[ActionResult]] = {}
        if readonly:
            gathered = await asyncio.gather(
                *(self._adjudicate_execution(es, agents_dict, step) for es in readonly),
                return_exceptions=True,
            )
            for es, res in zip(readonly, gathered):
                if isinstance(res, BaseException):  # defensive; _adjudicate_execution already guards
                    logger.warning(
                        "adjudication_slot_failed",
                        extra={"step": step, "execution_id": es.execution_id, "error": str(res)},
                    )
                    res = []
                results_by_exec[es.execution_id] = res
        for es in world_infra:
            results_by_exec[es.execution_id] = await self._adjudicate_execution(es, agents_dict, step)

        await self._land_results_grouped(exec_states, results_by_exec, agents_dict, step)
        return results_by_exec

    @staticmethod
    def _result_conflict_keys(result: ActionResult) -> set[str]:
        """What one result changes when it lands; results with disjoint keys may land
        concurrently.

        ``entity_spawns`` yields no key: no other result can refer to a new thing, and
        ``spawn_entity`` has no await, so concurrent groups can't interleave inside it."""
        keys: set[str] = {result.action.agent_id}
        for effect in result.target_effects:
            keys.add(effect.agent_id)
        for change in result.entity_state_changes:
            keys.add("item:" + change.entity_id)
        # A body's condition can change only once per beat. Errands aren't keyed: they land in
        # the world in ``ErrandExecutor.start``, and landing never touches them.
        for npc_effect in result.npc_effects:
            keys.add("npc:" + npc_effect.npc_id)
        return keys

    async def _land_results_grouped(
        self,
        exec_states: List[ActionExecutionState],
        results_by_exec: Dict[str, List[ActionResult]],
        agents_dict: Dict[str, "Agent"],
        step: int,
    ) -> None:
        """Union results by conflict key; groups land concurrently, members serially in
        _landing_order. The unit is a result, not an execution: both sides of a conversation
        change only themselves and needn't wait on each other.
        """
        slots: List[tuple[str, ActionResult]] = [
            (es.execution_id, result)
            for es in exec_states
            for result in results_by_exec.get(es.execution_id, [])
        ]
        parent = list(range(len(slots)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        key_owner: Dict[str, int] = {}
        for idx, (_eid, result) in enumerate(slots):
            for key in self._result_conflict_keys(result):
                if key in key_owner:
                    ra, rb = find(key_owner[key]), find(idx)
                    if ra != rb:
                        parent[rb] = ra
                else:
                    key_owner[key] = idx

        groups: Dict[int, List[tuple[str, ActionResult]]] = {}
        for idx, slot in enumerate(slots):  # dicts keep insertion order → stable group order
            groups.setdefault(find(idx), []).append(slot)

        async def _apply_group(members: List[tuple[str, ActionResult]]) -> None:
            for eid, result in self._landing_order(members):
                with annotate_call(execution_id=eid):
                    await self.land_result_feedback(result, agents_dict, step)

        group_results = await asyncio.gather(
            *(_apply_group(g) for g in groups.values()),
            return_exceptions=True,
        )
        for gr in group_results:  # land_result_feedback already guards each item; defensive second layer
            if isinstance(gr, BaseException):
                logger.warning("apply_group_failed", extra={"step": step, "error": str(gr)})

    @staticmethod
    def _landing_order(
        members: List[tuple[str, ActionResult]],
    ) -> List[tuple[str, ActionResult]]:
        """Landing order within a group: feedback on someone's own action lands before others'
        actions on him.

        Topological sort: if result j hits X, every result whose actor is X lands before j.
        Each round takes the first ready result in input order (deterministic); a cycle (mutual
        strikes) falls back to the first remaining one. Groups are small, so O(n²) is fine.
        """
        actor_of = [result.action.agent_id for _eid, result in members]
        before: List[set[int]] = [set() for _ in members]
        for j, (_eid, result) in enumerate(members):
            hit = {e.agent_id for e in result.target_effects} - {actor_of[j]}
            for i, actor in enumerate(actor_of):
                if i != j and actor in hit:
                    before[j].add(i)

        ordered: List[int] = []
        remaining = list(range(len(members)))
        while remaining:
            ready = next((j for j in remaining if not (before[j] - set(ordered))), remaining[0])
            ordered.append(ready)
            remaining.remove(ready)
        return [members[j] for j in ordered]

    async def land_result_feedback(
        self,
        result: ActionResult,
        agents_dict: Dict[str, "Agent"],
        step: int,
    ) -> None:
        """Land one ActionResult's feedback, each item behind its own Rule-1 boundary: a failed
        writeback loses only itself, never the run or the result's other items. Dead agents
        get no writeback.
        """
        agent_obj = agents_dict.get(result.action.agent_id)
        if agent_obj is not None and agent_obj.is_active:
            # The situation anchor was cached at perception time; the feedback chain below reads
            # it and would otherwise say "I'm at A" next to "went from B to C".
            landed_at = (
                result.action.target.acted_on_place if result.action.target is not None else None
            )
            if landed_at:
                agent_obj.refresh_situation(
                    self._environment.spatial_for(agent_id=result.action.agent_id, step=step)
                )
            try:
                await agent_obj.finalize_ongoing_action(result=result, step=step)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "finalize_writeback_failed",
                    extra={"agent_id": result.action.agent_id, "step": step, "error": str(exc)},
                )
        for effect in result.target_effects:
            # A self-targeted action already wrote back through finalize_ongoing_action.
            if effect.agent_id == result.action.agent_id:
                continue
            target = agents_dict.get(effect.agent_id)
            if target is None or not target.is_active:
                continue
            try:
                await target.apply_target_effect(effect, from_agent_id=result.action.agent_id, step=step)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "target_effect_failed",
                    extra={"agent_id": effect.agent_id, "step": step, "error": str(exc)},
                )
        for change in result.entity_state_changes:
            try:
                self._environment.change_entity_state(change, acting_agent_id=result.action.agent_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "entity_state_change_failed",
                    extra={"agent_id": result.action.agent_id, "step": step, "error": str(exc)},
                )
        for spawn in result.entity_spawns:
            try:
                # A refusal is legitimate, but memory already says "I made it", so log at error
                # like a Rule 6 write failure.
                actor_id = result.action.agent_id
                if not self._environment.spawn_entity(
                    spawn, ground=self._environment.get_body_location(actor_id), actor_id=actor_id,
                ):
                    logger.error(
                        "entity_spawn_refused",
                        extra={
                            "agent_id": result.action.agent_id, "step": step,
                            "entity_name": spawn.name,
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "entity_spawn_failed",
                    extra={"agent_id": result.action.agent_id, "step": step, "error": str(exc)},
                )

        for npc_effect in result.npc_effects:
            try:
                self._environment.apply_npc_effect(npc_effect)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "npc_effect_failed",
                    extra={"npc_id": npc_effect.npc_id, "step": step, "error": str(exc)},
                )

    def release_stuck_participants(
        self,
        exec_state: ActionExecutionState,
        agents_dict: Dict[str, "Agent"],
    ) -> None:
        """Reset to idle the surviving participants of an execution that couldn't finalize
        normally; otherwise they stay IN_PROGRESS forever once it leaves the registry.
        Touches only action-lifecycle fields (null step)."""
        for pid in exec_state.participant_ids:
            agent = agents_dict.get(pid)
            if agent is None or not agent.is_active:
                continue
            if agent.personality.state.action_status != ActionStatus.IN_PROGRESS:
                continue
            other = self._executor_registry.get_active_for_agent(pid)
            if other is not None and other.execution_id != exec_state.execution_id:
                continue  # attached to another execution, which handles its own cleanup
            agent.personality.update_action_status(
                status=ActionStatus.IDLE, current_action=None, remaining_steps=0,
            )
            logger.warning(
                "stuck_participant_released",
                extra={"agent_id": pid, "execution_id": exec_state.execution_id},
            )

    async def force_teardown(
        self,
        agent: "Agent",
        agents_dict: Dict[str, "Agent"],
        step: int,
        *,
        cause: str,
        trigger: str,
    ) -> None:
        """Pull someone out of every active execution, by a force outside the world (death, or
        the director moving him).

        Not an interrupt (``InterruptCoordinator._apply_interrupt``), where the agent decides
        to stop: ``thought`` is empty, the subject's own writeback is skipped (he neither
        remembers nor failed anything), and the execution is removed even if ``interrupt``
        raises, since a leftover would keep driving a corpse. Surviving participants get their
        interrupt writeback and are released, or they'd stay IN_PROGRESS forever.

        ``cause`` is a third-person narrative phrase; ``trigger`` is a log-only label. The
        survivors' records wait in ``take_forced_records`` for the step's display stream.
        """
        self._forced_records.extend(await self._tear_down(
            agent, agents_dict, step, cause=cause, trigger=trigger, write_back_subject=False,
        ))

    async def teardown_with_writeback(
        self,
        agent: "Agent",
        agents_dict: Dict[str, "Agent"],
        step: int,
        *,
        cause: str,
        trigger: str,
    ) -> List[Dict[str, Any]]:
        """A forced teardown the subject remembers: another person in the world pulled him
        away, so it enters his memory like any abandoned task. ``thought`` is empty and
        ``cause`` comes from whoever pulled him. Returns one record per result
        (phase="interrupt").
        """
        return await self._tear_down(
            agent, agents_dict, step, cause=cause, trigger=trigger, write_back_subject=True,
        )

    async def _tear_down(
        self,
        agent: "Agent",
        agents_dict: Dict[str, "Agent"],
        step: int,
        *,
        cause: str,
        trigger: str,
        write_back_subject: bool,
    ) -> List[Dict[str, Any]]:
        """Shared by both forced paths; they differ only in whether the subject's writeback lands."""
        records: List[Dict[str, Any]] = []
        for exec_state in list(self._executor_registry.all_active()):
            if agent.agent_id not in exec_state.participant_ids:
                continue
            executor = self._executor_registry.get_executor(exec_state.action_type)
            results: List[ActionResult] = []
            if executor is not None:
                try:
                    with observe_stage(Stage.ACTION, agent_id=exec_state.initiator_id), \
                            annotate_call(**execution_annotations(self._directory, exec_state)):
                        results = await executor.interrupt(
                            exec_state,
                            step,
                            agents=agents_dict,
                            environment=self._environment,
                            interrupted_agent_id=agent.agent_id,
                            thought="",
                            cause=cause,
                        )
                except Exception as exc:  # noqa: BLE001 — teardown must not be abandoned because the executor failed
                    logger.warning(
                        "forced_teardown_interrupt_failed",
                        extra={
                            "trigger": trigger,
                            "agent_id": agent.agent_id,
                            "execution_id": exec_state.execution_id,
                            "step": step,
                            "error": str(exc),
                        },
                    )
            for result in results:
                if result.action.agent_id == agent.agent_id and not write_back_subject:
                    continue
                with annotate_call(execution_id=exec_state.execution_id):
                    await self.land_result_feedback(result, agents_dict, step)
                # One record per result, not just the subject's, or the other participants'
                # conversation never gets an ending in the feed.
                record = self.completion_record(
                    exec_state, result, agents_dict, phase="interrupt",
                )
                # Nobody inside chose to stop; any name here would read as that.
                record["interrupted_by"] = ""
                records.append(record)
            self._executor_registry.remove_active(exec_state.execution_id)
            self.release_stuck_participants(exec_state, agents_dict)

        # Otherwise next step's in_progress_at_step_start still counts him as busy.
        if agent.personality.state.action_status == ActionStatus.IN_PROGRESS:
            agent.personality.update_action_status(
                status=ActionStatus.IDLE, current_action=None, remaining_steps=0,
            )
        return records

    def completion_record(
        self,
        exec_state: ActionExecutionState,
        result: ActionResult,
        agents_dict: Dict[str, "Agent"],
        phase: str = "ongoing_complete",
    ) -> Dict[str, Any]:
        """Build one participant's terminal-action record from a finalized result.

        ``phase``: "ongoing_complete", "interrupt", or "settled" (opened and closed in one beat).
        Same shape for all three; an interrupt fills only the god-view ``outcome`` and isn't
        carried (see ``ActionExecutor.interrupt``).
        """
        agent_obj = agents_dict.get(result.action.agent_id)
        return {
            "agent_id": result.action.agent_id,
            "agent_name": self._directory.agent_name(result.action.agent_id),
            "is_main_character": agent_obj.is_main_character if agent_obj else False,
            # Stamped here, not by callers: a missing one makes the feed show a joint action twice.
            "execution_id": exec_state.execution_id,
            # JSON-native, NOT the live enum: json.dumps turns a str-Enum into "talk" on disk,
            # but the live path's str(...) gives "ActionType.TALK", silently breaking live
            # rendering while replay looks right.
            "action_type": str(getattr(result.action.action_type, "value", result.action.action_type or "")),
            "action_description": result.action.action_description,
            "location_id": self._environment.get_body_location(result.action.agent_id),
            "phase": phase,
            "outcome": result.outcome,            # full authority (web / snapshot / god view)
            "gist": result.gist,
            # Bystander view, one entry per place; dicts to stay JSON-native.
            "observations": [o.__dict__ for o in result.observations],
            "succeeded": result.succeeded,
            "failure_reason": result.failure_reason,  # 3p "why it failed"; renderers use it as is
            "detected": result.detected,          # COVERT exposure, as a structured flag
            # The layer only close watchers catch (see ActionResult.happening); empty = this act
            # has no such layer.
            "happening": result.happening,
            "adjudication_failed": result.adjudication_failed,  # null steps aren't carried
            "not_executed": result.not_executed,  # unmet-precondition non-event (muted when rendered)
            "dialogue": list(result.dialogue),
            # From landed effects (what happened), not exec_state's list (intent). Kept apart
            # from participant_ids: listeners didn't spend their turn.
            "overheard_by": [e.agent_id for e in result.target_effects if e.overheard],
            # Carry filters these out too: they have their own first-person memory.
            "acted_upon": [e.agent_id for e in result.target_effects if not e.overheard],
            "participant_ids": list(exec_state.participant_ids),  # every member self-filters
            # Lets an observer folding the execution's records pick the initiator's outcome.
            "initiator_id": exec_state.initiator_id,
            **action_semantics(result.action, result),
        }

    def merge_completion_into_records(
        self,
        exec_state: ActionExecutionState,
        results: List[ActionResult],
        agent_records: List[Dict[str, Any]],
        agents_dict: Dict[str, "Agent"],
    ) -> None:
        """Fold a born-zero execution's completion into this step's begin record (matched on
        execution_id, agent_id), so consumers see one record, keeping the begin record's
        perception context and its observations at places the completion isn't seen.

        No results means a null step: the begin records are marked ``adjudication_failed``,
        withheld from the observer and never carried.
        """
        if not results:
            for r in agent_records:
                if r.get("execution_id") == exec_state.execution_id:
                    r["adjudication_failed"] = True
            return
        by_key = {
            (r.get("execution_id"), r.get("agent_id")): r
            for r in agent_records if r.get("execution_id") is not None
        }
        for result in results:
            # "settled": neither an opening nor a closing, so neither tag fits.
            completion = self.completion_record(exec_state, result, agents_dict, "settled")
            rec = by_key.get((exec_state.execution_id, result.action.agent_id))
            if rec is None:
                agent_records.append(completion)
                continue
            completion["visible_agent_ids"] = list(rec.get("visible_agent_ids", []))
            completion["message_ids"] = list(rec.get("message_ids", []))
            # Only the begin record knows the decision's thought; completion_record must not.
            completion["inner_monologue"] = rec.get("inner_monologue", "")
            # The completion supersedes the opening only where it is itself seen. Elsewhere the
            # opening still happened: a one-step MOVE's origin saw him leave and appears in no
            # completion observation.
            covered = {o["location_id"] for o in completion["observations"]}
            completion["observations"] = [
                *(o for o in rec.get("observations") or [] if o.get("location_id") not in covered),
                *completion["observations"],
            ]
            completion["summary"] = (
                f"行动内容：{result.action.action_description}，行动结果：{result.outcome}"
            )
            rec.clear()
            rec.update(completion)
