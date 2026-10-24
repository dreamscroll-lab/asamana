"""Staged runtime loop for an Asamana world."""

from __future__ import annotations


import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping

from agent.agent import AgentStepPlan, snapshot_agent_state
from agent.decision import DecisionStatus
from agent.need import NeedEvaluation
from agent.personality import ActionStatus, activity_status_for
from core.context import annotate_call, observe_stage, set_log_context
from core.event_bus import NarrativeEventBus
from core.interfaces.action import COVERTABLE_ACTION_TYPES
from core.interfaces.action import Ref
from core.interfaces.actor import (
    ActorAmendments,
    ActorPublicAnswer,
    ActorMind,
    ActorPace,
    ActorTurn,
    ActorIntent,
    ActorIntentResult,
    ActorOperations,
)
from core.interfaces.relation import relation_to_snapshot_dict
from core.interfaces.directory import WorldDirectory
from core.interfaces.snapshot import EngineState, SnapshotProvider, WorldSnapshot, WorldState
from core.interfaces.trace import CONSCRIPTED, BeatKind, Stage, StepTrace, TraceSink, record_beat
from core.duration import describe_duration
from core.logging import get_logger
from core.validate import require_at_least
from engine.actor_channel import ActorChannel
from engine.actor_coordinator import ActorCoordinator
from engine.signals import perceived_signals
from engine.broadcast import BroadcastChannel
from engine.presence import attach_presence
from engine.clock import GlobalClock, WorldTime
from engine.cognition_maintenance import CognitionMaintenance
from engine.environment import IN_TRANSIT, UNPLACED, EnvironmentSystem
from engine.narrative_editor import NarrativeEditorSettings, NarrativeEditor
from engine.event_seq import EventSequencer
from engine.step_record import action_events, broadcast_events, injection_events, message_events
from engine.arbiter import ArbitratedAction, ExecutionArbiter
from engine.death_handler import DeathHandler
from engine.director import DirectorChannel
from engine.injection import Author, InjectionDispatcher, serialize_injection
from engine.intervention_receipt import build_receipt
from engine.executors.base import action_semantics
from engine.world_mutation import WorldMutationChannel
from engine.execution_processor import ExecutionProcessor
from engine.interrupt_coordinator import InterruptCoordinator
from engine.executors.movement import arrival_view, transit_view
from engine.executors.registry import ActionExecutorRegistry
from engine.message_system import MessageDelivery, MessageSystem
from engine.npc_runner import NpcRunner
from engine.run_control import RunController, StepWithdrawn
from engine.scheduler import AgentScheduler, StepExecutionPlan
from core.interfaces.llm import LLMRouter
from engine.world_pressure import WorldPressureEvaluator
if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.perception import Broadcast, PerceivedSignals, SpatialPerception


logger = get_logger(__name__)

@dataclass
class RuntimeStepResult:
    step: int
    world_time: str
    action_summaries: List[str]
    injection_summaries: List[str]
    delivered_message_count: int = 0
    scheduled_agent_ids: List[str] = field(default_factory=list)


class NarrativeRuntime:
    """Run a small world loop.

    Boundary: the Runtime talks to subsystems through channels.
    - Event signals flow in from BroadcastChannel / MessageSystem and are consumed by perception.
    - The two authors (NarrativeEditor / DirectorChannel) hand back landed injections as
      CommittedInjection; the Runtime only serializes and assembles them.
    - Each piece of information takes one channel and is perceived once: action outcomes via
      carry → ambient; events via broadcast/message the same step, never carried again.
    """

    def __init__(
        self,
        *,
        world_id: str,
        clock: GlobalClock,
        scheduler: AgentScheduler,
        environment: EnvironmentSystem,
        message_system: MessageSystem,
        narrative_editor_settings: NarrativeEditorSettings,
        snapshot_provider: SnapshotProvider,
        event_bus: NarrativeEventBus,
        broadcast_channel: BroadcastChannel,
        directory: WorldDirectory,
        executor_registry: ActionExecutorRegistry | None = None,
        maintenance: CognitionMaintenance | None = None,
        pressure_evaluator: WorldPressureEvaluator | None = None,
        llm_router: LLMRouter,
        trace_sink: TraceSink | None = None,
        start_event_seq: int = 0,
        landed_injections: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self._world_id = world_id
        self._clock = clock
        self._scheduler = scheduler
        self._environment = environment
        self._message_system = message_system
        self._snapshot_provider = snapshot_provider
        self._event_bus = event_bus
        self._broadcast_channel = broadcast_channel
        self._directory = directory
        # Empty by default: arbitration skips an action type with no executor. Pass
        # build_default_registry() for real routing.
        self._executor_registry: ActionExecutorRegistry = (
            executor_registry if executor_registry is not None else ActionExecutorRegistry()
        )
        self._maintenance = maintenance if maintenance is not None else CognitionMaintenance()
        self._pressure_evaluator = pressure_evaluator
        self._trace_sink = trace_sink
        # Trace is a pure audit log and must never gate the step: per-step flushes run
        # fire-and-forget on worker threads, tracked only so run()'s shutdown can drain them.
        # Losing the last flush on a crash is acceptable.
        self._trace_flush_tasks: set[asyncio.Task[None]] = set()
        # Reentrancy latch: two concurrent run() loops would interleave step state and
        # double-fire the shutdown in run()'s finally. Guarded on the state's owner so no
        # caller can bypass it.
        self._running: bool = False
        self._step_in_progress: bool = False
        self._event_seqs = EventSequencer(start_seq=start_event_seq)
        # Advances ongoing executions (tick) and lands completions/feedback; the main sweep,
        # death teardown and interrupts all land through it. Shares the same executor_registry.
        self._processor = ExecutionProcessor(
            executor_registry=self._executor_registry,
            environment=self._environment,
            message_system=self._message_system,
            directory=self._directory,
        )
        # Concurrency arbitration (planned_steps → arbitration map). Shares the same
        # executor_registry the runtime mutates elsewhere.
        self._arbiter = ExecutionArbiter(
            executor_registry=self._executor_registry,
            environment=self._environment,
            message_system=self._message_system,
            directory=self._directory,
            clock=self._clock,
            processor=self._processor,
        )
        # The actor a person performs a character through, once ``cast_actor`` hands it one. The
        # step loop never asks it: the character does. Each turn opening and closing, and each
        # entry of the actor's thread, goes out on the event bus, like steps.
        self._actor = ActorCoordinator(
            clock.config, on_turn=self._publish_actor_turn, on_entry=self._publish_actor_entry,
            world_id=world_id, trace_sink=trace_sink,
        )
        # Owns its edge-trigger state (_interrupt_seen_goals); lands torn-down actions
        # through the processor.
        self._interrupts = InterruptCoordinator(
            executor_registry=self._executor_registry,
            environment=self._environment,
            directory=self._directory,
            clock=self._clock,
            processor=self._processor,
        )
        # Vitality decay + reaping new deaths (remove from world, release items, tear down
        # executions, publish the death broadcast).
        self._deaths = DeathHandler(
            environment=self._environment,
            processor=self._processor,
            broadcast_channel=self._broadcast_channel,
        )
        # The tier without cognition, advanced one hop per step. Built here, not in the
        # container: all its dependencies are already runtime-owned.
        self._npc_runner = NpcRunner(
            environment=self._environment,
            message_system=self._message_system,
            directory=self._directory,
            world_id=self._world_id,
            seconds_per_step=self._clock.config.seconds_per_step,
        )
        # The two authors share one dispatcher → one mutation channel (per-author permissions
        # checked inside). Built here because the channel must share this runtime's processor:
        # moving someone tears down the execution this runtime ticks, which a processor on
        # another registry can't do.
        mutations = WorldMutationChannel(
            environment=self._environment,
            processor=self._processor,
            seconds_per_step=self._clock.config.seconds_per_step,
        )
        dispatcher = InjectionDispatcher(
            broadcast_channel=self._broadcast_channel,
            message_system=self._message_system,
            mutation_channel=mutations,
            environment=self._environment,
            directory=self._directory,
        )
        self._narrative_editor = NarrativeEditor(
            llm_router=llm_router,
            snapshot_provider=self._snapshot_provider,
            dispatcher=dispatcher,
            directory=self._directory,
            settings=narrative_editor_settings,
        )
        self._director = DirectorChannel(
            llm_router=llm_router,
            dispatcher=dispatcher,
            directory=self._directory,
        )
        # Both get the same history and each keeps only its own (separate ledgers, see
        # engine/injection.py).
        fired = list(landed_injections)
        self._narrative_editor.restore_state(fired)
        self._director.restore_state(fired)

    @property
    def director(self) -> DirectorChannel:
        """The human author's channel — the API submits directives through this."""
        return self._director

    @property
    def actor(self) -> ActorChannel:
        """The actor's channel: the turn waiting for the actor, and its withdrawal."""
        return self._actor.channel

    def answer_actor(self, answer: ActorPublicAnswer) -> None:
        """Answer the turn waiting for the actor (``ActorCoordinator.answer``)."""
        self._actor.answer(answer)

    def cast_actor(self, agent: Agent, *, pace: ActorPace) -> None:
        """Have a person perform ``agent`` through this world's actor, asked at ``pace``."""
        self._actor.perform(agent, pace=pace)

    def set_actor_pace(self, pace: ActorPace) -> None:
        self._actor.set_pace(pace)

    def _publish_actor_turn(self, turn: ActorTurn | None) -> None:
        self._event_bus.publish({"type": "actor_turn", "world_id": self._world_id, "turn": turn})

    def _publish_actor_entry(self, entry: dict[str, Any]) -> None:
        self._event_bus.publish({"type": "actor_entry", "world_id": self._world_id, "entry": entry})

    @property
    def actor_live_thread(self) -> list[dict[str, Any]]:
        """The actor's thread since the last committed step (``ActorCoordinator.live_thread``)."""
        return self._actor.live_thread

    async def actor_mind(self) -> ActorMind:
        """The performed character's mind as the actor sees it (``ActorCoordinator.mind``)."""
        return await self._actor.mind(step=self._clock.current_step)

    async def amend_actor_mind(self, amendments: ActorAmendments) -> bool:
        """Save the actor's amendments; whether they landed now (``ActorCoordinator.amend``)."""
        return await self._actor.amend(amendments)

    async def actor_intent(self, intent: ActorIntent) -> ActorIntentResult:
        return await self._actor.intent(intent)

    def actor_operations(self, turn_id: str, target: Ref) -> ActorOperations:
        return self._actor.operations(turn_id, target)

    def actor_break_off(self) -> None:
        """Break off what the performed character is doing (``ActorCoordinator.break_off``)."""
        self._actor.break_off()

    @property
    def clock(self) -> GlobalClock:
        """This world's clock. Read-only to callers — only ``run_step`` may tick it."""
        return self._clock

    @property
    def is_running(self) -> bool:
        """Whether a run loop is advancing this runtime.

        The only answer that also covers a run started via ``run`` directly (``run_world``),
        which ``NarrativeApplication``'s reset/delete guards must not orphan.
        """
        return self._running

    async def run(
        self,
        agents: List[Agent],
        *,
        total_steps: int,
        controller: "RunController | None" = None,
    ) -> List[RuntimeStepResult]:
        """Advance ``total_steps`` steps (required, positive), or fewer if a stop is requested.

        No run-forever mode: each step costs several LLM calls per character, so run length is
        the caller's call. A ``controller`` pauses/resumes/stops at step boundaries; each step
        persists at its end, so this loses no step.
        """

        require_at_least(total_steps, 1, "total_steps")
        if self._running:
            raise ValueError(f"World {self._world_id!r} is already running.")
        self._running = True
        results: list[RuntimeStepResult] = []
        try:
            step_index = 0
            while step_index < total_steps:
                if controller is not None:
                    # Park here while paused; wake to break cleanly on stop.
                    await controller.wait_if_paused()
                    if controller.stop_requested:
                        break
                results.append(await self.run_step(agents))
                step_index += 1
        finally:
            # Don't close the narrative editor here: generation spans steps (started at N, consumed
            # by _consume_ready at N+k), and single-step advancing (the director's "inject one,
            # step, look") returns from run() every step, so it would never consume an injection
            # while still paying for gate + generation calls. aclose() closes it at real shutdown.
            #
            # Drain each agent's memory-write worker so a clean shutdown loses no trailing
            # writes. Worker failures are already swallowed (Rule 1); gather isolates one
            # aclose failure from the rest.
            await asyncio.gather(
                *(agent.memory_system.aclose() for agent in agents),
                return_exceptions=True,
            )
            # Drain fire-and-forget trace flushes (errors are swallowed inside the sink).
            if self._trace_flush_tasks:
                await asyncio.gather(*self._trace_flush_tasks, return_exceptions=True)
            # Final flush-all safety net: deferred writes (memory refine, the event-plan task)
            # trace with a stale step frozen by create_task, whose segment was flushed long ago.
            # flush(world_id) with no step writes every remaining segment. Runs after all
            # producers are closed, so no buffer race.
            if self._trace_sink is not None:
                await self._trace_sink.flush(self._world_id)
            # Release the re-entry latch last: if it opened during the awaits above, a second run
            # could slip in while this shutdown is closing the memory-write queue it just started.
            self._running = False
        return results

    async def aclose(self) -> None:
        """Called when the world actually shuts down (session destroyed/reset), not per run return.

        Handles only the cross-step state that run()'s finally deliberately leaves alone: the
        in-flight event-generation task.
        """
        await self._narrative_editor.aclose()

    @property
    def step_in_progress(self) -> bool:
        """Whether this step's traces can still receive arbitration and execution annotations."""
        return self._step_in_progress

    async def run_step(self, agents: List[Agent]) -> RuntimeStepResult:
        if self._step_in_progress:
            raise ValueError(f"World {self._world_id!r} already has a step in progress.")
        self._step_in_progress = True
        try:
            return await self._run_step(agents)
        finally:
            self._step_in_progress = False

    async def _run_step(self, agents: List[Agent]) -> RuntimeStepResult:
        step_start = time.perf_counter()
        agents_dict = {agent.agent_id: agent for agent in agents}
        # Sample the living set before any tick/decay/commit: death handling finds new deaths as
        # "alive at step start ∧ dead at step end", whatever the lethal source.
        pre_step_active = {a.agent_id for a in agents if a.is_active}
        world_time = self._clock.tick()
        step = world_time.step
        # Step-wide log context; gathered child tasks copy it and observe_stage() layers
        # agent_id/stage on top.
        set_log_context(world_id=self._world_id, step=str(step))
        logger.info(
            "step_start",
            extra={
                "world_id": self._world_id,
                "step": step,
                "agent_count": len(agents),
            },
        )
        self._environment.begin_step(step=step, world_time=world_time)
        # Records a step that raised never took would otherwise show up as this step's.
        self._processor.take_forced_records()
        # Self-limiting conditions expire here, before agent_spatials is built (not in the
        # step-end maintenance), or someone who woke up would look restrained for a step. No
        # ambient or memory write: code writing "he slowly came to" would add drama.
        for agent in agents:
            agent.expire_condition(step)
        # The broadcast channel is a deliver_step queue, so it needs no clearing; collect()
        # below dequeues this step's due broadcasts.

        # Injection: land the injections ready for this step. They enter perception through
        # the channels; the returned CommittedInjection is only assembled into the snapshot.
        # Director first, then the narrative editor: a person's intent shouldn't queue behind
        # automatic generation. This is the world's only injection point and its position is
        # relied on (see "Where it lands" in engine/world_mutation.py). Each author runs under
        # its own stage, so a trace tells a human's beat from the editor's.
        _t0 = time.perf_counter()
        with observe_stage(Stage.DIRECTOR):
            # The queue holds validated plans, so this is pure dispatch with no planning LLM.
            with annotate_call(settles_prior_action=True):
                # A director move tears down what the person was doing: start-of-step
                # settlement, which review separates from new cognition (same marker below).
                director_injections = await self._director.drain(step=step, agents=agents_dict)
        # observe_stage also covers background generation tasks started inside.
        with observe_stage(Stage.NARRATIVE_EDITOR):
            editor_injection = await self._narrative_editor.poll_injection(
                current_step=step,
                world_time=world_time,
                world_id=self._world_id,
                all_agents=agents_dict,
                agent_locations=self._environment.body_locations(agents_dict),
                locations=self._environment.space.all_places(),
                entities=self._environment.all_live_entities(),
            )
        # Injections land without an LLM call of their own: each goes into the trace as a beat
        # of the step it lands in, under its author's stage.
        for stage, committed in (
            *((Stage.DIRECTOR, c) for c in director_injections),
            *(((Stage.NARRATIVE_EDITOR, editor_injection),) if editor_injection is not None else ()),
        ):
            record_beat(self._trace_sink, BeatKind.INJECTION, stage, {
                "narrative_desc": committed.record.narrative_desc,
                "affected_names": committed.record.affected_names,
                "location_label": committed.record.location_label,
                "dispatched_to": committed.record.dispatched_to,
            }, world_id=self._world_id, step=step)
        # Reap whoever an injection just killed, before anything acts this step: otherwise the
        # dead keep ticking their actions, get interrupt calls and complete talks until the
        # step-end pass. Taken out of pre_step_active so that pass doesn't wrap them up twice.
        await self._deaths.process_new_deaths(agents, pre_step_active, step)
        pre_step_active = {aid for aid in pre_step_active if agents_dict[aid].is_active}
        self._sync_dead_ids(agents_dict)
        # Amendments to a character's mind saved between steps land before it perceives this one.
        for agent in agents:
            await agent.land_amendments(step)
        injection_ms = round((time.perf_counter() - _t0) * 1000, 1)
        logger.debug("injection_check_done", extra={"step": step, "elapsed_ms": injection_ms, "director_injections": len(director_injections), "editor_injected": editor_injection is not None})

        _t1 = time.perf_counter()
        deliveries = await self._message_system.deliver_for_agents(
            step=step,
            # The dead have no inbox and aren't in the world-wide broadcast set.
            agent_ids=[agent.agent_id for agent in agents if agent.is_active],
            environment=self._environment,
        )
        message_ms = round((time.perf_counter() - _t1) * 1000, 1)
        logger.debug("message_delivery_done", extra={"step": step, "elapsed_ms": message_ms})

        # Taken once, before ticks, so pressure, perception and planning share one consistent
        # world state.
        agent_spatials: dict[str, SpatialPerception] = {
            agent.agent_id: self._environment.spatial_for(
                agent_id=agent.agent_id,
                step=step,
                world_time=world_time.time_label,
            )
            for agent in agents
        }
        for spatial in agent_spatials.values():
            attach_presence(
                spatial, directory=self._directory, agents=agents_dict,
                environment=self._environment,
            )

        # This step's due broadcasts: events injected above and death notices published during
        # the previous step's execution.
        broadcasts_this_step = self._broadcast_channel.collect(step=step)
        # What reached each living agent this step; one definition for pressure and each
        # character's own reading of it.
        agent_signals = {
            agent.agent_id: perceived_signals(
                location_id=agent_spatials[agent.agent_id].location_id,
                inbox=deliveries.inbox_for(agent.agent_id),
                broadcasts=broadcasts_this_step,
                ambient=agent_spatials[agent.agent_id].ambient_events,
                acted_upon=agent.personality.state.acted_upon,
            )
            for agent in agents
            if agent.is_active
        }

        # --- pressure → perceive → interrupt → tick → plan → commit ---
        # Perception reads the pre-tick world (agent_spatials).
        # pressure/interrupt before tick: an agent decides whether to continue before executing.
        # pressure before perceive, never concurrently: perceive writes relations
        #   (apply_interaction for message senders) that pressure reads (_sender_relations /
        #   _co_located); concurrency would make pressure's view nondeterministic and break
        #   replay. Ordered by this coupling alone: perceive doesn't read external_goals.
        # perceive before interrupt: an interrupted agent's perception memory should include the
        # signal that triggered the interrupt.

        pressure_ms = await self._apply_world_pressure(
            agents_dict,
            signals=agent_signals,
            agent_spatials=agent_spatials,
            world_time_label=world_time.time_label,
            step=step,
        )

        _t_perceive = time.perf_counter()
        await self._perceive_all_agents(
            agents=agents_dict,
            deliveries=deliveries,
            agent_spatials=agent_spatials,
            broadcasts=broadcasts_this_step,
            step=step,
        )
        perceive_ms = round((time.perf_counter() - _t_perceive) * 1000, 1)

        # Who is IN_PROGRESS at step start, sampled before interrupts and ticks mutate
        # action_status. They don't decide this step: their completion reaches ambient only on
        # the next step's carry, so deferring lets them perceive the result first. Interrupted
        # agents are subtracted below.
        in_progress_at_step_start = {
            agent_id
            for agent_id, agent in agents_dict.items()
            if agent.personality.state.action_status == ActionStatus.IN_PROGRESS
        }

        _t_interrupt = time.perf_counter()
        with annotate_call(settles_prior_action=True):
            interrupt_records = await self._interrupts.evaluate_interrupts(
                deliveries, broadcasts_this_step, agents_dict, step, signals=agent_signals,
            )
        interrupt_ms = round((time.perf_counter() - _t_interrupt) * 1000, 1)

        # Torn-down agents re-enter cognition this step: the set means "its own action took
        # this step", and an interrupt is the world stopping it. Freezing them a step would lose,
        # unrecoverably:
        #   · pending_external_goals: rewritten every step, never in memory, trigger already
        #     delivered.
        #   · Perception emotion and need_activation: only evaluated in plan_step.
        # Safe: apply_interrupt has landed feedback synchronously and reset the state.
        # Every participant is subtracted (interrupt_records has one entry each), not just the
        # interrupter: the other side of an overturned talk is free too.
        in_progress_at_step_start -= {r["agent_id"] for r in interrupt_records}
        # Only the one who chose to stop (agent_id == interrupted_by) tells the cadence gate
        # "this was worth dropping what I was doing"; collateral companions go through as usual.
        interrupters = {
            r["agent_id"] for r in interrupt_records
            if r.get("interrupted_by") == r["agent_id"]
        }

        # Advance ongoing multi-step executions after pressure/interrupt so agents
        # decide whether to continue before executing the next tick.
        _t2 = time.perf_counter()
        # Executions finished this step leave the registry; keep them to report arrivals.
        settling = list(self._executor_registry.all_active())
        with annotate_call(settles_prior_action=True):
            tick_records = await self._processor.tick_ongoing_executions(agents_dict, step)
        executor_ms = round((time.perf_counter() - _t2) * 1000, 1)
        logger.debug("executor_tick_done", extra={"step": step, "elapsed_ms": executor_ms})

        # Apply passive vitality decay before scheduling so dead agents are excluded this step.
        await self._deaths.apply_vitality_decay(agents, step)
        self._sync_dead_ids(agents_dict)

        # Scheduler excludes the step-start IN_PROGRESS set, then admits only the idle bodies
        # its cadence gate judges worth a decision. The gate reads pending_external_goals, so
        # the pressure phase must stay ahead of planning.
        plan = self._scheduler.plan(
            agents,
            step=step,
            in_progress_at_step_start=in_progress_at_step_start,
            interrupters=interrupters,
            signals=agent_signals,
        )

        plan_start = time.perf_counter()
        planned_steps = await self._plan_execution(
            plan=plan,
            agents=agents,
            in_progress_at_step_start=in_progress_at_step_start,
            deliveries=deliveries,
            step=step,
            world_time_label=world_time.time_label,
            agent_spatials=agent_spatials,
            broadcasts=broadcasts_this_step,
        )
        plan_elapsed_ms = (time.perf_counter() - plan_start) * 1000.0
        # Count deciders (spent LLM calls), idle (conscriptable) and busy (listed only to be
        # compellable, see _plan_execution) separately: decider/idle gauges the cadence gate, and
        # folding busy into idle would make it drift with how many are on long tasks.
        decider_count = len(plan.ordered_agents())
        busy_count = sum(
            1 for _phase, _agent, p in planned_steps
            if p.agent_id in in_progress_at_step_start
        )
        logger.info(
            "planning_phase_complete",
            extra={
                "world_id": self._world_id,
                "step": step,
                "decider_count": decider_count,
                "idle_body_count": len(planned_steps) - decider_count - busy_count,
                "busy_body_count": busy_count,
                "elapsed_ms": round(plan_elapsed_ms, 2),
            },
        )

        exec_start = time.perf_counter()
        arbitration, seize_records = await self._arbiter.arbitrate(
            planned_steps, agents_dict, step, in_progress_at_step_start
        )
        agent_records = await self._commit_execution(planned_steps, arbitration, agents_dict)
        # Same-tick completion sweep. The decision step counts as step 1, so a duration-1
        # execution is born with remaining_steps==0 and must complete this step; ticks run only
        # at step start and miss it. Every remaining<=0 execution (incl. create_failed) is
        # finalized and merged into its agent_record in place, so consumers see one record.
        # remaining_steps<=0 is the system's only completion criterion (same as tick).
        pending = [
            exec_state
            for exec_state in list(self._executor_registry.all_active())
            if exec_state.remaining_steps <= 0
        ]
        if pending:
            # Same tick, two phases: concurrent adjudication + grouped parallel settlement.
            # Completion results merge into agent_records in pending order, so the record order
            # in web/snapshot/carry doesn't depend on which concurrent completion finished first.
            results_by_exec = await self._processor.finalize_executions_batch(pending, agents_dict, step)
            for exec_state in pending:
                self._processor.merge_completion_into_records(
                    exec_state,
                    results_by_exec.get(exec_state.execution_id, []),
                    agent_records,
                    agents_dict,
                )
                self._executor_registry.remove_active(exec_state.execution_id)
        # NPCs advance one hop. After the same-tick sweep, so an errand just taken (born-zero) is
        # visible, though taking it uses up this tick (see ``NpcRunner._advance_one``). Before
        # carry, so its traces make this tick's delivery. After every agent's execution, so
        # whoever saw him last tick can stop him now; reversed, interception is a tick late.
        # Deterministic, zero LLM: all the discretion happened on the ERRAND tick.
        await self._npc_runner.advance(step, agents_dict)
        # After the same-tick sweep, where a born-zero PHYSICAL's lethal target_effect lands.
        # Tick-path deaths (ExecutionProcessor.tick_ongoing_executions) are covered by the same
        # pass.
        await self._deaths.process_new_deaths(agents, pre_step_active, step)
        self._sync_dead_ids(agents_dict)
        exec_elapsed_ms = (time.perf_counter() - exec_start) * 1000.0
        logger.info(
            "execution_phase_complete",
            extra={
                "world_id": self._world_id,
                "step": step,
                # Total bodies with entries (including mid-action ones listed only so they can be
                # compelled), not how many acted this step.
                "agent_count": len(planned_steps),
                "elapsed_ms": round(exec_elapsed_ms, 2),
            },
        )
        _t_maintain = time.perf_counter()
        await self._maintenance.run(agents, step=step)
        maintain_ms = round((time.perf_counter() - _t_maintain) * 1000, 1)

        # Carry step outcomes to next step's ambient (rule-based, no LLM).
        # Every adjudication of this step must come before this line: the covert material window
        # relies on it to never read its own tick. Guarded by
        # test_covert_material_never_includes_current_step.
        self._carry_step_observations(
            agent_records=agent_records,
            tick_records=tick_records,
        )
        self._record_step_happenings(
            agent_records=agent_records,
            tick_records=tick_records,
            step=step,
        )

        # The two authors keep separate ledgers (so "the quota only constrains the LLM editor" is
        # structural) and merge into one stream here: observers / replay / frontend ask one
        # question. authored_by says who wrote each; the director's landed first.
        # Receipt: who this tick actually moved. Stamped here because the phases it waits on
        # (pressure / interrupt / admission) finish only now. Without it, a delivered-but-ignored
        # intervention looks exactly like a bug.
        decided_ids = {agent.agent_id for agent in plan.ordered_agents()}
        interrupted_ids = {r["agent_id"] for r in interrupt_records}
        # Receipts only for the director: they answer "who did this push move", and the reader is
        # whoever gave the order.
        injections = [*director_injections, *([editor_injection] if editor_injection is not None else [])]
        serialized_events: list[dict[str, Any]] = []
        for injection in injections:
            payload = serialize_injection(injection.record)
            # What onlookers at each place were given to perceive of the change, in the shape of
            # an action's observations: the place's record shows what happened there.
            payload["observations"] = [
                {"location_id": o.location_id, "text": o.text, "strength": o.strength}
                for o in injection.observations
            ]
            if injection.record.authored_by is Author.DIRECTOR:
                payload["receipt"] = build_receipt(
                    injection.target_ids,
                    agents=agents_dict,
                    decided_ids=decided_ids,
                    interrupted_ids=interrupted_ids,
                )
            serialized_events.append(payload)
        displaced_ids = {aid for injection in injections for aid in injection.displaced_ids}
        agent_records_now = self._agent_records(agents_dict, displaced_ids=displaced_ids)
        # The way each mover walked, by initiator; it belongs to the beat his MOVE ended.
        routes = {
            exec_state.initiator_id: view
            for exec_state in [*settling, *pending]
            if (view := arrival_view(exec_state, self._environment)) is not None
        }

        # Multi-step executions reach observers as three beats: opening → in progress → result.
        # Without the middle one there are hours of silence and both ends restate the purpose.
        # Dedupe: prefer agent_records when an agent finished and immediately started anew.
        agent_ids_in_records = {r["agent_id"] for r in agent_records}
        ongoing_records = [
            r for r in tick_records
            if r.get("phase") in ("ongoing_complete", "ongoing_tick")
            and r.get("agent_id") not in agent_ids_in_records
        ]
        # Interrupt records (phase="interrupt") are display only; restore doesn't read them. An
        # interrupted agent may also have a new agent_record; execution_ids differ and the step
        # record folds by execution_id, so there's no clash.
        #
        # Failed adjudications are dropped from the display stream: nothing happened in the
        # world. Don't add a frontend flag instead: the display only knows real failure /
        # foiled, and would draw it as a real defeat.
        observable_records = (
            agent_records + ongoing_records + interrupt_records + seize_records
            + self._processor.take_forced_records()
        )
        all_action_records = [r for r in observable_records if not r.get("adjudication_failed")]
        # The executor logged the cause; this logs the consequence, connecting "the world looks
        # frozen" back to failing judge calls. One count per step, only when non-empty: provider
        # jitter comes in clusters.
        if dropped := [r for r in observable_records if r.get("adjudication_failed")]:
            logger.warning(
                "null_steps_withheld_from_observer",
                extra={
                    "world_id": self._world_id,
                    "step": step,
                    "dropped": len(dropped),
                    "of_actions": len(observable_records),
                    "agent_ids": [r.get("agent_id") for r in dropped],
                    "action_types": sorted({str(r.get("action_type")) for r in dropped}),
                },
            )

        # The step's record: each event once, in emission order, stamped with its seq. Message
        # receivers are placed where they perceived from this step (the delivery reached them
        # there, before anyone moved).
        perceived_at = {aid: spatial.location_id for aid, spatial in agent_spatials.items()}
        events = self._event_seqs.order_events(
            messages=message_events(
                deliveries,
                where={
                    aid: perceived_at.get(aid) or self._environment.get_body_location(aid)
                    for aid in deliveries.inboxes
                },
            ),
            broadcasts=broadcast_events(
                broadcasts_this_step,
                where={aid: perceived_at[aid] for aid in agent_signals if aid in perceived_at},
            ),
            actions=action_events(all_action_records, routes=routes),
            injections=injection_events(serialized_events),
        )

        _t_snapshot = time.perf_counter()
        # A step commits whole or not at all: a snapshot missing a part would be the newest commit
        # point, and restore would resume from it. Unbuilt, the previous snapshot stays the commit
        # point and the next step's snapshot carries everything forward.
        snapshot: WorldSnapshot | None = None
        try:
            snapshot = await self._step_snapshot(
                step=step,
                world_time=world_time,
                agents=agent_records_now,
                live_agents=agents_dict,
                events=events,
                schedule=plan.as_dict(),
            )
        except Exception as exc:  # noqa: BLE001 — Rule 6: log error, don't kill run
            logger.error(
                "snapshot_build_failed",
                extra={"world_id": self._world_id, "step": step, "error": str(exc)},
            )
        if snapshot is not None:
            try:
                await self._snapshot_provider.save(self._world_id, step, snapshot)
            except Exception as exc:  # noqa: BLE001 — Rule 6: log error, don't kill run
                logger.error(
                    "snapshot_save_failed",
                    extra={"world_id": self._world_id, "step": step, "error": str(exc)},
                )
            # Live observers get the very record that was saved, so live and replay read one shape.
            self._event_bus.publish({
                "type": "step", "world_id": self._world_id, "step": step, "snapshot": snapshot,
            })
        snapshot_ms = round((time.perf_counter() - _t_snapshot) * 1000, 1)

        tick_narratives = [r["outcome"] for r in tick_records]
        step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0
        # Per-phase wall-clock, shared by the step_complete log and the persisted StepTrace.
        phase_ms = {
            "injection": injection_ms,
            "message": message_ms,
            "pressure": pressure_ms,
            "perceive": perceive_ms,
            "interrupt": interrupt_ms,
            "executor": executor_ms,
            "plan": round(plan_elapsed_ms, 1),
            "exec": round(exec_elapsed_ms, 1),
            "cognition": maintain_ms,
            "snapshot": snapshot_ms,
        }
        logger.info(
            "step_complete",
            extra={
                "world_id": self._world_id,
                "step": step,
                "agent_count": len(agents),
                "elapsed_ms": round(step_elapsed_ms, 2),
                "scheduled_agent_count": len(plan.ordered_agents()),
                "ongoing_agent_count": len({r["agent_id"] for r in tick_records}),
                "phase_ms": phase_ms,
            },
        )
        # Fire-and-forget the per-step flush (see _trace_flush_tasks).
        if self._trace_sink is not None:
            self._trace_sink.record_step(
                StepTrace(
                    world_id=self._world_id,
                    step=step,
                    world_time=world_time.clock_payload(),
                    wall_ms=round(step_elapsed_ms, 1),
                    phase_ms=phase_ms,
                    timestamp=datetime.now().isoformat(),
                )
            )
            flush_task = asyncio.create_task(self._trace_sink.flush(self._world_id, step))
            self._trace_flush_tasks.add(flush_task)
            flush_task.add_done_callback(self._trace_flush_tasks.discard)
        return RuntimeStepResult(
            step=step,
            world_time=world_time.iso_label(),
            # Null steps stay out (see the filter above); filtered again because agent_records
            # is unfiltered.
            action_summaries=[
                record["outcome"] for record in agent_records
                if not record.get("adjudication_failed")
            ] + tick_narratives,
            injection_summaries=[event["narrative_desc"] for event in serialized_events],
            delivered_message_count=len(deliveries.delivered_messages),
            scheduled_agent_ids=[agent.agent_id for agent in plan.ordered_agents()],
        )

    async def _step_snapshot(
        self,
        *,
        step: int,
        world_time: WorldTime,
        agents: Dict[str, Dict[str, Any]],
        live_agents: Dict[str, Agent],
        events: List[Dict[str, Any]],
        schedule: Dict[str, Any],
    ) -> WorldSnapshot:
        """The committed step's record, from what each subsystem holds at the commit point."""
        environment_state = self._environment.world_state()
        relations = {
            f"{relation.from_id}->{relation.to_id}": relation_to_snapshot_dict(relation)
            for agent in live_agents.values()
            for relation in await agent.relation_system.all_relations()
        }
        return WorldSnapshot(
            world_id=self._world_id,
            step=step,
            timestamp=datetime.now(),
            world_time=world_time.clock_payload(),
            state=WorldState(
                agents=agents,
                npcs=environment_state["npcs"],
                entities=environment_state["entities"],
                relations=relations,
            ),
            events=events,
            engine=EngineState(
                pending_messages=await self._message_system.peek_pending(),
                # Broadcasts not yet due. Death notices (deliver_step = next step) straddle this
                # boundary; unsaved, nobody would know of the death after a restore.
                pending_broadcasts=self._broadcast_channel.peek_pending(),
                # Next seq to assign; restore (NarrativeApplication._build_runtime →
                # start_event_seq) resumes strictly increasing.
                event_seq=self._event_seqs.next_seq,
                active_executions=self._active_execution_payloads(step),
                environment=self._environment.engine_state(),
                schedule=schedule,
            ),
            actor_thread=self._actor.take_thread(),
        )

    def _active_execution_payloads(self, step: int) -> List[Dict[str, Any]]:
        """The acts under way, as the snapshot keeps them. One that can't be saved is left out
        and logged (Rule 6) rather than costing the whole snapshot; a restore leaves its
        participants idle."""
        payloads: List[Dict[str, Any]] = []
        for execution in self._executor_registry.all_active():
            try:
                payloads.append(execution.to_payload())
            except TypeError as exc:
                logger.error(
                    "active_execution_unsaveable",
                    extra={
                        "world_id": self._world_id, "step": step,
                        "execution_id": execution.execution_id, "error": str(exc),
                    },
                )
        return payloads

    def _agent_records(
        self,
        agents_dict: Dict[str, "Agent"],
        *,
        displaced_ids: "set[str]",
    ) -> Dict[str, Dict[str, Any]]:
        """Every agent at the end of the step, for ``WorldState.agents``.

        ``snapshot_agent_state`` is what the agent knows of itself; this adds what only the world
        knows, which agent/ must never see: where the body is (a waypoint room, or IN_TRANSIT
        between two; the agent's own ``current_location`` changes only on arrival), the route of
        an execution still under way, and whether the director moved it this step.
        """
        records: Dict[str, Dict[str, Any]] = {}
        for agent_id, agent in agents_dict.items():
            record = snapshot_agent_state(agent)
            body = self._environment.get_body_location(agent_id)
            # The dead are taken off the map (UNPLACED); the body lies where he fell, which is his
            # own last location.
            if body != UNPLACED:
                record["location_id"] = body
            records[agent_id] = record
        # In-flight movers get {from,to,path,arrivals,elapsed,total} so a renderer walks them
        # along the route instead of teleporting. transit_view (movement executor) owns the keys.
        # Whoever he carries rides the same trip, but only while still beside him: one let go
        # on the way is no longer on it.
        for exec_state in self._executor_registry.all_active():
            view = transit_view(exec_state)
            if view is None:
                continue
            here = self._environment.get_body_location(exec_state.initiator_id)
            for agent_id in exec_state.participant_ids or [exec_state.initiator_id]:
                if agent_id != exec_state.initiator_id and (
                    self._environment.get_body_location(agent_id) != here
                ):
                    continue
                if (record := records.get(agent_id)) is not None:
                    record["transit"] = view
        # Moved by the director this step: transit's counterpart, "he didn't walk at all".
        for agent_id in displaced_ids:
            if (record := records.get(agent_id)) is not None:
                record["displaced"] = True
        return records

    def _sync_dead_ids(self, agents: "dict[str, Agent]") -> None:
        """After deaths, push the new set of dead to every agent.

        Perception (the source of that cache) runs before vitality decay and the execution
        phase, so people who died this step aren't in it; planning and feedback both read it for
        relation rendering, and without the push someone just dead would render as alive.
        """
        dead = frozenset(aid for aid, agent in agents.items() if not agent.is_active)
        for agent in agents.values():
            agent.note_deaths(dead)

    async def _perceive_all_agents(
        self,
        agents: Dict[str, "Agent"],
        deliveries: "MessageDelivery",
        agent_spatials: "dict[str, SpatialPerception]",
        broadcasts: "List[Broadcast]",
        step: int,
    ) -> None:
        """Run perception memorization for every active agent (phase order: see run_step)."""
        # Cached into each agent via perceive_step; read by plan-time relation "已死亡" marks and
        # feedback-time memory relation context.
        dead_ids = frozenset(aid for aid, a in agents.items() if not a.is_active)

        async def _perceive_one(
            agent: "Agent",
            spatial: "SpatialPerception",
            inbox: "Any",
            agent_broadcasts: "List[Broadcast]",
        ) -> None:
            # PERCEPTION + agent_id for this task; the memory-importance LLM inside
            # perceive_step overrides the stage to MEMORY (keeping agent_id).
            with observe_stage(Stage.PERCEPTION, agent_id=agent.agent_id):
                await agent.perceive_step(
                    spatial=spatial,
                    inbox=inbox,
                    broadcasts=agent_broadcasts,
                    step=step,
                    dead_ids=dead_ids,
                )

        perceivers: list[str] = []
        tasks = []
        for agent in agents.values():
            if not agent.is_active:
                continue
            spatial = agent_spatials.get(agent.agent_id)
            if spatial is None:
                continue
            agent_broadcasts = BroadcastChannel.for_location(broadcasts, spatial.location_id)
            inbox = deliveries.inbox_for(agent.agent_id)
            # External pressure isn't persisted in perception (it's a live signal only; plan_step
            # consumes pending_external_goals).
            perceivers.append(agent.agent_id)
            tasks.append(_perceive_one(agent, spatial, inbox, agent_broadcasts))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for agent_id, result in zip(perceivers, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "agent_perception_failed",
                    extra={"agent_id": agent_id, "step": step, "error": str(result)},
                )

    def _record_step_happenings(
        self,
        *,
        agent_records: list[dict],
        tick_records: list[dict],
        step: int,
    ) -> None:
        """Record the full strings of this step's actions into each place's recent happenings,
        readable only by covert adjudication.

        Same records as ``_carry_step_observations`` (degraded strings for onlookers); a
        separate pass because the selection differs.

        Two gates; drop either and it leaks:
        - ``COVERTABLE_ACTION_TYPES`` (coarse whitelist): new types are excluded by default.
        - ``ActionResult.happening`` (executor-authorized): which beat carries the layer; a
          talk's complete beat carries the transcript, its interrupt beat the interrupter's
          private thought.

        Deliberately excluded:
        - Interrupted actions: the outcome holds the interrupter's private thought
          (``format_interrupt_reason_3p``), and an interrupted talk has no transcript.
        - NPC doings (``note_npc_outcome``): they don't reach onlookers' ambient either, and
          someone hiding shouldn't get what the rest of the world can't.
        - Entity state changes: ``EntityStateChange`` has only the onlookers' ``perception``,
          no fuller narration to hand over.

        So covert acts see only same-place completed TALK / WORK / ERRAND and often come back
        empty. That's the mechanism's boundary, not a defect; paying off more often needs new
        features, not wiring in the exclusions above.

        Records are merged by (place, full string) with holders unioned, then written once:
        listeners attach only to the initiator's record (``SocialExecutor`` listener_effects), so
        per-record writes would store a duplicate missing the listeners, and a listener's covert
        act would bring back what he'd already heard.
        """
        merged: dict[tuple[str, str], set[str]] = {}
        for r in agent_records + tick_records:
            if r.get("action_type") not in COVERTABLE_ACTION_TYPES:
                continue
            if r.get("adjudication_failed"):   # null step: nothing happened in the world
                continue
            doer_id = r.get("agent_id")
            # Fine-grained gate: read the executor's authorization, don't infer it. Empty = this
            # beat has no such layer (progress beats are content-free, interrupt beats private).
            happening = r.get("happening")
            # The place comes from the record (where this happened), not from the observation
            # (where onlookers saw it). Exclude both pseudo-places: IN_TRANSIT would be treated
            # as one shared "room" by travelers, and UNPLACED isn't a place at all.
            loc = r.get("location_id")
            if not doer_id or not happening or not loc or loc in (IN_TRANSIT, UNPLACED):
                continue
            # Listeners, like participants, already hold this: it isn't new to them, and handing it
            # over would let the adjudication rule success on something they already knew. They're
            # outside participant_ids for turn-economy reasons, not cognitive ones.
            merged.setdefault((loc, happening), set()).update(
                r.get("participant_ids") or [doer_id], r.get("overheard_by") or (),
            )

        for (loc_id, happening), holders in merged.items():
            self._environment.record_happening(
                location_id=loc_id, outcome=happening, step=step, doer_ids=tuple(holders),
            )

    def _carry_step_observations(
        self,
        *,
        agent_records: list[dict],
        tick_records: list[dict],
    ) -> None:
        """Carry this step's action outcomes forward as traces in the next step's ambient_events.

        ``_record_step_happenings`` keeps the full strings of the same records for covert
        adjudication. Events aren't carried: they were already perceived this step via
        broadcast/message. Each onlooker's own cognition interprets the carried outcomes;
        COVERT outcomes go through the private channel.
        """
        # Onlookers get the record's observation (degraded), never outcome (privileged:
        # transcripts, interrupted purpose). Empty observation → nothing to perceive
        # (SEND_MESSAGE, an unexposed covert act).
        # Carry attaches every execution member and spatial_for self-excludes them;
        # participant_ids defaults to [agent_id].
        location_outcomes: dict[str, list[tuple[tuple[str, ...], str, float | None]]] = defaultdict(list)
        # Each participant of a joint action produces a record with the same observation; dedupe
        # so onlookers perceive and remember it once.
        seen_carries: set[tuple[str, tuple[str, ...], str]] = set()
        # Don't filter by action_type: executors authorize observations per record (unexposed
        # COVERT and SEND_MESSAGE declare none); a type filter would exclude COVERT entirely.
        for r in agent_records + tick_records:
            # Null step: nothing happened in the world.
            if r.get("adjudication_failed"):
                continue
            doer_id = r.get("agent_id")
            if not doer_id:
                continue
            # The person acted on is self-excluded like execution members: he already got his own
            # memory from the effect (see completion_record's acted_upon). That's also why he's in
            # this onlooker memory's related_agents: the event is about him.
            doer_ids = tuple(dict.fromkeys(
                [*(r.get("participant_ids") or [doer_id]), *(r.get("acted_upon") or ())]
            ))
            for o in r.get("observations") or []:
                place, text, strength = o.get("location_id"), o.get("text"), o.get("strength")
                # IN_TRANSIT isn't a place to watch from (travelers would share one "room").
                # Check the landing place, not the record: a multi-hop move's first-beat record is
                # IN_TRANSIT, but its "the origin watches him go" must still land.
                if not place or not text or place == IN_TRANSIT:
                    continue
                key = (place, doer_ids, text)
                if key in seen_carries:
                    continue
                seen_carries.add(key)
                location_outcomes[place].append((doer_ids, text, strength))

        for loc_id, entries in location_outcomes.items():
            for doer_ids, observation, strength in entries:
                self._environment.record_carry_observation(
                    location_id=loc_id,
                    observation=observation,
                    strength=strength,
                    doer_ids=doer_ids,
                )

    async def _plan_one_agent(
        self,
        agent: "Agent",
        phase: str,
        deliveries: MessageDelivery,
        step: int,
        world_time_label: str,
        broadcasts: "List[Broadcast]",
        spatial: "SpatialPerception | None" = None,
    ) -> tuple[str, "Agent", AgentStepPlan]:
        """Build perception inputs for one agent and run its plan_step.

        broadcasts are the ones collect dequeued this step (the channel is already empty and
        can't be queried again), filtered by the agent's location."""
        agent_messages = deliveries.inbox_for(agent.agent_id)
        if spatial is None:
            spatial = self._environment.spatial_for(
                agent_id=agent.agent_id,
                step=step,
                world_time=world_time_label,
            )
        local_broadcasts = BroadcastChannel.for_location(broadcasts, spatial.location_id)
        plan_result = await agent.plan_step(
            step=step,
            spatial=spatial,
            inbox=agent_messages,
            broadcasts=local_broadcasts,
        )
        return (phase, agent, plan_result)

    async def _apply_world_pressure(
        self,
        agents: Dict[str, "Agent"],
        *,
        signals: "Mapping[str, PerceivedSignals]",
        agent_spatials: "dict[str, SpatialPerception]",
        world_time_label: str,
        step: int,
    ) -> float:
        """Evaluate what the world presses on each agent this step and land it; returns the
        phase's wall-clock ms.

        The only writer of ``pending_external_goals``; it rewrites the field for every agent
        each step:
        - The evaluator omits agents with no acute signal, so landing only returned entries
          would pin a quiet agent's last pressure forever.
        - ``plan_step`` must not self-clear it: agents the cadence gate skips never reach it,
          and would later decide under long-gone pressure.

        No entry → empty; the field never survives a step boundary.
        """
        started = time.perf_counter()
        if self._pressure_evaluator is None:
            logger.debug(
                "world_pressure_done",
                extra={"step": step, "elapsed_ms": 0, "skipped": True},
            )
            return 0.0

        pressure_map = await self._pressure_evaluator.evaluate(
            agents=agents,
            signals=signals,
            world_time_label=world_time_label,
            agent_spatials=agent_spatials,
        )

        agents_with_pressure = 0
        for agent_id, agent in agents.items():
            goals = pressure_map.get(agent_id, [])
            agent.pending_external_goals = goals
            if goals:
                agents_with_pressure += 1

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.debug(
            "world_pressure_done",
            extra={
                "step": step,
                "elapsed_ms": elapsed_ms,
                "agents_with_pressure": agents_with_pressure,
                "skipped": False,
            },
        )
        return elapsed_ms

    def _idle_body_plan(
        self,
        agent: "Agent",
        *,
        status: DecisionStatus,
        deliveries: MessageDelivery,
        step: int,
        world_time_label: str,
        agent_spatials: "dict[str, SpatialPerception] | None",
    ) -> AgentStepPlan:
        """Plan entry for an agent with no decision of its own this step: planning crashed
        (FAILED) or the cadence gate didn't ask (NOT_SCHEDULED). The arbiter's
        ``action is None`` branch skips it.

        The entry is mandatory: ``ExecutionArbiter.arbitrate`` treats a co-participant absent
        from ``planned_steps`` as 「毫无回应」 and rejects the conscripting agent's action.

        So ``spatial`` must be real perception (passive join reads
        ``plan.spatial.visible_agent_ids``) and ``inbox`` the real inbox (it becomes
        ``message_ids``). ``need_evaluation`` may be empty: only the start path
        (``begin_ongoing_step``) reads it; ``join_ongoing_action`` doesn't, and feedback reads
        the dominant need from persisted state.
        """
        spatial = agent_spatials.get(agent.agent_id) if agent_spatials else None
        if spatial is None:
            spatial = self._environment.spatial_for(
                agent_id=agent.agent_id, step=step, world_time=world_time_label,
            )
        return AgentStepPlan(
            agent_id=agent.agent_id,
            step=step,
            spatial=spatial,
            inbox=deliveries.inbox_for(agent.agent_id),
            broadcasts=[],
            need_evaluation=NeedEvaluation(
                dominant_need=None, scores={}, active_needs=[],
                short_term_goals=[], long_term_goals=[],
                prompt_context="",
            ),
            action=None,
            decision_status=status,
        )

    async def _plan_execution(
        self,
        *,
        plan: StepExecutionPlan,
        agents: List["Agent"],
        in_progress_at_step_start: set[str],
        deliveries: MessageDelivery,
        step: int,
        world_time_label: str,
        broadcasts: "List[Broadcast]",
        agent_spatials: "dict[str, SpatialPerception] | None" = None,
    ) -> List[tuple[str, Agent, AgentStepPlan]]:
        # Planning is parallel across the whole scheduled set: plan_step writes nothing shared.
        # Main-before-background is an arbitration property, not a planning barrier: gather
        # preserves input order, so concatenating batches keeps initiative order. Provider load
        # is bounded by LLMRouter's global semaphore.
        ordered = [
            (batch.phase, agent)
            for batch in plan.batches
            for agent in batch.agents
        ]
        # These are the agents the cadence gate admitted; record it before the LLM runs, so
        # the starvation clock advances even if this agent's planning then crashes (a crashed
        # decision is still a decision we paid for — retrying it every step would be worse).
        for _phase, agent in ordered:
            agent.personality.mark_decided(step)

        results = await asyncio.gather(
            *[
                self._plan_one_agent(
                    agent=agent,
                    phase=phase,
                    deliveries=deliveries,
                    step=step,
                    world_time_label=world_time_label,
                    broadcasts=broadcasts,
                    spatial=agent_spatials.get(agent.agent_id) if agent_spatials else None,
                )
                for phase, agent in ordered
            ],
            return_exceptions=True,
        )
        # Rule 4 per slot. A failed slot becomes a FAILED idle plan (action=None, tier-1 no-op),
        # kept so the agent can still be conscripted this step.
        planned_steps: List[tuple[str, "Agent", AgentStepPlan]] = []
        for (phase, agent), result in zip(ordered, results):
            # A withdrawn step ends here, not just this agent's plan.
            if isinstance(result, StepWithdrawn):
                raise result
            if isinstance(result, BaseException):
                logger.warning(
                    "agent_plan_failed",
                    extra={"agent_id": agent.agent_id, "step": step, "error": str(result)},
                )
                planned_steps.append((phase, agent, self._idle_body_plan(
                    agent,
                    status=DecisionStatus.FAILED,
                    deliveries=deliveries,
                    step=step,
                    world_time_label=world_time_label,
                    agent_spatials=agent_spatials,
                )))
                continue
            planned_steps.append(result)

        # Agents the cadence gate didn't ask: zero LLM calls, but conscriptable bodies (see
        # _idle_body_plan), so being pulled into a conversation doesn't first burn a discarded
        # decision. Appended after the deciders to keep initiative order; idle bodies never
        # initiate.
        #
        # Mid-action bodies need entries too: without one arbitration rules them "毫无回应" (no
        # response), and ``Conscription.COMPEL``, whose point is taking busy people, would only
        # work on idle ones.
        scheduled = {agent.agent_id for _phase, agent in ordered}
        for agent in agents:
            if not agent.is_active:
                continue
            if agent.agent_id in scheduled:
                continue
            phase = "main" if agent.is_main_character else "background"
            planned_steps.append((phase, agent, self._idle_body_plan(
                agent,
                status=DecisionStatus.NOT_SCHEDULED,
                deliveries=deliveries,
                step=step,
                world_time_label=world_time_label,
                agent_spatials=agent_spatials,
            )))
        return planned_steps

    def _stamp_verdicts(
        self,
        planned_steps: List[tuple[str, "Agent", AgentStepPlan]],
        arbitration: dict[str, ArbitratedAction],
    ) -> None:
        """Stamp the arbitration verdict on every body that decided this step (see the
        ``_commit_execution`` call site).

        Observability only, through the trace channel: cognitive outputs don't carry an extra
        field just for review.
        """
        if self._trace_sink is None:
            return
        for _phase, _agent, plan in planned_steps:
            if plan.action is None:
                continue  # no intent: not scheduled / chose not to act / decision failed
            resolved = arbitration.get(plan.agent_id)
            exec_state = self._executor_registry.get_active_for_agent(plan.agent_id)
            # Read the carried completion result: ``create_failed``, the single funnel for
            # arbitration vetoes and precondition failures, sets ``not_executed``. Not the
            # first-beat marker: an infeasible action never has that beat, so it would always
            # read as executed.
            carried = exec_state.extra.get("completed_result") if exec_state is not None else None
            fields: dict[str, Any] = {}
            if resolved is None:
                # ``executor.start()`` raised in ``_enact`` (logged ``executor_start_failed``):
                # the machine breaking, not the world's verdict, so it's marked apart.
                fields["verdict"] = "start_failed"
            elif resolved.is_passive_join:
                fields["verdict"] = CONSCRIPTED
            elif resolved.refused:
                # The invitee refused: it happened, as a failure, so it isn't "rejected".
                fields["verdict"] = "refused"
            elif carried is not None and carried.not_executed:
                # Ruled infeasible and settled as a failure this step; the reason is in the
                # result text.
                fields["verdict"] = "rejected"
            else:
                fields["verdict"] = "executed"
                if exec_state is not None and exec_state.remaining_steps > 0:
                    # Only the start is marked; later beats read as the same act continuing, so
                    # review needn't chain steps.
                    fields["spans_steps"] = exec_state.remaining_steps + 1
            self._trace_sink.annotate_recorded_call(
                self._world_id, step=plan.step, agent_id=plan.agent_id,
                stage=Stage.DECISION.value, **fields,
            )

    async def _commit_one_agent(
        self,
        agent: "Agent",
        plan: AgentStepPlan,
        resolved: ArbitratedAction,
    ) -> Dict[str, Any]:
        """Begin one agent's action (IN_PROGRESS, no outcome writeback) and return its
        begin-marker record.

        Every admitted action is an execution: the doer calls begin_ongoing_step, a
        conscripted co-participant join_ongoing_action. Completion and all writeback happen
        later in finalize: same step for a born-zero execution (same-step sweep), else via
        tick. There is no commit_step.
        """
        exec_state = self._executor_registry.get_active_for_agent(plan.agent_id)
        # The record describes what the agent actually does: on passive join, the joined action.
        # His own plan.action was dropped by conscription (kept via defer_decided_intent) and
        # must never leak into the executed record.
        record_action = resolved.action_result.action if resolved.is_passive_join else plan.action
        # Mirror the executor's remaining (start counts as step 1) so
        # personality.action_remaining_steps matches exec_state; born-zero → 0, finalized by the
        # same-step sweep.
        remaining = exec_state.remaining_steps if exec_state else max(0, record_action.estimated_steps - 1)

        if resolved.is_passive_join:
            # His own decided action is discarded; keep the intent for next step's decision (the
            # perception that drove it lasts one step). None if his decision failed. Before the
            # join, so his own act comes to nothing before the one he is drawn into begins.
            if plan.action is not None:
                agent.defer_decided_intent(plan.action, plan.step)
            await agent.join_ongoing_action(
                # His own line (``participant_action_desc``), never the join record's outcome:
                # that names him in the third person and quotes the initiator's purpose.
                description=record_action.action_description,
                target_id=exec_state.initiator_id if exec_state else "",
                activity_status=activity_status_for(exec_state.action_type if exec_state else record_action.action_type),
                estimated_steps=remaining,
                total_steps=exec_state.estimated_steps if exec_state else record_action.estimated_steps,
                step=plan.step,
            )
            # Two trace marks for review (cognitive outputs carry no review-only field):
            #  · Motivation is voided (join_ongoing_action doesn't submit the need evaluation) →
            #    unadopted, so review won't flag a never-queued goal as unpursued.
            #  · Decision survives via defer_decided_intent, so it stays adopted; its conscription
            #    mark exempts it from the decision↔action check. Marked now: a multi-step
            #    interaction's action call happens only on its final step.
            if self._trace_sink is not None:
                self._trace_sink.mark_call_unadopted(
                    self._world_id, step=plan.step, agent_id=plan.agent_id,
                    stage=Stage.MOTIVATION.value, reason=CONSCRIPTED,
                )
        else:
            await agent.begin_ongoing_step(
                plan=plan, estimated_steps=remaining,
                total_steps=exec_state.estimated_steps if exec_state else record_action.estimated_steps,
            )

        # Invariant guard: arbitration gave him an execution the registry lacks. Landing
        # normally would leave him in progress forever, never deciding again, so keep the
        # landing (needs, goals, dropped intent) and reset only the action status.
        if exec_state is None and (resolved.is_passive_join or resolved.ongoing_execution_id is not None):
            logger.error(
                "stranded_participant_released",
                extra={
                    "agent_id": agent.agent_id, "step": plan.step,
                    "execution_id": resolved.ongoing_execution_id or "",
                    "passive_join": resolved.is_passive_join,
                },
            )
            agent.release_action(step=plan.step)

        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.personality.soul.name,
            "is_main_character": agent.is_main_character,
            # The beat, not the scheduler batch (that's schedule metadata). Always the opening
            # beat here, so `outcome` only restates the intent; a born-zero act becomes "settled"
            # when its completion folds in, or is withheld as a null step (see
            # execution_processor.merge_completion_into_records).
            "phase": "begin",
            # Expected duration, the opening beat's own news. Rendered here as natural duration:
            # step counts never cross into the story.
            "duration_label": (
                describe_duration(
                    exec_state.estimated_steps if exec_state else record_action.estimated_steps,
                    self._clock.config.seconds_per_step,
                )
                if remaining > 0 else ""
            ),
            # Progress, for the same bar the middle beat draws — see execution_processor. On the
            # opening beat it is the first stroke of it.
            "elapsed_steps": (exec_state.estimated_steps - remaining) if exec_state else 0,
            # TALK's stated topic: everyone in it knows it from the start (see AgentAction).
            "talk_topic": exec_state.talk_topic if exec_state else "",
            "total_steps": exec_state.estimated_steps if exec_state else 0,
            # Lets the same-step sweep fold the completion into this record in place.
            "execution_id": resolved.ongoing_execution_id,
            "location_id": resolved.location_id,
            "visible_agent_ids": list(resolved.visible_agent_ids),
            "message_ids": list(resolved.message_ids),
            # ActionType | str; unwrap. A live enum would read "ActionType.TALK" on the live path
            # (the disk path launders it and hides the mistake).
            "action_type": str(getattr(record_action.action_type, "value", record_action.action_type or "")),
            # All execution members, so carry self-excludes every participant; a passive-join
            # record is under the conscript's id but describes the initiator's action.
            "participant_ids": list(exec_state.participant_ids) if exec_state else [agent.agent_id],
            # See execution_processor.completion_record — who started this execution.
            "initiator_id": exec_state.initiator_id if exec_state else agent.agent_id,
            # Render-neutral semantic membrane (deed / target kind / affected entities)
            # so the 2D renderer can differentiate effects (e.g. take a thing vs smash it)
            # — facts only, never render instructions.
            **action_semantics(record_action, resolved.action_result),
            # Deed and result stay two fields: the read model renders the prose as the card body
            # and the outcome under it. Don't glue them into one digest: the feed would print its
            # scaffolding and repeat the outcome.
            "action_description": record_action.action_description,
            # Why he chose it: the only channel out of the sim for the deliberation. Rides this
            # record because the decision happened on this beat. Empty for a conscript:
            # ``record_action`` is then the initiator's, and would print one man's thought under
            # another's name.
            "inner_monologue": "" if resolved.is_passive_join else record_action.inner_monologue,
            "outcome": resolved.action_result.outcome,
            "gist": resolved.action_result.gist,
            # Degraded onlooker view, one per place (carry reads this); a JSON-native dict, no
            # Python objects.
            "observations": [o.__dict__ for o in resolved.action_result.observations],
            "succeeded": True,
            "failure_reason": "",
            "detected": False,
            "dialogue": [],
        }

    async def _commit_execution(
        self,
        planned_steps: List[tuple[str, "Agent", AgentStepPlan]],
        arbitration: dict[str, ArbitratedAction],
        agents_dict: Dict[str, "Agent"],
    ) -> List[Dict[str, Any]]:
        agent_records: List[Dict[str, Any]] = []
        if not planned_steps:
            return agent_records

        # Commits only mutate their own agent, so each consecutive phase group gathers in
        # parallel while main→bg record order holds. Agents absent from arbitration (decision
        # unavailable) get no commit or record; keying off arbitration membership keeps the two
        # in lockstep even if a skipped agent was conscripted elsewhere.
        groups: List[List[tuple[str, "Agent", AgentStepPlan]]] = []
        current_phase: str | None = None
        for item in planned_steps:
            phase, agent, plan = item
            if plan.agent_id not in arbitration:
                # An intent with no verdict: its execution failed to start (logged by the
                # arbiter). Nothing to commit; the character decides afresh next step.
                if plan.action is not None:
                    agent.drop_unstarted_intent(plan.action, plan.step)
                continue
            if phase != current_phase:
                groups.append([item])
                current_phase = phase
            else:
                groups[-1].append(item)

        # Stamp what the world did with each intent on its decision call, so review needn't
        # infer it from roundabout signals (execution_id, talk_role…) each wrong at some edge.
        # Only decisions with an intent:
        #   executed     became his own action (multi-step ones carry spans_steps from the
        #                execution, not the decision's often-off estimated_steps)
        #   conscripted  folded into someone else's action (motivation voided too)
        #   rejected     ruled infeasible, settled as a failure that step
        #   refused      the one invited refused; a failure that happened, remembered
        #   start_failed admitted but couldn't start: infrastructure failure, not a verdict
        self._stamp_verdicts(planned_steps, arbitration)

        for group in groups:
            group_records = await asyncio.gather(
                *[
                    self._commit_one_agent(
                        agent=agent,
                        plan=plan,
                        resolved=arbitration[plan.agent_id],
                    )
                    for _phase, agent, plan in group
                ],
                return_exceptions=True,
            )
            # Rule 4 per slot: a failed commit (typically OSError on save) drops its record and
            # logs error (Rule 6); the agent replans next step.
            for (_phase, _agent, plan), record in zip(group, group_records):
                if isinstance(record, BaseException):
                    logger.error(
                        "agent_commit_failed",
                        extra={"agent_id": plan.agent_id, "step": plan.step, "error": str(record)},
                    )
                    continue
                agent_records.append(record)

        return agent_records
