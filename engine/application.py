"""Application-facing orchestration for world build and runtime execution."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from config.models import Config
from core.container import Container
from core.context import get_log_context, observe_stage, set_log_context
from core.factory import ComponentKind, ProviderFactory
from core.interfaces.trace import Stage
from core.interfaces.world_config import WorldConfig
from core.logging import get_logger
from core.interfaces.perception import Broadcast
from core.interfaces.snapshot import WorldSnapshot
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTime
from engine.cognition_maintenance import CognitionMaintenance
from engine.director import DirectivePrompt, DirectiveResult, DirectorChannel, NpcOnMenu
from engine.event import EventSettings
from engine.executors import build_default_registry
from engine.message_system import MessageSystem
from engine.run_control import RunController, RunState
from engine.runtime import NarrativeRuntime, RuntimeStepResult
from engine.scheduler import AgentScheduler
from engine.world_pressure import WorldPressureEvaluator
from world import World, WorldBuilder, WorldCatalog
from world.initializer import WorldInitializer

logger = get_logger(__name__)


@dataclass
class RuntimeSession:
    """In-process runtime session for one built world.

    ``controller``/``task`` are cleared when the run loop ends; ``status`` outlives it
    (COMPLETED / FAILED) until the next run.
    """

    world: World
    runtime: NarrativeRuntime
    status: RunState = RunState.IDLE
    controller: RunController | None = field(default=None)
    task: "asyncio.Task[None] | None" = field(default=None)


class NarrativeApplication:
    """Own world build and runtime orchestration outside the interaction layer."""

    def __init__(
        self,
        container: Container,
        config: Config,
        *,
        catalog: WorldCatalog | None = None,
    ) -> None:
        self._container = container
        self._config = config
        self._builder = WorldBuilder(container)
        self._sessions: dict[str, RuntimeSession] = {}
        # When set, build_world registers each world. None in tests, so builds never touch disk.
        self._catalog = catalog
        # Held only so no in-flight flush is garbage-collected mid-write (see _flush_trace_async).
        self._trace_flush_tasks: set["asyncio.Task[None]"] = set()
        # One lifecycle lock per world: restore, reset and delete replace _sessions[world_id]
        # across several awaits, and an overwritten session may still own an unreachable run loop.
        # Don't remove the lock in delete: setdefault would create a new one, and waiters on the
        # old and new locks would proceed together. Cost: one small lock per world ever touched.
        self._lifecycle_locks: dict[str, "asyncio.Lock"] = {}
        # Set by shutdown: no run may start once the process is going down.
        self._shutting_down = False

    def _world_config_for(self, template: str) -> WorldConfig:
        """The ``WorldConfig`` for one named map.

        The deployment supplies only the kind (``world.config``); clock, calendar and locations
        come off the map, chosen per world.
        """
        return ProviderFactory.create(
            self._config.world.config,
            kind=ComponentKind.WORLD,
            template=template,
        )

    def _candidates(self, template: str | None, available: Sequence[str]) -> list[WorldConfig]:
        """The maps this build may choose between.

        A named ``template`` is the only candidate; otherwise ``TemplateSelector`` picks from all
        available, and raises on none (Rule 3). The names come from the interaction layer: the
        engine must not learn that a map is a directory on disk.
        """
        if template is not None:
            return [self._world_config_for(template)]
        return [self._world_config_for(name) for name in available]

    def has_session(self, world_id: str) -> bool:
        return world_id in self._sessions

    def get_session(self, world_id: str) -> RuntimeSession:
        try:
            return self._sessions[world_id]
        except KeyError as exc:
            raise ValueError(f"World {world_id!r} has no active session.") from exc

    def world_time(self, world_id: str) -> WorldTime:
        """The live session's current moment."""
        return self.get_session(world_id).runtime.clock.current

    async def build_world(
        self,
        theme: str,
        *,
        template: str | None = None,
        available_templates: Sequence[str] = (),
    ) -> World:
        """Build a world and register its runtime session.

        The map is chosen per world: a named ``template``, else picked from
        ``available_templates`` by the theme. With neither the build fails; a deployment-wide
        default map would hide that. The chosen map is persisted with the world (see
        ``TiledWorldConfig.to_runtime_context``).
        """

        world = await self._builder.build(
            theme=theme,
            world_configs=self._candidates(template, available_templates),
            min_agents=self._config.world.min_agents,
            max_agents=self._config.world.max_agents,
            max_npcs=self._config.engine.max_npcs,
        )
        # Flush the build segment (world-init LLM traces carry no step → build.jsonl).
        await self._container.trace_sink.flush(world.world_id)
        self._sessions[world.world_id] = RuntimeSession(
            world=world,
            runtime=self._build_runtime(world),
        )
        if self._catalog is not None:
            self._catalog.register(
                world.world_id,
                theme=theme,
                world_name=world.analysis.world_name,
                created_at=datetime.now(),
            )
        logger.info(
            "application_world_registered",
            extra={
                "world_id": world.world_id,
                "world_name": world.analysis.world_name,
                "agent_count": len(world.agents),
            },
        )
        return world

    def _lifecycle_lock(self, world_id: str) -> "asyncio.Lock":
        """This world's lifecycle lock (see ``_lifecycle_locks``)."""
        return self._lifecycle_locks.setdefault(world_id, asyncio.Lock())

    def _refuse_while_mutating(self, world_id: str) -> None:
        """Refuse to start a run while this world's lifecycle is being changed.

        A sync check, not an await: ``start_run`` must stay sync, and holding the lock for a
        whole run would block reset/delete forever. Callers run atomically to their first
        await, so reading ``locked()`` closes the window in which a run would start on a session
        being replaced and keep advancing a runtime nobody can reach.
        """
        lock = self._lifecycle_locks.get(world_id)
        if lock is not None and lock.locked():
            raise ValueError(
                f"World {world_id!r} is being restored, reset or deleted — cannot start a run."
            )

    @staticmethod
    def _run_is_live(session: RuntimeSession) -> bool:
        """Whether a run loop is advancing this session: a background ``task`` or a bare
        ``runtime.run`` await (``run_world``)."""
        if session.task is not None and not session.task.done():
            return True
        return session.runtime.is_running

    async def restore_session(self, world_id: str) -> World:
        """Restore a previously built world from storage and register its session.

        Runs under the lifecycle lock, re-checking inside: assembly spans several awaits, and
        two concurrent restores would let the later overwrite the earlier, opening the way for
        two run loops on one world. An already restored session is reused.
        """

        async with self._lifecycle_lock(world_id):
            existing = self._sessions.get(world_id)
            if existing is not None:
                return existing.world
            return await self._restore_session_locked(world_id)

    async def _restore_session_locked(self, world_id: str) -> World:
        initializer = WorldInitializer(self._container, max_npcs=self._config.engine.max_npcs)
        world = await initializer.restore(world_id)
        steps = await self._container.snapshot.list_steps(world_id)
        latest_step = max((s for s in steps if s > 0), default=0)
        fired_events = await self._collect_fired_events(world_id, steps)
        # Shared by in-flight messages, pending broadcasts and the event-seq counter.
        # Don't skip step 0: the `s > 0` filter above only sets the clock, and a never-run
        # world's step-0 snapshot still carries pending_messages (same rule in
        # WorldInitializer._restore_environment_state). Its metadata has no event_seq.
        latest = await self._container.snapshot.load(world_id, latest_step)
        await self._restore_pending_messages(world_id, latest)
        # Keeps the event seq strictly increasing across restart (engine.event_seq.EventSequencer).
        start_event_seq = _event_seq_of(latest)
        self._sessions[world_id] = RuntimeSession(
            world=world,
            runtime=self._build_runtime(
                world,
                start_step=latest_step,
                fired_events=fired_events,
                start_event_seq=start_event_seq,
                pending_broadcasts=latest.pending_broadcasts if latest else None,
            ),
        )
        logger.info(
            "application_world_session_restored",
            extra={
                "world_id": world_id,
                "world_name": world.analysis.world_name,
                "agent_count": len(world.agents),
            },
        )
        return world

    async def reset_session(self, world_id: str) -> World:
        """Reset a world to its initial state (step 0) and register a fresh session.

        Rolls every store back to its step-0 baseline first, then rebuilds through the canonical
        ``restore`` path, so the agents carry baseline state (not step-N minds on a step-0
        clock). Memory keeps its backstory seeds and traces keep the build segment: only what
        the *run* produced is purged.

        Runs under the lifecycle lock, re-checking inside that no loop is running: the
        rollback spans many awaits, and a loop started meanwhile would be orphaned.
        """
        async with self._lifecycle_lock(world_id):
            return await self._reset_session_locked(world_id)

    async def _reset_session_locked(self, world_id: str) -> World:
        existing = self._sessions.get(world_id)
        if existing is not None and self._run_is_live(existing):
            raise ValueError(f"World {world_id!r} is running — stop the run before resetting.")

        initializer = WorldInitializer(self._container, max_npcs=self._config.engine.max_npcs)
        agent_store = self._container.agent_store
        snapshot_store = self._container.snapshot

        # Clear relations first, or runtime-created pairs survive the reset.
        await agent_store.clear_relations(world_id)
        for agent_id in await agent_store.list_agent_ids(world_id):
            initial_state = await agent_store.load_initial_agent_state(world_id, agent_id)
            if initial_state is not None:
                await agent_store.save_agent_state(world_id, agent_id, initial_state)
            for rel in await agent_store.load_all_initial_relations(world_id, agent_id):
                await agent_store.save_relation(rel)

        # Drop in-flight messages and post-step-0 snapshots so restore re-seeds the
        # environment from step 0 and no stale message is delivered after the reset.
        await self._container.message_provider.clear(world_id)
        await snapshot_store.delete_steps_after(world_id, 0)

        # Traces rewind with the clock (see TraceSink.delete_run_traces).
        self._container.trace_sink.delete_run_traces(world_id)

        world = await initializer.restore(world_id)

        # restore() warmed from the still-polluted collection, hence the re-warm after the purge.
        for agent in world.agents.values():
            await agent.memory_system.purge_runtime_memories()
            await agent.memory_system.warm_recent_memories(current_step=0)

        # Also stops in-flight event generation, written for the step-N world.
        if existing is not None:
            await existing.runtime.aclose()

        self._sessions[world_id] = RuntimeSession(
            world=world,
            runtime=self._build_runtime(world, start_step=0),
        )
        logger.info(
            "world_session_reset",
            extra={"world_id": world_id, "world_name": world.analysis.world_name},
        )
        return world

    async def run_world(self, world_id: str, *, steps: int) -> list[RuntimeStepResult]:
        """Advance a previously built world session and wait for it: the tests' driver (the API
        runs worlds in the background through ``start_run``).

        ``NarrativeRuntime.run`` rejects reentrancy itself, but its latch is per-runtime and
        can't see a reset/delete swapping the session, hence the lifecycle gate here.
        """

        self._refuse_while_mutating(world_id)
        session = self.get_session(world_id)
        return await session.runtime.run(session.world.agent_list(), total_steps=steps)

    # ------------------------------------------------------------------
    # Background run control (API-facing): run / pause / resume / stop.
    # ------------------------------------------------------------------

    def start_run(self, world_id: str, *, steps: int) -> None:
        """Start advancing a world *steps* steps in the background (non-blocking).

        Raises if a run is already active, or the lifecycle is mid-change
        (see ``_refuse_while_mutating``).
        """

        self._refuse_while_mutating(world_id)
        if self._shutting_down:
            raise ValueError("The server is shutting down — cannot start a run.")
        session = self.get_session(world_id)
        if self._run_is_live(session):
            raise ValueError(f"World {world_id!r} is already running.")
        controller = RunController()
        session.controller = controller
        self._set_status(world_id, session, RunState.RUNNING)
        session.task = asyncio.create_task(
            self._run_loop(world_id, session, controller, steps),
            name=f"run:{world_id}",
        )
        logger.info("world_run_started", extra={"world_id": world_id, "steps": steps})

    async def shutdown(self) -> None:
        """Stop every background run at its next step boundary and wait for them all.

        A step writes memories as it goes and its snapshot only at the end, so a run killed
        mid-step restores to the previous snapshot and replays the step, remembering it twice.
        The wait is unbounded: the process manager's stop timeout is the ceiling.
        """
        self._shutting_down = True
        live = {
            world_id: session.task
            for world_id, session in self._sessions.items()
            if session.task is not None and not session.task.done()
        }
        if not live:
            return
        for world_id in live:
            # None once the loop is past its last step and only finishing up.
            if self._sessions[world_id].controller is not None:
                self.stop_run(world_id)
        logger.info("shutdown_waiting_for_runs", extra={"world_ids": sorted(live)})
        await asyncio.gather(*live.values(), return_exceptions=True)
        logger.info("shutdown_runs_stopped", extra={"world_ids": sorted(live)})

    def _set_status(self, world_id: str, session: RuntimeSession, state: RunState) -> None:
        """Update a session's run state and announce it on the event bus, like step events."""
        session.status = state
        self._container.event_bus.publish(
            {"type": "status", "world_id": world_id, "status": state.value}
        )

    async def _run_loop(
        self,
        world_id: str,
        session: RuntimeSession,
        controller: RunController,
        steps: int,
    ) -> None:
        completed = False
        try:
            await session.runtime.run(
                session.world.agent_list(),
                total_steps=steps,
                controller=controller,
            )
            completed = True
            self._set_status(world_id, session, RunState.COMPLETED)
            logger.info(
                "world_run_finished",
                extra={"world_id": world_id, "stopped": controller.stop_requested},
            )
        except asyncio.CancelledError:
            self._set_status(world_id, session, RunState.COMPLETED)
            raise
        except Exception as exc:  # noqa: BLE001 — surface any run failure as FAILED status
            self._set_status(world_id, session, RunState.FAILED)
            logger.error(
                "world_run_failed",
                extra={"world_id": world_id, "error": str(exc)},
            )
        finally:
            session.controller = None
            session.task = None
        # The director queue is drained only at the start of a step, so a directive accepted
        # during the run's last step (or while it was stopping) would otherwise wait for a run
        # nobody starts. Not after a failed run: a step that keeps failing would loop.
        if (
            completed
            and session.runtime.director.pending_count()
            and self._sessions.get(world_id) is session
        ):
            try:
                self.start_run(world_id, steps=1)
            except ValueError as exc:
                logger.warning(
                    "directive_landing_run_refused", extra={"world_id": world_id, "error": str(exc)}
                )

    # ------------------------------------------------------------------
    # Director surface. The API talks to this, never to `session.runtime.director`, so the
    # web layer doesn't depend on how a runtime is assembled.
    # ------------------------------------------------------------------

    def _director_inputs(self, world_id: str) -> tuple[DirectorChannel, dict[str, Any]]:
        """The channel plus the selection menu every director call is composed against.

        Assembled here because the director channel doesn't hold EnvironmentSystem (same
        boundary as EventSystem), and once, so submit and the developer instruments below never
        show the model two different worlds.
        """
        session = self.get_session(world_id)
        director = session.runtime.director
        environment = session.world.environment
        return director, {
            "all_agents": session.world.agents,
            "npcs": [
                NpcOnMenu(npc=npc, location_id=environment.get_body_location(npc.npc_id))
                for npc in environment.all_npcs()
            ],
            "locations": environment.space.all_places(),
            # Newest first: the director usually wants whatever just appeared. Don't truncate: an
            # author who can't reach something has a broken tool; if it grows to hundreds, filter
            # by the instruction instead. Sorted once: rendering, index mapping and id resolution
            # all read this list.
            "entities": sorted(
                environment.all_live_entities(),
                key=lambda e: (-e.created_step, e.entity_id),
            ),
            "world_time": session.runtime.clock.current,
        }

    def directive_prompt(self, world_id: str, text: str) -> DirectivePrompt:
        """The prompt a directive *would* be parsed with — composed, not sent.

        Developer instrument (the director console): the production prompt, with no LLM call
        and no effect on the world.
        """
        director, menu = self._director_inputs(world_id)
        return director.build_prompt(text, **menu)

    def interpret_directive(
        self, world_id: str, raw_response: str,
    ) -> tuple[DirectiveResult, dict[str, Any] | None]:
        """Run a raw LLM response through production validation — queueing nothing.

        The other half of the director console: shows whether a response would be accepted and
        which refusal path caught it. A parsed plan is described and dropped.
        """
        director, menu = self._director_inputs(world_id)
        result, plan = director.interpret(
            raw_response,
            all_agents=menu["all_agents"],
            npcs=menu["npcs"],
            locations=menu["locations"],
            entities=menu["entities"],
        )
        return result, None if plan is None else director.describe_plan(plan)

    @contextmanager
    def _director_trace_scope(self, world_id: str):
        """Make a director LLM call land in the trace explorer where it belongs.

        These calls run in an HTTP request, outside the step loop, so nothing else sets
        ``world_id`` / step / stage for them. The step is the one the world is parked on: the
        one the directive was composed against.

        - The context is RESTORED, not cleared: ``clear_log_context()`` would blank the
          caller's fields.
        - The flush (a paused world never reaches the per-step flush) goes to a worker thread
          unawaited: a disk write must not sit in the request path.
        """
        session = self.get_session(world_id)
        step = session.runtime.clock.current_step
        previous = get_log_context()
        set_log_context(world_id=world_id, step=str(step))
        try:
            with observe_stage(Stage.DIRECTOR):
                yield
        finally:
            set_log_context(**previous)
            self._flush_trace_async(world_id, step)

    def _flush_trace_async(self, world_id: str, step: int) -> None:
        """Write the trace segment in the background; never wait for it.

        Losing a flush to a crash is acceptable for an audit log. No running loop means no
        flush: skipping an audit write beats raising into the work being audited.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        task = asyncio.create_task(self._container.trace_sink.flush(world_id, step))
        self._trace_flush_tasks.add(task)
        task.add_done_callback(self._trace_flush_tasks.discard)

    async def submit_directive(self, world_id: str, text: str) -> "DirectiveResult":
        """Parse a human's sentence into an intervention, queue it, **and make sure it lands**.

        Parsed here, not in the step loop, so an impossible directive is refused while someone
        is still looking (see DirectorChannel).

        A world that wouldn't otherwise advance is advanced one step, or the user watches a map
        that never moves. **Except while a run is going or stopping**: ``step_world`` would
        re-park it (or overwrite STOPPING), and ``_run_loop`` steps once more for a directive
        that lands after the run's last drain.
        """
        director, menu = self._director_inputs(world_id)
        with self._director_trace_scope(world_id):
            result = await director.submit(text, **menu)
        # Outside the trace scope, or the whole step's cognition would be billed to the director.
        if result.accepted and self.run_status(world_id) not in (RunState.RUNNING, RunState.STOPPING):
            self.step_world(world_id)
        return result

    def step_world(self, world_id: str) -> None:
        """Advance exactly one step, then leave the world parked.

        - A live loop (running or paused): its controller releases one step and re-parks.
        - No loop (idle / completed / freshly restored): start a one-step run.
        """
        session = self.get_session(world_id)
        if session.controller is not None:
            self._refuse_while_stopping(world_id, session)
            session.controller.step_once()
            # Status names the resting state: nothing fires a status event when the loop
            # re-parks, so RUNNING would never be corrected.
            self._set_status(world_id, session, RunState.PAUSED)
            logger.info("world_stepped_once", extra={"world_id": world_id})
            return
        self.start_run(world_id, steps=1)

    def pause_run(self, world_id: str) -> None:
        """Pause a running world at the next step boundary."""

        session = self.get_session(world_id)
        if session.controller is None:
            raise ValueError(f"World {world_id!r} is not running.")
        self._refuse_while_stopping(world_id, session)
        session.controller.pause()
        self._set_status(world_id, session, RunState.PAUSED)
        logger.info("world_run_paused", extra={"world_id": world_id})

    def resume_run(self, world_id: str) -> None:
        """Resume a paused world."""

        session = self.get_session(world_id)
        if session.controller is None:
            raise ValueError(f"World {world_id!r} is not running.")
        self._refuse_while_stopping(world_id, session)
        session.controller.resume()
        self._set_status(world_id, session, RunState.RUNNING)
        logger.info("world_run_resumed", extra={"world_id": world_id})

    @staticmethod
    def _refuse_while_stopping(world_id: str, session: RuntimeSession) -> None:
        """A stop is final: the controller ignores pause/resume/step after it, so reporting their
        state would hide STOPPING until the loop exits."""
        if session.controller is not None and session.controller.stop_requested:
            raise ValueError(f"World {world_id!r} is stopping.")

    def stop_run(self, world_id: str) -> None:
        """Request a running world to stop at the next step boundary."""

        session = self.get_session(world_id)
        if session.controller is None:
            raise ValueError(f"World {world_id!r} is not running.")
        session.controller.stop()
        self._set_status(world_id, session, RunState.STOPPING)
        logger.info("world_run_stopping", extra={"world_id": world_id})

    def run_status(self, world_id: str) -> RunState | None:
        """Return the run state for a world, or None if it has no live session."""

        session = self._sessions.get(world_id)
        return session.status if session is not None else None

    async def delete_world(self, world_id: str) -> None:
        """Permanently remove a world and all of its persisted state.

        Stops any active run, then purges every store the world touched:
        snapshots, agent state + relations, memory vectors, in-flight messages,
        traces, and the catalog entry. Irreversible.

        Runs under the lifecycle lock: once the stopped loop exits the session looks idle, and a
        concurrent ``/run`` would otherwise write to stores being purged.
        """

        async with self._lifecycle_lock(world_id):
            await self._delete_world_locked(world_id)

    async def _delete_world_locked(self, world_id: str) -> None:
        session = self._sessions.get(world_id)
        if session is not None and session.task is not None and not session.task.done():
            if session.controller is not None:
                session.controller.stop()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        if session is not None:
            await session.runtime.aclose()
        self._sessions.pop(world_id, None)

        container = self._container
        await container.snapshot.delete_world(world_id)
        await container.agent_store.delete_world(world_id)
        await container.vector_store.delete_world(world_id)
        await container.message_provider.clear(world_id)
        container.trace_sink.delete_world(world_id)
        if self._catalog is not None:
            self._catalog.remove(world_id)
        logger.info("world_deleted", extra={"world_id": world_id})

    async def _collect_fired_events(
        self, world_id: str, steps: list[int]
    ) -> list[dict[str, object]]:
        """Gather serialized world events persisted across a world's snapshots.

        Used to rehydrate EventSystem throttle state on restore so a restored world
        does not re-trigger a fresh window quota.
        """
        fired_events: list[dict[str, object]] = []
        for step in sorted(s for s in steps if s > 0):
            snapshot = await self._container.snapshot.load(world_id, step)
            if snapshot is not None:
                fired_events.extend(snapshot.events_this_step)
        return fired_events

    async def _restore_pending_messages(
        self, world_id: str, snapshot: "WorldSnapshot | None"
    ) -> None:
        """Re-enqueue in-flight messages so a restored world keeps delivering them.

        The in-memory queue is lost across processes. The snapshot's ``pending_messages`` is
        authoritative, so clear-then-enqueue is idempotent for an in-process re-restore.
        Broadcasts are refilled in ``_build_runtime``: their channel belongs to the runtime.
        """
        provider = self._container.message_provider
        await provider.clear(world_id)
        if snapshot is None:
            return
        for message in snapshot.pending_messages:
            await provider.enqueue(message)

    def _build_runtime(
        self,
        world: World,
        *,
        start_step: int = 0,
        fired_events: list[dict[str, object]] | None = None,
        start_event_seq: int = 0,
        pending_broadcasts: "list[Broadcast] | None" = None,
    ) -> NarrativeRuntime:
        # Pass the frozen clock config by reference; a field-by-field copy can only drop a field
        # (dropping `calendar` sends every non-Chinese-calendar world back to "九月初一").
        clock = GlobalClock(world.clock_config, start_step=start_step)
        message_system = MessageSystem(
            self._container.message_provider,
            world_id=world.world_id,
        )
        broadcast_channel = BroadcastChannel()
        # The only place to restore not-yet-due broadcasts (death notices among them): the
        # channel is fresh on every _build_runtime.
        for broadcast in pending_broadcasts or ():
            broadcast_channel.publish(broadcast)
        runtime = NarrativeRuntime(
            world_id=world.world_id,
            clock=clock,
            scheduler=AgentScheduler(
                seed=world.world_id,
                main_max_idle_steps=self._config.engine.main_max_idle_steps,
                background_max_idle_steps=self._config.engine.background_max_idle_steps,
            ),
            environment=world.environment,
            message_system=message_system,
            event_settings=EventSettings(
                core_tension=world.analysis.core_tension,
                narrative_theme=world.analysis.narrative_theme,
                enabled=self._config.engine.event_system_enabled,
                check_interval=self._config.engine.event_check_interval,
                max_events_per_window=self._config.engine.max_events_per_window,
                quota_window=self._config.engine.event_quota_window,
            ),
            snapshot_provider=self._container.snapshot,
            agent_store=self._container.agent_store,
            event_bus=self._container.event_bus,
            broadcast_channel=broadcast_channel,
            directory=world.directory,
            executor_registry=build_default_registry(
                self._container.llm_router,
                world.directory,
                seconds_per_step=clock.config.seconds_per_step,
                world_start_second_of_day=WorldTime.from_step(0, clock.config).elapsed_seconds,
            ),
            maintenance=CognitionMaintenance(
                memory_decay_interval=self._config.engine.decay_interval,
                long_term_goal_revision_interval=self._config.engine.long_term_goal_revision_interval,
                compression_enabled=self._config.engine.compression_enabled,
                reflection_enabled=self._config.engine.reflection_enabled,
                relation_evolution_enabled=self._config.engine.relation_evolution_enabled,
                long_term_goal_revision_enabled=self._config.engine.long_term_goal_revision_enabled,
            ),
            pressure_evaluator=WorldPressureEvaluator(llm_router=self._container.llm_router),
            llm_router=self._container.llm_router,
            trace_sink=self._container.trace_sink,
            start_event_seq=start_event_seq,
            fired_events=fired_events or (),
        )
        return runtime


def _event_seq_of(snapshot: "WorldSnapshot | None") -> int:
    """Next-to-assign event-seq counter, from ``metadata["event_seq"]``.

    Persisted by ``NarrativeRuntime.run_step``; 0 when absent.
    """
    if snapshot is None:
        return 0
    raw = snapshot.metadata.get("event_seq")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0
