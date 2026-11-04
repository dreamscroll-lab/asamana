"""Application-facing orchestration for world build and runtime execution."""

from __future__ import annotations


import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Sequence
from uuid import uuid4

from config.models import Config
from core.container import Container
from core.context import get_log_context, observe_stage, set_log_context
from core.factory import ComponentKind, ProviderFactory
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
from core.interfaces.trace import Stage
from core.interfaces.world_config import WorldConfig
from core.logging import get_logger
from core.interfaces.perception import Broadcast
from core.interfaces.snapshot import WorldSnapshot
from engine.broadcast import BroadcastChannel
from engine.clock import GlobalClock, WorldTime
from engine.cognition_maintenance import CognitionMaintenance
from engine.director import DirectivePrompt, DirectiveResult, DirectorChannel, NpcOnMenu
from engine.narrative_editor import NarrativeEditorSettings
from engine.executors import build_default_registry
from engine.message_system import MessageSystem
from engine.run_control import RunController, RunState, StepWithdrawn
from engine.runtime import NarrativeRuntime, RuntimeStepResult
from engine.scheduler import AgentScheduler
from engine.world_pressure import WorldPressureEvaluator
from world import ActorCasting, World, WorldBuilder, WorldCatalog
from world.initializer import WorldInitializer
from world.setup import MapPlaces, SetupRejected, WorldSetup
from world.stored_config import StoredWorldConfig
from worlds import TiledWorldConfig  # noqa: F401  registers the built-in WorldConfig kinds

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
    ) -> str:
        """Build a world's setup and catalog the world; returns its id.

        Nothing runs yet: the world is materialized when it is confirmed (``confirm_world``),
        and until then its setup may be revised. The map is chosen per world: a named
        ``template``, else picked from ``available_templates`` by the theme. With neither the
        build fails; a deployment-wide default map would hide that.
        """

        world_id = str(uuid4())
        setup = await self._builder.build(
            theme=theme,
            world_id=world_id,
            world_configs=self._candidates(template, available_templates),
            min_agents=self._config.world.min_agents,
            max_agents=self._config.world.max_agents,
            max_npcs=self._config.engine.max_npcs,
        )
        if self._catalog is not None:
            self._catalog.register(
                world_id,
                theme=theme,
                world_name=setup.lore.world_name,
                created_at=datetime.now(),
            )
        logger.info(
            "application_world_registered",
            extra={
                "world_id": world_id,
                "world_name": setup.lore.world_name,
                "agent_count": len(setup.characters),
            },
        )
        return world_id

    async def confirm_world(self, world_id: str, *, character_id: str | None = None) -> World:
        """Lock a world's setup and materialize the world from it: step 0 (with the setup frozen
        into it), the agents and their stores, and a session to run them. The setup file goes once
        all of that stands.

        ``character_id`` casts the actor in the same act: the cast is part of the setup, so this
        is the only place it is chosen, and never after the director has seen every secret.
        Confirming a confirmed world again just returns it. A materialization that fails leaves
        the world unconfirmed, and the next attempt clears what it wrote.
        """
        async with self._lifecycle_lock(world_id):
            entry = self._catalog.get(world_id) if self._catalog is not None else None
            if self._catalog is not None and self._catalog.is_confirmed(world_id):
                if character_id is not None:
                    raise ValueError(f"World {world_id!r} is confirmed; its cast is locked.")
                if (session := self._sessions.get(world_id)) is not None:
                    return session.world
                return await self._restore_session_locked(world_id, casting=self._casting(world_id))
            casting = None
            if character_id is not None:
                if self._catalog is None or entry is None:
                    raise ValueError(f"World {world_id!r} is not cataloged.")
                if self._catalog.director_reviewed(world_id):
                    raise ValueError(f"World {world_id!r} was reviewed by the director; nobody can be cast.")
                setup = await WorldSetup.load(self._container.snapshot, world_id)
                if setup is None:
                    raise ValueError(f"World {world_id!r} has no setup.")
                if character_id not in {d.agent_id for d in setup.characters}:
                    raise ValueError(f"World {world_id!r} has no character {character_id!r} to cast the actor as")
                casting = ActorCasting(character_id=character_id)
            await self._clear_materialized(world_id)
            initializer = WorldInitializer(self._container, max_npcs=self._config.engine.max_npcs)
            world = await initializer.initialize(world_id, casting=casting)
            runtime = self._build_runtime(world, casting=casting)
            # The catalog before the session: a live session the catalog calls unconfirmed or
            # uncast would run without the viewer filters the casting turns on. A failed write
            # leaves no session, and confirming again starts over from the setup file.
            if self._catalog is not None:
                try:
                    if casting is not None:
                        self._catalog.set_actor_casting(world_id, casting)
                    self._catalog.set_confirmed(world_id, True)
                except Exception:
                    await runtime.aclose()
                    raise
            self._sessions[world_id] = RuntimeSession(world=world, runtime=runtime)
            # Last: the setup now lives frozen in step 0. Left behind by a failure here, the file
            # still holds what step 0 froze, and a confirmed world takes no revision.
            await self._container.snapshot.delete_world_setup(world_id)
            logger.info(
                "world_confirmed",
                extra={"world_id": world_id, "agent_count": len(world.agents), "cast": character_id},
            )
            return world

    async def _clear_materialized(self, world_id: str) -> None:
        """Drop what an earlier, failed confirmation wrote; the setup file and the frozen map stay."""
        container = self._container
        await container.snapshot.delete_steps_after(world_id, -1)
        await container.vector_store.delete_world(world_id)
        await container.message_provider.clear(world_id)

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
            return await self._restore_session_locked(world_id, casting=self._casting(world_id))

    async def _restore_session_locked(
        self, world_id: str, *, casting: ActorCasting | None,
    ) -> World:
        initializer = WorldInitializer(self._container, max_npcs=self._config.engine.max_npcs)
        world = await initializer.restore(world_id, casting=casting)
        steps = await self._container.snapshot.list_steps(world_id)
        latest_step = max((s for s in steps if s > 0), default=0)
        landed_injections = await self._collect_landed_injections(world_id, steps)
        # Shared by in-flight messages, pending broadcasts and the event-seq counter.
        # Don't skip step 0: the `s > 0` filter above only sets the clock, and a never-run
        # world's step-0 snapshot still carries pending messages (same rule in
        # WorldInitializer._restore_environment_state).
        latest = await self._container.snapshot.load(world_id, latest_step)
        await self._restore_pending_messages(world_id, latest)
        # Keeps the event seq strictly increasing across restart (engine.event_seq.EventSequencer).
        start_event_seq = latest.engine.event_seq if latest else 0
        self._sessions[world_id] = RuntimeSession(
            world=world,
            runtime=self._build_runtime(
                world,
                casting=casting,
                start_step=latest_step,
                landed_injections=landed_injections,
                start_event_seq=start_event_seq,
                pending_broadcasts=latest.engine.pending_broadcasts if latest else None,
            ),
        )
        logger.info(
            "application_world_session_restored",
            extra={
                "world_id": world_id,
                "world_name": world.setup.lore.world_name,
                "agent_count": len(world.agents),
            },
        )
        return world

    async def reset_session(self, world_id: str) -> World:
        """Reset a world to its initial state (step 0) and register a fresh session.

        Drops everything after step 0, then rebuilds through the canonical ``restore`` path,
        which resumes from step 0's snapshot: agent states and relations come back as built.
        Memory keeps its backstory seeds and traces keep the build segment: only what the *run*
        produced is purged.

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

        # Out before the stores are touched: a reset that fails halfway must not leave the step-N
        # session in place, running on against stores already rewound. With none, the next access
        # restores from what storage holds. Closing also stops in-flight event generation,
        # written for the step-N world.
        if existing is not None:
            del self._sessions[world_id]
            await existing.runtime.aclose()

        # Drop in-flight messages and post-step-0 snapshots so restore resumes from step 0 and
        # no stale message is delivered after the reset.
        await self._container.message_provider.clear(world_id)
        await self._container.snapshot.delete_steps_after(world_id, 0)

        # Traces rewind with the clock (see TraceSink.delete_run_traces).
        self._container.trace_sink.delete_run_traces(world_id)

        # restore() resumes at step 0 and purges every memory written after it. The casting
        # stands: it belongs to the world's setup, not to its run.
        casting = self._casting(world_id)
        world = await initializer.restore(world_id, casting=casting)

        self._sessions[world_id] = RuntimeSession(
            world=world,
            runtime=self._build_runtime(world, casting=casting, start_step=0),
        )
        logger.info(
            "world_session_reset",
            extra={"world_id": world_id, "world_name": world.setup.lore.world_name},
        )
        return world

    # ------------------------------------------------------------------
    # Actor casting. Chosen once while the world is still being set up; afterwards the only
    # change is the actor's one-way exit.
    # ------------------------------------------------------------------

    def _casting(self, world_id: str) -> ActorCasting | None:
        return self._catalog.actor_casting(world_id) if self._catalog is not None else None

    async def open_director_review(self, world_id: str) -> None:
        """Show the user the whole cast, as the director. One way: whoever has read every
        character's secrets can't then live in the world as one of them, so the world takes no
        actor afterwards."""
        async with self._lifecycle_lock(world_id):
            if self._catalog is None or self._catalog.get(world_id) is None:
                raise ValueError(f"World {world_id!r} is not cataloged.")
            if self.performing_character(world_id) is not None:
                raise ValueError(f"An actor performs in world {world_id!r}; the director's review is closed.")
            self._catalog.mark_director_review(world_id)

    async def release_actor(self, world_id: str) -> World:
        """The actor steps out for good; the character goes back to AI cognition.

        It stays main tier, and nobody can be cast in this world again. Stop the run first: a
        step suspended on the actor's turn is discarded by the stop (see ``stop_run``).
        """
        async with self._lifecycle_lock(world_id):
            casting = self._casting(world_id)
            if casting is None or casting.exited:
                raise ValueError(f"No actor is performing a character in world {world_id!r}.")
            return await self._recast_locked(
                world_id, ActorCasting(character_id=casting.character_id, exited=True),
            )

    async def record_director_read(self, world_id: str) -> None:
        """Keep a private read and actor casting mutually exclusive."""
        async with self._lifecycle_lock(world_id):
            if self.performing_character(world_id) is not None:
                raise ValueError(f"An actor performs in world {world_id!r}; private director reads are closed.")
            # Uncataloged worlds cannot cast an actor, so they need no casting gate.
            if (
                self._catalog is not None
                and self._catalog.get(world_id) is not None
                and not self._catalog.is_confirmed(world_id)
            ):
                self._catalog.mark_director_review(world_id)

    async def revise_setup(
        self, world_id: str, revise: Callable[[WorldSetup, MapPlaces], WorldSetup],
    ) -> WorldSetup:
        """Apply a revision to a world's setup.

        Open until the world is confirmed: confirming locks the setup, and nothing is
        materialized before it, so a revision rewrites the setup file alone.
        ``revise`` gets the setup and the world's map. Raises ``ValueError``
        when the setup is closed, and ``SetupRejected`` when ``revise`` refuses the revision;
        either way nothing changes.
        """
        async with self._lifecycle_lock(world_id):
            if self._catalog is None or self._catalog.get(world_id) is None:
                raise ValueError(f"World {world_id!r} is not cataloged.")
            if self._catalog.is_confirmed(world_id):
                raise ValueError(f"World {world_id!r} is confirmed; its setup is locked.")
            snapshots = self._container.snapshot
            stored = await snapshots.load_world_setup(world_id)
            config = await snapshots.load_world_config(world_id)
            if stored is None or config is None:
                raise ValueError(f"World {world_id!r} has no setup to revise.")
            places = MapPlaces.of(StoredWorldConfig(config))
            try:
                setup = revise(WorldSetup.from_dict(stored), places)
            except ValueError as exc:
                raise SetupRejected(str(exc)) from exc
            await snapshots.save_world_setup(world_id, setup.as_dict())
            return setup

    def performing_character(self, world_id: str) -> str | None:
        """The character an actor performs in this world now; None once nobody does."""
        casting = self._casting(world_id)
        return casting.character_id if casting is not None and not casting.exited else None

    def actor_turn(self, world_id: str) -> ActorTurn | None:
        """The turn waiting for the actor, if this world's live session has one."""
        session = self._sessions.get(world_id)
        return session.runtime.actor.pending_turn if session is not None else None

    async def actor_intent(self, world_id: str, intent: ActorIntent) -> ActorIntentResult:
        """Interpret or submit actor intent in the suspended character's decision trace."""
        session = self.get_session(world_id)
        turn = session.runtime.actor.pending_turn
        step = turn.step if turn is not None else session.runtime.clock.current_step
        previous = get_log_context()
        # HTTP requests don't inherit the suspended step's observation context.
        set_log_context(world_id=world_id, step=str(step))
        try:
            with observe_stage(Stage.DECISION, agent_id=turn.agent_id if turn is not None else ""):
                return await session.runtime.actor_intent(intent)
        finally:
            set_log_context(**previous)
            self._flush_trace_async(world_id, step)

    def actor_operations(self, world_id: str, turn_id: str, target: Ref) -> ActorOperations:
        return self.get_session(world_id).runtime.actor_operations(turn_id, target)

    def actor_break_off(self, world_id: str) -> None:
        """Break off what the performed character is doing. Raises ``ActorAnswerRejected`` when
        it is doing nothing, ``ValueError`` when nobody is performed in this session."""
        self.get_session(world_id).runtime.actor_break_off()

    async def actor_thread(self, world_id: str, *, steps: int) -> list[dict[str, Any]]:
        """The actor's thread over the latest ``steps`` committed steps, as their snapshots keep
        it, then what the step under way has recorded so far."""
        store = self._container.snapshot
        thread: list[dict[str, Any]] = []
        for step in sorted(await store.list_steps(world_id))[-steps:]:
            snapshot = await store.load(world_id, step)
            if snapshot is not None:
                thread.extend(snapshot.actor_thread)
        session = self._sessions.get(world_id)
        if session is not None:
            thread.extend(session.runtime.actor_live_thread)
        return thread

    async def set_actor_pace(self, world_id: str, pace: ActorPace) -> None:
        """Set when the performed character's actor is asked to decide; kept with the casting."""
        async with self._lifecycle_lock(world_id):
            casting = self._casting(world_id)
            if casting is None or casting.exited:
                raise ValueError(f"No actor is performing a character in world {world_id!r}.")
            assert self._catalog is not None
            self._catalog.set_actor_casting(world_id, replace(casting, pace=pace))
            session = self._sessions.get(world_id)
            if session is not None:
                session.runtime.set_actor_pace(pace)

    async def actor_mind(self, world_id: str) -> ActorMind:
        """The performed character's mind as the actor sees it; ``ValueError`` if nobody is
        performed in this session."""
        return await self.get_session(world_id).runtime.actor_mind()

    async def amend_actor_mind(self, world_id: str, amendments: ActorAmendments) -> bool:
        """Save the actor's amendments to the character's mind; whether they landed now or wait
        for the next step. Raises ``ActorAnswerRejected`` for one the mind doesn't list."""
        return await self.get_session(world_id).runtime.amend_actor_mind(amendments)

    def answer_actor_turn(
        self,
        world_id: str,
        *,
        turn_id: str,
        answer: ActorPublicAnswer,
    ) -> None:
        """Answer the open turn, which must be ``turn_id``: an answer meant for a turn that has
        since closed must not land on the next one.

        Raises ``ActorAnswerRejected`` when the turn refuses the answer (it stays open), and
        ``ValueError`` when that turn is no longer waiting or takes another kind of answer.
        """
        turn = self.actor_turn(world_id)
        if turn is None or turn.turn_id != turn_id:
            raise ValueError("That turn is no longer waiting for the actor")
        self.get_session(world_id).runtime.answer_actor(answer)

    async def _recast_locked(self, world_id: str, casting: ActorCasting) -> World:
        """Rebuild the session under ``casting``, then record it.

        Rebuilt, not patched: whether a character is actor-controlled, and its tier, are set
        when it is constructed, the same way on every restore. The new session is built first,
        so a casting the world can't take (an unknown character) changes nothing, and the old one
        stays until the catalog has the casting: a refused write puts it back, or the live
        session would drive the character one way while the catalog and viewers say the other.
        """
        existing = self._sessions.get(world_id)
        if existing is not None and self._run_is_live(existing):
            raise ValueError(
                f"World {world_id!r} is running — stop the run before changing who performs a character."
            )
        world = await self._restore_session_locked(world_id, casting=casting)
        assert self._catalog is not None
        try:
            self._catalog.set_actor_casting(world_id, casting)
        except Exception:
            recast = self._sessions.pop(world_id)
            if existing is not None:
                self._sessions[world_id] = existing
            await recast.runtime.aclose()
            raise
        if existing is not None:
            await existing.runtime.aclose()
        logger.info(
            "actor_casting_changed",
            extra={
                "world_id": world_id,
                "agent_id": casting.character_id,
                "exited": casting.exited,
            },
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
            if self._trace_flush_tasks:
                await asyncio.gather(*self._trace_flush_tasks, return_exceptions=True)
            return
        for world_id in live:
            # None once the loop is past its last step and only finishing up.
            if self._sessions[world_id].controller is not None:
                self.stop_run(world_id)
        logger.info("shutdown_waiting_for_runs", extra={"world_ids": sorted(live)})
        await asyncio.gather(*live.values(), return_exceptions=True)
        if self._trace_flush_tasks:
            await asyncio.gather(*self._trace_flush_tasks, return_exceptions=True)
        logger.info("shutdown_runs_stopped", extra={"world_ids": sorted(live)})

    def _set_status(self, world_id: str, session: RuntimeSession, state: RunState) -> None:
        """Update a session's run state and announce it on the event bus, like step events."""
        session.status = state
        self._container.event_bus.publish(
            {"type": "status", "world_id": world_id, "status": state.value}
        )

    async def _discard_session(self, world_id: str, session: RuntimeSession) -> None:
        """Drop a session whose step stopped halfway (withdrawn at the actor's turn, or raised).

        What the step wrote is in this session's memory and partly in the stores; running on
        from it would commit a step that never happened whole. The next restore resumes at the
        last committed step and rolls the rest back.
        """
        if self._sessions.get(world_id) is session:
            del self._sessions[world_id]
        try:
            await session.runtime.aclose()
        except Exception as exc:  # noqa: BLE001 — the session is already gone; closing is cleanup
            logger.warning("session_close_failed", extra={"world_id": world_id, "error": str(exc)})

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
        except StepWithdrawn:
            self._set_status(world_id, session, RunState.COMPLETED)
            await self._discard_session(world_id, session)
            logger.info("world_run_abandoned_at_actor_turn", extra={"world_id": world_id})
        except Exception as exc:  # noqa: BLE001 — surface any run failure as FAILED status
            self._set_status(world_id, session, RunState.FAILED)
            logger.error(
                "world_run_failed",
                extra={"world_id": world_id, "error": str(exc)},
            )
            await self._discard_session(world_id, session)
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
        boundary as NarrativeEditor), and once, so submit and the developer instruments below never
        show the model two different worlds.
        """
        session = self.get_session(world_id)
        director = session.runtime.director
        environment = session.world.environment
        return director, {
            "all_agents": session.world.agents,
            "agent_locations": environment.body_locations(session.world.agents),
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
        session = self._sessions.get(world_id)
        # An open step still annotates its calls; the runtime flushes them at commit or run exit.
        if session is not None and session.runtime.step_in_progress:
            return
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
        # A step waiting on the actor would never reach the boundary: withdraw the turn, and
        # the run loop discards the half-run step.
        session.runtime.actor.abandon()
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
        if session is not None and session.runtime.is_running and session.task is None:
            raise ValueError(f"World {world_id!r} is running — wait for the run before deleting.")
        if session is not None and session.task is not None and not session.task.done():
            if session.controller is not None:
                session.controller.stop()
            session.runtime.actor.abandon()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        if session is not None:
            await session.runtime.aclose()
        self._sessions.pop(world_id, None)

        container = self._container
        await container.snapshot.delete_world(world_id)
        await container.vector_store.delete_world(world_id)
        await container.message_provider.clear(world_id)
        container.trace_sink.delete_world(world_id)
        if self._catalog is not None:
            self._catalog.remove(world_id)
        logger.info("world_deleted", extra={"world_id": world_id})

    async def _collect_landed_injections(
        self, world_id: str, steps: list[int]
    ) -> list[dict[str, object]]:
        """Gather serialized world events persisted across a world's snapshots.

        Used to rehydrate NarrativeEditor throttle state on restore so a restored world
        does not re-trigger a fresh window quota.
        """
        landed_injections: list[dict[str, object]] = []
        for step in sorted(s for s in steps if s > 0):
            snapshot = await self._container.snapshot.load(world_id, step)
            if snapshot is not None:
                landed_injections.extend(snapshot.events_of("injection"))
        return landed_injections

    async def _restore_pending_messages(
        self, world_id: str, snapshot: "WorldSnapshot | None"
    ) -> None:
        """Re-enqueue in-flight messages so a restored world keeps delivering them.

        The in-memory queue is lost across processes. The snapshot's pending messages are
        authoritative, so clear-then-enqueue is idempotent for an in-process re-restore.
        Broadcasts are refilled in ``_build_runtime``: their channel belongs to the runtime.
        """
        provider = self._container.message_provider
        await provider.clear(world_id)
        if snapshot is None:
            return
        for message in snapshot.engine.pending_messages:
            await provider.enqueue(message)

    def _build_runtime(
        self,
        world: World,
        *,
        casting: ActorCasting | None = None,
        start_step: int = 0,
        landed_injections: list[dict[str, object]] | None = None,
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
        executor_registry = build_default_registry(
            self._container.llm_router,
            world.directory,
            seconds_per_step=clock.config.seconds_per_step,
            world_start_second_of_day=WorldTime.from_step(0, clock.config).elapsed_seconds,
        )
        # The registry is fresh on every _build_runtime too: the acts a restored world was in
        # the middle of go on from where its snapshot left them.
        for execution in world.active_executions:
            executor_registry.add_active(execution)
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
            narrative_editor_settings=NarrativeEditorSettings(
                narrative_theme=world.setup.lore.narrative_theme,
                enabled=self._config.engine.narrative_editor_enabled,
                check_interval=self._config.engine.injection_check_interval,
                max_injections_per_window=self._config.engine.max_injections_per_window,
                quota_window=self._config.engine.injection_quota_window,
            ),
            snapshot_provider=self._container.snapshot,
            event_bus=self._container.event_bus,
            broadcast_channel=broadcast_channel,
            directory=world.directory,
            executor_registry=executor_registry,
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
            landed_injections=landed_injections or (),
        )
        # Wired here, once: the step loop never asks who performs a character.
        if casting is not None and not casting.exited:
            runtime.cast_actor(world.agents[casting.character_id], pace=casting.pace)
        return runtime
