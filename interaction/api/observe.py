"""Read-only observation routes + the live WebSocket feed.

Render-neutral: every payload is semantic world state (steps, the relationship
graph). No coordinates/layout — a client lays it out. A 2D renderer turns the
symbolic state into a map on the client; a text view ignores the geometry.

Public character, NPC and place identities are independent of viewpoint. While an actor performs a character,
steps, profiles, graphs and the feed answer as that character perceived the world
(``interaction.actor_view``); once the actor exits, they answer in full again.

The WebSocket pushes off the in-process event bus (subscribe, not poll): on
connect it sends the current run state, the turn waiting for the actor (if one is) and the latest
persisted step (a ``latest_step`` frame), then forwards each live ``step``, ``status``,
``actor_turn`` and ``actor_entry`` event for its world. An ``actor_turn`` frame carries the whole
turn when one opens and null when it closes; an ``actor_entry`` frame one new entry of the actor's
thread, whose history a client reads from ``GET /actor/thread``.
"""

from __future__ import annotations

from typing import Any

from core.logging import get_logger
from world.stored_config import StoredWorldConfig

# WebSocket must be importable at module scope: with `from __future__ import annotations`,
# FastAPI resolves the `websocket: WebSocket` hint against module globals, so a function-local
# import makes it read `websocket` as a query param and reject every handshake. Only
# create_app imports this module, and it already requires fastapi.
from fastapi import WebSocket, WebSocketDisconnect

from interaction.api.app import ApiServices
from interaction.actor_view import graph_for, profile_for, step_view_for
from interaction.api.serialization import to_jsonable
from interaction.models import (
    AgentProfile,
    CharacterIdentity,
    GraphNode,
    PlaceIdentity,
    StepView,
    WorldGraph,
    relations_from_snapshot,
)

logger = get_logger(__name__)


def build_observe_router(services: ApiServices) -> Any:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(tags=["observe"])
    manager = services.manager
    application = services.application
    container = services.container
    event_bus = container.event_bus

    def _step(world_id: str, view: StepView) -> dict[str, Any]:
        viewer = application.performing_character(world_id)
        return to_jsonable(step_view_for(view, viewer) if viewer is not None else view)

    def _profile(world_id: str, profile: AgentProfile) -> dict[str, Any]:
        viewer = application.performing_character(world_id)
        return to_jsonable(profile_for(profile, viewer) if viewer is not None else profile)

    async def _step_payload(world_id: str, step: int) -> dict[str, Any]:
        replayer = manager.create_replayer(world_id)
        try:
            view = await replayer.get_step(step)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _step(world_id, view)

    # ----- steps ----------------------------------------------------------

    @router.get("/api/worlds/{world_id}/steps")
    async def list_steps(world_id: str) -> list[int]:
        return await manager.list_steps(world_id)

    @router.get("/api/worlds/{world_id}/steps/{step}")
    async def get_step(world_id: str, step: int) -> dict[str, Any]:
        return await _step_payload(world_id, step)

    # ----- relationship graph ---------------------------------------------

    @router.get("/api/worlds/{world_id}/graph")
    async def get_graph(world_id: str, step: int | None = None) -> dict[str, Any]:
        if step is not None:
            snapshot = await manager.snapshot_provider.load(world_id, step)
        else:
            snapshot = await manager.latest_snapshot(world_id)
        if snapshot is None:
            return to_jsonable(WorldGraph())
        profiles = await manager.character_profiles(world_id)
        nodes = [
            GraphNode(
                id=aid,
                name=s.get("agent_name") or "某人",
                role=profile.role if (profile := profiles.get(aid)) else "",
                is_main_character=bool(s.get("is_main_character", False)),
            )
            for aid, s in snapshot.state.agents.items()
        ]
        graph = WorldGraph(nodes=nodes, edges=relations_from_snapshot(snapshot))
        viewer = application.performing_character(world_id)
        return to_jsonable(graph_for(graph, viewer) if viewer is not None else graph)

    @router.get("/api/worlds/{world_id}/agents")
    async def list_cast(world_id: str) -> list[dict[str, Any]]:
        """The world's public identities, identical for every viewer."""
        if await manager.get_world(world_id) is None:
            raise HTTPException(status_code=404, detail=f"World not found: {world_id}")
        profiles = await manager.character_profiles(world_id)
        return [
            to_jsonable(CharacterIdentity(
                agent_id=agent_id,
                name=profile.name,
                gender=profile.gender,
                age=profile.age,
                role=profile.role,
                background=profile.background,
                appearance=profile.appearance,
                color=profile.color,
            ))
            for agent_id, profile in sorted(profiles.items())
        ]

    @router.get("/api/worlds/{world_id}/npcs")
    async def list_npcs(world_id: str) -> list[dict[str, Any]]:
        """The world's NPC identities, independent of current presence and viewpoint."""
        identities = await manager.npc_identities(world_id)
        if identities is None:
            raise HTTPException(status_code=404, detail=f"No world setup for: {world_id}")
        return [to_jsonable(identity) for _, identity in sorted(identities.items())]

    @router.get("/api/worlds/{world_id}/places")
    async def list_places(world_id: str) -> list[dict[str, Any]]:
        """The world's places, fixed at build and identical for every viewer."""
        data = await manager.snapshot_provider.load_world_config(world_id)
        if not data:
            raise HTTPException(status_code=404, detail=f"No world config for: {world_id}")
        places = StoredWorldConfig(data).get_places()
        return [
            to_jsonable(PlaceIdentity(
                location_id=place_id, name=place.name, description=place.description,
            ))
            for place_id, place in sorted(places.items())
        ]

    @router.get("/api/worlds/{world_id}/agents/{agent_id}/profile")
    async def get_profile(world_id: str, agent_id: str) -> dict[str, Any]:
        """Static identity from the world's setup (name/role/background/traits…).

        Distinct from the per-step ``AgentView``s on a StepView, which carry
        the mutable current state. Used for the creation-review cards and the
        per-step agent detail panel.
        """
        profiles = await manager.character_profiles(world_id)
        profile = profiles.get(agent_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Agent profile not found: {agent_id}")
        return _profile(world_id, profile)

    # ----- live WebSocket feed (push) -------------------------------------

    @router.websocket("/ws/{world_id}")
    async def ws_feed(websocket: WebSocket, world_id: str) -> None:
        await websocket.accept()
        logger.info("ws_connected", extra={"world_id": world_id})

        # Subscribe before sending the initial frames: a step published while the snapshot loads
        # would otherwise be missed for good (live mode doesn't backfill; the map jumps N → N+2).
        # Subscribe to this world only: the bounded queue drops the oldest when full, so another
        # world's burst would push out this connection's frames (see NarrativeEventBus).
        # Everything after subscribe sits inside the try, so a disconnect while the initial
        # frames load still unsubscribes.
        queue = event_bus.subscribe(world_id)
        try:
            await _serve(websocket, world_id, queue)
        except WebSocketDisconnect:
            logger.info("ws_disconnected", extra={"world_id": world_id})
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_error", extra={"world_id": world_id, "error": str(exc)})
        finally:
            event_bus.unsubscribe(queue)

    async def _serve(websocket: WebSocket, world_id: str, queue: Any) -> None:
        """Send initial frames, start the push task, and block reading inbound frames (only to
        detect disconnects).

        The push task lives only inside this function, so every exit path (disconnect, send
        failure, cancellation) goes through the same finally and no task is left holding the
        socket.
        """
        import asyncio

        # Initial frames: current run state, then the latest persisted step.
        state = application.run_status(world_id)
        await websocket.send_json(
            {"type": "status", "data": {"world_id": world_id, "status": state.value if state else None}}
        )
        turn = application.actor_turn(world_id)
        if turn is not None:
            await websocket.send_json({"type": "actor_turn", "data": to_jsonable(turn)})
        steps = await manager.list_steps(world_id)
        # Skip the step the initial frame already covered (subscribing first causes this overlap).
        sent_step = -1
        if steps:
            try:
                latest = await manager.create_replayer(world_id).get_step(max(steps))
                await websocket.send_json({"type": "latest_step", "data": _step(world_id, latest)})
                sent_step = latest.step
            except ValueError:
                pass

        async def _pump() -> None:
            # Guard the whole loop: a payload value to_jsonable can't handle (an enum, set or
            # dataclass in a dict[str, Any] field) would kill this unawaited task silently,
            # leaving the socket open and "connected" while the queue fills.
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    etype = event.get("type")
                    if etype == "step":
                        view = StepView.from_runtime_payload(event)
                        if view.step <= sent_step:
                            continue
                        await websocket.send_json({"type": "step", "data": _step(world_id, view)})
                    elif etype == "actor_turn":
                        await websocket.send_json(
                            {"type": "actor_turn", "data": to_jsonable(event.get("turn"))}
                        )
                    elif etype == "actor_entry":
                        await websocket.send_json({"type": "actor_entry", "data": event.get("entry")})
                    elif etype == "status":
                        await websocket.send_json(
                            {"type": "status",
                             "data": {"world_id": world_id, "status": event.get("status")}}
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ws_pump_failed", extra={"world_id": world_id, "error": str(exc)}
                )
                # Close the socket so the outer receive_text returns and the finally
                # unsubscribes; otherwise the page shows connected but never updates.
                try:
                    await websocket.close()
                except Exception:  # noqa: BLE001 — already closed
                    pass

        pump = asyncio.create_task(_pump())
        try:
            # Block on inbound frames purely to detect client disconnect.
            while True:
                await websocket.receive_text()
        finally:
            pump.cancel()

    return router
