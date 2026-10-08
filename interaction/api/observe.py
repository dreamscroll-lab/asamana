"""Read-only observation routes + the live WebSocket feed.

Render-neutral: every payload is semantic world state (steps, the relationship
graph). No coordinates/layout — a client lays it out. A 2D renderer turns the
symbolic state into a map on the client; a text view ignores the geometry.

The WebSocket pushes off the in-process event bus (subscribe, not poll): on
connect it sends the latest persisted step and current run state, then forwards
each live ``step`` and ``status`` event for its world.
"""

from __future__ import annotations

from typing import Any

from core.logging import get_logger

# WebSocket must be importable at module scope: with `from __future__ import annotations`,
# FastAPI resolves the `websocket: WebSocket` hint against module globals, so a function-local
# import makes it read `websocket` as a query param and reject every handshake. Only
# create_app imports this module, and it already requires fastapi.
from fastapi import WebSocket, WebSocketDisconnect

from interaction.api.app import ApiServices
from interaction.api.serialization import to_jsonable
from interaction.models import (
    AgentProfile,
    GraphNode,
    StepEvent,
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

    async def _step_payload(world_id: str, step: int) -> dict[str, Any]:
        replayer = manager.create_replayer(world_id)
        try:
            event = await replayer.get_step(step)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return to_jsonable(event)

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
                role=profiles.get(aid, {}).get("role", ""),
                is_main_character=bool(s.get("is_main_character", False)),
                # Fixed identity color — same value the map/card use.
                color=s.get("color", ""),
            )
            for aid, s in snapshot.agent_states.items()
        ]
        return to_jsonable(WorldGraph(nodes=nodes, edges=relations_from_snapshot(snapshot)))

    def _make_profile(agent_id: str, entry: dict[str, Any]) -> AgentProfile:
        return AgentProfile(
            agent_id=agent_id,
            name=entry.get("name", ""),
            role=entry.get("role", ""),
            age=entry.get("age"),
            gender=entry.get("gender", ""),
            background=entry.get("background", ""),
            appearance=entry.get("appearance", ""),
            color=entry.get("color", ""),
            core_traits=list(entry.get("core_traits", [])),
            core_values=list(entry.get("core_values", [])),
            self_image=entry.get("self_image", ""),
            life_goal=entry.get("life_goal", ""),
            secret=entry.get("secret", ""),
            is_main_character=bool(entry.get("is_main_character", False)),
        )

    @router.get("/api/worlds/{world_id}/agents")
    async def list_profiles(world_id: str) -> list[dict[str, Any]]:
        """The world's whole cast as persisted at step 0, the same list at every step.

        A stable identity key can't come from the per-step ``agent_states`` (the roster would
        depend on the loaded step); the map deals each agent a skin from this list, so a body
        is the same in every step, replay and session.
        """
        profiles = await manager.character_profiles(world_id)
        return [
            to_jsonable(_make_profile(agent_id, entry))
            for agent_id, entry in sorted(profiles.items())
        ]

    @router.get("/api/worlds/{world_id}/agents/{agent_id}/profile")
    async def get_profile(world_id: str, agent_id: str) -> dict[str, Any]:
        """Static identity persisted at step 0 (name/role/background/traits…).

        Distinct from the per-step ``agent_states`` on a StepEvent, which carry
        the mutable current state. Used for the creation-review cards and the
        per-step agent detail panel.
        """
        profiles = await manager.character_profiles(world_id)
        entry = profiles.get(agent_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Agent profile not found: {agent_id}")
        return to_jsonable(_make_profile(agent_id, entry))

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
        steps = await manager.list_steps(world_id)
        # Skip the step the initial snapshot already covered (subscribing first causes this overlap).
        sent_step = -1
        if steps:
            try:
                latest = await manager.create_replayer(world_id).get_step(max(steps))
                await websocket.send_json({"type": "snapshot", "data": to_jsonable(latest)})
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
                        se = StepEvent.from_runtime_payload(event)
                        if se.step <= sent_step:
                            continue
                        await websocket.send_json({"type": "step", "data": to_jsonable(se)})
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
