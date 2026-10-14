"""An environment as a snapshot saves it, and back: its share of the world state plus the
engine's bookkeeping."""

from __future__ import annotations

from typing import Any

from engine.environment import EnvironmentSystem


def saved(env: EnvironmentSystem) -> dict[str, Any]:
    return {**env.world_state(), "engine": env.engine_state()}


def restore(env: EnvironmentSystem, payload: dict[str, Any]) -> None:
    env.restore_state(
        npcs=payload.get("npcs", {}), entities=payload.get("entities", {}),
        engine=payload.get("engine", {}),
    )


def npc_outcomes(env: EnvironmentSystem) -> dict[str, dict[str, Any]]:
    """This beat's outcome per Npc, only for those something happened to."""
    return {nid: n["outcome"] for nid, n in env.world_state()["npcs"].items() if n["outcome"]}
