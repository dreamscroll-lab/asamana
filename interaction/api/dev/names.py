"""Cast id → display name for developer payloads."""

from __future__ import annotations

from interaction.world_manager import WorldManager


async def agent_names(manager: WorldManager, world_id: str) -> dict[str, str]:
    profiles = await manager.character_profiles(world_id)
    names = {aid: p.get("name", aid) for aid, p in profiles.items()}
    snapshot = await manager.latest_snapshot(world_id)
    if snapshot is not None:
        for aid, state in snapshot.agent_states.items():
            names.setdefault(aid, state.get("agent_name", aid))
    return names
