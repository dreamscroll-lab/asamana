"""A world built and confirmed, as a director who revises nothing would leave it."""

from __future__ import annotations

from typing import Any

from engine.application import NarrativeApplication
from world import World


async def confirmed_world(application: NarrativeApplication, theme: str, **build: Any) -> World:
    world_id = await application.build_world(theme, **build)
    return await application.confirm_world(world_id)
