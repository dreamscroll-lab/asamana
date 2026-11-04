"""The closed vocabularies a client names cognition in: every value and its label.

World-independent: the same for every world, every viewer and every surface that offers them
(the setup's initial emotion and needs, the actor's emotion amendment).
"""

from __future__ import annotations

from typing import Any

from agent.need import NeedType
from agent.personality import EmotionType


def build_vocabulary_router() -> Any:
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/api/vocabulary")
    async def get_vocabulary() -> dict[str, list[dict[str, str]]]:
        return {
            "emotions": [{"value": e.value, "label": e.label} for e in EmotionType],
            "needs": [{"value": n.value, "label": n.label} for n in NeedType],
        }

    return router
