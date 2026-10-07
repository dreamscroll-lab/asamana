"""World construction package.

This package owns the build-stage narrative pipeline (theme → world) and the world model
(``world.models``) the runtime engine reads; the build pipeline itself never runs at runtime.

The three build-time narrative invariants (``core_tension`` / ``narrative_theme`` /
``narrative_pitch``) are defined on ``world.models.ThemeAnalysis``: hard anchors
during build, deliberately **not** consumed by runtime cognition prompts.
"""

from world.catalog import DEFAULT_CATALOG_PATH, WorldCatalog
from world.models import (
    AgentDefinition,
    AgentTier,
    HistoricalEventSeed,
    LocationSeed,
    RelationSeed,
    ThemeAnalysis,
    ThemeFigure,
    World,
    WorldEntity,
    WorldEntitySeed,
    WorldEntityType,
)

__all__ = [
    "AgentDefinition",
    "AgentTier",
    "HistoricalEventSeed",
    "LocationSeed",
    "RelationSeed",
    "ThemeAnalysis",
    "ThemeFigure",
    "World",
    "WorldBuilder",
    "DEFAULT_CATALOG_PATH",
    "WorldCatalog",
    "WorldEntity",
    "WorldEntitySeed",
    "WorldEntityType",
    "WorldInitializer",
]


def __getattr__(name: str):
    """Lazy import builder and initializer to avoid circular imports."""

    if name == "WorldBuilder":
        from world.builder import WorldBuilder

        return WorldBuilder
    if name == "WorldInitializer":
        from world.initializer import WorldInitializer

        return WorldInitializer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
