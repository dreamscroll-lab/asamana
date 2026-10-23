"""World construction package.

This package owns the build-stage narrative pipeline (theme → world) and the world model
(``world.models``) the runtime engine reads; the build pipeline itself never runs at runtime.

The three build-time narrative invariants (``core_tension`` / ``narrative_theme`` /
``narrative_pitch``) are defined on ``world.models.ThemeAnalysis``: hard anchors
during build, deliberately **not** consumed by runtime cognition prompts.
"""

from world.builder import WorldBuilder
from world.catalog import DEFAULT_CATALOG_PATH, ActorCasting, WorldCatalog
from world.initializer import WorldInitializer
from world.models import World

__all__ = [
    "ActorCasting",
    "DEFAULT_CATALOG_PATH",
    "World",
    "WorldBuilder",
    "WorldCatalog",
    "WorldInitializer",
]
