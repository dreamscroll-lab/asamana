"""The world-building stages ``world.builder.WorldBuilder`` runs — one module per LLM stage."""

# Rule 2: a world-building LLM call gets one retry on an unusable reply, then raises.
BUILD_ATTEMPTS = 2
