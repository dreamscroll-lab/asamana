"""Interaction layer: world management, observation, replay, the CLI and the REST + WebSocket API.

Read-only toward a world's simulation, except where a person acts on it through a documented
route: lifecycle (build, confirm, run control, delete), the setup revised before confirmation
(``api.setup``), the director reaching into a running world (``api.direct``, see
``engine.director``) and the actor performing one character (``api.actor``). Anything here that
renders a world reports its state; it never owns it.
"""
