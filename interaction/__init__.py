"""Interaction layer helpers.

Read-only toward the world (it can start and stop one, not touch it), except ``api.direct``,
where a human reaches into a running world (see ``engine.director``). Anything here that
renders a world reports its state; it never owns it.
"""
