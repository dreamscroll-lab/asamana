"""Development-phase tuning toolkit for Asamana.

This package is *not* part of the production runtime. It captures the
intermediate inputs/outputs of long pipelines (world building first, per-step
cognition later) so they can be browsed and scored during prompt tuning.

Separation contract: ``tuning`` depends on production code, never the reverse.
Deleting this package leaves the production runtime fully functional. The only
production-side seam is the optional ``on_phase`` observation hook on
``WorldBuilder.build()`` (default ``None`` — production never passes it).
"""
