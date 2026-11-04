"""Memory contracts shared below ``agent/``: what kind of record a memory is."""

from __future__ import annotations

from enum import Enum


class MemoryKind(str, Enum):
    """EVENT is something the agent went through, INSIGHT what reflection drew from several
    events, SUMMARY what compression made of a period."""

    EVENT = "event"
    INSIGHT = "insight"
    SUMMARY = "summary"
