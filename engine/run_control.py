"""Cooperative run control for the runtime loop — pause / resume / stop.

The runtime loop cooperates with a :class:`RunController` at each step boundary:
it awaits :meth:`RunController.wait_if_paused` before running a step and checks
:attr:`RunController.stop_requested` to break early. Because every step persists
its snapshot at step end, pausing or stopping on a step boundary loses no step —
the world is always resumable from the last persisted step.

``RunState`` is the world-level run lifecycle the orchestration layer maintains
per session and the API layer reports; the controller itself only distinguishes
running-vs-paused plus a stop flag.
"""

from __future__ import annotations

import asyncio
from enum import Enum


class RunState(str, Enum):
    IDLE = "idle"          # session exists, no run loop active
    RUNNING = "running"    # run loop advancing steps
    PAUSED = "paused"      # run loop parked at a step boundary
    STOPPING = "stopping"  # stop requested, loop will end at next boundary
    COMPLETED = "completed"  # finished its requested steps or stopped cleanly
    FAILED = "failed"      # run loop raised


class RunController:
    """Step-boundary pause/resume/stop signal for one world's run loop.

    Single-process, single event loop: all mutation happens on the asyncio loop
    thread, so no locking is required.
    """

    def __init__(self) -> None:
        # Set = free to run; cleared = paused. Starts runnable.
        self._resume = asyncio.Event()
        self._resume.set()
        self._stop_requested = False
        # One-shot: let exactly one more step through, then park again. See step_once.
        self._single_step = False

    @property
    def is_paused(self) -> bool:
        return not self._resume.is_set() and not self._stop_requested

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def pause(self) -> None:
        """Park the loop at the next step boundary (no-op once stopping)."""
        if not self._stop_requested:
            self._resume.clear()

    def resume(self) -> None:
        """Release a paused loop (no-op once stopping)."""
        if not self._stop_requested:
            self._resume.set()

    def step_once(self) -> None:
        """Let exactly one more step through, then park again (no-op once stopping).

        The director's rhythm: inject, advance ONE step, see what it caused. It also keeps the
        render-gated observer feed in lockstep with a world that would otherwise run ahead.
        """
        if self._stop_requested:
            return
        self._single_step = True
        self._resume.set()

    def stop(self) -> None:
        """Request the loop to end at the next boundary; unblock if paused."""
        self._stop_requested = True
        self._single_step = False
        self._resume.set()  # release any wait_if_paused waiter so the loop can exit

    async def wait_if_paused(self) -> None:
        """Block while paused; returns immediately when running or stopping.

        The one-shot latch is consumed HERE, after the wait and before the step: that is what
        makes it exactly one step.
        """
        await self._resume.wait()
        if self._single_step and not self._stop_requested:
            self._single_step = False
            self._resume.clear()
