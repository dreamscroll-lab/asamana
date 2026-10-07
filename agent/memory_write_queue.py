"""Serial background-write worker for a MemorySystem.

Drains already-created write coroutines (memory refine / persist) in order on one worker
task. It owns only its queue and worker; it never touches the memory store, embedding, or LLM.
"""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

from core.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def _entered(ctx: contextvars.Context) -> "Iterator[None]":
    """Apply every contextvar in *ctx* in the *current* frame; restore each on exit.

    Don't use ``Context.run`` or ``create_task`` with a context: the former can't run a
    coroutine, and the latter adds a scheduling hop, making it likelier a job is dropped
    before it starts at shutdown. The worker starts from an empty context (see
    ``_ensure_worker``), so this only layers values on; it needn't clear variables absent
    from ctx — whatever the previous job set is restored by that job's own finally.
    """
    tokens = [(var, var.set(value)) for var, value in ctx.items()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class MemoryWriteQueue:
    """Single-worker FIFO queue for a memory system's background writes."""

    def __init__(self, *, agent_id: str) -> None:
        self._agent_id = agent_id
        self._queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self._worker: "asyncio.Task[None] | None" = None

    def enqueue(self, coro: "Any") -> None:
        """Enqueue a write coroutine with its enqueue-time context (agent, step, execution).

        Without it every job would run under the first enqueuer's contextvars, since the worker
        is one long-lived task. Carry the whole context, not chosen fields: the next contextvar
        someone adds would break the same way.
        """
        self._ensure_worker()
        self._queue.put_nowait((coro, contextvars.copy_context()))

    async def drain(self) -> None:
        """Barrier before maintenance (compression/reflection): they must never read
        half-finished in-flight writes (phantom / orphan sources)."""
        await self._queue.join()

    async def close(self) -> None:
        """Drain and stop the worker (for run() teardown: no lost tail, no dangling task)."""
        if self._worker is None or self._worker.done():
            return
        await self._queue.join()
        self._queue.put_nowait(None)  # stop sentinel
        await self._worker
        self._worker = None

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            # Start the worker with an empty context: ``create_task`` would otherwise freeze the
            # creator's contextvars into a task that runs everyone's jobs. Each job applies its
            # own enqueue-time context instead (see _worker_loop).
            self._worker = asyncio.create_task(self._worker_loop(), context=contextvars.Context())

    async def _worker_loop(self) -> None:
        """Await jobs in order. Failures are logged and swallowed (Rule 1: the provisional
        entry is already in _entries, so a failed write must not kill the run)."""
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                job, ctx = item
                with _entered(ctx):

                    await job
            except Exception as exc:  # noqa: BLE001 — Rule 1
                logger.warning(
                    "memory_refine_failed",
                    extra={"agent_id": self._agent_id, "error": str(exc)},
                )
            finally:
                self._queue.task_done()
