"""Backgrounded ``python -m tuning …`` subprocesses, shared by the audit and stage-suite routes."""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)

# Project root = the dir that holds the `tuning` package, so every subprocess
# (`python -m tuning …`) resolves the module. This file is interaction/api/dev/jobs.py.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _Job:
    """One backgrounded `python -m tuning …` subprocess.

    Production never imports tuning; the `interaction → tuning` boundary stays a subprocess
    call. ``kind`` is also the concurrency key: at most one job per (kind, world) runs at a time.
    """

    job_id: str
    kind: str  # "audit" | "stage:<key>"
    world_id: str
    cmd: list[str]
    status: str = "running"  # running | completed | failed
    returncode: int | None = None
    log_tail: list[str] = field(default_factory=list)  # last ~200 subprocess stdout lines
    # The job's background task. The event loop holds tasks only weakly, so an unreferenced
    # task can be GC'd mid-run; the job keeps the reference and lives until process exit.
    task: "asyncio.Task[None] | None" = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "world_id": self.world_id,
            "cmd": " ".join(self.cmd[2:]),  # drop the interpreter + -m
            "status": self.status,
            "returncode": self.returncode,
            "log_tail": self.log_tail,
        }


class JobRegistry:
    """Every job this process started, by id; at most one running per (kind, world)."""

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._job_by_slot: dict[tuple[str, str], str] = {}  # (kind, world_id) → job_id

    def get(self, job_id: str) -> _Job | None:
        return self._jobs.get(job_id)

    def spawn(self, kind: str, world_id: str, cmd: list[str]) -> dict[str, Any]:
        """Start a job in the (kind, world) slot, refusing a second one while it runs."""
        from fastapi import HTTPException

        prior = self._job_by_slot.get((kind, world_id))
        if prior and self._jobs.get(prior) and self._jobs[prior].status == "running":
            raise HTTPException(status_code=409, detail=f"A {kind} job is already running for this world")
        job = _Job(job_id=uuid.uuid4().hex, kind=kind, world_id=world_id, cmd=cmd)
        self._jobs[job.job_id] = job
        self._job_by_slot[(kind, world_id)] = job.job_id
        job.task = asyncio.create_task(_run_job(job), name=f"{kind}:{job.job_id}")
        return job.as_dict()


async def _run_job(job: _Job) -> None:
    """Drive one `python -m tuning …` subprocess, streaming stdout into log_tail.

    Inherits the server's env (the LLM key is already present — world runs need it),
    so no key is typed into the UI. The subprocess writes its report to disk; the
    client re-fetches the report route on completion.
    """
    logger.info(
        "dev_job_start",
        extra={"world_id": job.world_id, "job_id": job.job_id, "kind": job.kind,
               "cmd": " ".join(job.cmd)},
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *job.cmd,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        # Read raw chunks rather than readline: at DEBUG the child's httpx client logs a whole
        # request body as one line, which overruns StreamReader's 64 KB line limit and aborts the job.
        def _record(text: str) -> None:
            job.log_tail.append(text[:2000])  # cap: a debug line can be MBs
            if len(job.log_tail) > 200:  # cap: keep only the tail
                del job.log_tail[: len(job.log_tail) - 200]

        buf = b""
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            *complete, buf = buf.split(b"\n")
            for raw in complete:
                _record(raw.decode("utf-8", "replace").rstrip())
        if buf:  # trailing line with no final newline
            _record(buf.decode("utf-8", "replace").rstrip())
        await proc.wait()
        job.returncode = proc.returncode
        job.status = "completed" if proc.returncode == 0 else "failed"
        logger.info(
            "dev_job_done",
            extra={"world_id": job.world_id, "job_id": job.job_id, "kind": job.kind,
                   "returncode": proc.returncode},
        )
    except Exception as exc:  # noqa: BLE001 — a job failure must not crash the server
        job.status = "failed"
        job.log_tail.append(f"[job error] {exc}")
        logger.error("dev_job_failed",
                     extra={"world_id": job.world_id, "kind": job.kind, "error": str(exc)})


def build_jobs_router(jobs: JobRegistry) -> Any:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(tags=["dev"])

    @router.get("/api/dev/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job.as_dict()

    return router
