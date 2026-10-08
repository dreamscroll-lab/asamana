"""Developer tooling routes: LLM traces, post-hoc audit, stage suites, workbench.

These are server-side developer instruments (observability), not user-facing
world observation. They stay on the backend and leave the observed world
untouched — reports are produced by the ``python -m tuning`` CLI, and the
director console parses without injecting.

The front end is the SPA's ``#/dev`` screen; no HTML is served from here.
"""

from __future__ import annotations

from typing import Any

from interaction.api.app import ApiServices
from interaction.api.dev.audit import build_audit_router
from interaction.api.dev.director_console import build_director_console_router
from interaction.api.dev.jobs import JobRegistry, build_jobs_router
from interaction.api.dev.playground import build_playground_router
from interaction.api.dev.recall import build_recall_router
from interaction.api.dev.stages import build_stages_router
from interaction.api.dev.trace import build_trace_router


def build_dev_router(services: ApiServices) -> Any:
    from fastapi import APIRouter

    router = APIRouter()
    jobs = JobRegistry()
    router.include_router(build_trace_router(services))
    router.include_router(build_audit_router(services, jobs))
    router.include_router(build_jobs_router(jobs))
    router.include_router(build_stages_router(services, jobs))
    router.include_router(build_playground_router(services))
    router.include_router(build_director_console_router(services))
    router.include_router(build_recall_router(services))
    return router
