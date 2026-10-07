"""Blocking IO in the file backends may live only in sync helpers, never in an async method body.

One event loop serves every world, so one world reading its relation files stalls all others,
including their LLM callbacks. Reads, writes and whole-world deletes go through ``asyncio.to_thread``.

A static AST guard, not a timing assertion (timing is flaky on CI); on violation it names the file,
method, line and call. ``mkdir`` isn't listed: one or two cheap syscalls that don't grow with data,
and moving it would add a thread hop to every save. The list covers what grows with data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROVIDERS = (
    "providers/agent_store/file.py",
    "providers/snapshot/file.py",
    "providers/vector_store/file.py",
    "providers/trace/file.py",
)

_BLOCKING = frozenset({
    "read_text", "write_text", "read_bytes", "write_bytes", "open",
    "glob", "rglob", "iterdir", "stat", "exists",
    "unlink", "rmtree", "replace",
})

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", _PROVIDERS)
def test_async_methods_do_no_blocking_io_directly(module: str) -> None:
    tree = ast.parse((_ROOT / module).read_text(encoding="utf-8"))
    offences: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _BLOCKING
            ):
                offences.append(f"{module}:{node.lineno} {fn.name}() 直接调用了 {node.func.attr}()")
    assert not offences, (
        "阻塞 IO 留在了 async 方法体里 —— 搬进一个同步 helper,再用 asyncio.to_thread 调它:\n"
        + "\n".join(offences)
    )


def test_the_guard_can_actually_see_a_violation() -> None:
    """Prove the probe can tell the difference before trusting the 0 it reports."""
    tree = ast.parse(
        "import asyncio\n"
        "class P:\n"
        "    async def load(self, path):\n"
        "        return path.read_text()\n"
    )
    found = [
        node.func.attr
        for fn in ast.walk(tree)
        if isinstance(fn, ast.AsyncFunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _BLOCKING
    ]
    assert found == ["read_text"]


def test_passing_a_blocking_method_to_to_thread_is_not_a_violation() -> None:
    """``to_thread(target.read_bytes)`` passes the function rather than calling it, so it isn't a violation."""
    tree = ast.parse(
        "import asyncio\n"
        "class P:\n"
        "    async def load(self, path):\n"
        "        return await asyncio.to_thread(path.read_bytes)\n"
    )
    found = [
        node.func.attr
        for fn in ast.walk(tree)
        if isinstance(fn, ast.AsyncFunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _BLOCKING
    ]
    assert found == []
