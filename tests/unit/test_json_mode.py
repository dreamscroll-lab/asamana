"""A call that PARSES JSON must ASK for JSON.

Without json_mode the failures are not truncation: the model emits garbage in value position
(``"estimated_steps": I,`` / ``, 1`` / ``blocked content``), and each one costs an agent its step,
because an unparseable decision degrades to "do nothing" (CLAUDE.md Rule 1). json_mode masks those
tokens during sampling, so the malformed payload is never generated.

Two things are pinned here, and the second is the one that bites:
  1. Every LLM call site in agent/ engine/ world/ that parses its reply asks for json_mode.
     Forgetting is silent: the failure rate simply comes back.
  2. The OpenAI-compatible endpoint REJECTS response_format with a 400 unless the prompt
     contains the word "json". Every runtime caller would swallow that as "the LLM failed", so the
     provider raises instead: asking for JSON without telling the model is a bug, not a
     degradation (Rule 3).
"""

from __future__ import annotations

import ast
import asyncio
import glob
import re

import pytest

from core.interfaces.llm import LLMMessage
from providers.llm.openai_compat import OpenAICompatProvider

# The one call in the sim that wants prose. Everything else parses its reply.
FREE_TEXT = {("agent/memory.py", "MEMORY_SUMMARIZATION")}


def _llm_call_sites() -> list[tuple[str, int, str, bool, bool]]:
    """(file, line, scene, parses_json, asks_for_json) for every LLM call in the sim."""
    sites = []
    for f in sorted(
        glob.glob("agent/**/*.py", recursive=True)
        + glob.glob("engine/**/*.py", recursive=True)
        + glob.glob("world/**/*.py", recursive=True)
    ):
        src = open(f).read()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute) or fn.attr not in ("complete", "complete_with_retry"):
                continue
            if not node.args:
                continue
            # Match on the receiver (some llm router), not the first-arg shape: matching only
            # `LLMScene.X` would let a call site that passes the scene as a variable slip past.
            if "llm" not in ast.unparse(fn.value):
                continue
            first = node.args[0]
            lines = src.splitlines(keepends=True)
            after = sum(len(line) for line in lines[: node.end_lineno])
            nxt = re.search(r"\n    (?:async )?def ", src[after:])
            body = src[after : after + (nxt.start() if nxt else 4000)]
            parses = bool(re.search(r"extract_json|self\._parse\(|_parse_\w*\(\s*(response|resp)\.content", body))
            asks = any(k.arg == "json_mode" for k in node.keywords)
            scene = first.attr if isinstance(first, ast.Attribute) else ast.unparse(first)
            sites.append((f, node.lineno, scene, parses, asks))
    return sites


def test_every_json_parsing_call_asks_for_json() -> None:
    offenders = [
        f"{f}:{line} {scene}"
        for f, line, scene, parses, asks in _llm_call_sites()
        if parses and not asks and (f, scene) not in FREE_TEXT
    ]
    assert not offenders, (
        "these calls parse the reply as JSON but never asked the model to produce JSON — "
        f"they will keep silently losing agent-steps: {offenders}"
    )


def test_the_free_text_call_is_the_only_one_not_asking() -> None:
    # If this list grows, someone added a prose call (fine — declare it) or forgot json_mode
    # on a JSON call (not fine). Either way it must be a decision, not a drift.
    silent = {
        (f, scene) for f, _line, scene, _parses, asks in _llm_call_sites() if not asks
    }
    assert silent == FREE_TEXT, f"unexpected calls not asking for JSON: {silent - FREE_TEXT}"


def test_the_site_detector_is_not_vacuous() -> None:
    # A guard that inspects nothing passes trivially. It must actually be finding call sites.
    sites = _llm_call_sites()
    assert len(sites) > 20
    assert sum(1 for *_, parses, _ in sites if parses) > 20


def test_asking_for_json_without_telling_the_model_is_a_bug_not_a_degradation() -> None:
    # The precondition under test is checked before any request is built, so no network.
    provider = OpenAICompatProvider(
        model="deepseek-v3.2", base_url="https://example.invalid/v1", api_key="test-key",
    )
    with pytest.raises(ValueError, match="json"):
        # Sent as-is, the endpoint would 400; the exception would land in some caller's
        # `except`, and a plain bug would wear the costume of a flaky model forever.
        asyncio.run(
            provider.complete([LLMMessage(role="user", content="讲个笑话")], json_mode=True)
        )
