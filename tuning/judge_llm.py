"""The judge LLM: the single construction point shared by all of tuning.

The judge uses its own ``llm.judge`` declaration, separate from the scene it grades: a model that
both generates and grades its own output scores too leniently. Vendor coordinates (``base_url`` /
``api_key_env``) live only in config's endpoint directory, so the judge takes them from there
too; there's no second copy in code.
"""

from __future__ import annotations

from typing import Any

from core.interfaces.llm import LLMProvider
from providers.llm.catalog import create_llm, resolve_catalog


def create_judge(config: Any, model: str | None = None, **params: Any) -> LLMProvider:
    """Build a judge provider from ``llm.judge``; a non-empty ``model`` swaps the model only, not the vendor.

    ``params`` overrides the endpoint's dialect defaults key by key. Don't pass dialect keys (e.g.
    ``enable_thinking``) through it: the call site doesn't know which endpoint it's talking to, so
    that forces one vendor's parameter on all of them. Thinking on/off belongs in
    ``llm.providers.<name>.params`` and ``llm.judge.params``.
    """
    spec = config.llm.judge_llm()
    return create_llm(
        spec.provider,
        model or spec.model,
        resolve_catalog(config.llm.providers),
        timeout=spec.timeout,
        **{**config.llm.judge_params(), **params},
    )
