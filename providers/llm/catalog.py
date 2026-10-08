"""LLM endpoint catalog: turns each provider's endpoint declared in config into an instance.

OpenAI-compatible vendors differ only in base_url and key env var, so they all share the
``OpenAICompatProvider`` transport and adding one needs no code.

- Config is the only source. Don't keep a list of common vendors in code: it would give a
  second answer to "where does this name connect". Common settings live as comments in
  ``config/config.example.yaml`` and ``docs/configuration.md``, where going stale doesn't change
  behavior.
- Don't add per-vendor SDKs: ``core.rate_gate.classify_rate_limit`` duck-types
  ``status_code`` / ``response``, so another exception type would silently disable the rate
  gate for that vendor.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.factory import ComponentKind, ProviderFactory
from core.interfaces.llm import LLMProvider
from providers.llm.openai_compat import OpenAICompatProvider

@dataclass(frozen=True)
class Endpoint:
    """How to reach one endpoint: URL, key env var, and whether it accepts temperature."""

    base_url: str
    api_key_env: str
    accepts_temperature: bool = True


#: LLM providers handled directly by ``ProviderFactory`` rather than the catalog: ``mock`` is a
#: registered test double, not an endpoint. Config can't declare it as a compatible endpoint;
#: shadowing it would silently switch the simulation to a real endpoint, or run production on
#: canned replies.
_FACTORY_PROVIDERS = frozenset({"mock"})

def resolve_catalog(providers: Mapping[str, Any] | None) -> dict[str, Endpoint]:
    """Map ``llm.providers`` to the endpoints this deployment can reach.

    ``api_key_env`` has a default (``config.models.DEFAULT_LLM_KEY_ENV``), so several providers
    all relying on it is a config error: one variable can't hold two keys, and one provider
    would authenticate with another's credentials and fail with an auth error that doesn't
    point at the config. Naming the same variable explicitly is allowed, since that states the
    intent (e.g. a gateway that really shares credentials).
    """
    catalog: dict[str, Endpoint] = {}
    defaulted: list[str] = []
    for name, declared in (providers or {}).items():
        if name in _FACTORY_PROVIDERS:
            raise ValueError(
                f"llm.providers.{name!r} is reserved: {name!r} is handled by ProviderFactory "
                "and cannot be declared as an OpenAI-compatible endpoint"
            )
        catalog[name] = Endpoint(
            declared.base_url, declared.api_key_env, declared.accepts_temperature,
        )
        if "api_key_env" not in getattr(declared, "model_fields_set", {"api_key_env"}):
            defaulted.append(name)
    if len(defaulted) > 1:
        shared = catalog[defaulted[0]].api_key_env
        raise ValueError(
            f"llm providers {', '.join(sorted(defaulted))} all fall back to {shared!r}; "
            "one variable cannot hold several accounts' keys — give each its own 'api_key_env'"
        )
    return catalog


def create_llm(
    provider: str,
    model: str,
    catalog: Mapping[str, Endpoint],
    *,
    timeout: float | None = None,
    **params: Any,
) -> LLMProvider:
    """Create an LLM for a provider and model; the only place LLMs are created.

    A name in the catalog is an OpenAI-compatible endpoint; anything else goes to
    ``ProviderFactory`` (the ``mock`` registration).

    ``timeout`` is how long we wait and is kept apart from ``params``, which are all sent to the
    endpoint. ``None`` uses the transport default.
    """
    endpoint = catalog.get(provider)
    if endpoint is None:
        if provider not in _FACTORY_PROVIDERS:
            # Config is the only way to add a provider, so this message doubles as its docs: list
            # the known providers and how to declare a new one, not just "Unknown".
            raise ValueError(
                f"Unknown LLM provider {provider!r}. Declared providers: "
                f"{', '.join(sorted(catalog)) or '(none)'}; plus "
                f"{', '.join(sorted(_FACTORY_PROVIDERS))}. To add {provider!r}, give it a "
                "'base_url' under llm.providers."
            )
        return ProviderFactory.create(provider, kind=ComponentKind.LLM, **params)
    if not model:
        raise ValueError(f"llm provider {provider!r} must name a 'model'")
    api_key = os.environ.get(endpoint.api_key_env)
    if not api_key:
        raise ValueError(f"{endpoint.api_key_env} is not set (required by provider {provider!r})")
    kwargs: dict[str, Any] = {
        "call_params": params,
        "accepts_temperature": endpoint.accepts_temperature,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAICompatProvider(
        model=model, base_url=endpoint.base_url, api_key=api_key, **kwargs,
    )
