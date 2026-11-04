"""Assemble a ``Container`` from config: the composition root for providers.

It lives here, not in ``core``, because it names concrete providers; ``core`` holds only the
container's contract-typed shape.
"""

from __future__ import annotations

import inspect
import os
from importlib import import_module

from config.models import Config, LLMProviderConfig
from core.container import Container
from core.event_bus import NarrativeEventBus
from core.factory import ComponentKind, ProviderFactory
from core.interfaces.embedding import EmbeddingProvider
from core.interfaces.llm import LLMProvider, LLMRouter, LLMScene
from core.interfaces.trace import TraceSink
from core.logging import get_logger
from core.rate_gate import RateGate
from providers.embedding.keyless import KeylessEmbeddingProvider
from providers.llm.catalog import Endpoint, create_llm, resolve_catalog
from providers.llm.keyless import KeylessLLMProvider

logger = get_logger(__name__)

_BUILTIN_MODULES = (
    "providers.embedding",
    "providers.llm",
    "providers.message",
    "providers.snapshot",
    "providers.trace",
    "providers.vector_store",
)
_BUILTINS_REGISTERED = False


def build_container(config: Config) -> Container:
    """Assemble every provider the config selects."""
    _register_builtin_components()
    trace_sink = _build_trace_sink(config)
    # Per assembly, not a module-level singleton: with several worlds in one process, the next
    # config would overwrite a global table and silently switch endpoints.
    catalog = resolve_catalog(config.llm.providers)
    rate_gates = _build_rate_gates(config)
    model_keys = _model_keys_present(config, catalog)
    embedding: EmbeddingProvider
    if model_keys:
        embedding = _create_embedding(config)
        if _is_networked(embedding):
            from providers.embedding.gated import GatedEmbeddingProvider

            embedding = GatedEmbeddingProvider(embedding, _embedding_gate(config))
        llm_router = _build_llm_router(
            config, catalog, trace_sink=trace_sink, rate_gates=rate_gates,
        )
    else:
        embedding = KeylessEmbeddingProvider()
        llm_router = LLMRouter(
            {scene: KeylessLLMProvider() for scene in LLMScene}, trace_sink=trace_sink,
        )
    return Container(
        trace_sink=trace_sink,
        model_keys=model_keys,
        create_llm=lambda provider, model, **params: create_llm(provider, model, catalog, **params),
        llm_router=llm_router,
        embedding=embedding,
        vector_store=ProviderFactory.create(
            config.vector_store.provider,
            kind=ComponentKind.VECTOR_STORE,
            **config.vector_store.params,
        ),
        message_provider=ProviderFactory.create(
            config.message.provider,
            kind=ComponentKind.MESSAGE,
            **config.message.params,
        ),
        snapshot=ProviderFactory.create(
            config.snapshot.provider,
            kind=ComponentKind.SNAPSHOT,
            **config.snapshot.params,
        ),
        event_bus=NarrativeEventBus(),
    )


def _model_keys_present(config: Config, catalog: dict[str, Endpoint]) -> bool:
    """Are the keys this config uses all set (True) or all unset (False)? Partly set raises.

    The set is what the config actually reaches: the endpoints behind every scene and the judge,
    plus a networked embedding's key. A half-keyed deployment would build worlds and then fail
    inside them, so it is a config error, not a mode.
    """
    providers = {config.llm.scene_llm(scene).provider for scene in LLMScene}
    providers.add(config.llm.judge_llm().provider)
    names = {catalog[name].api_key_env for name in providers if name in catalog}
    # A networked embedding takes ``api_key_env``, a local one doesn't (the embedding tests hold
    # every registration to this).
    embedding_cls = ProviderFactory.registration(
        config.embedding.provider, kind=ComponentKind.EMBEDDING,
    )
    key_param = inspect.signature(embedding_cls.__init__).parameters.get("api_key_env")
    if key_param is not None:
        names.add(config.embedding.params.get("api_key_env", key_param.default))
    missing = sorted(name for name in names if not os.environ.get(name))
    if missing and len(missing) < len(names):
        raise ValueError(
            "Model keys must be all set or all unset; missing: " + ", ".join(missing)
        )
    if missing:
        logger.warning("model_keys_missing", extra={"key_vars": missing})
    return not missing


def _register_builtin_components() -> None:
    global _BUILTINS_REGISTERED

    if _BUILTINS_REGISTERED:
        return

    for module_name in _BUILTIN_MODULES:
        import_module(module_name)

    _BUILTINS_REGISTERED = True


def _build_rate_gates(config: Config) -> dict[str, RateGate]:
    """One gate per **endpoint** in use, keyed by its ``llm.providers`` entry name.

    Vendors cap RPM per model and an entry names one model, so the entry, not the account, is
    what a limit attaches to: two entries on one credential don't stall each other. ``rpm`` /
    ``tpm`` fall back to the ``rate_limit`` defaults. A provider with no entry (``mock``) takes
    no gate.
    """

    def _limited(declared: "LLMProviderConfig") -> RateGate:
        return RateGate(
            base_cooldown=config.rate_limit.cooldown_base,
            max_cooldown=config.rate_limit.cooldown_max,
            max_retry_after=config.rate_limit.retry_after_max,
            rpm_limit=config.rate_limit.rpm if declared.rpm is None else declared.rpm,
            tpm_limit=config.rate_limit.tpm if declared.tpm is None else declared.tpm,
        )

    gates: dict[str, RateGate] = {}
    for scene in LLMScene:
        name = config.llm.scene_llm(scene).provider
        declared = config.llm.providers.get(name)
        if declared is not None and name not in gates:
            gates[name] = _limited(declared)
    return gates


def _embedding_gate(config: Config) -> RateGate:
    """The embedding endpoint's own gate; ``reserve_rate=False`` makes it cooldown-only.

    Keep it out of the ``_build_rate_gates`` table: an "embedding" key would collide with an LLM
    endpoint of that name and silently leave it cooldown-only.
    """
    return RateGate(
        base_cooldown=config.rate_limit.cooldown_base,
        max_cooldown=config.rate_limit.cooldown_max,
        max_retry_after=config.rate_limit.retry_after_max,
    )


def _build_llm_router(
    config: Config,
    catalog: dict[str, Endpoint],
    *,
    trace_sink: TraceSink | None = None,
    rate_gates: dict[str, RateGate] | None = None,
) -> LLMRouter:
    gates = rate_gates or {}
    scene_providers: dict[LLMScene, LLMProvider] = {}
    scene_gates: dict[LLMScene, RateGate] = {}
    for scene in LLMScene:
        chosen = config.llm.scene_llm(scene)
        # ``mock`` uses exact registration (``MockLLMProvider(**kwargs)``), so an unknown keyword
        # raises TypeError on the spot: a scene with no endpoint params must pass no extra kwargs.
        scene_providers[scene] = create_llm(
            chosen.provider, chosen.model, catalog,
            timeout=chosen.timeout, **config.llm.params_for(scene),
        )
        if chosen.provider in gates:
            scene_gates[scene] = gates[chosen.provider]
    return LLMRouter(
        scene_providers,
        max_concurrent=config.engine.max_concurrent_llm,
        trace_sink=trace_sink,
        rate_gates=scene_gates,
        capture_thinking=config.observability.capture_thinking,
    )


def _create_embedding(config: Config) -> EmbeddingProvider:
    """The embedding provider itself, ungated — the caller asks whether it is networked first."""

    return ProviderFactory.create(
        config.embedding.provider,
        kind=ComponentKind.EMBEDDING,
        **config.embedding.params,
    )


def _is_networked(embedding: EmbeddingProvider) -> bool:
    """Does this embedding hit a remote endpoint (and so need a gate)?

    Asked of the provider, not a table here: a new networked provider just sets ``api_key_env``,
    which the ``EmbeddingProvider`` contract requires.
    """

    return embedding.api_key_env is not None


def _build_trace_sink(config: Config) -> TraceSink:
    """Build the observability trace sink. Disabled → NullTraceSink (no overhead)."""

    if not config.observability.enabled:
        return ProviderFactory.create("null", kind=ComponentKind.TRACE)
    return ProviderFactory.create(
        config.observability.provider,
        kind=ComponentKind.TRACE,
        **config.observability.params,
    )
