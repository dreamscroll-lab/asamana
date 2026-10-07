"""Provider registry and creation helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ComponentKind(str, Enum):
    """Supported factory namespaces."""

    LLM = "llm"
    EMBEDDING = "embedding"
    VECTOR_STORE = "vector_store"
    AGENT_STORE = "agent_store"
    MESSAGE = "message"
    SNAPSHOT = "snapshot"
    WORLD = "world"
    TRACE = "trace"


class ProviderFactory:
    """Scoped registry for providers and world configs."""

    _registry: dict[ComponentKind, dict[str, type]] = {
        kind: {}
        for kind in ComponentKind
    }

    @classmethod
    def register(cls, name: str, *, kind: ComponentKind | str):
        normalized_kind = ComponentKind(kind)

        def decorator(provider_cls: type) -> type:
            contract = _default_contract_for(normalized_kind)
            if contract is not None and not issubclass(provider_cls, contract):
                raise TypeError(
                    f"{provider_cls.__name__} must implement {contract.__name__} "
                    f"to register as {normalized_kind.value}:{name}"
                )

            bucket = cls._registry[normalized_kind]
            existing = bucket.get(name)
            if existing is not None and existing is not provider_cls:
                raise ValueError(
                    f"Duplicate registration for {normalized_kind.value}:{name}"
                )

            bucket[name] = provider_cls
            return provider_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        *,
        kind: ComponentKind | str | None = None,
        **kwargs: Any,
    ) -> Any:
        registration = cls.registration(name, kind=kind)
        try:
            return registration(**kwargs)
        except TypeError as exc:
            scope = f"{ComponentKind(kind).value}:" if kind is not None else ""
            raise TypeError(
                f"Failed to construct {scope}{name} with arguments {sorted(kwargs)}"
            ) from exc

    @classmethod
    def registered_names(cls, *, kind: ComponentKind | str | None = None) -> list[str]:
        if kind is not None:
            return sorted(cls._registry[ComponentKind(kind)])

        names = {
            name
            for bucket in cls._registry.values()
            for name in bucket
        }
        return sorted(names)

    @classmethod
    def registration(
        cls,
        name: str,
        *,
        kind: ComponentKind | str | None = None,
    ) -> type:
        if kind is not None:
            normalized_kind = ComponentKind(kind)
            registration = cls._registry[normalized_kind].get(name)
            if registration is None:
                available = ", ".join(cls.registered_names(kind=normalized_kind))
                raise ValueError(
                    f"Unknown {normalized_kind.value}: '{name}'. "
                    f"Available: {available or '<none>'}"
                )
            return registration

        matches = [
            provider_cls
            for bucket in cls._registry.values()
            for registered_name, provider_cls in bucket.items()
            if registered_name == name
        ]
        if not matches:
            available = ", ".join(cls.registered_names())
            raise ValueError(
                f"Unknown provider: '{name}'. Available providers: {available or '<none>'}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous provider name: '{name}'. Specify a component kind."
            )
        return matches[0]


def _default_contract_for(kind: ComponentKind) -> type | None:
    if kind == ComponentKind.LLM:
        from core.interfaces.llm import LLMProvider

        return LLMProvider
    if kind == ComponentKind.EMBEDDING:
        from core.interfaces.embedding import EmbeddingProvider

        return EmbeddingProvider
    if kind == ComponentKind.VECTOR_STORE:
        from core.interfaces.vector_store import VectorStoreProvider

        return VectorStoreProvider
    if kind == ComponentKind.AGENT_STORE:
        from core.interfaces.agent_store import AgentStoreProvider

        return AgentStoreProvider
    if kind == ComponentKind.MESSAGE:
        from core.interfaces.message import MessageProvider

        return MessageProvider
    if kind == ComponentKind.SNAPSHOT:
        from core.interfaces.snapshot import SnapshotProvider

        return SnapshotProvider
    if kind == ComponentKind.WORLD:
        from core.interfaces.world_config import WorldConfig

        return WorldConfig
    if kind == ComponentKind.TRACE:
        from core.interfaces.trace import TraceSink

        return TraceSink
    return None
