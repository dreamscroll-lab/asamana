"""Dependency injection container for Asamana.

Holds the assembled providers by their ``core/interfaces`` contracts. Assembly from config lives in
``providers.assembly.build_container``, so ``core`` never imports a concrete provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.event_bus import NarrativeEventBus
from core.interfaces.agent_store import AgentStoreProvider
from core.interfaces.embedding import EmbeddingProvider
from core.interfaces.llm import LLMProvider, LLMRouter
from core.interfaces.message import MessageProvider
from core.interfaces.snapshot import SnapshotProvider
from core.interfaces.trace import TraceSink
from core.interfaces.vector_store import VectorStoreProvider


@dataclass
class Container:
    """Application dependency container."""

    llm_router: LLMRouter
    embedding: EmbeddingProvider
    vector_store: VectorStoreProvider
    agent_store: AgentStoreProvider
    message_provider: MessageProvider
    snapshot: SnapshotProvider
    event_bus: NarrativeEventBus
    trace_sink: TraceSink
    # False: started without model keys, so ``llm_router`` and ``embedding`` raise
    # ``ModelKeysMissing`` on use (see ``providers.assembly``).
    model_keys: bool
    # ``create_llm(provider, model, *, timeout=None, **params)``: a one-off provider outside
    # ``llm_router``, for the dev playground: untraced, and ungated, since rate gates live on the
    # router's per-scene path.
    create_llm: Callable[..., LLMProvider]

    def describe(self) -> dict[str, str]:
        return {
            "llm_router": type(self.llm_router).__name__,
            "embedding_provider": type(self.embedding).__name__,
            "vector_store_provider": type(self.vector_store).__name__,
            "agent_store_provider": type(self.agent_store).__name__,
            "message_provider": type(self.message_provider).__name__,
            "snapshot_provider": type(self.snapshot).__name__,
            "trace_sink": type(self.trace_sink).__name__,
        }
