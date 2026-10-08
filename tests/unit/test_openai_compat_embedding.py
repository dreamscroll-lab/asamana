"""Embedding and LLM are independent axes: give endpoint + key + model and it connects."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from core.factory import ComponentKind, ProviderFactory
from providers.embedding.openai_compat import OpenAICompatEmbeddingProvider


@dataclass
class _Datum:
    index: int
    embedding: list[float]


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [_Datum(index=i, embedding=[float(i)] * 3) for i in range(len(kwargs["input"]))]
        return type("Resp", (), {"data": data})()


def _provider(monkeypatch, **overrides) -> OpenAICompatEmbeddingProvider:
    monkeypatch.setenv("SOME_PROVIDER_KEY", "k")
    provider = OpenAICompatEmbeddingProvider(**{
        "model": "text-embedding-3-large",
        "dimension": 3,
        "base_url": "https://api.example.com/v1",
        "api_key_env": "SOME_PROVIDER_KEY",
        **overrides,
    })
    provider._client = type("C", (), {"embeddings": _FakeEmbeddings()})()
    return provider


def test_registered_so_config_can_select_it() -> None:
    assert "openai_compat" in ProviderFactory.registered_names(kind=ComponentKind.EMBEDDING)


def test_endpoint_key_and_model_are_all_it_needs(monkeypatch) -> None:
    monkeypatch.setenv("SOME_PROVIDER_KEY", "k")
    provider = OpenAICompatEmbeddingProvider(
        model="text-embedding-3-large", dimension=3072,
        base_url="https://api.example.com/v1", api_key_env="SOME_PROVIDER_KEY",
    )

    assert provider.dimension == 3072
    assert provider.api_key_env == "SOME_PROVIDER_KEY"
    assert provider._client.api_key == "k"
    assert str(provider._client.base_url).rstrip("/") == "https://api.example.com/v1"


def test_a_missing_key_fails_fast_and_names_the_variable(monkeypatch) -> None:
    monkeypatch.delenv("SOME_PROVIDER_KEY", raising=False)

    with pytest.raises(ValueError, match="SOME_PROVIDER_KEY"):
        OpenAICompatEmbeddingProvider(
            model="m", dimension=3, base_url="https://x.invalid/v1",
            api_key_env="SOME_PROVIDER_KEY",
        )


def test_a_text_embeds_dense_only(monkeypatch) -> None:
    provider = _provider(monkeypatch)

    out = asyncio.run(provider.embed("a"))

    assert out.dense == [0.0, 0.0, 0.0]
    # Compatible endpoints give no learned sparse vectors; hybrid retrieval degrades to pure dense,
    # which is the normal path, not an error.
    assert out.sparse is None


def test_endpoint_params_pass_through_verbatim(monkeypatch) -> None:
    """Keys like ``dimensions`` belong to the endpoint and pass through uninterpreted, same rule as
    the LLM side."""
    provider = _provider(monkeypatch, dimensions=1024)

    asyncio.run(provider.embed("a"))

    assert provider._client.embeddings.calls[0]["extra_body"] == {"dimensions": 1024}


def test_a_repeated_single_text_is_not_embedded_twice(monkeypatch) -> None:
    provider = _provider(monkeypatch)

    first = asyncio.run(provider.embed("同一句话"))
    second = asyncio.run(provider.embed("同一句话"))

    assert first.dense == second.dense
    assert len(provider._client.embeddings.calls) == 1
    # The cache hands out copies on both ends: mutating a return value must not poison the next hit.
    second.dense[0] = 99.0
    assert asyncio.run(provider.embed("同一句话")).dense[0] != 99.0


def test_every_registered_embedding_says_whether_it_bills_to_an_account() -> None:
    """A networked provider missing ``api_key_env`` silently loses rate-limit backpressure: it
    hits an account at full speed and the limiter never knows. So every embedding implementation
    must declare itself in this table."""
    import inspect

    from core.factory import ComponentKind, ProviderFactory

    expected = {"dashscope": True, "openai_compat": True, "in_memory": False}
    registry = ProviderFactory._registry[ComponentKind.EMBEDDING]

    assert set(registry) == set(expected), "新增了 embedding 实现就更新这张表"
    for name, networked in expected.items():
        takes_key = "api_key_env" in inspect.signature(registry[name].__init__).parameters
        assert takes_key is networked, f"{name}: api_key_env 的有无与「是否联网」对不上"


def test_a_local_provider_names_no_account() -> None:
    """The contract defaults to ``None``, so local implementations correctly get no gate without
    doing anything."""
    from core.interfaces.embedding import EmbeddingProvider
    from providers.embedding.in_memory import InMemoryEmbeddingProvider

    assert EmbeddingProvider.api_key_env is None
    assert InMemoryEmbeddingProvider(dimension=4).api_key_env is None
