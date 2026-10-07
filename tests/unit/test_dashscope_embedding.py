"""Unit tests for the DashScope native embedding provider (dense + learned sparse)."""

from __future__ import annotations

import pytest

from core.factory import ComponentKind, ProviderFactory
from providers.embedding.dashscope import DashScopeEmbeddingProvider, _parse_embeddings


def test_parse_embeddings_dense_and_sparse() -> None:
    """Native response → (dense, sparse{index:value})."""
    payload = {
        "output": {
            "embeddings": [
                {"text_index": 0, "embedding": [0.3, 0.4],
                 "sparse_embedding": [{"index": 9, "token": "李", "value": 0.82},
                                      {"index": 5, "token": "暗", "value": 0.5}]},
            ]
        }
    }
    (out,) = _parse_embeddings(payload)
    assert out.dense == [0.3, 0.4]
    assert out.sparse == {9: 0.82, 5: 0.5}


def test_parse_embeddings_no_sparse_is_none() -> None:
    """No sparse_embedding field → sparse=None (graceful fallback)."""
    payload = {"output": {"embeddings": [{"text_index": 0, "embedding": [0.1]}]}}
    out = _parse_embeddings(payload)
    assert out[0].sparse is None


def test_registered_under_embedding_namespace() -> None:
    """Registered as 'dashscope' in the EMBEDDING namespace so config can select it directly."""
    assert "dashscope" in ProviderFactory.registered_names(kind=ComponentKind.EMBEDDING)


def test_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    try:
        DashScopeEmbeddingProvider()
        assert False, "应在缺 key 时抛错"
    except ValueError:
        pass


def test_an_empty_response_raises_instead_of_storing_an_empty_vector(monkeypatch) -> None:
    """On quota or parameter errors the endpoint returns 200 with an error body, which parses to
    no vector. Raise rather than store an empty one (per Rule 1 the caller loses only this
    embedding's recall; the memory itself is kept)."""
    import asyncio

    import pytest

    monkeypatch.setenv("EMBEDDING_API_KEY", "test-key")
    provider = DashScopeEmbeddingProvider()

    class _Resp:
        @staticmethod
        def raise_for_status() -> None: ...

        @staticmethod
        def json() -> dict:
            return {"output": {"embeddings": []}}

    async def _post(*_args, **_kwargs):
        return _Resp()

    monkeypatch.setattr(provider._client, "post", _post)

    with pytest.raises(ValueError, match="0 vectors for 1 text"):
        asyncio.run(provider.embed("甲"))


def _provider_answering(monkeypatch, statuses: list[int | str], *, retry_after: str | None = None):
    """A provider whose endpoint answers with ``statuses`` in turn (200 = one valid embedding,
    "connect" = the connection fails); a rejection carries ``retry_after`` if given."""
    import httpx

    monkeypatch.setenv("EMBEDDING_API_KEY", "k")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        if status == "connect":
            raise httpx.ConnectError("refused", request=request)
        if status != 200:
            headers = {"retry-after": retry_after} if retry_after is not None else {}
            return httpx.Response(status, json={"code": "Throttling"}, headers=headers)
        return httpx.Response(200, json={"output": {"embeddings": [{"text_index": 0, "embedding": [0.5]}]}})

    provider = DashScopeEmbeddingProvider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: SLF001

    async def _no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("providers.embedding.dashscope.asyncio.sleep", _no_wait)
    return provider, calls


@pytest.mark.asyncio
async def test_rate_limit_is_retried_inside_the_provider(monkeypatch, caplog) -> None:
    """429/503 are transient backpressure: the provider retries them (CLAUDE.md Provider Rules),
    and each retry is logged so a run can tell a retried 429 from one that never came."""
    provider, calls = _provider_answering(monkeypatch, [429, 503, 200])
    result = await provider.embed("x")
    assert result.dense == [0.5]
    assert calls == [429, 503, 200]
    retries = [r for r in caplog.records if r.getMessage() == "embedding_retry"]
    assert [(r.attempt, r.reason) for r in retries] == [(1, "429"), (2, "503")]


@pytest.mark.asyncio
async def test_a_failed_connection_is_retried(monkeypatch, caplog) -> None:
    provider, calls = _provider_answering(monkeypatch, ["connect", 200])
    result = await provider.embed("x")
    assert result.dense == [0.5]
    assert calls == ["connect", 200]
    assert [r.reason for r in caplog.records if r.getMessage() == "embedding_retry"] == ["ConnectError"]


@pytest.mark.asyncio
async def test_a_connection_that_keeps_failing_still_raises(monkeypatch) -> None:
    import httpx

    provider, calls = _provider_answering(monkeypatch, ["connect"])
    with pytest.raises(httpx.ConnectError):
        await provider.embed("x")
    assert len(calls) == 4  # the first try plus three retries


@pytest.mark.asyncio
async def test_rate_limit_that_persists_still_raises(monkeypatch) -> None:
    import httpx

    provider, calls = _provider_answering(monkeypatch, [429])
    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed("x")
    assert len(calls) > 1  # retried before giving up


@pytest.mark.asyncio
async def test_a_client_error_is_not_retried(monkeypatch) -> None:
    import httpx

    provider, calls = _provider_answering(monkeypatch, [400, 200])
    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed("x")
    assert calls == [400]


@pytest.mark.asyncio
async def test_a_long_retry_after_goes_up_to_the_shared_gate(monkeypatch) -> None:
    """Waiting out an hour-long Retry-After inside one call would hang whatever needed the
    embedding; it raises at once, and the gate above caps and shares the cooldown."""
    import httpx

    provider, calls = _provider_answering(monkeypatch, [429, 200], retry_after="3600")
    with pytest.raises(httpx.HTTPStatusError):
        await provider.embed("x")
    assert calls == [429]


@pytest.mark.asyncio
async def test_a_routine_retry_after_is_still_waited_out(monkeypatch) -> None:
    """A throttle asking for tens of seconds is waited out and retried: giving up would lose the
    memory whose embedding this was."""
    provider, calls = _provider_answering(monkeypatch, [429, 200], retry_after="30")
    result = await provider.embed("x")
    assert result.dense == [0.5]
    assert calls == [429, 200]
