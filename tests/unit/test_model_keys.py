"""A deployment holds every model key its config uses, or none: none starts a keyless backend that
serves stored worlds and refuses anything that would call a model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from config.models import LLMConfig
from core.container import Container
from core.interfaces.llm import LLMMessage, LLMScene
from core.model_keys import ModelKeysMissing
from interaction.api import create_app
from world import WorldCatalog

_LLM_KEY = "ASAMANA_TEST_LLM_KEY"
_JUDGE_KEY = "ASAMANA_TEST_JUDGE_KEY"
_EMBEDDING_KEY = "ASAMANA_TEST_EMBEDDING_KEY"


@pytest.fixture()
def keyed_config(test_config):
    """The test config pointed at networked endpoints: an LLM, a judge on its own account, and
    an embedding."""
    config = test_config.model_copy(deep=True)
    config.llm = LLMConfig(
        providers={
            "p1": {"base_url": "https://p1.invalid/v1", "model": "m", "api_key_env": _LLM_KEY},
            "p2": {"base_url": "https://p2.invalid/v1", "model": "j", "api_key_env": _JUDGE_KEY},
        },
        default_provider="p1",
        judge={"provider": "p2"},
    )
    config.embedding.provider = "openai_compat"
    config.embedding.params = {
        "model": "e", "dimension": 4, "base_url": "https://e.invalid/v1",
        "api_key_env": _EMBEDDING_KEY,
    }
    return config


def _set_keys(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in (_LLM_KEY, _JUDGE_KEY, _EMBEDDING_KEY):
        if name in names:
            monkeypatch.setenv(name, "k")
        else:
            monkeypatch.delenv(name, raising=False)


def test_all_keys_set_builds_the_real_providers(keyed_config, monkeypatch) -> None:
    _set_keys(monkeypatch, _LLM_KEY, _JUDGE_KEY, _EMBEDDING_KEY)
    container = Container.from_config(keyed_config)

    assert container.model_keys is True
    assert type(container.llm_router.get(LLMScene.WORLD_BUILDING)).__name__ == "OpenAICompatProvider"
    assert container.embedding.dimension == 4


@pytest.mark.asyncio
async def test_no_keys_starts_keyless(keyed_config, monkeypatch) -> None:
    _set_keys(monkeypatch)
    container = Container.from_config(keyed_config)

    assert container.model_keys is False
    with pytest.raises(ModelKeysMissing):
        await container.llm_router.complete(
            LLMScene.WORLD_BUILDING, [LLMMessage(role="user", content="hi")],
        )
    with pytest.raises(ModelKeysMissing):
        await container.embedding.embed("hi")
    with pytest.raises(ModelKeysMissing):
        container.embedding.dimension


@pytest.mark.parametrize("present", [
    (_LLM_KEY, _JUDGE_KEY),          # embedding missing
    (_LLM_KEY, _EMBEDDING_KEY),      # judge's own account missing
    (_EMBEDDING_KEY,),
])
def test_partly_set_keys_refuse_to_start(keyed_config, monkeypatch, present) -> None:
    _set_keys(monkeypatch, *present)
    missing = {_LLM_KEY, _JUDGE_KEY, _EMBEDDING_KEY} - set(present)

    with pytest.raises(ValueError, match="all set or all unset") as exc:
        Container.from_config(keyed_config)
    assert all(name in str(exc.value) for name in missing)


def test_an_empty_value_counts_as_unset(keyed_config, monkeypatch) -> None:
    _set_keys(monkeypatch)
    monkeypatch.setenv(_LLM_KEY, "")

    assert Container.from_config(keyed_config).model_keys is False


def test_a_config_without_endpoints_needs_no_keys(container) -> None:
    assert container.model_keys is True


@pytest.fixture()
def keyless_client(keyed_config, monkeypatch) -> TestClient:
    _set_keys(monkeypatch)
    keyed_config.web.dev_tools_enabled = True
    container = Container.from_config(keyed_config)
    return TestClient(create_app(container, keyed_config, catalog=WorldCatalog(None)))


def test_keyless_backend_reports_it_and_still_serves_reads(keyless_client) -> None:
    assert keyless_client.get("/api/deployment").json()["model_keys"] is False
    assert keyless_client.get("/api/worlds").status_code == 200


@pytest.mark.parametrize("method, path, body", [
    ("post", "/api/worlds", {"theme": "长安"}),
    ("post", "/api/worlds/w1/run", {"steps": 1}),
    ("post", "/api/worlds/w1/step", None),
    ("post", "/api/worlds/w1/direct", {"text": "下雨"}),
    ("post", "/api/worlds/w1/reset", None),
    ("post", "/api/llm/replay", {"messages": [{"role": "user", "content": "hi"}]}),
    ("post", "/api/worlds/w1/audit/run", {}),
    ("post", "/api/dev/stages/decide/run", {"world_id": "w1"}),
    ("post", "/api/worlds/w1/memory/recall", {"agent_id": "a1", "query": "q"}),
    ("post", "/api/worlds/w1/director/prompt", {"text": "下雨"}),
    ("post", "/api/worlds/w1/director/interpret", {"response": "{}"}),
])
def test_keyless_backend_refuses_model_calls(keyless_client, method, path, body, caplog) -> None:
    with caplog.at_level("INFO", logger="interaction.api.app"):
        response = getattr(keyless_client, method)(path, json=body)

    assert response.status_code == 503
    assert "asamana.sh keys" in response.json()["detail"]
    refused = [r for r in caplog.records if r.getMessage() == "model_call_refused"]
    assert [(r.method, r.path) for r in refused] == [(method.upper(), path)]


def test_keyless_cli_refuses_to_build(keyed_config, monkeypatch, capsys) -> None:
    import asyncio

    from interaction.cli import run_cli_async

    _set_keys(monkeypatch)
    container = Container.from_config(keyed_config)

    assert asyncio.run(run_cli_async(container, keyed_config, ["build", "长安"])) == 1
    assert "asamana.sh keys" in capsys.readouterr().err
