"""The endpoint catalog has one source: config. Adding a provider must not touch code."""

from __future__ import annotations

import pytest

from config.models import LLMProviderConfig
from providers.llm.catalog import create_llm, resolve_catalog


def _providers(**decls: str | tuple[str, str]) -> dict[str, LLMProviderConfig]:
    """``name=base_url`` uses the default credential; ``name=(base_url, KEY_ENV)`` names it explicitly."""
    return {
        name: LLMProviderConfig(base_url=decl[0], api_key_env=decl[1])
        if isinstance(decl, tuple) else LLMProviderConfig(base_url=decl)
        for name, decl in decls.items()
    }


def test_a_provider_works_from_config_alone(monkeypatch) -> None:
    """Adding a provider takes three lines of YAML and no code. That is what "pluggable" means.

    Code must not hold a provider list: every new provider would need an edit and a release, and
    the question "where does this prefix connect" would have two answers.
    """
    monkeypatch.setenv("LLM_API_KEY", "step-key")

    catalog = resolve_catalog(_providers(stepfun="https://api.stepfun.com/v1"))
    provider = create_llm("stepfun", "step-2", catalog)

    assert str(provider._client.base_url).rstrip("/") == "https://api.stepfun.com/v1"
    assert provider._client.api_key == "step-key"
    assert provider.model == "step-2"


def test_a_provider_must_say_where_it_is() -> None:
    """``base_url`` has no default. The schema rejects a missing one immediately (Rule 3) instead of
    letting it surface at runtime as a connection error that hides the missing config."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        LLMProviderConfig(api_key_env="STEPFUN_API_KEY")


def test_one_provider_needs_no_key_name_but_two_do(monkeypatch) -> None:
    """With one provider the credential defaults to ``LLM_API_KEY``; no variable name needed.

    Two providers with no key names is a config error: one variable can't hold two keys, and
    letting it through makes one provider authenticate with the other's credential and fail with no
    hint that the config is mixed up.
    """
    monkeypatch.setenv("LLM_API_KEY", "the-only-key")

    one = resolve_catalog(_providers(kimi="https://api.moonshot.cn/v1"))
    assert create_llm("kimi", "kimi-k2.6", one)._client.api_key == "the-only-key"

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        resolve_catalog(_providers(
            kimi="https://api.moonshot.cn/v1", zhipu="https://open.bigmodel.cn/api/paas/v4",
        ))



def test_the_same_provider_can_be_pointed_at_another_region(monkeypatch) -> None:
    """MiniMax international site or a self-hosted gateway: change those two config lines and keep the
    model prefix."""
    monkeypatch.setenv("MINIMAX_GLOBAL_KEY", "global-key")

    catalog = resolve_catalog(_providers(
        minimax=("https://api.minimax.io/v1", "MINIMAX_GLOBAL_KEY"),
    ))
    provider = create_llm("minimax", "MiniMax-M2", catalog)

    assert str(provider._client.base_url).rstrip("/") == "https://api.minimax.io/v1"
    assert provider._client.api_key == "global-key"


def test_an_unknown_provider_says_what_is_known_and_how_to_add_one() -> None:
    """Config is the only interface for adding a provider, so this error message is the docs: a
    misspelled prefix must get more than just "Unknown"."""
    with pytest.raises(ValueError) as exc:
        create_llm("arkk", "doubao", resolve_catalog(_providers(
            kimi="https://api.moonshot.cn/v1",
        )))

    message = str(exc.value)
    assert "kimi" in message and "mock" in message     # which providers this config knows
    assert "llm.providers" in message                  # where to add another one


def test_config_cannot_hijack_a_name_that_routes_elsewhere() -> None:
    """``mock`` is a test double, not an endpoint. Letting config redefine it as an OpenAI-compatible
    vendor would let production and tests impersonate each other, so it is rejected."""
    with pytest.raises(ValueError, match="reserved"):
        resolve_catalog(_providers(mock="https://evil.invalid/v1"))


def test_a_provider_outside_the_catalog_falls_through_to_the_factory() -> None:
    """``mock`` is not an OpenAI-compatible vendor; it still goes through ProviderFactory."""
    from providers.llm.mock import MockLLMProvider

    assert isinstance(create_llm("mock", "", resolve_catalog(None)), MockLLMProvider)


# ---------------------------------------------------------------------------
# Credential = account identity
# ---------------------------------------------------------------------------


def test_two_providers_read_two_different_keys(monkeypatch) -> None:
    """The other half of pluggable: two providers in one config each use their own key.

    Keys can't be tied to a single env var, or mixing vendors becomes impossible.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "kimi-key")
    catalog = resolve_catalog(_providers(
        deepseek=("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        kimi=("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    ))

    assert create_llm("deepseek", "deepseek-flash", catalog)._client.api_key == "ds-key"
    assert create_llm("kimi", "kimi-k2.6", catalog)._client.api_key == "kimi-key"



def test_a_missing_key_fails_fast_and_names_the_variable(monkeypatch) -> None:
    """Config errors get no fallback (Rule 3), and the error must say which env var to set."""
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    catalog = resolve_catalog(_providers(
        zhipu=("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
    ))

    with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
        create_llm("zhipu", "glm-4.7", catalog)


def test_a_scene_without_a_model_is_rejected(monkeypatch) -> None:
    """Vendor given, model missing. Don't fall back to a hard-coded default model: a wrong config
    would keep running, just on a different model."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    catalog = resolve_catalog(_providers(
        deepseek=("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    ))

    with pytest.raises(ValueError, match="must name a 'model'"):
        create_llm("deepseek", "", catalog)


def test_timeout_is_ours_and_never_reaches_the_endpoint(monkeypatch) -> None:
    """``timeout`` configures our client; ``params`` is sent as-is to the endpoint. They are separate
    channels."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    catalog = resolve_catalog(_providers(
        qwen=("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    ))

    provider = create_llm("qwen", "qwen-plus", catalog, timeout=300, enable_search=True)

    assert provider._call_params == {"enable_search": True}
    assert provider._client.timeout == 300


def test_no_timeout_given_leaves_the_transport_default(monkeypatch) -> None:
    """Defaults live only in the transport layer; neither the catalog nor config keeps a copy."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    catalog = resolve_catalog(_providers(
        qwen=("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    ))

    assert create_llm("qwen", "qwen-plus", catalog)._client.timeout == 60.0


def test_one_config_can_route_two_scenes_to_two_providers(monkeypatch) -> None:
    """End to end: one config routes two scenes to two providers, each with its own endpoint, key,
    and rate-limit gate."""
    from config.models import Config
    from core.container import _build_llm_router, _build_rate_gates
    from core.interfaces.llm import LLMScene

    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-key")
    monkeypatch.setenv("STEPFUN_API_KEY", "step-key")

    local = {"provider": "in_memory", "params": {}}
    config = Config(
        llm={
            "default_provider": "qwen",
            "scenes": {"agent_decision_main": {"provider": "stepfun"}},
            "providers": {
                "qwen": {
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-plus",
                    "api_key_env": "DASHSCOPE_API_KEY",
                    "params": {"enable_thinking": False},
                },
                "stepfun": {
                    "base_url": "https://api.stepfun.com/v1",
                    "model": "step-2",
                    "api_key_env": "STEPFUN_API_KEY",
                },
            },
        },
        embedding=local, vector_store=local, agent_store=local,
        message=local, snapshot=local,
        world={"config": "tiled"},
        observability={"enabled": False},
    )
    catalog = resolve_catalog(config.llm.providers)
    gates = _build_rate_gates(config)
    router = _build_llm_router(config, catalog, rate_gates=gates)

    main = router.get(LLMScene.AGENT_DECISION_MAIN)
    other = router.get(LLMScene.AGENT_INTERRUPT_DECISION)
    assert (main._client.api_key, other._client.api_key) == ("step-key", "qwen-key")
    assert "stepfun" in str(main._client.base_url)
    assert "dashscope" in str(other._client.base_url)
    # Vendor-specific params go only to the vendor that understands them.
    assert main._call_params == {}
    assert other._call_params == {"enable_thinking": False}
    # Rate limits are per endpoint.
    assert router._rate_gates[LLMScene.AGENT_DECISION_MAIN] is gates["stepfun"]
    assert router._rate_gates[LLMScene.AGENT_INTERRUPT_DECISION] is gates["qwen"]


def test_an_endpoint_that_pins_temperature_gets_none_sent(monkeypatch) -> None:
    """Kimi fixes temperature (k2.6 non-thinking accepts only 0.6, k3 only 1.0) and returns 400 for
    any other value, so it must not be sent at all."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    catalog = resolve_catalog({
        "kimi": LLMProviderConfig(
            base_url="https://api.moonshot.cn/v1",
            api_key_env="MOONSHOT_API_KEY",
            accepts_temperature=False,
        ),
    })

    assert create_llm("kimi", "kimi-k2.6", catalog)._accepts_temperature is False


def test_temperature_is_sent_unless_an_endpoint_refuses_it(monkeypatch) -> None:
    """Sent unless declared otherwise. Most endpoints accept it; the switch exists for the few that
    reject it."""
    monkeypatch.setenv("LLM_API_KEY", "k")

    assert create_llm(
        "stepfun", "step-2", resolve_catalog(_providers(stepfun="https://api.stepfun.com/v1")),
    )._accepts_temperature is True
