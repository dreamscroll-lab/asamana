from __future__ import annotations

import re
from pathlib import Path

import pytest

from config import load_config
from config.loader import CONFIG_ENV_VAR
from providers.llm.catalog import resolve_catalog


def test_load_test_config() -> None:
    config = load_config(Path("config/config.test.yaml"))

    assert config.embedding.provider == "in_memory"
    assert config.embedding.params["dimension"] == 4
    assert config.vector_store.provider == "in_memory"
    assert config.world.config == "tiled"
    assert config.world.max_agents == 3
    assert config.engine.max_events_per_window == 1
    assert config.logging.output == "stdout"
    assert config.logging.fmt == "console"


def test_load_config_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.from-env.yaml"
    config_path.write_text(Path("config/config.test.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))

    config = load_config()

    assert config.world.max_agents == 3
    assert config.engine.event_check_interval == 2


def test_a_placeholder_default_applies_when_the_var_is_unset_or_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("config/config.test.yaml").read_text(encoding="utf-8")
        + "\nweb:\n  dev_tools_enabled: ${ASAMANA_TEST_FLAG:-true}\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("ASAMANA_TEST_FLAG", raising=False)
    assert load_config(config_path).web.dev_tools_enabled is True
    monkeypatch.setenv("ASAMANA_TEST_FLAG", "")
    assert load_config(config_path).web.dev_tools_enabled is True
    monkeypatch.setenv("ASAMANA_TEST_FLAG", "false")
    assert load_config(config_path).web.dev_tools_enabled is False


def test_a_placeholder_without_default_still_requires_the_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("config/config.test.yaml").read_text(encoding="utf-8")
        + "\nweb:\n  host: ${ASAMANA_TEST_HOST}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ASAMANA_TEST_HOST", raising=False)

    with pytest.raises(ValueError, match="ASAMANA_TEST_HOST"):
        load_config(config_path)


def test_load_config_applies_engine_and_logging_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(
        "\n".join(
            [
                "llm:",

                "  default_provider: mock",
                "embedding:",
                "  provider: in_memory",
                "vector_store:",
                "  provider: in_memory",
                "agent_store:",
                "  provider: in_memory",
                "message:",
                "  provider: in_memory",
                "snapshot:",
                "  provider: in_memory",
                "world:",
                "  config: tiled",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.engine.max_concurrent_llm == 16
    assert config.logging.level == "INFO"
    assert config.logging.fmt == "json"
    assert config.logging.output == "stdout"


def test_load_config_rejects_non_mapping_documents(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config(config_path)


# ---------------------------------------------------------------------------
# Per-scene call parameters (SceneLLM) — "how to call" beyond "which model"
# ---------------------------------------------------------------------------


def _llm_section(**overrides: object) -> dict[str, object]:
    """A minimal llm config: default provider ``p1`` (model ``m``), ``overrides`` per scene."""
    section: dict[str, object] = {
        "providers": {"p1": {"base_url": "https://p1.invalid/v1", "model": "m"}},
        "default_provider": "p1",
    }
    if overrides:
        section["scenes"] = dict(overrides)
    return section


def test_a_scene_names_a_provider_and_a_model() -> None:
    """Provider and model are two fields, not one ``vendor/model`` string — the latter would need
    parsing back in several places and its own guard against "vendor without model", which the
    schema should do for free."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(**_llm_section())
    chosen = config.scene_llm(LLMScene.WORLD_BUILDING)

    assert (chosen.provider, chosen.model) == ("p1", "m")
    assert config.params_for(LLMScene.WORLD_BUILDING) == {}


def test_a_scene_not_listed_takes_the_default_provider() -> None:
    """Most scenes need no entry at all — copying the same provider and model into a dozen scenes
    means a dozen edits per model change."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(**_llm_section(
        world_building={"provider": "p1", "model": "big"},
    ))

    assert config.scene_llm(LLMScene.AGENT_DECISION_MAIN).model == "m"
    assert config.scene_llm(LLMScene.WORLD_BUILDING).model == "big"


def test_a_scene_entry_must_name_its_provider() -> None:
    """A scene record must say on its own which endpoint it uses without consulting defaults —
    especially when two vendors are connected, where an omitted provider has no answer."""
    import pydantic

    from config.models import LLMConfig

    with pytest.raises(pydantic.ValidationError):
        LLMConfig(**_llm_section(world_building={"model": "big"}))


def test_omitting_model_takes_that_providers_own_model() -> None:
    """Switching provider switches the model too — don't carry the previous vendor's model name to
    the new endpoint, where that model doesn't exist."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(
        providers={
            "p1": {"base_url": "https://p1.invalid/v1", "model": "m", "api_key_env": "P1_KEY"},
            "p2": {"base_url": "https://p2.invalid/v1", "model": "other", "api_key_env": "P2_KEY"},
        },
        default_provider="p1",
        scenes={"world_building": {"provider": "p2"}},
    )
    chosen = config.scene_llm(LLMScene.WORLD_BUILDING)

    assert (chosen.provider, chosen.model) == ("p2", "other")


def test_a_scene_name_that_is_not_a_scene_is_rejected() -> None:
    """A misspelled scene name is rejected by the schema on the spot, not silently ignored."""
    import pydantic

    from config.models import LLMConfig

    with pytest.raises(pydantic.ValidationError):
        LLMConfig(**_llm_section(agent_decison_main={"provider": "p1"}))


def test_call_params_reach_the_provider_verbatim(monkeypatch) -> None:
    """Endpoint params pass through to extra_body verbatim — nothing in between enumerates or
    rewrites their keys.

    That's the whole point: the schema of keys like ``enable_thinking`` / ``enable_search`` belongs
    to the endpoint, so switching models should change one word in config, not code.
    """
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene
    from providers.llm.catalog import create_llm, resolve_catalog

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    params = {"enable_thinking": True, "enable_search": True}
    config = LLMConfig(**_llm_section(
        world_building={"provider": "p1", "model": "big", "params": params},
    ))
    chosen = config.scene_llm(LLMScene.WORLD_BUILDING)

    provider = create_llm(
        chosen.provider, chosen.model, resolve_catalog(config.providers),
        **config.params_for(LLMScene.WORLD_BUILDING),
    )

    assert provider._call_params == params


def test_connection_coordinates_are_not_call_params() -> None:
    """``base_url`` / ``api_key_env`` come with the endpoint catalog and stay out of call params —
    otherwise they'd flow with ``params`` into extra_body and be sent as endpoint dialect."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(
        providers={"p1": {
            "base_url": "https://gateway.internal/v1",
            "api_key_env": "INTERNAL_KEY",
            "params": {"enable_search": True},
        }},
        default_provider="p1",
    )

    assert config.params_for(LLMScene.WORLD_BUILDING) == {"enable_search": True}


def test_timeout_is_a_field_of_its_own_not_a_params_key() -> None:
    """``params`` goes to the endpoint whole, while how long to wait is our business — the endpoint
    doesn't and shouldn't see it.

    They look alike but are treated differently, hence two fields: mixed together, the config's
    shape would mislead.
    """
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(**_llm_section(
        world_building={"provider": "p1", "timeout": 300, "params": {"enable_search": True}},
    ))

    assert config.scene_llm(LLMScene.WORLD_BUILDING).timeout == 300
    assert config.params_for(LLMScene.WORLD_BUILDING) == {"enable_search": True}


def test_a_scene_timeout_overrides_its_providers() -> None:
    """The provider gives this vendor's default and a scene extends it in place — theme analysis
    with search enabled far exceeds the normal timeout."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(
        providers={"p1": {"base_url": "https://p1.invalid/v1", "model": "m", "timeout": 90}},
        default_provider="p1",
        scenes={"world_building": {"provider": "p1", "timeout": 300}},
    )

    assert config.scene_llm(LLMScene.AGENT_DECISION_MAIN).timeout == 90
    assert config.scene_llm(LLMScene.WORLD_BUILDING).timeout == 300


def test_no_timeout_anywhere_leaves_it_to_the_transport() -> None:
    """A number should have one default — it lives on ``OpenAICompatProvider``; config doesn't copy
    it."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    assert LLMConfig(**_llm_section()).scene_llm(LLMScene.WORLD_BUILDING).timeout is None


def test_one_scenes_call_params_never_reach_another() -> None:
    """``params`` belongs only to the scene that writes it.

    Silently inheriting an expensive switch (thinking / web search) is much worse than a missing
    setting: some unnoticed scene would think with web search every step, with nothing in the
    config showing it.
    """
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(**_llm_section(
        world_building={"provider": "p1", "params": {"enable_search": True}},
    ))

    assert config.params_for(LLMScene.WORLD_BUILDING) == {"enable_search": True}
    assert config.params_for(LLMScene.CAST_DESIGN) == {}


def test_provider_params_reach_every_scene_on_it_and_a_scene_can_override_them() -> None:
    """Provider-level defaults reach every scene on that provider and can be overridden in place.

    The default is a visible config entry pinned by this test. Hidden as an implicit constructor
    default it looks like dead code, and removing it turns thinking on in every scene (world
    building becomes several times slower).
    """
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(
        providers={"p1": {
            "base_url": "https://p1.invalid/v1", "model": "m",
            "params": {"enable_thinking": False},
        }},
        default_provider="p1",
        scenes={"world_building": {
            "provider": "p1", "model": "big",
            "params": {"enable_thinking": True, "enable_search": True},
        }},
    )

    # A scene without params gets the provider default — not an empty dict.
    assert config.params_for(LLMScene.AGENT_DECISION_MAIN) == {"enable_thinking": False}
    # Written keys override one by one; the remaining defaults still come along.
    assert config.params_for(LLMScene.WORLD_BUILDING) == {
        "enable_thinking": True, "enable_search": True,
    }


def test_one_providers_dialect_never_leaks_onto_another() -> None:
    """Dialect keys are scoped to the provider, not global.

    ``enable_thinking`` is a Bailian key; sent to kimi in a mixed setup it just draws a 400 — and
    that 400 lands in some caller's except, disguised as model flakiness.
    """
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(
        providers={
            "p1": {"base_url": "https://p1.invalid/v1", "model": "m",
                   "api_key_env": "P1_KEY", "params": {"enable_thinking": False}},
            "p2": {"base_url": "https://p2.invalid/v1", "model": "other",
                   "api_key_env": "P2_KEY"},
        },
        default_provider="p1",
        scenes={"agent_decision_main": {"provider": "p2"}},
    )

    assert config.params_for(LLMScene.AGENT_DECISION_MAIN) == {}
    assert config.params_for(LLMScene.AGENT_INTERRUPT_DECISION) == {"enable_thinking": False}


def test_the_mock_provider_survives_endpoint_call_params() -> None:
    """Provider params are splatted into every scene's instance, mock included — it must accept
    them."""
    from core.factory import ComponentKind, ProviderFactory

    provider = ProviderFactory.create(
        "mock", kind=ComponentKind.LLM, enable_thinking=False, enable_search=True, timeout=300,
    )

    assert provider.fixed_response == "mock response"   # still usable; params ignored


# ---------------------------------------------------------------------------
# Offline judge — declared separately from scenes
# ---------------------------------------------------------------------------


def test_the_judge_has_its_own_declaration() -> None:
    """The judge grades exactly those scenes' output, so each picks its own model — the same model
    generating and grading itself is lenient."""
    from config.models import LLMConfig
    from core.interfaces.llm import LLMScene

    config = LLMConfig(
        providers={
            "p1": {"base_url": "https://p1.invalid/v1", "model": "m", "api_key_env": "P1_KEY"},
            "strict": {"base_url": "https://strict.invalid/v1", "model": "big",
                       "api_key_env": "STRICT_KEY", "params": {"enable_thinking": False}},
        },
        default_provider="p1",
        scenes={"world_building": {"provider": "p1", "model": "other"}},
        judge={"provider": "strict", "params": {"enable_thinking": True}},
    )

    judge = config.judge_llm()
    assert (judge.provider, judge.model) == ("strict", "big")
    assert config.judge_params() == {"enable_thinking": True}
    # Neither affects the other.
    assert config.scene_llm(LLMScene.WORLD_BUILDING).model == "other"


def test_an_unset_judge_falls_back_to_the_default_provider() -> None:
    """No judge declared falls back to the default vendor — it runs, but the judge shares a model
    with the scenes it grades."""
    from config.models import LLMConfig

    config = LLMConfig(**_llm_section())

    judge = config.judge_llm()
    assert (judge.provider, judge.model) == ("p1", "m")
    assert config.judge_params() == {}


def _shipped_endpoint_configs() -> list[Path]:
    """Configs that hit real endpoints: default, reference sample, vendor presets.
    config.test.yaml is all mock and excluded."""
    root = [p for p in sorted(Path("config").glob("config*.yaml")) if p.name != "config.test.yaml"]
    return root + sorted(Path("config/presets").glob("*.yaml"))


def _provide_deployment_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Placeholders for the ``${VAR}`` without a default that shipped configs require (e.g. a
    workspace id inside a base_url); deployment supplies the real values."""
    for path in _shipped_endpoint_configs():
        for name in re.findall(r"\$\{([A-Z0-9_]+)\}", path.read_text(encoding="utf-8")):
            monkeypatch.setenv(name, "test")


def test_the_shipped_configs_declare_a_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configs hitting real endpoints must declare the judge explicitly, or review silently turns
    into self-review.

    Globbed rather than listed: adding another vendor config keeps this invariant automatically.
    """
    _provide_deployment_vars(monkeypatch)
    paths = _shipped_endpoint_configs()
    assert paths
    for path in paths:
        config = load_config(path)
        assert config.llm.judge is not None, path
        assert config.llm.judge_llm().model, path


def test_the_shipped_configs_resolve_their_llm_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading checks the schema only; the endpoint catalog has its own rules (e.g. several
    providers can't silently share the default key variable) that otherwise surface only when
    the server starts."""
    _provide_deployment_vars(monkeypatch)
    for path in _shipped_endpoint_configs():
        assert resolve_catalog(load_config(path).llm.providers), path


def test_the_shipped_configs_keep_dev_tools_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev tools spawn subprocesses and hit paid LLMs with arbitrary prompts; someone doing the
    documented one-click deploy must not get them by default."""
    monkeypatch.delenv("ASAMANA_DEV_TOOLS", raising=False)
    _provide_deployment_vars(monkeypatch)
    for path in _shipped_endpoint_configs():
        assert load_config(path).web.dev_tools_enabled is False, path


def test_cast_size_is_capped() -> None:
    """The cast-size cap must itself be capped: theme analysis's output budget grows
    quadratically with it (relations are C(N,2) pairs), and a fat-fingered large number would push
    max_tokens past what the endpoint accepts — reported as an endpoint error."""
    import pydantic

    from config.models import MAX_CAST_SIZE, WorldConfigSection

    assert WorldConfigSection(config="tiled", max_agents=MAX_CAST_SIZE).max_agents == MAX_CAST_SIZE
    for field in ("min_agents", "max_agents"):
        with pytest.raises(pydantic.ValidationError):
            WorldConfigSection(config="tiled", **{field: MAX_CAST_SIZE + 1})


def test_the_largest_cast_fits_the_endpoint_output_cap() -> None:
    """MAX_CAST_SIZE and MAX_TOKENS_MULTIPLIER move together: raising either can push the
    world-building max_tokens past the smallest output cap among the shipped models (131072)."""
    from config.models import MAX_CAST_SIZE, EngineConfig
    from world.builders.theme_analyzer import _theme_max_tokens

    assert _theme_max_tokens(MAX_CAST_SIZE, EngineConfig().max_npcs) <= 131_072


def test_an_unknown_logging_level_is_a_config_error() -> None:
    """Fail fast (Rule 3): a typo must not quietly log at another level."""
    import pydantic

    from config.models import LoggingConfig

    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(level="VERBOSE")
    assert LoggingConfig(level="DEBUG").level == "DEBUG"
