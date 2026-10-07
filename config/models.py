"""Typed configuration models for Asamana."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.interfaces.llm import LLMScene

#: Env var read when a vendor omits ``api_key_env``. A deployment usually connects one vendor and
#: needn't name a variable; with two vendors one variable can't hold two keys, and
#: ``resolve_catalog`` reports it immediately.
DEFAULT_LLM_KEY_ENV = "LLM_API_KEY"

#: Maximum characters per world. See ``WorldConfigSection.max_agents``; at 16 the world-building
#: budget passes the 131072-token output cap of GLM-5.3 and qwen3.8-max.
MAX_CAST_SIZE = 15


class AsamanaConfigModel(BaseModel):
    """Base config model with strict field validation."""

    model_config = ConfigDict(extra="forbid")


class ProviderConfig(AsamanaConfigModel):
    provider: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class WorldConfigSection(AsamanaConfigModel):
    """No ``params``: the only one would be the template name, and templates are chosen per world
    (by the caller or from the theme), not per deployment.
    """

    config: str = Field(min_length=1)
    # A one-person show has no emergence. Falling short fails the build (Rule 2) rather than
    # padding with filler characters. ``None`` = no minimum.
    min_agents: int | None = Field(default=None, ge=1, le=MAX_CAST_SIZE)
    # Capped because theme analysis's output budget grows with C(N,2) relation pairs (see
    # ``theme_analyzer._theme_max_tokens``): a mistyped large number would surface as an endpoint
    # max_tokens error, not as this config line. ``None`` = no maximum.
    max_agents: int | None = Field(default=None, ge=1, le=MAX_CAST_SIZE)

    @model_validator(mode="after")
    def _cast_bounds_are_satisfiable(self) -> "WorldConfigSection":
        if self.min_agents is not None and self.max_agents is not None:
            if self.min_agents > self.max_agents:
                raise ValueError(
                    f"min_agents ({self.min_agents}) exceeds max_agents ({self.max_agents})"
                )
        return self


class SceneLLM(AsamanaConfigModel):
    """The provider, model and timeout a scene actually uses after resolution: the return value
    of ``LLMConfig.scene_llm()``, not a config shape. ``timeout`` ``None`` means the transport
    default."""

    provider: str = Field(min_length=1)
    model: str = ""
    timeout: float | None = Field(default=None, gt=0)


class SceneOverride(AsamanaConfigModel):
    """An LLM config that differs from the default: a scene, or ``judge``.

    ``provider`` is required so the entry names its endpoint on its own. Omitted ``model`` takes
    that provider's declared model, so switching provider never carries another vendor's model
    name to an endpoint where it doesn't exist.

    ``params`` is passed verbatim to the endpoint (see ``OpenAICompatProvider``), uninterpreted:
    the endpoint owns that schema. It applies only to the declaring scene; leaking it would leave
    other scenes silently searching or thinking every step.
    """

    provider: str = Field(min_length=1)
    model: str | None = None
    # Ours and never sent, so a first-class field, not a ``params`` key.
    timeout: float | None = Field(default=None, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)


class LLMProviderConfig(AsamanaConfigModel):
    """One LLM endpoint: where it is, whose key, default model, and endpoint params.

    There is no vendor list in code: any OpenAI-compatible endpoint works by declaring
    ``base_url``.

    ``api_key_env`` names a variable, not the key, so this config can be committed (default: see
    ``DEFAULT_LLM_KEY_ENV``). Credentials live on the endpoint, not on scenes: a scene-level key
    could pair "DashScope endpoint with a DeepSeek key", which surfaces only as a baffling auth
    failure. Two accounts on one vendor = two declarations with the same ``base_url`` and
    different ``api_key_env``.

    ``params`` are this endpoint's dialect defaults, overridden per key by a scene's ``params``.
    Never global: another vendor answers unknown dialect keys with a 400.
    """

    base_url: str = Field(min_length=1)
    model: str = ""
    api_key_env: str = Field(default=DEFAULT_LLM_KEY_ENV, min_length=1)
    # This endpoint's default timeout; a scene can extend it. ``None`` = transport default.
    timeout: float | None = Field(default=None, gt=0)
    # ``False`` = never send temperature; the endpoint uses its own fixed value. Not a pinned
    # number: the fixed value varies with model and thinking mode (Kimi k2.6: 0.6 without thinking,
    # 1.0 with; anything else is a 400). Cost: per-scene temperatures have no effect there.
    accepts_temperature: bool = True
    # This endpoint's rate quota. Vendors grant RPM per model and a declaration binds exactly one
    # model, so it lives here. ``None`` = the account-level default from ``rate_limit``.
    rpm: int | None = Field(default=None, ge=0)
    tpm: int | None = Field(default=None, ge=0)
    params: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(AsamanaConfigModel):
    """Scene-to-provider routing: ``providers`` declares endpoints; ``scenes`` lists only the
    scenes that differ from ``default_provider``."""

    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    # Required: guessing a default (e.g. from the credential name) would turn a config error into
    # an auth failure.
    default_provider: str = Field(min_length=1)
    # Exceptions only. Keys must be valid scene names; a typo is rejected by the schema.
    scenes: dict[LLMScene, SceneOverride] = Field(default_factory=dict)
    # Offline judge (``python -m tuning``). Not a scene: it scores the scenes' output, so it must
    # pick its model independently.
    judge: SceneOverride | None = None

    def _resolve(self, override: SceneOverride | None) -> SceneLLM:
        """Override → effective provider, model, timeout; ``None`` = all from the default."""
        provider = override.provider if override is not None else self.default_provider
        declared = self.providers.get(provider)
        model = override.model if override is not None and override.model is not None else None
        if model is None:
            model = declared.model if declared is not None else ""
        timeout = override.timeout if override is not None else None
        if timeout is None and declared is not None:
            timeout = declared.timeout
        return SceneLLM(provider=provider, model=model, timeout=timeout)

    def scene_llm(self, scene: LLMScene) -> SceneLLM:
        """Scene entry's provider/model/timeout, else ``default_provider`` and that provider's
        declaration (see ``SceneOverride``)."""
        return self._resolve(self.scenes.get(scene))

    def judge_llm(self) -> SceneLLM:
        """Without a ``judge`` declaration this falls back to the default vendor, i.e. the judge
        grades its own scenes leniently; declare it explicitly."""
        return self._resolve(self.judge)

    def provider_params(self, provider: str) -> dict[str, Any]:
        """This provider's dialect defaults. Connection fields (``base_url`` / ``api_key_env``)
        travel in the catalog (``providers.llm.catalog.resolve_catalog``), not in call params.
        """
        declared = self.providers.get(provider)
        return dict(declared.params) if declared is not None else {}

    def params_for(self, scene: LLMScene) -> dict[str, Any]:
        """The effective provider's dialect defaults with the scene's own ``params`` on top.

        Uses the provider after override, so a scene switching provider never carries the old
        endpoint's keys (the new one would answer with a 400).
        """
        override = self.scenes.get(scene)
        own = override.params if override is not None else {}
        return {**self.provider_params(self.scene_llm(scene).provider), **own}

    def judge_params(self) -> dict[str, Any]:
        """The judge's constructor params: same as ``params_for``, but the override comes from
        ``judge`` instead of a scene."""
        own = self.judge.params if self.judge is not None else {}
        return {**self.provider_params(self.judge_llm().provider), **own}


class EngineConfig(AsamanaConfigModel):
    # Process-wide cap on in-flight chat requests (LLMRouter's semaphore): "how many at once",
    # orthogonal to rate_limit's "how many per minute". It also bounds tpm's accounting lag
    # (in-flight tokens ≤ max_concurrent_llm × max_tokens, see core.rate_gate). Not embedding.
    max_concurrent_llm: int = Field(default=16, gt=0)
    decay_interval: int = Field(default=5, gt=0)
    # Long-term goal revision period (steps), independent of reflection.
    long_term_goal_revision_interval: int = Field(default=30, gt=0)
    # Maintenance-phase switches for ablations, ANDed with each interval. decay has none (it's a
    # base mechanism).
    compression_enabled: bool = Field(default=True)
    reflection_enabled: bool = Field(default=True)
    relation_evolution_enabled: bool = Field(default=True)
    long_term_goal_revision_enabled: bool = Field(default=True)
    # False: check_and_inject short-circuits, for purely agent-driven runs / tuning baselines.
    event_system_enabled: bool = Field(default=True)
    # Sliding-window quota: at most max_events_per_window in the last event_quota_window steps.
    # Not a lifetime total: a long run would spend it early and get no outside force afterwards.
    # Clustering within a window is allowed.
    max_events_per_window: int = Field(default=2, ge=0)
    event_quota_window: int = Field(default=20, gt=0)
    event_check_interval: int = Field(default=6, gt=0)
    # AgentScheduler's gate: an idle agent with no pressure, urgent need or strong emotion is pulled
    # back into decision at least every this many steps. Set in world time (one step = one hour,
    # a waking day ≈ 16 hours). Don't use these to patch problems in the gate's other criteria.
    main_max_idle_steps: int = Field(default=2, gt=0)
    background_max_idle_steps: int = Field(default=6, gt=0)
    # Cap on Npcs (actors with a body but no cognition). The cost is prompt space, not memory: each
    # takes a line in main-character decision prompts. 0 = none.
    max_npcs: int = Field(default=6, ge=0)


class RateLimitConfig(AsamanaConfigModel):
    """Rate-limit defaults (core.rate_gate.RateGate).

    One gate per ``llm.providers`` declaration in use, plus one for embedding. rpm / tpm are
    defaults for endpoints without their own; cooldown_* applies to all gates.

    cooldown_* (always on): after a 429/overload an endpoint's calls share a cooldown window;
    it doubles from base up to max while rejections continue, and one success resets it. A
    response's Retry-After takes precedence, up to retry_after_max.

    rpm / tpm (0 = off): rolling 60s window; rpm is strict admission (chat only), tpm counts
    measured input+output tokens, not estimates. Embedding takes part in cooldown only.
    """

    cooldown_base: float = Field(default=2.0, gt=0)
    cooldown_max: float = Field(default=30.0, gt=0)
    retry_after_max: float = Field(default=120.0, gt=0)
    rpm: int = Field(default=0, ge=0)
    tpm: int = Field(default=0, ge=0)


class ObservabilityConfig(AsamanaConfigModel):
    """LLM observability (tracing) configuration.

    When enabled, every LLM call is recorded with its cognition stage and each
    runtime step's wall-clock total is captured, written as JSONL under
    ``params.base_dir``. Disabled selects a NullTraceSink (zero overhead).
    """

    enabled: bool = Field(default=True)
    provider: str = Field(default="jsonl", min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    # Write full thinking text into the trace. Off by default: it is often several times the
    # answer. Not a scene ``params`` key, which goes to the endpoint verbatim. Thinking token counts
    # are always recorded: they're the only basis for sizing max_tokens.
    capture_thinking: bool = Field(default=False)


class LoggingConfig(AsamanaConfigModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    fmt: Literal["json", "console"] = "json"
    output: Literal["stdout", "file", "stdout+file"] = "stdout"
    file_path: Optional[str] = None
    max_size_mb: int = Field(default=20, gt=0)
    backup_count: int = Field(default=5, ge=0)


class WebConfig(AsamanaConfigModel):
    """Web server (FastAPI/uvicorn) configuration.

    ``cors_origins``: ``["*"]`` is fine locally and behind a same-origin proxy (WebSockets
    bypass CORS).

    ``dev_tools_enabled`` gates the trace explorer, audit runner, prompt-replay playground and
    ``/observability`` dashboard. They spawn subprocesses and send arbitrary prompts to the paid
    LLM, so never enable it in a public deployment.
    """

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=7860, gt=0)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    dev_tools_enabled: bool = Field(default=False)


class Config(AsamanaConfigModel):
    llm: LLMConfig
    embedding: ProviderConfig
    vector_store: ProviderConfig
    agent_store: ProviderConfig
    message: ProviderConfig
    snapshot: ProviderConfig
    world: WorldConfigSection
    engine: EngineConfig = Field(default_factory=EngineConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    web: WebConfig = Field(default_factory=WebConfig)
