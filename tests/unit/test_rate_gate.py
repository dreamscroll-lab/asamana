"""Unit tests for the endpoint-scoped rate gate (cooldown + RPM/TPM governor)."""

from __future__ import annotations

import pytest

from core.interfaces.llm import LLMMessage, LLMProvider, LLMResponse, LLMRouter, LLMScene
from core.rate_gate import RateGate, classify_rate_limit
from providers.embedding.gated import GatedEmbeddingProvider


class _FakeClock:
    """Deterministic clock whose ``sleep`` advances time (records total slept)."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.t += seconds


def _gate(clock: _FakeClock, **kw: object) -> RateGate:
    # jitter=0 keeps sleep durations exact for assertions.
    return RateGate(jitter=0.0, now=clock.now, sleep=clock.sleep, **kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------- classify

class _Resp:
    def __init__(self, status: int, retry_after: str | None = None) -> None:
        self.status_code = status
        self.headers = {"retry-after": retry_after} if retry_after is not None else {}


def test_classify_openai_style_rate_limit_with_retry_after() -> None:
    exc = Exception("rate limited")
    exc.status_code = 429  # type: ignore[attr-defined]
    exc.response = _Resp(429, retry_after="3.5")  # type: ignore[attr-defined]
    limited, retry_after = classify_rate_limit(exc)
    assert limited is True
    assert retry_after == 3.5


def test_classify_httpx_style_overload_without_header() -> None:
    exc = Exception("overloaded")
    exc.response = _Resp(529)  # type: ignore[attr-defined]
    limited, retry_after = classify_rate_limit(exc)
    assert limited is True
    assert retry_after is None


def test_classify_ignores_non_backpressure_errors() -> None:
    exc = Exception("bad request")
    exc.status_code = 400  # type: ignore[attr-defined]
    assert classify_rate_limit(exc) == (False, None)
    assert classify_rate_limit(Exception("no status at all")) == (False, None)


# ---------------------------------------------------------------- gate timing

@pytest.mark.asyncio
async def test_acquire_is_instant_without_cooldown() -> None:
    clock = _FakeClock()
    gate = _gate(clock)
    await gate.acquire()
    assert clock.slept == 0.0


@pytest.mark.asyncio
async def test_penalize_honors_retry_after_verbatim() -> None:
    clock = _FakeClock()
    gate = _gate(clock, base_cooldown=2.0, max_cooldown=30.0)
    gate.penalize(retry_after=5.0)  # authoritative within max_retry_after, ignores base/max
    await gate.acquire()
    assert clock.slept == 5.0


@pytest.mark.asyncio
async def test_retry_after_is_capped() -> None:
    """A spent quota can ask for hours; waiting that out would stall every world on the endpoint
    with no error, so the wait stops at the ceiling and calls fail into their fallbacks."""
    clock = _FakeClock()
    gate = _gate(clock, max_retry_after=120.0)
    gate.penalize(retry_after=3600.0)
    await gate.acquire()
    assert clock.slept == 120.0


@pytest.mark.asyncio
async def test_hits_within_one_cooldown_do_not_stack() -> None:
    """A burst of concurrent 429s is one rate-limit event: in-flight requests bounced during
    cooldown mustn't push it to max."""
    clock = _FakeClock()
    gate = _gate(clock, base_cooldown=2.0, max_cooldown=30.0)
    for _ in range(5):
        gate.penalize()
    await gate.acquire()
    assert clock.slept == 2.0


@pytest.mark.asyncio
async def test_each_round_still_failing_doubles_up_to_max() -> None:
    """Only a bounce after cooldown means sustained overrun; that doubles per round."""
    clock = _FakeClock()
    gate = _gate(clock, base_cooldown=2.0, max_cooldown=6.0)
    waits = []
    for _ in range(4):
        gate.penalize()
        before = clock.slept
        await gate.acquire()
        waits.append(clock.slept - before)
    assert waits == [2.0, 4.0, 6.0, 6.0]


@pytest.mark.asyncio
async def test_a_success_between_rounds_starts_over_at_base() -> None:
    clock = _FakeClock()
    gate = _gate(clock, base_cooldown=2.0, max_cooldown=30.0)
    gate.penalize()
    await gate.acquire()
    gate.penalize()  # second round → 4
    await gate.acquire()
    gate.note_success()
    gate.penalize()
    before = clock.slept
    await gate.acquire()
    assert clock.slept - before == 2.0


@pytest.mark.asyncio
async def test_retry_after_within_a_cooldown_still_extends_it() -> None:
    """Retry-After is the vendor's authoritative value; it pushes the deadline out even during
    cooldown."""
    clock = _FakeClock()
    gate = _gate(clock, base_cooldown=2.0, max_cooldown=30.0)
    gate.penalize()
    gate.penalize(retry_after=10.0)
    await gate.acquire()
    assert clock.slept == 10.0


def test_construction_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        RateGate(base_cooldown=10.0, max_cooldown=5.0)


def test_construction_rejects_negative_limits() -> None:
    with pytest.raises(ValueError):
        RateGate(rpm_limit=-1)
    with pytest.raises(ValueError):
        RateGate(tpm_limit=-1)


# ---------------------------------------------------------------- slice B: RPM

@pytest.mark.asyncio
async def test_rpm_admits_up_to_limit_then_waits_for_slot() -> None:
    clock = _FakeClock()
    gate = _gate(clock, rpm_limit=2, window_seconds=60.0)
    await gate.acquire()  # t=0, slot 1
    await gate.acquire()  # t=0, slot 2
    assert clock.slept == 0.0
    await gate.acquire()  # window full → wait until the oldest ages out at t=60
    assert clock.slept == 60.0


@pytest.mark.asyncio
async def test_rpm_disabled_never_waits() -> None:
    clock = _FakeClock()
    gate = _gate(clock, rpm_limit=0)
    for _ in range(100):
        await gate.acquire()
    assert clock.slept == 0.0


@pytest.mark.asyncio
async def test_rpm_ignored_when_reserve_rate_false() -> None:
    clock = _FakeClock()
    gate = _gate(clock, rpm_limit=1, window_seconds=60.0)
    # A separate-quota caller (embedding) never consumes RPM slots regardless of count.
    for _ in range(5):
        await gate.acquire(reserve_rate=False)
    assert clock.slept == 0.0


# ---------------------------------------------------------------- slice B: TPM

@pytest.mark.asyncio
async def test_tpm_under_budget_admits_immediately() -> None:
    clock = _FakeClock()
    gate = _gate(clock, tpm_limit=100, window_seconds=60.0)
    gate.note_usage(10, 10)  # 20 < 100
    await gate.acquire()
    assert clock.slept == 0.0


@pytest.mark.asyncio
async def test_tpm_over_budget_waits_for_window_to_slide() -> None:
    clock = _FakeClock()
    gate = _gate(clock, tpm_limit=100, window_seconds=60.0)
    gate.note_usage(60, 50)  # 110 >= 100 at t=0
    await gate.acquire()  # over budget → wait until that usage ages out at t=60
    assert clock.slept == 60.0


@pytest.mark.asyncio
async def test_note_usage_is_noop_when_tpm_disabled() -> None:
    clock = _FakeClock()
    gate = _gate(clock, tpm_limit=0)
    for _ in range(50):
        gate.note_usage(1000, 1000)
    assert len(gate._usage) == 0  # nothing accumulates when disabled
    await gate.acquire()
    assert clock.slept == 0.0


@pytest.mark.asyncio
async def test_cooldown_and_rate_take_the_max_wait() -> None:
    clock = _FakeClock()
    gate = _gate(clock, tpm_limit=100, window_seconds=60.0)
    gate.note_usage(200, 0)  # over budget → tpm wants 60
    gate.penalize(retry_after=10.0)  # cooldown wants 10
    await gate.acquire()
    assert clock.slept == 60.0  # the binding (larger) wait governs


# ---------------------------------------------------------------- router wiring

class _SpyGate:
    """Records gate interactions; substitutes for RateGate at the call boundary."""

    def __init__(self) -> None:
        self.acquired = 0
        self.reserve_flags: list[bool] = []
        self.penalized: list[float | None] = []
        self.successes = 0
        self.usages: list[tuple[int, int]] = []

    async def acquire(self, *, reserve_rate: bool = True) -> None:
        self.acquired += 1
        self.reserve_flags.append(reserve_rate)

    def penalize(self, retry_after: float | None = None) -> None:
        self.penalized.append(retry_after)

    def note_success(self) -> None:
        self.successes += 1

    def note_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.usages.append((input_tokens, output_tokens))


class _ScriptedProvider(LLMProvider):
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc

    async def complete(self, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ) -> LLMResponse:
        if self._exc is not None:
            raise self._exc
        return LLMResponse(content="ok", input_tokens=3, output_tokens=5, model="fake")

    async def stream(self, messages, temperature=0.7):  # pragma: no cover - unused here
        yield "ok"


def _router(provider: LLMProvider, gate: _SpyGate) -> LLMRouter:
    return LLMRouter(
        {scene: provider for scene in LLMScene},
        rate_gates={scene: gate for scene in LLMScene},
    )


@pytest.mark.asyncio
async def test_router_acquires_and_notes_success() -> None:
    gate = _SpyGate()
    router = _router(_ScriptedProvider(), gate)
    await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="hi")])
    assert gate.acquired == 1
    assert gate.reserve_flags == [True]  # chat reserves an RPM slot
    assert gate.successes == 1
    assert gate.penalized == []
    assert gate.usages == [(3, 5)]  # actual tokens fed to the TPM governor


@pytest.mark.asyncio
async def test_router_penalizes_on_backpressure_and_reraises() -> None:
    exc = Exception("429")
    exc.status_code = 429  # type: ignore[attr-defined]
    exc.response = _Resp(429, retry_after="7")  # type: ignore[attr-defined]
    gate = _SpyGate()
    router = _router(_ScriptedProvider(exc=exc), gate)
    with pytest.raises(Exception):
        await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="hi")])
    assert gate.acquired == 1
    assert gate.penalized == [7.0]
    assert gate.successes == 0


@pytest.mark.asyncio
async def test_router_does_not_penalize_on_ordinary_error() -> None:
    exc = Exception("boom")  # no status → not backpressure
    gate = _SpyGate()
    router = _router(_ScriptedProvider(exc=exc), gate)
    with pytest.raises(Exception):
        await router.complete(LLMScene.AGENT_DECISION_MAIN, [LLMMessage(role="user", content="hi")])
    assert gate.penalized == []


# ---------------------------------------------------------------- embedding wrapper

class _FakeEmbedding:
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc
        self.calls = 0

    async def embed(self, text: str):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return "vec"

    @property
    def dimension(self) -> int:
        return 3


@pytest.mark.asyncio
async def test_gated_embedding_success_path() -> None:
    gate = _SpyGate()
    inner = _FakeEmbedding()
    wrapped = GatedEmbeddingProvider(inner, gate)  # type: ignore[arg-type]
    assert await wrapped.embed("hello") == "vec"
    assert await wrapped.embed("world") == "vec"
    assert wrapped.dimension == 3
    assert gate.acquired == 2
    assert gate.reserve_flags == [False, False]  # separate quota: no RPM/TPM reservation
    assert gate.successes == 2
    assert gate.penalized == []
    assert gate.usages == []  # embeddings carry no token counts


@pytest.mark.asyncio
async def test_gated_embedding_penalizes_on_backpressure() -> None:
    exc = Exception("429")
    exc.response = _Resp(429)  # type: ignore[attr-defined]
    gate = _SpyGate()
    wrapped = GatedEmbeddingProvider(_FakeEmbedding(exc=exc), gate)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        await wrapped.embed("hello")
    assert gate.acquired == 1
    assert gate.penalized == [None]
    assert gate.successes == 0


# ---------------------------------------------------------------------------
# One gate per account: rate limits are per API account, not per process
# ---------------------------------------------------------------------------


def _config_with(**scene_providers: str):
    """A config with local providers for everything but llm; the providers used declare coordinates
    inline.

    Pass ``scene=provider`` (provider name only; the model is always ``m``)."""
    from config.models import Config

    llm: dict = {
        "default_provider": "mock",
        "scenes": {
            scene: {"provider": name} for scene, name in scene_providers.items()
        },
        # These tests care about "who shares a gate with whom", not vendors' real addresses, so
        # coordinates derive from names.
        "providers": {
            name: {
                "base_url": f"https://{name}.invalid/v1",
                "model": "m",
                "api_key_env": f"{name.upper()}_API_KEY",
            }
            for name in set(scene_providers.values()) - {"mock"}
        },
    }
    local = {"provider": "in_memory", "params": {}}
    return Config(
        llm=llm,
        embedding=local, vector_store=local, agent_store=local,
        message=local, snapshot=local,
        world={"config": "tiled"},
        observability={"enabled": False},
    )


def _gates(config):
    from core.container import _build_rate_gates

    return _build_rate_gates(config)


def test_two_accounts_do_not_share_a_gate() -> None:
    """Kimi's 429 is no reason to put DeepSeek calls to sleep, and rpm/tpm are counted per vendor
    anyway."""
    gates = _gates(_config_with(agent_decision_main="kimi", world_pressure="deepseek"))

    assert set(gates) == {"kimi", "deepseek"}
    assert gates["kimi"] is not gates["deepseek"]


def test_scenes_on_one_provider_share_its_gate() -> None:
    """Calls to the same endpoint must back off together; backing off separately keeps adding load
    while it's shedding."""
    gates = _gates(_config_with(agent_decision_main="qwen", world_pressure="qwen"))

    assert set(gates) == {"qwen"}


def test_a_provider_that_hits_no_account_gets_no_gate() -> None:
    """``mock`` counts against no account quota; a gate for it is pointless overhead and
    misleading."""
    assert _gates(_config_with()) == {}


def test_the_embedding_never_shares_an_llm_gate() -> None:
    """The embedding endpoint has its own limit even with the same key as chat; a 429 on one side
    shouldn't stall the other.

    It doesn't even enter this table: the keys are ``llm.providers`` entry names, so an LLM
    endpoint named embedding would compete for the slot and a collision would silently drop rpm/tpm.
    """
    from core.container import _embedding_gate

    config = _config_with(agent_decision_main="qwen")
    gates = _gates(config)

    assert set(gates) == {"qwen"}
    assert _embedding_gate(config) is not gates["qwen"]


def test_an_llm_entry_named_embedding_keeps_its_own_limits() -> None:
    """The namespaces are separate: an LLM endpoint that happens to be named embedding keeps its own
    rpm. With one shared table it would be overwritten by the embedding gate, which only carries
    cooldown, and silently lose its limit."""
    config = _config_with(agent_decision_main="embedding")
    config.llm.providers["embedding"].rpm = 42

    assert _gates(config)["embedding"]._rpm_limit == 42


def test_whether_an_embedding_needs_a_gate_is_self_reported(monkeypatch) -> None:
    """Whether a provider is networked is self-reported by whether it reads credentials. No
    provider→networked table here: switching embedding vendors would require remembering to update
    it, and missing that silently loses backpressure."""
    from core.container import _is_networked
    from providers.embedding.dashscope import DashScopeEmbeddingProvider
    from providers.embedding.in_memory import InMemoryEmbeddingProvider

    monkeypatch.setenv("DASHSCOPE_EMBED_KEY", "k")
    networked = DashScopeEmbeddingProvider(api_key_env="DASHSCOPE_EMBED_KEY")

    assert _is_networked(networked) is True
    assert _is_networked(InMemoryEmbeddingProvider(dimension=4)) is False


# ---------------------------------------------------------------------------
# The unit of rate limiting is the endpoint, not the account: vendors set RPM per model
# ---------------------------------------------------------------------------


def _two_model_config():
    """One account, two models, different RPM each: how vendors actually allocate quota per
    model."""
    from config.models import Config
    from core.interfaces.llm import LLMScene

    local = {"provider": "in_memory", "params": {}}
    return Config(
        llm={
            "providers": {
                "fast": {"base_url": "https://v/v1", "model": "f",
                         "api_key_env": "K", "rpm": 50},
                "pro": {"base_url": "https://v/v1", "model": "p",
                        "api_key_env": "K", "rpm": 5},
            },
            "default_provider": "fast",
            "scenes": {LLMScene.WORLD_BUILDING.value: {"provider": "pro"}},
        },
        embedding=local, vector_store=local, agent_store=local,
        message=local, snapshot=local,
        world={"config": "tiled"}, observability={"enabled": False},
        rate_limit={"tpm": 500000},
    )


def test_two_models_on_one_account_do_not_share_a_gate() -> None:
    """Vendors set RPM per model, so the unit of rate limiting is the endpoint, not the account:
    two models on one key each have their own quota, and exceeding one is no reason to stop the
    other."""
    gates = _gates(_two_model_config())

    assert set(gates) == {"fast", "pro"}
    assert (gates["fast"]._rpm_limit, gates["pro"]._rpm_limit) == (50, 5)


def test_an_endpoint_without_its_own_numbers_takes_the_account_defaults() -> None:
    """Unset means the account-level ``rate_limit`` default: most deployments connect to one vendor,
    no need to repeat it per entry."""
    config = _two_model_config()
    config.llm.providers["fast"].rpm = None

    gates = _gates(config)

    assert gates["fast"]._rpm_limit == config.rate_limit.rpm   # default 0 = unlimited
    assert gates["fast"]._tpm_limit == 500000                  # tpm: account level


def test_scenes_on_one_endpoint_share_its_gate(monkeypatch) -> None:
    """Quota belongs to the endpoint, not the scene: giving each scene its own 50 RPM is no limit at
    all."""
    from core.interfaces.llm import LLMScene

    monkeypatch.setenv("K", "test-key")
    config = _two_model_config()
    gates = _build_scene_gates(config)
    on_fast = {id(gates[s]) for s in LLMScene if s is not LLMScene.WORLD_BUILDING}

    assert len(on_fast) == 1
    assert gates[LLMScene.WORLD_BUILDING] is not gates[LLMScene.AGENT_DECISION_MAIN]


def _build_scene_gates(config):
    from core.container import _build_llm_router, _build_rate_gates
    from providers.llm.catalog import resolve_catalog

    gates = _build_rate_gates(config)
    router = _build_llm_router(config, resolve_catalog(config.llm.providers), rate_gates=gates)
    return router._rate_gates
