"""Dev-router tests: audit execution, the prompt playground, and the director console.

The `python -m tuning audit` subprocess itself is integration-level (a real world + LLM key) and
not unit-tested: production never imports tuning, so the only boundary to mock is the process
launch.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.interfaces.trace import LLMCallTrace
from world import WorldCatalog

from interaction.api import create_app


def _client(container, test_config) -> TestClient:
    # The dev router is gated off by default; these tests exercise it, so enable it.
    test_config.web.dev_tools_enabled = True
    return TestClient(create_app(container, test_config, catalog=WorldCatalog(None)))


def _call(call_id: str = "c-1", step: int | None = 3) -> LLMCallTrace:
    return LLMCallTrace(
        world_id="w1",
        stage="decision",
        scene="agent_decision_main",
        prompt_messages=[{"role": "system", "content": "你是甲"},
                         {"role": "user", "content": "此刻你要做什么"}],
        response_content='{"selected_index": 0}',
        temperature=0.7,
        max_tokens=500,
        input_tokens=10,
        output_tokens=5,
        model="mock",
        latency_ms=12.0,
        json_mode=True,
        timestamp=datetime.now().isoformat(),
        agent_id="agent-1",
        step=step,
        call_id=call_id,
    )


# --------------------------------------------------------------------------- #
# Prompt playground: trace-call lookup + live replay (pure production)
# --------------------------------------------------------------------------- #

def test_trace_call_detail_returns_full_record(container, test_config) -> None:
    container.trace_sink.read_calls = lambda world_id, **k: [_call("c-1")]  # type: ignore[method-assign]
    client = _client(container, test_config)
    r = client.get("/api/worlds/w1/trace/calls/c-1")
    assert r.status_code == 200
    body = r.json()
    assert body["scene"] == "agent_decision_main"
    assert body["prompt_messages"][0]["content"] == "你是甲"
    assert body["temperature"] == 0.7 and body["max_tokens"] == 500
    # The debug console uses this to restore the JSON mode of the original production call.
    assert body["json_mode"] is True


def test_trace_call_detail_404_for_unknown_id(container, test_config) -> None:
    container.trace_sink.read_calls = lambda world_id, **k: [_call("c-1")]  # type: ignore[method-assign]
    client = _client(container, test_config)
    assert client.get("/api/worlds/w1/trace/calls/nope").status_code == 404


def test_trace_calls_filter_by_call_id_prefix(container, test_config) -> None:
    """Chips and logs show only the first few characters, so filter by prefix (case-insensitive).
    Calls that don't match are dropped."""
    container.trace_sink.read_calls = lambda world_id, **k: [  # type: ignore[method-assign]
        _call("abcd1234ff"), _call("9900aabbcc"),
    ]
    client = _client(container, test_config)
    body = client.get("/api/worlds/w1/trace/calls", params={"call_id": "ABCD1234"}).json()
    assert body["filters"]["call_id"] == "ABCD1234"
    assert [c["call_id"] for g in body["groups"] for c in g["items"]] == ["abcd1234ff"]
    assert body["total"]["calls"] == 1
    # No matches gives an empty view; it doesn't quietly fall back to everything.
    miss = client.get("/api/worlds/w1/trace/calls", params={"call_id": "zzzz"}).json()
    assert miss["groups"] == [] and miss["total"]["calls"] == 0
    # Without the parameter, you get everything as usual.
    assert client.get("/api/worlds/w1/trace/calls").json()["total"]["calls"] == 2


@pytest.mark.asyncio
async def test_trace_build_names_the_figure_a_persona_call_built(container, test_config) -> None:
    """Persona-generation traces carry agent_id, which the build view uses to show names. If the
    world fails to build at this step and there's no snapshot, the name is still in the call's own
    extra."""
    from core.interfaces.snapshot import WorldSnapshot
    from engine.clock import WorldTime, WorldTimeConfig

    persona = dataclasses.replace(
        _call("p-1", step=None), stage="world_init", scene="persona_generation",
        agent_id="agent-1", extra={"agent_name": "李世民"},
    )
    container.trace_sink.read_calls = lambda world_id, **k: [persona]  # type: ignore[method-assign]
    await container.snapshot.save("w1", 0, WorldSnapshot(
        world_id="w1", step=0, timestamp=datetime.now(),
        world_time=WorldTime.from_step(0, WorldTimeConfig()).clock_payload(),
        agent_states={"agent-1": {"agent_id": "agent-1", "agent_name": "李世民"}},
        agent_relations={}, metadata={"character_profiles": {"agent-1": {"name": "李世民"}}},
    ))
    client = _client(container, test_config)

    [item] = [c for g in client.get("/api/worlds/w1/trace/build").json()["groups"] for c in g["items"]]
    assert item["agent_name"] == "李世民"
    assert item["extra"]["agent_name"] == "李世民"


def test_trace_build_filter_by_call_id_prefix(container, test_config) -> None:
    """Build-time calls have ids too, so this filter works in the build view as well."""
    container.trace_sink.read_calls = lambda world_id, **k: [  # type: ignore[method-assign]
        _call("beef0001", step=None), _call("beef0002", step=None),
    ]
    client = _client(container, test_config)
    body = client.get("/api/worlds/w1/trace/build", params={"call_id": "beef0001"}).json()
    assert [c["call_id"] for g in body["groups"] for c in g["items"]] == ["beef0001"]
    assert client.get("/api/worlds/w1/trace/build").json()["total"]["calls"] == 2


def test_trace_calls_filter_by_keywords(container, test_config) -> None:
    """Every term must match (case-insensitive), searching prompt / reply / thinking / error /
    extra."""
    plain = _call("c-plain")
    hot = dataclasses.replace(
        _call("c-hot"), response_content='{"Action": "TALK"}', extra={"dominant_need": "安全"}
    )
    container.trace_sink.read_calls = lambda world_id, **k: [plain, hot]  # type: ignore[method-assign]
    client = _client(container, test_config)

    def ids(**params: str) -> list[str]:
        body = client.get("/api/worlds/w1/trace/calls", params=params).json()
        return [c["call_id"] for g in body["groups"] for c in g["items"]]

    assert ids(q="talk") == ["c-hot"]
    assert ids(q="安全") == ["c-hot"]
    assert ids(q="你是甲") == ["c-plain", "c-hot"]
    assert ids(q="你是甲 talk") == ["c-hot"]
    assert ids(q="你是甲 不存在") == []
    assert ids(q="  ") == ["c-plain", "c-hot"]


def test_trace_build_filter_by_keywords(container, test_config) -> None:
    hot = dataclasses.replace(_call("b-hot", step=None), thinking="先想想人设")
    container.trace_sink.read_calls = lambda world_id, **k: [  # type: ignore[method-assign]
        _call("b-plain", step=None), hot,
    ]
    client = _client(container, test_config)
    body = client.get("/api/worlds/w1/trace/build", params={"q": "人设"}).json()
    assert [c["call_id"] for g in body["groups"] for c in g["items"]] == ["b-hot"]
    assert body["total"]["calls"] == 1


def test_trace_dimensions_carry_each_steps_world_time_as_an_object(container, test_config) -> None:
    """World time in the step summary has the same shape as StepEvent.world_time: label +
    hour/minute."""
    from core.interfaces.trace import StepTrace

    clock = {"iso": "step=0003 time=09:00 elapsed=0s", "hour": 9, "minute": 0, "label": "辰时"}
    container.trace_sink.read_calls = lambda world_id, **k: [_call("c-1")]  # type: ignore[method-assign]
    container.trace_sink.read_step_summaries = lambda world_id: [  # type: ignore[method-assign]
        StepTrace("w1", 3, clock, 5.0, "ts"),
    ]
    client = _client(container, test_config)
    r = client.get("/api/worlds/w1/trace/dimensions")
    assert r.status_code == 200
    (step,) = r.json()["steps"]
    assert step["world_time"] == {"label": "辰时", "hour": 9, "minute": 0}


def test_llm_replay_by_scene_returns_completion(container, test_config) -> None:
    # config.test.yaml wires every scene to the `mock` provider → deterministic.
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={
        "scene": "agent_decision_main",
        "messages": [{"role": "user", "content": "改过的提示词"}],
        "temperature": 0.5,
        "max_tokens": 100,
    })
    assert r.status_code == 200
    body = r.json()
    assert "content" in body and body["provider_spec"] == "mock"
    assert "latency_ms" in body


def test_llm_replay_carries_scene_call_params(container, test_config, monkeypatch) -> None:
    """A rerun must carry the scene's endpoint params. Without enable_thinking: false, a thinking
    endpoint spends all of max_tokens on reasoning and returns empty content, so it's no longer the
    same call."""

    from core.factory import ComponentKind, ProviderFactory

    from config.models import SceneOverride
    from core.interfaces.llm import LLMScene

    test_config.llm.scenes = {
        LLMScene.AGENT_DECISION_MAIN: SceneOverride(
            provider="mock", params={"enable_thinking": False},
        ),
    }
    seen: list[dict] = []
    real = ProviderFactory.create

    def _spy(name, *, kind=None, **kwargs):
        if kind == ComponentKind.LLM:
            seen.append(kwargs)
        return real(name, kind=kind, **kwargs)

    monkeypatch.setattr(ProviderFactory, "create", _spy)
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={
        "scene": "agent_decision_main",
        "messages": [{"role": "user", "content": "改过的提示词"}],
    })
    assert r.status_code == 200
    assert seen == [{"enable_thinking": False}]


def test_llm_replay_params_override_the_scene_params_key_by_key(container, test_config, monkeypatch) -> None:
    """Switching models often means switching endpoint dialect (some models only accept
    enable_thinking: true). Overrides are applied key by key over the scene params, unmentioned keys
    follow the scene, and the response shows what was actually sent."""
    from core.factory import ComponentKind, ProviderFactory

    from config.models import SceneOverride
    from core.interfaces.llm import LLMScene

    test_config.llm.scenes = {
        LLMScene.AGENT_DECISION_MAIN: SceneOverride(
            provider="mock", params={"enable_thinking": False, "top_p": 0.8},
        ),
    }
    seen: list[dict] = []
    real = ProviderFactory.create

    def _spy(name, *, kind=None, **kwargs):
        if kind == ComponentKind.LLM:
            seen.append(kwargs)
        return real(name, kind=kind, **kwargs)

    monkeypatch.setattr(ProviderFactory, "create", _spy)
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={
        "scene": "agent_decision_main",
        "messages": [{"role": "user", "content": "改过的提示词"}],
        "params": {"enable_thinking": True},
    })
    assert r.status_code == 200
    assert seen == [{"enable_thinking": True, "top_p": 0.8}]
    assert r.json()["params"] == {"enable_thinking": True, "top_p": 0.8}


def test_llm_replay_rejects_params_that_are_not_an_object(container, test_config) -> None:
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={
        "scene": "agent_decision_main",
        "messages": [{"role": "user", "content": "x"}],
        "params": ["enable_thinking"],
    })
    assert r.status_code == 400


def test_llm_replay_without_a_scene_falls_back_to_the_default_provider(container, test_config) -> None:
    """Trying just a model name shouldn't force picking a scene first. With no scene given, it uses
    ``default_provider``."""
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={
        "model": "some-other-model",
        "messages": [{"role": "user", "content": "改过的提示词"}],
    })

    assert r.status_code == 200
    assert r.json()["provider_spec"] == "mock/some-other-model"


def test_llm_replay_rejects_empty_messages(container, test_config) -> None:
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={"scene": "agent_decision_main", "messages": []})
    assert r.status_code == 400


def test_llm_replay_rejects_unknown_scene(container, test_config) -> None:
    client = _client(container, test_config)
    r = client.post("/api/llm/replay", json={
        "scene": "not_a_scene",
        "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Audit execution: validation (before any subprocess is spawned)
# --------------------------------------------------------------------------- #

def test_audit_judge_reports_the_configured_endpoint(container, test_config) -> None:
    """The page is filled with the judge this deployment will actually use. The vendor is reported
    but not accepted as input: endpoint URL and credentials hang off that provider declaration, so
    changing it is a deployment change."""
    from config.models import SceneOverride

    test_config.llm.judge = SceneOverride(provider="house", model="strict-v1")
    client = _client(container, test_config)

    body = client.get("/api/dev/audit/judge").json()

    assert body == {"provider": "house", "model": "strict-v1"}


def test_audit_run_404_when_world_has_no_traces(container, test_config, tmp_path) -> None:
    # The audit base dir comes from config (not the trace provider's internals).
    test_config.observability.params = {"base_dir": str(tmp_path)}  # empty root → world dir absent
    client = _client(container, test_config)
    assert client.post("/api/worlds/ghost/audit/run", json={}).status_code == 404


def test_audit_run_400_on_unknown_scope(container, test_config, tmp_path) -> None:
    test_config.observability.params = {"base_dir": str(tmp_path)}
    (tmp_path / "w1").mkdir()
    client = _client(container, test_config)
    r = client.post("/api/worlds/w1/audit/run", json={"scopes": ["sams", "bogus"]})
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]


def test_audit_clear_removes_only_the_report_dir(container, test_config, tmp_path) -> None:
    test_config.observability.params = {"base_dir": str(tmp_path)}
    world = tmp_path / "w1"
    (world / "audit").mkdir(parents=True)
    (world / "audit" / "summary.json").write_text("{}", encoding="utf-8")
    (world / "step_0001.jsonl").write_text("{}", encoding="utf-8")
    client = _client(container, test_config)

    r = client.delete("/api/worlds/w1/audit")
    assert r.status_code == 200 and r.json()["cleared"] is True
    assert not (world / "audit").exists()
    assert (world / "step_0001.jsonl").exists()  # the trace itself must not be deleted along with it
    # Idempotent: nothing to clear is still success, just cleared=False.
    assert client.delete("/api/worlds/w1/audit").json()["cleared"] is False


def test_audit_clear_rejects_world_id_escaping_the_base(container, test_config, tmp_path) -> None:
    test_config.observability.params = {"base_dir": str(tmp_path / "traces")}
    (tmp_path / "traces").mkdir()
    victim = tmp_path / "audit"
    victim.mkdir()
    client = _client(container, test_config)

    # The client doesn't normalize %2e%2e away, so the path parameter reaches world_id as-is.
    assert client.delete("/api/worlds/%2e%2e/audit").status_code == 400
    assert victim.exists()


def test_job_status_404_for_unknown_job(container, test_config) -> None:
    client = _client(container, test_config)
    assert client.get("/api/dev/jobs/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Stage suites: the registry + pre-run validation (before the subprocess)
# --------------------------------------------------------------------------- #

def test_stage_catalog_matches_the_tuning_registry(container, test_config) -> None:
    """The registry comes from `python -m tuning stages`; production doesn't import tuning.

    This also guards that boundary: if someone changed dev/stages.py to import it directly, the
    table would still come back, but nothing would check the subprocess contract (the CLI prints
    JSON) anymore.
    """
    client = _client(container, test_config)
    body = client.get("/api/dev/stages").json()
    keys = [s["key"] for s in body["stages"]]
    assert "decision" in keys and "perception" in keys
    decision = next(s for s in body["stages"] if s["key"] == "decision")
    assert decision["judged"] is True
    assert decision["criteria"] == ["fields", "action_fit", "coherence"]
    assert decision["scenarios"], "场景名取自 scenarios/decision.json"
    # perception is a purely deterministic suite (no judge), so the frontend hides the judge model
    # input.
    assert next(s for s in body["stages"] if s["key"] == "perception")["judged"] is False
    assert set(body["runs"]) == set(keys)


def test_stage_run_rejects_unknown_stage_and_scenario(container, test_config) -> None:
    client = _client(container, test_config)
    assert client.post("/api/dev/stages/nope/run", json={"world_id": "w1"}).status_code == 400
    r = client.post("/api/dev/stages/decision/run", json={"world_id": "w1"})
    assert r.status_code == 404, "世界不存在 → 404，不该去 spawn 子进程"


def test_stage_scenarios_returns_the_file_verbatim(container, test_config) -> None:
    """Scenario browsing returns the file as-is. The page displays and edits it, so the code
    shouldn't pick fields for it."""
    client = _client(container, test_config)
    body = client.get("/api/dev/stages/decision/scenarios").json()
    assert body["path"] == "tuning/scenarios/decision.json"
    names = [sc["name"] for sc in body["file"]["scenarios"]]
    assert "talk_ally" in names
    # The scenario payload comes through as-is (everything injected lives here), and top-level keys
    # other than scenarios are kept.
    assert "memories" in body["file"]["scenarios"][0]["scenario"]
    assert "_comment" in body["file"]
    assert client.get("/api/dev/stages/nope/scenarios").status_code == 400


def test_stage_scenarios_keeps_perception_knob_sets(container, test_config) -> None:
    """perception scenario files carry knob_sets, which must reach the page and survive in edited
    scenarios at run time."""
    client = _client(container, test_config)
    body = client.get("/api/dev/stages/perception/scenarios").json()
    assert "knob_sets" in body["file"]


# --------------------------------------------------------------------------- #
# Scenario library CRUD: writes tuning/scenarios/<stage>.json itself
# --------------------------------------------------------------------------- #

def _scenario_file(stage: str) -> Path:
    return Path(__file__).resolve().parents[2] / "tuning" / "scenarios" / f"{stage}.json"


@pytest.fixture
def scenario_sandbox():
    """Back up the real scenario file and restore it afterwards. CRUD tests really write to disk
    and must leave no trace."""
    import shutil

    path = _scenario_file("perception")  # perception: purely deterministic, no scoring criteria
    backup = path.read_bytes()
    yield path
    path.write_bytes(backup)
    del shutil


def test_scenario_name_must_be_a_safe_directory_name() -> None:
    """A scenario name is also a report directory name, so names with / or .. must be rejected, or
    reports get written outside the directory."""
    from tuning.scenario_store import ScenarioError, validate

    for bad in ("../escape", "a/b", "", " ", "x" * 65, "-leading"):
        with pytest.raises(ScenarioError, match="invalid scenario name"):
            validate({"name": bad}, criteria=(), stage="perception")
    assert validate({"name": "ok.name-1_2"}, criteria=(), stage="perception") == "ok.name-1_2"


def test_scenario_rejects_criteria_focus_typos() -> None:
    """A typo in criteria_focus would silently focus on nothing, so check it against the stage's
    scoring criteria."""
    from tuning.scenario_store import ScenarioError, validate

    with pytest.raises(ScenarioError, match="unknown criteria_focus"):
        validate(
            {"name": "n", "criteria_focus": ["action_fit", "nope"]},
            criteria=("fields", "action_fit", "coherence"),
            stage="decision",
        )


def test_scenario_crud_round_trip(container, test_config, scenario_sandbox) -> None:
    """Create → rename → delete, each step written to the file; top-level keys other than
    scenarios survive throughout."""
    client = _client(container, test_config)
    before = json.loads(scenario_sandbox.read_text(encoding="utf-8"))

    created = client.post(
        "/api/dev/stages/perception/scenarios",
        json={"name": "added_by_test", "description": "d", "scenario": {}},
    )
    assert created.status_code == 200, created.text
    on_disk = json.loads(scenario_sandbox.read_text(encoding="utf-8"))
    assert [s["name"] for s in on_disk["scenarios"]][-1] == "added_by_test", "追加在末尾"
    assert on_disk["knob_sets"] == before["knob_sets"], "旋钮组不能被顶掉"

    # Duplicate name → 400, rather than quietly writing two scenarios with the same name (their
    # report directories would collide).
    dup = client.post("/api/dev/stages/perception/scenarios", json={"name": "added_by_test"})
    assert dup.status_code == 400 and "already has" in dup.json()["detail"]

    renamed = client.put(
        "/api/dev/stages/perception/scenarios/added_by_test",
        json={"name": "renamed_by_test", "scenario": {}},
    )
    assert renamed.status_code == 200 and renamed.json()["name"] == "renamed_by_test"
    names = [s["name"] for s in json.loads(scenario_sandbox.read_text(encoding="utf-8"))["scenarios"]]
    assert "added_by_test" not in names and "renamed_by_test" in names

    gone = client.delete("/api/dev/stages/perception/scenarios/renamed_by_test")
    assert gone.status_code == 200
    after = json.loads(scenario_sandbox.read_text(encoding="utf-8"))
    assert after == before, "增删一轮回到原样(含键序与内容)"


def test_scenario_writes_keep_the_hand_written_formatting(scenario_sandbox) -> None:
    """Format is fixed at indent=2 / no escaping of non-ASCII / trailing newline. Otherwise editing
    one scenario in the UI produces a whole-file diff."""
    from tuning import scenario_store

    data = scenario_store.load(scenario_sandbox)
    scenario_store.save(scenario_sandbox, data)
    text = scenario_sandbox.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")
    assert '\\u' not in text, "中文必须原样,不能被转义成 \\uXXXX"
    assert text.startswith("{\n  ")


def test_stage_run_rejects_unknown_scenario_names(container, test_config) -> None:
    """Running a subset: names are checked here, not when the subprocess starts."""
    client = _client(container, test_config)
    r = client.post(
        "/api/dev/stages/decision/run",
        json={"world_id": "w1", "scenario_names": ["talk_ally", "nope"]},
    )
    assert r.status_code == 400 and "nope" in r.json()["detail"]
    bad = client.post(
        "/api/dev/stages/decision/run", json={"world_id": "w1", "scenario_names": "talk_ally"},
    )
    assert bad.status_code == 400 and "list of strings" in bad.json()["detail"]


def test_stage_report_404_before_any_run(container, test_config) -> None:
    client = _client(container, test_config)
    r = client.get("/api/dev/stages/decision/runs/ghost")
    assert r.status_code == 404
    assert "tuning validate decision" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# Director debug console: assemble a prompt / validate a response. Neither changes the world.
# --------------------------------------------------------------------------- #

async def _built_world(container, test_config, catalog: WorldCatalog) -> str:
    """A built world with persisted state — the dev routes rehydrate it like /direct does."""
    from engine.application import NarrativeApplication

    app = NarrativeApplication(container, test_config, catalog=catalog)
    world = await app.build_world("宫廷权谋", template="changan_iso")
    catalog.set_confirmed(world.world_id)
    return world.world_id


def _directive_json(**overrides) -> str:
    payload = {
        "reason": "导演要放个消息",
        "feasible": True,
        "refusal": "",
        "broadcast": {"content": "钟声大作。", "severity": "medium",
                      "location_scope": None, "phenomenon": "none"},
        "message": None,
        "mutations": [],
        "narrative_desc": "钟声大作",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_director_prompt_hands_back_the_production_prompt_and_its_menus(
    mock_build_container, test_config,
) -> None:
    """The console must test the production prompt, including the call settings (scene /
    temperature / limit). Otherwise a prompt tuned here runs under different settings in
    production."""
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config) as client:
        r = client.post(f"/api/worlds/{world_id}/director/prompt", json={"text": "敲响全城的钟"})

    assert r.status_code == 200
    body = r.json()
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "不加戏" in body["messages"][0]["content"]
    assert "敲响全城的钟" in body["messages"][1]["content"]   # the director's words reach the prompt verbatim
    assert body["scene"] == "directive" and body["json_mode"] is True
    # Read the production constant instead of copying the literal, so tuning max_tokens doesn't
    # require editing this too.
    from engine.director import _MAX_TOKENS

    assert body["temperature"] == 0.2 and body["max_tokens"] == _MAX_TOKENS
    # Index → name: the LLM only outputs indices, and this table is the only way to check who it
    # pointed at.
    assert body["menus"]["cast"] and "1" in body["menus"]["cast"]
    # Same shape as StepEvent.world_time: label is for people, hour/minute is the machine clock.
    assert set(body["world_time"]) == {"label", "hour", "minute"}
    assert body["world_time"]["label"] and isinstance(body["world_time"]["hour"], int)


@pytest.mark.asyncio
async def test_director_prompt_needs_a_directive(mock_build_container, test_config) -> None:
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    with _client(mock_build_container, test_config) as client:
        assert client.post(f"/api/worlds/{world_id}/director/prompt", json={"text": " "}).status_code == 400


@pytest.mark.asyncio
async def test_director_console_never_reaches_the_paths_that_change_the_world(
    mock_build_container, test_config, monkeypatch,
) -> None:
    """Validating a response means checking whether it would be accepted, not accepting it.

    The product paths (submit_directive / step_world) are replaced with tripwires here. If the
    console went through them, a world that should only be observed would advance a step and get a
    memory injected that can never be removed.
    """
    from engine.application import NarrativeApplication

    def _boom(*args, **kwargs):
        raise AssertionError("dev console must not touch the world")

    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)
    monkeypatch.setattr(NarrativeApplication, "submit_directive", _boom)
    monkeypatch.setattr(NarrativeApplication, "step_world", _boom)

    with _client(mock_build_container, test_config) as client:
        prompt = client.post(f"/api/worlds/{world_id}/director/prompt", json={"text": "敲响全城的钟"})
        verdict = client.post(
            f"/api/worlds/{world_id}/director/interpret", json={"response": _directive_json()},
        )

    assert prompt.status_code == 200
    assert verdict.status_code == 200
    body = verdict.json()
    assert body["accepted"] is True
    assert body["preview"] == "钟声大作"
    assert body["plan"]["channels"] == ["broadcast"]
    assert body["plan"]["broadcast"]["location"] == "全域"     # scope=null → "全域", not a bare id


@pytest.mark.asyncio
async def test_director_interpret_relays_the_refusal_a_director_would_hear(
    mock_build_container, test_config,
) -> None:
    """Rejection is first-class: the sentence the console shows is exactly what the director will
    hear."""
    catalog = WorldCatalog(None)
    world_id = await _built_world(mock_build_container, test_config, catalog)

    with _client(mock_build_container, test_config) as client:
        r = client.post(f"/api/worlds/{world_id}/director/interpret", json={
            "response": _directive_json(feasible=False, refusal="没说清是谁,请指名道姓。",
                                        broadcast=None),
        })

    assert r.status_code == 200
    assert r.json() == {"accepted": False, "reason": "没说清是谁,请指名道姓。",
                        "preview": "", "plan": None}


@pytest.mark.asyncio
async def test_director_routes_404_on_an_unknown_world(mock_build_container, test_config) -> None:
    with _client(mock_build_container, test_config) as client:
        assert client.post("/api/worlds/nope/director/prompt", json={"text": "放把火"}).status_code == 404
        assert client.post(
            "/api/worlds/nope/director/interpret", json={"response": "{}"},
        ).status_code == 404


@pytest.mark.asyncio
async def test_spawned_job_keeps_a_reference_to_its_task() -> None:
    """A background task with no reference can be garbage-collected midway, so the reference is
    kept on the job."""
    import asyncio
    import sys

    from interaction.api.dev.jobs import JobRegistry

    registry = JobRegistry()
    spawned = registry.spawn("audit", "w1", [sys.executable, "-c", "pass"])
    job = registry.get(spawned["job_id"])
    assert job is not None
    try:
        assert job.task is not None
        await job.task
        assert job.status == "completed"
    finally:
        # Once the reference is lost it can't be awaited, and since it holds the subprocess pipes
        # the loop can't close. Cleaning up like this on assertion failure keeps the test run from
        # hanging instead of failing.
        leftover = asyncio.all_tasks() - {asyncio.current_task()}
        await asyncio.gather(*leftover, return_exceptions=True)
