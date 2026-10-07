"""Tuning judge endpoint resolution: the vendor comes from config's endpoint catalog; the model can
change per run."""

from __future__ import annotations

import pytest

from config.models import LLMConfig
from tuning.judge_llm import create_judge


def _config(**judge: object) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(llm=LLMConfig(
        providers={
            "house": {"base_url": "https://house.invalid/v1", "model": "m",
                      "api_key_env": "JUDGE_KEY", "params": {"enable_thinking": False}},
        },
        default_provider="house",
        judge={"provider": "house", "model": "strict-v1", **judge},
    ))


def test_the_judge_comes_from_the_configured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_KEY", "k")

    judge = create_judge(_config())

    assert judge.model == "strict-v1"
    assert str(judge._client.base_url).rstrip("/") == "https://house.invalid/v1"


def test_a_per_run_model_changes_the_model_not_the_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The page/CLI passes a model name. The vendor follows the credentials; changing it means
    editing config."""
    monkeypatch.setenv("JUDGE_KEY", "k")

    judge = create_judge(_config(), "other-v2")

    assert judge.model == "other-v2"
    assert str(judge._client.base_url).rstrip("/") == "https://house.invalid/v1"


def test_the_caller_params_win_over_the_endpoint_dialect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether a judge thinks is that judge's call; the endpoint dialect is only the default."""
    monkeypatch.setenv("JUDGE_KEY", "k")

    judge = create_judge(_config(), enable_thinking=True)

    assert judge._call_params["enable_thinking"] is True


def test_a_missing_key_says_which_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUDGE_KEY", raising=False)

    with pytest.raises(ValueError, match="JUDGE_KEY"):
        create_judge(_config())
