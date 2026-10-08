from __future__ import annotations

import pytest

from core.factory import ComponentKind, ProviderFactory


def test_factory_rejects_unknown_provider() -> None:
    try:
        ProviderFactory.create("does_not_exist")
    except ValueError as exc:
        assert "Unknown provider" in str(exc)
    else:
        raise AssertionError("ProviderFactory should reject unknown providers")


def test_factory_rejects_unknown_provider_with_kind() -> None:
    with pytest.raises(ValueError, match="Unknown message"):
        ProviderFactory.create("does_not_exist", kind=ComponentKind.MESSAGE)


def test_factory_requires_kind_for_ambiguous_names(container) -> None:
    with pytest.raises(ValueError, match="Ambiguous provider name"):
        ProviderFactory.create("in_memory")


def test_factory_creates_scoped_provider_when_kind_is_supplied(container) -> None:
    provider = ProviderFactory.create("in_memory", kind=ComponentKind.MESSAGE)

    assert type(provider).__name__ == "InMemoryMessageProvider"
