from __future__ import annotations

import pytest

from core.factory import ComponentKind, ProviderFactory


def test_factory_rejects_unknown_provider_with_kind() -> None:
    with pytest.raises(ValueError, match="Unknown message"):
        ProviderFactory.create("does_not_exist", kind=ComponentKind.MESSAGE)


def test_factory_creates_scoped_provider(container) -> None:
    provider = ProviderFactory.create("in_memory", kind=ComponentKind.MESSAGE)

    assert type(provider).__name__ == "InMemoryMessageProvider"
