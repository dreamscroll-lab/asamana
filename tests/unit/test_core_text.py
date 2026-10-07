"""Unit tests for the shared slugify primitive."""

from __future__ import annotations

from core.text import slugify


def test_slugify_ascii_name() -> None:
    assert slugify("Li Shimin") == "li-shimin"


def test_slugify_collapses_non_alphanumeric() -> None:
    assert slugify("  Foo__Bar!! 99 ") == "foo-bar-99"


def test_slugify_cjk_falls_back_to_prefixed_hash() -> None:
    # CJK folds away entirely → deterministic, prefixed, value-stable fallback.
    out = slugify("李世民", fallback_prefix="agent")
    assert out.startswith("agent-")
    assert out == slugify("李世民", fallback_prefix="agent")  # stable
    assert out != slugify("李建成", fallback_prefix="agent")  # value-unique


def test_slugify_fallback_prefix_is_parameterized() -> None:
    assert slugify("张三", fallback_prefix="seed").startswith("seed-")


def test_slugify_default_prefix() -> None:
    assert slugify("。。。").startswith("item-")
