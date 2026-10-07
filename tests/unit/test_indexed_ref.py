"""IndexedRef: the only safe way for an LLM to reference items in a known set (see "LLM Indexed
Reference Pattern" in CLAUDE.md).

These tests pin its tolerance boundary: which forms are recognized and which must be dropped. The
costs on either side are asymmetric, so the line can't be drawn by feel:
- Too strict: a dropped ``location_scope`` doesn't mean "no place", it means "broadcast to the
  whole world"; dropped recipients mean "nobody got this urgent message".
- Too loose: guessing a digit inside a name as an index silently points at a different person,
  which is worse than dropping it.
"""

from __future__ import annotations

from core.interfaces.llm import IndexedRef

_IDS = ["id_a", "id_b", "id_c"]


def test_plain_integers_map_one_based() -> None:
    assert IndexedRef(_IDS).resolve([1, 3]) == ["id_a", "id_c"]


def test_numeric_strings_are_understood() -> None:
    """An index written as a JSON string is one of the most common forms."""
    assert IndexedRef(_IDS).resolve(["2"]) == ["id_b"]
    assert IndexedRef(_IDS).resolve([2.0]) == ["id_b"]


def test_the_hash_form_from_the_prompt_is_understood() -> None:
    """Candidates appear as ``#2`` in the prompt and models often echo that form back. Dropping it
    costs too much."""
    ref = IndexedRef(_IDS)
    for decorated in ("#2", "# 2", "第2", "2号", "2.", "2)"):
        assert ref.resolve([decorated]) == ["id_b"], decorated


def test_a_name_that_happens_to_contain_a_digit_is_not_an_index() -> None:
    """A name in a reference field is already a violation, and reading "李世民3号" as person 3 is
    worse than dropping it. Dropping gives a visible empty result; a wrong guess silently binds the
    wrong person."""
    assert IndexedRef(_IDS).resolve(["李世民3号", "西市", "", None]) == []


def test_out_of_range_and_duplicates_are_dropped() -> None:
    ref = IndexedRef(_IDS)
    assert ref.resolve([0, 4, 99, -1]) == []      # 1-based; out of range is dropped
    assert ref.resolve([2, 2, "#2"]) == ["id_b"]  # duplicates collapse


def test_an_empty_candidate_set_resolves_to_nothing() -> None:
    assert IndexedRef([]).resolve([1]) == []
