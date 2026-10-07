"""How retrieval candidates are scored and selected: R/I/R weighting, the relevance floor, MMR.

Pure functions over ``Memory`` records. ``RankedMemory`` keeps every factor so the development
tools can explain a ranking without a second, drifting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict

from agent.memory_types import Memory, MemoryKind
from core.interfaces.vector_store import SearchResult


# Fixed R/I/R weights shared by every caller; no per-situation dynamic weights (YAGNI).
#
# All three factors contribute: raising relevance only makes recalled memories older, and zeroing
# recency or importance lowers the hit rate. Don't justify raising relevance with offline recall
# metrics: their semantic/proper-noun ground truth can't penalize forgetting what just happened,
# which is what recency (working memory) is for.
#
# TODO(retrieval-weights): if runtime shows one fixed set can't serve different situations
#   (crisis favoring importance, social favoring relevance, …), introduce mode-based dynamic
#   weights with evidence, and actually wire them in.
RETRIEVAL_WEIGHTS: Dict[str, float] = {"relevance": 0.7, "recency": 0.2, "importance": 0.1}

# Hybrid retrieval (dense_weight, sparse_weight). Sums to 1, so fused is a [0,1] weighted average
# (sparse_similarity is cosine, not an unbounded dot product). Neither weight may be zero: either
# pure stream does worse.
#
# Don't weight the two memory streams separately: production queries are long semantic rewrites
# where sparse is weak, so leaning factual on sparse would depress factual scores, and the
# streams' fused scores would land on different scales under one RETRIEVAL_SCORE_FLOOR.
# Coupled to RETRIEVAL_SCORE_FLOOR: don't tune one without the other.
VECTOR_WEIGHTS: tuple[float, float] = (0.7, 0.3)

# Relevance floor on the fused score, applied *before* min-max normalization (after which no
# absolute floor means anything). Don't gate on dense: an exact proper-noun hit can have high
# sparse and middling dense, and those are the hits most worth keeping.
#
# It only filters junk; it isn't a relevance threshold. For "what I know about someone", use an
# identity-scoped constraint (recall_about_agent), not a higher number.
#
# Don't raise it: from 0.4 up it drops true answers steeply; by 0.5 many queries recall nothing.
RETRIEVAL_SCORE_FLOOR: float = 0.3


@dataclass
class RankedMemory:
    """One candidate's intermediate ranking values, which double as diagnostics.

    Production reads only ``memory`` / ``score``; the rest lets the recall inspector explain a
    ranking without a separate shadow scorer that would drift.

    dropped: ``None`` = selected; ``"floor"`` = below the relevance floor; ``"mmr_duplicate"`` =
    near-duplicate of selected content; ``"top_k"`` = scored too low to make the cut.
    """

    memory: Memory
    dense: float          # raw cosine
    sparse: float         # raw sparse overlap score
    fused: float          # dense×w_d + sparse×w_s; raw relevance used for ranking
    relevance: float = 0.0   # these three are min-max normalized within this candidate set
    recency: float = 0.0
    importance: float = 0.0
    score: float = 0.0       # final score after weighting + decay
    dropped: str | None = None


def mmr_select(ranked: list[RankedMemory], *, top_k: int) -> list[RankedMemory]:
    """Select high-scoring memories while filtering near-duplicates; marks each item's
    ``dropped`` in place for the dev tools."""

    selected: list[RankedMemory] = []
    seen_contents: list[str] = []
    for item in sorted(ranked, key=lambda r: r.score, reverse=True):
        if len(selected) >= top_k:
            item.dropped = "top_k"
            continue
        if any(simple_similarity(item.memory.stored_content, seen) > 0.9 for seen in seen_contents):
            item.dropped = "mmr_duplicate"
            continue
        item.dropped = None
        selected.append(item)
        seen_contents.append(item.memory.stored_content)
    return selected


def _minmax(values: list[float]) -> list[float]:
    """Linearly normalize values to [0,1] (all equal → all 0.5)."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _raw_recency(memory: Memory, current_step: int) -> float:
    """Raw recency ``0.99^Δstep`` before normalization: retrieval's model of working memory.

    ref_step prefers last_accessed_step: rehearsal raises recency, which is how obsession
    persists. decay_score stays out of recency (the decay floor handles it). Summaries get ×0.7
    so one repeatedly revived by touch() can't outrank genuinely recent events. Insights get no
    extra bias: importance, decay protection and link-aware dedup already cover them.
    """
    ref_step = memory.last_accessed_step if memory.last_accessed_step >= 0 else memory.created_step
    recency = 0.99 ** max(0, current_step - ref_step)
    if memory.kind == MemoryKind.SUMMARY:
        recency *= 0.7
    return recency


def score_and_rank(
    candidates: list[tuple[Memory, SearchResult]], current_step: int, weights: dict[str, float]
) -> list[RankedMemory]:
    """Per-query min-max normalization of the three factors → linear combination → decay floor.

    Normalize because the factors live on different scales and relevance has the smallest
    variance: added raw, the nominally heaviest weight would be the weakest. Linear, not a
    product: any one high enough factor can lift a memory. Decay floor: decay_score < 0.2
    multiplies the score by it, so deeply decayed memories sink but are never cut off.
    """

    if not candidates:
        return []
    rel_n = _minmax([float(r.score) for _, r in candidates])
    rec_n = _minmax([_raw_recency(m, current_step) for m, _ in candidates])
    imp_n = _minmax([float(m.importance) for m, _ in candidates])
    out: list[RankedMemory] = []
    for (memory, result), rn, ren, in_ in zip(candidates, rel_n, rec_n, imp_n):
        score = weights["relevance"] * rn + weights["recency"] * ren + weights["importance"] * in_
        if memory.decay_score < 0.2:
            score *= memory.decay_score
        out.append(
            RankedMemory(
                memory=memory,
                dense=float(result.dense_score),
                sparse=float(result.sparse_score),
                fused=float(result.score),
                relevance=rn,
                recency=ren,
                importance=in_,
                score=score,
            )
        )
    return out


def simple_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()
