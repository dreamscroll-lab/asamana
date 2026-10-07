"""Audit dimensions (the director's 7-dimension review) + scope evaluation mechanism.

The scoring dimensions are a fixed set of 7 (4 basic + 3 quality) with global weights summing to
100; together they make up the final world quality score:

  basic (75%)    1 persona & worldview consistency 20  2 causal coherence 20
                 3 action efficacy & social yield 20   4 narrative closure & convergence 15
  quality (25%)  5 dramatic arc & conflict intensity 15  6 emergent interest & surprise 5
                 7 character arc & growth 5

A scope is only the granularity of evidence (single/multi agent × single/multi step +
initialization). One dimension needs several scopes to be seen in full (causal "amnesia" spans
steps, "ineffective socializing" spans agents, closure needs the global view). So each scope
scores the dimensions it can see, and the final result aggregates per dimension.

Aggregation has two levels, one ladder all the way up; scope totals and the world score both
derive from it:

  L1 each judge call's score
  L2 scope_dim = mean of every score for this dimension within the scope  — call counts stop here
  L3 dim       = Σ evidence share × L2 / Σ share                            — shares in ``_EVIDENCE_WEIGHTS``
  L4 world     = Σ effective weight × L3 / Σ effective weight               — effective weight = global weight × coverage
     scope total = Σ global weight × L2 (that scope) / Σ weight             — same L2, not comparable across scopes

L2 is the key: without collapsing to one number, "how many agents, how many steps" would quietly
reweight the scopes.

- Dimension names/tiers/weights/inspection points are fixed in ``DIMENSIONS``; scope → dimension
  coverage is fixed in ``_SCOPE_DEFS``; each scope's evidence share within a dimension is fixed in
  ``_EVIDENCE_WEIGHTS`` (``_validate_evidence_weights`` keeps the three tables aligned).
- The scoring criteria for each (scope, dimension) (negative red flags, as a list) are configured
  in ``audit_criteria.yaml``.
- Not applicable → null: dimensions the judge can't assess for this fragment (e.g. closure in a
  sliced audit, growth when nothing happened) are returned as null / omitted and drop out of the
  weighting (no cold deduction). See the judge prompt.
- A scope that didn't score a dimension drops out of that dimension's weighting; the remaining
  scopes are renormalized and a score still comes out, but ``coverage`` is reported alongside.
  Two audits with different coverage aren't comparable. Corroborating scopes' shares are capped,
  see ``_shares``. Three causes, reported separately: not run (not recorded), scoring version
  changed (``stale``, rerun), asked but not answered (``abstained``, check criteria / prompt).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from core.logging import get_logger

logger = get_logger(__name__)

Unit = str  # "stage" | "step" | "none" — the inner scoring unit


# ---------------------------------------------------------------------------
# The fixed 7 review dimensions (global weights sum to 100)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditDimension:
    id: str
    name: str
    category: str   # "basic" | "quality"
    weight: float   # global weight (%); the 7 dimensions sum to 100
    inspect: str    # core inspection point (a one-line fault-finding guide)


DIMENSIONS: dict[str, AuditDimension] = {
    "persona_worldview": AuditDimension(
        "persona_worldview", "人设与世界观一致性", "basic", 20,
        "言行是否偏离其性格设定 / 身份 / 世界观常识（含时间线常识、已死者不复现）"),
    "causal_coherence": AuditDimension(
        "causal_coherence", "因果逻辑连贯性", "basic", 20,
        "行动 / 情绪 / 需求能否从其感知 / 动机 / 记忆找到明确因果来源,不凭空、不失忆、不臆造事实"),
    "action_efficacy": AuditDimension(
        "action_efficacy", "行动有效性与社交率", "basic", 20,
        "是否无意义空转 / 重复打转;社交是否真正改变了世界状态或关系,而非流于形式"),
    "narrative_convergence": AuditDimension(
        "narrative_convergence", "叙事终局性与收敛度", "basic", 15,
        "是否朝核心冲突 / 主题 / 长期目标收敛推进,而非无限发散、各行其是或戛然而止"),
    "dramatic_arc": AuditDimension(
        "dramatic_arc", "戏剧性弧线与冲突强度", "quality", 15,
        "情绪 / 压力 / 冲突曲线是否有起伏张弛与节奏,而非一条平直线"),
    "emergent_novelty": AuditDimension(
        "emergent_novelty", "涌现趣味与意料之外", "quality", 5,
        "是否有认知差 / 情报差带来的误会 / 反转 / 博弈,而非俗套直线剧本"),
    "character_growth": AuditDimension(
        "character_growth", "角色弧光与成长性", "quality", 5,
        "经历重大变故后,底层情绪 / 需求 / 行为模式是否发生可见质变"),
}


# ---------------------------------------------------------------------------
# Fidelity metrics (sass only): not a narrative dimension and not part of the world score
# ---------------------------------------------------------------------------
#
# sass audits whether the cognition machinery thinks from what it was given (output ↔ its own
# input): engineering quality, not narrative quality, so it has its own metrics.
#
# It stays out of the world score structurally: ``aggregate_dimensions`` iterates only
# ``DIMENSIONS``, which doesn't hold these ids. Its weight is carried by fidelity_score, reported
# alongside; a share of the world score would let engineering quality dilute narrative quality.
#
# weight here is a relative weight within the scope (outside the global 100 budget), used only for sass's own total.
FIDELITY_METRICS: dict[str, AuditDimension] = {
    "grounding": AuditDimension(
        "grounding", "言必有据", "basic", 60,
        "输出断言的事实,能否在这次调用拿到的输入里找到出处;不凭空添人 / 添物 / 添往事"),
    "instruction_adherence": AuditDimension(
        "instruction_adherence", "照章办事", "basic", 40,
        "输出本身守没守规矩:该有的口吻在不在、字段之间自不自洽、给的是不是这次要它给的东西"),
}

# Metric lookup table: the seven narrative dimensions + the fidelity metrics. ``_SCOPE_DEFS`` looks both up by id the same way.
METRICS: dict[str, AuditDimension] = {**DIMENSIONS, **FIDELITY_METRICS}


@dataclass(frozen=True)
class AuditMetric:
    """One dimension scored within a scope: the dimension's fixed info + the scope's own criteria."""

    id: str
    name: str
    category: str
    weight: float
    inspect: str
    criteria: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AuditScope:
    key: str
    label: str
    purpose: str
    metrics: list[AuditMetric]
    unit: Unit


# ---------------------------------------------------------------------------
# scope → the dimensions it can provide evidence for (order = display order); criteria come from YAML per (scope, dimension)
# (key, Chinese label, purpose, unit, [dimension_ids])
# ---------------------------------------------------------------------------

_SCOPE_DEFS: list[tuple[str, str, str, Unit, list[str]]] = [
    # What only sass can see: it's the only scope that sees what this call was fed (the engine declares
    # it through ``given_facts`` / candidate lists; see the producer table in audit_reconstruct). So
    # it judges fidelity, not narrative; its metrics are separate (FIDELITY_METRICS) and don't enter the world score.
    ("single_agent_single_step", "1 agent · 1 step",
     "一次思考内部:输出是否忠于它自己的输入(不凭空添事实、不越出给定的框)",
     "stage", ["grounding", "instruction_adherence"]),
    # What only sams can see: the timeline, whether a person is the same person over time. Here
    # persona_worldview judges drift across steps (gradually becoming someone else, crossing hard
    # constraints); sass's dimension of the same name judges OOC within one step. Two different things.
    # unit=none: growth / arc / loops / cross-step causal continuity are properties of the whole
    # trajectory, not per step. Scoring the whole fits them, and avoids a step × dimension grid big
    # enough to push the judge into long thinking and time out.
    ("single_agent_multi_step", "1 agent · N steps",
     "多步中某 agent 的人设一致 / 因果连贯 / 有效性 / 成长 / 弧线",
     "none", ["persona_worldview", "causal_coherence", "action_efficacy",
              "character_growth", "dramatic_arc"]),
    # What only mass can see: the cross-agent view of what holds within a single step — who knows what
    # (no knowledge from nowhere), whom reactions land on, whether actions in the same environment
    # contradict. sass/sams have one agent and can't compare; mams is too coarse.
    # It doesn't judge action_efficacy: whether an action worked requires seeing its result, and
    # actions often span steps with results landing in a different step. sams (whole trajectory) and
    # mams (whole world) see this dimension in full; a mass vote would only misjudge multi-step actions as idling.
    ("multi_agent_single_step", "N agents · 1 step",
     "单步内 agent 之间:谁知道什么、反应打在谁身上、同环境行动是否互不矛盾",
     "none", ["causal_coherence"]),
    ("multi_agent_multi_step", "N agents · N steps", "整体叙事:收敛 / 弧线 / 涌现 / 有效性 / 世界观",
     "none", ["narrative_convergence", "dramatic_arc", "emergent_novelty",
              "action_efficacy", "persona_worldview"]),
    ("initialization", "Initialization", "开局设定:人设 / 关系自洽 + 历史时间线",
     "none", ["persona_worldview", "causal_coherence"]),
]

_DEFAULT_CRITERIA_PATH = Path(__file__).parent / "audit_criteria.yaml"


def _load_criteria(path: Path) -> dict:
    """Read audit_criteria.yaml → {scope: {dimension: {criteria:[...]}}}. A missing or broken file raises.

    Don't degrade to empty: the criteria are the deduction checklist ("matches → deduct"). Without
    them the judge has only dimension names and can only give a gut-feel score, which isn't this
    scoring system any more. A config error fails immediately per Rule 3.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"audit criteria file missing: {path}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"audit criteria file unparseable: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"audit criteria file must be a mapping: {path}")
    return data


def build_scopes(criteria_path: Path | None = None) -> dict[str, AuditScope]:
    """Build SCOPES from the code-fixed scope → dimension structure + the YAML-configured criteria.

    Every (scope, dimension) must have criteria; a single missing one raises. The two tables live
    apart (Python and YAML), and a dimension missing criteria would reach the prompt as a bare name:
    deduction scoring silently stops for it while scores look normal. Don't turn this back into
    "degrade if missing".
    """
    path = criteria_path or _DEFAULT_CRITERIA_PATH
    cfg = _load_criteria(path)
    scopes: dict[str, AuditScope] = {}
    missing: list[str] = []
    for key, label, purpose, unit, dim_ids in _SCOPE_DEFS:
        sc_cfg = cfg.get(key) or {}
        metrics: list[AuditMetric] = []
        for did in dim_ids:
            d = METRICS[did]
            criteria = [str(c) for c in ((sc_cfg.get(did) or {}).get("criteria") or [])]
            if not criteria:
                missing.append(f"{key}.{did}")
            metrics.append(AuditMetric(
                id=d.id, name=d.name, category=d.category, weight=d.weight, inspect=d.inspect,
                criteria=criteria))
        scopes[key] = AuditScope(key, label, purpose, metrics, unit)
    if missing:
        # Report everything at once: raising on one missing entry at a time would take several runs to see how many are missing.
        raise ValueError(
            f"no scoring criteria for {', '.join(missing)} in {path}; "
            "every dimension a scope evaluates needs its own list of failure modes")
    return scopes


SCOPES: dict[str, AuditScope] = build_scopes()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_metric_block(metrics: list[AuditMetric]) -> str:
    """Dimension list: name (tier) + inspection point + scoring criteria as a list.

    Weights aren't injected: the judge scores each dimension 0-100 independently, and weights are only
    used in code for the world score. Showing them could be misread as "this dimension matters more →
    deduct harder/softer". The tier (basic/quality) stays, since the system prompt calibrates
    severity by it.
    """
    lines: list[str] = []
    for i, m in enumerate(metrics, 1):
        tier = "基础" if m.category == "basic" else "提升"
        lines.append(f"{i}. {m.name}（{m.id}·{tier}）")
        lines.append(f"   审查点：{m.inspect}")
        lines.append("   命中以下任一即扣分：")
        lines.extend(f"     - {c}" for c in m.criteria)
    return "\n".join(lines)


def output_schema(metrics: list[AuditMetric], unit: Unit) -> str:
    cells = ", ".join(f'"{m.id}": 0-100或null' for m in metrics)
    if unit == "none":
        scores = "{" + cells + "}"
    else:
        placeholder = "<拍的标题，原样抄评分对象里的「拍：」那一行>" if unit == "stage" else "<步号>"
        tail = "拍" if unit == "stage" else "步"
        scores = '{"' + placeholder + '": {' + cells + "}, ...每个" + tail + "一项}"
    return '{"rationale": "先写打分理由（指出被扣分处与扣分幅度）", "scores": ' + scores + "}"


def clamp(v: object) -> int | None:
    """0-100 integer scale; non-integer/missing/null → None (treated as unscored/not applicable, skipped when aggregating)."""
    try:
        s = int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return max(0, min(100, s))


def weighted_total(scores: dict, metrics: list[AuditMetric], unit: Unit) -> float | None:
    """Weighted total of one call in this scope (0-100, by dimension weight); only for secondary scope-level display."""
    w = {m.id: m.weight for m in metrics}
    num = den = 0.0
    cells = scores.values() if unit != "none" else [scores]
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        for m in metrics:
            s = clamp(cell.get(m.id))
            if s is None:
                continue
            num += w[m.id] * s
            den += w[m.id]
    return round(num / den, 1) if den else None


# ---------------------------------------------------------------------------
# Evidence shares: how much say each scope has within each dimension (each dimension sums to 100)
# ---------------------------------------------------------------------------
#
# This is the only source of scope weight in the world score; sample counts must not decide it
# (see L2 in the module docstring). Shares follow what each scope alone can see: sams has the
# timeline, mams the global view, and mass only the single-step cross-section of causality — it
# structurally can't see across steps and shouldn't dominate a dimension called "coherence". sass
# isn't in the table: it audits fidelity (engineering) and has its own fidelity_score.
_EVIDENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "persona_worldview": {
        "single_agent_multi_step": 60, "multi_agent_multi_step": 20, "initialization": 20},
    "causal_coherence": {
        "single_agent_multi_step": 55, "multi_agent_single_step": 30, "initialization": 15},
    "action_efficacy": {
        "single_agent_multi_step": 60, "multi_agent_multi_step": 40},
    "narrative_convergence": {"multi_agent_multi_step": 100},
    "dramatic_arc": {"single_agent_multi_step": 50, "multi_agent_multi_step": 50},
    "emergent_novelty": {"multi_agent_multi_step": 100},
    "character_growth": {"single_agent_multi_step": 100},
}

# Corroborating only: build-time evidence's share is capped at its design value and never inflated
# by runtime scopes being absent or abstaining. It audits a different object from a different
# period and must not pass as runtime evidence (init habitually scores full marks). Freed weight is
# redistributed only among runtime scopes; with no runtime scope present, the dimension counts as
# having no evidence. mass has a narrow view but still audits the same runtime object, so it isn't on this list.
_CORROBORATING_ONLY = frozenset({"initialization"})


def _validate_evidence_weights() -> None:
    """_EVIDENCE_WEIGHTS and _SCOPE_DEFS must describe the same thing; a mismatch is a config error and raises immediately (Rule 3).

    The two tables answer "which dimensions does this scope score" and "how much of this dimension is
    it", and one missing cell silently drops a line of evidence.
    """
    declared: dict[str, set[str]] = {did: set() for did in DIMENSIONS}
    for key, _label, _purpose, _unit, dim_ids in _SCOPE_DEFS:
        for did in dim_ids:
            if did in declared:
                declared[did].add(key)
    for did, scopes in declared.items():
        weighted = set(_EVIDENCE_WEIGHTS.get(did) or {})
        if weighted != scopes:
            raise ValueError(
                f"evidence weights for {did!r} cover {sorted(weighted)} "
                f"but _SCOPE_DEFS scores it in {sorted(scopes)}")


_validate_evidence_weights()


def scope_digest(key: str, scopes: dict[str, AuditScope] | None = None) -> str:
    """A scope's scoring-version fingerprint: which dimensions it scores, their criteria, and their share of the world score.

    It must cover the scope → dimension table, not just the criteria file. Incremental merging keeps
    old scope scores in the world score, and after a dimension is added to a scope, the old file
    simply doesn't have it, which looks exactly like the judge abstaining. The table lives in
    Python, where a file fingerprint can't see it.
    """
    sc = (scopes or SCOPES)[key]
    payload = json.dumps(
        [[m.id, m.criteria, _EVIDENCE_WEIGHTS.get(m.id, {}).get(key)] for m in sc.metrics],
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Two-level aggregation (source of the world score): collapse within each scope first, then weight by evidence share
# ---------------------------------------------------------------------------

ScopeDimScores = dict[str, dict[str, list[float]]]  # dim_id -> scope_key -> [this dimension's scores within the scope]


def mean_by_scope(per_scope: ScopeDimScores) -> dict[str, dict[str, float]]:
    """L2: {dim: {scope: [scores]}} → {scope: {dim: mean}}. From here on, how many calls a scope made has no bearing on its weight."""
    out: dict[str, dict[str, float]] = {}
    for did, by_scope in per_scope.items():
        for key, vals in by_scope.items():
            if vals:
                out.setdefault(key, {})[did] = round(sum(vals) / len(vals), 1)
    return out


def _shares(weights: dict[str, float], present: set[str]) -> dict[str, float]:
    """Effective shares of the scopes present (sum = 1); empty if no runtime scope is present (the dimension has no evidence).

    Corroborating scopes (``_CORROBORATING_ONLY``) get their design share, capped; weight freed by
    absent scopes is redistributed only among runtime scopes. Otherwise, when another scope abstains,
    the build-time vote gets amplified, which is exactly the "must not pass as runtime evidence"
    problem. With every scope present, the result is the design table itself.
    """
    runtime = {k for k in present if k not in _CORROBORATING_ONLY}
    if not runtime:
        return {}
    corroborating = {k: weights[k] / 100 for k in present - runtime}
    rest = 1.0 - sum(corroborating.values())
    pool = sum(weights[k] for k in runtime)
    return {**corroborating, **{k: weights[k] / pool * rest for k in runtime}}


def aggregate_dimensions(per_scope: ScopeDimScores,
                         audited: set[str] | None = None,
                         stale: set[str] | None = None) -> tuple[dict, float | None, float | None]:
    """{dim: {scope: [scores]}} → (7-dimension scorecard, world score, evidence coverage).

    Scopes present are normalized by ``_EVIDENCE_WEIGHTS`` (see ``_shares``), so running only some
    scopes still produces a score, but coverage is reported alongside: two audits with different
    coverage aren't comparable. ``audited`` is the set of scopes with data on disk, used to report
    "ran but didn't answer" separately from "didn't run".
    """
    card: dict[str, dict] = {}
    num = den = 0.0
    cov_num = cov_den = 0.0
    for did, d in DIMENSIONS.items():
        weights = _EVIDENCE_WEIGHTS[did]
        by_scope = per_scope.get(did) or {}
        present = {k: vals for k, vals in by_scope.items() if vals and k in weights}
        total_w = sum(weights.values())
        got_w = sum(weights[k] for k in present)
        shares = _shares(weights, set(present))
        sources = {
            k: {"score": round(sum(v) / len(v), 1), "samples": len(v),
                # With empty shares (no evidence for the dimension), record the share as 0: the evidence exists but wasn't admitted, so it shouldn't appear to have a say.
                "weight": weights[k], "share": round(shares.get(k, 0.0), 3),
                # Scores from a scope whose scoring version changed still count (incremental is intended), but they weren't produced under the current version.
                "stale": k in (stale or set())}
            for k, v in present.items()
        }
        mean = (round(sum(sc["share"] * sc["score"] for sc in sources.values()), 1)
                if shares else None)
        coverage = round(got_w / total_w, 3) if total_w else 0.0
        # The three causes of a missing score look identical but call for different actions, so report them separately:
        #   not run          — the cost wasn't spent; run it if you want it
        #   version changed  — it ran, but the scoring version then didn't include this dimension, so it was never asked → rerun the scope
        #   ran, no answer   — asked but not given: the judge abstained or dropped the field → check criteria / prompt
        silent = (audited or set()) & set(weights) - set(present)
        outdated = sorted(silent & (stale or set()))
        abstained = sorted(silent - set(outdated))
        # This dimension's effective weight in the world score = design weight × fraction of evidence obtained.
        # Half the evidence means half the say: after one witness abstains the other owns the dimension,
        # which is still the best estimate, but it shouldn't weigh as much as a fully evidenced dimension.
        # coverage=0 means weight 0, so "unscored dimensions don't count" isn't a separate rule but this one's endpoint.
        effective = round(d.weight * coverage, 2)
        card[did] = {"name": d.name, "category": d.category, "weight": d.weight,
                     "effective_weight": effective, "inspect": d.inspect, "score": mean,
                     "coverage": coverage, "sources": sources,
                     "abstained": abstained, "stale": outdated}
        cov_num += d.weight * coverage
        cov_den += d.weight
        if mean is not None:
            num += effective * mean
            den += effective
    world = round(num / den, 1) if den else None
    return card, world, (round(cov_num / cov_den, 3) if cov_den else None)


def scope_totals(per_scope: ScopeDimScores,
                 scopes: dict[str, AuditScope] | None = None) -> dict[str, float | None]:
    """Each scope's total: L2 weighted by dimension weight, on the same ladder as the world score.

    Don't add a second aggregation path (like averaging each entry's total): the same data would
    give two numbers that don't agree, and scope scores would stop being a breakdown of the world
    score. Not comparable across scopes: each has a different dimension mix and denominator (mass has
    one dimension; sass uses a different family of metrics).
    """
    means = mean_by_scope(per_scope)
    out: dict[str, float | None] = {}
    for key, sc in (scopes or SCOPES).items():
        got = means.get(key) or {}
        num = den = 0.0
        for m in sc.metrics:
            if (s := got.get(m.id)) is not None:
                num += m.weight * s
                den += m.weight
        out[key] = round(num / den, 1) if den else None
    return out
