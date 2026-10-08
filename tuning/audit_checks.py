"""Deterministic pre-checks feeding the judges — what code can settle, code settles.

Two checks, each feeding one scope:

- ``check_relations`` → initialization. Are relation categories reciprocal (one side declares it,
  does the other point back)? A pure string-structure check: it understands no form-of-address
  semantics and ignores age — whether address and age agree is a semantic call left to the
  LLM judge.
- ``check_bindings``  → sass. Is the index the LLM picked in the list it was given? This is
  computable: both sides are in ``LLMCallTrace.extra`` (the output index + the engine's
  declared candidate mapping); having the judge count indices is costlier and less accurate
  than code.

The judge rules only on what code can't (semantics). Both checks' output goes into the prompt as a
deterministic pre-check section (for reference, not the final verdict).
"""

from __future__ import annotations

from typing import Any

from tuning.audit_reconstruct import AgentStepView, WorldAuditView


def _split_label(label: str) -> tuple[str, str]:
    """'兄弟:弟弟' → ('兄弟', '弟弟'); no colon → (label, ''). Pure string split, no semantics."""
    for sep in (":", "："):
        if sep in label:
            cat, _, role = label.partition(sep)
            return cat.strip(), role.strip()
    return label.strip(), ""


def check_relations(view: WorldAuditView) -> list[dict[str, Any]]:
    """Structural reciprocity check on relation categories; returns a list of findings (empty = nothing found)."""
    findings: list[dict[str, Any]] = []
    for rel in view.init.get("initial_relations", []):
        if not isinstance(rel, dict):
            continue
        a, b = str(rel.get("from", "")), str(rel.get("to", ""))
        fwd = {_split_label(str(x))[0] for x in (rel.get("labels") or [])}
        rev = {_split_label(str(x))[0] for x in (rel.get("reverse_labels") or [])}
        # Check both directions: fwd-rev alone misses the half where B declares and A doesn't point back,
        # which is just as much "a category declared by one side and not reciprocated by the other".
        for cat in fwd - rev:
            findings.append({"kind": "category_not_reciprocal", "from": a, "to": b,
                             "detail": f"{a}→{b} 有关系类别「{cat}」，但 {b}→{a} 未回指该类别"})
        for cat in rev - fwd:
            findings.append({"kind": "category_not_reciprocal", "from": b, "to": a,
                             "detail": f"{b}→{a} 有关系类别「{cat}」，但 {a}→{b} 未回指该类别"})
    return findings


def check_bindings(sv: AgentStepView) -> list[dict[str, Any]]:
    """Violations in one cell that code can settle: an index outside the list, a failed call, or unparseable output.

    Out-of-range indices are declared by the engine (``extra.dropped_indices``, recorded by
    decision.py when it parses indices). Don't recompare them here: that would copy decision.py's
    "which slot checks which list" table into a second source of truth that silently goes stale.

    The output goes into the prompt as the deterministic pre-check; none of it costs judge calls.
    """
    findings: list[dict[str, Any]] = []
    for c in sv.calls:
        for detail in c.extra.get("dropped_indices") or []:
            findings.append({"kind": "index_off_menu", "stage": c.stage, "scene": c.scene,
                             "detail": f"{c.stage}：{detail} —— 越界的序号被静默丢弃"})
        if c.parse_ok is False or not c.ok:
            findings.append({"kind": "unusable_output", "stage": c.stage, "scene": c.scene,
                             "detail": f"{c.stage}/{c.scene} 这次调用失败或输出无法解析"})
    return findings
