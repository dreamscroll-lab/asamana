"""Scaffold shared by every stage validation suite (``tuning/validation*.py``).

A suite owns its scenarios, its per-scenario run and its deterministic checks; everything
around them — running scenarios concurrently, turning a crashed scenario into a failed row,
averaging judge scores, writing ``summary.{json,md}`` — lives here, so a fix lands in every
suite at once.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

from config.models import Config
from core.interfaces.llm import LLMProvider, LLMRouter, LLMScene
from core.interfaces.trace import TraceSink
from core.logging import get_logger

from tuning.trace import write_json

logger = get_logger(__name__)

#: Code-layer time coordinates leaking into narrative text: "第3步", "3个步骤", "step=12", "steps". Sim
#: steps are always interpolated as Arabic numerals, so only digit-led forms count — "一步 / 三步 / 脚步" are narrative words, not leaks.
STEP_LEAK = re.compile(r"\d+\s*个?步|\bsteps?\b", re.IGNORECASE)

#: One column = (header, function that pulls the cell from a scenario row).
Column = tuple[str, Callable[[dict], Any]]


def text_leak(
    text: Any, *, agent_ids: Iterable[str], location_ids: Iterable[str] = ()
) -> str | None:
    """Describe the first agent id / location id / bare step found in ``text``, else None."""
    t = str(text or "")
    for aid in agent_ids:
        if aid and aid in t:
            return f"泄漏 agent id「{aid}」"
    for lid in location_ids:
        if lid and lid in t:
            return f"泄漏 location id「{lid}」"
    if STEP_LEAK.search(t):
        return f"出现裸 step→「{t[:24]}…」"
    return None


def make_judge_router(judge_provider: LLMProvider, sink: TraceSink, config: Config) -> LLMRouter:
    """A router that sends every scene to the judge, traced into the scenario's sink."""
    return LLMRouter(
        {scene: judge_provider for scene in LLMScene},
        trace_sink=sink, max_concurrent=config.engine.max_concurrent_llm,
    )


def load_scenarios(scenarios_path: str | None, default: Path) -> list[dict]:
    path = Path(scenarios_path) if scenarios_path else default
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def focus_of(row: dict) -> str:
    return "/".join(row.get("criteria_focus", []))


def aggregate(
    world_id: str, rows: list[dict], criteria: Sequence[str], **extra: Any
) -> dict[str, Any]:
    """Average each criterion over the rows that scored it (0 = not judged / not applicable)."""
    avg = {}
    for c in criteria:
        scores = [r["scores"][c] for r in rows if r["scores"].get(c, 0) > 0]
        avg[c] = round(sum(scores) / len(scores), 2) if scores else 0
    return {
        "world_id": world_id,
        "generated_at": datetime.now().isoformat(),
        **extra,
        "scenario_count": len(rows),
        "criteria_avg": avg,
        "all_deterministic_passed": all(r["deterministic_passed"] for r in rows),
        "scenarios": rows,
    }


def summary_md(
    summary: dict, *, title: str, avg_label: str, columns: Sequence[Column],
    header_extra: str = "",
) -> str:
    avg = summary["criteria_avg"]
    lines = [
        f"# {title} 验证汇总",
        f"- world_id: `{summary['world_id']}`",
        f"- 生成时间: {summary['generated_at']}",
        f"- 场景数: {summary['scenario_count']}  ·  "
        f"确定性全过: {'✅' if summary['all_deterministic_passed'] else '❌'}{header_extra}",
        f"- {avg_label}: " + " · ".join(f"{c}={v}" for c, v in avg.items()),
        "",
        "| 场景 | " + " | ".join(h for h, _ in columns) + " | 确定性 | 总评 |",
        "|---|" + "---|" * len(columns) + "---|---|",
    ]
    for r in summary["scenarios"]:
        cells = [str(cell(r)) for _, cell in columns]
        det = "✅" if r["deterministic_passed"] else "❌"
        overall = (r.get("overall", "") or "").replace("|", "/")[:50]
        lines.append(f"| {r['name']} | " + " | ".join(cells) + f" | {det} | {overall} |")
    lines.append("\n## Flagged issues")
    flagged = [r for r in summary["scenarios"] if r.get("issues")]
    for r in flagged:
        lines.append(f"\n**{r['name']}**")
        lines.extend(f"- {i}" for i in r["issues"])
    if not flagged:
        lines.append("\n（无）")
    return "\n".join(lines) + "\n"


def _failed_row(
    meta: dict, exc: BaseException, criteria: Sequence[str], default_kind: str | None
) -> dict:
    row: dict[str, Any] = {
        "name": meta["name"],
        "criteria_focus": meta.get("criteria_focus", []),
        "scores": {c: 0 for c in criteria},
        "overall": f"运行失败: {exc}",
        "semantic_judged": False,
        "deterministic_passed": False,
        "issues": [f"运行异常: {exc}"],
    }
    if default_kind is not None:
        row["kind"] = meta.get("kind", default_kind)
    return row


async def run_suite(
    stage: str,
    world_id: str,
    scenarios: list[dict],
    run_one: Callable[[dict], Awaitable[dict]],
    *,
    out_dir: Path,
    criteria: Sequence[str],
    render_md: Callable[[dict], str],
    default_kind: str | None = None,
    summary_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every scenario concurrently; a crashed scenario becomes a failed row, never aborts the suite."""
    results = await asyncio.gather(*[run_one(m) for m in scenarios], return_exceptions=True)
    rows: list[dict] = []
    for meta, res in zip(scenarios, results):
        if isinstance(res, BaseException):
            logger.warning("scenario_failed", extra={"scenario": meta["name"], "error": str(res)})
            rows.append(_failed_row(meta, res, criteria, default_kind))
        else:
            rows.append(res)
    summary = aggregate(world_id, rows, criteria, **(summary_extra or {}))
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_md(summary), encoding="utf-8")
    logger.info("validation_complete", extra={"stage": stage, "world_id": world_id, "scenarios": len(rows)})
    return summary
