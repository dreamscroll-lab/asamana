"""CLI for the development-phase tuning toolkit.

    python -m tuning build "<theme>"           # run a traced world build
    python -m tuning plan "<world_id>"         # dry-run plan_step for a built world
    python -m tuning stages                    # list the stage suites as JSON
    python -m tuning validate <stage> "<world_id>"
                                               # one cognition stage suite (see `stages`)
                                               #   [--scenario a,b,c] [--scenarios <path>]
                                               #   [--judge-model <m>]
    python -m tuning audit "<world_id>"        # post-hoc world quality audit (reads data/traces, LLM judge)
                                               #   [--scope sass,mams] [--agent <id|name>] [--steps 1-10]
    python -m tuning leak "<world_id>"         # TALK dialogue cross-column leak metric (reads data/traces, LLM judge)
                                               #   [--steps 1-10] [--judge-model <m>]
    python -m tuning progress "<world_id>"     # cross-scene progress of repeated talks between the same pair (LLM judge)
                                               #   [--limit N] [--judge-model <m>]

Bootstraps its own config + container (see ``bootstrap_application`` below for why
it must not share the production entrypoint's). Reports are browsed in the web dev
tool (Developer tools → Debug), which drives ``stages`` / ``validate`` / ``audit``
as subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from config.loader import load_config
from core.container import Container
from config.models import Config
from core.logging import configure_logging, get_logger

from tuning.build_harness import run_build
from tuning.phase_harness.plan_step import run_plan_step
from tuning.scenario_store import ScenarioError
from tuning.stages import all_stages, stage, stage_keys, write_scenario_subset
from tuning.audit import run_audit
from tuning.judge_dialogue_leak import run_dialogue_leak
from tuning.judge_dialogue_progress import run_dialogue_progress

logger = get_logger(__name__)

_TRACE_DIR = "./data/tuning_traces"
# Post-hoc audit reads production runtime traces (not the tuning harness dir).
_RUNTIME_TRACE_DIR = "data/traces"


def bootstrap_application(config_path: str | None = None) -> tuple[Config, Container]:
    """Config + container for a tuning process, with **production tracing switched off**.

    Tuning calls must reach only tuning's own sink, never the world's real trace
    (`data/traces/<world>/`): audits and the observer read it, and debug calls mixed in would
    falsify its stats and trajectories with no way to tell them apart.

    `config.observability.enabled=False` before building makes `Container.from_config` install a
    NullTraceSink for `llm_router`, so whichever router a call site uses, the production channel has
    nowhere to write: guaranteed by construction, not by nobody happening to flush.

    Doesn't reuse `main.bootstrap_application`: a switch only the dev tools need would push dev
    logic into the production entrypoint.
    """
    config = load_config(config_path)
    configure_logging(**config.logging.model_dump())
    config.observability.enabled = False
    return config, Container.from_config(config)


async def _cmd_build(theme: str) -> int:
    config, container = bootstrap_application()
    world_id, world = await run_build(container, config, theme, trace_dir=_TRACE_DIR)
    print(f"Build complete: {world.analysis.world_name!r}")
    print(f"  world_id: {world_id}")
    print(f"  trace:    {_TRACE_DIR}/{world_id}/build.jsonl")
    return 0


async def _cmd_plan(world_id: str) -> int:
    config, container = bootstrap_application()
    step = await run_plan_step(container, config, world_id, trace_dir=_TRACE_DIR)
    print(f"plan_step dry-run complete for world {world_id!r} (step {step})")
    print(f"  trace: {_TRACE_DIR}/{world_id}/plan_step.jsonl")
    return 0


def _cmd_stages() -> int:
    print(json.dumps([s.as_dict() for s in all_stages()], ensure_ascii=False, indent=2))
    return 0


async def _cmd_validate(
    stage_key: str, world_id: str, scenarios: str | None, scenario: str | None,
    judge_model: str | None,
) -> int:
    spec = stage(stage_key)
    if spec is None:
        print(f"Unknown validation target {stage_key!r}. Available: {', '.join(stage_keys())}",
              file=sys.stderr)
        return 1
    # Running a few scenarios = trimming the scenario file down to them and taking the same path.
    # When tuning a prompt you want one scenario, one call, an immediate result; the full suite
    # means waiting on a dozen-plus LLM calls.
    #
    # Trim before bootstrap: a misspelled scenario name shouldn't demand an API key before telling you it's misspelled.
    picked = [n.strip() for n in (scenario or "").split(",") if n.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        if picked:
            path = Path(tmp) / f"{spec.key}.json"
            try:
                write_scenario_subset(spec, picked, path)
            except ScenarioError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            scenarios = str(path)
        config, container = bootstrap_application()
        kwargs = {"judge_model": judge_model} if judge_model and spec.judged else {}
        summary = await spec.runner(
            container, config, world_id,
            scenarios_path=scenarios, trace_dir=_TRACE_DIR, **kwargs,
        )

    avg = summary.get("criteria_avg", {})
    print(f"{spec.key} validation complete for world {world_id!r}: "
          f"{summary.get('scenario_count', 0)} scenarios")
    if spec.criteria:
        print("  semantic avg  " + "  ".join(f"{c}={avg.get(c, '—')}" for c in spec.criteria))
    if "all_deterministic_passed" in summary:
        print(f"  deterministic: {'all passed ✅' if summary['all_deterministic_passed'] else 'FAILURES ❌'}")
    print(f"  report: {_TRACE_DIR}/{world_id}/validation/{spec.key}/summary.md")
    return 0


def _parse_steps(spec: str | None) -> list[int] | None:
    """'1-10' / '3,5,7' / '4' → list[int]; None → everything."""
    if not spec:
        return None
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out)) or None


# Short scope aliases → full keys (CLI convenience; full keys are accepted too).
_SCOPE_ALIAS = {
    "sass": "single_agent_single_step",
    "sams": "single_agent_multi_step",
    "mass": "multi_agent_single_step",
    "mams": "multi_agent_multi_step",
    "init": "initialization",
}


def _or_dash(value: float | None, why: str) -> str:
    return f"{value}" if value is not None else f"—（{why}）"


def _parse_scopes(spec: str | None) -> list[str] | None:
    """'sass,mams' / 'single_agent_single_step' → [full keys]; None → all. Unknown aliases raise."""
    if not spec:
        return None
    from tuning.audit_metrics import SCOPES
    out: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        key = _SCOPE_ALIAS.get(tok, tok)
        if key not in SCOPES:
            raise SystemExit(f"unknown scope {tok!r}; valid: {sorted(SCOPES)} or aliases {sorted(_SCOPE_ALIAS)}")
        out.append(key)
    return out or None


def _parse_csv(spec: str | None) -> list[str] | None:
    """'agent-x,Li Shimin' → [tokens]; None → all."""
    if not spec:
        return None
    out = [t.strip() for t in spec.split(",") if t.strip()]
    return out or None


async def _cmd_audit(world_id: str, judge_model: str | None, steps: str | None,
                     scope: str | None, agent: str | None) -> int:
    config, container = bootstrap_application()
    kwargs = {"judge_model": judge_model} if judge_model else {}
    summary = await run_audit(
        container, config, world_id,
        steps=_parse_steps(steps), scopes=_parse_scopes(scope), agents=_parse_csv(agent),
        trace_dir=_RUNTIME_TRACE_DIR, **kwargs,
    )
    short = {v: k for k, v in _SCOPE_ALIAS.items()}
    cov = summary["coverage"]
    print(f"audit complete for world {world_id!r}: 世界总分 {_or_dash(summary['world_score'], '未取证')}"
          f"  取证覆盖 {cov:.0%}" if cov is not None else
          f"audit complete for world {world_id!r}: 世界总分 {_or_dash(summary['world_score'], '未取证')}")
    print(f"  忠实度(sass，不进世界分): {_or_dash(summary['fidelity_score'], '未跑')}"
          f"   ran: {', '.join(summary['ran_scopes'])}")
    if summary["mixed_provenance"]:
        print("  ⚠ 参与世界分的 scope 来自不同的判分标准 / judge 模型（增量合并），分数是缝出来的")
    print("  按维度（取证份额加权）:")
    for did, d in summary["dimensions"].items():
        tier = "基础" if d["category"] == "basic" else "提升"
        src = "、".join(f"{short.get(k, k)} {v['share']:.0%}" for k, v in (d["sources"] or {}).items()) or "无"
        print(f"    [{tier} {d['weight']:g}%] {d['name']}：{_or_dash(d['score'], '未取证')}"
              f"  (覆盖 {d['coverage']:.0%}；来自 {src})")
    print(f"  report: {_RUNTIME_TRACE_DIR}/{world_id}/audit/summary.json")
    return 0


async def _cmd_dialogue_leak(world_id: str, judge_model: str | None, steps: str | None) -> int:
    config, container = bootstrap_application()
    kwargs = {"judge_model": judge_model} if judge_model else {}
    s = await run_dialogue_leak(
        container, config, world_id,
        steps=_parse_steps(steps), trace_dir=_RUNTIME_TRACE_DIR, **kwargs,
    )
    def _row(label: str, d: dict) -> str:
        sr = "—" if d["scene_rate"] is None else f"{d['scene_rate']:.1%}"
        lr = "—" if d["line_rate"] is None else f"{d['line_rate']:.2%}"
        return (f"  {label}：场次 {d['scenes']}/{s['judged']} = {sr}"
                f"，句 {d['lines']}/{s['total_lines']} = {lr}")
    either = "—" if s["either_scene_rate"] is None else f"{s['either_scene_rate']:.1%}"
    print(f"dialogue leak for world {world_id!r}: {s['judged']}/{s['dialogues']} 场已判"
          f"（{s['unjudged']} 场判定未发生，不计入）")
    print(_row("越界(说出对方才知道的事)", s["crossings"]))
    print(_row("让步(丢掉自己的认定)    ", s["concessions"]))
    print(f"  两者居其一的场次：{s['either_scenes']}/{s['judged']} = {either}")
    dv = s["divergence"]; c = dv["counts"]
    vr = "—" if dv["voiced_rate"] is None else f"{dv['voiced_rate']:.1%}"
    print(f"  两栏有实质分歧的场次：{dv['scenes_with_divergence']}/{s['judged']}"
          f"（演出来 {c['voiced']} / 绕开 {c['avoided']} / 抹平 {c['flattened']}）")
    print(f"  分歧演出率：{vr}   [无分歧 {c['none']}，判不出 {c['unknown']}]")
    print(f"  report: {_RUNTIME_TRACE_DIR}/{world_id}/audit/dialogue_leak.json")
    return 0


async def _cmd_dialogue_progress(world_id: str, judge_model: str | None) -> int:
    config, container = bootstrap_application()
    kwargs = {"judge_model": judge_model} if judge_model else {}
    s = await run_dialogue_progress(container, config, world_id,
                                    trace_dir=_RUNTIME_TRACE_DIR, **kwargs)
    c = s["counts"]
    rate = "—" if s["advance_rate"] is None else f"{s['advance_rate']:.1%}"
    print(f"dialogue progress for world {world_id!r}: {s['dialogues']} 场 / {s['pairs']} 组人")
    print(f"  后续场次(N≥2)：{s['followups']}，已判 {s['judged']}（{s['unjudged']} 场判定未发生）")
    print(f"  推进 {c['advance']} / 原地重复 {c['restate']}   [判不出 {c['unknown']}]")
    print(f"  推进率：{rate}")
    top = list(s["pair_sizes"].items())[:3]
    print("  谈得最多的几对：" + "，".join(f"{k} {v}次" for k, v in top))
    print(f"  report: {_RUNTIME_TRACE_DIR}/{world_id}/audit/dialogue_progress.json")
    return 0


def _parse_flag_str(args: list[str], flag: str, *, default: str | None) -> str | None:
    for i, tok in enumerate(args):
        if tok == flag and i + 1 < len(args):
            return args[i + 1]
    return default


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python -m tuning <build|plan|stages|validate|audit|leak|progress> [args]",
              file=sys.stderr)
        return 1
    command, rest = argv[0], argv[1:]
    if command == "build":
        if not rest:
            print('Usage: python -m tuning build "<theme>"', file=sys.stderr)
            return 1
        return asyncio.run(_cmd_build(rest[0]))
    if command == "plan":
        if not rest:
            print('Usage: python -m tuning plan "<world_id>"', file=sys.stderr)
            return 1
        return asyncio.run(_cmd_plan(rest[0]))
    if command == "stages":
        return _cmd_stages()
    if command == "validate":
        if len(rest) < 2:
            print(f'Usage: python -m tuning validate <stage> "<world_id>" '
                  f'[--scenario a,b,c] [--scenarios <path>] [--judge-model <m>]\n'
                  f'  stages: {", ".join(stage_keys())}', file=sys.stderr)
            return 1
        flags = rest[2:]
        return asyncio.run(_cmd_validate(
            rest[0], rest[1],
            _parse_flag_str(flags, "--scenarios", default=None),
            _parse_flag_str(flags, "--scenario", default=None),
            _parse_flag_str(flags, "--judge-model", default=None),
        ))
    if command == "audit":
        if not rest:
            print('Usage: python -m tuning audit "<world_id>" [--scope sass,mams] '
                  '[--agent <id|name>] [--steps 1-10] [--judge-model <m>]',
                  file=sys.stderr)
            return 1
        judge_model = _parse_flag_str(rest[1:], "--judge-model", default=None)
        steps = _parse_flag_str(rest[1:], "--steps", default=None)
        scope = _parse_flag_str(rest[1:], "--scope", default=None)
        agent = _parse_flag_str(rest[1:], "--agent", default=None)
        return asyncio.run(_cmd_audit(rest[0], judge_model, steps, scope, agent))
    if command == "leak":
        if not rest:
            print('Usage: python -m tuning leak "<world_id>" [--steps 1-10] [--judge-model <m>]',
                  file=sys.stderr)
            return 1
        return asyncio.run(_cmd_dialogue_leak(
            rest[0],
            _parse_flag_str(rest[1:], "--judge-model", default=None),
            _parse_flag_str(rest[1:], "--steps", default=None),
        ))
    if command == "progress":
        if not rest:
            print('Usage: python -m tuning progress "<world_id>" [--judge-model <m>]', file=sys.stderr)
            return 1
        return asyncio.run(_cmd_dialogue_progress(
            rest[0], _parse_flag_str(rest[1:], "--judge-model", default=None)))
    print(f"Unknown command: {command!r}", file=sys.stderr)
    return 1
