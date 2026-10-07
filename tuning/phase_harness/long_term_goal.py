"""Long-term goal revision (NeedEngine.revise_long_term_goals) dry-run"""

from __future__ import annotations

from datetime import datetime

from agent.goals import GoalEntity
from agent.personality import parse_emotion_type
from config.models import Config
from core.container import Container
from core.context import clear_log_context, set_log_context
from core.logging import get_logger

from tuning.phase_harness.common import emotion_view, isolate_agent, restore_traced
from tuning.phase_harness.scenario import resolve_ref
from tuning.trace import InMemoryTraceSink, PhaseTrace

logger = get_logger(__name__)


async def run_long_term_goal(
    container: Container,
    config: Config,
    world_id: str,
    *,
    scenario: dict | None = None,
    sink: InMemoryTraceSink | None = None,
    trace_dir: str = "./data/tuning_traces",
) -> int:
    """Isolated dry-run of the long-term-goal revision stage for a restored world.

    Drives the real production ``NeedEngine.revise_long_term_goals`` (the agent's
    first-person LLM review). The scenario injects the current long-term goals (before), short-term
    goals (reference only), emotion, and recent experience (spanning objective facts and lived
    feelings). recent_memory_texts is passed in directly (deterministic, skipping the vector-recall
    scaffolding); everything else is the agent's own state. ``life_goal`` comes from the restored
    real personality and serves as the immovable north star.

    Persistence is isolated by isolate_agent's shadow store; baseline ./data is never written.
    Restores per scenario.
    """
    trace_sink = sink if sink is not None else InMemoryTraceSink()
    traced_container, world, run_step, _ = await restore_traced(container, config, world_id, trace_sink)
    name_by_id = world.directory.all_agent_names()
    id_by_name = {v: k for k, v in name_by_id.items()}
    valid_ids = set(world.agents.keys())
    scenario = scenario or {}

    ref = scenario.get("agent")
    if ref:
        aid = resolve_ref(str(ref), id_by_name, valid_ids)
    else:
        aid = next((a for a, ag in world.agents.items() if ag.is_main_character), next(iter(valid_ids)))
    agent = world.agents[aid]
    isolate_agent(agent, name_by_id)

    emo = scenario.get("emotion")
    if isinstance(emo, dict):
        agent.personality.update_emotion(
            primary=parse_emotion_type(str(emo.get("type", "neutral"))),
            intensity=float(emo.get("intensity", 0.4)),
            valence=float(emo.get("valence", 0.0)),
            triggered_by="scene",
        )

    # Inject the current long-term goals (the starting point for review) and short-term goals (reference only).
    before_goals = [str(g) for g in scenario.get("before_long_term", []) if str(g).strip()]
    if before_goals:
        agent.personality.set_long_term_goals(list(before_goals))
    before_goals = [e.text for e in agent.personality.state.long_term_goal_entities]
    short_term = [str(g) for g in scenario.get("short_term", []) if str(g).strip()]
    if short_term:
        agent.personality.set_short_term_goal_entities([
            GoalEntity(id=f"stg-ltg-{i}", text=t, goal_type="short_term", created_step=run_step)
            for i, t in enumerate(short_term)
        ])
    life_goal_before = str(getattr(agent.personality.soul, "life_goal", "") or "")
    recent = [str(t) for t in scenario.get("recent", []) if str(t).strip()]

    soul = agent.personality.soul
    set_log_context(world_id=world_id, agent_id=aid, step=str(run_step))
    error = None
    output: list[str] | None = None
    try:
        output = await agent.need_engine.revise_long_term_goals(
            personality=agent.personality,
            recent_memory_texts=recent,
            is_main_character=agent.is_main_character,
        )
    except Exception as exc:  # noqa: BLE001 — one scenario's failure must not abort the run
        logger.warning("long_term_goal_dryrun_failed",
                       extra={"world_id": world_id, "agent_id": aid, "error": str(exc)})
        error = str(exc)
    clear_log_context()

    # revise_long_term_goals is pure computation and doesn't write personality; the dry-run shows the
    # proposal it returns (None = direction unchanged, keep before_goals).
    after_goals = list(output) if output is not None else list(before_goals)
    life_goal_after = str(getattr(agent.personality.soul, "life_goal", "") or "")
    revised = output is not None

    trace_sink.record_phase(PhaseTrace(
        world_id=world_id,
        phase="stage.long_term_goal",
        step=run_step,
        agent_id=aid,
        inputs={
            "agent_name": name_by_id.get(aid, aid),
            "is_main_character": agent.is_main_character,
            "role": getattr(soul, "role", ""),
            "core_traits": list(getattr(soul, "core_traits", [])),
            "core_values": list(getattr(soul, "core_values", [])),
            "self_image": getattr(soul, "self_image", ""),
            "life_goal": life_goal_before,
            "emotion": emotion_view(agent.personality.state.emotion),
            "before_long_term": before_goals,
            "short_term": short_term,
            "recent": recent,
        },
        outputs={
            "agent_name": name_by_id.get(aid, aid),
            "revised": revised,
            "output_goals": list(output) if output is not None else None,
            "before_long_term": before_goals,
            "after_long_term": after_goals,
            "life_goal_before": life_goal_before,
            "life_goal_after": life_goal_after,
            "error": error,
        },
        timestamp=datetime.now().isoformat(),
    ))
    logger.info("tuning_long_term_goal_dry_run_complete",
                extra={"world_id": world_id, "step": run_step, "agent_id": aid, "revised": revised})
    return run_step
