"""Movement action executor for MOVE actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.interfaces.action import ActionResult, ActionTarget, ActionType, AgentAction, Observed, Ref
from core.interfaces.directory import WorldDirectory
from core.logging import get_logger
from core.duration import describe_duration, describe_seconds
from engine.environment import IN_TRANSIT
from core.interfaces.execution import TickResult
from engine.executors.base import ActionExecutionState, ActionExecutor, Conscription
from engine.narration import (
    format_intent_clause, format_interrupt_reason_3p, format_interrupt_thought, observed_here,
    scene_line,
)
from engine.scene import observe_location

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

# Transit keys on the execution state; only this module knows them (see transit_view).
_ORIGIN_KEY = "origin"
_DESTINATION_KEY = "destination"
_PATH_KEY = "path"            # list[str]: ordered waypoints [origin, …, destination]
_ARRIVALS_KEY = "arrivals"    # list[int]: elapsed step on which path[i] is reached (arrivals[0]=0)
_SECONDS_KEY = "travel_seconds"  # int: how long the whole route takes


def _arrival_steps(path: list[str], environment: "EnvironmentSystem", seconds_per_step: int) -> list[int]:
    """The elapsed step on which each waypoint is reached; the last entry is the trip's duration.

    Each waypoint's cumulative walking time is rounded to steps on its own, with a floor of 1. Don't
    round edges and sum them: that makes every hop cost at least a step, and the trip's time becomes
    "hops × step length" whatever the distance. Several waypoints can share a step: a short route in a
    coarse world is crossed within one step.
    """
    arrivals = [0]
    elapsed_seconds = 0
    for a, b in zip(path, path[1:]):
        elapsed_seconds += environment.space.edge_seconds(a, b)
        arrivals.append(max(1, round(elapsed_seconds / seconds_per_step)))
    return arrivals


def _passed_at(path: list[str], arrivals: list[int], elapsed: int) -> list[str]:
    """The intermediate waypoints the mover goes through on step ``elapsed``, in route order."""
    return [path[i] for i in range(1, len(path) - 1) if arrivals[i] == elapsed]


def _transit_location_at(path: list[str], arrivals: list[int], elapsed: int) -> str:
    """Where the mover stands at the end of step ``elapsed``: the furthest intermediate waypoint
    reached this step (visible and interruptible there), else IN_TRANSIT. Never the destination,
    which ``complete()`` owns: waypoints passed on the final step are only passed through."""
    passed = _passed_at(path, arrivals, elapsed) if elapsed < arrivals[-1] else []
    return passed[-1] if passed else IN_TRANSIT


def _next_hop(path: list[str], here: str) -> str:
    """The waypoint id after ``here``; "" if ``here`` is the destination or not on the path.

    Bystanders see a direction, not a destination, so ambient lines report only the next hop.
    """
    try:
        i = path.index(here)
    except ValueError:
        return ""
    return path[i + 1] if i + 1 < len(path) else ""


def _reached_node_at(path: list[str], arrivals: list[int], elapsed: int) -> str:
    """Where the mover lands if interrupted at ``elapsed``: the nearer endpoint of the current
    leg (≥ halfway → far end). Per leg, not per journey, so it is always a real place."""
    if not path:
        return ""
    total = arrivals[-1] if arrivals else 0
    if elapsed >= total:
        return path[-1]
    for i in range(len(path) - 1):
        lo, hi = arrivals[i], arrivals[i + 1]
        if lo <= elapsed < hi:
            frac = (elapsed - lo) / (hi - lo) if hi > lo else 0.0
            return path[i + 1] if frac >= 0.5 else path[i]
    return path[0]


def carried_bodies(
    actor_id: str, candidate_ids: list[str], agents: dict[str, "Agent"],
    environment: "EnvironmentSystem",
) -> list[str]:
    """The people in ``candidate_ids`` that ``actor_id`` can take along: alive (moving a
    corpse would put it back into the world) and co-located.

    No "can he resist" gate: a carry-off buys a displacement, not obedience; he can fight or
    walk back after arrival. Being busy is the arbiter's call (``Conscription.COMPEL``).
    Asked once at start; mid-trip death or relocation tears the whole trip down via
    ``participant_ids``.
    """
    here = environment.get_body_location(actor_id)
    kept: list[str] = []
    for aid in candidate_ids:
        agent = agents.get(aid)
        if agent is None or not agent.is_active:
            continue
        if environment.get_body_location(aid) != here:
            continue
        kept.append(aid)
    return kept


def _landing_target(landed_id: str, carried_ids: list[str]) -> ActionTarget:
    """The final stub's target: the landing place plus the carried bodies in ``claims``, so
    the last beat says who was carried, like the first two."""
    return ActionTarget(
        acts_on=[Ref.place(landed_id)],
        claims=[Ref.agent(aid) for aid in carried_ids],
    )


def _carried_of(state: ActionExecutionState) -> list[str]:
    """The people this trip carries: every participant except the actor (``participant_ids``
    is the only record)."""
    return [pid for pid in state.participant_ids if pid != state.initiator_id]


def transit_view(state: ActionExecutionState) -> dict[str, object] | None:
    """The render-side transit contract for an in-flight MOVE, else None. God-view only.

    ``path`` is the waypoint route; ``arrivals[i]`` is the elapsed step on which the mover
    reaches ``path[i]``, so the renderer matches the engine on weighted legs.
    """
    if state.action_type != ActionType.MOVE:
        return None
    return {
        "from_location_id": state.extra.get(_ORIGIN_KEY),
        "to_location_id": state.extra.get(_DESTINATION_KEY),
        "path": list(state.extra.get(_PATH_KEY, [])),
        "arrivals": list(state.extra.get(_ARRIVALS_KEY, [])),
        "elapsed_steps": state.estimated_steps - state.remaining_steps,
        "total_steps": state.estimated_steps,
    }


def arrival_view(state: ActionExecutionState, environment: "EnvironmentSystem") -> dict[str, object] | None:
    """The trip that ended this step, in ``transit_view``'s shape (elapsed == total), else None.

    Only for a MOVE that ran out its steps with the mover standing at its destination, having
    left somewhere else. A trip crossed within one step never has a transit, so this is the only
    record of the route it took.
    """
    if state.action_type != ActionType.MOVE or state.remaining_steps > 0:
        return None
    destination = state.extra.get(_DESTINATION_KEY)
    if destination is None or destination == state.extra.get(_ORIGIN_KEY):
        return None
    if environment.get_body_location(state.initiator_id) != destination:
        return None
    return transit_view(state)


class MovementExecutor(ActionExecutor):
    """Executor for MOVE actions: departs to IN_TRANSIT in start(), arrives in complete().
    Feasibility is EnvironmentSystem's; a blocked move is ``create_failed``."""

    # complete() lands the arrival via environment.move_body.
    mutates_world_during_complete = True
    # Carrying someone off doesn't ask his consent.
    conscription = Conscription.COMPEL

    def __init__(self, directory: WorldDirectory, seconds_per_step: int = 3600) -> None:
        self._directory = directory
        self._seconds_per_step = seconds_per_step

    def _company_clause(self, carried_ids: list[str]) -> str:
        """Companion clause ("带着张三、王五"), "" when alone, so call sites interpolate it
        unconditionally.

        The verb is the neutral "带着", not "押着": the same mechanism serves abduction and
        rescue, and the wording must not characterize the situation.
        """
        if not carried_ids:
            return ""
        return "带着" + "、".join(self._directory.agent_name(aid) for aid in carried_ids)

    def _carried_result(
        self, agent_id: str, *, step: int, actor_name: str, origin_name: str,
        land_name: str, landed_id: str, arrived: bool, estimated_steps: int,
        outcome: str, observations: list[Observed], thought: str = "", gist: str = "",
    ) -> ActionResult:
        """The carried person's own participant result (not a ``TargetAgentEffect``): his
        location is landed by the MOVE branch of ``Agent._apply_feedback``, the same path as
        the actor's.

        ``outcome`` / ``observations`` are the actor's (delivery dedupes them); only
        ``factual_memory`` differs. Used for both arrival and a mid-way stop.
        """
        tail = f"到了{land_name}。" if arrived else f"到{land_name}便停下了。"
        stub = AgentAction(
            agent_id=agent_id, step=step, action_type=ActionType.MOVE,
            action_description=f"被{actor_name}带着前往{land_name}",
            target=_landing_target(landed_id, [agent_id]),
            estimated_steps=estimated_steps,
        )
        return ActionResult(
            action=stub,
            expected_outcome="",
            outcome=outcome,
            gist=gist,
            observations=observations,
            succeeded=arrived,
            factual_memory=(
                f"我被{actor_name}带着，身不由己地从{origin_name}{tail}"
                + format_interrupt_thought(thought)
            ),
        )

    def _carry_along(
        self, carried_ids: list[str], location_id: str, environment: "EnvironmentSystem",
    ) -> None:
        """Move the carried bodies to the actor's location, world side only. The agent side
        (``current_location``) is landed by feedback from ``_carried_result``; missing either
        side splits where others see him from where he thinks he is."""
        for aid in carried_ids:
            environment.move_body(body_id=aid, location_id=location_id)

    def _passthrough_line(
        self,
        *,
        subject: str,
        path: list[str],
        here: str,
        environment: "EnvironmentSystem",
    ) -> str:
        """3p bystander beat for a mover on a real waypoint this step, shared by start and tick.
        Call only when ``here`` != IN_TRANSIT. Reports only the next hop, never the destination."""
        # Resolve from ``here``, not his current location, so start()'s world mutation can stay last.
        location = environment.narrative_location_name(here)
        nxt = _next_hop(path, here)
        toward = environment.narrative_location_name(nxt) if nxt else ""
        body = (
            f"{subject}途经此地，往{toward}方向去了。" if toward
            else f"{subject}途经此地，未作停留。"
        )
        return scene_line(location, body)

    def _passthrough_observations(
        self,
        *,
        subject: str,
        path: list[str],
        passed: list[str],
        environment: "EnvironmentSystem",
    ) -> list[Observed]:
        """One "passing through" beat at each waypoint in ``passed``."""
        return [
            Observed(
                location_id=wp,
                text=self._passthrough_line(subject=subject, path=path, here=wp, environment=environment),
            )
            for wp in passed
        ]

    def _departure_line(
        self, *, subject: str, path: list[str], environment: "EnvironmentSystem",
    ) -> str:
        """3p bystander beat at the origin: "he left, heading that way" (next hop only).

        Needed explicitly: perception has no cross-step diff, so a departure is otherwise
        silent. Director teleports use the ``else`` sentence (``_departure_line`` in
        ``engine/world_mutation.py``)."""
        origin = path[0] if path else ""
        nxt = _next_hop(path, origin)
        toward = environment.narrative_location_name(nxt) if nxt else ""
        body = (
            f"{subject}离开此地，往{toward}方向去了。" if toward
            else f"{subject}离开了此地。"
        )
        return scene_line(environment.narrative_location_name(origin), body)

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        feasibility = environment.check_move_feasibility(action.agent_id, action.target.acted_on_place)
        actor_name = self._directory.agent_name(action.agent_id)
        origin_id = environment.get_body_location(action.agent_id)
        origin_name = environment.narrative_location_name(origin_id)

        if not feasibility.ok:
            stub = AgentAction(
                agent_id=action.agent_id,
                step=step,
                action_type=ActionType.MOVE,
                action_description=action.action_description or "move",
                target=action.target,
            )
            # No scene_line prefix: the sentence carries its own places.
            outcome = f"{actor_name}本想动身离开{origin_name}，却因{feasibility.reason}未能成行。"
            failure = ActionResult(
                action=stub,
                expected_outcome=action.expected_outcome,
                outcome=outcome,
                observations=observed_here(environment, action.agent_id, outcome),
                succeeded=False,
                failure_reason=feasibility.reason,   # structured fact (no route), not LLM-invented
                factual_memory=feasibility.reason,
            )
            return ActionExecutionState.create_failed(
                action_type=ActionType.MOVE, initiator_id=action.agent_id,
                failure_result=failure, started_step=step,
                purpose=action.action_description or "move",
            )

        to_id = feasibility.resolved_id
        dest_name = environment.narrative_location_name(to_id)
        path = list(feasibility.path) or [origin_id, to_id]
        arrivals = _arrival_steps(path, environment, self._seconds_per_step)
        duration = arrivals[-1]
        travel_label = describe_seconds(feasibility.travel_seconds)

        # Start is elapsed=1.
        here = _transit_location_at(path, arrivals, elapsed=1)
        # Companions and the departure line must be computed before move_body: both depend on
        # the actor still being at the origin. The departure is declared in
        # opening_observations (runtime._carry_step_observations delivers it), never recorded
        # directly, and is independent of duration: a same-step arrival was still seen leaving.
        carried = carried_bodies(
            action.agent_id, action.target.claimed_agents, agents, environment,
        )
        party = f"{actor_name}{self._company_clause(carried)}"
        departure = self._departure_line(subject=party, path=path, environment=environment)
        # A duration-1 move's opening is replaced by complete's outcome.
        opening = "" if duration <= 1 else (
            # describe_seconds already includes "约".
            f"{party}从{origin_name}动身前往{dest_name}，路程{travel_label}。"
        )
        # Waypoints passed on the first step read like a mid-trip tick; mid-edge is unseen. On a
        # one-step trip complete() reports them.
        passthroughs = [] if duration <= 1 else self._passthrough_observations(
            subject=party, path=path, passed=_passed_at(path, arrivals, 1), environment=environment,
        )
        state = ActionExecutionState.create(
            action_type=ActionType.MOVE,
            initiator_id=action.agent_id,
            # The carried are participants, so one-body-one-action, teardown and self-filtering
            # come from the existing machinery.
            participant_ids=[action.agent_id, *carried],
            purpose=f"从{origin_name}前往{dest_name}",
            started_step=step,
            estimated_steps=duration,
            opening_outcome=opening,
            # Origin sees him leave; each waypoint he goes through sees him pass.
            opening_observations=[Observed(location_id=origin_id, text=departure), *passthroughs],
            target=action.target,
            # Don't drop: without the "why" in memory an agent can't notice he's going in circles.
            expected_outcome=action.expected_outcome,
        )
        # Ids only; names are resolved on use.
        state.extra[_DESTINATION_KEY] = to_id
        state.extra[_ORIGIN_KEY] = origin_id
        state.extra[_PATH_KEY] = path
        state.extra[_ARRIVALS_KEY] = arrivals
        state.extra[_SECONDS_KEY] = feasibility.travel_seconds
        # The world mutation must stay last: if anything above raises, the world is untouched,
        # instead of a body stranded on IN_TRANSIT with no execution to move it again. Hence
        # neither ``_departure_line`` nor ``_passthrough_line`` may read his current location.
        environment.move_body(body_id=action.agent_id, location_id=here)
        self._carry_along(carried, here, environment)
        return state

    def claim_bodies(
        self,
        action: "AgentAction",
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
    ) -> list[str]:
        """Whom this trip will carry, by start()'s criteria. An infeasible trip carries nobody,
        or it would tear down the other person's action for nothing."""
        candidates = action.target.claimed_agents
        if not candidates:
            return []
        if not environment.check_move_feasibility(action.agent_id, action.target.acted_on_place).ok:
            return []
        return carried_bodies(action.agent_id, candidates, agents, environment)

    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        elapsed = state.estimated_steps - state.remaining_steps
        destination_id = state.extra.get(_DESTINATION_KEY, "")
        dest_name = environment.narrative_location_name(destination_id)
        actor_name = self._directory.agent_name(state.initiator_id)
        path = list(state.extra.get(_PATH_KEY, []))
        arrivals = list(state.extra.get(_ARRIVALS_KEY, []))

        still = _carried_of(state)
        party = f"{actor_name}{self._company_clause(still)}"

        here = _transit_location_at(path, arrivals, elapsed) if path else IN_TRANSIT
        environment.move_body(body_id=state.initiator_id, location_id=here)
        self._carry_along(still, here, environment)

        if here != IN_TRANSIT:
            narrative = self._passthrough_line(
                subject=party, path=path, here=here,
                environment=environment,
            )
            # Waypoints crossed earlier this step; the one he stands on is reported below.
            passed_by = self._passthrough_observations(
                subject=party, path=path, passed=_passed_at(path, arrivals, elapsed)[:-1],
                environment=environment,
            )
        else:
            passed_by = []
            # Mid-edge: carry skips IN_TRANSIT, so this reaches only the event stream.
            location = observe_location(environment, state.initiator_id)  # the just-advanced position ("途中")
            elapsed_label = describe_duration(elapsed, self._seconds_per_step)
            remaining_label = describe_duration(state.remaining_steps, self._seconds_per_step)
            narrative = scene_line(
                location,
                f"{party}正赶往{dest_name}的路上，"
                f"已走了{elapsed_label}，预计还需{remaining_label}。",
            )
        # Movement is fully public: observation == outcome.
        return [TickResult(
            agent_id=state.initiator_id, outcome=narrative,
            observations=[*passed_by, *observed_here(environment, state.initiator_id, narrative)],
        )]

    async def complete(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[ActionResult]:
        stored = self._stored_result(state)
        if stored is not None:
            return [stored]  # feasibility failure (no move performed)
        destination_id = state.extra.get(_DESTINATION_KEY, "")
        origin_id = state.extra.get(_ORIGIN_KEY, "")
        dest_name = environment.narrative_location_name(destination_id)
        origin_name = environment.narrative_location_name(origin_id)
        carried = _carried_of(state)
        company = self._company_clause(carried)
        actor = agents.get(state.initiator_id)
        if actor is None or actor.is_active:
            # Moving a dead body would put it back into the world.
            environment.move_body(body_id=state.initiator_id, location_id=destination_id)
        self._carry_along(carried, destination_id, environment)
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.MOVE,
            action_description=state.purpose,
            target=_landing_target(destination_id, carried),
            estimated_steps=state.estimated_steps,
        )
        duration_label = describe_seconds(state.extra[_SECONDS_KEY])
        actor_name = self._directory.agent_name(state.initiator_id)
        outcome = (
            f"{actor_name}{company}从{origin_name}出发，赶了{duration_label}的路，抵达{dest_name}。"
        )
        # Intent goes only into his own memory; others see that he arrived, not why.
        intent = format_intent_clause(state.expected_outcome)
        path = list(state.extra.get(_PATH_KEY, []))
        arrivals = list(state.extra.get(_ARRIVALS_KEY, []))
        observations = [
            # Waypoints crossed on the final step.
            *self._passthrough_observations(
                subject=f"{actor_name}{company}", path=path,
                passed=_passed_at(path, arrivals, state.estimated_steps) if path else [],
                environment=environment,
            ),
            *observed_here(environment, state.initiator_id, outcome),
        ]
        # All participants share outcome and observations (carry dedupes); memories differ.
        return [
            ActionResult(
                action=stub,
                expected_outcome=state.expected_outcome or f"抵达{dest_name}",
                outcome=outcome,
                observations=observations,
                succeeded=True,
                factual_memory=(
                    f"{intent}{company}从{origin_name}前往了{dest_name}，路上花了{duration_label}。"
                ),
            ),
            *(
                self._carried_result(
                    aid, step=step, actor_name=actor_name, origin_name=origin_name,
                    land_name=dest_name, landed_id=destination_id, arrived=True,
                    estimated_steps=state.estimated_steps,
                    outcome=outcome, observations=observations,
                )
                for aid in carried
            ),
        ]

    async def interrupt(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem | None" = None,
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        dest_id = state.extra.get(_DESTINATION_KEY, "")
        origin_id = state.extra.get(_ORIGIN_KEY, "")
        dest_name = environment.narrative_location_name(dest_id) if environment is not None else "此处"
        origin_name = environment.narrative_location_name(origin_id) if environment is not None else "此处"
        path = list(state.extra.get(_PATH_KEY, []))
        arrivals = list(state.extra.get(_ARRIVALS_KEY, []))
        elapsed = state.estimated_steps - state.remaining_steps
        actor_name = self._directory.agent_name(state.initiator_id)
        cut = f"因{cause}而停下" if cause else "停下"
        # The thought belongs to whoever broke off, which may be a carried person, never
        # automatically the captor.
        breaker_id = interrupted_agent_id or state.initiator_id
        breaker_name = self._directory.agent_name(breaker_id)
        by_initiator = breaker_id == state.initiator_id
        thought_part = format_interrupt_thought(thought) if by_initiator else ""

        # Land at a real place: the nearer end of the current leg (see _reached_node_at).
        land_at = _reached_node_at(path, arrivals, elapsed) if path else origin_id
        land_name = environment.narrative_location_name(land_at) if environment is not None else "此处"

        carried = _carried_of(state)
        party = f"{actor_name}{self._company_clause(carried)}"
        if land_at == dest_id:
            # Already past halfway on the last leg: counts as arrived.
            base = f"{party}从{origin_name}赶往{dest_name}途中遭打断，仍抵达了{dest_name}。"
            factual = f"从{origin_name}赶往{dest_name}途中遭打断，仍抵达了{dest_name}{thought_part}"
            succeeded = True
        elif land_at == origin_id:
            # Hasn't left the first leg → stays put.
            base = f"{party}刚从{origin_name}动身前往{dest_name}，便停下，仍在{origin_name}。"
            factual = f"刚从{origin_name}动身前往{dest_name}，就{cut}，仍在{origin_name}{thought_part}"
            succeeded = False
        else:
            # Stopped at a real waypoint mid-way.
            base = f"{party}从{origin_name}赶往{dest_name}途中，行至{land_name}便停下。"
            factual = f"从{origin_name}赶往{dest_name}途中，行至{land_name}时{cut}{thought_part}"
            succeeded = False

        # "Why he stopped" only on the breaker's own outcome; no bystander line (see
        # ActionExecutor.interrupt).
        breaker_outcome = base + format_interrupt_reason_3p(breaker_name, thought)

        actor = agents.get(state.initiator_id)
        if environment is not None and land_at and (actor is None or actor.is_active):
            # Death teardown also comes through here; a removed body must not be put back.
            environment.move_body(body_id=state.initiator_id, location_id=land_at)
        if environment is not None and land_at:
            # Same landing as the actor, even if the actor is dead: otherwise the captive is
            # left on IN_TRANSIT forever.
            self._carry_along(carried, land_at, environment)
        if environment is None:
            # API misuse: without environment a body in transit stays stranded.
            logger.error(
                "movement_interrupt_without_environment",
                extra={"agent_id": state.initiator_id, "stranded_at": IN_TRANSIT},
            )

        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.MOVE,
            action_description=state.purpose,
            target=_landing_target(land_at, _carried_of(state)),
            estimated_steps=state.estimated_steps,
        )
        intent = format_intent_clause(state.expected_outcome)
        return [
            ActionResult(
                action=stub,
                expected_outcome=state.expected_outcome or f"抵达{dest_name}",
                outcome=breaker_outcome if by_initiator else base,
                gist=base,
                succeeded=succeeded,
                factual_memory=f"{intent}{factual}",
            ),
            *(
                self._carried_result(
                    aid, step=step, actor_name=actor_name, origin_name=origin_name,
                    land_name=land_name, landed_id=land_at, arrived=(land_at == dest_id),
                    estimated_steps=state.estimated_steps,
                    outcome=breaker_outcome if aid == breaker_id else base, observations=[],
                    thought=thought if aid == breaker_id else "", gist=base,
                )
                for aid in carried
            ),
        ]
