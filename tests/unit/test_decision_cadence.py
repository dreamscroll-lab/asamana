"""Decision cadence: who the engine spends a decision loop on, and who stays a free body.

Admitting every active agent every step costs two LLM calls per agent per world-hour,
including for characters with nothing to react to and no urge to act — and a planned agent
later conscripted into someone else's conversation throws its decision away outright.

The cadence gate (``AgentScheduler._should_decide``) admits only the idle bodies worth
asking. The ones it skips are NOT dropped from the step: the runtime gives them a stub plan
(``action=None`` / ``NOT_SCHEDULED``, zero LLM calls) so they remain conscriptable. That
combination is the whole design — a skipped agent costs nothing and loses nothing.

The gate is a *scheduling* decision ("does the engine ask you"), never a cognitive one
("what do you decide" stays with the LLM: ``act=false`` → ``NO_ACTION``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent.decision import ActionType, AgentAction, DecisionStatus
from agent.motivation import ExternalDriveType, ExternalGoal
from agent.need import NEED_URGENT_INTENSITY
from agent.personality import EMOTION_STRONG_INTENSITY, EmotionState, StateLayer
from core.interfaces.action import ActionTarget, Ref
from core.interfaces.urgency import Urgency
from engine.environment import EnvironmentSystem
from engine.message_system import MessageDelivery
from engine.scheduler import AgentScheduler, CadenceReason


def _ext(urgency: Urgency = Urgency.NORMAL) -> ExternalGoal:
    """A pressure surface at a given urgency. The gate reads ``.urgency``, so a pressure stub
    must be a real ExternalGoal — the pressure evaluator never emits bare strings."""
    return ExternalGoal(
        text="someone is at the gate",
        source_id="world",
        urgency=urgency,
        drive_type=ExternalDriveType.EVENT,
    )

from tests.unit.test_runtime_smoke import _arb_runtime, _build_agent, _plan_for
from worlds.tiled import TiledWorldConfig


# ---------------------------------------------------------------------------
# The gate itself — a pure policy over structured state, so it is tested on stubs.
# ---------------------------------------------------------------------------


class _StubPersonality:
    def __init__(self, state: StateLayer) -> None:
        self._state = state

    @property
    def state(self) -> StateLayer:
        return self._state


@dataclass
class _StubAgent:
    agent_id: str
    is_main_character: bool = False
    is_active: bool = True
    pending_external_goals: list = field(default_factory=list)
    # Defaults describe a settled agent with nothing going on: it decided recently, its needs
    # sit at the rest baseline, its mood is flat. Each test perturbs exactly one signal, so a
    # failure names the condition that broke.
    last_decision_step: int = 1
    need_intensity: float = 0.3          # _NEED_REST_BASELINE: noticeable but not urgent
    emotion_intensity: float = 0.2       # EmotionState's own default

    def __post_init__(self) -> None:
        self.personality = _StubPersonality(StateLayer(
            agent_id=self.agent_id,
            last_decision_step=self.last_decision_step,
            need_intensities={"safety": self.need_intensity},
            emotion=EmotionState(intensity=self.emotion_intensity),
        ))


def _admitted(agents, *, step: int, **kw) -> set[str]:
    scheduler = AgentScheduler(main_max_idle_steps=2, background_max_idle_steps=6, **kw)
    plan = scheduler.plan(agents, step=step, in_progress_at_step_start=set())
    return {a.agent_id for a in plan.ordered_agents()}


def test_settled_agent_with_nothing_happening_is_not_asked() -> None:
    """The gate must actually close — otherwise none of this saves anything.

    Nothing outside, nothing inside, decided last step: there is nothing for this character
    to think about, so the engine does not pay to have it think.
    """
    assert _admitted([_StubAgent("a")], step=2) == set()


def test_never_decided_agent_is_always_admitted() -> None:
    """Cold start. last_decision_step == 0 means 「从未决策过」.

    Without this the starvation clock would read `1 - 0 = 1 < 2` on a fresh world's first
    step and open the simulation on a step where nobody thought at all.
    """
    assert _admitted([_StubAgent("a", last_decision_step=0)], step=1) == {"a"}


def test_agent_under_external_pressure_is_admitted() -> None:
    """The reactive half. The gate reads pending_external_goals — the pressure evaluator's
    verdict for this step, already the cognitive translation of "raw signal → worth acting
    on". It gates on that verdict, NOT on the raw signal: keying on the raw signal re-admits
    the co-located ambient chatter pressure already dismissed and wakes everyone
    every step."""
    pressed = _StubAgent("b", pending_external_goals=[_ext(Urgency.NORMAL)])
    quiet = _StubAgent("a")
    assert _admitted([pressed, quiet], step=2) == {"b"}


def test_low_urgency_pressure_alone_does_not_admit() -> None:
    """LOW is 「可留意(背景信号)」 — below the Urgency scale's actionability line. Pressure
    noting a faint, unpointed signal must not by itself force a decision; that is the world
    being noticed, not the world bearing on the character. (It still colors motivation if the
    agent decides for another reason, and pressure re-evaluates it next step — nothing lost.)"""
    noted = _StubAgent("a", pending_external_goals=[_ext(Urgency.LOW)])
    assert _admitted([noted], step=2) == set()


def test_pressure_admits_when_any_goal_reaches_normal() -> None:
    """A list is admitted on its strongest goal, not its emptiness: a NORMAL sitting beside a
    LOW still presses. The floor is NORMAL — the first 「值得响应」 band."""
    mixed = _StubAgent("a", pending_external_goals=[_ext(Urgency.LOW), _ext(Urgency.NORMAL)])
    assert _admitted([mixed], step=2) == {"a"}


def test_low_only_pressure_still_admits_once_starved() -> None:
    """Dropping LOW from the pressure condition must not strand an agent: the starvation clock
    still catches it. LOW pressure never resets that clock, so 「可留意」 signals cannot silence
    a character forever."""
    lulled = _StubAgent("a", last_decision_step=1, pending_external_goals=[_ext(Urgency.LOW)])
    assert _admitted([lulled], step=7) == {"a"}  # 7 - 1 >= background idle bound (6)


def test_urgent_need_admits_without_any_external_signal() -> None:
    """The self-driven half. A gate keyed only on external signals would hard-code a reactive
    personality onto every character — nobody could ever move on their own ambition — and the
    world would deadlock: nobody acting means no signals arise to wake anybody."""
    driven = _StubAgent("a", need_intensity=NEED_URGENT_INTENSITY)
    calm = _StubAgent("b", need_intensity=NEED_URGENT_INTENSITY - 0.01)
    assert _admitted([driven, calm], step=2) == {"a"}


def test_strong_emotion_admits_without_any_external_signal() -> None:
    stirred = _StubAgent("a", emotion_intensity=EMOTION_STRONG_INTENSITY)
    flat = _StubAgent("b", emotion_intensity=EMOTION_STRONG_INTENSITY - 0.01)
    assert _admitted([stirred, flat], step=2) == {"a"}


def test_merely_pronounced_emotion_does_not_admit() -> None:
    """The gate reads "强烈" (0.75), NOT "明显" (0.45).

    LLM-authored emotion usually sits well above 0.45, so "明显" (a prose-rendering band)
    carries no information. Keyed on it, the gate fires on most agent-steps and the cadence
    mechanism becomes a no-op while every other test stays green.
    """
    typical = _StubAgent("a", emotion_intensity=0.70)   # a typical intensity
    assert _admitted([typical], step=2) == set()


def test_starvation_bound_is_shorter_for_main_characters() -> None:
    """With nothing pushing either way, a person still decides what to do next eventually.
    A main character is asked back sooner — that is what the narrative tier buys (scheduling
    priority), and it is the ONLY thing it buys here: an admitted background agent runs the
    exact same LLM cognition path (CLAUDE.md §5 — no rule-vs-LLM split by agent)."""
    hero = _StubAgent("hero", is_main_character=True, last_decision_step=1)
    extra = _StubAgent("extra", is_main_character=False, last_decision_step=1)

    assert _admitted([hero, extra], step=2) == set()          # neither is starved yet
    assert _admitted([hero, extra], step=3) == {"hero"}       # main bound (2) reached
    assert _admitted([hero, extra], step=7) == {"hero", "extra"}  # background bound (6) reached


def test_each_condition_reports_which_one_admitted() -> None:
    """The gate reports WHY it let someone in, and that attribution is the tuning channel.

    Two of the thresholds are empirical rather than derived, so tuning has to ask "which
    conditions actually carry this world, and which are inert?"; a bare admitted/skipped count
    cannot answer that.

    Attribution is first-match-wins, so an agent that is both under pressure and starved reads
    as 「pressure」 — the honest label: the world is bearing on it, the clock is incidental.
    """
    gate = AgentScheduler(main_max_idle_steps=2, background_max_idle_steps=6)

    def reason(agent, *, step=2):
        return gate._should_decide(agent, step=step)  # noqa: SLF001

    assert reason(_StubAgent("a", last_decision_step=0)) is CadenceReason.NEVER_DECIDED
    assert reason(_StubAgent("a", pending_external_goals=[_ext()])) is CadenceReason.PRESSURE
    assert reason(_StubAgent("a", need_intensity=NEED_URGENT_INTENSITY)) is CadenceReason.NEED
    assert reason(_StubAgent("a", emotion_intensity=EMOTION_STRONG_INTENSITY)) is CadenceReason.EMOTION
    assert reason(_StubAgent("a", last_decision_step=1), step=7) is CadenceReason.STARVED
    assert reason(_StubAgent("a")) is None  # nothing pressing → skipped
    # LOW-only pressure is below the actionability line → not a PRESSURE admit.
    assert reason(_StubAgent("a", pending_external_goals=[_ext(Urgency.LOW)])) is None

    # Both under pressure and starved → pressure is the reason, not the clock.
    both = _StubAgent("a", last_decision_step=1, pending_external_goals=[_ext()])
    assert reason(both, step=7) is CadenceReason.PRESSURE


def test_in_progress_body_is_never_asked_even_when_starving() -> None:
    """Occupancy outranks the gate: an agent mid-action does not re-decide, however long it
    has been. The two questions stay separate — who CAN act vs. who is ASKED to think."""
    busy = _StubAgent("a", last_decision_step=1, pending_external_goals=[_ext()])
    plan = AgentScheduler(main_max_idle_steps=2, background_max_idle_steps=6).plan(
        [busy], step=99, in_progress_at_step_start={"a"},
    )
    assert plan.ordered_agents() == []


def test_dead_agent_is_never_asked() -> None:
    assert _admitted([_StubAgent("a", is_active=False, last_decision_step=0)], step=1) == set()


# ---------------------------------------------------------------------------
# The other half: a skipped agent is a FREE BODY, not an absent one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unscheduled_agent_is_still_conscriptable(container) -> None:
    """**The core invariant.** An agent the gate skipped can still be pulled into someone
    else's action — and it never spent a decision to get there.

    This is easy to break: ``ExecutionArbiter`` classifies a co-participant absent from
    ``planned_steps`` as 「毫无回应」 and rejects the *conscripting* agent's entire action. If
    the runtime stops emitting stub plans for skipped agents, conscription dies silently, and
    TALK is the majority of all actions.

    Here B is not scheduled (it has no plan of its own), yet A's TALK still enrolls it.
    """
    world_id = "world-cadence-conscript"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="魏徵", is_main_character=False)
    agents = {"agent-a": a, "agent-b": b}
    rt = _arb_runtime(container, world_id, agents, env)

    talk = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.TALK,
        action_description="商议对策", target=ActionTarget(acts_on=[Ref.agent("agent-b")], claims=[Ref.agent("agent-b")]),
        estimated_steps=2,
    )
    # A decided; B was never asked — exactly what the runtime hands the arbiter for a
    # gate-skipped agent (action=None, NOT_SCHEDULED, zero LLM calls spent).
    planned = [
        ("main", a, _plan_for(a, talk, env)),
        ("background", b, _plan_for(b, None, env, decision_status=DecisionStatus.NOT_SCHEDULED)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    # A's TALK was admitted — NOT rejected as 「毫无回应」.
    a_aa = arb["agent-a"]
    assert a_aa.is_passive_join is False
    assert a_aa.ongoing_execution_id is not None
    exec_state = rt._executor_registry.get_active_for_agent("agent-b")  # noqa: SLF001
    assert exec_state is not None and "agent-b" in exec_state.participant_ids

    # B was conscripted into it, having spent no decision of its own.
    b_aa = arb["agent-b"]
    assert b_aa.is_passive_join is True


@pytest.mark.asyncio
async def test_runtime_gives_skipped_agents_a_free_body_plan(container) -> None:
    """The runtime emits one plan entry per active idle agent — deciders and skipped alike —
    so `planned_ids` stays complete (see the test above for why that matters). The skipped
    ones carry action=None / NOT_SCHEDULED, which the arbiter reads as "free body": no
    arbitration entry, nothing committed, zero state change.
    """
    world_id = "world-cadence-stub"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="魏徵", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    # An empty scheduler plan = the gate admitted nobody this step.
    empty_plan = AgentScheduler().plan([], step=5, in_progress_at_step_start=set())
    planned = await rt._plan_execution(  # noqa: SLF001
        plan=empty_plan, agents=[a, b], in_progress_at_step_start=set(),
        deliveries=MessageDelivery(step=5), step=5,
        world_time_label="清晨", agent_spatials=None, broadcasts=[],
    )

    by_id = {p.agent_id: p for _phase, _agent, p in planned}
    assert set(by_id) == {"agent-a", "agent-b"}       # both present → both conscriptable
    for plan in by_id.values():
        assert plan.action is None
        assert plan.decision_status is DecisionStatus.NOT_SCHEDULED
        # The passive-join path reads these two off the plan; a hollow stub would break it.
        assert plan.spatial is not None
        assert plan.inbox == []


@pytest.mark.asyncio
async def test_admitted_agents_have_their_decision_step_recorded(container) -> None:
    """The starvation clock only advances for agents actually asked to think. If it advanced
    for skipped ones too, nobody would ever starve and the gate would latch shut."""
    world_id = "world-cadence-clock"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    env.place_agent(agent_id="agent-b", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    b = _build_agent(container, world_id=world_id, agent_id="agent-b", name="魏徵", is_main_character=False)
    rt = _arb_runtime(container, world_id, {"agent-a": a, "agent-b": b}, env)

    # Only A is admitted this step.
    plan = AgentScheduler().plan([a], step=5, in_progress_at_step_start=set())
    await rt._plan_execution(  # noqa: SLF001
        plan=plan, agents=[a, b], in_progress_at_step_start=set(),
        deliveries=MessageDelivery(step=5), step=5,
        world_time_label="清晨", agent_spatials=None, broadcasts=[],
    )

    assert a.personality.state.last_decision_step == 5
    assert b.personality.state.last_decision_step == 0   # never asked → still starving


# ---------------------------------------------------------------------------
# Two things the gate quietly depends on.
# ---------------------------------------------------------------------------


def test_last_decision_step_survives_a_state_round_trip() -> None:
    """``_copy_state`` rebuilds StateLayer field by field (no dataclasses.replace), and BOTH
    ``PersonalityLayer.state`` and ``restore_state`` route through it. Miss the field there
    and it silently resets to 0 on every state read: every agent looks 「从未决策过」 and the
    gate admits everyone, a no-op that still passes every other test.
    """
    from agent.personality import PersonalityLayer, SoulLayer

    personality = PersonalityLayer(soul=SoulLayer(name="李世民", role="prince", agent_id="a"))
    personality.mark_decided(42)
    assert personality.state.last_decision_step == 42          # read path

    personality.restore_state(personality.state)
    assert personality.state.last_decision_step == 42          # restore path


@pytest.mark.asyncio
async def test_pressure_is_rewritten_every_step_and_never_goes_stale(container) -> None:
    """``pending_external_goals`` must be the pressure evaluator's verdict for THIS step —
    empty included — not a high-water mark.

    The evaluator omits agents with no acute signal from its result, so iterating only the
    result would pin a quiet agent's last pressure in place forever. ``plan_step`` cannot
    mask that by clearing the field either — an agent the cadence gate skips never reaches
    it. Were this to regress, the stale goals would be read by the agent's own next decision
    as if the world were still pressing on it.
    """
    from agent.motivation import ExternalDriveType, ExternalGoal
    from core.interfaces.action import Urgency

    world_id = "world-cadence-pressure"
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    env.place_agent(agent_id="agent-a", location_id="palace")
    a = _build_agent(container, world_id=world_id, agent_id="agent-a", name="李世民", is_main_character=True)
    rt = _arb_runtime(container, world_id, {"agent-a": a}, env)

    class _Pressure:
        """Speaks for agent-a on the first call, then falls silent (no acute signal)."""

        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                return {}
            return {"agent-a": [ExternalGoal(
                text="有人在殿外求见",
                source_id="agent-b",
                urgency=Urgency.HIGH,
                drive_type=ExternalDriveType.OBLIGATION,
            )]}

    rt._pressure_evaluator = _Pressure()  # noqa: SLF001

    await rt.run_step([a])
    assert a.pending_external_goals, "step 1: the evaluator spoke, the goals must land"

    await rt.run_step([a])
    assert a.pending_external_goals == [], "step 2: it fell silent → the field must be empty, not stale"


def test_interrupter_is_admitted_over_a_gate_that_would_otherwise_skip_it() -> None:
    """An interrupt means, within one step: kill the current action → land the interrupt
    feedback → run a new cognition cycle. That third part is part of what an interrupt is, not
    something the cadence gate gets to veto.

    The agent here is exactly what the gate wants to skip (just decided, no pressure, needs and
    emotion at rest). Its own judgment that the matter is worth dropping its work for outranks
    the third-party verdict PRESSURE reads; vetoing it would tear down an action and put nothing
    in its place.
    """
    gate = AgentScheduler()
    settled = _StubAgent("a")

    assert gate._should_decide(settled, step=2) is None  # noqa: SLF001
    assert gate._should_decide(settled, step=2, interrupted=True) is CadenceReason.INTERRUPTED  # noqa: SLF001


def test_collateral_participant_takes_the_ordinary_gate() -> None:
    """The co-participant caught up in it never said it was worth dropping its work; someone
    else broke up its joint action. Its body is free (not in the in_progress set), but whether it
    gets to think is still up to the gate."""
    settled, interrupter = _StubAgent("b"), _StubAgent("a")
    plan = AgentScheduler().plan(
        [interrupter, settled], step=2,
        in_progress_at_step_start=set(),   # runtime has already removed both
        interrupters={"a"},                # but only a is the one who chose to stop
    )
    assert [a.agent_id for a in plan.ordered_agents()] == ["a"]
