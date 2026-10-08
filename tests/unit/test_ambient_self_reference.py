"""Execution-body membership invariant: ambient_events never contain the outcome of the
agent itself (or of an action it took part in).

runtime._carry_step_observations writes ActionResult.outcome into ambient; without filtering,
spatial_for(agent_id=X) would put X's own outcome into X's ambient_events, and X would read a
self-referential "XX did something somewhere" as an external signal.

Ambient entries carry actor_ids (every member of the execution body), not a single actor_id: a
TALK passive-join record lands under the recruit's id while describing the initiator's action, so
only set membership filters out both. spatial_for drops entries with `agent_id ∈ actor_ids`.
"""

from __future__ import annotations



from engine.clock import WorldTime
from engine.environment import EnvironmentSystem
from core.interfaces.place import Place


def _make_env_with_two_agents() -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="hall", name="hall", description="",
        connections={}, is_public=True, capacity=50,
    ))
    env.place_agent(agent_id="actor", location_id="hall")
    env.place_agent(agent_id="bystander", location_id="hall")
    return env


def _world_time(step: int) -> WorldTime:
    return WorldTime(step=step, elapsed_seconds=step * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Self-reference filter: the actor doesn't read ambient with actor_id=self
# ─────────────────────────────────────────────────────────────────────────────


def test_actor_does_not_read_own_ambient_outcome() -> None:
    """After the actor calls record_carry_observation(actor_ids=(self,)), the actor's own
    spatial.ambient_events must not contain it; bystanders should see it.
    """
    env = _make_env_with_two_agents()
    env.record_carry_observation(
        location_id="hall",
        observation="actor 在大厅里部署了一些事情",
        actor_ids=("actor",),
    )
    env.begin_step(step=1, world_time=_world_time(1))

    spatial_actor = env.spatial_for(agent_id="actor")
    spatial_bystander = env.spatial_for(agent_id="bystander")

    # The actor doesn't read its own outcome
    assert not any("actor 在大厅里部署" in ev.content for ev in spatial_actor.ambient_events), (
        "F-fixture-1 违规:actor 通过 ambient 读到自己的 outcome 描述。"
    )
    # The bystander reads it
    assert any("actor 在大厅里部署" in ev.content for ev in spatial_bystander.ambient_events)


def test_ambient_without_actor_ids_visible_to_everyone() -> None:
    """actor_ids=() (environment events / system-level ambient) is visible to every agent at the
    location (the actor included)."""
    env = _make_env_with_two_agents()
    env.record_carry_observation(
        location_id="hall",
        observation="天气骤变,乌云密布",
        # actor_ids not passed → ()
    )
    env.begin_step(step=1, world_time=_world_time(1))

    for agent_id in ("actor", "bystander"):
        spatial = env.spatial_for(agent_id=agent_id)
        assert any("天气骤变" in ev.content for ev in spatial.ambient_events), (
            f"{agent_id} 应能看到 actor_id=None 的环境事件"
        )


def test_ambient_attributed_to_other_actor_still_visible_to_me() -> None:
    """A's outcome should be read by B (only "own" is filtered, not "others")."""
    env = _make_env_with_two_agents()
    env.record_carry_observation(
        location_id="hall",
        observation="bystander 在角落沉思",
        actor_ids=("bystander",),
    )
    env.begin_step(step=1, world_time=_world_time(1))

    spatial_actor = env.spatial_for(agent_id="actor")
    # The actor should read the bystander's outcome
    assert any("bystander 在角落沉思" in ev.content for ev in spatial_actor.ambient_events)


def test_multi_participant_filtered_for_every_member() -> None:
    """Core self-exclusion case: ambient from a multi-party action belongs to all participants,
    and none of them reads it.

    Mirrors TALK passive-join (see the module docstring): both members are filtered, while a third
    party (observer) still sees it.
    """
    env = _make_env_with_two_agents()
    env.place_agent(agent_id="observer", location_id="hall")
    env.record_carry_observation(
        location_id="hall",
        observation="被 actor 邀入密议",  # lands under bystander
        actor_ids=("actor", "bystander"),            # every member of the execution body
    )
    env.begin_step(step=1, world_time=_world_time(1))

    for member in ("actor", "bystander"):
        spatial = env.spatial_for(agent_id=member)
        assert not any("邀入密议" in ev.content for ev in spatial.ambient_events), (
            f"自我排除违规:执行体成员 {member} 读到了自己参与的行动 outcome。"
        )
    spatial_observer = env.spatial_for(agent_id="observer")
    assert any("邀入密议" in ev.content for ev in spatial_observer.ambient_events), (
        "非参与者(observer)应能看到该 ambient。"
    )
