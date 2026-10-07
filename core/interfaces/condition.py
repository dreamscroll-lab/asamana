"""BodyCondition — an ongoing condition imposed on a body that persists across steps.

Serves both ``Agent`` and ``Npc``: being bound has nothing to do with cognition.

Motivation
==========

Objects have ``WorldEntity.state``; without this, people would have no slot for lasting imposed
conditions ("bound", "poisoned", "imprisoned"). Ambient outcomes and memory decay by design, so
they can't carry a fact that must shape the next decision and adjudication: a restrainer would
tie the same person up again, and the restrained would act, and be judged, as free.

Why a description and no taxonomy
=================================

No ``ConditionKind`` enum: the space of states is open-ended, and a closed taxonomy would force
the judge to cram "knocked out" into "restrained".

No capability consequences ("which ActionTypes this blocks") either: whether a bound person can
still eavesdrop is situational interpretation (CLAUDE.md §5), and every reader of a condition
has an LLM present. A consequence set frozen at write time would keep applying after the guard
has walked off.

No free ``dict[str, str]`` state bag either: the engine would end up string-matching a
schemaless field. The narrative axis is open; the structural axis is not.

Boundaries
==========

- Read: ``agent/`` cognition may only see someone else's condition through the perception
  bundle ``SpatialPerception.visible_agents`` (information asymmetry, same discipline as
  "``agent/`` must not use ``WorldDirectory``"). God-view code in ``engine/`` (executor
  adjudication, world_pressure, director) already holds the ``Agent`` legitimately and reads
  ``personality.state.condition`` directly.
- Write: executors only declare it in ``TargetAgentEffect``; the single place it lands is
  ``Agent.apply_target_effect`` (the Executor/Feedback boundary rule).
- Render: the only renderer is ``core.prompts.render_condition``; call sites must not build
  the text themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BodyCondition:
    """The condition a body is in right now.

    ``description`` is narrative-layer text (no ids or step counts); the other fields are
    code-layer coordinates. ``since_step`` reaches text only via ``render_condition``, as a
    natural duration the judge can weigh ("tied up for three days, the guard has grown lax").

    Frozen: a condition is settled by one adjudication and replaced whole, which also lets
    ``_copy_state`` alias it.
    """

    description: str                # narrative layer, ≤15 chars: "hands bound behind", "drugged, limbs limp"
    source_agent_id: str = ""       # code layer; empty = not caused by anyone (poison / blizzard / illness)
    since_step: int = 0             # code layer; rendered as a natural duration, never as a step count
    until_step: int | None = None   # None = needs outside help to end; set = self-limiting (drug wears off, sleeper wakes)


def condition_to_dict(condition: BodyCondition | None) -> dict[str, object] | None:
    """BodyCondition → JSON-native dict (None passes through).

    ``FileAgentStore`` reads via ``AgentState(**data)`` without rebuilding types, so a stored
    dataclass would come back as a dict and diverge from ``InMemoryAgentStore``.
    """
    if condition is None:
        return None
    return {
        "description": condition.description,
        "source_agent_id": condition.source_agent_id,
        "since_step": condition.since_step,
        "until_step": condition.until_step,
    }


def condition_from_dict(data: object) -> BodyCondition | None:
    """dict → BodyCondition. Any unreadable input, or no description, returns None; never
    raises. None is the normal state, not error tolerance."""
    if not isinstance(data, dict):
        return None
    description = str(data.get("description", "") or "").strip()
    if not description:
        return None
    until = data.get("until_step")
    return BodyCondition(
        description=description,
        source_agent_id=str(data.get("source_agent_id", "") or ""),
        since_step=_as_int(data.get("since_step"), 0),
        until_step=None if until is None else _as_int(until, 0),
    )


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


#: How long a "needs outside help" condition on an Npc lasts before lifting on its own. People
#: get no such fallback: they can break free; an Npc never acts, so one interception would erase
#: it and its errand. Shorter and interception buys nothing; longer is as good as erasure. In
#: seconds, not steps: step length varies by world, how long a rope holds doesn't.
NPC_CONDITION_FALLBACK_SECONDS = 12 * 3600


def npc_condition_fallback_steps(seconds_per_step: int) -> int:
    """The fallback duration in steps, at least 1 (0 would untie it on the same beat)."""
    return max(1, math.ceil(NPC_CONDITION_FALLBACK_SECONDS / max(1, seconds_per_step)))
