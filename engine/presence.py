"""Presence assembly for perception: the single place that builds "who can I see, what can I
call them, and what state are they in".

It merges two authorities, neither of which should own the other:

- Identity (name / gender / age / bio) comes from ``WorldDirectory``, which must not grow any
  mutable-state query (that would create a second source of truth).
- The per-step mutable half comes from live objects: an agent's condition from ``Agent``, an
  Npc's busy flag and condition from ``Npc``.

It must live in one place, shared by the runtime and the tuning harness: scattered copies drift
(the harness's scene override replaces the present set, and a copy that forgets to rebuild leaves
identity and condition on the old set), and the drift fails silently. For the same reason
``visible_agent_ids`` is a read-only view derived from ``visible_agents``, not a parallel field.

Agents and Npcs go through one entry point (``attach_presence``): with two functions, callers
forget the second and Npcs silently render as "#1 某人" (someone) with no bio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from core.interfaces.directory import WorldDirectory
from core.interfaces.perception import (
    NpcIdentity,
    PerceivedIdentity,
    PerceivedNpc,
    PerceivedPresence,
    SpatialPerception,
)
from core.prompts import render_condition

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem


def reset_visible(spatial: SpatialPerception, agent_ids: Iterable[str]) -> None:
    """Replace "who is here" wholesale (for the harness's scene override); identity and
    condition are cleared, so this must be followed by ``attach_presence``.

    There is no "change only the ids" path: keeping the old identity/condition guarantees drift.
    """
    spatial.visible_agents = {agent_id: PerceivedPresence() for agent_id in agent_ids}


def _attach_npcs(
    spatial: SpatialPerception,
    *,
    directory: WorldDirectory,
    environment: "EnvironmentSystem",
) -> None:
    """The present-Npc half; called by ``attach_presence``, not exposed on its own.

    Identity from ``WorldDirectory``, busy and condition from the live ``Npc``. A missing bio
    stays empty and is omitted when rendered.
    """
    identities = directory.npc_identity_map(spatial.visible_npc_ids)
    spatial.visible_npcs = {
        npc_id: PerceivedNpc(
            identity=identities.get(npc_id) or NpcIdentity(),
            busy=bool((npc := environment.get_npc(npc_id)) is not None and npc.busy),
            condition=(
                render_condition(npc.condition) if npc is not None and npc.condition else ""
            ),
        )
        for npc_id in spatial.visible_npc_ids
    }


def attach_presence(
    spatial: SpatialPerception,
    *,
    directory: WorldDirectory,
    agents: Mapping[str, "Agent"],
    environment: "EnvironmentSystem",
) -> None:
    """Rebuild both "who can I see" tables for the current present set, in place.

    Agents go into ``visible_agents``, Npcs into ``visible_npcs``. Call this after the present
    set is final (after ``spatial_for`` or ``reset_visible``).

    Presence is expressed by the key existing, not by non-empty fields: everyone in view is named
    from the directory, and one it can't name has an empty ``identity`` (rendered as "某人", never a
    bare id) yet is still in the table; an empty ``condition`` omits the line.
    """
    identities = directory.agent_identity_map(spatial.visible_agent_ids)
    spatial.visible_agents = {
        agent_id: PerceivedPresence(
            identity=identities.get(agent_id) or PerceivedIdentity(),
            condition=(
                render_condition(agent.personality.state.condition)
                if (agent := agents.get(agent_id)) is not None else ""
            ),
        )
        for agent_id in spatial.visible_agent_ids
    }
    _attach_npcs(spatial, directory=directory, environment=environment)
