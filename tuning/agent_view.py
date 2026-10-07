"""Shape an initialized Agent into a JSON-safe, browsable post-init view.

Lives in tuning (not production): the build harness calls this to turn the
``world.agents`` produced by ``WorldInitializer`` into the structured product
shown under the build trace's "initialization" phase. Read-only — reuses the
production ``snapshot_agent_state`` for runtime state and augments it with the
soul (identity) and need profile that the compact snapshot omits.
"""

from __future__ import annotations

from typing import Any

from agent.agent import Agent, snapshot_agent_state


def initialized_agent_view(agent: Agent) -> dict[str, Any]:
    """Return a comprehensive, JSON-safe snapshot of one agent right after init."""
    soul = agent.personality.soul
    state = agent.personality.state
    view = dict(snapshot_agent_state(agent))  # emotion, goals(+entities), needs, location, vitality...
    view["soul"] = {
        "core_traits": list(soul.core_traits),
        "core_values": list(soul.core_values),
        "hard_constraints": list(soul.hard_constraints),
        "self_image": soul.self_image,
        "background": soul.background,
        "life_goal": soul.life_goal,
        "secret": soul.secret,
        "age": soul.age,
        "gender": soul.gender,
        "role": soul.role,
    }
    view["need_profile"] = {
        # Static needs come from soul.innate_needs; current intensities (seeded, then evolved by feedback) come from state.need_intensities.
        "active_needs": [
            {
                "type": n.type.value,
                "intensity": round(state.need_intensities.get(n.type.value, n.intensity), 3),
                "weight": round(n.weight, 3),
            }
            for n in soul.innate_needs if not n.is_hidden
        ],
        "hidden_needs": [n.type.value for n in soul.innate_needs if n.is_hidden],
        "long_term_goals": [e.text for e in state.long_term_goal_entities],
    }
    return view
