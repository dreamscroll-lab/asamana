"""What a director's intervention actually DID: the intervention receipt.

An intervention can be delivered correctly and still move nobody (the pressure evaluator decides
it doesn't push anyone), which from outside looks exactly like a bug. The receipt that matters
most is that negative one: delivered, but nobody took it seriously.

All four signals are already computed during the step; this just surfaces them:

| column | source | answers |
|---|---|---|
| delivered_to | delivery resolution / broadcast audience | whose senses it reached |
| pressure     | ``pending_external_goals`` (the pressure phase's verdict) | who actually felt pushed, and how hard |
| decided      | the scheduler's admission list this step | who was pulled into a decision loop |
| interrupted  | the interrupt coordinator's records | who dropped what they were doing |

It is an observation read model over the pressure, scheduling and interrupt phases, which
``DirectorChannel`` knows nothing about, so it lives here as a pure function rather than in the
channel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

from core.interfaces.directory import WorldDirectory
from core.interfaces.urgency import Urgency

if TYPE_CHECKING:
    from agent.agent import Agent


# urgency is a code-layer scale and the receipt is read by people: translate it on the way out.
_URGENCY_WORDS: dict[Urgency, str] = {
    Urgency.LOW: "轻微",
    Urgency.NORMAL: "一般",
    Urgency.HIGH: "紧急",
    Urgency.CRITICAL: "危急",
}


def build_receipt(
    target_ids: Sequence[str],
    *,
    agents: Mapping[str, "Agent"],
    directory: WorldDirectory,
    decided_ids: set[str],
    interrupted_ids: set[str],
) -> dict[str, Any]:
    """Condense the people an intervention touched plus this step's three signals into a
    human-readable receipt.

    ``target_ids`` are the people the intervention touched (direct recipients, broadcast
    audience, those a mutation acted on). The other three columns are filtered to this set:
    the receipt answers "who did this push move", not "who moved anywhere this step".
    """
    def named(agent_id: str) -> dict[str, str]:
        return {"agent_id": agent_id, "name": directory.agent_name(agent_id)}

    pressure: list[dict[str, Any]] = []
    for agent_id in target_ids:
        agent = agents.get(agent_id)
        if agent is None:
            continue
        # This step's pressure verdict. Take the heaviest: the receipt answers "how hard did this
        # land on them", not a list of every thought it stirred.
        goals = list(agent.pending_external_goals)
        if not goals:
            continue
        top = max(goals, key=lambda g: g.urgency.level)
        pressure.append({**named(agent_id), "urgency": _URGENCY_WORDS.get(top.urgency, "一般")})

    return {
        "delivered_to": [named(aid) for aid in target_ids],
        "pressure": pressure,
        "decided": [named(aid) for aid in target_ids if aid in decided_ids],
        "interrupted": [named(aid) for aid in target_ids if aid in interrupted_ids],
    }
