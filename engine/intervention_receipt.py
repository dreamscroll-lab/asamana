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

if TYPE_CHECKING:
    from agent.agent import Agent


def build_receipt(
    target_ids: Sequence[str],
    *,
    agents: Mapping[str, "Agent"],
    decided_ids: set[str],
    interrupted_ids: set[str],
) -> dict[str, Any]:
    """Condense the people an intervention touched plus this step's three signals into a receipt.

    ``target_ids`` are the people the intervention touched (direct recipients, broadcast
    audience, those a mutation acted on). The other three columns are filtered to this set:
    the receipt answers "who did this push move", not "who moved anywhere this step".
    People are ids and urgency is the ``Urgency`` value: a reader resolves names and words itself.
    """
    pressure: list[dict[str, str]] = []
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
        pressure.append({"agent_id": agent_id, "urgency": top.urgency.value})

    return {
        "delivered_to": list(target_ids),
        "pressure": pressure,
        "decided": [aid for aid in target_ids if aid in decided_ids],
        "interrupted": [aid for aid in target_ids if aid in interrupted_ids],
    }
