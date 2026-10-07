"""Death handling for the runtime step: passive vitality decay, then reaping this step's new
deaths (body removal, item drop, execution teardown, death broadcast).

Holds no per-step runtime state and never calls back into the runtime. Agent vitality is the
source of truth for death.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from agent.need import NeedType
from core.interfaces.action import EntityStateChange
from core.interfaces.perception import Broadcast, BroadcastType
from core.interfaces.severity import Severity
from core.logging import get_logger
from engine.broadcast import BroadcastChannel
from engine.environment import IN_TRANSIT, EnvironmentSystem
from engine.execution_processor import ExecutionProcessor

if TYPE_CHECKING:
    from agent.agent import Agent


logger = get_logger(__name__)

_VITALITY_DECAY_PER_STEP: float = 0.001
_PHYSIO_DECAY_THRESHOLD: float = 0.6
_PHYSIO_DECAY_FACTOR: float = 0.003


class DeathHandler:
    """Applies vitality decay and reaps new deaths."""

    def __init__(
        self,
        *,
        environment: EnvironmentSystem,
        processor: ExecutionProcessor,
        broadcast_channel: BroadcastChannel,
    ) -> None:
        self._environment = environment
        self._processor = processor
        self._broadcast_channel = broadcast_channel

    async def apply_vitality_decay(self, agents: List["Agent"], step: int) -> None:
        """Apply per-step passive vitality decay and PHYSIOLOGICAL-urgency extra decay."""
        for agent in agents:
            if not agent.is_active:
                continue
            decay = _VITALITY_DECAY_PER_STEP
            physio = agent.personality.state.need_intensities.get(NeedType.PHYSIOLOGICAL.value, 0.0)
            starving = physio > _PHYSIO_DECAY_THRESHOLD
            if starving:
                decay += (physio - _PHYSIO_DECAY_THRESHOLD) * _PHYSIO_DECAY_FACTOR
            # Causes carry their own final punctuation (see TargetAgentEffect.death_cause);
            # identity is added when the death notice is assembled.
            agent.apply_vitality_damage(
                decay, step=step, death_cause="饥馁力竭而亡。" if starving else "油尽灯枯。"
            )

    async def process_new_deaths(
        self, agents: List["Agent"], pre_step_active: set[str], step: int
    ) -> None:
        """The single world-side sink for deaths: every agent alive at step start
        (pre_step_active) and dead now is wrapped up here, whatever the cause.

        The death notice is an ordinary perception signal; survivors' reactions come from the
        perception layer.
        """
        agents_dict = {a.agent_id: a for a in agents}
        for agent in agents:
            if agent.agent_id not in pre_step_active or agent.is_active:
                continue
            # Tear down every execution the dead agent is in; otherwise they keep ticking, put
            # the corpse back into the world on completion, and leave survivors hanging. Force
            # teardown, not interrupt: the dead don't "decide to stop".
            await self._processor.force_teardown(
                agent, agents_dict, step,
                cause=f"{agent.personality.soul.name}已经失去生命力",
                trigger="death",
            )
            death_location = self._environment.get_body_location(agent.agent_id)
            # The notice says "on the road" for a death in transit; the rewrite to the origin
            # below is only so items have somewhere to land.
            death_place = (
                self._environment.narrative_location_name(death_location)
                if death_location == IN_TRANSIT or self._environment.space.has(death_location)
                else None
            )
            # Death in transit: items drop at the origin. IN_TRANSIT is a pseudo-location; items
            # left there resolve to no room and are lost for good.
            if death_location == IN_TRANSIT:
                death_location = (
                    self._environment.transit_origin(agent.agent_id) or death_location
                )

            # Drops go through the change_entity_state sink; never edit placement directly.
            for item in self._environment.get_items_of(agent.agent_id):
                if not item.is_takeable:
                    continue
                self._environment.change_entity_state(
                    EntityStateChange(
                        entity_id=item.entity_id,
                        location_id=death_location,
                    )
                )
            self._environment.remove_body(agent.agent_id)

            # The role lets people who didn't know the deceased tell who it was (the notice is
            # world-wide). Don't add final punctuation: the cause carries its own (see
            # TargetAgentEffect.death_cause). deliver_step is the next step because this step's
            # perception has already run.
            soul = agent.personality.soul
            identity = f"{soul.name}（{soul.role}）" if soul.role else soul.name
            content = f"{identity}{agent.death_cause}" if agent.death_cause else f"{identity}生命力耗尽。"
            if death_place:
                content = f"在{death_place}，{content}"
            self._broadcast_channel.publish(Broadcast(
                content=content,
                source="system",
                broadcast_type=BroadcastType.WORLD_EVENT,
                deliver_step=step + 1,
                location_scope=None,
                severity=Severity.HIGH,
            ))
            logger.info(
                "death_processed",
                extra={"agent_id": agent.agent_id, "step": step, "location": death_location},
            )

