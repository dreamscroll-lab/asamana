"""Physical action executor for PHYSICAL actions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from agent.personality import EmotionType, parse_emotion_type
from agent.relation import (
    NEUTRAL_AFFECTION, NEUTRAL_TRUST, format_relation_block, parse_relation_direction,
)
from core.context import GivenFacts, annotate_call, observe_stage
from core.interfaces.action import (
    KIND_OBJECT, ActionResult, ActionType, AgentAction, Deed, EntityStateChange,
    NpcEffect, TargetAgentEffect, parse_deed,
)
from core.interfaces.directory import WorldDirectory
from core.coerce import coerce_float, coerce_int, coerce_str
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.duration import describe_duration
from core.interfaces.condition import BodyCondition, npc_condition_fallback_steps
from core.prompts import (
    render_npc,
    CLOSED_WORLD_FACT_RULE,
    VITALITY_DAMAGE_DEFINITION,
    VITALITY_RELIEF_CAP,
    VITALITY_RELIEF_DEFINITION,
    deed_options,
    SituationVoice,
    condition_line,
    emotion_legend,
    person_referent,
    render_condition,
    render_entity,
    vitality_line,
)
from core.interfaces.execution import TickResult
from engine.executors.base import (
    ActionExecutionState,
    ActionExecutor,
)
from engine.environment import SALIENT_AMBIENT_STRENGTH
from engine.narration import SAME_PLACE_VERDICT_RULE, ensure_actor_named, observed_here, scene_line
from engine.scene import (
    IDLE_BYSTANDER_VERDICT_RULE, SceneVisibility, assemble_scene_context, observe_location,
    scene_npc_view,
)

if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.action import ActionTarget
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem
    from world.models import WorldEntity

logger = get_logger(__name__)

#: Phrasings in the judge's ``new_entity_state`` that explicitly say the thing no longer exists.
#: Only this tier counts, not "damaged / cracked / shattered"-style "broken but still there"
#: (see the gone-state guard in ``_resolve_deed``).
_GONE_STATE = re.compile(
    r"destroy(ed)?|销毁|焚毁|烧毁|毁灭|不复存在|不再存在|化为灰烬|荡然无存"
)

# The LLM judges relation direction; magnitudes are constants shared by both sides, so one
# interaction changes them symmetrically. Same scale as TALK (_DIRECTION_TALK_DELTAS), each
# tier 0.01 heavier; an attempted negative act weighs the same as hostile words. Keep them
# small: relations shift over many interactions, not in one step.
_REL_DELTA_POSITIVE: tuple[float, float] = (0.04, 0.05)
_REL_DELTA_NEGATIVE_DONE: tuple[float, float] = (-0.03, -0.03)
_REL_DELTA_NEGATIVE_UNDONE: tuple[float, float] = (-0.02, -0.02)


def _relation_delta(direction: str, succeeded: bool) -> tuple[float, float] | None:
    """Map LLM-judged relation direction to the calibrated (trust, affection) delta."""
    if direction == "positive":
        return _REL_DELTA_POSITIVE
    if direction == "negative":
        return _REL_DELTA_NEGATIVE_DONE if succeeded else _REL_DELTA_NEGATIVE_UNDONE
    return None


def _affected_person_id(
    deed: Deed,
    *,
    actor_id: str,
    prior_owner_id: str | None,
    recipient_id: str | None,
) -> str | None:
    """Whose hands this act took the thing out of, or put it into.

    Keyed on the deed, as the mutation is, so the two can't disagree. Ignores success: an
    attempted robbery still has a victim.
    """
    if deed is Deed.RELINQUISH:
        return recipient_id or None
    if prior_owner_id and prior_owner_id != actor_id:
        return prior_owner_id
    return None


#: The affected person's second-person event line per deed, for the reaction stage.
_POSSESSION_EVENT_LINES: dict[Deed, str] = {
    Deed.SEIZE:      "{actor}要从你手里夺走{item}",
    Deed.DESTROY:    "{actor}要毁掉你手里的{item}",
    Deed.RELINQUISH: "{actor}要把{item}交到你手上",
    Deed.OPERATE:    "{actor}动了你手里的{item}",
}


# Tiers for the loss from one action (scale: core.prompts.VITALITY_DAMAGE_DEFINITION), distinct
# from vitality_label's current-vitality tiers. Say how much, never why.
def _damage_label(damage: float) -> str:
    if damage <= 0.0:
        return "分毫未损"
    if damage < 0.3:
        return "略有损耗"
    if damage < 0.7:
        return "生命力大损"
    return "危及生命"


# Tiers for relief (scale: core.prompts.VITALITY_RELIEF_DEFINITION). Say how much, never why.
def _relief_label(relief: float) -> str:
    if relief < 0.05:
        return "稍稍缓过来一点"
    if relief < 0.15:
        return "生命力回了一些"
    return "从濒死边上被拉了回来"


# Onlooker strength for a light wound or a missed blow: recorded at MEDIUM importance (≥0.4,
# <0.65) and above background agents' perception threshold (0.35). Serious wounds and restraint
# use SALIENT_AMBIENT_STRENGTH and record as HIGH.
_LIGHT_VIOLENCE_STRENGTH: float = 0.45


def _bystander_strength(verdict: "_Verdict") -> float | None:
    """How conspicuous this act is to onlookers; None = ordinary ambient.

    - Noticeable loss (≥ 0.3) on either side, or a lasting condition imposed: strong signal.
      Either side, because someone can get hurt without the deed being a strike.
    - Any other strike, even a miss: one tier lower.
    - restrain alone and relief don't count: they also cover helping someone.
    """
    hurt = max(verdict.target_damage, verdict.actor_damage)
    if hurt >= 0.3 or verdict.target_condition_desc or verdict.actor_condition_desc:
        return SALIENT_AMBIENT_STRENGTH
    if verdict.deed == Deed.STRIKE:
        return _LIGHT_VIOLENCE_STRENGTH
    return None


@dataclass(frozen=True)
class _Subject:
    """What this act is aimed at, resolved once before adjudication. The deed menu, the
    mutation and the affected person all read possession facts from here so they can't
    disagree; those facts are derived properties, never stored copies.
    """

    actor_id:        str
    item_type:       str                  # routing key code derives from the slot, not an LLM label (§6)
    target_id:       str
    target_block:    str                  # the rendered 【目标】 block (narrative layer, no ids)
    item:            "WorldEntity | None"  # the registered entity this act affects, else None
    target_agent:    "Agent | None"        # non-empty only when acting on a person
    recipient_id:    str                   # recipient of a handover (meaningful only on the entity slot)
    recipient_agent: "Agent | None"

    @property
    def is_person(self) -> bool:
        """This act lands on a body, thinking or mindless; the judge weighs both alike. Only
        where consequences land differs (``is_npc``)."""
        return self.item_type in ("agent", "npc")

    @property
    def is_npc(self) -> bool:
        """Consequences go through ``NpcEffect`` rather than ``TargetAgentEffect``: there's no
        inner life to write to."""
        return self.item_type == "npc"

    @property
    def prior_owner_id(self) -> str | None:
        """Who held this thing before the act (no registered entity, or it's on the ground →
        None)."""
        return self.item.owner_id if self.item is not None else None

    @property
    def actor_holds(self) -> bool:
        return self.prior_owner_id is not None and self.prior_owner_id == self.actor_id

    @property
    def seize_available(self) -> bool:
        """Takeable and not already in his own hands; read by both the judge menu and
        ``_resolve_deed``."""
        return self.item is not None and self.item.is_takeable and not self.actor_holds

    @property
    def needs_relation(self) -> bool:
        """Ask the judge for relation only when the act lands on a person with cognition or on
        what belongs to one. Never for an Npc: asking invites the judge to give a body a stance."""
        return (
            (self.is_person and not self.is_npc)
            or bool(self.recipient_id)
            or (self.prior_owner_id is not None and not self.actor_holds)
        )


@dataclass(frozen=True)
class _Verdict:
    """World facts adjudicated by the functional judge (single source of truth)."""

    success: bool
    # outcome: third person, names actor + target, full authority. Physical acts are fully
    # public, so it doubles as the onlooker observation channel.
    outcome: str
    fact: str                  # first person (actor's view), the actor's own memory channel
    failure_reason: str        # 3p "why it failed" (≤20 chars, "" on success); never in observation
    relation_dir: str          # "positive" | "negative" | "neutral" — actor toward target
    # Don't add an actor relief slot: nothing gates it, so self-healing would skip REST's time cost.
    actor_damage: float
    # Kept separate, not netted at parse time: narrative channels read each direction.
    target_damage: float       # only meaningful when target is a person
    target_relief: float       # only meaningful when target is a person
    new_entity_state: str      # only meaningful when target is a tracked entity
    deed: Deed                 # what the actor was doing, regardless of success; the judge's
    # call because only it reads the intent ("掰开" vs "砸烂" the same gate). Drives the
    # entity mutation and the observer's picture.
    #
    # Lasting conditions, split by person (see ActionResult.actor_condition_set). The only
    # defaults in the class: most acts leave none, while every field above must be answered.
    target_condition_desc: str = ""
    target_condition_steps: int = 0    # >0 = lifts on expiry; 0 = needs outside help
    frees_target: bool = False         # this act lifted the target's existing condition
    actor_condition_desc: str = ""
    actor_condition_steps: int = 0
    frees_actor: bool = False          # lifted the actor's existing condition (broke free / let go)


class PhysicalExecutor(ActionExecutor):
    """Executor for PHYSICAL actions: a two-stage pipeline in complete() (born-zero).

    1. ``_judge`` (functional referee) adjudicates world facts: success, damage, entity state.
       The actor's personality is evidence, not the viewpoint.
    2. ``_llm_target_reaction`` (in-character) writes the counterpart's subjective experience
       from the judged facts; it never decides physical facts.

    Routing follows ``ActionTarget.acted_on_kind`` (from the decision slot, never an LLM): a
    person is co-location checked; anything else is looked up in the environment, and a thing
    it can't find is narrated only.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        directory: WorldDirectory,
        *,
        seconds_per_step: int = 3600,
    ) -> None:
        self._llm = llm_router
        self._directory = directory
        # Only for conditions: durations down to the judge, and the step unit taught up so it can
        # answer in steps.
        self._seconds_per_step = seconds_per_step

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        agent_id = action.agent_id
        actor_name = self._directory.agent_name(agent_id)
        location = observe_location(environment, agent_id)
        # Fixed by which decision slot was filled, never a self-reported category (§6).
        item_type = action.target.acted_on_kind
        target_id = action.target.acts_on[0].id if action.target.acts_on else ""
        description = action.action_description or "physical action"

        feasibility = environment.check_physical_feasibility(agent_id, target_id or None, item_type)
        reason = "" if feasibility.ok else feasibility.reason
        # Rendered here (env has no names); never reveal where the target really is.
        if reason and item_type == "agent" and target_id:
            target_name = self._directory.agent_name(target_id)
            reason = f"{target_name}不在{location}，无法对其施加物理行动"

        # The recipient may have walked off since planning; same check as a person target.
        recipient_id = next(iter(action.target.reached_agents), "")
        if not reason and recipient_id:
            recipient_check = environment.check_physical_feasibility(agent_id, recipient_id, "agent")
            if not recipient_check.ok:
                # "其" rather than "他": the recipient's gender isn't known here.
                reason = (
                    f"{self._directory.agent_name(recipient_id)}不在{location}，"
                    "无法把东西交到其手上"
                )

        if reason:
            stub = AgentAction(
                agent_id=agent_id,
                step=step,
                action_type=ActionType.PHYSICAL,
                action_description=description,
                target=action.target,
            )
            outcome = scene_line(location, f"{actor_name}本想做「{description}」，却因{reason}未能做成。")
            failure = ActionResult(
                action=stub,
                expected_outcome=action.expected_outcome,
                outcome=outcome,
                observations=observed_here(environment, agent_id, outcome),
                succeeded=False,
                failure_reason=reason,   # structured fact (target absent / out of reach), not LLM-invented
                factual_memory=f"尝试物理行动未果：{reason}",
            )
            return ActionExecutionState.create_failed(
                action_type=ActionType.PHYSICAL, initiator_id=agent_id,
                failure_result=failure, started_step=step, purpose=description,
            )

        # Stash only ids for complete(), never names.
        state = ActionExecutionState.create(
            action_type=ActionType.PHYSICAL,
            initiator_id=agent_id,
            participant_ids=[agent_id],
            purpose=description,
            started_step=step,
            estimated_steps=1,
            opening_outcome="",
            target=action.target,
            expected_outcome=action.expected_outcome,
        )
        state.extra["resolved_id"] = feasibility.resolved_id or target_id
        if recipient_id:
            state.extra["recipient_id"] = recipient_id
        return state

    async def _resolve_subject(
        self, state: ActionExecutionState, *,
        agents: dict[str, "Agent"], environment: "EnvironmentSystem", step: int,
    ) -> "_Subject":
        """Resolve what this act is aimed at, once, before adjudication (see ``_Subject``)."""
        agent_id = state.initiator_id
        target = state.target
        item_type = target.acted_on_kind if target is not None else "none"
        target_id = (target.acts_on[0].id if target.acts_on else "") if target is not None else ""
        # A lookup key, not an id: may be the LLM's free text. Look up with it, never bind it.
        lookup_key = state.extra.get("resolved_id", "") or target_id
        recipient_id = state.extra.get("recipient_id", "") or ""

        item = None
        target_agent = None
        if item_type == "agent":
            target_agent = agents.get(target_id) if target_id else None
            target_block = await self._person_target_block(
                agents.get(agent_id), target_agent, agent_id, target_id, step=step,
            )
        elif item_type == "npc":
            # Only what's visible face to face. The no-cognition note must be said explicitly,
            # or the judge hands a mindless body a stance that lands in memory. Don't list
            # judgeable deeds: naming some pushes the rest out of view. The "it didn't happen"
            # clause keeps ``success`` honest when the act needed his understanding; a condition
            # still lands either way.
            npc = environment.get_npc(target_id) if target_id else None
            target_block = (
                f"{render_npc(scene_npc_view(npc))}\n"
                "（他没有自己的认知：不权衡、不拒绝，也不答应。所以凡是要他领会、答应、"
                "照办才成立的部分，一概不会发生——**若这一动图的正是那个，就是没成**；"
                "而落在他身上的动作照常裁，真发生了的照写。）"
                if npc is not None
                else self._directory.agent_name(target_id)
            )
        else:
            # Look up first, don't whitelist type strings: a missed type would silently be judged
            # as an abstract noun. Scope the name lookup to the actor's location so a namesake
            # elsewhere can't answer.
            item = environment.find_item(
                lookup_key, environment.get_body_location(agent_id)
            ) if lookup_key else None
            target_block = (
                self._entity_target_block(item, actor_id=agent_id) if item is not None
                else self._unresolved_target_block(target_id, item_type)
            )
            # Holder and recipient need the same evidence as a person target (vitality, emotion,
            # relation), which the scene block doesn't give.
            holder = (
                agents.get(item.owner_id)
                if item is not None and item.owner_id and item.owner_id != agent_id else None
            )
            recipient = agents.get(recipient_id) if recipient_id else None
            for role, other in (("持有者", holder), ("接收人", recipient)):
                if other is not None:
                    target_block += f"\n\n【{role}】\n" + await self._person_target_block(
                        agents.get(agent_id), other, agent_id, other.agent_id,
                        step=step, role=role,
                    )

        return _Subject(
            actor_id=agent_id, item_type=item_type, target_id=target_id,
            target_block=target_block, item=item,
            target_agent=target_agent, recipient_id=recipient_id,
            recipient_agent=agents.get(recipient_id) if recipient_id else None,
        )

    def _declare_entity_change(
        self, verdict: "_Verdict", subject: "_Subject", *,
        location: str, environment: "EnvironmentSystem",
    ) -> list[EntityStateChange]:
        """How the world changes because of this act, decided by deed alone. ``_resolve_deed``
        has already aligned the deed with what the world allows."""
        item = subject.item
        if item is None or not verdict.success:
            return []
        actor_name = self._directory.agent_name(subject.actor_id)
        if verdict.deed == Deed.DESTROY:
            return [EntityStateChange(
                entity_id=item.entity_id,
                new_state=verdict.new_entity_state or "destroyed",
                destroyed=True,
                perception=scene_line(location, f"{actor_name}摧毁了{item.name}。"),
            )]
        if verdict.deed == Deed.SEIZE:
            return [EntityStateChange(
                entity_id=item.entity_id,
                owner_id=subject.actor_id,
                perception=scene_line(location, f"{actor_name}拿起了{item.name}。"),
            )]
        if verdict.deed == Deed.RELINQUISH:
            recipient_id = subject.recipient_id
            return [EntityStateChange(
                entity_id=item.entity_id,
                owner_id=recipient_id or None,
                location_id=None if recipient_id else environment.get_body_location(subject.actor_id),
                perception=scene_line(location, (
                    f"{actor_name}把{item.name}交到了{self._directory.agent_name(recipient_id)}手上。"
                    if recipient_id else f"{actor_name}放下了{item.name}。"
                )),
            )]
        return [EntityStateChange(
            entity_id=item.entity_id,
            new_state=verdict.new_entity_state or "used",
            perception=scene_line(location, f"{actor_name}操作/使用了{item.name}。"),
        )]

    def _resolve_counterpart(
        self, verdict: "_Verdict", subject: "_Subject", *,
        agents: dict[str, "Agent"], environment: "EnvironmentSystem",
    ) -> tuple[str | None, "Agent | None"]:
        """Who besides the actor is materially affected: (id, live object). The target for an
        act on a person; for a thing, whose hands it left or went into (by deed).

        Returned separately: the relation delta needs only the id, the reaction needs the
        object, and one boolean would let a missing object swallow the relation change.
        """
        if subject.is_npc:
            # Npcs have no relations or reactions; consequences go through npc_effects.
            return None, None
        if subject.is_person:
            return (subject.target_id or None), subject.target_agent
        if subject.item is None:
            return None, None
        candidate = _affected_person_id(
            verdict.deed,
            actor_id=subject.actor_id,
            prior_owner_id=subject.prior_owner_id,
            recipient_id=subject.recipient_id or None,
        )
        other = agents.get(candidate) if candidate else None
        # Only someone present and alive experienced it.
        if (
            other is None
            or not other.is_active
            or environment.get_body_location(candidate) != environment.get_body_location(subject.actor_id)
        ):
            return None, None
        return candidate, other

    async def _counterpart_effect(
        self, verdict: "_Verdict", subject: "_Subject", counterpart: "Agent", *,
        step: int,
    ) -> TargetAgentEffect:
        """The counterpart's effect: experience from the reaction stage, harm from the judge."""
        actor_name = self._directory.agent_name(subject.actor_id)
        is_person = subject.is_person
        # The judge's 3p outcome, not the actor's first-person description: an embedded "I"
        # would be read as the receiver himself, and the intent may contradict the verdict.
        event_line = verdict.outcome
        template = (
            "" if is_person or subject.item is None
            else _POSSESSION_EVENT_LINES.get(verdict.deed, "")
        )
        if template and subject.item is not None:
            event_line = (
                f"{template.format(actor=actor_name, item=subject.item.name)}"
                f"，结果：{'做成了' if verdict.success else '没能做成'}"
            )
        effect = await self._llm_target_reaction(
            target_agent=counterpart,
            actor_id=subject.actor_id,
            actor_name=actor_name,
            event_line=event_line,
            succeeded=verdict.success,
            # Harm applies only to acts on people.
            target_damage=verdict.target_damage if is_person else 0.0,
            target_relief=verdict.target_relief if is_person else 0.0,
            step=step,
        )
        # Net of relief, the same number that lands.
        net_damage = effect.vitality_damage
        # See TargetAgentEffect.death_cause for the format.
        if (
            net_damage > 0
            and is_person
            and counterpart.personality.state.vitality - net_damage <= 0.0
        ):
            effect.death_cause = f"被{actor_name}耗尽最后一滴生命力，已失去生命。"
        # Declared only; lands in Agent.apply_target_effect. Neither set = leave his condition alone.
        if verdict.target_condition_desc:
            effect.condition_set = BodyCondition(
                description=verdict.target_condition_desc,
                source_agent_id=subject.actor_id,
                since_step=step,
                # 0 = needs outside help, never expires on its own.
                until_step=(
                    (step + verdict.target_condition_steps)
                    if verdict.target_condition_steps > 0 else None
                ),
            )
        elif verdict.frees_target:
            effect.condition_cleared = True
        return effect

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
            return [stored]  # feasibility failure

        agent_id = state.initiator_id
        actor_name = self._directory.agent_name(agent_id)
        location = observe_location(environment, agent_id)
        target = state.target
        description = state.purpose
        agent = agents.get(agent_id)

        subject = await self._resolve_subject(
            state, agents=agents, environment=environment, step=step,
        )

        verdict = await self._judge(
            agent,
            description=description,
            expected_outcome=state.expected_outcome,
            scene=assemble_scene_context(
                agent_id, environment=environment, directory=self._directory,
                agents=agents, with_background=True, visibility=SceneVisibility.GOD,  # functional judge
            ).text,
            target_block=subject.target_block,
            is_person=subject.is_person,
            has_entity=subject.item is not None,
            seize_available=subject.seize_available,
            actor_holds=subject.actor_holds,
            needs_relation=subject.needs_relation,
            recipient_line=self._recipient_line(subject.recipient_id, subject.recipient_agent),
            step=step,
        )

        # No adjudication: null step, nothing fabricated (Rule 1).
        if verdict is None:
            return [self._adjudication_failed_result(target, state.expected_outcome, step, agent_id, description, location)]

        # Both read verdict.deed only, so the world change and the affected person can't disagree.
        entity_state_changes = self._declare_entity_change(
            verdict, subject, location=location, environment=environment,
        )
        counterpart_id, counterpart_agent = self._resolve_counterpart(
            verdict, subject, agents=agents, environment=environment,
        )

        relation_updates: list[tuple[str, float, float]] = []
        if counterpart_id and agent is not None:
            delta = _relation_delta(verdict.relation_dir, verdict.success)
            if delta is not None:
                relation_updates = [(counterpart_id, *delta)]

        target_effects: list[TargetAgentEffect] = []
        if counterpart_agent is not None:
            target_effects.append(await self._counterpart_effect(
                verdict, subject, counterpart_agent, step=step,
            ))

        # An Npc takes only a condition; harm, emotion and relation have nowhere to go. Its
        # errand stalls via the condition (see ``NpcEffect``).
        npc_effects: list[NpcEffect] = []
        if subject.is_npc and subject.target_id:
            if verdict.target_condition_desc:
                npc_effects.append(NpcEffect(
                    npc_id=subject.target_id,
                    condition_set=BodyCondition(
                        description=verdict.target_condition_desc,
                        source_agent_id=agent_id,
                        since_step=step,
                        # 0 ("needs outside help") falls back to a floor: an Npc can't free
                        # itself (see NPC_CONDITION_FALLBACK_SECONDS).
                        until_step=step + (
                            verdict.target_condition_steps
                            if verdict.target_condition_steps > 0
                            else npc_condition_fallback_steps(self._seconds_per_step)
                        ),
                    ),
                ))
            elif verdict.frees_target:
                npc_effects.append(NpcEffect(
                    npc_id=subject.target_id, condition_cleared=True,
                ))

        # Same rules as the target's condition.
        actor_condition_set: "BodyCondition | None" = None
        if verdict.actor_condition_desc:
            actor_condition_set = BodyCondition(
                description=verdict.actor_condition_desc,
                # No source agent: the verdict doesn't say who; don't guess.
                since_step=step,
                until_step=(
                    (step + verdict.actor_condition_steps)
                    if verdict.actor_condition_steps > 0 else None
                ),
            )

        stub = AgentAction(
            agent_id=agent_id,
            step=step,
            action_type=ActionType.PHYSICAL,
            action_description=description,
            target=target,
            estimated_steps=1,
        )
        outcome = scene_line(location, ensure_actor_named(verdict.outcome, actor_name))
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome,
            outcome=outcome,       # 3p full authority
            observations=observed_here(
                environment, agent_id, outcome, strength=_bystander_strength(verdict),
            ),
            succeeded=verdict.success,
            failure_reason=verdict.failure_reason,   # 3p authoritative "why it failed"
            factual_memory=verdict.fact,    # 1p → the actor's own memory channel
            relation_updates=relation_updates,
            target_effects=target_effects,
            npc_effects=npc_effects,
            vitality_damage=verdict.actor_damage,
            actor_condition_set=actor_condition_set,
            actor_condition_cleared=(actor_condition_set is None and verdict.frees_actor),
            entity_state_changes=entity_state_changes,
            deed=verdict.deed.value,  # what he was seen doing — carried even when he failed
        )]

    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        return []  # PHYSICAL is born-zero (duration-1): no ticks, completes on its start step

    async def _person_target_block(
        self,
        agent: "Agent | None",
        target_agent: "Agent | None",
        agent_id: str,
        target_id: str,
        *,
        step: int,
        role: str = "目标",
    ) -> str:
        """The person evidence block for the judge, with relations in both directions (his
        hostility decides how hard he resists). ``role``: "目标", "持有者" or "接收人"."""
        target_name = self._directory.agent_name(target_id)
        if target_agent is None:
            return f"{target_name}（人，详情不明）"

        lines = [
            f"{target_name}（人）",
            target_agent.personality.to_prompt_context(include_emotion=True, include_goals=False),
            vitality_line(target_agent.personality.state.vitality,
                          voice=SituationVoice.THIRD, lead=""),
        ]
        # Otherwise the judge weighs his resistance as if he were free; duration is evidence too.
        target_condition = condition_line(
            target_agent.personality.state.condition, voice=SituationVoice.THIRD,
            now_step=step, seconds_per_step=self._seconds_per_step, lead="",
        )
        if target_condition:
            lines.append(target_condition)
        if agent is not None:
            # Read-only: a missing relation renders as neutral and stays missing.
            rel_ab = await agent.relation_system.load_existing(target_id)
            lines.append(f"行动者对{role}的关系：" + format_relation_block(
                labels=rel_ab.labels if rel_ab else [],
                trust=rel_ab.trust_objective if rel_ab else NEUTRAL_TRUST,
                affection=rel_ab.affection_objective if rel_ab else NEUTRAL_AFFECTION,
            ))
            rel_ba = await target_agent.relation_system.load_existing(agent_id)
            lines.append(f"{role}对行动者的关系：" + format_relation_block(
                labels=rel_ba.labels if rel_ba else [],
                trust=rel_ba.trust_objective if rel_ba else NEUTRAL_TRUST,
                affection=rel_ba.affection_objective if rel_ba else NEUTRAL_AFFECTION,
                include_legend=False,  # the legend only needs to appear once
            ))
        return "\n".join(lines)

    def _unresolved_target_block(self, target_id: str, item_type: str) -> str:
        """What the judge reads when the target isn't in the world: the actor's own words on the
        free-text branch, else a descriptive referent (``target_id`` is then a real id, which
        must never enter a prompt)."""
        if item_type == KIND_OBJECT:
            return target_id or "某物"
        return self._directory.entity_name(target_id)

    def _entity_target_block(self, item: "WorldEntity", *, actor_id: str) -> str:
        type_label = "物品" if item.is_takeable else "据点标记"
        # Ownership matters: taking or damaging someone else's thing is a contest or an offense.
        owner_name = self._directory.agent_name(item.owner_id) if item.owner_id else None
        # No content: the outcome goes verbatim to onlookers, and the whole room would read the
        # letter. Takeable is from the actor's view, to match the deed options.
        return f"【{type_label}】{render_entity(item, owner_name=owner_name, held_by_viewer=item.owner_id == actor_id)}"

    def _recipient_line(self, recipient_id: str, recipient_agent: "Agent | None") -> str:
        """The man the actor means to hand the thing to — a roster line, so it goes through
        ``person_referent`` like every other one. Empty when nobody was named."""
        if not recipient_id:
            return ""
        entry = self._directory.describe(recipient_id)
        condition = (
            render_condition(recipient_agent.personality.state.condition)
            if recipient_agent is not None else ""
        )
        referent = person_referent(
            self._directory.agent_name(recipient_id),
            entry.gender if entry is not None else "",
            "在场",
            *((condition,) if condition else ()),
        )
        return f"\n- 要交到谁手上：{referent}"

    async def _judge(
        self,
        agent: "Agent | None",
        *,
        description: str,
        expected_outcome: str,
        scene: str,
        target_block: str,
        is_person: bool,
        has_entity: bool,
        seize_available: bool,
        actor_holds: bool,
        needs_relation: bool,
        recipient_line: str,
        step: int,
    ) -> "_Verdict | None":
        """Neutral third-party adjudication of world facts. Returns ``None`` when adjudication
        can't happen (LLM failure, missing actor), so the caller fabricates nothing."""
        if agent is None:
            logger.warning("physical_judge_no_actor", extra={"description": description})
            return None

        expected_part = f"\n- 期望结果：{expected_outcome}" if expected_outcome else ""
        # Evidence for the escape hatch: whether a bound man struggling gets free.
        actor_condition_line = condition_line(
            agent.personality.state.condition, voice=SituationVoice.THIRD,
            now_step=step, seconds_per_step=self._seconds_per_step,
        )
        actor_vitality_line = vitality_line(
            agent.personality.state.vitality, voice=SituationVoice.THIRD, lead="",
        )

        relation_field = (
            ' "relation": "positive或negative或neutral（行动者对此事所涉他人的关系走向）",'
            if needs_relation else ""
        )
        person_fields = ""
        target_condition_fields = ""
        # Person-only explanations toggle with the schema, or they tempt the model to emit the keys.
        relief_rules = ""
        relief_definition = ""
        target_condition_doc = ""
        condition_pairing_rule = ""
        # A verdict on a person never changes item ownership: if the judge wrote a handover,
        # both sides' memories would disagree with the world state.
        possession_rule = ""
        if is_person:
            possession_rule = (
                "\n- 这一行动落在人身上，**不会让任何东西易手**：行动描述里若要从他身上夺取、"
                "或交到他手上某样东西，那一部分不成立——只裁身体上实际发生了什么，不得写成谁拿到了、"
                "谁交出了什么。"
            )
            person_fields = (
                ' "target_damage": 0.0到1.0之间的数字（目标受到的生命力损耗）,'
                f' "target_relief": 0.0到{VITALITY_RELIEF_CAP}之间的数字（这一动当场为目标挽回的生命力，'
                '不论他亏在哪里；没有这样的举动一律为 0）,'
            )
            relief_definition = f"\n{VITALITY_RELIEF_DEFINITION}"
            relief_rules = (
                "\n- 挽回生命力必有所凭借："
                "行动确实可以挽回生命力或此人的身份本事确能如此，target_relief 才可以大于 0；否则一律 0，**不得凭空挽回**。\n"
                "- 亏在哪里都算：负伤、力竭、饥渴、寒冷皆可，不要只认伤。相应地，"
                "止血包扎、递食喂水、解开勒缚、生火取暖，只要符合常量，都算。\n"
                "- target_relief 只管**当场挽回**（把眼下最要命的那一样解掉、让他缓过来），不管复元："
                "真正完全挽回不在这一行动中发生。它同样必须与 fact、outcome 自洽"
                "——没写出相应的举动就不该有挽回。"
            )
            # Conditions follow the verdict (§5b). Step counts are a controlled upward channel
            # because the step unit is taught here; it's a world constant, safe for the prefix cache.
            target_condition_doc = f"""\
- target_condition：这一行动在下面【目标】那一栏的那个人身上留下的、之后一直成立的身体状态；只有真跨越此刻的才算——一次擦伤、一次推搡、一次没能得手的尝试都不留下，给空字符串。写的是旁人当面看得出来的样子，不是他的心情、他的打算，也不是对局面的评价。
- target_condition_steps：这个处境若会自行消退，按「一步{describe_duration(1, self._seconds_per_step)}」折成整数步；若须旁人动手或他自己挣脱才会消失，填 0。
- frees_target：仅当这一动**解开了**目标原有的持续处境才为 true；默认 false——一记打在被缚者身上的拳并不会解开他的绳子。
"""
            # Only needed when both groups exist; keep it next to the schema, not between them.
            condition_pairing_rule = (
                "- 两个处境槽各写各的人，别串：落在**行动者**身上的写进 actor_condition，"
                "落在**目标**身上的才写进 target_condition。\n"
            )
            target_condition_fields = (
                ', "frees_target": true或false（这一动是否解除了目标原有的持续处境；'
                '目标本来就没有持续处境、或这一动与之无关，一律 false）,'
                ' "target_condition": "这一动在下面【目标】那一栏的那个人身上留下的、之后一直成立的身体状态（≤15字）；'
                '照实写你刚裁定的那一桩，别套用惯常说法；没留下任何持续处境就给空字符串",'
                ' "target_condition_steps": 这个处境大约多少步之后会自行消退的整数'
                '（须外力解除才会消失的填 0）'
            )
        # Always present: it doesn't depend on the target, and frees_actor is the only way a
        # bound man gets free.
        actor_condition_doc = f"""\
- actor_condition：这一行动之后落在**行动者自己**身上的、一直成立的身体状态——**不论是谁造成的**：旁人当场把他制住的，也算在他身上。同样只有真跨越此刻的才算，没有就给空字符串；写旁人当面看得出来的样子，不是他的心绪。
- actor_condition_steps：这个处境若会自行消退，按「一步{describe_duration(1, self._seconds_per_step)}」折成整数步；若须旁人动手或他自己挣脱才会消失，填 0。
- frees_actor：仅当这一动**解开了**行动者原有的持续处境才为 true（他挣脱了、有人放开了他）；默认 false。
- 他原有的处境若仍然成立，处境槽就**留空**：填了就是**替换**——别换个说法把它重述一遍。
{condition_pairing_rule}"""
        actor_condition_fields = (
            ', "frees_actor": true或false（这一动是否解除了行动者原有的持续处境；'
            '他本来就没有持续处境、或这一动与之无关，一律 false）,'
            ' "actor_condition": "这一动之后落在行动者自己身上的、一直成立的身体状态（≤15字）；'
            '照实写你刚裁定的那一桩；没有就给空字符串",'
            ' "actor_condition_steps": 这个处境大约多少步之后会自行消退的整数'
            '（须外力解除才会消失的填 0）'
        )
        entity_fields = ""
        if has_entity:
            entity_fields = (
                ' "new_entity_state": "行动后**这件东西自己**变成什么样的状态词（≤12字）'
                '（如 open/used/torn/activated 等）；'
                '不写它在谁手上、在哪儿（那不是状态），也不写另一样东西的状态；'
                '这件东西若被彻底毁掉，那由 deed 答 destroy，这里不写",'
            )

        # deed is the act, answered regardless of success; candidates narrow by target shape.
        if is_person:
            deed_field = (
                ' "deed": "行动者在对他做什么，二选一：'
                f'{deed_options("strike", "restrain")}",'
            )
        elif has_entity:
            # Narrow by ownership too: an impossible option only invites a verdict code must
            # downgrade. Handing over is an entity deed only, never restrain, or real transfers
            # would route to the one that mutates nothing.
            extra_deed = ("seize",) if seize_available else ("relinquish",) if actor_holds else ()
            deed_field = (
                ' "deed": "行动者在对它做什么，选一个：'
                f'{deed_options(*extra_deed, "operate", "destroy")}",'
            )
        else:
            deed_field = ""  # no identifiable target → nothing to choose; code falls back to exert

        # The shape-dependent fields are constants per target shape, so calls of the same shape
        # still share a cacheable system prefix.
        system = f"""\
【裁决任务】
你是中立的世界裁决者。根据下方材料客观裁定一次物理行动的结果：是否成功、造成了什么后果。

【裁决原则】
- 不得因行动者身份强势或性格果决就一律判成功；双方实力、体力、现场条件都必须计入。
- 在场旁人是裁决证据：结合各自的背景/性情，权衡他们对此事的立场（会阻拦、会协助、还是袖手旁观），
  据此影响成败与代价——但有人在场不机械地等于失败。
- 在场者标注的「此刻意图」是他们此刻**同时**在试图做的事、结果**尚未落定**（引号内是各人自己的说法）：
  据此权衡他们会不会以及如何影响这次行动（介入、协助、妨碍、分神、自顾不暇等），从而左右成败与代价——
  但**不要**把某人尚未落定的意图当成已经发生的事实去叙述（「意图介入」不等于「已经发生」）。
- 对人的物理行动**未必是攻击**(也可能是搀扶/救治/递物/制止等有益或中性之举,见意图)：
  没有耗损的行动 target_damage 应为 0，切勿见「物理行动」就判出损耗。
- 损耗数值必须与 fact 的描述一致：fact 里确实耗去了对方什么才给相应 damage，未遂/没落到实处则不应有损耗。**耗在哪里都算**：负伤、力竭、饥渴、寒冷皆可，不要只认伤——一场把人拖得脱力的纠缠同样是损耗。{relief_rules}
- 不管是对于自己还是目标，损耗应该分致命和非致命，如果是致命损耗值会高些，如果是非致命，比如体力消耗等等，这些损耗值可以很小，比如单纯的体力消耗可以建议小于等于0.099。
- 输出处境必须与生命力变化要符合认知常识，比如说如果说某个人的处境是“已经死亡”，但是生命力却还是满的，这种属于不符合认知常识。
- 对人行动即便成功也未必零代价（对方反抗、自身消耗）。
- 除非体力状态明确表示无生命力，否则不可以直接捏造死亡事实。
- 对于行动动作是否成功和行动结果的客观事实一定要分开判断，动作成功并不代表客观事实一定成立，比如行动者想看清某物是不是一封信，「看」这个动作本身可能成功了，但那物到底是不是信，得结合现场情况综合判定。
- 物件上写着、记着什么，你并不知道，材料里也没有给出：行动者要读、要看其中写了什么时，东西不在他手上就读不到；
  不论成败，都**不得**替他写出读到了什么，也不得从物件的描述里推想出它的内容。
- 如果【目标】与【行动】意图存在明显的冲突，可以先思考冲突的原因，如果没有得到解释理由，可以直接判定行动失败。
- 若行动者要把手中之物交到某人手上，对方接不接是他自己的事：可能接过、可能推拒、可能被旁人截下。据双方关系、性情与现场权衡，不得默认必然交到。
- 若他要拿的东西标着「由某人持有」，那就是从那人手上、身上拿，不是捡起一件无主之物：那人此刻醒着还是睡着、留不留神、护不护得住，连同双方实力与旁人立场，都要计入成败与代价。东西此刻在谁手上以「由某人持有」为准，物件描述里写的所在可能只是它原先的位置。{possession_rule}
{SAME_PLACE_VERDICT_RULE}
{IDLE_BYSTANDER_VERDICT_RULE}
{CLOSED_WORLD_FACT_RULE}

【输出】
{VITALITY_DAMAGE_DEFINITION}{relief_definition}
严格输出以下 JSON，不要任何多余内容。**先在 reason 里简述裁断依据**(权衡了双方哪些实力/伤势/现场条件、为何如此判),再据此给出结论字段——先想清楚再下判，不要凭空给数字。
- outcome：**第三人称、点名行动者与目标**、客观描述实际发生与结果（发生了什么？结果是什么？旁观者会看到的，内容具体清晰，简洁明了，不超过40字）。
- fact：**第一人称（行动者「我」视角）**一句话描述客观描述实际发生与结果（发生了什么？结果是什么？行动者自己会记住的，内容具体清晰，简洁明了，不超过60字）。
- outcome 与 fact 都**只写「谁做了什么、结果如何」，不要带时间或地点前缀**（如「在书房，…」）——地点由系统统一标注，你再写一遍会重复。
- damage 字段的值一定要于fact, outcome自洽合理，比如如果fact, outcome中说目标已经没有生命力了，那target_damage就应该是1。
- deed：行动者**在做的动作本身**，与成败无关——就算没做成，他做的仍是那个动作。
- why：**仅当 success 为 false**时给出没成的缘由，第三人称、≤20字、只写外部可陈述的（如对方早有防备、那东西比看上去沉得多），不写行动者的心绪或懊丧；success 为 true 时给空字符串。
{target_condition_doc}{actor_condition_doc}{{"reason": "一两句裁断依据（≤100字）",{deed_field} "success": true或false, "outcome": "第三人称、点名行动者与目标、客观描述实际发生（≤40字）", "fact": "第一人称、行动者「我」视角描述客观描述实际发生与结果（≤60字）", "why": "没成的缘由（≤20字）；成了给空字符串",{relation_field}{person_fields}{entity_fields} "actor_damage": 行动者自身的损耗，按上方 damage 刻度{target_condition_fields}{actor_condition_fields}}}
"""
        user = f"""\
【行动者】（裁决输入：性格与状态影响行动水准，但你不站在他的立场）
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}
{actor_vitality_line}{actor_condition_line}

【目标】
{target_block}

【现场】
{scene if scene else "（现场情况不明）"}

【行动】
- 意图：{description}{expected_part}{recipient_line}

依上面说定的 JSON 格式给出裁决（reason 在前），只输出 JSON、不写任何多余内容。"""
        # Same source as the user message above: every fact this adjudication can see.
        facts = (
            GivenFacts()
            .add("行动者体力", actor_vitality_line)
            .add("行动者处境", actor_condition_line)
            .add("目标", target_block)
            .add("现场", scene or "情况不明")
            .add("意图", description)
            .add("期望结果", expected_outcome)
            .add("受事", recipient_line.partition("：")[2])
        )
        try:
            with annotate_call(given_facts=facts):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    # Estimate each target shape and size max_tokens by the larger:
                    # common: reason (≤100 chars ≈150) + deed (enum ~8) + success (~3)
                    #   + outcome (40 chars ≈60) + fact (60 chars ≈90) + why (≤20 chars ≈30)
                    #   + actor_damage (~3) + keys (~45) ≈389 tok
                    # actor condition group (always): actor_condition (≤15 chars ≈23) + steps (~3)
                    #   + frees_actor (~3) + keys (~20) ≈438 tok
                    # person shape: + relation (~8) + target_damage (~3) + target_relief (~3)
                    #   + target_condition (≤15 chars ≈23) + steps (~3) + frees_target (~3)
                    #   + keys (~25) ≈511 tok
                    # thing shape: + relation (~8, only when the act lands on someone's
                    #   possession) + new_entity_state (~8) ≈454 tok
                    # The person shape is the largest.
                    temperature=0.7,
                    max_tokens=output_budget(511),
                    json_mode=True,
                )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "physical_judge_llm_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            return None

        relation_dir = parse_relation_direction(str(data.get("relation", "neutral"))).value
        actor_name = self._directory.agent_name(agent.agent_id)
        fact = str(data.get("fact", "")).strip() or description
        new_entity_state = str(data.get("new_entity_state", "")) if has_entity else ""
        succeeded = coerce_bool(data.get("success"), True)
        return _Verdict(
            success=succeeded,
            # The judge sometimes fills why on success too. Unsliced: the prompt caps it.
            failure_reason=(str(data.get("why", "")).strip() if not succeeded else ""),
            # Third person, never first. description is first-person wording, so quote it in 「」;
            # raw it renders as "常何我不再挣扎…".
            outcome=str(data.get("outcome", "")).strip() or f"{actor_name}着手做「{description}」",
            fact=fact,
            relation_dir=relation_dir,
            actor_damage=coerce_float(data.get("actor_damage", 0.0), default=0.0, minimum=0.0, maximum=1.0),
            target_damage=(
                coerce_float(data.get("target_damage", 0.0), default=0.0, minimum=0.0, maximum=1.0)
                if is_person else 0.0
            ),
            target_relief=(
                coerce_float(data.get("target_relief", 0.0), default=0.0, minimum=0.0, maximum=VITALITY_RELIEF_CAP)
                if is_person else 0.0
            ),
            new_entity_state=new_entity_state,
            # Target conditions only on verdicts on people, so "the door was broken open" isn't
            # taken as someone's condition. Unsliced: the prompt caps it.
            target_condition_desc=(
                coerce_str(data.get("target_condition", "")).strip() if is_person else ""
            ),
            target_condition_steps=(
                coerce_int(data.get("target_condition_steps", 0), default=0, minimum=0)
                if is_person else 0
            ),
            frees_target=(coerce_bool(data.get("frees_target"), False) if is_person else False),
            actor_condition_desc=coerce_str(data.get("actor_condition", "")).strip(),
            actor_condition_steps=coerce_int(
                data.get("actor_condition_steps", 0), default=0, minimum=0
            ),
            frees_actor=coerce_bool(data.get("frees_actor"), False),
            deed=self._resolve_deed(
                data.get("deed", ""),
                is_person=is_person,
                has_entity=has_entity,
                seize_available=seize_available,
                actor_holds=actor_holds,
                target_damage=coerce_float(data.get("target_damage", 0.0), default=0.0, minimum=0.0, maximum=1.0),
                new_entity_state=new_entity_state,
            ),
        )

    @staticmethod
    def _resolve_deed(
        label: object, *, is_person: bool, has_entity: bool, seize_available: bool,
        actor_holds: bool, target_damage: float, new_entity_state: str = "",
    ) -> Deed:
        """The judge's deed label, reconciled with the facts it also reported:
        - Reported damage makes it a STRIKE.
        - A SEIZE or RELINQUISH the world can't honour becomes OPERATE.
        - A ``new_entity_state`` saying the thing is gone makes it DESTROY.
        Defaults are conservative (restrain, operate): an unparseable label must not invent
        violence.

        Never promote a deed to RELINQUISH because a recipient was named: that's intent. If he
        smashed the thing instead, DESTROY is the truth and the recipient gets nothing.
        """
        if is_person:
            deed = parse_deed(label, default=Deed.RESTRAIN)
            if target_damage > 0.0:
                return Deed.STRIKE
            return deed if deed in (Deed.STRIKE, Deed.RESTRAIN) else Deed.RESTRAIN
        if has_entity:
            deed = parse_deed(label, default=Deed.OPERATE)
            # This guard promotes to a harsher deed, against the conservative-default rule.
            # Accepted: a thing described as gone but left in the world corrupts every later
            # scene it appears in.
            if deed != Deed.DESTROY and _GONE_STATE.search(new_entity_state):
                logger.warning(
                    "deed_destroy_inferred_from_state",
                    extra={"declared_deed": str(label), "new_entity_state": new_entity_state},
                )
                return Deed.DESTROY
            if deed == Deed.SEIZE and not seize_available:
                return Deed.OPERATE
            if deed == Deed.RELINQUISH and not actor_holds:
                return Deed.OPERATE
            return (
                deed
                if deed in (Deed.SEIZE, Deed.RELINQUISH, Deed.OPERATE, Deed.DESTROY)
                else Deed.OPERATE
            )
        return Deed.EXERT

    def _adjudication_failed_result(
        self, target: "ActionTarget | None", expected_outcome: str, step: int,
        agent_id: str, description: str, location: str,
    ) -> ActionResult:
        """Null step when adjudication can't happen: no memory, no effects of any kind."""
        # Never perceived; only a trace/snapshot marker.
        outcome = scene_line(location, f"{self._directory.agent_name(agent_id)}着手「{description}」，一时未能确知结果如何。")
        stub = AgentAction(
            agent_id=agent_id,
            step=step,
            action_type=ActionType.PHYSICAL,
            action_description=description,
            target=target,
            estimated_steps=1,
        )
        return ActionResult(
            action=stub,
            expected_outcome=expected_outcome,
            outcome=outcome,
            succeeded=False,
            adjudication_failed=True,
        )

    async def _llm_target_reaction(
        self,
        *,
        target_agent: "Agent",
        actor_id: str,
        actor_name: str,
        event_line: str,
        succeeded: bool,
        target_damage: float,
        target_relief: float,
        step: int,
    ) -> TargetAgentEffect:
        """B's subjective experience (in-character): emotion, relation direction, memory. Damage
        is the judge's input, never decided here.

        ``event_line`` is the complete sentence of what B went through and must never carry the
        actor's first-person wording: a foreign 「我」 in a 「你」 frame is read as B himself.
        """
        # Neutral wording: PHYSICAL on a person isn't necessarily an attack.
        rel = await target_agent.relation_system.perceive_existing(
            actor_id, emotion=target_agent.personality.state.emotion,
        )
        relation_block = format_relation_block(
            labels=rel.labels if rel else [],
            trust=rel.trust if rel else NEUTRAL_TRUST,
            affection=rel.affection if rel else NEUTRAL_AFFECTION,
        )
        # Positive = net loss, negative = net relief; Agent.apply_target_effect splits by sign.
        net_vitality_damage = target_damage - target_relief
        # Net, because the line reports where his body ended up; omitted at exactly zero so a
        # helping hand isn't "分毫未损". Neutral frame: event_line tells the cause.
        if net_vitality_damage > 0:
            harm_line = f"\n- 你身上的变化：{_damage_label(net_vitality_damage)}。"
        elif net_vitality_damage < 0:
            harm_line = f"\n- 你身上的变化：{_relief_label(-net_vitality_damage)}。"
        else:
            harm_line = ""
        # His condition before this act; without it a bound man might remember dodging.
        target_condition_line = condition_line(
            target_agent.personality.state.condition, voice=SituationVoice.SECOND,
            now_step=step, seconds_per_step=self._seconds_per_step,
        )
        system = f"""\
此刻你完全代入这个角色,以第二人称「你」说出你经历了什么、此刻的情绪,以及这件事让你对对方的观感变化——这是你一贯的记事方式,与处境无关。

【你要说的】
从你的视角说出你经历了什么、此刻的情绪，以及这件事让你对对方的观感变化。

【注意】
- 始终以你自己的视角表达。
- 你的情绪强度要与这件事对你的轻重相称。
- fact 里提到对方时，不妨带上你认识的名字便于日后回想（惯以称谓相称的话，可附成「称谓（名字）」）；这只是建议，不必生硬套用。

【约束】
- 禁止捏造不存在的客观事实。比如臆想从未发生过的事。

【输出】
情绪字段刻度：
{emotion_legend()}
严格输出以下 JSON，不要任何多余内容：
{{"fact": "一句话描述你经历了什么（≤30字）", "emotion_type": "{EmotionType.prompt_list()}中选一", "emotion_intensity": 按上方 intensity 刻度, "emotion_valence": 按上方 valence 刻度, "relation": "positive或negative或neutral（你对对方的观感变化）"}}
"""
        user = f"""\
【你是谁】
{target_agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{target_condition_line}

【刚刚发生在你身上的事】
- {event_line}{harm_line}
- 你与{actor_name}的关系：{relation_block}

依上面说定的 JSON 格式说出你的经历与反应，只输出 JSON、不写任何多余内容。"""
        # Only the certain part: event_line's viewpoint doesn't fit first-person memory.
        fallback_fact = f"{actor_name}对我做了一件事。"
        try:
            # Attribute the trace and memory to B, and mark his step as passive for the audit.
            with observe_stage(Stage.ACTION, agent_id=target_agent.agent_id), annotate_call(
                given_facts=GivenFacts()  # same source as the user message above
                .add("我的处境", target_condition_line)
                .add("刚刚发生在我身上的事", f"{event_line}{harm_line}")
                .add(f"我与{actor_name}的关系", relation_block),
                action_owner=target_agent.personality.soul.name,
                acted_upon=event_line,
            ):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    # fact (≤30 chars ≈45) + emotion_type (~5) + intensity/valence/relation (3×3=9)
                    # + 5 keys (~25) ≈84 tok.
                    max_tokens=output_budget(84),
                json_mode=True,
                )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "physical_reaction_llm_failed",
                extra={"agent_id": target_agent.agent_id, "error": str(exc)},
            )
            # No memory (Rule 1 tier-1): an empty line would only dilute recall. Harm still
            # lands: it came from the judge.
            return TargetAgentEffect(
                agent_id=target_agent.agent_id,
                factual_memory="",
                vitality_damage=net_vitality_damage,
            )

        relation_dir = parse_relation_direction(str(data.get("relation", "neutral"))).value
        delta = _relation_delta(relation_dir, succeeded)
        emotion_type = parse_emotion_type(str(data.get("emotion_type", "neutral")))
        return TargetAgentEffect(
            agent_id=target_agent.agent_id,
            factual_memory=str(data.get("fact", "")).strip() or fallback_fact,
            emotion_type=emotion_type.value,
            emotion_intensity=coerce_float(data.get("emotion_intensity", 0.3), default=0.0, minimum=0.0, maximum=1.0),
            emotion_valence=coerce_float(data.get("emotion_valence", 0.0), default=0.0, minimum=-1.0, maximum=1.0),
            relation_toward_actor=(actor_id, *delta) if delta is not None else None,
            vitality_damage=net_vitality_damage,
        )


