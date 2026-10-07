"""Decision engine for autonomous actions.

* ``_ACTION_SPACE`` is a static catalogue of action types. Entries say what a type means;
  targets aren't enumerated there because the prompt's situation section already lists them.
* ``_llm_select`` (one path for every agent) hands that catalogue plus the full in-character
  context to the LLM, which decides which option fits, why, and which target to bind.
* ``_parse_llm_selection`` only validates the bound target structurally (TALK targets present,
  MOVE targets reachable) and rejects illegal selections (the step ends ``DecisionStatus.FAILED``).
  Rule code stays structural; behaviour comes from internal state, not weighted heuristics.

The decision prompt's inputs play three roles:
* Direction  = need + short_term_goals: what my action moves toward;
* Lens       = persona (temperament / values / self-image) + current emotion: how I think,
               weigh and react;
* Reality    = hard constraints (bindable options + bottom lines) + soft references
               (relations / memories / signals): the only ground I can act on and judge from.
Through the lens, toward the direction, within reality, pick one concrete feasible action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Sequence

from agent.memory_types import Memory
from agent.need import NeedEvaluation
from agent.perception import PerceptionPacket
from agent.relation import describe_relations, render_relation_lines
from agent.personality import ActionStatus, PersonalityLayer
from core.context import (
    GivenFacts,
    annotate_active_call,
    annotate_call,
    note_active_call_adoption,
)
from core.interfaces.action import (
    KIND_ITEM,
    KIND_OBJECT,
    ActionTarget, ActionType, AgentAction, ErrandOrder, Ref,
)
from core.duration import describe_duration, describe_seconds
from core.interfaces.llm import IndexedRef, LLMMessage, LLMRouter, LLMScene, extract_json_object, output_budget
from core.interfaces.urgency import Urgency, parse_urgency
from core.logging import get_logger
from core.interfaces.perception import PerceivedIdentity, Situation, SpatialPerception, VisibleEntity
from core.prompts import (
    SIGNAL_CAPS,
    render_npc,
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    DECEASED_MARK,
    condition_line,
    MEMORY_ORDER_HINT,
    URGENCY_SCALE_DESCRIPTION,
    SituationVoice,
    order_memories_chrono,
    person_referent,
    render_entity,
    render_memory,
    render_memory_lines,
    render_situation_header,
    situation_location,
    urgency_label,
    vitality_line,
)

logger = get_logger(__name__)


@dataclass
class ActionIntent:
    """
    Structured intent produced at plan time and consumed by the ActionExecutor.
    Separates *what* the agent wants to achieve (plan) from *how* it unfolds (execution).
    """

    purpose: str                         # natural-language goal description
    estimated_steps: int = 1

    # Movement (MOVE)
    destination: str | None = None

    # Errand (ERRAND): a core type, so the payload stays strongly typed through
    # ``AgentAction.intent`` (an ``Any``, since core can't import this module).
    errand: "ErrandOrder | None" = None


@dataclass
class ActionCandidate:
    """One entry in the agent's action space.

    ``description`` says what the type means in the abstract and doesn't enumerate targets (the
    prompt's situation section shows them). The LLM picks the type by index and binds targets
    through the index slots in its response (see ``_TARGET_BINDERS``).
    """

    action_type: ActionType
    description: str


# Static action catalogue.  Behaviour comes from the LLM's choice of type and
# target, not from filtering this list at runtime.
_ACTION_SPACE: List[ActionCandidate] = [
    ActionCandidate(ActionType.TALK,         "与当前场景中某个可见角色面对面同步交谈（1v1，双方共同参与多步完成，生成完整对话，影响双边关系；必须在同一地点），但是交谈内容有可能会被同一地点的人听到。"),
    ActionCandidate(ActionType.SEND_MESSAGE, "向特定角色发送异步消息（对方下一步收到），或向场景内所有人广播通知（当步生效）；适合单边传达任意意图，收件人不在场也能送达"),
    ActionCandidate(ActionType.MOVE,         "移动到一个可前往的地点；若这一动还要把在场的某人强行一并带走（不问他愿不愿意），另填 move_carry_indices"),
    ActionCandidate(ActionType.WORK,         "专注完成某一件自身的事情（写作、练习、制作、经营、整理等）"),
    ActionCandidate(ActionType.PHYSICAL,     "对特定目标施加直接的、即时的物理干预或身体动作（攻击、推倒、搀扶、保护、制止、夺取、使用、拿起、放下、破坏等，可有益可有害）；对人动手填 physical_person_index，对物动手填 physical_entity_index，对此处听人吩咐做事的动手另有专属的一栏（见输出格式），三者必居其一；若是把手里的东西交给在场某人，另填 physical_recipient_index"),
    # State only what it can do and its hard constraints. Don't say it may fail, and don't say it
    # "changes nothing". That is a code-layer fact (the executor has no world write access), and
    # to the character it reads as "this option is useless", so models almost stop picking it.
    ActionCandidate(ActionType.COVERT,       "藏在暗处观察此处的某人某事，探听我本来无从得知的情形和情况——听清他们谈的是什么、看清他们做的是什么、看清此处发生了什么事、读到写着的是什么；须与目标同在一地"),
    ActionCandidate(ActionType.REST,         "暂时搁置外部事务进行休息（生理疲劳、情绪耗尽、心情低落、生命力亏损均可）；需要多个步骤，休息得越久，恢复的生命力越多"),
    ActionCandidate(ActionType.ERRAND,       "派一个听人吩咐做事的替我跑一趟别处：先定他去哪，再看要不要让他带上东西、指名一个人、**替我把一句话原样说出去**（指了人就说给那人听，没指人就在那处当众说）——填了哪几样就办哪几样。他没有心智：只会走到、交到、把那句话一字不改地说出口、把**眼睛看得见的**（谁在场、有什么东西、刚出了什么事）带回来；他不会打听、不会问话、不会看人脸色，更不会替我劝人、引开人、盯梢。我这一拍交代完就自由，不必等他回来"),
]

# Fixed expectation for SEND_MESSAGE: an async action can only confirm "the message went out"
# this step. Pinned, outcome (delivered) matches expected (delivery), so downstream (emotion gap,
# memory) needs no async special case; the real effect arrives later via replies + goal progress.
_SEND_MESSAGE_EXPECTED_OUTCOME = "把想传达的话送到对方面前"


def _entity_owner_name(entity: VisibleEntity, *, self_id: str, spatial: SpatialPerception) -> str | None:
    """Narrative referent for the holder: "我" for myself, else the perceived name, "某人" if
    unknown (never an id); None for things on the ground.

    Don't query the directory: names in an agent's prompt may only come from perception (a holder
    is co-located, so their name is already in visible_agents).
    """
    holder = entity.holder_id
    if not holder:
        return None
    if holder == self_id:
        return "我"
    presence = spatial.visible_agents.get(holder)
    return ((presence.identity.name if presence is not None else "") or "某人")


def _reject(reason: str) -> None:
    """Discard the LLM's selection, recording why on the call's trace.

    A rejection burns the beat (tier-1: a frozen beat beats a fabricated one), yet the response is
    HTTP-200 valid JSON, so ``ok``/``parse_ok`` read clean; without this the reason is invisible.
    """
    note_active_call_adoption(False, reason)
    return None


@dataclass
class LLMSelection:
    """Result of LLM choosing one ``ActionCandidate`` from the action space."""

    action_type: ActionType
    target: ActionTarget
    action_description: str
    inner_monologue: str
    estimated_steps: int = 1
    urgency: Urgency = Urgency.NORMAL
    llm_model: str = "llm"
    expected_outcome: str = ""
    # SEND_MESSAGE only: the verbatim words for the recipients / everyone present (narrative
    # layer, delivered and read by recipients). A different voice from action_description (my own
    # memory of the act). Empty for other actions.
    message_content: str = ""
    # ERRAND only: who runs it, where to, and what to do there. None for other actions.
    errand: "ErrandOrder | None" = None


def _text(payload: dict, key: str) -> str:
    """Conservative read of a string slot: anything that isn't a string reads as empty."""
    raw = payload.get(key)
    return raw.strip() if isinstance(raw, str) else ""


@dataclass(frozen=True)
class _Slots:
    """Raw target-related slots plus this step's candidate lists.

    Each list is a namespace: which list an index resolves against is decided by the field
    name, never by a category the LLM reports. ``packet`` absent = a test parsing directly;
    then we bind without validating.
    """

    payload: dict
    packet: "PerceptionPacket | None"
    visible_ids: List[str]      # present people: allow-list for TALK / PHYSICAL on a person / carry
    roster_ids: List[str]       # reachable people (incl. absent): list for the person_indices family
    reachable_ids: List[str]    # adjacent locations: destination_index / errand_destination_index
    entity_ids: List[str]       # things within reach: physical_entity_index
    entity_types: dict[str, str]
    npc_ids: List[str]          # people here with no mind of their own: errand_npc_index / physical_npc_index
    # What I'm holding: a membership set, not an indexed list (see ``_own_item_ids``).
    own_item_ids: List[str]
    # Filled indices that fell outside their list. IndexedRef.resolve drops them silently, so
    # record them here or they are lost. Entries name only slot and number, so audits don't
    # depend on slot names or prompt layout.
    dropped: List[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return self.packet is not None

    def filled(self, key: str) -> bool:
        """The LLM filled this index field. Channel choice depends only on whether it was filled,
        not whether it resolves: a filled but out-of-range index is a hallucination and must
        fail in its own channel, not slide into the next channel and match something there."""
        raw = self.payload.get(key)
        return isinstance(raw, int) and not isinstance(raw, bool)

    def flag(self, key: str) -> bool:
        """Declarative boolean slot (bound to no list, so no index). Only a real true counts:
        missing / null / string all mean undeclared — the point is to separate an explicit
        declaration from an empty answer."""
        return self.payload.get(key) is True

    def indices(self, key: str) -> list:
        """Tolerate a scalar int where an array was expected (wrap it); anything else invalid → empty."""
        raw = self.payload.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool):
            return [raw]
        return raw if isinstance(raw, list) else []

    def one(self, key: str, ids: Sequence[str]) -> str | None:
        if not self.grounded or not self.filled(key):
            return None
        raw = self.payload.get(key)
        resolved = IndexedRef(ids).resolve([raw])
        if not resolved:
            self.dropped.append(f"{key} 填了 #{raw}，而这份名单只有 {len(ids)} 项")
        return resolved[0] if resolved else None

    def many(self, key: str, ids: Sequence[str]) -> List[str]:
        """Resolve several indices, dropping duplicates and out-of-range ones."""
        if not self.grounded:
            return []
        raw = self.indices(key)
        resolved = IndexedRef(ids).resolve(raw)
        if len(resolved) < len({str(v) for v in raw}):
            self.dropped.append(
                f"{key} 填了 {raw}，只有 {len(resolved)} 个落在这份名单（共 {len(ids)} 项）内")
        return resolved


@dataclass(frozen=True)
class _Bound:
    """Binding result: the target plus any payload unique to the action type.

    ``description_suffix`` is used only by TALK (appends what will be said); ``errand`` only by
    ERRAND — the errand's content (where to, what to do there) doesn't fit in target, since it
    isn't a "who does this point at" relation.
    """

    target: ActionTarget
    description_suffix: str = ""
    errand: "ErrandOrder | None" = None


def _bind_send(slots: _Slots) -> "_Bound | None":
    """Two delivery channels: ``person_indices`` = directed (all recipients go into acts_on; a
    group send is one act); ``message_announce`` = announce to everyone here (empty acts_on).

    Don't treat an empty acts_on as a broadcast: "address everyone" and "the person I want isn't
    on the list" both look empty, and the lenient reading treats a beat that never touched the
    world as carried out. So reject when neither channel is used, or when no index resolves.
    If both are given, directed wins (declaration order is priority, as in ``_bind_physical``).
    """
    recipients = slots.many("person_indices", slots.roster_ids)
    if slots.grounded and slots.indices("person_indices") and not recipients:
        return _reject("send_recipients_unresolvable")
    if slots.grounded and not recipients and not slots.flag("message_announce"):
        return _reject("send_without_channel")
    return _Bound(ActionTarget(acts_on=[Ref.agent(aid) for aid in recipients]))


def _bind_talk(slots: _Slots) -> "_Bound | None":
    """Face-to-face 1v1. The first person is the counterpart (acts_on + claims: the talk spends
    their turn); the rest are listeners (reaches only, so they stay free this step).

    No counterpart, or an absent one, rejects the step: letting it through would embed a parser
    diagnostic ("wanted to talk but didn't know to whom") into narrative memory.

    Models often put the real words in ``message_content`` (a SEND field); TALK has no delivery
    channel, so fold them into the description here, in code, or they are lost.
    """
    people = slots.many("person_indices", slots.roster_ids)
    if slots.grounded:
        if not people:
            return _reject("talk_without_target")
        if people[0] not in slots.visible_ids:
            return _reject("talk_target_not_present")
    target = (
        ActionTarget(
            acts_on=[Ref.agent(people[0])],
            claims=[Ref.agent(people[0])],
            reaches=[Ref.agent(aid) for aid in people[1:]],
        )
        if people else ActionTarget()
    )
    spoken = _text(slots.payload, "message_content")
    return _Bound(target, f"（我打算说：{spoken}）" if spoken else "")


def _bind_covert(slots: _Slots) -> "_Bound | None":
    """A covert act's object (person, place or thing) lives only in action_description; no
    structured target. Binding would need a channel per kind, or a classification question for the
    model (forbidden by CLAUDE.md §6).

    Don't add channels because some downstream can't read a target: whether COVERT has one follows
    from what it is, not from who consumes it.
    """
    return _Bound(ActionTarget())


def _bind_physical(slots: _Slots) -> "_Bound | None":
    """Four mutually exclusive channels: a person with cognition, a person without, a thing, or
    something abstract off the lists. Kind comes from the channel, never from the LLM.

    The no-cognition person has its own channel because nothing downstream fits it: merged,
    adjudication would write emotion/relation/memory (``TargetAgentEffect``) against an id with no
    ``Agent`` and silently fail into a null step.

    Never fall back across namespaces: retrying an out-of-range person index against the entity
    list turns "pin someone down" into "grab something" while the description still names a person.

    The recipient is a modifier on the entity channel, not a fourth channel; unresolved, it rejects
    the step, since the recipient is the substance of a hand-over. A PHYSICAL with no target is a
    broken decision, not a weaker act.
    """
    acted_on: "Ref | None" = None
    recipient_id: str | None = None
    if slots.filled("physical_person_index"):
        person_id = slots.one("physical_person_index", slots.roster_ids)
        if person_id is not None and person_id in slots.visible_ids:   # a person must be present
            acted_on = Ref.agent(person_id)
    elif slots.filled("physical_npc_index"):
        npc_id = slots.one("physical_npc_index", slots.npc_ids)
        if npc_id is not None:
            acted_on = Ref.npc(npc_id)
    elif slots.filled("physical_entity_index"):
        entity_id = slots.one("physical_entity_index", slots.entity_ids)
        if entity_id is not None:
            acted_on = Ref.entity(entity_id, slots.entity_types.get(entity_id) or KIND_ITEM)
        if slots.filled("physical_recipient_index"):
            recipient_id = slots.one("physical_recipient_index", slots.roster_ids)
            if recipient_id is None or recipient_id not in slots.visible_ids:
                return _reject("physical_recipient_unresolvable")
    else:
        abstract = _text(slots.payload, "item_id")
        if abstract:
            acted_on = Ref.entity(abstract, KIND_OBJECT)
    if slots.filled("physical_recipient_index") and recipient_id is None:
        # A recipient without the entity channel: a hand-over with nothing to hand over.
        return _reject("physical_recipient_without_entity")
    if slots.grounded and acted_on is None:
        return _reject("physical_without_target")
    # The recipient goes into reaches, not acts_on: the act is on the thing (it changes hands)
    # and he is only affected. Not claims either — handing something over doesn't spend his turn.
    return _Bound(ActionTarget(
        acts_on=[acted_on] if acted_on else [],
        reaches=[Ref.agent(recipient_id)] if recipient_id else [],
    ))


def _bind_move(slots: _Slots) -> "_Bound | None":
    """Acts on the destination. Carried people only have their turn taken (COMPEL); they aren't
    the act's object — acts_on and claims must not be mixed.

    Only present people can be carried (decide and the executor each check). Out-of-range carry
    indices are dropped one by one: carrying is an add-on to a complete move, deliberately unlike
    the hand-over recipient ("unresolvable → reject"). No destination rejects the step.
    """
    destination = slots.one("destination_index", slots.reachable_ids)
    if slots.grounded and destination is None:
        return _reject("move_without_destination")
    carried = [aid for aid in slots.many("move_carry_indices", slots.roster_ids)
               if aid in slots.visible_ids]
    return _Bound(ActionTarget(
        acts_on=[Ref.place(destination)] if destination else [],
        claims=[Ref.agent(aid) for aid in carried],
    ))


def _bind_errand(slots: _Slots) -> "_Bound | None":
    """Acts on the runner: the act is "I give him an errand", so he is the object.

    Never put him in ``claims``: arbitration looks claims ids up in the agents table and a miss
    rejects the whole action; he isn't there (he has no turn to spend).

    The other four slots are orthogonal axes (where to / carry what / find whom / say what), not an
    action table; "deliver", "drop off", "pass a message" are combinations (see ``ErrandOrder``).
    Destination alone is legal ("go and have a look").

    - The item must already be in my hands. It indexes the printed entity list (as
      ``physical_entity_index`` does); "in my hands" is a check, not a namespace. Picking something
      someone else holds rejects the step; never quietly drop the item (his memory already says
      "I had it sent over").
    - "Find whom" is an addressing axis shared by item and message: filled = directed, empty = it
      lands at the place. A message with no recipient is therefore not an error.
    - Recipients come from the reachable roster, absent people included; whether he's there is
      settled at execution (a "not there" report is itself narrative).
    """
    npc_id = slots.one("errand_npc_index", slots.npc_ids)
    if slots.grounded and npc_id is None:
        return _reject("errand_without_bearer")
    destination = slots.one("errand_destination_index", slots.reachable_ids)
    if slots.grounded and destination is None:
        return _reject("errand_without_destination")
    item_id = slots.one("errand_item_index", slots.entity_ids)
    if slots.grounded and slots.filled("errand_item_index") and (
        item_id is None or item_id not in slots.own_item_ids
    ):
        return _reject("errand_item_not_in_hand")
    recipient_id = slots.one("errand_recipient_index", slots.roster_ids)
    message = _text(slots.payload, "errand_message")
    # Words need a listener: a recipient hears them alone, errand_announce says them aloud. With
    # neither, they are my instructions to the runner and fold into action_description (as TALK
    # folds message_content); the errand still runs, nobody recites them.
    # Don't read an empty recipient as "announce": the sender can't see who's at the destination,
    # so an empty slot isn't evidence of "shout it to strangers". Gated only here, so
    # ErrandOrder.message always has a real listener.
    suffix = ""
    if message and not slots.flag("errand_announce") and not recipient_id:
        suffix, message = f"（我吩咐他：{message}）", ""
    order = (
        ErrandOrder(
            npc_id=npc_id, destination_id=destination,
            item_id=item_id or "", recipient_id=recipient_id or "", message=message,
        )
        if npc_id and destination else None
    )
    return _Bound(
        ActionTarget(acts_on=[Ref.npc(npc_id)] if npc_id else []),
        suffix,
        errand=order,
    )


def _bind_nothing(slots: _Slots) -> "_Bound | None":
    """WORK / REST act only on oneself.

    No catch-all: a slot filled under another type (``destination_index``, ``item_id``) isn't
    bound, or the record and the narrative would name different things.
    """
    return _Bound(ActionTarget())


# One binder per action type; it owns all target resolution and structural validation for that
# type, so reading or adding a type touches one place. Default = acts only on oneself.
_TARGET_BINDERS: dict[ActionType, Callable[[_Slots], "_Bound | None"]] = {
    ActionType.SEND_MESSAGE: _bind_send,
    ActionType.TALK: _bind_talk,
    ActionType.COVERT: _bind_covert,
    ActionType.PHYSICAL: _bind_physical,
    ActionType.MOVE: _bind_move,
    ActionType.ERRAND: _bind_errand,
}


class DecisionStatus(str, Enum):
    """Four-valued decision result.

    - ``ACTED``         — acts: carries an executable ``AgentAction``.
    - ``NO_ACTION``     — declines: a successful decision that deliberately doesn't act this
      beat (the LLM returned ``act:false``).
    - ``FAILED``        — decision failed: the decision LLM was unavailable / output unparseable
      (even after retry).
    - ``NOT_SCHEDULED`` — never entered the decision loop this step: the scheduler's cadence gate
      kept it out (see ``AgentScheduler.plan``) and no LLM call ran. Not a cognitive outcome but
      "the engine didn't ask this beat" — the body stays idle and can still be recruited.

    The last three mean no autonomous action and no narrative/state footprint (Rule 1 tier-1);
    they differ only in the code layer (status + log level), which cadence-gate tuning needs to
    tell "nobody was asked" from "asked but declined/failed".
    """

    ACTED = "acted"
    NO_ACTION = "no_action"
    FAILED = "failed"
    NOT_SCHEDULED = "not_scheduled"


@dataclass(frozen=True)
class DecisionResult:
    """Return value of ``decide``: a status plus an action (non-empty only when ACTED)."""

    status: DecisionStatus
    action: AgentAction | None = None

    @classmethod
    def acted(cls, action: AgentAction) -> "DecisionResult":
        return cls(DecisionStatus.ACTED, action)

    @classmethod
    def no_action(cls) -> "DecisionResult":
        return cls(DecisionStatus.NO_ACTION)

    @classmethod
    def failed(cls) -> "DecisionResult":
        return cls(DecisionStatus.FAILED)


class DecisionEngine:
    """Choose a next action from current cognition state."""

    def __init__(
        self,
        llm_router: LLMRouter,
        *,
        seconds_per_step: int = 3600,
        world_start_second_of_day: int = 0,
    ) -> None:
        self._llm_router = llm_router
        self._seconds_per_step = seconds_per_step
        # Time of day at world start (injected by the world layer): "today/yesterday" in memory
        # prefixes counts midnights crossed, which a duration alone can't give.
        self._world_start_second_of_day = world_start_second_of_day

    def _step_duration_hint(self) -> str:
        """Human-readable single-step duration for LLM prompt anchoring."""
        return describe_duration(1, self._seconds_per_step)

    async def decide(
        self,
        *,
        personality: PersonalityLayer,
        packet: PerceptionPacket,
    ) -> DecisionResult:
        """Produce one decision (ACTED / NO_ACTION / FAILED, see ``DecisionStatus``).

        Every agent takes the same path: one in-character prompt + AGENT_DECISION_MAIN + retry.
        """

        return await self._llm_select(personality, packet, _ACTION_SPACE)

    def _build_action(
        self,
        selection: LLMSelection,
        personality: PersonalityLayer,
        packet: PerceptionPacket,
    ) -> AgentAction:
        """Assemble the executable ``AgentAction`` from a parsed selection."""
        # SEND_MESSAGE is async: see _SEND_MESSAGE_EXPECTED_OUTCOME.
        if selection.action_type == ActionType.SEND_MESSAGE:
            expected_outcome = _SEND_MESSAGE_EXPECTED_OUTCOME
        else:
            expected_outcome = (
                selection.expected_outcome
                or self._expected_outcome(selection.action_type, packet.internal_context.need_evaluation)
            )
        intent = ActionIntent(
            purpose=selection.action_description,
            estimated_steps=selection.estimated_steps,
            destination=selection.target.acted_on_place if selection.action_type == ActionType.MOVE else None,
            errand=selection.errand,
        )
        return AgentAction(
            agent_id=personality.soul.agent_id,
            step=packet.spatial.current_step,
            action_type=selection.action_type,
            action_description=selection.action_description,
            target=selection.target,
            inner_monologue=selection.inner_monologue,
            expected_outcome=expected_outcome,
            estimated_steps=selection.estimated_steps,
            urgency=selection.urgency,
            remaining_steps=selection.estimated_steps,
            status=ActionStatus.IN_PROGRESS,
            llm_model=selection.llm_model,
            reason=selection.inner_monologue if selection.inner_monologue else "llm selected",
            # SEND_MESSAGE: the verbatim words to deliver; otherwise mirrors action_description
            # (unread downstream).
            content=(
                selection.message_content
                if selection.action_type == ActionType.SEND_MESSAGE and selection.message_content
                else selection.action_description
            ),
            intent=intent,
        )

    # ------------------------------------------------------------------
    # LLM-driven selection
    # ------------------------------------------------------------------

    async def _llm_select(
        self,
        personality: PersonalityLayer,
        packet: PerceptionPacket,
        candidates: Sequence[ActionCandidate],
    ) -> DecisionResult:
        """Unified decision path: full in-character context + structured choice.

        ``act:false`` → NO_ACTION; LLM error or no parseable selection → FAILED (never a fabricated
        default action); otherwise ACTED.

        Retries once at the semantic level: ``complete_with_retry`` only retries transport
        failures, so an HTTP-200 JSON missing a field would otherwise burn the beat. The retry
        resends unchanged — no relaxed requirements, no fields filled in for the LLM; fabricated
        cognition is far worse than a lost beat. Transport retry stays in the provider.
        """

        system_prompt, user_prompt, given_facts = self._build_decision_prompt(
            personality, packet, candidates)
        result = await self._attempt_selection(
            personality, packet, candidates, system_prompt, user_prompt, given_facts)
        if result.status is DecisionStatus.FAILED:
            logger.info(
                "decision_retry_after_unusable_output",
                extra={
                    "agent_id": personality.soul.agent_id,
                    "step": packet.spatial.current_step,
                },
            )
            result = await self._attempt_selection(
                personality, packet, candidates, system_prompt, user_prompt, given_facts
            )
        return result

    async def _attempt_selection(
        self,
        personality: PersonalityLayer,
        packet: PerceptionPacket,
        candidates: Sequence[ActionCandidate],
        system_prompt: str,
        user_prompt: str,
        given_facts: Sequence[str],
    ) -> DecisionResult:
        """One decision LLM call + parse. ``FAILED`` here is retryable by the caller."""

        agent_id = personality.soul.agent_id
        step = packet.spatial.current_step
        # Code-layer diagnostics (→ LLMCallTrace.extra): record the prompt's #N index ↔ display
        # text mapping on the trace so audits needn't regex the prompt. Each table matches an
        # index contract in the prompt:
        #   action_menu           = 0-based, matches _format_candidate_line; selected_index
        #                           indexes candidates directly.
        #   person_candidates     = 1-based, matches #N in "我能触及的人" (_message_roster order).
        #   destination_candidates= 1-based, matches #N in "我可前往" (reachable_locations order).
        #   item_candidates       = 1-based, matches #N in "我够得着的东西" (visible_entities order;
        #                           short name e.name, plus the holder for held items so audits can
        #                           tell whether it was free to take).
        #   npc_candidates        = 1-based, matches #N in the people-here list (visible_npcs
        #                           order); shared by errand_npc_index and physical_npc_index.
        #   The recipient (physical_recipient_index) uses the person_candidates indices; no
        #   separate map. Errand recipient indices also use person_candidates;
        #   errand_item_index uses a subset of item_candidates (only what I hold), so no map either.
        action_menu = {
            str(i): (
                c.action_type.value.upper()
                if hasattr(c.action_type, "value")
                else str(c.action_type).upper()
            )
            for i, c in enumerate(candidates)
        }
        person_candidates = {
            str(i): who.name for i, (aid, who) in enumerate(_message_roster(packet), 1)
        }
        destination_candidates = {
            str(i): (rl.view.name or "某处")
            for i, rl in enumerate(packet.spatial.reachable_locations, 1)
        }
        npc_candidates = {
            str(i): (packet.spatial.visible_npcs[nid].identity.name or "某人")
            for i, nid in enumerate(packet.spatial.visible_npc_ids, 1)
        }
        item_candidates = {
            str(i): (
                f"{e.name or '某物'}（{owner}）" if (
                    owner := _entity_owner_name(e, self_id=agent_id, spatial=packet.spatial)
                ) else (e.name or "某物")
            )
            for i, e in enumerate(packet.spatial.visible_entities, 1)
        }
        try:
            with annotate_call(
                given_facts=list(given_facts),
                action_menu=action_menu,
                person_candidates=person_candidates,
                destination_candidates=destination_candidates,
                item_candidates=item_candidates,
                npc_candidates=npc_candidates,
                location=situation_location(packet.spatial),
            ):
                response = await self._llm_router.complete_with_retry(
                    LLMScene.AGENT_DECISION_MAIN,
                    [
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    temperature=0.7,
                    # inner_monologue≤60 chars + action_description≤40 + message_content≤60 (SEND)
                    # + expected_outcome≤30 + several index/enum fields; est. ~380 tok. The last three
                    # may each carry a bracketed date (ABSOLUTE_TIME_RULE, ≈8 chars×3 ≈36 tok) → ~416.
                    # Per-type extras are mutually exclusive: PHYSICAL recipient index ≈11 or
                    # item_id≤10 chars ≈20 → ≤~436; MOVE move_carry_indices ≈12 → ~439; SEND
                    # message_announce ≈8 → ~447. ERRAND is costliest: ~416 + errand_message≤60
                    # chars ≈90 + four index fields ≈44 + errand_announce ≈8 → ~558.
                    max_tokens=output_budget(558),
                    json_mode=True,
                )
        except Exception as exc:
            logger.warning(
                "decision_llm_failed",
                extra={"agent_id": agent_id, "step": step, "error": str(exc)},
            )
            return DecisionResult.failed()

        payload = extract_json_object(response.content)
        if not isinstance(payload, dict):
            note_active_call_adoption(False, "no_json_object")
            logger.warning("decision_unparseable", extra={"agent_id": agent_id, "step": step})
            return DecisionResult.failed()

        # Declining (act:false) is a successful decision not to act, kept apart from parse failure
        # at the source: no executor, memory or state change, info log only. Only an explicit
        # false (JSON or the string "false") counts; anything ambiguous is not a decline.
        act = payload.get("act")
        if act is False or (isinstance(act, str) and act.strip().lower() == "false"):
            # Adopted: the output was used and says "do nothing"; marking it discarded would log
            # deliberate restraint as failure.
            note_active_call_adoption(True)
            logger.info("decision_no_action", extra={"agent_id": agent_id, "step": step})
            return DecisionResult.no_action()

        selection = self._parse_llm_selection(payload, candidates, packet)
        if selection is None:  # _parse_llm_selection already recorded the reason on the trace
            logger.warning("decision_unparseable", extra={"agent_id": agent_id, "step": step})
            return DecisionResult.failed()
        note_active_call_adoption(True)
        selection.llm_model = response.model or "llm"
        action = self._build_action(selection, personality, packet)
        target = action.target
        target_ref = ",".join(target.acted_on_ids)
        logger.info(
            "decision_acted",
            extra={
                "agent_id": agent_id,
                "step": step,
                "action_type": action.action_type.value,
                "target": target_ref,
                "description": action.action_description,
                "urgency": action.urgency,
                "estimated_steps": action.estimated_steps,
            },
        )
        return DecisionResult.acted(action)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_decision_prompt(
        self,
        personality: PersonalityLayer,
        packet: PerceptionPacket,
        candidates: Sequence[ActionCandidate],
    ) -> tuple[str, str, List[str]]:
        """(system, user, given_facts).

        given_facts are the facts put in front of the agent, tagged by channel, attached to
        ``LLMCallTrace.extra`` so audits can judge whether the output invented facts. Built here
        alongside the prompt so both share one source: a channel added to the prompt but not
        declared makes the audit flag sourced statements as fabricated. Intents, goals and lists
        aren't facts (the lists are declared via action_menu and friends).
        """
        awareness = packet.internal_context
        spatial = packet.spatial

        lines: List[str] = []
        facts = GivenFacts()
        # Inputs split three ways (see the module docstring): lens × direction × reality.

        # Situation header: the only carrier of when and where, at the very top of user content
        # (§3). The location is the perceived location_view, never a bare location_id; the reality
        # section doesn't repeat it — "我可前往" is relative to this header.
        situation_header = render_situation_header(
            Situation.from_spatial(spatial), voice=SituationVoice.FIRST,
        )
        if situation_header:
            lines.append(situation_header)
            lines.append("")
            facts.add("此刻何时何地", situation_header)

        # 1) 【我是谁】 lens = persona + current emotion (how I think / weigh / react); emotion is
        # the post-perception one (awareness.emotion).
        lines.append("【我是谁】（我怎么想、怎么权衡、怎么反应，都由它决定）")
        lines.append(personality.to_prompt_context(include_goals=False, include_emotion=False))
        lines.append(f"我此刻的心境：{awareness.emotion.summary()}")
        facts.add("我此刻的心境", awareness.emotion.summary())
        lines.append("")

        # 2) 【我要往哪儿】 direction = need + intent + external pressure (what actions move toward)
        lines.append("【我要往哪儿】（我的行动应当朝它迈进，而不是漫无目的）")
        lines.append(f"我最迫切的需求与目标：{awareness.need_evaluation.prompt_context or '（暂无特别迫切的）'}")
        long_term_goals = getattr(awareness.need_evaluation, "long_term_goals", None)
        if long_term_goals:
            # Discrete items as a list (Prompt Design Principle 4), not one run-on line.
            lines.append("我的长期目标（我的未来指引，长期行动方向）：")
            lines.extend(f"  - {g}" for g in long_term_goals[:3])
        # External drivers are things happening outside (not his own intent), so declare them as
        # facts; needs and goals are direction, not facts.
        facts.add("外部驱动", [getattr(eg, "text", "")
                              for eg in getattr(awareness.need_evaluation, "external_goals", [])])
        for eg in getattr(awareness.need_evaluation, "external_goals", []):
            # Neutral prefix; urgency_label ("【留意】"/"【一般】"/"【紧急】"/"【危急】") carries the
            # urgency, so don't presume it's pressing.
            lines.append(f"外部驱动：{urgency_label(eg.urgency)} {eg.text}")
        lines.append("")

        # 3) 【供我参考的】 soft references = relations / memories / insights / ambient signals /
        # messages (inform judgment, don't limit options). Discrete items as lists.
        # Recalled memories get a relative-recency prefix (event kind): durations via
        # seconds_per_step, day boundaries from the clock phase.
        _now = spatial.current_step
        _sps = self._seconds_per_step
        _day_start = self._world_start_second_of_day
        lines.append("【供我参考的】（帮我判断，但不限定我能做什么）")
        lines.append(describe_relations(awareness.relevant_relations))
        # Declare relations with describe_relations' own rendering: a second rendering would drop
        # gender and add float noise, while the point is to declare exactly what it saw.
        facts.add("关系", render_relation_lines(
            awareness.relevant_relations, limit=len(awareness.relevant_relations)))
        _factual = render_memory_lines(
            awareness.factual_memories, now_step=_now, seconds_per_step=_sps, world_start_second_of_day=_day_start)
        _experiential = render_memory_lines(
            awareness.experiential_memories, now_step=_now, seconds_per_step=_sps, world_start_second_of_day=_day_start)
        facts.add("客观经历", _factual).add("主观理解", _experiential)
        lines.append(f"我近来的经历（客观发生；{MEMORY_ORDER_HINT}）：\n{_bullet_block(_factual)}")
        lines.append(f"我近来的经历（我当前的主观理解与感受；{MEMORY_ORDER_HINT}）：\n{_bullet_block(_experiential)}")
        if awareness.recent_foiled_attempts:
            # Foiled attempts are a transient "that beat didn't land" reminder, not memory (the
            # world didn't change). Same MEMORY_ORDER_HINT contract as the memory blocks: the list
            # ascends by step, so the LLM can tell which failed just now.
            facts.add("没能做成的事", awareness.recent_foiled_attempts)
            lines.append(
                f"我近来没能做成的事（尝试做过但是没有做成，世界并未因此改变；对于这类事情，如果重复多次（3次以及3次以上）均失败了，一定要调整策略，不要再重试了。"
                f"同一件事试过不止一次的会注明次数；{MEMORY_ORDER_HINT}）：\n"
                + "\n".join(f"- {t}" for t in awareness.recent_foiled_attempts)
            )
        if awareness.insights:
            facts.add("已形成的判断", render_memory_lines(
                awareness.insights, now_step=_now, seconds_per_step=_sps, world_start_second_of_day=_day_start))
            lines.append(f"我已经形成的判断：\n{_insights_with_sources(awareness.insights, awareness.insight_sources, now_step=_now, seconds_per_step=_sps, world_start_second_of_day=_day_start)}")
        if awareness.period_summaries:
            _periods = render_memory_lines(
                awareness.period_summaries, now_step=_now, seconds_per_step=_sps, world_start_second_of_day=_day_start)
            facts.add("那段时期的印象", _periods)
            lines.append(f"那段时期给我的整体印象：\n{_bullet_block(_periods)}")
        # One block per perception channel (public goings-on / world broadcast / direct message),
        # items listed under one header; an empty channel drops its block.
        if spatial.ambient_events:                  # location-level public events / action outcomes
            lines.append("我注意到（周围的动静）：")
            lines.extend(f"- {ev.content}" for ev in spatial.ambient_events[:SIGNAL_CAPS["ambient"]])
            facts.add("周围的动静", [ev.content for ev in spatial.ambient_events[:SIGNAL_CAPS["ambient"]]])
        if packet.broadcasts:
            lines.append("世界传来的消息：")
            facts.add("世界传来的消息", [b.content for b in packet.broadcasts[:SIGNAL_CAPS["broadcast"]]])
            # Content only: broadcast_type is a code-layer category (its only value, WORLD_EVENT,
            # carries no information) and the header already names the channel; rendering it
            # into first-person text would leak across layers.
            lines.extend(f"- {b.content}" for b in packet.broadcasts[:SIGNAL_CAPS["broadcast"]])
        # The narrator's sender_name is already "不知来源", so it reads as an unknown sender like
        # any other.
        if packet.inbox:
            lines.append("有人捎话给我：")
            lines.extend(
                f"- {msg.sender_name or '某人'}：{msg.content}"
                for msg in packet.inbox[:SIGNAL_CAPS["inbox"]]
            )
            facts.add("有人捎话", [
                f"{msg.sender_name or '某人'}：{msg.content}"
                for msg in packet.inbox[:SIGNAL_CAPS["inbox"]]
            ])
        lines.append("")

        # 4) 【我所处的现实】 hard constraints = bottom lines + bindable options (the only place I
        # can act). #N indices sit near the output (no opaque-id hallucination).
        lines.append("【我所处的现实】（我只能在这里头选择行动类型——没列出的，我此刻够不着）")
        # Condition goes first in this section (§3 strong position): it is the hardest constraint
        # on what I can do right now and reframes any list read before it. Deliberately not in
        # 【我是谁】 — that's persona (how I think and weigh); mixed in, "hands tied behind my
        # back" would read like a personality trait.
        _self_condition = condition_line(
            personality.state.condition,
            voice=SituationVoice.FIRST, now_step=_now, seconds_per_step=_sps, lead="",
        )
        if _self_condition:
            lines.append(_self_condition)
            facts.add("我的处境", _self_condition)
        # Vitality sits with condition: without it, someone near exhaustion keeps acting until
        # vitality hits zero.
        _self_vitality = vitality_line(
            personality.state.vitality, voice=SituationVoice.FIRST,
            lead="", omit_when_full=True,
        )
        if _self_vitality:
            lines.append(_self_vitality)
            facts.add("我的体力", _self_vitality)
        if personality.soul.hard_constraints:
            lines.append("我绝不逾越的底线：")
            for constraint in personality.soul.hard_constraints:
                lines.append(f"  - {constraint}")
        # One person roster (present + reachable absent), numbering shared by person_indices and
        # physical_person_index. Marked "在场" / "不在场": TALK and PHYSICAL-on-a-person may only
        # pick present people; only SEND_MESSAGE may target absent ones. (COVERT doesn't bind
        # from this list — its object lives in action_description.)
        roster = _message_roster(packet)
        if roster:
            lines.append(f"我能触及的人（默认都是有生命力的人，除非明确标识{DECEASED_MARK}）：")
            for idx, (aid, who) in enumerate(roster, 1):
                here = "在场" if aid in spatial.visible_agent_ids else "不在场"
                # Condition shares the parentheses with gender and presence (person_referent's
                # marks). Only present people carry one: for the absent it could only be a belief
                # in memory, not a live field (information asymmetry).
                cond = (p.condition if (p := spatial.visible_agents.get(aid)) else "")
                referent = person_referent(who.name, who.gender, here, cond)
                lines.append(f"  #{idx} {referent}")
                # The roster isn't just the hard frame to pick from; it is itself fact: who exists,
                # who is here, who isn't. Undeclared, the judge can't catch an invented person or
                # see that he's seeking someone absent.
                facts.add("我能触及的人", referent)
        else:
            lines.append("我能触及的人：无")
            facts.add("我能触及的人", "无")
        if spatial.visible_npcs:
            # One list for both "who else is here" and "whom I can send" (two would print the same
            # people twice). Only people here can be sent. The header must say busy people can't
            # be sent but can be stopped: errands exclude them, while stopping a runner only makes
            # sense while he's busy.
            lines.append(
                "此处听人吩咐做事的（不自己拿主意，也不会推辞：没在忙的我吩咐一声就去办；"
                "正在忙的派不出去，但拦得住。**他们不在上面「我能触及的人」名单里** —— "
                "跟他们打交道只有当面吩咐(ERRAND)或动手(PHYSICAL)这两条路，SEND_MESSAGE, TALK等其他行动不能选这个名单的人）："
            )
            for idx, npc_id in enumerate(spatial.visible_npc_ids, 1):
                rendered = render_npc(spatial.visible_npcs[npc_id], index=idx)
                lines.append(f"  {rendered}")
                facts.add("此处听人吩咐做事的", rendered.split(" ", 1)[-1])
        if spatial.reachable_locations:
            # Reachable = every location reachable by shortest path (not just adjacent), with travel
            # time so distance can be weighed.
            lines.append("我可前往（含需赶路的远处；脚程越远越要想清楚是否值得）：")
            for idx, rl in enumerate(spatial.reachable_locations, 1):
                name = rl.view.name or "某处"
                desc_part = f"：{rl.view.description}" if rl.view.description else ""
                dist = describe_seconds(rl.travel_seconds)
                lines.append(f"  #{idx} {name}（脚程{dist}）{desc_part}")
                facts.add("我可前往", f"{name}（脚程{dist}）")
        else:
            lines.append("我可前往：无处")
            facts.add("我可前往", "无处")
        if spatial.visible_entities:
            lines.append("我够得着的东西：")
            # render_entity is the single format (shared with scene and adjudication). Things in
            # anyone's hands are listed here too; holding is an attribute of the entry, not a
            # separate indexed list (see ``_own_item_ids``).
            for idx, e in enumerate(spatial.visible_entities, 1):
                rendered = render_entity(
                    e, owner_name=_entity_owner_name(
                        e, self_id=personality.soul.agent_id, spatial=spatial),
                    content=e.content,
                    held_by_viewer=e.holder_id == personality.soul.agent_id)
                lines.append(f"  #{idx} {rendered}")
                facts.add("我够得着的东西", rendered)
        else:
            # Say "nothing here" explicitly: omitted, the schema still asks for an entity index
            # from an unprinted list, so models fill physical_entity_index=1 and get rejected as
            # physical_without_target (assemble_scene_context treats "nobody here" the same way).
            lines.append("我够得着的东西：无（此处没有，别人手上也没有）")
            facts.add("我够得着的东西", "无（此处没有，别人手上也没有）")
        lines.append("")

        # Exclude only types that are structurally impossible (empty target set); otherwise the LLM
        # picks from an empty list anyway. Omit the block when nothing is restricted. SEND_MESSAGE
        # is never excluded: directed messages reach the absent and announcements address whoever
        # is here; parsing enforces one channel (see _bind_send).
        present_ids = spatial.visible_agent_ids
        # "Anyone to lay hands on" ≠ "anyone to talk to": no-cognition people can't be talked to
        # but can be stopped or held (``physical_npc_index``). present_ids alone would rule out
        # PHYSICAL when only a runner is here — exactly when you'd want to stop him.
        touchable = bool(present_ids) or bool(spatial.visible_npcs)
        restrictions: List[str] = []
        if not present_ids:
            talk_out = "无人可当面交谈：不可选 TALK。"
            restrictions.append(
                talk_out + "此处只有听人吩咐做事的，动手仍可以落到他们身上（physical_npc_index）"
                "——但动手是拦住、按住、夺他手里的东西，**不是差遣**：按住他并不会让他替我跑一趟。"
                if spatial.visible_npcs
                else talk_out + "COVERT / PHYSICAL 也不能以人为对象。"
            )
        if not spatial.reachable_locations:
            restrictions.append("无处可前往：不可选 MOVE。")
        if not spatial.visible_entities and not touchable:
            restrictions.append("我够不着任何东西、也无人在场：不可选 PHYSICAL。")
        # Busy or restrained runners are excluded here, not just marked: senders pick them anyway
        # despite the "正忙" mark, and this hard-rule block is where the model excludes options
        # first. Must agree with ``ErrandExecutor.start`` (rejects on busy and condition). They stay
        # on the list (they are here); each line's marks say why (``render_npc``).
        free_npcs = [
            nid for nid, seen in spatial.visible_npcs.items()
            if not seen.busy and not seen.condition
        ]
        if not spatial.visible_npcs:
            restrictions.append("此处没有听人吩咐做事的：不可选 ERRAND。")
        elif not free_npcs:
            restrictions.append(
                "此处听人吩咐做事的都腾不出身，这一拍谁也派不出去：不可选 ERRAND。"
                "**别改用传讯去够他** —— 他不在「我能触及的人」名单上，传讯到不了他；"
                "硬挑一个序号，话就落到名单上那个不相干的人头上。要么等他回来，要么我自己跑一趟。"
            )
        if restrictions:
            lines.append("【此刻够不着的行动】（硬规则：先据此排除，再在剩下的里挑；若都够不着，就 act:false 什么都不做）")
            for r in restrictions:
                lines.append(f"  - {r}")
            lines.append("")

        # 【我能做的】 candidate actions — volatile per beat, so they stay in the user message
        # (§3: bulk fillable context in the middle).
        lines.append("【我能做的】")
        for index, candidate in enumerate(candidates):
            lines.append(_format_candidate_line(index, candidate))
        lines.append("")

        # The user message ends with a one-line output reminder near the generation point (§3);
        # the full schema is in the system prefix.
        lines.append("我依上面说定的规矩与 JSON 格式，说出这一拍的选择，只输出 JSON、不写任何多余内容。")

        # Prefix cache: role, task, constraints and the full schema go in the system message,
        # byte-identical across beats (step_hint is a per-world constant); per-beat content goes
        # in the user message.
        #
        # The two time rules aren't duplicates: ABSOLUTE_TIME_RULE covers writing (a future time
        # needs a date); the earlier rule covers reading — choosing on an assumed hour writes no
        # time, so the write rule can't catch it. Keep the read rule before the write rule.
        step_hint = self._step_duration_hint()
        system = f"""你此刻完全代入一个角色，以第一人称「我」思考、权衡、抉择。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【我要做的】
- 此刻我要为接下来定一件事：从【我能做的】里挑出**恰好一个**最该做的行动，作为我推进需求与目标的下一步；这一动若指向外部的人 / 地点 / 物，就把它落到具体对象上，若只关乎我自己，就没有对象可落。
- 选之前我先逐条看清【我能做的】里每个选项的**说明**——别只看类型名，要按它实际能做什么来挑那个真正贴合我此刻需求和目标的（比如对方在不在场、是同步当面还是异步传讯、是改变状态还是只是观察等等）。
- 动手前我先把【我所处的现实】里那几份带 #序号 的名单看清楚——“我能触及的人”有谁(标了在场/不在场)、“可前往”何处、“我够得着的东西”有哪些；我要落到的对象，必须用对应名单里的 #序号来指认，名单上没有的，我此刻就够不着、绝不凭空写一个名字或编号。
- 想清楚该行动是否需要持续做，如果是，estimated_steps表示的是持续这个行动的时长，可以大些，否则，estimated_steps表示完成该行动的时长。对于单次行动和持续行动都符合的条件下，优先选择持续行动。
- 选择地点，物品，人物时可以先理解它们的描述并做出合理的选择。如果没有合理的选择，宁愿不选也不要乱选。

【我被约束的】
- 我想说话/动手的人、想去的地方、想用的东西，只能从【我所处的现实】里带#序号的名单里挑，用序号指认——没列出的，我此刻够不着。
- “我可前往”的每一处都是一大片地方，里头还有许多具体的去处。我想去的地方若不在名单上，多半就落在其中某一处里头——我照名单上各处的说明判断它最可能在哪一处，就去那一处；若我心里本没有非去不可的地方，就不必硬挑一个。
- 当面说话/动手（TALK/COVERT/PHYSICAL-对人）只能挑“我能触及的人”里标(在场)的；只有传讯（SEND_MESSAGE）可以发给标(不在场)的人。
- 动手（PHYSICAL）必须指认出下手的对象，且**只指认一个**：对“我能触及的人”里的人就填 physical_person_index，对“此处听人吩咐做事的”里的人就填 physical_npc_index（两份名单各数各的序号，拿错名单就打到别人身上了），对物就填 physical_entity_index，名单外的抽象之物才自填 item_id——四者填其一，一个都不填等于没动手，这一拍就白费了。
- 唯一的例外是「把手里的东西交出去」：这一动受动的仍是那样东西（照填 physical_entity_index），只是另可填 physical_recipient_index 指认我要交给谁——他必须是“我能触及的人”里标(在场)的。我只是把东西放下、不交给谁，就不填它。
- 派人跑腿（ERRAND）时我交代他的话——去看什么、看完回来报我、别惊动人——写进 action_description，绝不写进 errand_message：那一栏他会一字不改说出口，我的交代一旦写进去，就等于我自己把此行的目的喊了出来。
- 如果要进行的行动涉及到人，行动一定要与【相关人物关系】中描述的关系符合常识，除非有特殊情况。
- 不要脱离我此刻的需求与目标空转，也别被与我无关的事带跑。
- 不要重复我刚刚做完的事，要往前推进。
{CLOSED_WORLD_FACT_RULE_FIRST_PERSON}
- 我不臆造我没感知到的人、事、动向。同时，注意区分已经发生，正在发生，将要发生的客观事实，不能混为一谈，比如“我要去超市”，“我正在去超市的路上”，“我已经到了超市”，这三个是不一样的客观事实。
- 在行动前，需要先判断该事之前是不是做过，如果做过现在又重复做，需要先清楚为什么要重复做。宁可什么都不做也不要做一些无意义的重复的事情。
- 行动前需要分析清楚此刻的自身处境，做出的行动选择需要符合常理，比如我正在被别人攻击，行动选择却是我现在要偷偷观察别人怎么打我，这是明显的不符合常理的。
- 如果近期经历连续的行动失败，可以考虑换一种行动方式，不要过于一直执着于一种行动方式。比如如果交谈持续失败，可以采用物理方式等等。改变策略的建议：换行动方式，换地点，换目标等等。
- SEND_MESSAGE/TALK选person_indices的时候要注意，如果你的action_description，message_content不想让person_indices知道，那你就不要选，否则会造成信息泄漏，比如我现在想与A讨论关于B的坏事，那person_indices应该选择是A，而不是B。
- 只传一句话就用 SEND_MESSAGE：一拍即达、不占人手、对方在哪儿都收得到。派人跑一趟值得占用一个人，是因为它能做 SEND_MESSAGE 做不到的事——把东西送到某人手上、把那边的情形看回来。只为传一句话就派人跑腿，是白费一个人和一个来回；而要打听消息、要人回答我、要说服或引开谁，派人跑腿办不到，他只会走到、说出、看见。
- 行动前我先认准此刻是什么时候（下面会说明此刻是何年何月何日、几时）：眼下的时辰决定这一行动合不合时宜。如果出现时间错乱，需要先思考为什么会出现时间错乱，同时也会有可能付出必要的行动。
- 对于人的生死我要特别关注，如果出现死者复现的情况，我需要分析是什么原因，比如某人之前已经死亡了，但是现在又感知到他在场，这个我需要特别关注，如果有必要可以做出相应的行动。
- 在此处无事可做或陷入僵局时，换个地方也是一个选择：所在之处不同，能看到、听到的也不同。
- 我综合所有信息后，如果存在明显的事实性冲突，我必须要思考为什么会有这样的冲突，同时也会有可能付出必要的行动。
- 如果行动中涉及到人，先确认“我能触及的人”有没有这个人，如果没有，则不要选择该行动，除非有特殊情况。
- 我接下来只做一件事：从【我能做的】里选一个。

【输出】
我用严格 JSON 说出我的选择，不写任何多余内容。字段按我心里成形的先后排列，我就照这个次序往下写：
先在 inner_monologue 里把此刻的思考和权衡想清楚 → 决定这一拍要不要出手(act) → 挑定行动类型(selected_index)
→ **这一动若指向外部对象，从名单里指认它(序号)** → 对象已经定死（或这一动本就无对象可指认），我再写出我要做什么、要说什么。
只要 act=true，action_description 就**必须**写：它是这一动唯一的实质内容——哪怕这一动只关乎我自己、没有任何对象可指认，也照样要写清我此刻在做什么。
{{
  "inner_monologue": "<第一人称，我此刻心里是怎么思考的、怎么权衡的、为什么倾向某个选择，20-60字（先想，再据此定 act 与下面的 selected_index）>",
  "act": <true=我要行动；false=我通盘想过，此刻确实没有任何值得行动的事（比如没有该回应的、该推进的等等），于是什么都不做——这是正当选择，不为填满这一拍硬编一个行动，也不用它逃避该面对的抉择。填 false 时，下面 selected_index 及其余字段一律省略>,
  "selected_index": <仅当 act=true：我选的行动，【我能做的】里 0 到 N 的整数>,
  "person_indices": <整数数组，序号取自"我能触及的人"(1起)。SEND_MESSAGE 可填**多个**序号群发给指定多人；TALK 只填一个(多填只取第一个)，TALK 只能选标(在场)。SEND_MESSAGE 可选(不在场)的>,
  "message_announce": <仅当 SEND_MESSAGE 且我不指名任何人、就是要向我此刻所在地的众人当众宣告：填 true，同时 person_indices 留空。定向传讯给指定的人则省略此字段。**SEND_MESSAGE 这两条通道必须走一条**：既没填 person_indices、又没写 true，这一行动就作废了——我要找的人若不在"我能触及的人"名单上，那就是此刻传不到他，我该改做别的，而不是把话说给空气听>,
  "destination_index": <若我选 MOVE，填"我可前往"里的序号(1起)，不过如果没有合理的地点，也可以省略，不要乱填；否则省略>,
  "move_carry_indices": <整数数组，仅当我选 MOVE 且要把人强行一并带走：填"我能触及的人"里那些标(在场)者的序号(1起)；我独自走，或没有非带不可的人，就省略此字段>,
  "physical_person_index": <若我选 PHYSICAL 且要**对"我能触及的人"里的人**动手：填其中某个标(在场)者的序号(1起)；对此处听人吩咐做事的动手改填 physical_npc_index，其他情况则省略>,
  "physical_npc_index": <若我选 PHYSICAL 且要对**此处听人吩咐做事的**那些人动手(拦下、按住、夺他手里的东西)：填"此处听人吩咐做事的"里的序号(1起)；否则省略>,
  "physical_entity_index": <若我选 PHYSICAL 且要**对物**动手：填"我够得着的东西"里那样东西的序号(1起)；对人动手则省略>,
  "physical_recipient_index": <仅当我这一动是把手中之物交到某人手上：填"我能触及的人"里某个标(在场)者的序号(1起)；只是把东西放下、或这一动与交付无关，一律省略>,
  "item_id": "<仅当 PHYSICAL 的目标不在上面两份名单里(抽象之物，如'门'、'地面')才自填名称，≤10字；能用序号就别用它；非 PHYSICAL 省略>",
  "errand_npc_index": <若我选 ERRAND：填"此处听人吩咐做事的"里那个人的序号(1起)；否则省略。**必填**——这一动就是吩咐他去办，没有他就不成立>,
  "errand_destination_index": <若我选 ERRAND：填"我可前往"里他要去的那个地方的序号(1起)；否则省略。**必填**——一趟不去别处的差事不成立，那种事我自己伸手就够得到>,
  "errand_item_index": <仅当我要让他带上某样东西：填"我够得着的东西"里那一件的序号(1起)，**只能选标了「由我持有」的**——不能带别人手上、也带不了在地上的东西，得先拿到手才托得出去。不带东西就省略>,
  "errand_recipient_index": <仅当这一趟是**只限于对某个人的**：填"我能触及的人"里的序号(1起)。**这一栏决定东西和话落到哪儿**——填了就是交到他手上、只说给他听；不填就是把东西搁在那处、把话在那处当众说出。只是去看看情形则省略>,
  "errand_announce": <仅当我要他到了那处把 errand_message *当众*广播宣告给在场的所有听时：填 true。我看不见那处有谁，所以公开是我主动选的，不是默认；只说给某个人听、或那只是我对他的交代，就省略此字段>,
  "errand_message": "<他到了那头**说出口**的原话。**先定听众**：填了 errand_recipient_index 就只说给那一个人听；填了 errand_announce=true 就是当众宣告。两个都没有，那这句话就不会被他说出口——它会被当成我打发他时的交代收起来。写成能照着说出口的话：第一人称，对听的人说，20-60字。这趟差事是干什么的已经写在 action_description 里，不必在这儿重复。不带话就省略>",
  "action_description": "<**act=true 时必填，任何行动都不可省**。第一人称，写我打算做什么，内容要具体完整：若上面指认了对象，就落到那个人 / 地点 / 物上；若这一动只关乎我自己(无对象可指认)，就直接写我此刻要做的事。描述具体清晰，简洁明了，15-40字>",
  "message_content": "<仅当 SEND_MESSAGE：我此刻要对**上面 person_indices 指认的那些收件人**(或在场众人)当面说出的**原话**，第一人称直接对其说，像真的在传话/宣告——不是对我自己动作的描述，也不要以第三人称提到对方；20-60字。其他行动一律省略此字段>",
  "expected_outcome": "<我期望这一步带来什么，10-30字>",
  "estimated_steps": <整数，>=1；每步{step_hint}，完成某件事情或者持续做某件事的时长，MOVE 由距离自动决定，填 1 即可；若选 REST，休息时长看两头——身体亏空越深(越接近垮掉)越要久，手头待办越多/越急则越短(只够缓口气)，在两者间权衡取值。>,
  "urgency": <"low"|"normal"|"high"|"critical"，仅 SEND_MESSAGE 有效；照我上面写下的原话有多急来定，对方据此判断是否中断手头事>
}}
（urgency 取值含义：{URGENCY_SCALE_DESCRIPTION}）

定这一拍做什么、落笔写上面那几栏之前，还有两条：
- 上面若列了「截止时间已经到的事」，一般就先响应它——那是我自己定下时间的事、眼下已经该做的。
  要是这一拍我另有更该做的而把它放着，我心里得清楚为什么。
{ABSOLUTE_TIME_RULE_FIRST_PERSON}"""
        return system, "\n".join(lines), facts

    # ------------------------------------------------------------------
    # LLM response parsing
    # ------------------------------------------------------------------

    def _parse_llm_selection(
        self,
        content: "str | dict",
        candidates: Sequence[ActionCandidate],
        packet: PerceptionPacket | None = None,
    ) -> LLMSelection | None:
        # Accepts a parsed dict (from _llm_select) or a raw string (tests pass JSON text directly).
        payload = content if isinstance(content, dict) else extract_json_object(content)
        if not isinstance(payload, dict):
            return _reject("no_json_object")
        raw_index = payload.get("selected_index")
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            return _reject("missing_selected_index")
        if not 0 <= raw_index < len(candidates):
            return _reject("selected_index_out_of_range")
        chosen = candidates[raw_index]

        description = payload.get("action_description")
        if not isinstance(description, str) or not description.strip():
            # SEND_MESSAGE models often fill only message_content ("what I said" ≈ "what I did");
            # the information isn't missing, so fall back to it. Other types have no such neighbour.
            spoken = payload.get("message_content")
            if chosen.action_type is not ActionType.SEND_MESSAGE or not isinstance(spoken, str) or not spoken.strip():
                return _reject("missing_action_description")
            description = spoken
            annotate_active_call(action_description_from_message=True)

        inner_monologue = _text(payload, "inner_monologue")
        expected_outcome = _text(payload, "expected_outcome")
        message_content = _text(payload, "message_content")
        steps_raw = payload.get("estimated_steps", 1)
        if isinstance(steps_raw, bool) or not isinstance(steps_raw, int) or steps_raw < 1:
            estimated_steps = 1
        else:
            estimated_steps = steps_raw
        urgency = parse_urgency(payload.get("urgency"), default=Urgency.NORMAL)

        # Targets resolve only via IndexedRef indices; raw ids from the LLM aren't accepted. Each
        # type's binding and structural checks belong to its binder (see _TARGET_BINDERS); the
        # default acts only on oneself.
        slots = _Slots(
            payload=payload,
            packet=packet,
            visible_ids=list(packet.spatial.visible_agent_ids) if packet else [],
            roster_ids=[aid for aid, _who in _message_roster(packet)] if packet else [],
            reachable_ids=(
                [rl.location_id for rl in packet.spatial.reachable_locations] if packet else []
            ),
            entity_ids=[e.entity_id for e in packet.spatial.visible_entities] if packet else [],
            entity_types=(
                {e.entity_id: e.entity_type for e in packet.spatial.visible_entities}
                if packet else {}
            ),
            npc_ids=list(packet.spatial.visible_npc_ids) if packet else [],
            # Sending something requires holding it (see ``_own_item_ids``).
            own_item_ids=_own_item_ids(packet) if packet else [],
        )
        bound = _TARGET_BINDERS.get(chosen.action_type, _bind_nothing)(slots)
        if slots.dropped:
            # Out-of-range indices were dropped silently and the action stands; record them so the
            # hallucination can be traced later.
            annotate_active_call(dropped_indices=list(slots.dropped))
        if bound is None:
            return None          # the binder already recorded the reason on this call's trace

        return LLMSelection(
            action_type=chosen.action_type,
            target=bound.target,
            errand=bound.errand,
            action_description=description.strip() + bound.description_suffix,
            inner_monologue=inner_monologue,
            estimated_steps=estimated_steps,
            urgency=urgency,
            llm_model="llm",
            expected_outcome=expected_outcome,
            message_content=message_content,
        )

    def _expected_outcome(self, action_type: ActionType, needs: NeedEvaluation) -> str:
        if action_type == ActionType.TALK:
            return "通过交谈更清晰地了解对方意图，减少不确定性。"
        if action_type == ActionType.REST:
            return "恢复平静，消除内心噪音。"
        if action_type == ActionType.ERRAND:
            return "把这个行动指派出去，等他回来带话。"
        if action_type == ActionType.PHYSICAL:
            return "通过直接的物理行为改变环境状态或影响他人。"
        if action_type == ActionType.COVERT:
            return "在不被察觉的情况下，探到我本来无从得知的事情。"
        dominant_label = _dominant_need_label(needs) if needs.dominant_need is not None else "稳定"
        return f"推进「{dominant_label}」需求。"


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _bullet_block(items: Sequence[str]) -> str:
    """Render as a bullet list; an empty set gives "无".

    Callers pass lines from ``render_memory_lines`` (oldest first, matching MEMORY_ORDER_HINT); the
    same lines go into ``given_facts``, so the audit sees exactly what the agent read.
    """
    return "\n".join(f"- {t}" for t in items) if items else "无"


def _insights_with_sources(
    insights: Sequence[Memory],
    insight_sources: dict[str, Sequence[Memory]] | None,
    *,
    now_step: int,
    seconds_per_step: int,
    world_start_second_of_day: int = 0,
) -> str:
    """Render insights one per line, citing source memories as "依据": the LLM sees a belief the
    agent induced (not a single experience) and can trace, confirm or doubt it.
    """
    if not insights:
        return "无"
    insight_sources = insight_sources or {}
    rendered: list[str] = []
    # Same oldest-first contract as the experience lists (order_memories_chrono).
    for insight in order_memories_chrono(insights):
        # render_memory renders insights as timeless beliefs (no recency prefix).
        line = f"- {render_memory(insight, now_step=now_step, seconds_per_step=seconds_per_step, world_start_second_of_day=world_start_second_of_day)}"
        sources = insight_sources.get(insight.id, [])
        if sources:
            # Sources are oldest-first too. Don't truncate them: they are length-capped by their own
            # prompt, and a cut sentence is indistinguishable from a badly written one. Size is
            # bounded by count (_compose_retrieval_result takes sources[:3]).
            snippets = "；".join(s.stored_content for s in order_memories_chrono(sources))
            line += f"（依据：{snippets}）"
        rendered.append(line)
    return "\n".join(rendered)


def _dominant_need_label(awareness: object) -> str:
    """Return a human-readable label for the dominant need, if any."""

    dominant = getattr(awareness, "dominant_need", None)
    if dominant is None:
        return "维持稳定"
    active_needs = getattr(awareness, "active_needs", []) or []
    for need in active_needs:
        if getattr(need, "type", None) == dominant:
            label = getattr(need, "label", "") or getattr(dominant, "value", str(dominant))
            return str(label)
    return getattr(dominant, "value", str(dominant))


def _message_roster(packet: PerceptionPacket) -> List[tuple[str, PerceivedIdentity]]:
    """People reachable by SEND_MESSAGE as (id, referent): present + message senders + known
    relations, deduplicated in order; absent people included (unlike TALK/COVERT).

    One source for both the prompt list and IndexedRef parsing, so indices line up. Gender comes
    with the name because the text I write about them carries pronouns; without it the model
    guesses and the guess gets embedded (see core.prompts.person_referent).
    """
    spatial = packet.spatial
    awareness = packet.internal_context
    out: List[tuple[str, PerceivedIdentity]] = []
    seen: set[str] = set()

    def _add(aid: str, name: str, gender: str) -> None:
        if aid and aid not in seen:
            seen.add(aid)
            # Roster names go into the prompt (narrative layer); unknown falls back to "某人",
            # never a bare id.
            out.append((aid, PerceivedIdentity(name=name or "某人", gender=gender)))

    for aid in spatial.visible_agent_ids:
        who = (p.identity if (p := spatial.visible_agents.get(aid)) else None)
        _add(aid, who.name if who else "", who.gender if who else "")
    # A sender's gender can only come from earlier perception, already carried into
    # relevant_relations via the relation channel (to_gender). Dedup keeps the first entry, so if
    # the same person also appears in relations below, look the gender up here first.
    rel_gender = {
        getattr(r, "target_agent_id", ""): getattr(r, "target_agent_gender", "")
        for r in (getattr(awareness, "relevant_relations", []) or [])
    }
    for msg in packet.inbox:
        # Skip senders without cognition: a runner's report is a Message too, but on this list it
        # only yields an undeliverable letter (``deliver_for_agents`` only has inboxes for agents).
        # ``sender_is_agent`` works even for a body never seen before.
        if not msg.sender_is_agent:
            continue
        _add(msg.sender_id, msg.sender_name or "", rel_gender.get(msg.sender_id, ""))
    for rel in getattr(awareness, "relevant_relations", []) or []:
        # The dead aren't contactable — no messages to corpses (they remain only in the relations
        # block, marked "（已死亡）", as remembered bonds).
        if getattr(rel, "deceased", False):
            continue
        _add(
            getattr(rel, "target_agent_id", ""),
            getattr(rel, "target_agent_name", ""),
            getattr(rel, "target_agent_gender", ""),
        )
    return out


def _own_item_ids(packet: PerceptionPacket) -> List[str]:
    """The things I'm holding: a membership set for the "in my hands" check, not an indexed list.

    ``errand_item_index`` counts against the printed entity list (``visible_entities``); a second
    numbering would make the displayed #3 and the parsed #3 different items. Filtered from the
    perception packet: the cognition path has no access to the environment.
    """
    return [
        e.entity_id for e in packet.spatial.visible_entities
        if e.holder_id and e.holder_id == packet.agent_id
    ]


def _format_candidate_line(index: int, candidate: ActionCandidate) -> str:
    action_type = candidate.action_type
    label = action_type.value.upper() if hasattr(action_type, "value") else str(action_type).upper()
    return f"{index}. [{label}] {candidate.description}"