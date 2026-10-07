"""Director channel: the only way a human author reaches into this world.

``EventSystem`` (the automatic LLM editor) is bound by a pacing gate, a quota and no-resolution
rules because an automatic LLM shouldn't have god powers; the director is bound by none of
them. The two share only the record, channel specs and dispatcher (``engine/injection.py``).

The director can never decide for an agent: pressure goes only through message / broadcast,
and what the agent does about it stays his own decision. No channel writes an action for him.

Parsing happens in ``submit()`` (one synchronous LLM call), so an unactionable instruction is
rejected on the spot and ``drain()`` in the step loop is pure dispatch with no LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence

from agent.personality import SECRET_LABEL
from core.context import annotate_active_call, annotate_call, note_active_call_adoption
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import (
    IndexedRef,
    LLMMessage,
    LLMRouter,
    LLMScene,
    coerce_bool,
    extract_json,
    leaks_index_ref,
    output_budget,
)
from core.interfaces.phenomenon import Phenomenon
from core.interfaces.severity import Severity
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    DECEASED_MARK,
    PHENOMENON_DEFINITION,
    SEVERITY_SCALE_DESCRIPTION,
    URGENCY_SCALE_DESCRIPTION,
    clip_text,
    person_referent,
    render_condition,
    render_location,
)
from engine.clock import WorldTime
from engine.injection import (
    Author,
    BroadcastSpec,
    CommittedInjection,
    InjectionDispatcher,
    InjectionLedger,
    MessageSpec,
    parse_broadcast_spec,
    parse_message_spec,
)
from world.models import Npc, WorldEntity, WorldEntityType
from core.interfaces.place import Place
from engine.world_mutation import (
    ConditionMutation,
    EntityMutation,
    Mutation,
    RelocateMutation,
    SpawnMutation,
    VitalityEffect,
    VitalityMutation,
    parse_spawn,
)

if TYPE_CHECKING:
    from agent.agent import Agent

logger = get_logger(__name__)

# Location descriptions are on the menu for identification (the director often names no
# place). This is a blowup guard, not a style limit; director calls are rare, so be generous.
_MENU_DESCRIPTION_CHARS = 500

# Module constants so the developer tools reproduce this call with the same settings.
_SCENE = LLMScene.DIRECTIVE
_TEMPERATURE = 0.2   # translation, not creation
# reason ≤80 chars (120) + feasible (3) + refusal ≤40 chars (60)
# + broadcast{scope 3 + content ≤80 chars 120 + severity 3 + phenomenon 3 + 4 fields 20} = 149
# + message{recipients ~6 + content ≤80 chars 120 + urgency 3 + 3 fields 15} = 144
# + mutations 5 × the longest kind (spawn, same length as a full entity){location/entity 3
#   + type/destroyed 3 + name/state ≤12 chars 18 + description ≤40 chars 60
#   + content ≤80 chars 120 + observation ≤40 chars 60 + kind 3 + 7 fields 35} = 5×302 = 1510
# + narrative_desc ≤60 chars (90) + top-level 7-field structure (35); ~2111 tok
#
# Mutations are budgeted at 5, not the prompt's "at most 3": `_parse_mutations` doesn't
# truncate (that would silently drop what the director asked for), so the realistic bound is higher.
_MAX_TOKENS = output_budget(2111)


@dataclass(frozen=True)
class DirectivePlan:
    """A validated director instruction: what the queue holds.

    At least one of ``broadcast`` / ``message`` / ``mutations`` is non-empty (empty plans are
    rejected at parse time).

    ``directive_text`` is the director's original words, carried to the record so a failed
    intervention can be told apart as badly worded vs. ignored by the world.
    """

    narrative_desc: str
    directive_text: str = ""
    broadcast: BroadcastSpec | None = None
    message: MessageSpec | None = None
    mutations: List[Mutation] = field(default_factory=list)


def _narrative_texts(plan: DirectivePlan) -> Iterable[str]:
    """Every LLM-written human-readable text in this plan, checked for index leaks.

    ``directive_text`` isn't listed: a person typed it, so a #1 there isn't a leak.
    """
    yield plan.narrative_desc
    if plan.broadcast is not None:
        yield plan.broadcast.content
    if plan.message is not None:
        yield plan.message.content
    for mutation in plan.mutations:
        yield mutation.observation


@dataclass(frozen=True)
class DirectiveResult:
    """``submit()``'s answer, returned straight to whoever submitted the instruction."""

    accepted: bool
    reason: str = ""     # on rejection: one line for the director on why it can't be carried out
    preview: str = ""    # on acceptance: what will happen this beat
    queued: int = 0      # plans waiting in the queue after accepting this one


@dataclass(frozen=True)
class DirectivePrompt:
    """Everything one director parse sends: two messages, call settings, and the menus.

    Shared by ``submit`` and the developer console
    (``interaction/api/dev/director_console.py``) so the console tests the production prompt.
    ``menus`` maps index → name (cast / location / entity) for trace annotation and for
    translating the LLM's indices back.
    """

    system: str
    user: str
    menus: dict[str, dict[str, str]]
    scene: LLMScene = _SCENE
    temperature: float = _TEMPERATURE
    max_tokens: int = _MAX_TOKENS


# The prompt states rules, not why they exist (that dilutes attention). Menu headings don't name
# schema paths; channels are explained in the world's terms.
_SYSTEM_PROMPT = f"""\
【你是谁】
你是这个世界的执行器。一个人(导演)用一句话说出他要这个世界发生什么,你把那句话翻成
一份可执行的投递计划。你不创作,只翻译。

【三条铁律】
1. **不加戏。** 他说什么就落实什么,不多不少:他没提的人不要牵扯,他没说的后果不要补写,
   他说得含糊就照含糊落实。宁可少做。
2. **不替人做决定。** 你可以让世界对一个人做任何事,但不能让他自己去做事。「让某人去做
   某事」只能落成一句送到他面前的话 —— 听不听是他自己的事。**那句话由你来写**:照导演
   的意思平实传达就行,不必交代是谁带来的、也不必等他补充细节,这不算加戏。
3. **两条路都走不通才拒绝。** 先看能不能直接落实;不能,就看能不能间接落实(把话送到他
   面前、让世界逼他)。都不行才 feasible=false,并用一句话说清卡在哪 —— 诚实的拒绝远好
   过含糊的落实。
   这四种情况必须拒绝:
   - 认不准他指的是谁 / 哪里 / 哪件东西(名单里没有,或有好几个都像);
   - 说不清究竟要发生什么;
   - 他要的是下面【你做不到的事】里的一件;
   - 这句话根本不是在说这个世界里的一件事。

【你能做的三件事】(按需选用,不必都用)
1. 让一件事**公开发生** —— 在某个地方,或笼罩全域;在场的人都会察觉。**这是最常用的一件**:
   一场火、一阵雨、一次骚动、一个传遍全城的消息,都用它,不需要动任何人或东西。
2. 把一段话**送到指定的人面前**。
3. **直接改变世界本身**,只有当导演要动的是名单上**某个具体的人或东西**,或要一件新东西出现
   时才用,五种:
   改变一件东西的状态、样子或上面写着的字,或毁掉它(只能是「这个世界里的东西」名单里的) /
   在某个地方放下一件世界里原本没有的东西 / 伤害、治好或杀死一个人 /
   把一个人挪到另一个地方(不问路途远近) / 让一个人陷入某种处境（阻拦，制住等等）
   或解开他原有的处境。最多 3 条。
   地方不是东西:一座城、一条街改不了 —— 那里发生了什么,用第 1 条讲出来。
   新东西只能落在某个地方,不会凭空到谁手里:要让某人得到它,就放在他所在的地方,
   别写成送到他手中。
   名单上标着「听吩咐办事」的人没有生死那一种:伤着他、治好他,都用处境来写。

【你做不到的事】(只有这些才拒绝)
- 造出名单上没有的人或地方。
- 让一件东西换到别人手里,或把它从一处挪到另一处。
- 直接改一个人的情绪、记忆或他与谁的关系 —— 想让他恨谁,就让一件值得恨的事发生。
- 对名单上标着「听吩咐办事」的人:所以给他递话、让他去做某事、杀死他。
「让某人去做某事」对其余的人**不在此列**:它照铁律 2 落成送到他面前的一句话,不要拒绝。

【怎么填这些字段】
- 名单里的身份与位置**只为认人认物**:导演常不报名字,只说「那个太子」「守门的那位」
  「桌上那把刀」。认出来之后就别再回头看它 —— 把名单里的细节写进你投出去的内容,
  就是加戏。
- **序号(#N)只填在引用位置**:recipients、location_scope,以及每条改变世界的
  person / location / entity(放下新东西的 location 也是)。
- **其余字段全是给人读的话**(包括说给导演听的 refusal):提到人就写名字,提到地方就写
  地名。读到它们的人手里没有这份名单,所以这些字段里出现 #序号、id、「第N步」都是错的。
- **每条改变世界的动作都必须带 observation**:一句旁人当场看得见的话。写不出来,说明这
  件事不该这么发生。
- 你写下的每一段给人读的话(各处的 content、observation、narrative_desc)都会被人记住、
  日后反复读到,所以提到将来的时点时(这是把话说清楚,不算加戏):
{ABSOLUTE_TIME_RULE}

【输出】严格输出 JSON,不要任何多余内容。字段顺序就是思考顺序,照着往下写:先在 reason
里想清楚他要什么、能不能落实、打算怎么落 → 给出 feasible / refusal → **先指认投给谁、
投到哪(序号),再写投出去的内容**(不先定下收件人,就写不出该对他说的话)→ 最后把实际
投出去的东西概括成 narrative_desc。
字段说明:
entity_type 仅 item(拿得走的物品)或 landmark(拿不走的固着物);
{SEVERITY_SCALE_DESCRIPTION};
{URGENCY_SCALE_DESCRIPTION};
{PHENOMENON_DEFINITION}
{{
  "reason": "他要的是什么;能不能落实;若能,打算怎么落 —— 不超过 80 字",
  "feasible": true 或 false,
  "refusal": "feasible=false 时,给他看的一句话:卡在哪、可以怎么改;写名字不写编号。不超过 40 字;可落实时留空字符串",
  "broadcast": {{"location_scope": 发生地的地点序号(笼罩全域就填 null,或整个不写这一项), "content": "公开发生了什么,不超过 80 字", "severity": "{Severity.prompt_choices()}", "phenomenon": "{Phenomenon.prompt_choices()}"}} 或 null,
  "message": {{"recipients": [人物序号], "content": "送到他们面前的话,不超过 80 字", "urgency": "low|normal|high|critical"}} 或 null,
  "mutations": [
    {{"kind": "entity", "entity": 物件序号, "destroyed": true 或 false, "state": "变化后的状态词,不超过 12 字,不变就留空", "description": "变化后它看上去的样子,不超过 40 字,不变就留空", "content": "改动后上面写着的全文,不超过 80 字,不变就留空", "observation": "旁人看见的一句话,不超过 40 字"}},
    {{"kind": "spawn", "location": 地点序号, "entity_type": "item 或 landmark", "name": "它叫什么,不超过 12 字", "description": "它看上去是什么样,不超过 40 字", "content": "只有写着字、记着东西的才有(信、告示、账册);大多数东西没有,留空字符串。有则不超过 80 字", "observation": "在场的人看见的一句话,不超过 40 字"}},
    {{"kind": "vitality", "person": 人物序号, "effect": "{VitalityEffect.prompt_choices()}", "observation": "旁人看见的一句话,不超过 40 字"}},
    {{"kind": "relocate", "person": 人物序号, "location": 地点序号, "observation": "旁人在**目的地**看见的一句话,不超过 40 字"}},
    {{"kind": "condition", "person": 人物序号, "condition": "他此后身处的处境,不超过 15 字;解开原有处境就留空字符串", "observation": "旁人看见的一句话,不超过 40 字"}}
  ],
  "narrative_desc": "这一拍实际投出去了什么,不超过 60 字。**把上面填过的序号换回名字再写**：写「某地起了大火」,不是「#18 起了大火」"
}}
mutations 没有就给 []。上面五种形状只是格式示例,不是让你五种都用。
最后回读一遍 content / observation / narrative_desc:里面只要还剩一个 # 号,就是写错了,改掉再交。
"""


@dataclass(frozen=True)
class NpcOnMenu:
    """An errand-runner on the menu plus its current location, passed in because the director
    channel doesn't hold ``EnvironmentSystem``."""

    npc: Npc
    location_id: str


def _people_ids(
    all_agents: Mapping[str, "Agent"], npcs: Sequence[NpcOnMenu],
) -> list[str]:
    """Index order of the people menu: agents first, then errand-runners. Shared by menu
    rendering and index resolution."""
    return [*all_agents.keys(), *(n.npc.npc_id for n in npcs)]


def _people_names(
    all_agents: Mapping[str, "Agent"], npcs: Sequence[NpcOnMenu],
) -> dict[str, str]:
    names = {aid: a.personality.soul.name for aid, a in all_agents.items()}
    names.update({n.npc.npc_id: n.npc.name for n in npcs})
    return names


def _without_npc_recipients(
    message: MessageSpec | None, npc_ids: set[str],
) -> MessageSpec | None:
    """Messages go only to people with an inbox. Errand-runners are filtered out even when named;
    if that empties the list, treat the message as absent."""
    if message is None:
        return None
    kept = [rid for rid in message.recipients if rid not in npc_ids]
    if len(kept) != len(message.recipients):
        logger.warning(
            "directive_message_to_npc_dropped",
            extra={"dropped": len(message.recipients) - len(kept)},
        )
    return replace(message, recipients=kept) if kept else None


class DirectorChannel:
    """Queue, parsing and dispatch for director instructions, plus its own injection ledger.

    The ledger is separate from ``EventSystem``'s so the editor's quota structurally can't see
    director injections.
    """

    def __init__(
        self,
        *,
        llm_router: LLMRouter,
        dispatcher: InjectionDispatcher,
        directory: WorldDirectory,
    ) -> None:
        self._llm_router = llm_router
        self._dispatcher = dispatcher
        self._directory = directory
        self._queue: "deque[DirectivePlan]" = deque()
        self._ledger = InjectionLedger(Author.DIRECTOR)

    def pending_count(self) -> int:
        return len(self._queue)

    def restore_state(self, fired_events: Iterable[Mapping[str, Any]]) -> None:
        """Restore the director's past injections. Gates nothing; it's only for complete replay."""
        self._ledger.restore(fired_events)

    async def submit(
        self,
        text: str,
        *,
        all_agents: Mapping[str, "Agent"],
        npcs: Sequence[NpcOnMenu],
        locations: Sequence[Place],
        entities: Sequence[WorldEntity],
        world_time: WorldTime,
    ) -> DirectiveResult:
        """Parse a free-text instruction into a plan and queue it; if nothing actionable comes
        out, reject it and inject nothing."""
        text = (text or "").strip()
        if not text:
            return DirectiveResult(accepted=False, reason="请先写下你想让世界发生什么。")
        if not all_agents:
            return DirectiveResult(accepted=False, reason="这个世界里还没有人。")

        prompt = self.build_prompt(
            text,
            all_agents=all_agents,
            npcs=npcs,
            locations=locations,
            entities=entities,
            world_time=world_time,
        )
        try:
            # Wrong index resolutions only show up against the menu, so annotate it.
            with annotate_call(
                directive_text=text,
                cast_menu=prompt.menus["cast"],
                location_menu=prompt.menus["location"],
                entity_menu=prompt.menus["entity"],
            ):
                response = await self._llm_router.complete(
                    prompt.scene,
                    [
                        LLMMessage(role="system", content=prompt.system),
                        LLMMessage(role="user", content=prompt.user),
                    ],
                    max_tokens=prompt.max_tokens,
                    temperature=prompt.temperature,
                    json_mode=True,
                )
        except Exception as exc:  # noqa: BLE001 — a failed call means rejection; never inject vague content
            logger.warning("directive_call_failed", extra={"error": str(exc)})
            return DirectiveResult(accepted=False, reason="没能读懂这条指令,请换一种说法。")

        result, plan = self.interpret(
            response.content,
            text=text,
            all_agents=all_agents,
            npcs=npcs,
            locations=locations,
            entities=entities,
        )
        if plan is None:
            return result
        self._queue.append(plan)
        return replace(result, queued=len(self._queue))

    def build_prompt(
        self,
        text: str,
        *,
        all_agents: Mapping[str, "Agent"],
        npcs: Sequence[NpcOnMenu],
        locations: Sequence[Place],
        entities: Sequence[WorldEntity],
        world_time: WorldTime,
    ) -> DirectivePrompt:
        """Assemble everything this parse would send, without sending; shared with the
        developer console."""
        names = _people_names(all_agents, npcs)
        return DirectivePrompt(
            system=_SYSTEM_PROMPT,
            user=self._build_user_prompt(
                text,
                all_agents=all_agents,
                npcs=npcs,
                locations=locations,
                entities=entities,
                world_time=world_time,
            ),
            menus={
                "cast": {
                    str(i): names.get(pid, "")
                    for i, pid in enumerate(_people_ids(all_agents, npcs), 1)
                },
                "location": {str(i): loc.name for i, loc in enumerate(locations, 1)},
                "entity": {str(i): e.name for i, e in enumerate(entities, 1)},
            },
        )

    def interpret(
        self,
        raw_response: str,
        *,
        text: str = "",
        all_agents: Mapping[str, "Agent"],
        npcs: Sequence[NpcOnMenu],
        locations: Sequence[Place],
        entities: Sequence[WorldEntity],
    ) -> tuple[DirectiveResult, DirectivePlan | None]:
        """Validate raw LLM output into an answer + a plan without queueing, so the developer
        console can run production validation without touching the world.

        ``plan is None`` ⇔ rejected, and ``result.reason`` is what to tell the director.
        """
        try:
            data = extract_json(raw_response)
        except Exception as exc:  # noqa: BLE001 — an unreadable response means rejection
            logger.warning("directive_parse_failed", extra={"error": str(exc)})
            return DirectiveResult(accepted=False, reason="没能读懂这条指令,请换一种说法。"), None

        if not isinstance(data, dict):
            note_active_call_adoption(False, reason="response_not_an_object")
            return DirectiveResult(accepted=False, reason="没能读懂这条指令,请换一种说法。"), None
        if not coerce_bool(data.get("feasible"), False):
            refusal = str(data.get("refusal", "")).strip()
            # Without this the trace shows a successful call and hides that nothing was injected.
            note_active_call_adoption(False, reason=refusal or "llm_declared_infeasible")
            return DirectiveResult(accepted=False, reason=refusal or "这条指令落实不了。"), None

        plan = self._parse_plan(
            data,
            directive_text=text,
            person_ids=_people_ids(all_agents, npcs),
            npc_ids={n.npc.npc_id for n in npcs},
            location_ids=[loc.place_id for loc in locations],
            entity_ids=[e.entity_id for e in entities],
        )
        if plan is None:
            # Declared feasible but nothing dispatchable parsed (out-of-range indices, mutations
            # without an observation): reject rather than inject an empty event.
            note_active_call_adoption(False, reason="no_dispatchable_channel")
            logger.warning("directive_plan_empty", extra={"text": text[:80]})
            return DirectiveResult(
                accepted=False, reason="这条指令没有落到具体的人、地点或物件上,请说得更确切些。",
            ), None

        leaked = next((t for t in _narrative_texts(plan) if leaks_index_ref(t)), None)
        if leaked is not None:
            # Record the leak but still dispatch. The first adoption verdict wins, so the True
            # below can't overwrite this.
            note_active_call_adoption(False, reason="index_ref_in_narrative")
            logger.warning("directive_index_ref_in_narrative", extra={"text": leaked[:80]})

        # The plan's shape is the first thing to check when reviewing translation vs. overreach.
        note_active_call_adoption(True)
        annotate_active_call(
            channels=self._channels_of(plan),
            mutation_kinds=[type(m).__name__ for m in plan.mutations],
            recipients=[self._directory.agent_name(a) for a in (plan.message.recipients if plan.message else [])],
            narrative_desc=plan.narrative_desc,
        )
        return DirectiveResult(accepted=True, preview=plan.narrative_desc), plan

    def describe_plan(self, plan: DirectivePlan) -> dict[str, Any]:
        """Lay out a plan with ids turned into names, for the developer console. Read-only."""
        broadcast = plan.broadcast
        message = plan.message
        return {
            "narrative_desc": plan.narrative_desc,
            "channels": self._channels_of(plan),
            "broadcast": None if broadcast is None else {
                "content": broadcast.content,
                "location": (
                    self._directory.location_name(broadcast.location_scope)
                    if broadcast.location_scope else "全域"
                ),
                "severity": broadcast.severity.value,
                "phenomenon": broadcast.phenomenon.value,
            },
            "message": None if message is None else {
                "content": message.content,
                "recipients": [self._directory.agent_name(a) for a in message.recipients],
                "urgency": message.urgency.value,
            },
            "mutations": [self._describe_mutation(m) for m in plan.mutations],
        }

    def _describe_mutation(self, mutation: Mutation) -> dict[str, Any]:
        if isinstance(mutation, EntityMutation):
            target = self._directory.entity_name(mutation.entity_id)
            changes = [c for c in (
                mutation.new_state,
                f"样子：{mutation.new_description}" if mutation.new_description else "",
                f"写着：{mutation.new_content}" if mutation.new_content else "",
            ) if c]
            detail = "摧毁" if mutation.destroyed else ("；".join(changes) or "—")
        elif isinstance(mutation, SpawnMutation):
            target = mutation.name
            detail = f"放在{self._directory.location_name(mutation.location_id)}"
        elif isinstance(mutation, VitalityMutation):
            target = self._body_name(mutation.body_id)
            detail = mutation.effect.value
        elif isinstance(mutation, ConditionMutation):
            target = self._body_name(mutation.body_id)
            detail = mutation.description or "解开处境"
        else:
            assert isinstance(mutation, RelocateMutation)
            target = self._body_name(mutation.body_id)
            detail = f"→ {self._directory.location_name(mutation.location_id)}"
        return {
            "kind": type(mutation).__name__,
            "target": target,
            "detail": detail,
            "observation": mutation.observation,
        }

    def _body_name(self, body_id: str) -> str:
        """Name of a person or an errand-runner (``agent_name`` only knows the former)."""
        entry = self._directory.describe(body_id)
        return entry.name if entry is not None else "某人"

    @staticmethod
    def _channels_of(plan: DirectivePlan) -> list[str]:
        return sorted(
            (["broadcast"] if plan.broadcast else [])
            + (["message"] if plan.message else [])
            + (["mutation"] if plan.mutations else [])
        )

    def _build_user_prompt(
        self,
        text: str,
        *,
        all_agents: Mapping[str, "Agent"],
        npcs: Sequence[NpcOnMenu],
        locations: Sequence[Place],
        entities: Sequence[WorldEntity],
        world_time: WorldTime,
    ) -> str:
        """The per-call menus. They are for identification only: no recap of recent events and
        no dynamic state (emotion, needs, activity)."""
        # People and errand-runners share one index space: two lists would invite filling one
        # list's index into the other's slot. Code decides the tier from the resolved id.
        names = _people_names(all_agents, npcs)
        agent_lines = "\n".join(
            [self._agent_line(i + 1, a) for i, a in enumerate(all_agents.values())]
            + [
                self._npc_line(len(all_agents) + i + 1, npc, names)
                for i, npc in enumerate(npcs)
            ]
        ) or "（无）"
        # ``render_location`` keeps the director's menu and what characters see in one format.
        location_lines = "\n".join(
            f"#{i + 1} {clip_text(render_location(loc), _MENU_DESCRIPTION_CHARS)}"
            for i, loc in enumerate(locations)
        ) or "（无）"
        entity_lines = "\n".join(
            self._entity_line(i + 1, e, names) for i, e in enumerate(entities)
        ) or "（无）"

        return f"""\
【当前世界时间】{world_time.time_label}

【这个世界里的人】
{agent_lines}

【这个世界里的地方】
{location_lines}

【这个世界里的东西】
{entity_lines}

【导演的指令】
{text}

照上面说定的 JSON 格式把这条指令翻出来(reason 在前),只输出 JSON、不写任何多余内容。
记住:忠实翻译,不加戏;落实不了就 feasible=false 并说清卡在哪。"""

    def _agent_line(self, index: int, agent: "Agent") -> str:
        """One line of identifying facts plus one of background. Every item helps match "the
        one at the gate" / "his brother" to a person: gender and age separate same-surname
        siblings, and death must be marked."""
        soul = agent.personality.soul
        facts = [f for f in (
            soul.role,
            f"现于{self._directory.location_name(agent.personality.state.current_location)}",
            # Needed to lift or replace a condition and to write an observation that fits it.
            render_condition(agent.personality.state.condition),
            "" if agent.is_active else DECEASED_MARK,
        ) if f]
        head = f"#{index} {soul.identity_text()}" + (f"（{'，'.join(facts)}）" if facts else "")
        # The secret too: an instruction may name a person by what only they know.
        detail = [line for line in (
            soul.background,
            f"{SECRET_LABEL}：{soul.secret}" if soul.secret else "",
        ) if line]
        return "\n".join([head, *(f"    {line}" for line in detail)])

    def _npc_line(self, index: int, npc: NpcOnMenu, names: Mapping[str, str]) -> str:
        """List form (``person_referent``) plus a background line. The "听吩咐办事" mark is
        what the prompt's errand-runner rules key on; the errand helps match "the one carrying
        the letter"."""
        body = npc.npc
        errand = body.errand
        head = person_referent(
            body.name, body.gender,
            f"{body.age}岁" if body.age else "",
            "听吩咐办事",
            f"现于{self._directory.location_name(npc.location_id)}",
            render_condition(body.condition),
            f"正替{names.get(errand.requester_id) or '某人'}办一趟差事" if errand is not None else "",
        )
        line = f"#{index} {head}"
        return f"{line}\n    {body.description}" if body.description else line

    def _entity_line(
        self, index: int, entity: WorldEntity, names: Mapping[str, str],
    ) -> str:
        """One line with name, state and placement, one with the description.

        State is needed to write what a thing becomes; placement tells two letters apart and
        is needed for the observation. Description goes on its own line because it usually
        ends with a full stop. Placement is either someone's hands or a place, never both.
        """
        if entity.owner_id:
            placement = f"在{names.get(entity.owner_id) or '某人'}手上"
        elif entity.location_id:
            placement = f"现于{self._directory.location_name(entity.location_id)}"
        else:
            placement = ""
        facts = [f for f in (
            f"此刻状态：{entity.state}" if entity.state else "",
            placement,
        ) if f]
        lines = [f"#{index} {entity.name}" + (f"（{'，'.join(facts)}）" if facts else "")]
        if entity.description:
            lines.append(f"    {entity.description}")
        # Content in full: an edit supplies the full revised text. The director has the god
        # view, so readability rules don't apply here.
        if entity.content:
            lines.append(f"    上面写着：{entity.content}")
        return "\n".join(lines)

    def _parse_plan(
        self,
        data: dict[str, Any],
        *,
        directive_text: str,
        person_ids: list[str],
        npc_ids: set[str],
        location_ids: Sequence[str],
        entity_ids: Sequence[str],
    ) -> DirectivePlan | None:
        location_ref = IndexedRef(location_ids)
        broadcast = parse_broadcast_spec(data.get("broadcast"), location_ref)
        message = _without_npc_recipients(
            parse_message_spec(data.get("message"), person_ids), npc_ids,
        )
        mutations = self._parse_mutations(
            data.get("mutations"),
            person_ids=person_ids,
            npc_ids=npc_ids,
            location_ref=location_ref,
            entity_ref=IndexedRef(entity_ids),
        )
        if broadcast is None and message is None and not mutations:
            return None
        narrative_desc = str(data.get("narrative_desc", "")).strip()
        if not narrative_desc:
            return None
        return DirectivePlan(
            narrative_desc=narrative_desc,
            directive_text=directive_text,
            broadcast=broadcast,
            message=message,
            mutations=mutations,
        )

    def _parse_mutations(
        self,
        raw: Any,
        *,
        person_ids: list[str],
        npc_ids: set[str],
        location_ref: IndexedRef,
        entity_ref: IndexedRef,
    ) -> List[Mutation]:
        """Parse mutations; any without an observation is dropped (an invariant).

        Mutations the channel would reject at commit (vitality on an errand-runner) are dropped
        here: submission is the only rejection point, so the director isn't told "accepted" for
        something that won't happen.
        """
        if not isinstance(raw, list):
            return []
        person_ref = IndexedRef(person_ids)
        out: List[Mutation] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            observation = str(item.get("observation", "")).strip()
            if not observation:
                logger.warning("directive_mutation_without_observation", extra={"kind": item.get("kind")})
                continue
            kind = str(item.get("kind", "")).strip().lower()
            if kind == "entity":
                resolved = entity_ref.resolve([item.get("entity")])
                if not resolved:
                    continue
                out.append(EntityMutation(
                    observation=observation,
                    entity_id=resolved[0],
                    new_state=str(item.get("state", "")).strip(),
                    new_description=str(item.get("description", "")).strip(),
                    new_content=str(item.get("content", "")).strip(),
                    destroyed=coerce_bool(item.get("destroyed"), False),
                ))
            elif kind == "spawn":
                spawn = parse_spawn(item, location_ref)
                if spawn is None:
                    continue
                # Dropped here, not at commit (see docstring).
                if spawn.entity_type not in {t.value for t in WorldEntityType}:
                    logger.warning("directive_spawn_unknown_kind", extra={"kind": spawn.entity_type})
                    continue
                out.append(spawn)
            elif kind == "vitality":
                resolved = person_ref.resolve([item.get("person")])
                # How much vitality a tier means is world_mutation's call.
                try:
                    effect = VitalityEffect(str(item.get("effect", "")).strip().lower())
                except ValueError:
                    continue
                if not resolved:
                    continue
                if resolved[0] in npc_ids:
                    logger.warning("directive_vitality_on_npc_dropped", extra={"effect": effect.value})
                    continue
                out.append(VitalityMutation(
                    observation=observation, body_id=resolved[0], effect=effect,
                ))
            elif kind == "relocate":
                who = person_ref.resolve([item.get("person")])
                where = location_ref.resolve([item.get("location")])
                if not who or not where:
                    continue
                out.append(RelocateMutation(
                    observation=observation, body_id=who[0], location_id=where[0],
                ))
            elif kind == "condition":
                who = person_ref.resolve([item.get("person")])
                if not who:
                    continue
                out.append(ConditionMutation(
                    observation=observation,
                    body_id=who[0],
                    description=str(item.get("condition", "") or "").strip(),
                ))
        return out

    async def drain(
        self, *, step: int, agents: Dict[str, "Agent"],
    ) -> list[CommittedInjection]:
        """Land the whole queue this step (no per-step rationing) and return the committed
        injections. The Runtime stamps receipts at end of step."""
        drained: list[CommittedInjection] = []
        while self._queue:
            plan = self._queue.popleft()
            committed = await self._dispatcher.dispatch(
                author=Author.DIRECTOR,
                step=step,
                agents=agents,
                narrative_desc=plan.narrative_desc,
                broadcast=plan.broadcast,
                message=plan.message,
                mutations=plan.mutations,
                directive_text=plan.directive_text,
            )
            if committed is not None:
                self._ledger.record(committed.event)
                drained.append(committed)
        return drained
