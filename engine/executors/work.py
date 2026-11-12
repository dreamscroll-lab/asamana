"""Work action executor for WORK actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from core.action_duration import action_duration_steps
from core.coerce import coerce_mapping_list, coerce_str
from core.context import GivenFacts, annotate_call
from core.interfaces.action import (
    ActionResult, ActionType, AgentAction, EntitySpawn, EntityStateChange,
)
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import (
    IndexedRef, LLMRouter, coerce_bool, output_budget,
)
from core.interfaces.perception import DEFAULT_ENTITY_STATE
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    condition_line,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    MEMORY_ORDER_HINT,
    SituationVoice,
    render_memory_lines,
    render_situation_header,
    vitality_line,
)
from core.duration import describe_duration
from core.interfaces.execution import TickResult
from engine.executors.base import (
    ActionExecutionState, ActionExecutor, adjudication_failed_result, complete_json,
)
from engine.narration import (
    ensure_doer_named, format_interrupt_reason_3p, format_interrupt_thought, observed_here,
    scene_line,
)
from engine.scene import SceneVisibility, assemble_scene_context, observe_location, situation_for

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.memory_types import Memory
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem
    from core.interfaces.entity import WorldEntity

logger = get_logger(__name__)

_WORK_RECALL_K = 5
# Above RETRIEVAL_SCORE_FLOOR, which only filters junk: a loosely related memory gets written into
# the fact or a product's content as if it bore on the job, while a missed one only leaves the
# content coarser. Don't go to 0.5: from there many purposes recall nothing.
_WORK_RECALL_SCORE_FLOOR: float = 0.45


@dataclass(frozen=True)
class _WorkVerdict:
    """Everything one WORK self-assessment produces, one piece per channel, from one LLM call.

    "What part still isn't done" lives in 1p ``fact``, not ``failure_reason`` (3p, empty on
    success): "mostly done, one part missing" is the common case, and the residue judge reads it
    from memory to set the next goal, so it must survive success.

    ``observation`` is not a copy of ``outcome``: outcome includes the product, which onlookers
    must not see.

    What the session left in the world is stated explicitly, never inferred from ``succeeded``:
    thinking something through succeeds with nothing made, and a half-made thing still exists.
    ``changes`` has one entry per thing, in the same shape as a conversation's
    (``talk_entities``): an entry naming a thing in hand alters or sets it down, one naming
    nothing is made on the spot. Repeated polishing alters the thing in hand instead of piling
    up near-duplicates; both may be given (a chair carved from a block).
    """

    succeeded: bool
    fact: str
    outcome: str
    observation: str
    failure_reason: str
    changes: tuple["_WorkChange", ...] = ()


@dataclass(frozen=True)
class _WorkChange:
    """How one thing stands after the session. Empty text fields are unchanged.

    ``carried`` is where the doer's ``holder`` would be in a conversation's entry
    (``talk_entities``): there two speakers can end up holding it, so a number picks one; here
    only the doer can, so it is a yes or no. For a thing that was lying here, yes is picking it up.
    """

    entity: "WorldEntity | None"   # a thing in reach (in hand or lying here); None = made this session
    name: str = ""
    state: str = ""
    description: str = ""
    content: str = ""
    carried: bool = True


class WorkExecutor(ActionExecutor):
    """Executor for WORK actions: finishing something of one's own, with no external object
    and no relation delta.

    Judged in character, first person: with no opponent, it's an honest self-assessment. It
    must rest on what the doer perceives (things at hand, who's present, his
    ``expected_outcome``): from persona alone the verdict is always "nothing came of it" and the
    goal is retried forever.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        directory: WorldDirectory,
        seconds_per_step: int = 3600,
        *,
        world_start_second_of_day: int = 0,
    ) -> None:
        self._llm = llm_router
        self._directory = directory
        self._seconds_per_step = seconds_per_step
        self._world_start_second_of_day = world_start_second_of_day

    def _outcome(self, agent_id: str, location: str, *, succeeded: bool) -> str:
        """Fallback 3p outcome when the judge gives none. Never quotes the purpose: it's a
        first-person sentence and breaks a third-person line."""
        doer_name = self._directory.agent_name(agent_id)
        body = (
            f"{doer_name}做完了手上的事。" if succeeded
            else f"{doer_name}忙活了一阵，终究没能做成手上的事。"
        )
        return scene_line(location, body)

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        description = action.action_description or "working"
        # No feasibility gate. Never quote description: it's private and first person, and this
        # string is also the onlookers' observation.
        opening = scene_line(
            observe_location(environment, action.agent_id),
            f"{self._directory.agent_name(action.agent_id)}着手做手上的事。",
        )
        return ActionExecutionState.create(
            action_type=ActionType.WORK,
            initiator_id=action.agent_id,
            participant_ids=[action.agent_id],
            purpose=description,
            started_step=step,
            estimated_steps=action_duration_steps(action.action_type, action.estimated_steps),
            target=action.target,
            opening_outcome=opening,
            opening_observations=observed_here(environment, action.agent_id, opening),
            expected_outcome=action.expected_outcome,
        )

    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        doer_name = self._directory.agent_name(state.initiator_id)
        location = observe_location(environment, state.initiator_id)
        elapsed = state.estimated_steps - state.remaining_steps
        if elapsed <= 0:
            body = f"{doer_name}着手做手上的事。"                  # same form as start
        else:
            body = f"{doer_name}手上的事仍在进行，已投入{describe_duration(elapsed, self._seconds_per_step)}。"
        progress = scene_line(location, body)
        return [TickResult(
            agent_id=state.initiator_id, outcome=progress,
            observations=observed_here(environment, state.initiator_id, progress),
        )]

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
            return [stored]
        location = observe_location(environment, state.initiator_id)
        situation_header = render_situation_header(
            situation_for(environment, state.initiator_id), voice=SituationVoice.FIRST,
        )
        # In-character evidence: no background or intents of others (information asymmetry); no
        # header, since situation_header already anchors time and place.
        scene = assemble_scene_context(
            state.initiator_id, environment=environment, directory=self._directory,
            agents=agents,  # only for condition marks, visible on sight
            include_header=False,
            voice=SituationVoice.FIRST,
            visibility=SceneVisibility.OWN_EYES,
            # Numbered for the changes' entity: his own things and what lies here. Others' things
            # stay out: taking one from someone is PHYSICAL.
            split_in_reach=True,
        )
        doer = agents.get(state.initiator_id)
        gen_result = await self._generate_outcome(
            agent=doer,
            purpose=state.purpose,
            recalled=await self._recall(doer, state.purpose, step),
            in_reach=scene.in_reach,
            now_step=step,
            duration_label=describe_duration(state.estimated_steps, self._seconds_per_step),
            situation_header=situation_header,
            scene=scene.text,
            expected_outcome=state.expected_outcome,
        )
        if gen_result is None:
            return [adjudication_failed_result(state, step, outcome=scene_line(
                location,
                f"{self._directory.agent_name(state.initiator_id)}着手做手上的事，一时未能确知是否做成。",
            ))]
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.WORK,
            action_description=state.purpose,
            target=state.target,
            estimated_steps=state.estimated_steps,
        )
        # A missing observation falls back to the template, never to outcome (it has the product).
        doer_name = self._directory.agent_name(state.initiator_id)
        fallback_3p = self._outcome(state.initiator_id, location, succeeded=gen_result.succeeded)
        outcome = (
            scene_line(location, ensure_doer_named(gen_result.outcome, doer_name))
            if gen_result.outcome else fallback_3p
        )
        observation = (
            scene_line(location, ensure_doer_named(gen_result.observation, doer_name))
            if gen_result.observation else fallback_3p
        )
        changes, spawns = _world_changes(
            gen_result.changes, doer_id=state.initiator_id, doer_name=doer_name,
            here=environment.get_body_location(state.initiator_id), location=location, step=step,
        )
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=outcome,            # 3p full authoritative record (with the product)
            observations=observed_here(environment, state.initiator_id, observation),
            happening=outcome,          # what close watching catches: the product observation omits
            succeeded=gen_result.succeeded,
            failure_reason=gen_result.failure_reason,   # 3p authoritative "why it didn't work"
            factual_memory=gen_result.fact,             # 1p self-assessed memory
            entity_spawns=spawns,
            entity_state_changes=changes,
        )]

    async def _interrupt_memory(
        self, agent: "Agent", environment: "EnvironmentSystem", *, happened: str, now_step: int,
    ) -> str:
        """The worker's own one-line record of the work that was cut off, or "" when no usable one
        came back (the caller then keeps the record code built: being interrupted is a fact)."""
        situation_header = render_situation_header(
            situation_for(environment, agent.agent_id), voice=SituationVoice.FIRST,
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场被打断的事——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角,我留下了什么没做完、此刻什么感受。始终用我自己的口吻。
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "一句话（不超过36字）"}
"""
        doer_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        doer_vitality = vitality_line(
            agent.personality.state.vitality, voice=SituationVoice.FIRST, lead="",
        )
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{doer_condition}
{doer_vitality}

【刚刚发生的事】
{happened}（时间地点见开头）。

依上面说定的 JSON 格式记下这一场，只输出 JSON、不写任何多余内容。"""
        with annotate_call(
            given_facts=GivenFacts()  # same source as user above
            .add("此刻何时何地", situation_header)
            .add("我的处境", doer_condition)
            .add("我的生命值", doer_vitality)
            .add("刚刚发生的事", happened),
            action_owner=agent.personality.soul.name,
        ):
            data = await complete_json(
                self._llm.complete, system, user,
                temperature=0.7,
                max_tokens=output_budget(69),  # fact ≤36 chars ≈54 + key and braces ~15; est. ~69 tok
                failure_event="work_interrupt_failed",
                log_extra={"agent_id": agent.agent_id, "step": now_step},
                required=("fact",),
            )
        return str(data["fact"]).strip() if data is not None else ""

    async def interrupt(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        agent = agents.get(state.initiator_id)
        thought_part = format_interrupt_thought(thought)
        location = observe_location(environment, state.initiator_id)
        # Only third-person cause may name what interrupted it (see base.interrupt).
        cut = f"因{cause}而中断" if cause else "被打断"

        factual = ""
        if agent is not None:
            factual = await self._interrupt_memory(
                agent, environment,
                happened=f"我做「{state.purpose}」，已投入{elapsed_label}，突然{cut}{thought_part}",
                now_step=step,
            )
        if not factual:
            factual = (
                f"在{location}做「{state.purpose}」做了{elapsed_label}便{cut}，没能收尾{thought_part}"
                if elapsed > 0
                else f"刚着手「{state.purpose}」就{cut}{thought_part}"
            )

        doer_name = self._directory.agent_name(state.initiator_id)
        gist = scene_line(location, (
            f"{doer_name}做手上的事做了{elapsed_label}，{cut}。"
            if elapsed > 0
            else f"{doer_name}刚着手做手上的事就{cut}。"
        ))
        outcome = gist + format_interrupt_reason_3p(doer_name, thought)
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=state.action_type,
            action_description=state.purpose,
            target=state.target,
        )
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=outcome,            # 3p full record (who, where, what, interrupted)
            gist=gist,
            succeeded=False,
            factual_memory=factual,     # 1p self-assessed memory
        )]

    async def _recall(self, agent: "Agent | None", purpose: str, step: int) -> "list[Memory]":
        """What the doer remembers about the job, recalled by its purpose. Empty on a read
        failure (Rule 6) or a missing doer."""
        if agent is None:
            return []
        try:
            return await agent.memory_system.retrieve(
                purpose, current_step=step, top_k=_WORK_RECALL_K, score_floor=_WORK_RECALL_SCORE_FLOOR,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: judge without recall
            logger.warning(
                "work_memory_retrieve_failed",
                extra={"agent_id": agent.agent_id, "step": step, "error": str(exc)},
            )
            return []

    async def _generate_outcome(
        self,
        *,
        agent: "Agent | None",
        purpose: str,
        duration_label: str,
        recalled: "list[Memory] | None" = None,
        now_step: int = 0,
        situation_header: str = "",
        scene: str = "",
        expected_outcome: str = "",
        in_reach: "tuple[WorldEntity, ...]" = (),  # index → item, same list as the scene lines
    ) -> "_WorkVerdict | None":
        """Adjudicate a finished WORK (1p fact, 3p outcome, why) in one call. Returns ``None``
        when adjudication can't happen, so the caller fabricates nothing.

        All four inputs are needed: ``scene`` (what's at hand, who's present),
        ``expected_outcome`` (a reference, not a gate: the decision LLM often writes downstream
        consequences into it), persona/energy/emotion, and ``recalled`` (what the doer knows
        about the job: where an earlier session left off, the facts a letter or plan is written
        from; without it a product's content is invented and then persisted).

        Accepted trade-off: ``scene`` is a snapshot at completion, so midway comings and goings
        are invisible to the judge.
        """
        if agent is None:
            logger.warning("work_self_judge_no_doer", extra={"purpose": purpose})
            return None

        header_part = f"{situation_header}\n\n" if situation_header else ""
        system = """\
我此刻代入这个角色,以第一人称「我」自评我刚做完的事——这是我一贯的记事方式,与处境无关。

【我要做的】
1. 先说清这段工夫我**实际做出了什么**:手上真正出来的东西、或推进到了哪一步。
2. 再据它判一件事:**我这个动作的意图,达成了没有。**

【我据以判断的】
- 手边的东西:所给现场列出的是这里稀罕、特定或会有人争的东西,这类东西没列出就是没有;其中标了「由…持有」的在别人手上,不是我可以径直取用的。
- 此刻与我同处一地的人:他们不会主动来影响我,但他们在场本身可能使我受到影响。
- 我的本事、此刻的精力与情绪:做事的水准由它们定。
- 我动手前期望的:只作参照,用来衡量我实际做出来的东西。
- 我想起的旧事:是我动手时已经知道的,写下的内容只能出自这些、现场和我的本事。

【我不能做的】
- **不能把那句打算当成已经发生的事**:它是我动手前的打算,这段工夫里究竟如何,正是我要判的。
  更不能把它换个说法抄一遍,充作我实际做出来的东西。
- **不能把过程当成产出**:比如反复斟酌而始终拿不出任何实在的东西,那就是没成。
- **不能把我期望的照抄成已经发生的**。
- 不能替同处一地的人编造他们并没有做出的举动。
- **不能把想起的旧事当成这段工夫里发生的**:旧事里的打算、上回做到的地步,只说明我从哪儿接手,
  这一回做成了什么仍要我来判。与这件事无关的旧事就放过,不硬写进 fact。

【我记事的规矩】
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + "\n" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【我要写的】
- fact:第一人称,我自己记下的这一场——做出了什么、成果是什么;有新东西产生要点明,产生了具体、
  实质的内容(哪怕没有东西成形)也要在这里写清楚;若有没了结的部分或压根没做成的,一并说清缘由。
  fact 只有我自己看得到,别人看不到也听不到,所以不会有内容泄漏的风险。
- success:上面第 2 步的判断。
- changes:这一场里**改变了或被我留下**的东西,一件一条,最多 3 条;什么都没变、也没留下什么,就给空数组。
  · entity:是所给现场「我手上的东西」或「此处地上的东西」里的哪一件,填它的序号;这一场**新做出来**的有形之物填 0。
  · name / state / description / content:它经这一场之后的名字、状态、样子、内容（如果有就写，没有就不写）。新做出来的
    name 必填,其余有则填;原有的那件东西,哪一栏变了才填哪一栏,没变留空。
  · carried:它此刻在哪儿:我带在身上 true,留在此处 false。地上没标「可取」的东西拿不走,只能填 false。
  原有的东西只是使用一下，比如比对了一遍、翻看了一遍之类,没有改变东西本身、也没挪动,就不写它;只是挪动了
  (把手上的放下、把地上的拿走之类的),也写一条:各栏留空,carried 照实填。
  **改名是换了身份,不是变了样**:它已经不再是原来那样东西了才改名;变旧、变短、变脏、
  写满了字,名字照旧——那个词是别人记忆里认得它的凭据,换掉它,那些记忆就指向了一件不
  存在的东西。
  **新做出来的和改了的是两回事,可以同时有**:拿一块料削出一把椅子,料变短了是一条,椅子
  出现了是另一条。
  以下情形**不写新做出来的东西**:
  · 这段工夫只在心里过了一遍,没有任何东西成形;
  · 我只是拿起来看看、核对一遍,什么也没添;
  · 我动的就是所给的那一件(手上的或地上的)——它变了样就写它那一条,不是另报一件新的出来;
  · 如果想要东西已经在世界中存在，那就不要再重复生成了，也不要生成一个相似的东西，这样没有意义；
  「进展」「头绪」「决心」「把握」这类**没有形体的东西一概不算**,写进来就是错。
  如果实在没有新东西产生那就留空，留空比捏造更好。
  **东西有没有留下,和事情成没成是两回事**:一件做到一半、离我期望的还差得远的东西,
  是没达成的意图,却是实实在在的一件东西——第一次做出它时照写不误。
- content 是那件东西**上面写着、记着的东西本身**，比如纸上写了
  什么、账上记了哪几笔、图上标了哪几处。拿到它的人读到的就是这一栏,所以要有具体、实质内容、写实写全,不写「写了一
  封信，一张图纸」「按时间地点逐条记下所见，但没有实质信息」这种说明概括或者介绍。
  · 它长什么样归 description,不归这里;description 是旁人一眼看得到的,上面写了什么不能在
    description 里写、也不能概括——那等于把内容亮给了所有人。
    没有写字记事的东西(一把刀、一张椅子)内容一律留空。所以这一栏可以选择填写，并不是必填。
  · 改写原有那件的内容时,给出它此刻的**全部**内容,不是只写新添的那一笔。
  · 如果内容涉及到时间，那时间尽量给具体的时间点，比如6月1日，六月初六等等，不要给相对时间，比如明日，三日后。注意时间不能乱填，需要结合当前时间确定，同时，格式与当前时间一致。
- outcome:把 fact 用第三人称说出来,点名我的名字;只说我做了什么、做出了什么,不复述其中的具体内容。
- observation:换一个站在旁边的人来看,他看得见的是什么。他看得见我在忙什么、忙成没忙成,
  但只看得到表面、看不到内在;什么都看不见的就不写,别把 outcome 抄一遍。
- why:没做成时,第三人称一句缘由,**只写旁人可陈述的**(所需的东西不齐、时候不够、有人在跟前);
  我的懊丧、恐惧、焦虑这些只属于我自己的,不写进这一栏。做成了给空字符串。

【输出】
严格输出以下 JSON,不要任何多余内容。**先写 fact**——成败与留下的东西都要从我实际做出来的
里头读,不是先拍板再补一句话去圆;后面三句叙述又要把那件东西说进去。
**success 与 carried 写裸布尔 true / false,entity 写裸整数,都不要加引号。**
{"fact": "第一人称、≤150字", "success": true或false, "changes": [{"entity": 序号或0, "name": "≤14字；不改名则空字符串", "state": "≤12字；没变则空字符串", "description": "≤24字；没变则空字符串", "content": "≤80字；没变或没有内容则空字符串", "carried": true或false}], "outcome": "第三人称、点名我的名字、≤40字、**单行不换行**", "observation": "第三人称、点名我的名字、≤30字、**单行不换行**", "why": "第三人称、≤20字；做成了给空字符串"}
"""
        # §3: the thing to judge and the yardstick go last, right before the output instructions.
        scene_part = f"\n【我此刻所处的现场】\n{scene}\n" if scene else ""
        expected_part = (
            f"\n【我动手前期望的】\n{expected_outcome}\n" if expected_outcome else ""
        )
        memory_lines = [
            line for line in render_memory_lines(
                recalled or [], now_step=now_step, seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day,
            ) if line
        ]
        memory_part = (
            f"\n【我想起的、可能与这件事有关的旧事（{MEMORY_ORDER_HINT}）】\n"
            + "\n".join(f"- {line}" for line in memory_lines) + "\n"
            if memory_lines else ""
        )
        doer_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        doer_vitality = vitality_line(
            agent.personality.state.vitality, voice=SituationVoice.FIRST, lead="",
        )
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{doer_condition}
{doer_vitality}
{memory_part}{scene_part}
【我动手前打算做的】
{purpose}

【这件事上花掉的时间】
{duration_label}（时间地点见开头，不必复述）
{expected_part}
依上面说定的 JSON 格式给出判断，只输出 JSON、不写任何多余内容。"""
        with annotate_call(
            given_facts=GivenFacts()  # same source as user above
            .add("此刻何时何地", situation_header)
            .add("我的处境", doer_condition)
            .add("我的生命值", doer_vitality)
            .add("我想起的旧事", memory_lines)
            .add("现场", scene)
            .add("我动手前打算做的", purpose)
            .add("花掉的时间", duration_label)
            .add("我动手前期望的", expected_outcome),
            # So review can tell what each change's entity pointed at.
            item_candidates={
                str(i): (item.name or "某物") for i, item in enumerate(in_reach, 1)
            },
            action_owner=agent.personality.soul.name,
        ):
            data = await complete_json(
                self._llm.complete, system, user,
                temperature=0.7,
                # fact (≤150 chars ≈ 225 tok) + success (~3)
                # + changes ≤3 × (entity ~3 + name ≤14 ≈21 + state ≤12 ≈18 + description ≤24 ≈36
                #   + content ≤80 ≈120 + carried ~3 + structure for 6 fields ~30) ≈ 3 × 231 = 693
                #   + the array's ~10
                # + outcome (≤40 ≈ 60) + observation (≤30 ≈ 45) + why (≤20 ≈ 30)
                # + structure for 6 fields (~30); est. ~1096 tok.
                max_tokens=output_budget(1096),
                failure_event="work_outcome_failed",
                log_extra={"agent_id": agent.agent_id, "step": now_step},
                # The verdict and the doer's own record of it; outcome and observation have
                # neutral stand-ins that claim no more than the verdict.
                required=("success", "fact"),
            )
        if data is None:
            return None
        succeeded = coerce_bool(data.get("success"), False)
        fact = str(data["fact"]).strip()
        # One line: the renderer treats a newline as a transcript separator and cuts there.
        outcome = " ".join(str(data.get("outcome", "")).split()).strip()
        # Never fall back to outcome here (it has the product).
        observation = " ".join(str(data.get("observation", "")).split()).strip()
        why = str(data.get("why", "")).strip() if not succeeded else ""
        return _WorkVerdict(
            succeeded=succeeded, fact=fact, outcome=outcome,
            # Unsliced: the prompt caps it.
            observation=observation, failure_reason=why,
            # Independent of succeeded (see _WorkVerdict).
            changes=_parsed_changes(data.get("changes"), in_reach, doer_id=agent.agent_id, step=now_step),
        )


def _parsed_changes(
    raw: Any, in_reach: "tuple[WorldEntity, ...]", *, doer_id: str, step: int,
) -> tuple[_WorkChange, ...]:
    """The answer's entries, each bound to the thing in reach it names. The only place an index
    is mapped back to a thing; an index naming nothing in reach drops its entry whole."""
    by_id = {item.entity_id: item for item in in_reach}
    ref = IndexedRef(item.entity_id for item in in_reach)
    parsed: list[_WorkChange] = []
    for entry in coerce_mapping_list(raw):
        index = coerce_str(entry.get("entity")).strip()
        if index == "0":
            entity = None
        else:
            resolved = ref.resolve([entry.get("entity")])
            if not resolved:
                logger.warning("work_change_rejected", extra={"reason": "entity_not_in_reach", "step": step})
                continue
            entity = by_id[resolved[0]]
        parsed.append(_WorkChange(
            entity=entity,
            name=coerce_str(entry.get("name")).strip(),
            state=coerce_str(entry.get("state")).strip(),
            description=coerce_str(entry.get("description")).strip(),
            content=coerce_str(entry.get("content")).strip(),
            # Left out, a thing stays where it was; a thing made defaults to the private tier.
            carried=coerce_bool(
                entry.get("carried"), entity is None or entity.owner_id == doer_id,
            ),
        ))
    return tuple(parsed)


def _world_changes(
    entries: tuple[_WorkChange, ...],
    *,
    doer_id: str,
    doer_name: str,
    here: str,
    location: str,
    step: int,
) -> tuple[list[EntityStateChange], list[EntitySpawn]]:
    """Turn the session's entries into the changes the world takes.

    A thing in reach takes its new fields; one in hand, set down, lands ``here`` in sight, and one
    lying here, carried, is picked up (a fixture can't be: it keeps its changes and stays). One
    thing settles once. A value copied back unchanged is no change, and an entry left with nothing to
    change is dropped. A thing made is private while carried (public would leak it through
    others' reachable lists) and public once left; always the default "item": WORK makes no
    landmarks.
    """
    changes: list[EntityStateChange] = []
    spawns: list[EntitySpawn] = []
    settled: set[str] = set()

    def _reject(reason: str) -> None:
        logger.warning("work_change_rejected", extra={"reason": reason, "step": step})

    for entry in entries:
        entity = entry.entity
        if entity is None:
            if not entry.name:
                _reject("made_without_a_name")
                continue
            spawns.append(EntitySpawn(
                name=entry.name,
                description=entry.description,
                content=entry.content,
                state=entry.state or DEFAULT_ENTITY_STATE,
                holder_id=doer_id if entry.carried else None,
                is_public=not entry.carried,
                # Announce something set down, so others learn it just appeared.
                perception=(
                    "" if entry.carried
                    else scene_line(location, f"{doer_name}把{entry.name}留在了此处。")
                ),
            ))
            continue
        if entity.entity_id in settled:
            _reject("entity_already_settled")
            continue
        name = "" if entry.name == entity.name else entry.name
        state = "" if entry.state == entity.state else entry.state
        description = "" if entry.description == entity.description else entry.description
        content = "" if entry.content == entity.content else entry.content
        held = entity.owner_id == doer_id
        set_down = held and not entry.carried
        picked_up = not held and entry.carried and entity.is_takeable
        if not held and entry.carried and not entity.is_takeable:
            # Only the carry is dropped; its other changes still land.
            logger.warning("work_carry_ignored", extra={"reason": "entity_not_takeable", "step": step})
        if not (name or state or description or content or set_down or picked_up):
            _reject("nothing_changed")
            continue
        settled.add(entity.entity_id)
        # Others see him put it down or take it, not what's new on the page.
        moved = (
            f"{doer_name}把{entity.name}放在了此处。" if set_down
            else f"{doer_name}拿起了{entity.name}。" if picked_up
            else ""
        )
        changes.append(EntityStateChange(
            entity_id=entity.entity_id,
            new_name=name,
            new_state=state,
            new_description=description,
            new_content=content,
            owner_id=doer_id if picked_up else None,
            location_id=here if set_down else None,
            perception=scene_line(location, moved) if moved else "",
        ))
    return changes, spawns
