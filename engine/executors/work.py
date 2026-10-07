"""Work action executor for WORK actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from core.context import GivenFacts, annotate_call
from core.interfaces.action import (
    ActionTarget, ActionResult, ActionType, AgentAction, EntitySpawn, EntityStateChange,
)
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import (
    IndexedRef, LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget,
)
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    condition_line,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    SituationVoice,
    render_situation_header,
    vitality_line,
)
from core.duration import describe_duration
from core.interfaces.execution import TickResult
from engine.executors.base import ActionExecutionState, ActionExecutor
from engine.narration import (
    ensure_actor_named, format_interrupt_reason_3p, format_interrupt_thought, observed_here,
    scene_line,
)
from engine.scene import SceneVisibility, assemble_scene_context, observe_location, situation_for

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem
    from world.models import WorldEntity

logger = get_logger(__name__)


@dataclass(frozen=True)
class _WorkVerdict:
    """Everything one WORK self-assessment produces, one piece per channel, from one LLM call.

    "What part still isn't done" lives in 1p ``fact``, not ``failure_reason`` (3p, empty on
    success): "mostly done, one part missing" is the common case, and the residue judge reads it
    from memory to set the next goal, so it must survive success.

    ``observation`` is not a copy of ``outcome``: outcome includes the product, which onlookers
    must not see.

    A product is stated explicitly, never inferred from ``succeeded``: thinking something through
    succeeds with nothing made, and a half-made thing still exists. Repeated polishing changes
    the thing in hand via ``updated_*`` instead of piling up near-duplicates; both may be given
    (a chair carved from a block).

    ``product_carried`` is a bool, not separate slots: a free-text product has no candidate list
    to bind to.
    """

    succeeded: bool
    fact: str
    outcome: str
    observation: str
    failure_reason: str
    updated_id: str = ""        # the item in hand this session changed; empty = none
    updated_name: str = ""      # its current name; empty = not renamed
    updated_state: str = ""     # its current state; empty = unchanged
    updated_desc: str = ""      # how it looks now; empty = unchanged
    updated_content: str = ""   # everything it now contains; empty = unchanged
    product_name: str = ""
    product_desc: str = ""
    product_content: str = ""
    product_carried: bool = True


class WorkExecutor(ActionExecutor):
    """Executor for WORK actions: finishing something of one's own, with no external object
    and no relation delta.

    Judged in character, first person: with no opponent, it's an honest self-assessment. It
    must rest on what the actor perceives (things at hand, who's present, his
    ``expected_outcome``): from persona alone the verdict is always "nothing came of it" and the
    goal is retried forever.
    """

    def __init__(self, llm_router: LLMRouter, directory: WorldDirectory, seconds_per_step: int = 3600) -> None:
        self._llm = llm_router
        self._directory = directory
        self._seconds_per_step = seconds_per_step

    def _outcome(self, agent_id: str, purpose: str, location: str, *, succeeded: bool) -> str:
        """Fallback 3p outcome when the judge gives none. Never quotes ``purpose``: it's a
        first-person sentence and breaks a third-person line."""
        actor_name = self._directory.agent_name(agent_id)
        body = (
            f"{actor_name}做完了手上的事。" if succeeded
            else f"{actor_name}忙活了一阵，终究没能做成手上的事。"
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
            estimated_steps=action.estimated_steps,
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
        actor_name = self._directory.agent_name(state.initiator_id)
        location = observe_location(environment, state.initiator_id)
        elapsed = state.estimated_steps - state.remaining_steps
        if elapsed <= 0:
            body = f"{actor_name}着手做手上的事。"                  # same form as start
        else:
            body = f"{actor_name}手上的事仍在进行，已投入{describe_duration(elapsed, self._seconds_per_step)}。"
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
            # Numbered for updated_index. Only things in hand: altering something on the ground
            # takes PHYSICAL first (half of the WORK/PHYSICAL boundary).
            split_own_entities=True,
        )
        gen_result = await self._generate_outcome(
            agent=agents.get(state.initiator_id),
            purpose=state.purpose,
            in_hand=scene.own_entities,
            now_step=step,
            duration_label=describe_duration(state.estimated_steps, self._seconds_per_step),
            location=location,
            situation_header=situation_header,
            scene=scene.text,
            expected_outcome=state.expected_outcome,
        )
        if gen_result is None:
            return [self._adjudication_failed_result(
                purpose=state.purpose, step=step, agent_id=state.initiator_id,
                estimated_steps=state.estimated_steps, expected_outcome=state.expected_outcome,
                location=location, target=state.target,
            )]
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.WORK,
            action_description=state.purpose,
            target=state.target,
            estimated_steps=state.estimated_steps,
        )
        # A missing observation falls back to the template, never to outcome (it has the product).
        actor_name = self._directory.agent_name(state.initiator_id)
        fallback_3p = self._outcome(
            state.initiator_id, state.purpose, location, succeeded=gen_result.succeeded,
        )
        outcome = (
            scene_line(location, ensure_actor_named(gen_result.outcome, actor_name))
            if gen_result.outcome else fallback_3p
        )
        observation = (
            scene_line(location, ensure_actor_named(gen_result.observation, actor_name))
            if gen_result.observation else fallback_3p
        )
        # Carried products are private (public would leak them via others' reachable lists);
        # left behind, they're public. Always the default "item": WORK makes no landmarks.
        spawns: list[EntitySpawn] = []
        changes: list[EntityStateChange] = []
        touched = (
            gen_result.updated_name or gen_result.updated_state or gen_result.updated_desc
            or gen_result.updated_content
        )
        if gen_result.updated_id and touched:
            changes.append(EntityStateChange(
                entity_id=gen_result.updated_id,
                new_name=gen_result.updated_name,
                new_state=gen_result.updated_state,
                new_description=gen_result.updated_desc,
                new_content=gen_result.updated_content,
                # No perception: others see him at the desk, not what's new on the page.
            ))
        if gen_result.product_name:
            carried = gen_result.product_carried
            spawns.append(EntitySpawn(
                name=gen_result.product_name,
                description=gen_result.product_desc,
                content=gen_result.product_content,
                holder_id=state.initiator_id if carried else None,
                is_public=not carried,
                # Announce something set down, so others learn it just appeared.
                perception=(
                    "" if carried
                    else scene_line(location, f"{actor_name}把{gen_result.product_name}留在了此处。")
                ),
            ))
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

    async def interrupt(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem | None" = None,
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        agent = agents.get(state.initiator_id)
        thought_part = format_interrupt_thought(thought)
        location = observe_location(environment, state.initiator_id) if environment is not None else "此处"
        place = f"在{location}"
        # Only third-person cause may name what interrupted it (see base.interrupt).
        cut = f"因{cause}而中断" if cause else "被打断"

        fallback = (
            f"{place}做「{state.purpose}」做了{elapsed_label}便{cut}，没能收尾{thought_part}"
            if elapsed > 0
            else f"刚着手「{state.purpose}」就{cut}{thought_part}"
        )
        factual = fallback
        if agent is not None:
            situation_header = (
                render_situation_header(situation_for(environment, state.initiator_id),
                                        voice=SituationVoice.FIRST)
                if environment is not None else ""
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
            actor_condition = condition_line(
                agent.personality.state.condition, voice=SituationVoice.FIRST,
                now_step=step, seconds_per_step=self._seconds_per_step,
        )
            actor_vitality = vitality_line(
                agent.personality.state.vitality, voice=SituationVoice.FIRST, lead="",
            )
            user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}
{actor_vitality}

【刚刚发生的事】
我做「{state.purpose}」，已投入{elapsed_label}，突然{cut}{thought_part}（时间地点见开头）。

依上面说定的 JSON 格式记下这一场，只输出 JSON、不写任何多余内容。"""
            try:
                with annotate_call(
                    given_facts=GivenFacts()  # same source as user above
                    .add("此刻何时何地", situation_header)
                    .add("我的处境", actor_condition)
                    .add("我的体力", actor_vitality)
                    .add("刚刚发生的事",
                         f"我做「{state.purpose}」，已投入{elapsed_label}，突然{cut}{thought_part}"),
                    action_owner=agent.personality.soul.name,
                ):
                    resp = await self._llm.complete(
                        LLMScene.AGENT_ACTION_NARRATION,
                        [
                            LLMMessage(role="system", content=system),
                            LLMMessage(role="user", content=user),
                        ],
                        temperature=0.7,
                        max_tokens=output_budget(69),  # fact ≤36 chars; est. ~69 tok
                    json_mode=True,
                    )
                factual = str(extract_json(resp.content).get("fact", "")).strip() or fallback
            except Exception as exc:
                logger.warning("work_interrupt_failed", extra={"agent_id": agent.agent_id, "error": str(exc)})

        actor_name = self._directory.agent_name(state.initiator_id)
        gist = scene_line(location, (
            f"{actor_name}做手上的事做了{elapsed_label}，{cut}。"
            if elapsed > 0
            else f"{actor_name}刚着手做手上的事就{cut}。"
        ))
        outcome = gist + format_interrupt_reason_3p(actor_name, thought)
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

    def _adjudication_failed_result(
        self, *, purpose: str, step: int, agent_id: str, estimated_steps: int,
        expected_outcome: str, location: str, target: "ActionTarget",
    ) -> ActionResult:
        """Null step when adjudication can't happen: no memory, no claimed result."""
        stub = AgentAction(
            agent_id=agent_id,
            step=step,
            action_type=ActionType.WORK,
            action_description=purpose,
            target=target,
            estimated_steps=estimated_steps,
        )
        return ActionResult(
            action=stub,
            expected_outcome=expected_outcome or purpose,
            # Never perceived; only a trace/snapshot marker.
            outcome=scene_line(location, f"{self._directory.agent_name(agent_id)}着手做手上的事，一时未能确知是否做成。"),
            succeeded=False,
            adjudication_failed=True,
        )

    async def _generate_outcome(
        self,
        *,
        agent: "Agent | None",
        purpose: str,
        duration_label: str,
        now_step: int = 0,
        location: str = "",
        situation_header: str = "",
        scene: str = "",
        expected_outcome: str = "",
        in_hand: "tuple[WorldEntity, ...]" = (),  # index → item, same list as the scene line
    ) -> "_WorkVerdict | None":
        """Adjudicate a finished WORK (1p fact, 3p outcome, why) in one call. Returns ``None``
        when adjudication can't happen, so the caller fabricates nothing.

        All three inputs are needed: ``scene`` (what's at hand, who's present),
        ``expected_outcome`` (a reference, not a gate: the decision LLM often writes downstream
        consequences into it), and persona/energy/emotion.

        Accepted trade-off: ``scene`` is a snapshot at completion, so midway comings and goings
        are invisible to the judge.
        """
        if agent is None:
            logger.warning("work_self_judge_no_actor", extra={"purpose": purpose})
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
- 我出发时想要的:只作参照,用来衡量我实际做出来的东西。

【我不能做的】
- **不能把那句打算当成已经发生的事**:它是我动手前的打算,这段工夫里究竟如何,正是我要判的。
  更不能把它换个说法抄一遍,充作我实际做出来的东西。
- **不能把过程当成产出**:比如反复斟酌而始终拿不出任何实在的东西,那就是没成。
- **不能把我想要的照抄成已经发生的**。
- 不能替同处一地的人编造他们并没有做出的举动。

【我记事的规矩】
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + "\n" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【我要写的】
- fact:第一人称,我自己记下的这一场——做出了什么、成果是什么，如果有新东西产生也需要在这里点明;若有没了结的部分或压根没做成的,
  一并说清缘由。
- success:上面第 2 步的判断。
- updated_index / updated_name / updated_state / updated_desc / updated_content:这一场我
  **动到了手上哪一件东西**,它又变成了什么样。先在所给现场「我手上的东西」那一行里指出是哪一件
  (填它的序号),再写它经这一场之后的名字、所属状态、现在的样子
  与现在的内容（如果有）;哪一样没变就留空哪一栏。
  **改名是换了身份,不是变了样**:它已经不再是原来那样东西了才改名;变旧、变短、变脏、
  写满了字,名字照旧——那个词是别人记忆里认得它的凭据,换掉它,那些记忆就指向了一件不
  存在的东西。
  **手上的东西一件也没动到,updated_index 就填 0**,后面几栏一并留空——我只是拿着它比对了
  一遍、翻看了一遍,它还是原样;手上空无一物时同样填 0。
- product_name / product_desc / product_content / product_carried:这一场**新做出来的**
  东西——有形体、摸得着的,写它叫什么、是个什么样子、承载着什么内容（如果有）;product_carried 说它此刻
  在哪儿:我带在身上 true,搁下没带走 false。
  **它与上面那件是两回事,可以同时有**:拿一块料削出一把椅子,料变短了(上面那几栏),
  椅子出现了(这几栏)。
  以下情形一律**留空**:
  · 这段工夫只在心里过了一遍,没有任何东西成形;
  · 我只是拿起来看看、核对一遍,什么也没添;
  · 我动的就是手上已有的那一件——它变了样该填上面那几栏,不是在这里另报一件出来;
  · 如果想要东西已经在世界中存在，那就不要再重复生成了，也不要生成一个相似的东西，这样没有意义；
  「进展」「头绪」「决心」「把握」这类**没有形体的东西一概不算**,写进来就是错。
  如果实在没有新东西产生那就留空，留空比捏造更好。
  **东西有没有留下,和事情成没成是两回事**:一件做到一半、离我想要的还差得远的东西,
  是没达成的意图,却是实实在在的一件东西——第一次做出它时照写不误。
- 内容(updated_content / product_content)是那件东西**上面写着、记着的东西本身**，比如纸上写了
  什么、账上记了哪几笔、图上标了哪几处。拿到它的人读到的就是这一栏,所以写实写全,不写「写了一
  封信，一张图纸」这种概括。
  · 它长什么样归 desc,不归这里;desc 是旁人一眼看得到的,上面写了什么不能在 desc 里写、
    也不能概括——那等于把内容亮给了所有人。
    没有写字记事的东西(一把刀、一张椅子)内容一律留空。所以这两个字段可以选择填写，并不是必填。
  · 改写手上那件的内容时,给出它此刻的**全部**内容,不是只写新添的那一笔。
  · 如果内容涉及到时间，那时间尽量给具体的时间点，比如6月1日，六月初六等等，不要给相对时间，比如明日，三日后。注意时间不能乱填，需要结合当前时间确定，同时，格式与当前时间一致。
- outcome:把 fact 用第三人称说出来,点名我的名字。
- observation:换一个站在旁边的人来看,他看得见的是什么。他看得见我在忙什么、忙成没忙成,
  但只看得到表面、看不到内在;什么都看不见的就不写,别把 outcome 抄一遍。
- why:没做成时,第三人称一句缘由,**只写旁人可陈述的**(所需的东西不齐、时候不够、有人在跟前);
  我的懊丧、恐惧、焦虑这些只属于我自己的,不写进这一栏。做成了给空字符串。

【输出】
严格输出以下 JSON,不要任何多余内容。**先写 fact**——成败与留下的东西都要从我实际做出来的
里头读,不是先拍板再补一句话去圆;后面三句叙述又要把那件东西说进去。
**success 与 product_carried 写裸布尔 true / false,updated_index 写裸整数,都不要加引号。**
{"fact": "第一人称、≤68字", "success": true或false, "updated_index": 整数（0=手上的东西一件也没动到）, "updated_name": "≤14字；不改名则空字符串", "updated_state": "≤12字；没变则空字符串", "updated_desc": "≤24字；没变则空字符串", "updated_content": "≤80字；没变或没有内容则空字符串", "product_name": "≤14字；没有则空字符串", "product_desc": "≤24字；没有则空字符串", "product_content": "≤80字；没有内容则空字符串", "product_carried": true或false, "outcome": "第三人称、点名我的名字、≤40字、**单行不换行**", "observation": "第三人称、点名我的名字、≤30字、**单行不换行**", "why": "第三人称、≤20字；做成了给空字符串"}
"""
        # §3: the thing to judge and the yardstick go last, right before the output instructions.
        scene_part = f"\n【我此刻所处的现场】\n{scene}\n" if scene else ""
        expected_part = (
            f"\n【我出发时想要的】\n{expected_outcome}\n" if expected_outcome else ""
        )
        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        actor_vitality = vitality_line(
            agent.personality.state.vitality, voice=SituationVoice.FIRST, lead="",
        )
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}
{actor_vitality}
{scene_part}
【我动手前打算做的】
{purpose}

【这件事上花掉的时间】
{duration_label}（时间地点见开头，不必复述）
{expected_part}
依上面说定的 JSON 格式给出判断，只输出 JSON、不写任何多余内容。"""
        try:
            with annotate_call(
                given_facts=GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("我的体力", actor_vitality)
                .add("现场", scene)
                .add("我动手前打算做的", purpose)
                .add("花掉的时间", duration_label)
                .add("我出发时想要的", expected_outcome),
                # So review can tell what updated_index pointed at.
                item_candidates={
                    str(i): (item.name or "某物") for i, item in enumerate(in_hand, 1)
                },
                action_owner=agent.personality.soul.name,
            ):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    # fact (≤68 chars ≈ 102 tok) + success (~3) + updated_index (~3)
                    # + updated_name (≤14 ≈ 21) + updated_state (≤12 ≈ 18) + updated_desc (≤24 ≈ 36)
                    # + updated_content (≤80 ≈ 120) + product_name (≤14 ≈ 21)
                    # + product_desc (≤24 ≈ 36) + product_content (≤80 ≈ 120)
                    # + product_carried (~3) + outcome (≤40 ≈ 60) + observation (≤30 ≈ 45)
                    # + why (≤20 ≈ 30) + structure for 14 fields (~70); est. ~688 tok.
                    max_tokens=output_budget(688),
                json_mode=True,
                )
            data = extract_json(response.content)
            succeeded = coerce_bool(data.get("success"), True)
            place = f"在{location}" if location else ""
            fact = str(data.get("fact", "")).strip() or (
                f"{place}花了{duration_label}做「{purpose}」，做成了。"
                if succeeded
                else f"{place}花了{duration_label}做「{purpose}」，终究没能做成。"
            )
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
                product_name=str(data.get("product_name", "")).strip(),
                product_desc=str(data.get("product_desc", "")).strip(),
                product_content=str(data.get("product_content", "")).strip(),
                # Default carried: the private tier.
                product_carried=coerce_bool(data.get("product_carried"), True),
                # The only place the index is mapped back to an id.
                updated_id=next(iter(IndexedRef(
                    [item.entity_id for item in in_hand]
                ).resolve([data.get("updated_index")])), ""),
                updated_name=str(data.get("updated_name", "")).strip(),
                updated_state=str(data.get("updated_state", "")).strip(),
                updated_desc=str(data.get("updated_desc", "")).strip(),
                updated_content=str(data.get("updated_content", "")).strip(),
            )
        except Exception as exc:
            logger.warning("work_outcome_failed", extra={"agent_id": agent.agent_id, "error": str(exc)})
            return None
