"""Covert action executor for COVERT actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.interfaces.action import ActionTarget, ActionResult, ActionType, AgentAction
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.interfaces.action import Observed
from core.logging import get_logger
from core.prompts import (
    CLOSED_WORLD_FACT_RULE, MEMORY_ORDER_HINT, SituationVoice, condition_line, recency_prefix,
)
from engine.environment import SALIENT_AMBIENT_STRENGTH
from core.duration import describe_duration
from core.context import GivenFacts, annotate_call
from core.interfaces.execution import TickResult
from engine.executors.base import (
    ActionExecutionState,
    ActionExecutor,
)
from engine.narration import (
    SAME_PLACE_VERDICT_RULE, ensure_actor_named, format_interrupt_reason_3p,
    format_interrupt_thought, observed_here, scene_line,
)
from engine.scene import (
    IDLE_BYSTANDER_VERDICT_RULE, SceneVisibility, assemble_scene_context, observe_location,
)

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

class CovertExecutor(ActionExecutor):
    """Executor for COVERT actions.

    Onlookers perceive nothing unless the act is ``detected``. COVERT doesn't see more; it sees
    the other layer: an action's full ``outcome`` instead of the onlooker string, a thing's
    ``content`` instead of its appearance. It never changes the world (that's PHYSICAL's);
    what it gains lands only in the actor's memory.

    achieved (``succeeded``) and ``detected`` are judged independently. achieved means "learned
    something he couldn't otherwise get", not "finished lurking".

    A functional third-party judge picks what he caught out of ``_happenings``; it must not
    generate intelligence, since inventions get embedded and recalled. If adjudication can't
    happen, the result is an ``adjudication_failed`` null step.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        directory: WorldDirectory,
        seconds_per_step: int = 3600,
        world_start_second_of_day: int = 0,
    ) -> None:
        self._llm = llm_router
        self._directory = directory
        self._seconds_per_step = seconds_per_step
        self._world_start_second_of_day = world_start_second_of_day

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        description = action.action_description or "covert action"
        location = observe_location(environment, action.agent_id)
        return ActionExecutionState.create(
            action_type=ActionType.COVERT,
            initiator_id=action.agent_id,
            participant_ids=[action.agent_id],
            purpose=description,
            started_step=step,
            estimated_steps=action.estimated_steps,
            opening_outcome=scene_line(
                location, f"{self._directory.agent_name(action.agent_id)}悄悄着手做「{description}」。"
            ),
            # Onlookers perceive something only on complete()'s detection verdict.
            opening_observations=[],
            target=action.target,
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
        elapsed = state.estimated_steps - state.remaining_steps
        duration_part = (
            f"，已持续{describe_duration(elapsed, self._seconds_per_step)}" if elapsed > 0 else ""
        )
        actor_name = self._directory.agent_name(state.initiator_id)
        location = observe_location(environment, state.initiator_id)
        outcome = scene_line(location, f"{actor_name}正秘密进行「{state.purpose}」{duration_part}。")
        # No observation: exposure is decided only by complete()'s detection verdict.
        return [TickResult(agent_id=state.initiator_id, outcome=outcome, observations=[])]

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
        agent = agents.get(state.initiator_id)
        location = observe_location(environment, state.initiator_id)
        verdict = await self._judge(
            agent,
            purpose=state.purpose,
            duration_label=describe_duration(state.estimated_steps, self._seconds_per_step),
            expected_outcome=state.expected_outcome,
            happenings=self._happenings(environment, state),
            scene=assemble_scene_context(
                state.initiator_id, environment=environment, directory=self._directory,
                agents=agents, with_background=True, visibility=SceneVisibility.GOD,  # functional judge
                # Content would be passed into fact as something that happened.
                reveal_intents=False,
                reveal_all_content=True,
            ).text,
            now_step=step,
        )
        if verdict is None:
            return [self._adjudication_failed_result(
                purpose=state.purpose, step=step, agent_id=state.initiator_id,
                estimated_steps=state.estimated_steps, expected_outcome=state.expected_outcome,
                location=location, target=state.target,
            )]
        achieved, detected, outcome, fact, why = verdict
        # Also the exposure observation's text, so one fix covers both channels.
        named_outcome = ensure_actor_named(outcome, self._directory.agent_name(state.initiator_id))
        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.COVERT,
            action_description=state.purpose,
            target=state.target,
            estimated_steps=state.estimated_steps,
        )
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=scene_line(location, named_outcome),  # 3p full (god-view web/snapshot + actor's ledger)
            observations=self._exposure_observations(environment, state.initiator_id, named_outcome, detected),
            succeeded=achieved,
            failure_reason=why,         # 3p authoritative "why it didn't work"
            factual_memory=fact,        # 1p → the actor's own memory channel
            detected=detected,          # structured exposure signal (runtime propagation reads it)
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
        agent = agents.get(state.initiator_id)
        location = observe_location(environment, state.initiator_id) if environment is not None else "此处"
        thought_part = format_interrupt_thought(thought)
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        planned_label = describe_duration(state.estimated_steps, self._seconds_per_step)

        fact = ""
        outcome = ""
        detected = False
        detection_known = False
        actor_name = self._directory.agent_name(state.initiator_id)
        if agent is not None:
            # Only third-person cause goes in; the interrupt signal is first-person prompt input
            # and would land in memory via fact.
            cut = f"因{cause}被迫中止" if cause else "被迫中止"
            interrupt_part = (
                f"\n- 中途变故：行动进行了{elapsed_label}（原计划{planned_label}）时，{cut}{thought_part}"
                if elapsed > 0
                else f"\n- 中途变故：行动刚开始就{cut}{thought_part}"
            )
            scene = (
                assemble_scene_context(
                    state.initiator_id, environment=environment, directory=self._directory,
                    agents=agents, with_background=True, visibility=SceneVisibility.GOD,  # functional judge
                # Content would be passed into fact as something that happened.
                reveal_intents=False,
                    reveal_all_content=True,
                ).text
                if environment is not None
                else ""
            )
            # achieved is fixed to False by code; only detected/fact are used. Still pass the
            # material: what he caught before being pulled away is real.
            verdict = await self._judge(
                agent,
                purpose=state.purpose,
                duration_label=planned_label,
                expected_outcome=state.expected_outcome,
                happenings=self._happenings(environment, state),
                scene=scene,
                now_step=step,
                interrupt_part=interrupt_part,
            )
            if verdict is not None:
                _achieved, detected, outcome, fact, _why = verdict
                detection_known = True
            # No verdict: still record the interruption (a real event), but claim no detection.
        # Same if the agent can't be found.

        if not fact:
            if detection_known:
                suffix = "，被人察觉了" if detected else "，幸未被察觉"
            else:
                suffix = ""  # adjudication failed — record the interruption, claim no detection outcome
            fact = (
                f"秘密行动「{state.purpose}」进行了{elapsed_label}后被迫中止{suffix}"
                if elapsed > 0
                else f"秘密行动「{state.purpose}」刚开始就被迫中止{suffix}"
            )
        if not outcome:
            suffix = ("，被人察觉了" if detected else "，幸未被察觉") if detection_known else ""
            outcome = (
                f"{actor_name}的秘密行动「{state.purpose}」进行了{elapsed_label}后被迫中止{suffix}"
                if elapsed > 0
                else f"{actor_name}的秘密行动「{state.purpose}」刚开始就被迫中止{suffix}"
            )


        stub = AgentAction(
            agent_id=state.initiator_id,
            step=step,
            action_type=ActionType.COVERT,
            action_description=state.purpose,
            target=state.target,
        )
        # "Why it stopped" goes only on the outcome, never into the exposure string, or his
        # thoughts would leak to onlookers.
        return [ActionResult(
            action=stub,
            expected_outcome=state.expected_outcome or state.purpose,
            outcome=scene_line(location, outcome + format_interrupt_reason_3p(actor_name, thought)),
            gist=scene_line(location, outcome),
            succeeded=False,
            # No failure_reason: phase="interrupt" already says it.
            factual_memory=fact,        # 1p → the actor's own memory channel
            detected=detected,          # structured exposure signal
        )]

    @staticmethod
    def _happenings(
        environment: "EnvironmentSystem | None", state: ActionExecutionState,
    ) -> list[tuple[int, str]]:
        """What happened here while he lurked: COVERT's only source of intelligence (without it
        the judge can only invent findings).

        The window starts at ``started_step - 1`` because all perception lags one beat; reading
        "what is happening right now" would be cheating.
        """
        if environment is None:
            return []
        return environment.recent_happenings(
            # The id, not observe_location's narrative name.
            environment.get_body_location(state.initiator_id),
            since_step=state.started_step - 1,
            # What you did yourself isn't something you found out.
            exclude_ids=tuple(state.participant_ids) or (state.initiator_id,),
        )

    @staticmethod
    def _exposure_observations(
        environment: "EnvironmentSystem", actor_id: str, body: str, detected: bool,
    ) -> list[Observed]:
        """What onlookers see on exposure (prefixed, social-signal strength); nothing if
        undetected. Authored here, not added downstream by the runtime."""
        if not detected or not body:
            return []
        return observed_here(
            environment, actor_id, f"[秘密行动暴露] {scene_line(observe_location(environment, actor_id), body)}",
            strength=SALIENT_AMBIENT_STRENGTH,
        )

    async def _judge(
        self,
        agent: "Agent | None",
        *,
        purpose: str,
        duration_label: str,
        expected_outcome: str,
        scene: str,
        happenings: "list[tuple[int, str]] | None" = None,
        now_step: int = 0,
        interrupt_part: str = "",
    ) -> "tuple[bool, bool, str, str, str] | None":
        """Neutral third-party adjudication. Returns (achieved, detected, outcome, fact, why).

        ``outcome`` is third person (also the exposure string); ``fact`` is first person, for
        the actor's memory. Returns ``None`` when adjudication can't happen (LLM failure,
        missing actor), so the caller fabricates nothing.
        """
        if agent is None:
            logger.warning("covert_judge_no_actor", extra={"purpose": purpose})
            return None

        expected_part = f"\n- 期望结果：{expected_outcome}" if expected_outcome else ""
        # Memory-shaped input, so it follows the memory-injection contract (ordering,
        # MEMORY_ORDER_HINT, recency_prefix). Say it even when empty: otherwise the judge
        # invents findings to match expected_outcome.
        happenings_block = "\n".join(
            "- " + recency_prefix(
                now_step=now_step, ref_step=at_step,
                seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day,
            ) + text
            for at_step, text in happenings
        ) if happenings else "（这段时间此处没有任何事发生——他无物可探）"
        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.THIRD,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )

        # Field order (§5b): achieved is read off fact, so it comes after it; before, the model
        # would commit to a boolean and invent intelligence to justify it. outcome before fact
        # keeps the findings out of context while the traces are written.
        system = """\
【裁决任务】
你是中立的世界裁决者。裁定一次秘密行动的两件事：他探到了什么（achieved），行迹是否被察觉（detected）。
下面会给你「此处先后发生的事」。**你的活是判断他这一趟接住了其中哪些**，不是替他编一份情报——那里面没有的，一个字都不许添。

【裁决原则】
- 在场不等于看见：判暴露必须存在可信的感知通道（视线、声响、留下的痕迹）。
- 对于能拿到什么信息需要符合物理常识和空间常识，比如在吵杂的环境或者说话声音本来就很小，那就有可能听不到内容，或者听到部分内容。
- 下面那些事**都发生在他秘密行动的这段时间里，落到他眼里耳里**，他当时就在场。**不要因为某条标着「几小时前」就判他够不着**——那只说明它落在这段时间的较早处，不是说他当时不在。他接没接住，取决于上一条说的物理与空间条件，不取决于时间远近。
- 不得因为现场有人就一律判暴露，也不得因为四下无人就一律判他探到了东西。
- 是否被发觉和这一趟探到了什么需要分别裁定，不要混为一谈。
- 不得因为行动者身份强势或性格果决就一律判成功，现场条件与行动难度同样起作用。
- achieved 问的是「有没有探到本来拿不到的东西」，不是「这次秘密行动做完没有」：走了一趟什么都没有拿到，就是 false。
- 此处这段时间没发生什么，或发生的都是他本来就看得见的，同样是 false。空手而归是常态，照实判。
- 结合在场者各自的背景/性情，权衡他们对此事的立场（倾向揭发、佯装不见、还是漠不关心）：
  即便察觉，立场不同的人是否会声张大不相同——这与「物理上看没看见」一并影响 detected。
- 在场者标注的「此刻正忙着手上的事」或「此刻没有任何动作」，说的是他此刻**同时**在不在忙别的：
  忙着的人更难留意到暗处，闲着、无所事事的人更容易撞见——据此权衡 detected。
  他忙的究竟是什么，你看不到，也不必知道。
- achieved 与 detected 相互独立，四种组合都有可能。

【他能知道什么】
- 能：「此处先后发生的事」里他接得住的部分；物件上写着的字（这一趟真读到了才算）；在场者外在可见的举止与神色。
- 不能：任何人的内心——性情来历、动机盘算、打算做什么。他起身往哪去可以写，他心里图什么不能写。
- 他是从暗处看和听的人，不会读心。
- 别把满屋子物件的内容都算作他读到了——他这一趟够得着、又真去看的才算。

【裁断纪律】
""" + SAME_PLACE_VERDICT_RULE + "\n" + IDLE_BYSTANDER_VERDICT_RULE + "\n" + CLOSED_WORLD_FACT_RULE + """

【输出】
严格输出以下 JSON，不要任何多余内容。**字段顺序照抄，不要调换**——后面的字段要用到前面的结论。
- reason：裁断依据——他藏在哪、有无可信的感知通道、此处这段时间有没有事发生（≤100字）。
- outcome：第三人称、点名行动者，只写**行迹**，即旁人会看到的那一幕（≤40字）。**不写他探到的内容**——那是他一个人知道的。
- fact：第一人称「我」，写**我看见、听见、读到了什么**（≤120字）。要写具体内容，别用「我暗中观察了一番」充数；没接住就说没接住。
- outcome 与 fact 都不要带时间或地点前缀（如「在密室，…」）——地点由系统统一标注。
{"reason": "一两句裁断依据（≤100字）", "detected": true或false, "outcome": "第三人称、点名行动者、只写行迹（≤40字）", "fact": "第一人称、我看见/听见/读到了什么（≤120字）", "achieved": true或false, "why": "什么都没探到的缘由（≤20字）；探到了给空字符串"}
"""
        user = f"""\
【行动者】（裁决输入：行动者的性格，人设与状态等信息影响行动水准，但你不站在他的立场）
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【行动】
- 秘密行动意图：{purpose}
- 投入时间：{duration_label}{expected_part}{interrupt_part}

【现场】
{scene if scene else "（现场情况不明）"}

【他秘密行动的这段时间里，落到他眼里耳里的事】（{MEMORY_ORDER_HINT}；他接住了其中哪些由你裁断）
{happenings_block}

依上面说定的 JSON 格式与字段先后给出裁决，只输出 JSON、不写任何多余内容。"""
        # Same source as the user message above: every fact this adjudication can see.
        facts = (
            GivenFacts()
            .add("行动者处境", actor_condition)
            .add("秘密行动意图", purpose)
            .add("投入时间", duration_label)
            .add("期望结果", expected_outcome)
            .add("中途变故", interrupt_part.partition("：")[2])
            .add("现场", scene or "情况不明")
            # Omitting it would make the audit flag well-sourced findings as invented.
            .add("他秘密行动时落到眼里耳里的事", happenings_block)
        )
        try:
            with annotate_call(given_facts=facts):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    # reason ≤100 chars ≈150 + outcome ≤40 chars ≈60 + fact ≤120 chars ≈180
                    # + why ≤20 chars ≈30 + 2 bools ≈6 + 6-field JSON structure ≈30 → ≈456 tok.
                    temperature=0.7,
                    max_tokens=output_budget(456),
                    json_mode=True,
                )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "covert_judge_llm_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            return None

        # Missing fields claim the least (Rule 1).
        achieved = coerce_bool(data.get("achieved"), False)
        detected = coerce_bool(data.get("detected"), False)
        actor_name = self._directory.agent_name(agent.agent_id)
        # Default "found nothing": coming away empty-handed is still a real experience.
        fact = str(data.get("fact", "")).strip() or f"我暗中留意「{purpose}」，什么也没探到。"
        # Third person, never first: it leaks to onlookers on exposure.
        outcome = str(data.get("outcome", "")).strip() or f"{actor_name}悄悄进行了「{purpose}」。"
        # The judge sometimes fills why on success too. Unsliced: the prompt caps it.
        why = str(data.get("why", "")).strip() if not achieved else ""
        return achieved, detected, outcome, fact, why

    def _adjudication_failed_result(
        self, *, purpose: str, step: int, agent_id: str, estimated_steps: int,
        expected_outcome: str, location: str, target: "ActionTarget",
    ) -> ActionResult:
        """Null step when adjudication can't happen: no memory, no claimed success or exposure."""
        stub = AgentAction(
            agent_id=agent_id,
            step=step,
            action_type=ActionType.COVERT,
            action_description=purpose,
            target=target,
            estimated_steps=estimated_steps,
        )

        return ActionResult(
            action=stub,
            expected_outcome=expected_outcome or purpose,
            # Never perceived; only a trace/snapshot marker.
            outcome=scene_line(location, f"{self._directory.agent_name(agent_id)}悄悄着手做「{purpose}」，一时未能确知是否办成、有无被人察觉。"),
            succeeded=False,
            detected=False,
            adjudication_failed=True,
        )


