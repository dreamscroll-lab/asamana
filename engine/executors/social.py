"""Social action executor for TALK actions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, List

from agent.relation import format_relation_block, parse_relation_direction, relation_legend
from core.context import GivenFacts, annotate_call, observe_stage
from core.interfaces.action import (
    ActionResult, ActionTarget, ActionType, AgentAction, Observed, Ref, TargetAgentEffect,
)
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    condition_line,
    CLOSED_WORLD_FACT_RULE,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    MEMORY_ORDER_HINT,
    SituationVoice,
    render_memory_lines,
    render_situation_header,
)
from core.duration import describe_duration
from core.interfaces.execution import TickResult
from engine.executors.base import (
    ActionExecutionState,
    ActionExecutor,
    Conscription,
    participant_action_desc,
)
from engine.narration import (
    format_interrupt_reason_3p, format_interrupt_thought, observed_here, scene_line,
)
from engine.scene import SceneVisibility, assemble_scene_context, observe_location, situation_for

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

# Each side's LLM-judged direction → (trust_delta, affection_delta): TALK's single source of
# relation change. One notch below PHYSICAL's _REL_DELTA_* (a deed persuades a little more than
# a word). Keep small: relations move by accumulation, and apply_interaction triples negative
# trust.
_DIRECTION_TALK_DELTAS: dict[str, tuple[float, float]] = {
    "positive": (0.03, 0.04),
    "neutral":  (0.01, 0.01),
    "negative": (-0.02, -0.02),
}

# Soft cap on generated turns. A long TALK means more elapsed time, not more turns.
_MAX_DIALOGUE_TURNS: int = 8

_EMPTY_CTX: dict[str, Any] = {"relation": None, "about_other": [], "about_topic": [], "recent": []}

# Recent factual memories injected per side, picked as for the goal judge and interrupt evaluation.
_TALK_RECENT_FACTUAL_K = 5


def _talk_target(other_id: str | None) -> ActionTarget:
    """A talk's target: the other party is acted on and claimed, filed exactly as the decision
    layer does, since this rebuilds the action for feedback."""
    if not other_id:
        return ActionTarget()
    return ActionTarget(acts_on=[Ref.agent(other_id)], claims=[Ref.agent(other_id)])


def _present_listeners(
    candidate_ids: "List[str]",
    *,
    environment: "EnvironmentSystem",
    agents: dict[str, "Agent"],
    anchor_id: str,
    exclude: "set[str]",
) -> List[str]:
    """The named listeners actually standing here. Checked again at complete: listeners don't
    spend their turn and may walk away mid-conversation."""
    here = environment.get_body_location(anchor_id)
    seen: set[str] = set(exclude)
    present: List[str] = []
    for pid in candidate_ids:
        if pid in seen:
            continue
        seen.add(pid)
        agent = agents.get(pid)
        if agent is None or not agent.is_active:
            continue
        if environment.get_body_location(pid) != here:
            continue
        present.append(pid)
    return present


def _known_relation_block(rel, *, include_legend: bool = True) -> str:
    """format_relation_block only if there's a real relationship; strangers get '' so none is
    invented."""
    if rel is None:
        return ""
    labels = list(getattr(rel, "labels", []) or [])
    trust = getattr(rel, "trust", None)
    if trust is None:
        trust = getattr(rel, "trust_objective", 0.5)
    affection = getattr(rel, "affection", None)
    if affection is None:
        affection = getattr(rel, "affection_objective", 0.0)
    if not labels and abs(trust - 0.5) < 0.05 and abs(affection) < 0.05:
        return ""
    return format_relation_block(labels=labels, trust=trust, affection=affection, include_legend=include_legend)


def _format_context(
    ctx: dict, owner_name: str, other_name: str, *, now_step: int, seconds_per_step: int,
    world_start_second_of_day: int = 0,
) -> str:
    """Render gathered context as list items; memories go through ``render_memory_lines`` for
    recency prefixes."""
    lines: list[str] = []

    rel = ctx.get("relation")
    if rel is not None:
        trust = getattr(rel, "trust", None)
        if trust is None:
            trust = getattr(rel, "trust_objective", 0.5)
        affection = getattr(rel, "affection", None)
        if affection is None:
            affection = getattr(rel, "affection_objective", 0.0)
        labels = list(getattr(rel, "labels", []) or [])
        # The caller adds the legend once for both blocks.
        block = format_relation_block(labels=labels, trust=trust, affection=affection, include_legend=False)
        lines.append(f"- 与{other_name}的关系：{block}")
        history = getattr(rel, "history_summary", "") or ""
        if history:
            lines.append(f"- 与{other_name}的过往：{history}")

    about_other = [
        c for c in render_memory_lines(
            ctx.get("about_other", []), now_step=now_step,
            seconds_per_step=seconds_per_step, world_start_second_of_day=world_start_second_of_day)
        if c
    ]
    if about_other:
        lines.append(f"- {owner_name}对{other_name}的认识（{MEMORY_ORDER_HINT}）：")
        lines.extend(f"  - {c}" for c in about_other)

    about_topic = [
        c for c in render_memory_lines(
            ctx.get("about_topic", []), now_step=now_step,
            seconds_per_step=seconds_per_step, world_start_second_of_day=world_start_second_of_day)
        if c
    ]
    if about_topic:
        # The heading doesn't name the purpose: it's the initiator's private calculation and
        # would leak into the other column. And it only claims "possibly related": these are
        # semantic neighbors, often not about the other person at all.
        lines.append(f"- {owner_name}记得的其他事，或许与这次交谈有关（{MEMORY_ORDER_HINT}）：")
        lines.extend(f"  - {c}" for c in about_topic)

    recent = [
        c for c in render_memory_lines(
            ctx.get("recent", []), now_step=now_step,
            seconds_per_step=seconds_per_step, world_start_second_of_day=world_start_second_of_day)
        if c
    ]
    if recent:
        lines.append(f"- {owner_name}近来的经历（{MEMORY_ORDER_HINT}）：")
        lines.extend(f"  - {c}" for c in recent)

    return "\n".join(lines)


#: The byte-identical system prefix for dialogue generation (prefix cache).
_DIALOGUE_SYSTEM: str = f"""\
【任务】
你是中立的叙事者。基于给定双方各自的处境，客观写出他们之间的一段对话。你不代入任何一方，只如实呈现这场交谈会怎样发生。

【约束】
- **谈完之后，有一点东西该不一样了，有实质的变化。** 面对面说话，是他们此刻改变「谁知道什么」的机会：把自己
  那一栏里对方还不知道的说出去一点、把话挑明、把某件事定死、翻脸、松口、死咬着不松口——都算，
  也远不止这几样，怎样都行，只要不是白谈一场。
- **不一定谈成。** 谈崩、被搪塞、话说一半就散，都是正当结局；不正当的是谈完之后，两人之间跟开口
  之前一模一样。同时，对于谈话双方来说，也不一定一味地对抗，谈不成或者不欢而散，而是要根据实际情况，灵活应对。
- 实在没有新东西可给的时候，让他说不出口、说不清楚、或是不肯说——**别让他把自己那一栏里写着
  「已经说过、已经问过」的话，换个说法再说一遍。**
- 对白要从各自的人设、情绪、彼此之间关系和位置与已知信息自然长出，而非为推进话题服务：上面那
  一点变化不是你安排出来的，是他们各自想要的东西逼出来的。
- **每一方只能说出他自己那一栏里有的东西。** A 栏里写着的事，B 未必知道、更未必认账；绝不可让任
  何一方承认、证实或否认一件只有对方那栏才写着的事，也不可替对方把他还没亮的底牌先亮出来。
- **两栏说法相左时，那不是要你调和的错漏**——那正是这场交谈的关节。他们各自都当自己那套是真的，
  把这个演出来，别把它抹平成一套说得通的说法。允许双方误解、话不投机乃至不欢而散。
- 不要说教、不要把心事直白说尽、不要背景信息倾倒、不要书面腔。
- 对于人物的生死问题，需要特别关注，如果出现死者复现的问题，比如某人已经死亡了，但是现在感知到他在场，这个我需要关注并思考什么这样。
- 语言简练口语，长短随情境，不必每句等长。
- 如果对话中要提及某人，一定要符合逻辑和常量和地理空间规则，比如双方不能命令一个不在场的人攻击对方。
- 谈完即止；轮数上限由输入给出。
- 落笔前先看清【现场】给的当前时间与当前地点，避免在对白里的时间和处所错乱。同时有一点必须要注意时间已经过了并不代表某件之前约定好的事情一定发生，它是否发生需要以给定的上下文为准。
{CLOSED_WORLD_FACT_RULE}

【输出】
严格输出以下 JSON，不要任何多余内容（speaker 用 1 代表 A、2 代表 B，对应下方【A】【B】两人）。
三个字段按下面的次序写：先 reason（想清楚这场交谈），再据此写 dialogue，再由 dialogue 推出 observation。
{{"reason": "…", "dialogue": [{{"speaker": 1或2, "line": "这句话，≤60字"}}, ...], "observation": "旁人视角看见的这场交谈（≤25字）"}}

reason：动笔写对白之前，先把这场交谈想一遍（≤80字）——两人各自此刻要什么、各自知道和不知道什么、
这话谈得下去还是谈不下去、谁会先撑不住。想到什么写什么，不限于这几样。它只为让你落笔时有据可依，
不进对白、不进 observation。

observation 是**一个不在近旁、听不见内容的路人**看到的样子：他只见这两人的神色、姿态。
- 不必每场都看出戏来：**多数交谈从外面看就是平平无奇的**，看不出什么就把这一项留空。
- 不要为了让这句话有内容，而捏造神色、姿态、收场加戏。
**主语不必你写**：系统会在你这句前面补上「谁与谁的交谈：」，你只接着写他们的举止与收场。
**绝不可写出任何谈话内容、话题或双方的意图**（那是他无从得知的）；也不要写他们心里怎么想。

落笔写上面那些对白之前，还有一条：
{ABSOLUTE_TIME_RULE}"""


class SocialExecutor(ActionExecutor):
    """Executor for TALK actions.

    - Always 1v1. Extra people named become listeners (ActionTarget.reaches): their bodies
      stay free and they only get an overheard TargetAgentEffect. Group announcements belong
      to SEND_MESSAGE.
    - The whole dialogue is generated once at complete(); ticks are content-free.
    - One neutral narrator voices both sides. Accepted cost: one prompt holds both sides'
      private memories, so asymmetry can leak at generation (and persist via memory_summary).
    - Each side's memory_summary is its own first-person record.
    """

    # By invitation: if the other person is busy, the talk fails (see Conscription).
    conscription = Conscription.INVITE

    def __init__(
        self, llm_router: LLMRouter, directory: WorldDirectory, seconds_per_step: int = 3600,
        *, world_start_second_of_day: int = 0,
    ) -> None:
        self._llm = llm_router
        self._directory = directory
        self._seconds_per_step = seconds_per_step
        # The memory prefix's "今日/昨日" counts midnights crossed (see render_memory).
        self._world_start_second_of_day = world_start_second_of_day

    def claim_bodies(
        self,
        action: AgentAction,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
    ) -> list[str]:
        """Whose turn this conversation takes, by start()'s criteria. An infeasible talk takes
        nobody's, or "he isn't here" would be reported as "he's busy"."""
        target_ids = action.target.acted_on_agents[:1]
        if not target_ids or target_ids[0] not in agents:
            return []
        if not environment.check_talk_feasibility(
            action.agent_id, target_ids, target_label="对方",
        ).ok:
            return []
        return [target_ids[0]]

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        target_ids = action.target.acted_on_agents
        # The dialogue is 1v1; anyone else the decision layer bound is REACHED, not acted on.
        if len(target_ids) > 1:
            target_ids = target_ids[:1]

        target_label = self._directory.agent_name(target_ids[0]) if target_ids else "对方"

        location = observe_location(environment, action.agent_id)

        feasibility = environment.check_talk_feasibility(
            action.agent_id, target_ids, target_label=target_label,
        )
        if not feasibility.ok:
            return self._failed_result(action, step, feasibility.reason, location)

        if target_ids[0] not in agents:
            return self._failed_result(action, step, f"没有找到{target_label}，无法交谈。", location)

        duration = max(action.estimated_steps, 1)
        participant_ids = [action.agent_id, target_ids[0]]
        initiator_name = self._directory.agent_name(action.agent_id)
        target_name = self._directory.agent_name(target_ids[0])
        # The opening is public; the privileged transcript appears only in complete's outcome.
        opening = scene_line(location, f"{initiator_name}与{target_name}开始聊了起来。")
        state = ActionExecutionState.create(
            action_type=ActionType.TALK,
            initiator_id=action.agent_id,
            participant_ids=participant_ids,
            purpose=action.action_description or "对话",
            started_step=step,
            estimated_steps=duration,
            opening_outcome=opening,
            opening_observations=observed_here(environment, action.agent_id, opening),
            target=action.target,
            expected_outcome=action.expected_outcome,
            listener_ids=_present_listeners(
                action.target.reached_agents, environment=environment, agents=agents,
                anchor_id=action.agent_id, exclude={action.agent_id, target_ids[0]},
            ),
        )
        state.extra["target_id"] = target_ids[0]
        return state

    def _failed_result(self, action: AgentAction, step: int, outcome: str, location: str) -> ActionExecutionState:
        """A TALK feasibility failure as an actor-only ``create_failed`` execution."""
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.TALK,
            action_description=action.action_description or "talk",
            target=action.target,
        )
        actor_name = self._directory.agent_name(action.agent_id)
        failure = ActionResult(
            action=stub,
            expected_outcome=action.expected_outcome,
            # Quoted: unquoted, his first-person words render as "甲本想我去劝他…".
            outcome=scene_line(location, f"{actor_name}本想做「{action.action_description}」，{outcome}"),
            # Nothing happened, and the intent is private.
            observations=[],
            succeeded=False,
            failure_reason=outcome,
            factual_memory=f"想要做：{action.action_description}，但是结果是：{outcome}",
        )
        return ActionExecutionState.create_failed(
            action_type=ActionType.TALK, initiator_id=action.agent_id,
            failure_result=failure, started_step=step,
            purpose=action.action_description or "对话",
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
        initiator_id = state.initiator_id
        initiator_name = self._directory.agent_name(initiator_id)
        target_name = self._directory.agent_name(state.extra.get("target_id", ""))
        location = observe_location(environment, initiator_id)
        if elapsed <= 0:
            body = f"{initiator_name}与{target_name}的交谈刚开始。"
        else:
            body = f"{initiator_name}与{target_name}的交谈仍在继续，已持续{describe_duration(elapsed, self._seconds_per_step)}。"
        narrative = scene_line(location, body)
        # Content-free, so public: observation == outcome.
        return [
            TickResult(
                agent_id=pid, outcome=narrative,
                observations=observed_here(environment, pid, narrative),
            )
            for pid in state.participant_ids
        ]

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
            return [stored]  # feasibility failure (no pairing)
        initiator_id = state.initiator_id
        target_id = state.extra.get("target_id", "")
        initiator_name = self._directory.agent_name(initiator_id)
        target_name = self._directory.agent_name(target_id)
        location = observe_location(environment, initiator_id)

        initiator = agents.get(initiator_id)
        target = agents.get(target_id) if target_id else None

        # Three-layer context per participant (relation + memories about other + about topic).
        initiator_ctx_r, target_ctx_r = await asyncio.gather(
            self._gather_context(initiator, target_id, target_name, state.purpose, step),
            self._gather_context(target, initiator_id, initiator_name, state.purpose, step),
            return_exceptions=True,
        )
        initiator_ctx = dict(_EMPTY_CTX) if isinstance(initiator_ctx_r, BaseException) else initiator_ctx_r
        target_ctx = dict(_EMPTY_CTX) if isinstance(target_ctx_r, BaseException) else target_ctx_r

        scene = assemble_scene_context(
            initiator_id, environment=environment, directory=self._directory, agents=agents,
            visibility=SceneVisibility.GOD,  # functional judge
        ).text

        dialogue: list[dict] = []
        bystander_view = ""
        if initiator is not None and target is not None:
            dialogue, bystander_view = await self._llm_full_dialogue(
                initiator=initiator,
                target=target,
                initiator_name=initiator_name,
                target_name=target_name,
                purpose=state.purpose,
                expected_outcome=state.expected_outcome,
                # Don't use min(estimated_steps, MAX): a 1-step TALK would become a monologue.
                turns=_MAX_DIALOGUE_TURNS,
                initiator_ctx=initiator_ctx,
                target_ctx=target_ctx,
                scene=scene,
                now_step=step,
            )

        transcript = (
            "\n".join(f"{t['speaker']}：{t['line']}" for t in dialogue) if dialogue else "无对话"
        )

        # No transcript, no conversation: ask nothing, or an infrastructure failure becomes the
        # narrative fact "we talked" (Rule 1). summary=None is the null-step exit.
        listener_ids: list[str] = []
        listener_facts: list[Any] = []
        summary_i: "tuple[str, bool, str, str] | None" = None
        summary_t: "tuple[str, bool, str, str] | None" = None
        if dialogue:
            # All first-person memories depend only on the transcript: one batch.
            listener_ids = _present_listeners(
                state.listener_ids, environment=environment, agents=agents,
                anchor_id=initiator_id, exclude={initiator_id, target_id},
            )
            _fb_i = f"与{target_name}就「{state.purpose}」谈了一场。"
            _fb_t = f"与{initiator_name}就「{state.purpose}」谈了一场。"
            summary_i_r, summary_t_r, *listener_facts = await asyncio.gather(
                self._memory_summary(
                    agent=initiator, other_id=target_id, other_name=target_name, transcript=transcript,
                    purpose=state.purpose, fallback_fact=_fb_i, is_initiator=True,
                    now_step=step, environment=environment, expected_outcome=state.expected_outcome,
                ),
                self._memory_summary(
                    agent=target, other_id=initiator_id, other_name=initiator_name, transcript=transcript,
                    purpose=state.purpose, fallback_fact=_fb_t, is_initiator=False,
                    now_step=step, environment=environment, expected_outcome=state.expected_outcome,
                ),
                *(
                    self._listener_memory(
                        agent=agents.get(pid), initiator_name=initiator_name,
                        target_name=target_name, transcript=transcript,
                        now_step=step, environment=environment,
                    )
                    for pid in listener_ids
                ),
                return_exceptions=True,
            )
            summary_i = summary_i_r if not isinstance(summary_i_r, BaseException) else None
            summary_t = summary_t_r if not isinstance(summary_t_r, BaseException) else None

        # outcome carries the transcript; onlookers only see demeanour and how it ended.
        gist = scene_line(location, f"{initiator_name}与{target_name}交谈。")
        outcome = scene_line(location, f"{initiator_name}与{target_name}的交谈：\n{transcript}")
        # Code supplies the subject: left to the LLM it writes "二人", losing "who".
        observations = observed_here(environment, initiator_id, scene_line(
            location,
            f"{initiator_name}与{target_name}的交谈：{bystander_view}" if bystander_view
            else f"{initiator_name}与{target_name}在交谈。",
        ))
        listener_effects: List[TargetAgentEffect] = []
        for pid, fact in zip(listener_ids, listener_facts):
            if isinstance(fact, BaseException) or not fact:
                logger.warning(
                    "talk_listener_memory_skipped",
                    extra={"agent_id": pid, "step": step},
                )
                continue
            listener_effects.append(
                TargetAgentEffect(agent_id=pid, factual_memory=fact, overheard=True)
            )

        results: List[ActionResult] = []
        results.append(self._talk_result(
            agent_id=initiator_id, step=step, purpose=state.purpose,
            action_description=state.purpose,     # initiator: the act he wrote himself
            estimated_steps=state.estimated_steps, rel_other_id=target_id,
            other_name=target_name, outcome=outcome, gist=gist, observations=observations,
            dialogue=dialogue,
            # Don't fall back to purpose: it's intent, not expectation.
            summary=summary_i, expected_outcome=state.expected_outcome,
            # Attached to the initiator's result only (attaching to both → listeners get it twice).
            target_effects=listener_effects,
        ))
        if target is not None:
            results.append(self._talk_result(
                agent_id=target_id, step=step, purpose=state.purpose,
                action_description=participant_action_desc(self._directory, target_id, state),
                estimated_steps=state.estimated_steps, rel_other_id=initiator_id,
                other_name=initiator_name, outcome=outcome, gist=gist,
                observations=observations,
                dialogue=dialogue,
                summary=summary_t, expected_outcome="",
            ))

        return results

    @staticmethod
    def _talk_result(
        *,
        agent_id: str,
        step: int,
        purpose: str,
        action_description: str,
        estimated_steps: int,
        rel_other_id: str,
        other_name: str,
        outcome: str,
        gist: str,
        observations: list[Observed],
        dialogue: list[dict],
        summary: "tuple[str, bool, str, str] | None",
        expected_outcome: str,
        target_effects: "list[TargetAgentEffect] | None" = None,
    ) -> ActionResult:
        """Build one participant's TALK result.

        ``purpose`` is the shared topic; ``action_description`` is this person's own act (they
        differ for the conscripted party). ``summary=None`` is a null step: no memory, no
        relation delta. Listeners' ``target_effects`` still go out then: what they heard
        doesn't depend on this summary.
        """
        stub = AgentAction(
            agent_id=agent_id, step=step, action_type=ActionType.TALK,
            action_description=action_description,
            target=_talk_target(rel_other_id),
            estimated_steps=estimated_steps,
        )
        if summary is None:
            return ActionResult(
                action=stub,
                expected_outcome=expected_outcome,
                outcome=outcome,
                gist=gist,
                observations=observations,
                succeeded=False,
                # No factual_memory: feedback skips null steps before reading it.
                dialogue=dialogue,
                relation_updates=[],
                adjudication_failed=True,
                target_effects=list(target_effects or []),
            )
        fact, succeeded, direction, why = summary
        td, ad = _DIRECTION_TALK_DELTAS.get(direction, _DIRECTION_TALK_DELTAS["neutral"])
        return ActionResult(
            action=stub,
            expected_outcome=expected_outcome,
            outcome=outcome,
            gist=gist,
            observations=observations,
            # A lurker hears the transcript. Not given on interrupt.
            happening=outcome,
            succeeded=succeeded,
            failure_reason=why,     # 3p authoritative "why the talk didn't work"
            factual_memory=fact,
            dialogue=dialogue,
            relation_updates=[(rel_other_id, td, ad)] if rel_other_id else [],
            target_effects=list(target_effects or []),
        )

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
        initiator_id = state.initiator_id
        target_id = state.extra.get("target_id", "")
        initiator_name = self._directory.agent_name(initiator_id)
        target_name = self._directory.agent_name(target_id)
        location = observe_location(environment, initiator_id) if environment is not None else "此处"
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        breaker = (
            self._directory.agent_name(interrupted_agent_id)
            if interrupted_agent_id and interrupted_agent_id in state.participant_ids
            else ""
        )
        if cause:
            cut = f"因{cause}而中断"
        elif breaker:
            cut = f"被{breaker}打断"
        else:
            cut = "被打断"
        # The breaker's thought is quoted only on his own result.
        what_happened = f"{initiator_name}与{target_name}的交谈{cut}。"
        breaker_id = (
            interrupted_agent_id if interrupted_agent_id in state.participant_ids else initiator_id
        )
        breaker_outcome = scene_line(
            location, what_happened + format_interrupt_reason_3p(breaker or initiator_name, thought),
        )
        others_outcome = scene_line(location, what_happened)
        results: List[ActionResult] = []
        for pid in (initiator_id, target_id):
            if not pid or pid not in agents:
                continue
            agent = agents[pid]
            is_initiator = (pid == initiator_id)
            other_id = target_id if is_initiator else initiator_id
            other_name = target_name if is_initiator else initiator_name
            # The breaker knows why; the other only sees him break off. Key on the interrupted
            # participant, not the trigger source (rarely a participant). Unknown → both know.
            is_triggered = (interrupted_agent_id is None or pid == interrupted_agent_id)
            # Own eyes, first person; _interrupt_summary supplies its own header.
            scene = (
                assemble_scene_context(
                    pid, environment=environment, directory=self._directory, agents=agents,
                    include_header=False, voice=SituationVoice.FIRST,
                    visibility=SceneVisibility.OWN_EYES,
                ).text
                if environment is not None else ""
            )
            relation_block = await self._present_relations_block(agent, environment)
            factual = await self._interrupt_summary(
                agent=agent, other_name=other_name, purpose=state.purpose,
                thought=thought, cause=cause, elapsed_label=elapsed_label,
                is_triggered=is_triggered, is_initiator=is_initiator, scene=scene,
                environment=environment, now_step=step, relation_block=relation_block,
            )
            stub = AgentAction(
                agent_id=pid, step=step, action_type=ActionType.TALK,
                action_description=(
                    state.purpose if is_initiator
                    else participant_action_desc(self._directory, pid, state)
                ),
                target=_talk_target(other_id),
                estimated_steps=state.estimated_steps,
            )
            # An interrupted conversation moves no relation.
            results.append(ActionResult(
                action=stub,
                # Only the initiator wrote an expectation.
                expected_outcome=state.expected_outcome if is_initiator else "",
                outcome=breaker_outcome if pid == breaker_id else others_outcome,
                gist=others_outcome,
                succeeded=False,
                factual_memory=factual,     # 1p (_interrupt_summary) → own memory
            ))
        return results

    async def _present_relations_block(self, agent: "Agent", environment: "EnvironmentSystem | None") -> str:
        """The agent's own relation to each co-present person it knows (strangers omitted)."""
        if environment is None:
            return ""
        loc = environment.get_body_location(agent.agent_id)
        # Cognitive tier only: an Npc holds no relation with anyone.
        present = [a for a in environment.agents_at(loc) if a != agent.agent_id]
        lines: list[str] = []
        for pid in present:
            try:
                rel = await agent.relation_system.perceive_existing(
                    pid, emotion=agent.personality.state.emotion)
            except Exception as exc:  # noqa: BLE001 — read failed: skip this relation block
                logger.warning(
                    "relation_perceive_failed",
                    extra={"agent_id": agent.agent_id, "target_id": pid, "error": str(exc)},
                )
                continue
            block = _known_relation_block(rel, include_legend=False)
            if block:
                lines.append(f"- 与在场的{self._directory.agent_name(pid)}：{block}")
        if not lines:
            return ""
        return "\n".join(lines) + f"\n（关系数值含义：{relation_legend()}）"

    async def _interrupt_summary(
        self,
        *,
        agent: "Agent",
        other_name: str,
        purpose: str,
        thought: str,
        cause: str,
        elapsed_label: str,
        is_triggered: bool,
        is_initiator: bool,
        scene: str,
        environment: "EnvironmentSystem | None" = None,
        now_step: int = 0,
        relation_block: str = "",
    ) -> str:
        """First-person record of an interrupted conversation (one per participant)."""
        thought_part = format_interrupt_thought(thought)
        initiator_line = (
            f"是我主动找{other_name}谈的" if is_initiator else f"{other_name}来找我谈"
        )
        opening = f"我主动找{other_name}" if is_initiator else f"{other_name}找我"
        # The trigger signal stays out (see base.interrupt).
        if cause:
            # Nobody chose to stop, so both share this sentence; the cause must be stated or the
            # model invents one from purpose.
            situation = f"这场谈话因{cause}而中断"
            fallback = f"{opening}就「{purpose}」谈到一半（{elapsed_label}），{situation}"
        elif is_triggered:
            situation = f"我中途撂下了这场谈话{thought_part}"
            fallback = f"{opening}就「{purpose}」谈到一半（{elapsed_label}），我中途撂下了这场谈话{thought_part}"
        else:
            situation = "对方突然中止了交谈"
            fallback = f"{opening}就「{purpose}」谈到一半，{situation}"

        if agent is None:
            return fallback

        scene_part = f"\n【现场】\n{scene}" if scene else ""
        relation_part = f"\n【我与在场者的关系】\n{relation_block}" if relation_block else ""
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场被打断的对话——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角,这场被打断的对话让我记住了什么、留下什么感受。始终用我自己的口吻。
提到我认识的人时,不妨带上其名字便于日后回想(惯以称谓相称的话,可附成「称谓（名字）」);这只是建议、不必生硬套用,陌生人用描述。
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "一句话（不超过36字）"}
"""
        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        # Without a time/place anchor the model takes place names from purpose.
        situation_header = (
            render_situation_header(situation_for(environment, agent.agent_id),
                                    voice=SituationVoice.FIRST)
            if environment is not None else ""
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【刚刚发生的事】
{initiator_line}，正就「{purpose}」交谈，已谈了{elapsed_label}，{situation}。{relation_part}{scene_part}

依上面说定的 JSON 格式记下这一场，只输出 JSON、不写任何多余内容。"""
        try:
            # Attribute the trace to this participant. talk_role lets review exempt the addressee
            # from the decision↔action consistency check.
            with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
                given_facts=GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("刚刚发生的事",
                     f"{initiator_line}，正就「{purpose}」交谈，已谈了{elapsed_label}，{situation}。")
                .add("我与在场者的关系", relation_block)
                .add("现场", scene),
                talk_role="initiator" if is_initiator else "addressee",
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
            return str(extract_json(resp.content).get("fact", "")).strip() or fallback
        except Exception as exc:
            logger.warning("talk_interrupt_summary_failed", extra={"agent_id": agent.agent_id, "error": str(exc)})
            return fallback

    async def _gather_context(
        self,
        agent: "Agent | None",
        other_id: str,
        other_name: str,
        purpose: str,
        step: int,
    ) -> dict:
        """Collect relation + memories (about the other agent, the topic, his recent past). Each
        read failure only empties its own column."""
        if agent is None:
            return dict(_EMPTY_CTX)

        relation = None
        try:
            relation = await agent.relation_system.perceive_existing(
                other_id, emotion=agent.personality.state.emotion,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: relation is None, continue
            logger.warning(
                "relation_perceive_failed",
                extra={"agent_id": agent.agent_id, "target_id": other_id, "error": str(exc)},
            )

        # Fetched first: otherwise the other columns take the latest key event and this timeline
        # loses its key link. Fetch extra (as Agent._recent_factual_lines) to survive dropping
        # entries without factual.
        recent: list = []
        try:
            pairs = agent.memory_system.sample_recent_events(step, top_k=_TALK_RECENT_FACTUAL_K + 8)
        except Exception as exc:  # noqa: BLE001 — read failed: drop recent events, continue
            logger.warning(
                "social_recent_memory_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            pairs = []
        recent = [fact for fact, _exp in pairs if fact is not None][:_TALK_RECENT_FACTUAL_K]
        seen: set[str] = {m.id for m in recent}

        # exclude_ids inside retrieval, so each column still fills its quota; run in order.
        # "What I know of him" filters by other_id, then ranks by purpose. Don't fall back to
        # retrieve(other_name): in a single-theme world that recalls the topic, not the person.
        # Empty means "I don't know him".
        about_other: list = []
        try:
            about_other = await agent.memory_system.recall_about_agent(
                other_id, purpose, current_step=step, top_k=5, exclude_ids=seen,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: drop this column, continue
            logger.warning(
                "social_memory_retrieve_failed",
                extra={"agent_id": agent.agent_id, "target_id": other_id, "error": str(exc)},
            )
        seen |= {m.id for m in about_other}
        about_topic: list = []
        try:
            about_topic = await agent.memory_system.retrieve(
                purpose, current_step=step, top_k=5, exclude_ids=seen,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: drop this column, continue
            logger.warning(
                "social_memory_retrieve_failed",
                extra={"agent_id": agent.agent_id, "target_id": other_id, "error": str(exc)},
            )

        return {
            "relation": relation, "about_other": about_other, "about_topic": about_topic,
            "recent": recent,
        }

    async def _llm_full_dialogue(
        self,
        *,
        initiator: "Agent",
        target: "Agent",
        initiator_name: str,
        target_name: str,
        purpose: str,
        expected_outcome: str,
        turns: int,
        initiator_ctx: dict,
        target_ctx: dict,
        scene: str,
        now_step: int,
    ) -> list[dict]:
        """Generate the full conversation in one functional call.

        Returns (turns, observation): turns are ``{"speaker_id", "speaker", "line"}``; ``([], "")``
        on failure, never a fabricated line. Speakers come back as indices (1=A, 2=B).
        """
        initiator_ctx_text = _format_context(
            initiator_ctx, initiator_name, target_name,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
            world_start_second_of_day=self._world_start_second_of_day,
        )
        target_ctx_text = _format_context(
            target_ctx, target_name, initiator_name,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
            world_start_second_of_day=self._world_start_second_of_day,
        )
        expected_part = f"\n对话发起者的期望：{expected_outcome}" if expected_outcome else ""
        scene_part = f"\n【现场】\n{scene}" if scene else ""
        # One legend for both relation blocks, only if either has one.
        has_relation = bool(initiator_ctx.get("relation") or target_ctx.get("relation"))
        legend_block = f"关系数值含义：{relation_legend()}\n\n" if has_relation else ""

        initiator_condition = condition_line(
            initiator.personality.state.condition, voice=SituationVoice.THIRD,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        target_condition = condition_line(
            target.personality.state.condition, voice=SituationVoice.THIRD,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        # Shared with the trace annotation, so the leak review reads exactly what the model saw.
        lane_a = (
            f"{initiator.personality.to_prompt_context(include_emotion=True, include_goals=False)}"
            f"{initiator_condition}\n{initiator_ctx_text}"
        )
        lane_b = (
            f"{target.personality.to_prompt_context(include_emotion=True, include_goals=False)}"
            f"{target_condition}\n{target_ctx_text}"
        )
        system = _DIALOGUE_SYSTEM
        user = f"""\
{legend_block}【A：{initiator_name}】
{lane_a}

【B：{target_name}】
{lane_b}

【对话目的】（以下是{initiator_name}单方面的说法与盘算，既非已定的事实，也非这场交谈该有的结局）
{initiator_name}想就「{purpose}」与{target_name}交谈。{expected_part}{scene_part}

【轮数上限】
至多 {turns} 轮，可更少；谈完即止。

依上面说定的 JSON 格式（speaker 用 1 代表{initiator_name}、2 代表{target_name}）输出，只输出 JSON、不写任何多余内容。"""
        # Same source as user above: every fact this dialogue generation can see.
        facts = (
            GivenFacts()
            .add(f"{initiator_name} 的处境", initiator_condition)
            .add(f"{initiator_name} 手上的", initiator_ctx_text)
            .add(f"{target_name} 的处境", target_condition)
            .add(f"{target_name} 手上的", target_ctx_text)
            .add("对话目的", f"{initiator_name}想就「{purpose}」与{target_name}交谈。")
            .add("发起者的期望", expected_outcome)
            .add("现场", scene)
            .add("轮数上限", turns)
        )
        # One semantic retry (as DecisionEngine._llm_select) for broken JSON or an
        # unattributable line; same prompt, same bar, never filling in lines.
        for attempt in (1, 2):
            reason = ""
            try:
                with annotate_call(
                    given_facts=facts,
                    dialogue_lanes={
                        "a": {"name": initiator_name, "text": lane_a},
                        "b": {"name": target_name, "text": lane_b},
                    },
                ):
                    response = await self._llm.complete(
                        LLMScene.AGENT_ACTION_NARRATION,
                        [
                            LLMMessage(role="system", content=system),
                            LLMMessage(role="user", content=user),
                        ],
                        temperature=0.8,
                        # Per turn {speaker, line ≤60 chars ≈ 90 tok} ≈ 100 tok, plus reason (≤80 chars
                        # ≈ 120 tok), observation (≤25 chars ≈ 38 tok) and the three fields' structure
                        # (~15 tok).
                        max_tokens=output_budget(turns * 100 + 173),
                        json_mode=True,
                    )
                data = extract_json(response.content)
                parsed = self._parse_dialogue(
                    data,
                    (initiator.agent_id, initiator_name),
                    (target.agent_id, target_name),
                )
                if parsed:
                    observation = " ".join(str(data.get("observation", "")).split()).strip()
                    return parsed, observation
                reason = "transcript_unattributable"
            except Exception as exc:
                reason = str(exc)
            logger.warning(
                "talk_dialogue_failed",
                extra={"initiator": initiator.agent_id, "attempt": attempt, "reason": reason},
            )
        return [], ""

    @staticmethod
    def _parse_dialogue(
        data: object, speaker_a: tuple[str, str], speaker_b: tuple[str, str]
    ) -> list[dict]:
        """Map ``{"dialogue":[{"speaker":1|2,"line":...}]}`` to turns tagged with each
        speaker's ``(agent_id, name)``.

        One unattributable line voids the whole transcript: a readable stub still looks like a
        finished conversation and gets summarized as one. Empty lines are skipped.
        """
        turns: list[dict] = []
        raw = data.get("dialogue", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return turns
        for t in raw:
            idx = t.get("speaker") if isinstance(t, dict) else None
            line = str(t.get("line", "")).strip() if isinstance(t, dict) else ""
            if isinstance(t, dict) and not line:
                continue
            who = speaker_a if idx in (1, "1") else speaker_b if idx in (2, "2") else None
            if who is None:
                logger.warning(
                    "talk_dialogue_unattributable",
                    extra={"speaker": repr(idx), "readable": len(turns), "total": len(raw)},
                )
                return []
            turns.append({"speaker_id": who[0], "speaker": who[1], "line": line})
        return turns

    async def _memory_summary(
        self,
        *,
        agent: "Agent | None",
        other_id: str,
        other_name: str,
        transcript: str,
        purpose: str,
        fallback_fact: str,
        is_initiator: bool,
        now_step: int = 0,
        environment: "EnvironmentSystem | None" = None,
        expected_outcome: str = "",
    ) -> "tuple[str, bool, str, str] | None":
        """First-person record of the conversation from this participant's view, judged in
        character. Returns (fact, succeeded, relation_dir, why), or ``None`` when it can't
        happen, which the caller turns into a null step.
        """
        if agent is None:
            logger.warning("talk_memory_summary_no_participant", extra={"other_id": other_id})
            return None

        rels = await self._present_relations_block(agent, environment)
        relation_part = f"\n【我与在场者的关系】\n{rels}" if rels else ""

        situation_header = (
            render_situation_header(situation_for(environment, agent.agent_id),
                                    voice=SituationVoice.FIRST)
            if environment is not None else ""
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        initiator_line = (
            f"是我主动找{other_name}谈的" if is_initiator else f"{other_name}来找我谈"
        )
        expected_part = f"\n原本期望：{expected_outcome}" if (is_initiator and expected_outcome) else ""
        # other_name stays out of system (prefix cache); the reminder at the end of user names him.
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场刚结束的对话——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角:这场对话达成我在意的目的了吗?发生了什么实质的事?谈完之后我对对方的整体观感有何变化。始终用我自己的口吻。

【约束】
 - fact 里提到我认识的人时,不妨带上其名字,便于日后回想;若我惯以称谓相称,可把名字附在称谓后(形如「称谓（名字）」)。这只是便于回想的建议,不必生硬套用、更不必每处都加括号;陌生人用描述,不臆造名字。
 - 要区分已经发生、正在发生、将要发生的事情：把“我现在要去超市”写成“我去了超市”、把“我命令他帮我收拾行李”写成“他已经帮我收拾行李”，都是禁止的。
 - 先理解此刻的时间和我此刻所在地点。
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "我视角一句话,我从这场对话得到了什么或发生了什么(≤100字)", "success": true或false, "why": "没能达成目的的缘由（≤20字）；达成了给空字符串", "relation": "positive或negative或neutral（我对对方的观感变化）"}
其中 why 是**说给旁人听的一句交代**:只写外面看得出的缘由(「对方不肯松口」「话没说到点上」),不写我的懊丧、盘算或对自己的评价——那些只属于 fact。

落笔写上面那句 fact 之前,还有一条:
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON
        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【刚结束的对话】
{initiator_line}，话题是「{purpose}」。{relation_part}{expected_part}
对话记录：
{transcript}

依上面说定的 JSON 格式记下这一场（relation 反映我对{other_name}的观感变化），只输出 JSON、不写任何多余内容。"""
        try:
            # Attribute to this participant; the addressee also gets conscripted_input for review.
            call_annotations = {
                "given_facts": GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("刚结束的对话", f"{initiator_line}，话题是「{purpose}」。")
                .add("我与在场者的关系", rels)
                # The addressee never saw the initiator's expectation; declaring it would blind
                # the leak review.
                .add("发起者的期望", expected_outcome if is_initiator else "")
                .add("对话记录", transcript),
                "talk_role": "initiator" if is_initiator else "addressee",
                "action_owner": agent.personality.soul.name,
            }
            if not is_initiator:
                call_annotations["conscripted_input"] = f"{initiator_line}，话题是「{purpose}」"
            with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
                **call_annotations
            ):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    # fact (≤100 chars ≈ 150 tok, including a possible bracketed date) + success
                    # + why (≤20 chars ≈ 30) + relation; est. ~211 tok.
                    max_tokens=output_budget(211),
                    json_mode=True,
                )
            data = extract_json(response.content)
            fact = str(data.get("fact", "")).strip() or fallback_fact
            succeeded = coerce_bool(data.get("success"), True)
            relation_dir = parse_relation_direction(str(data.get("relation", "neutral"))).value
            # Unsliced: the prompt caps it.
            why = str(data.get("why", "")).strip() if not succeeded else ""
            return fact, succeeded, relation_dir, why
        except Exception as exc:
            logger.warning("talk_memory_summary_failed", extra={"agent_id": agent.agent_id, "error": str(exc)})
            return None

    async def _listener_memory(
        self,
        *,
        agent: "Agent | None",
        initiator_name: str,
        target_name: str,
        transcript: str,
        now_step: int = 0,
        environment: "EnvironmentSystem | None" = None,
    ) -> str | None:
        """A listener's first-person record of what he heard: no success verdict or relation,
        since he said nothing. Not the raw transcript, which would embed the whole dialogue.

        ``None`` on failure, and no memory is written: a content-free line only dilutes recall.
        """
        if agent is None:
            return None

        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        situation_header = (
            render_situation_header(situation_for(environment, agent.agent_id),
                                    voice=SituationVoice.FIRST)
            if environment is not None else ""
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        system = """\
我此刻代入这个角色,以第一人称「我」记下我方才在旁边听到的一场对话——这是我一贯的记事方式,与处境无关。

【我要说的】
我听见了什么,以及这些话对我意味着什么。用我自己的口吻。

【我绝不这样写】
 1.不把自己写进那场对话:我一句话也没说,不能写成我问了、我答了、我劝了。
 2.不复述流水账:不逐句转写他们的对白,只记下落到我心里的那一点。
 3.不添他们没说过的话,也不替他们补上没挑明的意思。
 4.不混淆时态:将要发生的事不能写成已经发生。
 5.提到我认识的人时,不妨带上其名字,便于日后回想;惯以称谓相称的,可把名字附在称谓后
   (形如「称谓（名字）」)。这只是便于回想的建议,不必生硬套用、更不必每处都加括号;
   陌生人用描述,不臆造名字。
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "我视角一句话,我从旁听到的这场对话里知道了什么(≤60字)"}

落笔写上面那句 fact 之前,还有一条:
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【我在旁边听到的对话】
说话的是{initiator_name}与{target_name}，我在一旁听着。
对话记录：
{transcript}

依上面说定的 JSON 格式记下我听到了什么，只输出 JSON、不写任何多余内容。"""
        try:
            # Attribute to the listener, not the initiator.
            with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
                given_facts=GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("我在旁边听到的", f"说话的是{initiator_name}与{target_name}")
                .add("对话记录", transcript),
                talk_role="listener", action_owner=agent.personality.soul.name,
            ):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    # fact ≤60 chars ≈ 90 tok + JSON structure ~5; est. ~95 tok.
                    max_tokens=output_budget(95),
                    json_mode=True,
                )
            return str(extract_json(response.content).get("fact", "")).strip() or None
        except Exception as exc:
            logger.warning(
                "talk_listener_memory_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            return None
