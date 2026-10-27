"""Social action executor for TALK actions."""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from agent.relation import format_relation_block, parse_relation_direction, relation_legend
from core.context import GivenFacts, annotate_call, note_active_call_adoption, observe_actor, observe_stage
from core.interfaces.action import (
    ActionResult, ActionTarget, ActionType, AgentAction, DialogueTurn, Observed, Ref, TargetAgentEffect,
)
from core.interfaces.actor import BriefPerson, Conversation
from core.action_duration import action_duration_steps
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import LLMRouter, coerce_bool, output_budget
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
    person_referent,
    render_memory_lines,
    render_situation_header,
)
from core.duration import describe_duration
from core.interfaces.execution import TickResult
from engine.executors.base import (
    ActionExecutionState,
    ActionExecutor,
    Conscription,
    complete_json,
    participant_action_desc,
)
from engine.narration import (
    format_interrupt_reason_3p, format_interrupt_thought, observed_here, scene_line,
)
from engine.scene import SceneVisibility, assemble_scene_context, observe_location, situation_for

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.memory_types import Memory
    from agent.relation import PerceivedRelation
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

# Static, so it lives in system ahead of the rows it explains; after them the model has already
# read every label by its own prior.
_PRESENT_RELATIONS_INPUT: str = f"""\
【输入说明】
下面【我对在场者的关系】每行是我对一位在场者的关系：我是持有者，那位在场者是对方。各字段含义：{relation_legend()}
"""

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

# A live conversation: at most this many dialogue turns, both sides' together, and what the actor
# says in one at most this long. An AI turn's length cap is in its prompt.
#
# The turn cap is a backstop against a conversation that won't end, not a narrative constraint:
# the AI side isn't told it, and ends by the conversation's own course. Don't put the turns left
# in its prompt: a count of turns to spare reads as room to keep talking. The actor is shown it.
_LIVE_MAX_TURNS: int = 20
_LIVE_ACTOR_TURN_CHARS: int = 100

# What an AI side recalls as a live conversation goes on, by the other side's last line: the
# context gathered up front can't foresee what the other brings up. Recalled every turn, not every
# few: a turn that misses the new subject has the AI deny knowing it, and that denial is remembered.
# Lines shorter than _LIVE_RECALL_MIN_CHARS (assent, greetings) carry no subject.
#
# The floor is above RETRIEVAL_SCORE_FLOOR on purpose. That one only filters junk, and raising it
# loses true answers; here a missed recall costs nothing, while a loose one puts an unrelated
# memory in the speaker's mouth.
_LIVE_RECALL_MIN_CHARS: int = 6
_LIVE_RECALL_PER_TURN: int = 2
_LIVE_RECALL_TOTAL: int = 6
_LIVE_RECALL_SCORE_FLOOR: float = 0.45

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


def _topic_query(state: ActionExecutionState, *, is_initiator: bool, live: bool) -> str:
    """What a side recalls the talk's subject by; empty recalls none.

    The narrator writes both sides and knows both aims, and has no later recall: both sides recall
    by the purpose, or the addressee can't answer what the talk is about. In a live talk the
    addressee knows only the topic stated up front: don't recall by the initiator's private purpose,
    which would have him recall just what the hidden aim needs. With no topic he recalls nothing
    yet; mid-talk recall follows the other side's lines.
    """
    return state.purpose if is_initiator or not live else state.talk_topic


def _format_context(
    ctx: dict, owner_name: str, other_name: str, *, now_step: int, seconds_per_step: int,
    world_start_second_of_day: int = 0,
) -> str:
    """Render gathered context as list items; memories go through ``render_memory_lines`` for
    recency prefixes."""
    lines: list[str] = []

    rel: "PerceivedRelation | None" = ctx.get("relation")
    if rel is not None:
        # The legend is in each caller's system prompt.
        block = format_relation_block(
            labels=rel.labels, trust=rel.trust, affection=rel.affection, include_legend=False,
        )
        lines.append(f"- {owner_name}对{other_name}的关系：{block}")
        if rel.history_summary:
            lines.append(f"- 与{other_name}的过往：{rel.history_summary}")

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
- 注意人物生死信息的一致性：若材料同时描述某人已经死亡和此刻仍然在场，应结合给定上下文处理这一矛盾，不凭空补充解释。比如某人已经死亡了，但是现在看到他就在眼前。
- 语言简练口语，长短随情境，不必每句等长。
- 对话中涉及他人时，应符合逻辑、常识和地理空间限制，不能把不在场的人写成当场参与交谈或执行行动。比如不能命令一个不在场的人攻击对方。
- 谈完即止；轮数上限由输入给出。
- 落笔前先看清【现场】给的当前时间与当前地点，避免在对白里的时间和处所错乱。同时有一点必须要注意时间已经过了并不代表某件之前约定好的事情一定发生，它是否发生需要以给定的上下文为准。
{CLOSED_WORLD_FACT_RULE}

【说明】
两人关系各栏的含义：{relation_legend()}

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


@dataclass(frozen=True)
class _LiveSide:
    """One side of a live conversation: who, the name the transcript gives them, and what they
    bring to it (``_gather_context``)."""

    agent: "Agent"
    name: str
    ctx: dict[str, Any]
    is_initiator: bool


#: The byte-identical system prefix for an AI side's dialogue turn in a live conversation (prefix cache).
_LIVE_TURN_SYSTEM: str = f"""\
【我此刻】
我就是下面写的这个人，正和人当面说话。轮到我开口了。

【所给】
- 我是谁、我此刻的处境、我眼前的人和物。
- 我知道的：我和对方的关系、我对对方的认识、我记得的相关的事、我近来的经历。
- 这场对话：是谁找的谁、到现在说过的话。

【要做的】
- 先判断这场话说完了没有，再决定怎么开口。
- 没说完：接着对方的话，说出我这一轮的话。
- 说完了：收场。可以说一句告别、应承或撂下的话，也可以一句不说。

【怎样算说完】（有一条就算）
- 我想要的已经得到了，或已经明白得不到。
- 事情已经说定：对方已经答应了、告退了，或只回了一句应付的话，或者对方已经明确主动结束对话了。
- 再说下去，也不会有什么实质的内容和进展了。
- 谈崩了，或我此时此刻不想再谈了。
拿不准时就收场：当面说话本来就短。

【约束】
- 对方已经应承或告辞，我还一句接一句地叮嘱、重复说过的事——这是最该避免的。
- 我只说我自己知道的事：下面「我知道的」和到现在说过的话里没有的，我就不知道。
- 对方的话我未必信、也未必认同：说法和我知道的相冲突时，结合下面「我知道的」想清楚，不一味肯定，也不一味否定。认不认同、嘴上怎么说，取决于我和他的关系、我知道的和我的处境；嘴上和心里可以不一样。
- 说法相左时，我照自己认定的来，不替两边调和。
- 说话符合我人设，性格，当前的情绪以及与对方的关系等等。
- 只说我这一轮的话：不写动作旁白，不替对方说话。
- 口语、简练，长短随情境；也可随情境做一些适当的调整。比如闲聊是可以偶尔幽默，说正事时需要正经严肃等等。
- 提到不在场的人要合乎情理：比如不能指使不在场的人当场做事。
- 先看清此刻的时间和我所在的地方，别在话里把时间、处所说错。
{CLOSED_WORLD_FACT_RULE_FIRST_PERSON}

关系数值含义：{relation_legend()}

【输出】
严格输出以下 JSON，不要任何多余内容，三个字段按这个次序写：
{{"thought": "对方那句话的意思、这场话说完了没有、我要怎么回，≤60字", "end": true或false, "line": "我这一轮说出口的话，≤60字"}}
- thought：先弄清对方那句话的意思，对照我知道的和我的处境；再照上面【怎样算说完】判断这场话说完了没有；没说完，就想好我要怎么回。
- end：说完了就是 true，这一轮之后我不再说。
- line：说出口的原话。end 为 true 时是收场的话，可以留空。

开口之前，还有一条：
{ABSOLUTE_TIME_RULE_FIRST_PERSON}"""


class SocialExecutor(ActionExecutor):
    """Executor for TALK actions.

    - Always 1v1. Extra people named become listeners (ActionTarget.reaches): their bodies
      stay free and they only get an overheard TargetAgentEffect. Group announcements belong
      to SEND_MESSAGE.
    - The whole dialogue is generated once at complete(); ticks are content-free.
    - Between AI characters one neutral narrator voices both sides. Accepted cost: one prompt
      holds both sides' private memories, so asymmetry can leak at generation (and persist via
      memory_summary).
    - With the performed character the conversation is live instead, spoken turn by turn
      (``_live_dialogue``): its actor types its dialogue turns, the AI side speaks in character from its
      own column only. Either way the result is the same transcript.
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

        participant_ids = [action.agent_id, target_ids[0]]
        live = any(agents[pid].is_performed for pid in participant_ids if pid in agents)
        # A live conversation takes one step, so it is spoken the step it starts; a longer act
        # would ask the actor whether to break off a talk not yet spoken.
        duration = action_duration_steps(action.action_type, action.estimated_steps, live_talk=live)
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
            talk_topic=action.talk_topic,
        )
        state.extra["target_id"] = target_ids[0]
        return state

    def _failed_result(self, action: AgentAction, step: int, outcome: str, location: str) -> ActionExecutionState:
        """A TALK feasibility failure as a doer-only ``create_failed`` execution."""
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.TALK,
            action_description=action.action_description or "talk",
            target=action.target,
        )
        doer_name = self._directory.agent_name(action.agent_id)
        failure = ActionResult(
            action=stub,
            expected_outcome=action.expected_outcome,
            # Quoted: unquoted, his first-person words render as "甲本想我去劝他…".
            outcome=scene_line(location, f"{doer_name}本想做「{action.action_description}」，{outcome}"),
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

        # Three-layer context per participant (relation + memories about other + about topic),
        # the topic recalled by what each side knows of the talk (``_topic_query``). A performed side
        # recalls nothing: its actor remembers for itself, and a recall would only touch memories
        # nobody reads.
        # Asked of the sides now, not recorded at start: a talk a restore resumes may have been
        # begun under another casting, and a performed side always speaks for itself.
        live = any(side is not None and side.is_performed for side in (initiator, target))
        initiator_ctx_r, target_ctx_r = await asyncio.gather(
            self._gather_context(
                None if live and initiator is not None and initiator.is_performed else initiator,
                target_id, target_name, _topic_query(state, is_initiator=True, live=live), step,
            ),
            self._gather_context(
                None if live and target is not None and target.is_performed else target,
                initiator_id, initiator_name, _topic_query(state, is_initiator=False, live=live), step,
            ),
            return_exceptions=True,
        )
        initiator_ctx = dict(_EMPTY_CTX) if isinstance(initiator_ctx_r, BaseException) else initiator_ctx_r
        target_ctx = dict(_EMPTY_CTX) if isinstance(target_ctx_r, BaseException) else target_ctx_r

        dialogue: list[dict] = []
        bystander_view = ""
        if initiator is not None and target is not None:
            if live:
                dialogue = await self._live_dialogue(
                    sides=(
                        _LiveSide(initiator, initiator_name, initiator_ctx, is_initiator=True),
                        _LiveSide(target, target_name, target_ctx, is_initiator=False),
                    ),
                    listener_ids=state.listener_ids,
                    purpose=state.purpose,
                    talk_topic=state.talk_topic,
                    expected_outcome=state.expected_outcome,
                    environment=environment,
                    agents=agents,
                    step=step,
                )
            else:
                scene = assemble_scene_context(
                    initiator_id, environment=environment, directory=self._directory, agents=agents,
                    visibility=SceneVisibility.GOD,  # functional judge
                ).text
                dialogue, bystander_view = await self._llm_full_dialogue(
                    initiator=initiator,
                    target=target,
                    initiator_name=initiator_name,
                    target_name=target_name,
                    purpose=state.purpose,
                    talk_topic=state.talk_topic,
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
            # The transcript as data for review, on every side's record of it: the prompts carry
            # it only as text.
            with annotate_call(dialogue=[{"speaker": t["speaker"], "line": t["line"]} for t in dialogue]):
                summary_i_r, summary_t_r, *listener_facts = await asyncio.gather(
                    self._memory_summary(
                        agent=initiator, other_id=target_id, other_name=target_name, transcript=transcript,
                        purpose=state.purpose, is_initiator=True,
                        now_step=step, environment=environment, expected_outcome=state.expected_outcome,
                    ),
                    self._memory_summary(
                        agent=target, other_id=initiator_id, other_name=initiator_name, transcript=transcript,
                        purpose=state.purpose, talk_topic=state.talk_topic, is_initiator=False,
                        now_step=step, environment=environment, expected_outcome=state.expected_outcome,
                    ),
                    *(
                        self._listener_memory(
                            agent=agents.get(pid), initiator_id=initiator_id,
                            target_id=target_id, transcript=transcript, talk_topic=state.talk_topic,
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
            agent_id=initiator_id, step=step,
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
                agent_id=target_id, step=step,
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

        ``action_description`` is this person's own act (``participant_action_desc``).
        ``summary=None`` is a null step: no memory, no
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
        environment: "EnvironmentSystem",
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        initiator_id = state.initiator_id
        target_id = state.extra.get("target_id", "")
        initiator_name = self._directory.agent_name(initiator_id)
        target_name = self._directory.agent_name(target_id)
        location = observe_location(environment, initiator_id)
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
            scene = assemble_scene_context(
                pid, environment=environment, directory=self._directory, agents=agents,
                include_header=False, voice=SituationVoice.FIRST,
                visibility=SceneVisibility.OWN_EYES,
            ).text
            relation_block = await self._present_relations_block(agent, environment)
            factual = await self._interrupt_summary(
                agent=agent, other_id=other_id, other_name=other_name, purpose=state.purpose,
                talk_topic=state.talk_topic, thought=thought, cause=cause, elapsed_label=elapsed_label,
                is_triggered=is_triggered, is_initiator=is_initiator, scene=scene,
                environment=environment, now_step=step, relation_block=relation_block,
            )
            stub = AgentAction(
                agent_id=pid, step=step, action_type=ActionType.TALK,
                action_description=participant_action_desc(self._directory, pid, state),
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

    async def _present_relations_block(self, agent: "Agent", environment: "EnvironmentSystem") -> str:
        """The agent's own relations with the people present, as its memory writes render them
        (``Agent.relation_context``): strangers and empty relations are left out."""
        loc = environment.get_body_location(agent.agent_id)
        # Cognitive tier only: an Npc holds no relation with anyone.
        present = [a for a in environment.agents_at(loc) if a != agent.agent_id]
        try:
            return await agent.relation_context(present)
        except Exception as exc:  # noqa: BLE001 — read failed: the record goes without relations (Rule 6)
            logger.warning(
                "relation_context_failed", extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            return ""

    def _seen_person(self, agent_id: str) -> str:
        """A co-present person as a roster referent: face to face, gender shows at a glance, and a
        first-person record written without it guesses the pronoun into memory."""
        identity = self._directory.agent_identity_map([agent_id]).get(agent_id)
        if identity is None:
            return self._directory.agent_name(agent_id)
        return person_referent(identity.name, identity.gender)

    async def _interrupt_summary(
        self,
        *,
        agent: "Agent",
        other_id: str,
        other_name: str,
        purpose: str,
        thought: str,
        cause: str,
        talk_topic: str = "",
        elapsed_label: str,
        is_triggered: bool,
        is_initiator: bool,
        scene: str,
        environment: "EnvironmentSystem",
        now_step: int = 0,
        relation_block: str = "",
    ) -> str:
        """First-person record of an interrupted conversation (one per participant)."""
        thought_part = format_interrupt_thought(thought)
        other = self._seen_person(other_id)
        initiator_line = f"是我主动找{other}谈的" if is_initiator else f"{other}来找我谈"
        opening = f"我主动找{other_name}" if is_initiator else f"{other_name}找我"
        # The initiator knows his own purpose; the addressee only the topic stated up front, since
        # the dialogue is written when the talk completes.
        known = purpose if is_initiator else talk_topic
        topic = f"就「{known}」" if known else ""
        # The trigger signal stays out (see base.interrupt).
        if cause:
            # Nobody chose to stop, so both share this sentence; the cause must be stated or the
            # model invents one from purpose.
            situation = f"这场谈话因{cause}而中断"
            fallback = f"{opening}{topic}谈到一半（{elapsed_label}），{situation}"
        elif is_triggered:
            situation = f"我中途撂下了这场谈话{thought_part}"
            fallback = f"{opening}{topic}谈到一半（{elapsed_label}），我中途撂下了这场谈话{thought_part}"
        else:
            situation = "对方突然中止了交谈"
            fallback = f"{opening}{topic}谈到一半，{situation}"

        scene_part = f"\n【现场】\n{scene}" if scene else ""
        relation_part = f"\n【我对在场者的关系】\n{relation_block}" if relation_block else ""
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场被打断的对话——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角,这场被打断的对话让我记住了什么、留下什么感受。始终用我自己的口吻。
提到我认识的人时,不妨带上其名字便于日后回想(惯以称谓相称的话,可附成「称谓（名字）」);这只是建议、不必生硬套用,陌生人用描述。
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【约束】
 - 只记下面给出的经过。没说谈的是什么,就不写谈了什么,也不替对方揣测来意。

""" + _PRESENT_RELATIONS_INPUT + """
【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "一句话（不超过36字）"}
"""
        doer_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        # Without a time/place anchor the model takes place names from purpose.
        situation_header = render_situation_header(
            situation_for(environment, agent.agent_id), voice=SituationVoice.FIRST,
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        happened = f"{initiator_line}，正{topic}交谈，已谈了{elapsed_label}，{situation}。"
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{doer_condition}

【刚刚发生的事】
{happened}{relation_part}{scene_part}

依上面说定的 JSON 格式记下这一场，只输出 JSON、不写任何多余内容。"""
        # Attribute the trace to this participant. talk_role lets review exempt the addressee
        # from the decision↔action consistency check.
        with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
            given_facts=GivenFacts()  # same source as user above
            .add("此刻何时何地", situation_header)
            .add("我的处境", doer_condition)
            .add("刚刚发生的事", happened)
            .add("我对在场者的关系", relation_block)
            .add("现场", scene),
            talk_role="initiator" if is_initiator else "addressee",
            action_owner=agent.personality.soul.name,
        ):
            data = await complete_json(
                self._llm.complete, system, user,
                temperature=0.7,
                max_tokens=output_budget(69),  # fact ≤36 chars ≈54 + key and braces ~15; est. ~69 tok
                failure_event="talk_interrupt_summary_failed",
                log_extra={"agent_id": agent.agent_id, "step": now_step},
                required=("fact",),
            )
        fact = str(data["fact"]).strip() if data is not None else ""
        return fact or fallback

    async def _gather_context(
        self,
        agent: "Agent | None",
        other_id: str,
        other_name: str,
        topic_query: str,
        step: int,
    ) -> dict:
        """Collect relation + memories (about the other agent, the topic, his recent past). Each
        read failure only empties its own column; an empty topic query recalls no topic."""
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
        # "What I know of him" filters by other_id, then ranks by his name: the whole of him, not
        # the topic's slice, which the next column holds. Don't retrieve(other_name) unfiltered: in
        # a single-theme world that recalls the topic, not the person. Empty means "I don't know him".
        about_other: list = []
        try:
            about_other = await agent.memory_system.recall_about_agent(
                other_id, other_name, current_step=step, top_k=5, exclude_ids=seen,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: drop this column, continue
            logger.warning(
                "social_memory_retrieve_failed",
                extra={"agent_id": agent.agent_id, "target_id": other_id, "error": str(exc)},
            )
        seen |= {m.id for m in about_other}
        about_topic: list = []
        try:
            about_topic = [] if not topic_query else await agent.memory_system.retrieve(
                topic_query, current_step=step, top_k=5, exclude_ids=seen,
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
        talk_topic: str = "",
        initiator_ctx: dict,
        target_ctx: dict,
        scene: str,
        now_step: int,
    ) -> tuple[list[dict], str]:
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
        # What B knows of why A came: only the topic A states up front, never the purpose.
        stated = (
            f"{initiator_name}一开口就向{target_name}点明了来意：「{talk_topic}」；其余事情{target_name}并不知道。"
            if talk_topic else f"{initiator_name}没有事先点明来意，{target_name}起初并不知道{initiator_name}为何而来。"
        )
        system = _DIALOGUE_SYSTEM
        user = f"""\
【A：{initiator_name}】
{lane_a}

【B：{target_name}】
{lane_b}

【对话目的】（以下是{initiator_name}单方面的说法与意图，既非已定的事实，也非这场交谈该有的结局）
{initiator_name}想就「{purpose}」与{target_name}交谈。{expected_part}
{stated}{scene_part}

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
            .add("开口点明的来意", talk_topic)
            .add("发起者的期望", expected_outcome)
            .add("现场", scene)
            .add("轮数上限", turns)
        )
        # One semantic retry (as DecisionEngine._llm_select) for broken JSON or an
        # unattributable line; same prompt, same bar, never filling in lines.
        for attempt in (1, 2):
            with annotate_call(
                given_facts=facts,
                dialogue_lanes={
                    "a": {"name": initiator_name, "text": lane_a},
                    "b": {"name": target_name, "text": lane_b},
                },
            ):
                data = await complete_json(
                    self._llm.complete, system, user,
                    temperature=0.8,
                    # Per turn {speaker, line ≤60 chars ≈ 90 tok} ≈ 100 tok, plus reason (≤80 chars
                    # ≈ 120 tok), observation (≤25 chars ≈ 38 tok) and the three fields' structure
                    # (~15 tok).
                    max_tokens=output_budget(turns * 100 + 173),
                    failure_event="talk_dialogue_failed",
                    log_extra={"initiator": initiator.agent_id, "attempt": attempt},
                )
            if data is None:
                continue
            parsed = self._parse_dialogue(
                data,
                (initiator.agent_id, initiator_name),
                (target.agent_id, target_name),
            )
            if parsed:
                observation = " ".join(str(data.get("observation", "")).split()).strip()
                return parsed, observation
            note_active_call_adoption(False, "transcript_unattributable")
            logger.warning(
                "talk_dialogue_failed",
                extra={
                    "initiator": initiator.agent_id, "attempt": attempt,
                    "error": "transcript_unattributable",
                },
            )
        return [], ""

    async def _live_dialogue(
        self,
        *,
        sides: "tuple[_LiveSide, _LiveSide]",
        listener_ids: list[str],
        purpose: str,
        talk_topic: str,
        expected_outcome: str,
        environment: "EnvironmentSystem",
        agents: dict[str, "Agent"],
        step: int,
    ) -> list[dict]:
        """The live conversation, the initiator first, as turns like ``_llm_full_dialogue``'s.

        It ends when a side ends it, when the turns run out, or when an AI turn can't be had: what
        was said stands and nothing is filled in (Rule 1). Each turn, and the end, is told to the
        performed side as it is spoken.
        """
        dialogue: list[DialogueTurn] = []
        # Per AI side: what it recalled as the conversation went on (``_recall_mid_talk``).
        recalled: dict[str, list["Memory"]] = {side.agent.agent_id: [] for side in sides}
        spoken = 0
        while spoken < _LIVE_MAX_TURNS:
            me, other = sides[spoken % 2], sides[1 - spoken % 2]
            conversation = Conversation(
                other=BriefPerson(
                    agent_id=other.agent.agent_id, name=other.name,
                    gender=other.agent.personality.soul.gender, present=True,
                ),
                dialogue=tuple(dialogue),
                turns_left=_LIVE_MAX_TURNS - spoken,
                last=spoken == _LIVE_MAX_TURNS - 1,
                max_turn_chars=_LIVE_ACTOR_TURN_CHARS,
                listener_ids=tuple(_present_listeners(
                    listener_ids, environment=environment, agents=agents,
                    anchor_id=me.agent.agent_id,
                    exclude={side.agent.agent_id for side in sides},
                )),
            )
            if me.agent.is_performed:
                answer = await me.agent.actor_dialogue_turn(step=step, conversation=conversation)
                line, end = answer.line, answer.end
            else:
                mine = recalled[me.agent.agent_id]
                mine.extend(await self._recall_mid_talk(me, dialogue, mine, step))
                reply = await self._ai_turn(
                    me=me, conversation=conversation, purpose=purpose, talk_topic=talk_topic,
                    expected_outcome=expected_outcome, recalled=mine, environment=environment,
                    agents=agents, step=step,
                )
                line, end = reply if reply is not None else ("", True)
            turn = DialogueTurn(speaker_id=me.agent.agent_id, speaker=me.name, line=line)
            if line:
                dialogue.append(turn)
            end = end or conversation.last or not line
            for side in sides:
                side.agent.dialogue_turn_spoken(step=step, dialogue_turn=turn, end=end)
            spoken += 1
            if end:
                break
        return [dataclasses.asdict(t) for t in dialogue]

    async def _recall_mid_talk(
        self, me: "_LiveSide", dialogue: "list[DialogueTurn]", recalled: "list[Memory]", step: int,
    ) -> "list[Memory]":
        """What the other side's last line brings back to ``me``, beyond what it already holds.
        Empty on a read failure (Rule 6), a line too short to carry a subject, or a full quota."""
        if not dialogue or dialogue[-1].speaker_id == me.agent.agent_id:
            return []
        line = dialogue[-1].line
        room = min(_LIVE_RECALL_PER_TURN, _LIVE_RECALL_TOTAL - len(recalled))
        if len(line) < _LIVE_RECALL_MIN_CHARS or room <= 0:
            return []
        held = {m.id for column in ("about_other", "about_topic", "recent") for m in me.ctx.get(column, [])}
        held |= {m.id for m in recalled}
        try:
            return await me.agent.memory_system.retrieve(
                line, current_step=step, top_k=room, exclude_ids=held,
                score_floor=_LIVE_RECALL_SCORE_FLOOR,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: recall nothing this turn
            logger.warning(
                "talk_live_recall_failed",
                extra={"agent_id": me.agent.agent_id, "step": step, "error": str(exc)},
            )
            return []

    async def _ai_turn(
        self,
        *,
        me: "_LiveSide",
        conversation: Conversation,
        purpose: str,
        talk_topic: str,
        expected_outcome: str,
        recalled: "list[Memory]",
        environment: "EnvironmentSystem",
        agents: dict[str, "Agent"],
        step: int,
    ) -> "tuple[str, bool] | None":
        """An AI side's dialogue turn, in character, from its own column only: ``(line, end)``,
        or ``None`` when the call fails."""
        agent = me.agent
        other_name = conversation.other.name
        situation_header = render_situation_header(
            situation_for(environment, agent.agent_id), voice=SituationVoice.FIRST,
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=step, seconds_per_step=self._seconds_per_step,
        )
        scene = assemble_scene_context(
            agent.agent_id, environment=environment, directory=self._directory, agents=agents,
            include_header=False, voice=SituationVoice.FIRST, visibility=SceneVisibility.OWN_EYES,
        ).text
        scene_part = f"\n\n【我眼前】\n{scene}" if scene else ""
        known = _format_context(
            me.ctx, "我", other_name,
            now_step=step, seconds_per_step=self._seconds_per_step,
            world_start_second_of_day=self._world_start_second_of_day,
        )
        recalled_lines = [
            c for c in render_memory_lines(
                recalled, now_step=step, seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day)
            if c
        ]
        if recalled_lines:
            known = "\n".join([
                *([known] if known else []),
                f"- 话说到这里，我想起的（{MEMORY_ORDER_HINT}）：",
                *(f"  - {c}" for c in recalled_lines),
            ])
        known_part = f"\n\n【我知道的】\n{known}" if known else ""
        if me.is_initiator:
            expected_part = f"我原本期望：{expected_outcome}" if expected_outcome else ""
            opening = f"是我主动找{other_name}谈的，我想谈的是「{purpose}」。{expected_part}"
        elif talk_topic:
            opening = f"{other_name}来找我说话，来意是「{talk_topic}」。"
        else:
            opening = f"{other_name}来找我说话。"
        said = "\n".join(
            f"{'我' if t.speaker_id == agent.agent_id else t.speaker}：{t.line}"
            for t in conversation.dialogue
        ) or "（还没人开口，我先说。）"
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{condition}{scene_part}{known_part}

【这场对话】
{opening}
到现在说过的话：
{said}

依上面说定的 JSON 格式，先判断这场话说完了没有，再说出我这一轮的话；只输出 JSON、不写任何多余内容。"""
        facts = (
            GivenFacts()
            .add("此刻何时何地", situation_header)
            .add("我的处境", condition)
            .add("我眼前", scene)
            .add("我知道的", known)
            .add("这场对话", opening)
            .add("到现在说过的话", said)
        )
        call = (
            self._llm.complete_with_retry if agent.is_main_character else self._llm.complete
        )
        # Only a conversation with a performed character is spoken turn by turn.
        with observe_actor(), observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
            given_facts=facts, action_owner=agent.personality.soul.name,
        ):
            data = await complete_json(
                call, _LIVE_TURN_SYSTEM, user,
                temperature=0.8,
                # thought ≤60 chars ≈90 + end ≈3 + line ≤60 chars ≈90 + three fields'
                # structure ≈15; est. ~200 tok.
                max_tokens=output_budget(200),
                failure_event="talk_live_turn_failed",
                log_extra={"agent_id": agent.agent_id, "step": step},
            )
        if data is None:
            return None
        if "line" not in data and "end" not in data:
            note_active_call_adoption(False, "missing_dialogue_turn")
        return str(data.get("line", "")).strip(), coerce_bool(data.get("end"), False)

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
            turns.append(dataclasses.asdict(DialogueTurn(speaker_id=who[0], speaker=who[1], line=line)))
        return turns

    async def _memory_summary(
        self,
        *,
        agent: "Agent | None",
        other_id: str,
        other_name: str,
        transcript: str,
        purpose: str,
        is_initiator: bool,
        environment: "EnvironmentSystem",
        talk_topic: str = "",
        now_step: int = 0,
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
        relation_part = f"\n【我对在场者的关系】\n{rels}" if rels else ""

        situation_header = render_situation_header(
            situation_for(environment, agent.agent_id), voice=SituationVoice.FIRST,
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        other = self._seen_person(other_id)
        initiator_line = f"是我主动找{other}谈的" if is_initiator else f"{other}来找我谈"
        expected_part = f"\n原本期望：{expected_outcome}" if (is_initiator and expected_outcome) else ""
        # The initiator knows his own purpose; the addressee the topic stated up front, if any, and
        # what was said.
        if is_initiator:
            opening = f"{initiator_line}，意图是「{purpose}」。"
        elif talk_topic:
            opening = f"{initiator_line}，来意是「{talk_topic}」。"
        else:
            opening = f"{initiator_line}。"
        # other_name stays out of system (prefix cache); the reminder at the end of user names him.
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场刚结束的对话——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角:这场对话达成我在意的目的了吗?发生了什么实质的事?谈完之后我对对方的整体观感有何变化。始终用我自己的口吻。

""" + _PRESENT_RELATIONS_INPUT + """
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
        doer_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{doer_condition}

【刚结束的对话】
{opening}{relation_part}{expected_part}
对话记录：
{transcript}

依上面说定的 JSON 格式记下这一场（relation 反映我对{other_name}的观感变化），只输出 JSON、不写任何多余内容。"""
        # Attribute to this participant; the addressee also gets conscripted_input for review.
        call_annotations = {
            "given_facts": GivenFacts()  # same source as user above
            .add("此刻何时何地", situation_header)
            .add("我的处境", doer_condition)
            .add("刚结束的对话", opening)
            .add("我对在场者的关系", rels)
            # The addressee never saw the initiator's expectation; declaring it would blind
            # the leak review.
            .add("发起者的期望", expected_outcome if is_initiator else "")
            .add("对话记录", transcript),
            "talk_role": "initiator" if is_initiator else "addressee",
            "action_owner": agent.personality.soul.name,
        }
        if not is_initiator:
            call_annotations["conscripted_input"] = opening
        with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
            **call_annotations
        ):
            data = await complete_json(
                self._llm.complete, system, user,
                temperature=0.7,
                # fact (≤100 chars ≈ 150 tok, including a possible bracketed date) + success
                # + why (≤20 chars ≈ 30) + relation; est. ~211 tok.
                max_tokens=output_budget(211),
                failure_event="talk_memory_summary_failed",
                log_extra={"agent_id": agent.agent_id, "step": now_step},
                # The record the verdict is read off; a stand-in would remember a talk nobody
                # recorded. The talk itself happened, so a missing verdict claims the least.
                required=("fact",),
            )
        if data is None:
            return None
        succeeded = coerce_bool(data.get("success"), False)
        relation_dir = parse_relation_direction(str(data.get("relation", "neutral"))).value
        # Unsliced: the prompt caps it.
        why = str(data.get("why", "")).strip() if not succeeded else ""
        return str(data["fact"]).strip(), succeeded, relation_dir, why

    async def _listener_memory(
        self,
        *,
        agent: "Agent | None",
        initiator_id: str,
        target_id: str,
        transcript: str,
        environment: "EnvironmentSystem",
        talk_topic: str = "",
        now_step: int = 0,
    ) -> str | None:
        """A listener's first-person record of what he heard: no success verdict or relation,
        since he said nothing. Not the raw transcript, which would embed the whole dialogue.

        ``None`` on failure, and no memory is written: a content-free line only dilutes recall.
        """
        if agent is None:
            return None

        doer_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        situation_header = render_situation_header(
            situation_for(environment, agent.agent_id), voice=SituationVoice.FIRST,
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        speakers = f"说话的是{self._seen_person(initiator_id)}与{self._seen_person(target_id)}"
        # A listener knows the topic its initiator stated up front, as the addressee does.
        stated = f"{self._seen_person(initiator_id)}发起的对话，来意是「{talk_topic}」。" if talk_topic else ""
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
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{doer_condition}

【我在旁边听到的对话】
{speakers}，我在一旁听着。{stated}
对话记录：
{transcript}

依上面说定的 JSON 格式记下我听到了什么，只输出 JSON、不写任何多余内容。"""
        # Attribute to the listener, not the initiator.
        with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
            given_facts=GivenFacts()  # same source as user above
            .add("此刻何时何地", situation_header)
            .add("我的处境", doer_condition)
            .add("我在旁边听到的", speakers)
            .add("说话人点明的来意", talk_topic)
            .add("对话记录", transcript),
            talk_role="listener", action_owner=agent.personality.soul.name,
        ):
            data = await complete_json(
                self._llm.complete, system, user,
                temperature=0.7,
                # fact ≤60 chars ≈ 90 tok + JSON structure ~5; est. ~95 tok.
                max_tokens=output_budget(95),
                failure_event="talk_listener_memory_failed",
                log_extra={"agent_id": agent.agent_id, "step": now_step},
                required=("fact",),
            )
        return str(data["fact"]).strip() if data is not None else None
