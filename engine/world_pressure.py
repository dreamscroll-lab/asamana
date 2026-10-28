"""Per-agent external pressure evaluator.

One LLM call **per agent that has an acute signal of its own** (``PerceivedSignals``: an inbox
message, a perceivable broadcast, an ambient observation, or another's act that landed on it).
Each prompt contains ONLY what that agent knows: this step's signals, his own memories, and those
present as he perceives them. Information asymmetry is enforced *structurally*, not by instruction.

External pressure is a push from something that happened outside him this step. Signals are
numbered and each goal names the ones it comes from, so a goal that a memory alone would justify
has nothing to cite and is dropped; a pressure that only several signals together amount to is one
goal citing them all. Those present are reference, never a signal: being there is a state, not an
event, and an arrival or departure already reaches him as an ambient observation. Don't number them
among the signals: a goal citing someone's presence is how an old arrangement or his own plan gets
passed off as pressure.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, NamedTuple

from agent.motivation import ExternalDriveType, ExternalGoal
from agent.need import NeedType
from agent.personality import SECRET_LABEL
from agent.relation import format_relation_block
from core.context import GivenFacts, annotate_active_call, annotate_call, note_active_call_adoption, observe_stage
from core.interfaces.llm import IndexedRef, LLMMessage, LLMScene, LLMRouter, extract_json, output_budget
from core.interfaces.perception import PerceivedSignals, Situation
from core.interfaces.severity import Severity
from core.interfaces.trace import Stage
from core.interfaces.urgency import Urgency, parse_urgency
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    MEMORY_ORDER_HINT,
    EXTERNAL_DRIVE_TYPE_DEFINITION,
    render_condition,
    SituationVoice,
    URGENCY_SCALE_DESCRIPTION,
    AMBIENT_TAG,
    MESSAGE_TAG,
    broadcast_tag,
    person_referent,
    relation_legend,
    render_situation_header,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from agent.agent import Agent
    from core.interfaces.message import Message
    from core.interfaces.perception import AmbientEvent, Broadcast, SpatialPerception

logger = get_logger(__name__)


# What landed on the target, in the source list and on each such line alike.
_ACTED_UPON_TAG = "落到该角色身上的事"

_DRIVE_TYPE_MAP: dict[str, ExternalDriveType] = {t.value: t for t in ExternalDriveType}
_NEED_TYPE_MAP: dict[str, NeedType] = {n.value: n for n in NeedType}
# Severity is a code-layer scale: plain words in the prompt.
_SEVERITY_WORD: dict[Severity, str] = {
    Severity.LOW: "轻微",
    Severity.MEDIUM: "一般",
    Severity.HIGH: "重大",
}

# Derived from the enum so prompt options and parsing can't drift apart.
_NEED_OPTIONS = "；".join(f'"{n.value}"={n.description}' for n in NeedType)
_DRIVE_TOKENS = " | ".join(f'"{t.value}"' for t in ExternalDriveType)

# Persisted and read back into prompts (see _parse_goal); already the widest of the sibling goal
# fields. Both the field description and the output schema interpolate it: never write the number
# twice. Changing it requires recomputing _MAX_TOKENS.
_GOAL_TEXT_MAX_CHARS = 40
# Room to weigh each signal against his recent experience, not only its face value.
_REASON_MAX_CHARS = 120
# Per CLAUDE.md's max_tokens buffer rule: estimate ≈ reason (120 chars×1.5=180)
# + 3×(text 40 chars×1.5=60 + signals list ~6 + urgency/drive/need 9 + structure 25) + wrapper 10
# ≈ 490.
# Coverage is guarded by test_the_token_budget_covers_the_caps_the_prompt_declares.
_MAX_TOKENS = output_budget(490)

_SYSTEM_PROMPT = f"""\
你是第三者视角的外部压力分析者。给定单个角色（下文称"该角色"）、它**此刻感知到的信号**，以及帮你读懂这些信号的参考
（它的人设、正在做的事、近来的经历、同处一地的人），客观评估此刻有哪些**外部力量**正在推动该角色行动，输出其外部目标列表。

【压力的唯一来源 = 此刻感知到的信号】
 - 外部压力**只能源自下面编号的信号 #N**（收到的消息 / 可感知的广播 / 环境观察 / {_ACTED_UPON_TAG}），每个目标都必须指明它出自哪些信号。
 - 几条信号叠加才构成的压力，作为一个目标列出这几条信号，不要拆成多个目标；只列真正促成它的信号，不要把所有信号都列上。
 - 角色的人设、正在做的事、近来的经历、同处一地的人，只是供你判断"这条信号对这个人意味着什么"的**参考**，**它们本身不是压力来源**；
 - 角色自身的内在动机、长期目标**不属于外部压力**（那是其内在需求，由别处处理，不在此输出）。
 - 同处一地的人作为参考信息，自己本身不是外界压力的源头。

【强度锚定信号本身，必须遵守】
 - urgency 必须与**信号本身的烈度**匹配：模糊的、遥远的、并未指向该角色的迹象只构成 low/normal。
 - **不要因为该角色内在动机强烈或处境特殊，就把一个微弱信号拔高成 high/critical。**

【依据经历读懂信号】
 - 用该角色近来的经历判断信号的来龙去脉：它是否在回应该角色自己先前做过的事、该角色是否已在应对、是否是该角色先前没理会的事再次逼近等等但不限于只有这些场景。
 - 该角色已在应对的事（正在做、或刚做过），或者压力和【正在做的事】内容或者方向一样，不要再给出同样的压力；先前没理会、如今再次到来的，可视为加码。
 - 经历里的旧事（早先接到的安排、该角色原本的打算）若这一步没有信号牵动，不构成压力；不得凭经历推断信号里没有的内容。

【默认无压力，不要凑数】
 - 外部压力是例外而非常态。只有当信号**确实**对该角色构成可行动的外部推力时，才输出目标；
 - 若该角色此刻并未真正承压，**直接输出空数组 []**——这是完全正常且被期望的结果。
 - 若信号与角色无关且也不是角色应该关注的内容，即使信号来源与该角色有强烈的关系，也直接输出空数组。
 - 不要为了"显得有反应"或凑满名额而虚构、夸大压力；目标数量为 0 也很正常，**无需凑满 3 个**。
（注意：真有压力时仍须如实输出——这不是要你压制真实压力，而是不要无中生有。）

【依据关系判断】
 - 同处一地的人附带"该角色对其的关系"；收到的消息也会附上"该角色对发送者的关系"。据此判断亲疏：
 - 不得仅因某人在场或双方关系亲近，就推断其完全可信任，也不能因为关系不亲近，就推断一定产生威胁。好感度和信任度是判断背景依据，具体的压力需要结合具体的上下文判定。
 - 若该角色实际感知到对方背叛、胁迫或伤害等具体信号，可以据此判断威胁并生成相称的防范目标；不得虚构敌意或事件。
 - 同处一地的人是否会增强或降低压力信号，取决于该角色对其的关系、近来的经历和此刻的处境，需要结合具体情况判定。

【信息边界】
 - 你只会看到该角色所知的内容；绝不替它推断它无从得知的信息。
 - 对同处一地的人，只以所给的关系、过往和该角色的经历为已知；未给出的信息，一律视为该角色不知道，不得揣测。

【约束】
 - 注意时间规律和常理，即深夜对于一般事件的压力敏感度下降，但对突发事件的敏感度会上升。
 - 处于行动中时，对于周边环境的压力敏感度会下降。
 - 不要把没有发生过的事情或者可能发生的事情当作客观事实。
 - 不要过度推测、过度解读信号来给该角色制造压力。
{ABSOLUTE_TIME_RULE}

【说明】
关系各栏的含义：{relation_legend()}

每个角色最多 3 个目标，按紧急程度降序。每个目标包含：
- signals: 该压力出自哪些信号——【此刻感知到的信号】里的序号数组（与输入中的 #N 对应的整数），至少一个
- text: 指引该角色行动的一句话（中文，≤{_GOAL_TEXT_MAX_CHARS}字，用该角色当下视角；**不得使用事件的事后命名，也不要预测未来走向，也不要臆想当前情景。**）
- {URGENCY_SCALE_DESCRIPTION}
- {EXTERNAL_DRIVE_TYPE_DEFINITION}，取值之一：{_DRIVE_TOKENS}
- related_need: 该压力最终牵动的内在需求，取值之一（含义如下）或 null：{_NEED_OPTIONS}。\
通常 threat→safety，authority→esteem 或 safety，obligation→social 或 esteem，event 视内容而定。\
**urgency 为 high 或 critical 的目标必须给出明确 related_need（不可为 null）**。

只输出纯 JSON 对象，不要任何说明文字或 markdown。**先在 reason 里分析、再据此给 goals**——\
reason 必须是第一个键（先想后做）：
{{"reason": "<逐一审视每条信号，结合该角色的人设、正在做的事、近来经历与关系，判断这条信号对该角色意味着什么、是否真的与该角色有关或者该角色需要关注的、\
该角色是否已在应对、是否真构成可行动的外部压力、为什么；这些参考只是判断信号轻重的依据、本身不算压力源。第三者视角，≤{_REASON_MAX_CHARS}字。\
注意这是判断依据，goals 必须由它推出，不是先列 goals 再补理由>", "goals": [{{"signals": [1], "text": "……（≤{_GOAL_TEXT_MAX_CHARS}字）", "urgency": "high", "drive_type": "threat", "related_need": "safety"}}]}}
"""


class _Signal(NamedTuple):
    """One numbered signal: its prompt line, the person it names (a goal citing it carries them
    as a source), and its text as a recall cue."""

    line: str
    person_id: str | None
    cue: str


class WorldPressureEvaluator:
    """Compute per-agent external pressure with structural information boundaries."""

    def __init__(self, *, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def evaluate(
        self,
        *,
        agents: dict[str, "Agent"],
        signals: "Mapping[str, PerceivedSignals]",
        world_time_label: str,
        step: int,
        agent_spatials: "dict[str, SpatialPerception] | None" = None,
        doings: "Mapping[str, str] | None" = None,
    ) -> dict[str, list[ExternalGoal]]:
        """Return external goals per agent, from what reached each (``signals``).

        ``doings``: what each agent in the middle of an action is doing, in his own view.
        Agents without an acute signal of their own are skipped and omitted from the result.
        """
        tasks = [
            self._evaluate_one(
                agent_id=agent_id,
                agent=agent,
                signals=signals.get(agent_id, PerceivedSignals()),
                spatial=agent_spatials.get(agent_id) if agent_spatials else None,
                world_time_label=world_time_label,
                step=step,
                doing=(doings or {}).get(agent_id, ""),
            )
            for agent_id, agent in agents.items()
            # Otherwise world-wide death notices would get evaluated for the corpse. External
            # goals are motivation, which a character performed by a person takes from them.
            if agent.is_active and agent.reads_world_pressure
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, list[ExternalGoal]] = {}
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("world_pressure_agent_eval_failed", extra={"error": str(result)})
                continue
            agent_id, goals = result
            if goals is not None:
                out[agent_id] = goals
        return out

    async def _evaluate_one(
        self,
        *,
        agent_id: str,
        agent: "Agent",
        signals: PerceivedSignals,
        spatial: "SpatialPerception | None",
        world_time_label: str,
        step: int,
        doing: str,
    ) -> tuple[str, list[ExternalGoal] | None]:
        """Evaluate one agent. Returns (agent_id, goals) or (agent_id, None) when skipped."""
        if not signals:
            return agent_id, None
        inbox = list(signals.messages)

        visible_ids = list(spatial.visible_agent_ids) if spatial is not None else []
        co_located = await self._co_located(agent, spatial, visible_ids)
        # Co-located senders' relations are already in the 【同处一地的人】 block; only fill the gap.
        sender_relations = await self._sender_relations(
            agent, inbox, exclude_ids=set(visible_ids),
        )
        numbered = self._signals(
            inbox=inbox,
            broadcasts=list(signals.broadcasts),
            ambient=list(signals.ambient),
            acted_upon=list(signals.acted_upon),
            sender_relations=sender_relations,
            visible_ids=set(visible_ids),
        )
        recent_past = await agent.pressure_recent_past(
            step, cues=[s.cue for s in numbered], present=[c["id"] for c in co_located],
        )
        situation_header = render_situation_header(
            Situation(
                location_view=spatial.location_view if spatial is not None else None,
                time_label=world_time_label,
            ),
            voice=SituationVoice.THIRD,
        )
        prompt = self._build_agent_prompt(
            agent,
            signals=numbered,
            co_located=co_located,
            recent_past=recent_past,
            doing=doing,
            situation_header=situation_header,
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]
        # Same sources as the prompt above, so a review weighs goals against what the model saw.
        facts = (
            GivenFacts()
            .add("此刻感知到的信号", [s.line for s in numbered])
            .add("近来的经历", recent_past)
            .add("同处一地的人", [self._co_located_line(c) for c in co_located])
            .add("正在做的事", doing)
        )
        try:
            # Set inside each gather task so traces attribute per person without crosstalk.
            with observe_stage(Stage.PRESSURE, agent_id=agent_id), annotate_call(given_facts=facts):
                response = await self._llm_router.complete(
                    # The budget follows the caps the prompt declares; see _MAX_TOKENS.
                    LLMScene.WORLD_PRESSURE, messages, temperature=0.2, max_tokens=_MAX_TOKENS,
                    json_mode=True,
                )
        except Exception as exc:
            logger.warning("world_pressure_eval_failed", extra={"agent_id": agent_id, "error": str(exc)})
            return agent_id, []
        return agent_id, self._parse_agent_goals(
            response.content, signal_people=[s.person_id for s in numbered], agent_id=agent_id,
        )

    # ------------------------------------------------------------------

    async def _co_located(
        self,
        agent: "Agent",
        spatial: "SpatialPerception | None",
        visible_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Each co-located agent as the evaluated agent knows them: what perception shows of
        them, and his own relation to them.

        Don't add the other's soul (role, traits, background): the result reaches his goals, so
        it may weigh only what he knows. Read-only (load_existing, never create).
        """
        out: list[dict[str, Any]] = []
        for vid in visible_ids:
            rel = await agent.relation_system.load_existing(vid)
            presence = spatial.visible_agents.get(vid) if spatial is not None else None
            who = presence.identity if presence is not None else None
            out.append({
                "id": vid,
                "name": (who.name if who else "") or "某人",
                "gender": who.gender if who else "",
                "condition": presence.condition if presence is not None else "",
                "trust": round(rel.trust_objective, 2) if rel is not None else None,
                "affection": round(rel.affection_objective, 2) if rel is not None else None,
                "labels": list(rel.labels) if rel is not None else [],
                "history": rel.history_summary if rel is not None else "",
            })
        return out

    async def _sender_relations(
        self,
        agent: "Agent",
        inbox: list["Message"],
        *,
        exclude_ids: set[str],
    ) -> dict[str, str]:
        """sender_id → this character's own relation to that sender (read-only).

        The same "come at once" is a request from an ally and a threat from a nemesis. Filter on
        the message's own ``sender_is_agent``, not an id list: missing one non-agent id renders
        "关系未明确" (relation unclear) for a relation that can never exist.
        """
        out: dict[str, str] = {}
        for msg in inbox:
            sid = getattr(msg, "sender_id", "") or ""
            if msg.anonymous or not sid or not msg.sender_is_agent or sid in exclude_ids or sid in out:
                continue
            rel = await agent.relation_system.load_existing(sid)
            out[sid] = self._format_relation({
                "trust": round(rel.trust_objective, 2) if rel is not None else None,
                "affection": round(rel.affection_objective, 2) if rel is not None else None,
                "labels": list(rel.labels) if rel is not None else [],
            })
        return out

    @staticmethod
    def _format_relation(c: dict[str, Any]) -> str:
        """Render via ``format_relation_block``; the scale legend is in the system prompt."""
        trust, affection = c.get("trust"), c.get("affection")
        if trust is None and affection is None:
            labels = " | ".join(c.get("labels") or [])
            return f"[{labels}]" if labels else "关系未明确"
        return format_relation_block(
            labels=c.get("labels") or [], trust=trust, affection=affection, include_legend=False,
        )

    @classmethod
    def _co_located_line(cls, c: dict[str, Any]) -> str:
        who = person_referent(c["name"], c.get("gender", ""), *((c["condition"],) if c.get("condition") else ()))
        history = f"；与该角色的过往：{c['history']}" if c.get("history") else ""
        return f"{who} — 关系：{cls._format_relation(c)}{history}"

    @staticmethod
    def _signals(
        *,
        inbox: list["Message"],
        broadcasts: "list[Broadcast]",
        ambient: "list[AmbientEvent]",
        acted_upon: "Sequence[str]" = (),
        sender_relations: dict[str, str],
        visible_ids: set[str],
    ) -> list[_Signal]:
        """This step's signals in prompt order, each with the person it names.

        A signal names a person only where it says who: a signed letter's sender, or the one
        visible doer of an act seen. An anonymous letter names no one, or a goal would bind it to
        the sender the letter withholds.
        """
        out: list[_Signal] = []
        for bc in broadcasts:
            out.append(_Signal(
                f"{broadcast_tag(bc.location_scope)}（{_SEVERITY_WORD[bc.severity]}）：{bc.content}",
                None, bc.content,
            ))
        for msg in inbox:
            sender_name = msg.sender_name
            line = f"收到来自{sender_name}的{MESSAGE_TAG}：{msg.content}"
            rel = None if msg.anonymous else sender_relations.get(msg.sender_id)
            if rel:
                line += f"（该角色对{sender_name}的关系：{rel}）"
            sender = None if msg.anonymous or not msg.sender_id else msg.sender_id
            out.append(_Signal(line, sender, f"{sender_name}：{msg.content}"))
        for ev in ambient:
            doers = [d for d in ev.doer_agent_ids if d in visible_ids]
            out.append(_Signal(
                f"{AMBIENT_TAG}：{ev.content}", doers[0] if len(doers) == 1 else None, ev.content,
            ))
        # His own memory of it, so first person: 「我」 is the target.
        for fact in acted_upon:
            out.append(_Signal(f"{_ACTED_UPON_TAG}（该角色自己的记忆，「我」即该角色）：{fact}", None, fact))
        return out

    def _build_agent_prompt(
        self,
        agent: "Agent",
        *,
        signals: Sequence[_Signal],
        co_located: "Sequence[dict[str, Any]]" = (),
        recent_past: "Sequence[str]" = (),
        doing: str = "",
        situation_header: str,
    ) -> str:
        soul = agent.personality.soul
        activity = agent.personality.state.activity_status.label
        # An empty string is omitted by person_referent, so no condition leaves no trace.
        condition = render_condition(agent.personality.state.condition)
        traits = "、".join(soul.core_traits) if soul.core_traits else "（未明确）"
        values = "、".join(soul.core_values) if soul.core_values else "（未明确）"
        background = soul.background or "（未明确）"
        # Only the target's own secret: a co-present person's would let the pressure assessment
        # weigh what the target cannot know, and its result reaches the target's goals.
        secret_line = f"\n  {SECRET_LABEL}：{soul.secret}" if soul.secret else ""
        doing_line = f"\n  正在做的事：「{doing}」" if doing else ""
        header_part = f"{situation_header}\n\n" if situation_header else ""

        lines: list[str] = [f"""\
{header_part}【该角色（仅供识人参考，不是压力来源）】
{person_referent(soul.name, soul.gender, soul.role or '无', activity, condition)}
  性格特点：{traits}
  价值观：{values}
  背景：{background}{secret_line}{doing_line}
"""]
        if recent_past:
            lines.append(
                f"【该角色近来的经历（该角色自己的记忆，「我」即该角色；{MEMORY_ORDER_HINT}；仅供读懂信号，不是压力来源）】"
            )
            lines.extend(f"- {m}" for m in recent_past)
            lines.append("")
        if co_located:
            lines.append("【同处一地的人（该角色眼中的他们、该角色对其的关系与过往；仅供读懂信号，在场本身不是压力来源）】")
            lines.extend(f"- {self._co_located_line(c)}" for c in co_located)
            lines.append("")

        lines.append("【此刻感知到的信号（外部压力的唯一来源）】")
        lines.extend(f"#{i} {s.line}" for i, s in enumerate(signals, 1))
        if not signals:
            lines.append("（无）")

        lines.append("")
        lines.append(
            "请以第三者视角根据上面的信号客观评估该角色承受的外部压力，先给出分析 reason（先想），再据此给出 goals（后做），"
            "每个目标用 signals 指明它出自哪些信号；无压力时同样输出 reason，goals 输出空数组 []。"
        )
        return "\n".join(lines)

    def _parse_agent_goals(
        self, raw: str, *, signal_people: "Sequence[str | None]", agent_id: str = "?",
    ) -> list[ExternalGoal]:
        """Goals from the model's answer; ``signal_people[i]`` is whom signal #i+1 names.

        A goal's sources are everyone its signals name: a joint pressure keeps all of them.
        """
        signal_ref = IndexedRef(str(i) for i in range(len(signal_people)))
        try:
            data: Any = extract_json(raw)
        except ValueError:
            note_active_call_adoption(False, "invalid_json")
            logger.warning(
                "world_pressure_parse_failed",
                extra={"agent_id": agent_id, "raw_prefix": raw[:200]},
            )
            return []
        raw_goals = data.get("goals") if isinstance(data, dict) else None
        if not isinstance(raw_goals, list):
            note_active_call_adoption(False, "missing_goals")
            return []
        goals: list[ExternalGoal] = []
        for g in raw_goals:
            # A skip must be logged: it drops a pressure, and a systematic format switch would
            # otherwise go unnoticed.
            if not isinstance(g, dict):
                logger.warning(
                    "world_pressure_goal_dropped",
                    extra={"agent_id": agent_id, "reason": "not_an_object",
                           "item_type": type(g).__name__},
                )
                continue
            # The anchor is the structural guarantee that pressure comes from signals: a goal
            # citing none may have come from a memory alone. Never keep it unanchored.
            cited = g.get("signals")
            resolved = signal_ref.resolve(cited if isinstance(cited, list) else [cited])
            if not resolved:
                logger.warning(
                    "world_pressure_goal_dropped",
                    extra={"agent_id": agent_id, "reason": "no_signal", "signals": str(cited)[:20]},
                )
                continue
            sources = tuple(dict.fromkeys(
                pid for i in resolved if (pid := signal_people[int(i)]) is not None
            ))
            try:
                goals.append(self._parse_goal(g, source_ids=sources))
            except Exception as exc:  # noqa: BLE001
                # Don't narrow this: one escaping exception would lose this agent's whole round
                # of pressure instead of one item.
                logger.warning(
                    "world_pressure_goal_dropped",
                    extra={"agent_id": agent_id, "reason": "unparseable", "error": str(exc)},
                )
                continue
        goals.sort(key=lambda x: x.urgency.level, reverse=True)
        annotate_active_call(goals_returned=len(raw_goals), goals_accepted=len(goals))
        if raw_goals and not goals:
            note_active_call_adoption(False, "no_usable_goals")
        return goals

    def _parse_goal(self, g: dict[str, Any], *, source_ids: tuple[str, ...]) -> ExternalGoal:
        drive_type = _DRIVE_TYPE_MAP.get(g.get("drive_type", ""), ExternalDriveType.EVENT)
        raw_need = g.get("related_need")
        related_need: NeedType | None = _NEED_TYPE_MAP.get(raw_need, None) if raw_need else None
        urgency = parse_urgency(g.get("urgency"), default=Urgency.NORMAL)
        return ExternalGoal(
            # Length is capped in the prompt (_GOAL_TEXT_MAX_CHARS), never cut here: a cut goal
            # is persisted as a permanently broken sentence.
            text=str(g["text"]),
            source_ids=source_ids,
            urgency=urgency,
            drive_type=drive_type,
            related_need=related_need,
        )
