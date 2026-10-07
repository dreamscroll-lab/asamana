"""Per-agent external pressure evaluator.

One LLM call **per agent that has an acute signal of its own** (inbox message, a perceivable
broadcast, or an ambient observation). Each prompt contains ONLY that agent's perceivable
signals, so information asymmetry is enforced *structurally*, not by instruction. Pure
co-location does not trigger evaluation; it only modulates pressure for an agent with a signal.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.motivation import ExternalDriveType, ExternalGoal
from agent.need import NeedType
from agent.personality import SECRET_LABEL
from agent.relation import format_relation_block
from core.context import observe_stage
from core.interfaces.llm import IndexedRef, LLMMessage, LLMScene, LLMRouter, extract_json, output_budget
from core.interfaces.perception import Situation
from engine.broadcast import BroadcastChannel
from core.interfaces.severity import Severity
from core.interfaces.trace import Stage
from core.interfaces.urgency import Urgency, parse_urgency
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    EXTERNAL_DRIVE_TYPE_DEFINITION,
    render_condition,
    SituationVoice,
    URGENCY_SCALE_DESCRIPTION,
    person_referent,
    relation_legend,
    render_situation_header,
)

if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.message import Message
    from core.interfaces.perception import AmbientEvent, Broadcast, SpatialPerception

logger = get_logger(__name__)


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
_REASON_MAX_CHARS = 80
# Per CLAUDE.md's max_tokens buffer rule: estimate ≈ reason (80 chars×1.5=120)
# + 3×(text 40 chars×1.5=60 + urgency/drive/need/source 12 + structure 25) + wrapper 10
# ≈ 421.
# Coverage is guarded by test_the_token_budget_covers_the_caps_the_prompt_declares.
_MAX_TOKENS = output_budget(421)

_SYSTEM_PROMPT = f"""\
你是第三者视角的外部压力分析者。给定【单个目标角色】、它**自身能感知到的内容**，以及供你识人参考的角色信息，
客观评估此刻有哪些**外部力量**正在推动该角色行动，输出其外部目标列表。

【压力的唯一来源 = 感知内容】
 - 外部压力**只能源自该角色此刻感知到的内容**（收到的消息 / 可感知的广播 / 环境观察 / 同处者的在场）。
 - 角色的姓名、性格、背景只是供你判断"同一信号对这个人意味着什么"的**参考**，**它们本身不是压力来源**；
 - 角色自身的内在动机、长期目标**不属于外部压力**（那是其内在需求，由别处处理，不在此输出）。

【强度锚定信号本身，必须遵守】
 - urgency 必须与**感知内容本身的烈度**匹配：模糊的、遥远的、并未指向该角色的迹象只构成 low/normal。
 - **不要因为该角色内在动机强烈或处境特殊，就把一个微弱信号拔高成 high/critical。**

【默认无压力，不要凑数】
 - 外部压力是例外而非常态。只有当感知内容**确实**对该角色构成可行动的外部推力时，才输出目标；
 - 若该角色此刻并未真正承压，**直接输出空数组 []**——这是完全正常且被期望的结果。
 - 若感知内容与角色无关且也不是角色应该关注的内容，即使感知内容来源与他有强烈的关系，也直接输出空数组。
 - 不要为了"显得有反应"或凑满名额而虚构、夸大压力；目标数量为 0 也很正常，**无需凑满 3 个**。
（注意：真有压力时仍须如实输出——这不是要你压制真实压力，而是不要无中生有。）

【依据关系判断】
 - 每个同处者都附带"目标角色与其的关系"及其身份/性格/背景；收到的消息也会附上"目标角色与发送者的关系"。据此判断亲疏：
 - **与该角色关系亲近、信任度高的人，通常不是其威胁来源**，不要生成针对这类亲近者的"控制 / 威慑 / 防范"目标。

【信息边界】你只会看到该角色能感知的内容；绝不替它推断它无从得知的信息。

【约束】
 - 注意时间规律和常理，即深夜对于一般事件的压力敏感度下降，但对突发事件的敏感度会上升。
 - 处于行动中时，对于周边环境的压力敏感度会下降。
{ABSOLUTE_TIME_RULE}

每个角色最多 3 个目标，按紧迫度降序。每个目标包含：
- text: 指引该角色行动的一句话（中文，≤{_GOAL_TEXT_MAX_CHARS}字，用该角色当下视角；**不得使用事件的事后命名，也不要预测未来走向，也不要臆想当前情景。**）
- {URGENCY_SCALE_DESCRIPTION}
- {EXTERNAL_DRIVE_TYPE_DEFINITION}，取值之一：{_DRIVE_TOKENS}
- related_need: 该压力最终牵动的内在需求，取值之一（含义如下）或 null：{_NEED_OPTIONS}。\
通常 threat→safety，authority→esteem 或 safety，obligation→social 或 esteem，event 视内容而定。\
**urgency 为 high 或 critical 的目标必须给出明确 related_need（不可为 null）**。
- source: 压力来源——【同处一地】列表中角色的序号（与输入中的 #N 对应的整数）；若来自广播或世界事件则填 0

只输出纯 JSON 对象，不要任何说明文字或 markdown。**先在 reason 里分析、再据此给 goals**——\
reason 必须是第一个键（先想后做）：
{{"reason": "<逐一审视每条感知到的信号，结合该角色的人设与关系判断这条信号对他意味着什么、是否真的与他有关或者他需要关注的、是否真\
构成可行动的外部压力、为什么；人设/关系只是判断信号轻重的依据、本身不算压力源。第三者视角，≤{_REASON_MAX_CHARS}字。\
注意这是判断依据，goals 必须由它推出，不是先列 goals 再补理由>", "goals": [{{"text": "……（≤{_GOAL_TEXT_MAX_CHARS}字）", "urgency": "high", "drive_type": "threat", "related_need": "safety", "source": 0}}]}}
"""


class WorldPressureEvaluator:
    """Compute per-agent external pressure with structural information boundaries."""

    def __init__(self, *, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def evaluate(
        self,
        *,
        agents: dict[str, "Agent"],
        agent_inboxes: dict[str, list["Message"]],
        broadcasts: "list[Broadcast]",
        world_time_label: str,
        agent_spatials: "dict[str, SpatialPerception] | None" = None,
    ) -> dict[str, list[ExternalGoal]]:
        """Return external goals per agent.

        Agents without an acute signal of their own are skipped and omitted from the result.
        """
        tasks = [
            self._evaluate_one(
                agent_id=agent_id,
                agent=agent,
                all_agents=agents,
                inbox=agent_inboxes.get(agent_id, []),
                broadcasts=broadcasts,
                spatial=agent_spatials.get(agent_id) if agent_spatials else None,
                world_time_label=world_time_label,
            )
            for agent_id, agent in agents.items()
            # Otherwise world-wide death notices would get evaluated for the corpse.
            if agent.is_active
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
        all_agents: dict[str, "Agent"],
        inbox: list["Message"],
        broadcasts: "list[Broadcast]",
        spatial: "SpatialPerception | None",
        world_time_label: str,
    ) -> tuple[str, list[ExternalGoal] | None]:
        """Evaluate one agent. Returns (agent_id, goals) or (agent_id, None) when skipped."""
        location = agent.personality.state.current_location or (spatial.location_id if spatial else "")
        perceivable_bcs = BroadcastChannel.for_location(broadcasts, location)
        ambient = list(spatial.ambient_events) if spatial is not None else []

        # Pure co-location does not trigger: standing beside someone who did nothing is not the
        # world speaking to you.
        if not (inbox or perceivable_bcs or ambient):
            return agent_id, None

        visible_ids = list(spatial.visible_agent_ids) if spatial is not None else []
        co_located = await self._co_located(agent, all_agents, spatial, visible_ids)
        # Co-located senders' relations are already in the 【同处一地】 block; only fill the gap.
        sender_relations = await self._sender_relations(
            agent, inbox, exclude_ids=set(visible_ids),
        )
        source_ref = IndexedRef(visible_ids)
        situation_header = render_situation_header(
            Situation(
                location_view=spatial.location_view if spatial is not None else None,
                time_label=world_time_label,
            ),
            voice=SituationVoice.THIRD,
        )
        prompt = self._build_agent_prompt(
            agent,
            inbox=inbox,
            broadcasts=perceivable_bcs,
            ambient=ambient,
            co_located=co_located,
            sender_relations=sender_relations,
            situation_header=situation_header,
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]
        try:
            # Set inside each gather task so traces attribute per person without crosstalk.
            with observe_stage(Stage.PRESSURE, agent_id=agent_id):
                response = await self._llm_router.complete(
                    # The budget follows the caps the prompt declares; see _MAX_TOKENS.
                    LLMScene.WORLD_PRESSURE, messages, temperature=0.2, max_tokens=_MAX_TOKENS,
                    json_mode=True,
                )
        except Exception as exc:
            logger.warning("world_pressure_eval_failed", extra={"agent_id": agent_id, "error": str(exc)})
            return agent_id, []
        return agent_id, self._parse_agent_goals(
            response.content, source_ref=source_ref, agent_id=agent_id,
        )

    # ------------------------------------------------------------------

    async def _co_located(
        self,
        agent: "Agent",
        all_agents: dict[str, "Agent"],
        spatial: "SpatialPerception | None",
        visible_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Build the evaluated agent's view of each co-located agent: role + own relation.

        The relation is the evaluated agent's own view, what lets the LLM tell ally from foe.
        Read-only (load_relation, never create).
        """
        out: list[dict[str, Any]] = []
        for vid in visible_ids:
            rel = None
            try:
                rel = await agent.agent_store.load_relation(agent.world_id, agent.agent_id, vid)
            except Exception:  # noqa: BLE001 — a relation read failure must not abort evaluation
                rel = None
            other = all_agents.get(vid)
            other_soul = other.personality.soul if other is not None else None
            who = (
                (p.identity if (p := spatial.visible_agents.get(vid)) else None)
                if spatial is not None else None
            )
            name = (who.name if who else "") or "某人"
            out.append({
                "id": vid,
                "name": name,
                # God-view evaluation: gender from soul, like role/traits. Don't double-source it
                # with the perception channel's who: two sources diverge.
                "gender": other_soul.gender if other_soul is not None else "",
                "role": other_soul.role if other_soul is not None else "",
                "traits": list(other_soul.core_traits) if other_soul is not None else [],
                "values": list(other_soul.core_values) if other_soul is not None else [],
                "background": (other_soul.background or "") if other_soul is not None else "",
                # From the live agent, not the perception packet: same rule as gender.
                "condition": render_condition(other.personality.state.condition) if other is not None else "",
                "trust": round(rel.trust_objective, 2) if rel is not None else None,
                "affection": round(rel.affection_objective, 2) if rel is not None else None,
                "labels": list(rel.labels) if rel is not None else [],
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
            if not sid or not msg.sender_is_agent or sid in exclude_ids or sid in out:
                continue
            rel = None
            try:
                rel = await agent.agent_store.load_relation(agent.world_id, agent.agent_id, sid)
            except Exception:  # noqa: BLE001 — a relation read failure must not abort evaluation
                rel = None
            out[sid] = self._format_relation({
                "trust": round(rel.trust_objective, 2) if rel is not None else None,
                "affection": round(rel.affection_objective, 2) if rel is not None else None,
                "labels": list(rel.labels) if rel is not None else [],
            })
        return out

    @staticmethod
    def _format_relation(c: dict[str, Any]) -> str:
        """Render via ``format_relation_block``; the scale legend is given once near the list."""
        trust, affection = c.get("trust"), c.get("affection")
        if trust is None and affection is None:
            labels = " | ".join(c.get("labels") or [])
            return f"[{labels}]" if labels else "关系未明确"
        return format_relation_block(
            labels=c.get("labels") or [], trust=trust, affection=affection, include_legend=False,
        )

    def _build_agent_prompt(
        self,
        agent: "Agent",
        *,
        inbox: list["Message"],
        broadcasts: "list[Broadcast]",
        ambient: "list[AmbientEvent]",
        co_located: list[dict[str, Any]],
        sender_relations: dict[str, str],
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
        header_part = f"{situation_header}\n\n" if situation_header else ""

        lines: list[str] = [f"""\
{header_part}【目标角色（仅供识人参考，不是压力来源）】
{person_referent(soul.name, soul.gender, soul.role or '无', activity, condition)}
  性格：{traits}
  价值观：{values}
  背景：{background}{secret_line}

【该角色此刻感知到的内容（外部压力的唯一来源）】"""]

        # Co-located people and message senders share one scale: explain it once.
        if co_located or sender_relations:
            lines.append(f"（涉及的关系字段含义：{relation_legend()}）")

        has_any = False
        for bc in broadcasts:
            # A perceivable local broadcast is always at this character's location: "local"
            # suffices, and the location_scope id must not be rendered.
            scope = "全局" if bc.location_scope is None else "本地"
            lines.append(f"- 广播[{scope}，{_SEVERITY_WORD[bc.severity]}] {bc.content}")
            has_any = True
        for msg in inbox:
            sender_name = getattr(msg, "sender_name", None) or "某人"
            line = f"- 收到来自{sender_name}的消息：{msg.content}"
            rel = sender_relations.get(getattr(msg, "sender_id", "") or "")
            if rel:
                line += f"（该角色与{sender_name}的关系：{rel}）"
            lines.append(line)
            has_any = True
        for ev in ambient:
            lines.append(f"- 观察到：{ev.content}")
            has_any = True
        if co_located:
            lines.append("- 同处一地（这些人就在该角色身边，构成在场压力；附其身份与该角色对其的关系，供识人参考）：")
            for i, c in enumerate(co_located, 1):
                who = person_referent(
                    c["name"], c.get("gender", ""),
                    *((c["role"],) if c.get("role") else ()),
                    *((c["condition"],) if c.get("condition") else ()),
                )
                lines.append(f"    #{i} {who} — 关系：{self._format_relation(c)}")
                if c.get("traits"):
                    lines.append(f"        其性格：{'、'.join(c['traits'])}")
                if c.get("values"):
                    lines.append(f"        其价值观：{'、'.join(c['values'])}")
                if c.get("background"):
                    lines.append(f"        其背景：{c['background']}")
            has_any = True
        if not has_any:
            lines.append("（无）")

        lines.append("")
        lines.append(
            "请以第三者视角根据【感知到的内容】客观评估该角色承受的外部压力，先给出分析 reason（先想），再据此给出 goals（后做）；"
            "无压力时同样输出 reason，goals 输出空数组 []。"
        )
        return "\n".join(lines)

    def _parse_agent_goals(
        self, raw: str, *, source_ref: IndexedRef, agent_id: str = "?",
    ) -> list[ExternalGoal]:
        try:
            data: Any = extract_json(raw)
        except (ValueError, Exception):  # noqa: BLE001 — extract_json may raise JSONDecodeError
            logger.warning(
                "world_pressure_parse_failed",
                extra={"agent_id": agent_id, "raw_prefix": raw[:200]},
            )
            return []
        raw_goals = data.get("goals") if isinstance(data, dict) else None
        if not isinstance(raw_goals, list):
            return []
        goals: list[ExternalGoal] = []
        for g in raw_goals:
            # A bare string is exactly the required text and every other field has a default:
            # salvage it.
            item = {"text": g} if isinstance(g, str) else g
            # A skip must be logged: it drops a real pressure, and a systematic format switch
            # would otherwise go unnoticed.
            if not isinstance(item, dict):
                logger.warning(
                    "world_pressure_goal_dropped",
                    extra={"agent_id": agent_id, "reason": "not_an_object",
                           "item_type": type(g).__name__},
                )
                continue
            try:
                goals.append(self._parse_goal(item, agents_ref=source_ref))
            except Exception as exc:  # noqa: BLE001
                # Don't narrow this: one escaping exception would lose this agent's whole round
                # of pressure instead of one item.
                logger.warning(
                    "world_pressure_goal_dropped",
                    extra={"agent_id": agent_id, "reason": "unparseable", "error": str(exc)},
                )
                continue
        goals.sort(key=lambda x: x.urgency.level, reverse=True)
        return goals

    def _parse_goal(self, g: dict[str, Any], *, agents_ref: IndexedRef) -> ExternalGoal:
        drive_type = _DRIVE_TYPE_MAP.get(g.get("drive_type", ""), ExternalDriveType.EVENT)
        raw_need = g.get("related_need")
        related_need: NeedType | None = _NEED_TYPE_MAP.get(raw_need, None) if raw_need else None
        urgency = parse_urgency(g.get("urgency"), default=Urgency.NORMAL)
        raw_source = g.get("source")
        if raw_source:
            resolved = agents_ref.resolve([raw_source])
            source_id = resolved[0] if resolved else "world"
        else:
            source_id = "world"
        return ExternalGoal(
            # Length is capped in the prompt (_GOAL_TEXT_MAX_CHARS), never cut here: a cut goal
            # is persisted as a permanently broken sentence.
            text=str(g["text"]),
            source_id=source_id,
            urgency=urgency,
            drive_type=drive_type,
            related_need=related_need,
        )
