"""Memory importance evaluation.

Scores a just-written memory's importance in [0, 1], the scalar consumed by retrieval ranking,
decay and compression. Every agent is scored by an LLM acting as a functional third-person judge;
a structured rule is only the failure floor. Depends only on the LLM router, never on the memory
store, embedding or vector index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from core.context import observe_stage
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.prompts import MEMORY_IMPORTANCE_SCORE_DEFINITION

if TYPE_CHECKING:
    from agent.personality import PersonalityLayer


logger = get_logger(__name__)


def render_related_people(directory: dict[str, str] | None) -> str:
    """Render ``{id: name}`` as names for the prompt; a person with no name becomes the
    descriptive "某位相关人物", never the id."""
    if not directory:
        return "无"
    parts = [
        label or "某位相关人物"
        for label in directory.values()
    ]
    return "、".join(parts) if parts else "无"



class ImportanceEvaluator:
    """Scores memory importance [0,1] via LLM; structured rules are the failure floor."""

    def __init__(self, *, llm_router: LLMRouter, agent_id: str) -> None:
        self._llm_router = llm_router
        self._agent_id = agent_id

    async def evaluate(
        self,
        raw_content: str,
        *,
        personality: PersonalityLayer,
        related_agents: Sequence[str],
        related_directory: dict[str, str] | None = None,
        related_relations_text: str = "",
        dominant_need_label: str = "",
    ) -> float:
        """Score importance in [0, 1] via the LLM, for every agent.

        Importance depends on persona and relations, so it has no rule equivalent (CLAUDE.md §5);
        the rule score is only the LLM failure floor. ``related_relations_text`` matters: the same
        event weighs very differently for a close friend than for a political enemy.
        """
        name = personality.soul.name
        related_str = render_related_people(related_directory)
        # The agent-specific label from personality.need_label, not a generic Maslow description.
        need_line = f"当前主导需求：{dominant_need_label}\n" if dominant_need_label else ""
        long_term = personality.state.long_term_goals or []
        if long_term:
            long_term_block = "长期目标：\n" + "\n".join(f"  - {g}" for g in long_term[:3])
        else:
            long_term_block = "长期目标：无"
        relation_block = (
            f"【{name}与相关人的关系】\n{related_relations_text}\n\n"
            if related_relations_text
            else ""
        )
        # Functional third-person judgment, not first-person: the experiential stream and
        # emotion_valence already carry the subjective side, and layering it in again would make
        # comparable events' scores jitter with the writer's mood.
        # The system prompt says only "被评判者" so it stays byte-identical across agents (prefix
        # cache); the real name and persona go in the user prompt.
        system_prompt = f"""\
【角色】你是一个客观的记忆重要性评判器，第三视角评判刚发生的这件事对被评判者有多重要——
不代入被评判者的情绪，只做结构性判断。

【评判维度】综合权衡以下各轴给一个**整体**分（不是逐项打分求平均；每条都可正可负，也可不相关）：
 1. 需求关联：是否触及被评判者的需求，尤其主导需求
 2. 目标关联：是否推进或阻碍被评判者的长期/短期目标
 3. 关系影响：是否改变被评判者与他人的信任/亲疏/敌友格局
 4. 自我与价值：是否触动被评判者的核心价值观/行为底线/自我认知，或改变其人生轨迹
 5. 预期反差：是否出乎被评判者意料、颠覆其原有认知（越意外越难忘）
 6. 安危关切：是否关乎被评判者的安全、性命或生计
 7. 其他方面的重要性。
{MEMORY_IMPORTANCE_SCORE_DEFINITION}

【输出】严格输出以下 JSON，不要任何多余内容（reason 在前——先按维度分析，再给分）：
{{"reason": "一两句点出哪些维度最重，≤80字", "score": 0.0到1.0之间的小数}}"""
        user_prompt = f"""\
【被评判者 {name}】
{personality.to_prompt_context(include_emotion=True, include_goals=True)}
{need_line}{long_term_block}

【相关人】{related_str}
{relation_block}【刚发生的事】
{raw_content}

依上面说定的评判维度与 JSON 格式给出结论，只输出 JSON、不写任何多余内容。"""
        try:
            with observe_stage(Stage.MEMORY, agent_id=self._agent_id):
                response = await self._llm_router.complete(
                    LLMScene.MEMORY_IMPORTANCE,
                    [
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    temperature=0.4,
                    # reason ≤80 chars ≈120 tok + score ≈3 + JSON ≈10
                    max_tokens=output_budget(133),
                    json_mode=True,
                )
            data = extract_json(response.content)
            score = float(data.get("score", 0.5))
            logger.debug(
                "memory_importance_llm",
                extra={
                    "agent_id": self._agent_id,
                    "score": round(score, 3),
                    "reason": str(data.get("reason") or "")[:120],
                },
            )
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning(
                "memory_importance_llm_failed",
                extra={"agent_id": self._agent_id, "error": str(exc)},
            )
            return self._rule_evaluate_importance(
                raw_content,
                personality=personality,
                related_agents=related_agents,
            )

    def _rule_evaluate_importance(
        self,
        raw_content: str,
        *,
        personality: PersonalityLayer,
        related_agents: Sequence[str],
    ) -> float:
        """LLM failure floor, not a designed path for any agent. novelty is fixed at 0.5 because
        it can't be measured cheaply here."""

        raw_lower = raw_content.lower()
        short_term_goals = personality.state.short_term_goals or []
        goal_relevance = 0.0
        for goal in short_term_goals:
            tokens = {t for t in goal.lower().split() if len(t) > 1}
            if tokens and any(t in raw_lower for t in tokens):
                goal_relevance = 1.0
                break
        relation_weight = 1.0 if related_agents else 0.0
        emotion_intensity = float(personality.state.emotion.intensity)
        novelty = 0.5
        score = (
            0.1
            + 0.3 * goal_relevance
            + 0.2 * relation_weight
            + 0.3 * emotion_intensity
            + 0.2 * novelty
        )
        return max(0.0, min(1.0, score))
