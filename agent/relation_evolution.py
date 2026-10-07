"""Periodic relation evolver for every living agent: a functional, third-person objective judgment.

The counterpart of reflection (first-person, ≈EXPERIENTIAL): every `label_evolution_interval`
steps it reads recent memories, picks the people in the window, judges what each relation's
labels + summary should now be, and writes them back to `RelationSystem`.

- No current emotion: labels/summary are a slow objective baseline a passing mood must not shift.
- No ids across the boundary: the prompt uses only the agent's own referents (`to_name` /
  `to_gender`, never the global agents dict) plus `#N` indices resolved via IndexedRef.
- Single writer: besides world init, only this module writes labels and `history_summary`;
  `apply_interaction` never does.
- Label protection: blood ties never change; other structural bonds change only on a
  bond-changing event. Enforced only by the prompt. Don't add a code-side guard: it would need a
  theme vocabulary (Rule 7).
- Cost: no LLM call when the window has no candidate relations; output lists only changed
  relations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from agent.memory import MemorySystem
from agent.personality import PersonalityLayer
from agent.relation import (
    relation_legend,
    RelationSystem,
)
from core.interfaces.agent_store import AgentRelation
from core.context import annotate_call
from core.interfaces.llm import (
    IndexedRef,
    LLMMessage,
    LLMRouter,
    LLMScene,
    extract_json,
    output_budget,
)
from core.logging import get_logger
from core.prompts import (
    CLOSED_WORLD_FACT_RULE,
    MEMORY_ORDER_HINT,
    order_memories_chrono,
    person_referent,
)

if TYPE_CHECKING:
    from agent.memory_types import Memory

logger = get_logger(__name__)


# How often `evaluate` runs is `label_evolution_interval` in engine/cognition_maintenance.py; LOOKBACK_STEPS is how
# far back it reads when it does.
LABEL_EVOLUTION_LOOKBACK_STEPS: int = 25
MAX_RECENT_EVENTS_PER_TARGET: int = 4   # cap per-target memory lines in prompt to bound token cost


class RelationEvolution:
    """Update this agent's relation labels and summaries from recent memories."""

    def __init__(
        self,
        llm_router: LLMRouter,
        memory_system: MemorySystem,
        relation_system: RelationSystem,
        personality: PersonalityLayer,
        *,
        agent_id: str,
        lookback_steps: int = LABEL_EVOLUTION_LOOKBACK_STEPS,
    ) -> None:
        self._llm_router = llm_router
        self._memory_system = memory_system
        self._relation_system = relation_system
        self._personality = personality
        self._agent_id = agent_id
        self._lookback_steps = lookback_steps

    async def evaluate(
        self,
        current_step: int,
    ) -> List[tuple[str, list[str], str]]:
        """Run one cycle; returns ``(target_id, labels, summary)`` actually written, where ``[]`` /
        ``""`` means unchanged. Empty list on failure (Rule 1)."""

        candidate_ids = [
            aid
            for aid in self._memory_system.collect_recent_related_agents(
                current_step=current_step,
                lookback_steps=self._lookback_steps,
            )
            if aid != self._agent_id
        ]
        if not candidate_ids:
            return []

        candidate_ids = sorted(candidate_ids)

        contexts: list[_TargetContext] = []
        for target_id in candidate_ids:
            ctx = await self._build_target_context(
                target_id=target_id,
                current_step=current_step,
            )
            if ctx is None:
                continue
            if len(ctx.recent_memories) == 0:
                continue
            # Only relations that exist: collect_recent_related_agents also picks up third parties
            # seen in ambient text. The graph endpoint uses the same "is this a relation" test.
            if not ctx.relation.has_substance():
                continue
            contexts.append(ctx)

        if not contexts:
            return []

        ref = IndexedRef([ctx.target_id for ctx in contexts])
        system_prompt, user_prompt = self._build_prompt(contexts)
        # Index → other party, so an audit can tell whose relation flipped from target_index.
        targets = {str(i): (ctx.relation.to_name or "某人") for i, ctx in enumerate(contexts, 1)}

        try:
            with annotate_call(relation_targets=targets):
                response = await self._llm_router.complete(
                    LLMScene.RELATION_LABEL_EVOLUTION,
                    [
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    temperature=0.4,
                    # updates[]: one item = rationale ≤30 chars (45) + target_index (3) + ≤3 labels
                    # (33) + summary ≤25 chars (38) + JSON overhead (20) ≈ 140 tok; realistic max of
                    # 5 items ≈ 710 tok
                    max_tokens=output_budget(710),
                    json_mode=True,
                )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "label_evolution_llm_failed",
                extra={"agent_id": self._agent_id, "step": current_step, "error": str(exc)},
            )
            return []

        updates_payload = data.get("updates") if isinstance(data, dict) else None
        if not isinstance(updates_payload, list):
            return []

        applied: list[tuple[str, list[str], str]] = []
        for raw in updates_payload:
            if not isinstance(raw, dict):
                continue

            resolved = ref.resolve([raw.get("target_index")])
            if not resolved:
                logger.warning(
                    "label_evolution_bad_target_index",
                    extra={
                        "agent_id": self._agent_id,
                        "step": current_step,
                        "raw_target_index": raw.get("target_index"),
                    },
                )
                continue
            target_id = resolved[0]

            # Missing/invalid labels leave them unchanged (summary-only updates are allowed).
            cleaned_labels: list[str] = []
            new_labels_raw = raw.get("labels")
            if isinstance(new_labels_raw, list):
                cleaned_labels = [str(x).strip() for x in new_labels_raw if str(x).strip()]
            applied_summary = str(raw.get("summary") or "").strip()

            if not cleaned_labels and not applied_summary:
                continue

            wrote_labels: list[str] = []
            wrote_summary = ""
            try:
                if cleaned_labels:
                    await self._relation_system.replace_labels(
                        target_id, cleaned_labels, step=current_step
                    )
                    wrote_labels = cleaned_labels
                if applied_summary:
                    await self._relation_system.update_history_summary(
                        target_id, applied_summary, step=current_step
                    )
                    wrote_summary = applied_summary
            except Exception as exc:
                logger.warning(
                    "label_evolution_write_failed",
                    extra={
                        "agent_id": self._agent_id,
                        "target_id": target_id,
                        "error": str(exc),
                    },
                )
                continue

            if not wrote_labels and not wrote_summary:
                continue

            rationale = str(raw.get("rationale") or "").strip()
            logger.info(
                "label_evolution_applied",
                extra={
                    "agent_id": self._agent_id,
                    "step": current_step,
                    "target_id": target_id,
                    "new_labels": wrote_labels,
                    "new_summary": wrote_summary,
                    "rationale": rationale,
                },
            )
            applied.append((target_id, wrote_labels, wrote_summary))

        return applied

    async def _build_target_context(
        self,
        *,
        target_id: str,
        current_step: int,
    ) -> "_TargetContext | None":
        # Don't use ``get_or_create``: it would persist an empty row before the ``has_substance``
        # check, which ``significant_relations`` then renders into prompts as "某人：[未明确]".
        relation = await self._relation_system.load_existing(target_id)
        if relation is None:
            return None
        recent = self._memory_system.list_recent_memories_mentioning(
            target_id,
            current_step=current_step,
            lookback_steps=self._lookback_steps,
        )
        # Names only from the agent's own knowledge, never the global agents dict (information
        # asymmetry). If unknown, a descriptive referent, never the id.
        name = (relation.to_name or "").strip() or "某位与你有往来的人"
        return _TargetContext(
            target_id=target_id,
            target_name=name,
            target_gender=(relation.to_gender or "").strip(),
            relation=relation,
            recent_memories=recent[-MAX_RECENT_EVENTS_PER_TARGET:],
        )

    def _build_prompt(self, contexts: list["_TargetContext"]) -> tuple[str, str]:
        name = self._personality.soul.name
        persona = self._personality.to_prompt_context(include_goals=False,include_emotion=False)
        blocks = "\n".join(
            "\n".join(_format_target_block(idx, ctx))
            for idx, ctx in enumerate(contexts, start=1)
        )
        # System says only "被评判者" so it stays byte-identical (prefix cache); the name is in user.
        system_prompt = f"""\
【角色】你是一个客观的关系评判器。依据事实证据，**以第三视角**评判被评判者与近期出现的各人物
当前的关系应当是什么——你是裁判，不代入被评判者的情绪。

【事实边界】
{CLOSED_WORLD_FACT_RULE}
不要捏造任何没有发生过的客观事实，这段关系发生了什么只能源自于给定的上下文中。

【关系数值说明】
{relation_legend()}

【评判规则】
1. 先依据"最近相关事件"客观判断关系是否**真的**变了——多数关系应保持不变，不要为改而改。
2. labels 格式见上述【关系数值说明】，要求同一世界同一类型用一致措辞，不在近义词间漂移；
   涉及性别的称谓（父子/父女、兄弟/兄妹、夫/妻等）必须与双方标注的性别一致，不得凭名字臆断。
3. labels 分三类，演化规则不同（**硬约束**）：
   · **血亲**（父子/兄弟等基于生物亲缘）——身份关系，一旦设定**永不删除或改写**，只能在其上新增。
   · **其他结构性绑定**（婚姻/师承/契约/上下级隶属）——仅当近期出现明确的"绑定改变事件"
     （和离/断绝/违约/革职等）才可删改，不因情感漂移而变动。
   · **叙事性定性**（朋友/敌人/盟友/对手等）——可自由演化、新增、删除、替代。
4. summary 用**客观第三视角**概括这段关系近期的走向（如"近期数次交锋后信任明显下降"），
   不代入第一人称、不抒情。
5. 只对**确有变化**的关系输出条目；labels 与 summary 哪个变就给哪个，可只给其一。
6. 用整数序号引用关系（target_index 对应「近期出现的人物与当前关系」里的 #N）。

【输出】严格输出以下 JSON，不要任何多余内容（rationale 写在每项最前——先写依据的具体事件，再给序号与结论）：
{{"updates": [{{"rationale": "依据的具体事件，30个字以内。", "target_index": 1, "labels": ["...", "..."], "summary": "一句话概括走向，25个字以内。"}}]}}
若所有关系都无须改动，输出 {{"updates": []}}。"""
        user_prompt = f"""\
【被评判者】{name}

【{name} 的人设与价值观】
{persona}

【近期出现的人物与当前关系】（每条以 #N 编号；引用时用整数序号 N）：
{blocks}

依上面说定的评判规则与 JSON 格式给出裁定，只输出 JSON、不写任何多余内容。"""
        return system_prompt, user_prompt


class _TargetContext:
    """Per-target prompt context bundle."""

    __slots__ = ("target_id", "target_name", "target_gender", "relation", "recent_memories")

    def __init__(
        self,
        *,
        target_id: str,
        target_name: str,
        relation: AgentRelation,
        recent_memories: list["Memory"],
        target_gender: str = "",
    ) -> None:
        self.target_id = target_id
        self.target_name = target_name
        # Kinship labels are gendered ("父子" vs "父女", "兄弟" vs "兄妹", "夫" vs "妻"), and prompt
        # rule 3 never rewrites a blood-tie label once written, so a wrong guess here is permanent.
        self.target_gender = target_gender
        self.relation = relation
        self.recent_memories = recent_memories


def _format_target_block(idx: int, ctx: _TargetContext) -> list[str]:
    """Render one target block: referents only, no ids or raw steps."""
    rel = ctx.relation
    labels_text = " | ".join(rel.labels) if rel.labels else "未明确"
    block = [
        f"#{idx} {person_referent(ctx.target_name, ctx.target_gender)}",
        f"   当前标签：[{labels_text}]",
        f"   当前数值：信任度={rel.trust_objective:.2f}，"
        f"好感度={rel.affection_objective:.2f}，互动次数={rel.interaction_count}",
    ]
    if rel.history_summary:
        block.append(f"   关系概述：{rel.history_summary}")
    if ctx.recent_memories:
        block.append(f"   最近相关事件（{MEMORY_ORDER_HINT}）：")
        for memory in order_memories_chrono(ctx.recent_memories):
            content = (getattr(memory, "stored_content", "") or "").strip()
            if not content:
                content = (getattr(memory, "raw_content", "") or "").strip()
            if not content:
                continue
            block.append(f"     - {content}")
    else:
        block.append("   最近相关事件：（无明显事件）")
    return block
