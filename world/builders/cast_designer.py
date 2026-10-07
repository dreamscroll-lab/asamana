"""Cast design: the LLM call that gives every figure a canonical narrative role."""

from __future__ import annotations

import asyncio
import json

from core.interfaces.llm import IndexedRef, LLMMessage, LLMRouter, LLMScene, extract_json_object, output_budget
from core.logging import get_logger
from core.prompts import relation_legend
from core.coerce import coerce_list, coerce_mapping_list, coerce_str

from world.models import CastDesign, FigureCastRole, ThemeAnalysis
from world.builders import BUILD_ATTEMPTS

logger = get_logger(__name__)


class CastDesigner:
    """Single LLM call that assigns each figure a canonical narrative role."""

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def design(self, analysis: ThemeAnalysis) -> CastDesign:
        """Return a CastDesign with narrative roles for all figures in analysis."""

        system_prompt, user_prompt = self._build_prompt(analysis)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        last_exc: BaseException | None = None
        for attempt in range(BUILD_ATTEMPTS):
            try:
                response = await self._llm_router.complete(
                    # reason ≤120 chars ≈ 185 tok, plus per character = arc_summary (≤90 chars ≈ 135)
                    # + narrative_role (~6) + index (~3) + key_relationships (≤6 indices × 3) + structure (20)
                    # ≈ 182 tok; realistic max 8 characters ≈ 1456 + 185 = 1641.
                    LLMScene.CAST_DESIGN, messages, max_tokens=output_budget(1641), json_mode=True,
                )
                cast_design = self._parse(response.content, analysis)
                if cast_design is not None:
                    return cast_design
                last_exc = ValueError(
                    f"CastDesigner returned incomplete roles for world '{analysis.world_name}' "
                    f"(attempt {attempt + 1})"
                )
            except Exception as exc:
                last_exc = exc
            logger.warning(
                "cast_design_attempt_failed",
                extra={
                    "world_name": analysis.world_name,
                    "attempt": attempt + 1,
                    "reason": str(last_exc),
                },
            )
            if attempt < BUILD_ATTEMPTS - 1:
                await asyncio.sleep(1.0)
        raise ValueError(
            f"CastDesigner failed for world '{analysis.world_name}' after {BUILD_ATTEMPTS} attempts"
        ) from last_exc

    def _build_prompt(self, analysis: ThemeAnalysis) -> tuple[str, str]:
        """Return ``(system, user)`` split so the invariant scaffolding
        (task / constraints / schema) rides the provider's prefix cache.

        ``system`` = existing role one-liner + fully-static 【任务】/【约束】 +
        byte-fixed 【输出】 schema. ``user`` = volatile 【输入】 (world /
        core_tension / narrative_theme / narrative_pitch / figures / relations)
        + one short output reminder.
        """

        # ── system: fully-invariant scaffolding.
        schema = {
            # First: the roles are derived from this cross-cast analysis (CLAUDE.md §5).
            "reason": "动笔前先想清楚全体人物的分工，定位，关联等等（≤120字），只分析，不复述输入原文。",
            "roles": [
                {
                    # First: what follows is written about this person (CLAUDE.md §5b).
                    "index": "<该人物在上面人物列表中的编号（整数）>",
                    "narrative_role": "<2-4 字叙事角色描述，反映该人物在叙事中的功能性位置>",
                    "arc_summary": "人物弧摘要（一句话、≤90字，描述其在故事中的变化轨迹或叙事位置；只写他会被怎样推动、往哪个方向变，不写结局和结果）",
                    "key_relationships": ["<与其有重要叙事关系的人物编号（整数），至多 6 个>"],
                }
            ],
        }
        system_sections: list[str] = [
            "你负责为世界中的人物分配叙事角色与故事弧线。只返回 JSON，不包含任何其他内容。",
        ]
        system_sections.append(
            "【任务】\n"
            "先在 reason 里想清楚全体人物的分工，定位，关联等等，再据此为每个人物分配 narrative_role、一句话 arc_summary 以及 key_relationships。"
        )
        system_sections.append("\n".join([
            "【约束】",
            "- 每个 arc_summary 必须在本世界的核心张力与开局情境的引力场下推进，且与叙事主题一致。",
            "- 不要求每个角色都「与谁对立」——arc 可以是趋近、转变、告别、揭示、寻找、等待、目击、抉择等"
            "任一形态，由 core_tension 的形态决定。",
        ]))
        system_sections.append(
            "【输出】\n"
            "只返回符合以下 Schema 的紧凑 JSON，不要任何多余内容：\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        system_prompt = "\n\n".join(system_sections)

        # ── user: volatile 【输入】 + one-line output reminder.
        figures_text = "\n".join(
            f" #{i} {f.as_prompt()}" for i, f in enumerate(analysis.key_figures, 1)
        )
        # Render BOTH directions so the role/arc designer sees relational asymmetry
        # (e.g. A trusts B while B distrusts A) — a primary driver of divergent arcs.
        # CastDesigner is a god's-eye functional task, so full relational detail is
        # appropriate here (information asymmetry only constrains in-character cognition).
        relations_text = "\n".join(
            f" · {r.source_name} → {r.target_name}: "
            f"{' | '.join(r.labels) if r.labels else '未明确'} (trust={r.trust:.1f},affection={r.affection:.1f})；"
            f"{r.target_name} → {r.source_name}: "
            f"{' | '.join(r.effective_reverse_labels()) if r.effective_reverse_labels() else '未明确'} "
            f"(trust={r.effective_reverse_trust():.1f},affection={r.effective_reverse_affection():.1f})"
            for r in analysis.initial_relations
        )
        input_block = "\n".join([
            "【输入】",
            f"- 世界：{analysis.world_name}",
            f"- 核心张力（引擎）：{analysis.core_tension}",
            f"- 叙事主题：{analysis.narrative_theme}",
            f"- 开局情境聚焦：{analysis.narrative_pitch}",
            f"- 人物：\n{figures_text}",
            (f"- 关系字段说明：{relation_legend()}\n" if relations_text else "")
            + f"- 关系：\n{relations_text}",
        ])
        user_prompt = "\n\n".join([input_block, "严格按上述 JSON 输出，不写多余内容。"])
        return system_prompt, user_prompt

    def _parse(self, raw: str, analysis: ThemeAnalysis) -> CastDesign | None:
        payload = extract_json_object(raw)
        if payload is None:
            return None
        names = IndexedRef([f.name for f in analysis.key_figures])
        required: dict[str, set[str]] = {}
        for rel in analysis.initial_relations:
            required.setdefault(rel.source_name, set()).add(rel.target_name)
            required.setdefault(rel.target_name, set()).add(rel.source_name)
        by_name: dict[str, FigureCastRole] = {}
        for r in coerce_mapping_list(payload.get("roles")):
            resolved = names.resolve([r.get("index")])
            if not resolved:
                continue
            name = resolved[0]
            existing = names.resolve(coerce_list(r.get("key_relationships")))
            augmented = list(dict.fromkeys(existing + sorted(required.get(name, set()))))
            by_name[name] = FigureCastRole(
                name=name,
                narrative_role=coerce_str(r.get("narrative_role")),
                arc_summary=coerce_str(r.get("arc_summary")),
                key_relationships=augmented,
            )
        if len(by_name) < len(analysis.key_figures):
            return None
        return CastDesign(roles=list(by_name.values()))
