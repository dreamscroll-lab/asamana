"""Map selection: which installed world template a theme is built on."""

from __future__ import annotations

import asyncio
from typing import Sequence

from core.interfaces.llm import (
    IndexedRef,
    LLMMessage,
    LLMRouter,
    LLMScene,
    extract_json_object,
    output_budget,
)
from core.logging import get_logger
from core.coerce import coerce_str
from core.interfaces.world_config import WorldConfig

from world.builders import BUILD_ATTEMPTS

logger = get_logger(__name__)


_TEMPLATE_SELECTION_SYSTEM = """\
【角色】
你在为一个 AI 驱动的叙事仿真挑选世界地图。地图提供故事发生的**空间**——地理、
地点、地点之间的连接。角色、情节、冲突由另一套流程按主题生成，不归你管；你只回答
「这个故事应该发生在哪片土地上」。

【任务】
读下面给出的候选地图与用户主题，选出最能承载这个主题的一张。

【判据】
1. 时代与技术水平是否相容——主题里的人在这张图上能不能自然地生活与行动。
2. 空间结构是否支撑主题的核心张力——宫廷倾轧需要有中枢与层级，市井故事需要有街巷
   与聚集处，逃亡需要有边界与出口。
3. 尺度是否匹配——一场发生在几条街之内的故事，不必要一整座都城。

【原则】
- 没有完美匹配时，选**冲突最小**的那张，不要拒绝选择。
- 不要因为地图名里的专名与主题不同就排除它——地图是空间，专名会被主题覆盖。
- 只依据所给的候选信息判断，不要脑补地图上没有写明的东西。

【输出】
严格输出以下 JSON，不要任何多余内容：
{"reason": "≤60字，说明为何是这张而不是别的", "index": 候选序号（整数）}
"""


class TemplateSelector:
    """Pick the world map a theme should play out on.

    Which land a story belongs to is a judgement about setting, tension and
    scale — the subjective reading CLAUDE.md §5 puts on the LLM side, not a
    choice to push onto the user before they can start.

    The candidates arrive as ``WorldConfig``s and are read only through the
    contract (name / era / description). That is deliberate: enumerating what
    maps exist belongs to the interaction layer, which already answers that
    question — this class must not learn that a template is a directory.
    """

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def select(self, theme: str, candidates: Sequence[WorldConfig]) -> WorldConfig:
        """Return the candidate best suited to ``theme``.

        With nothing to choose from this is a configuration error and raises
        (Rule 3). With a single candidate the answer is already settled — no
        call is made. Otherwise the choice goes to the LLM, and failing to choose
        raises rather than falling back (Rule 2): a mismatched map is not a
        slightly worse world but a wrong substrate under every step that
        follows, and it cannot heal itself.
        """
        if not candidates:
            raise ValueError("No world templates to choose from.")
        if len(candidates) == 1:
            return candidates[0]

        system_prompt = _TEMPLATE_SELECTION_SYSTEM
        user_prompt = self._build_user_prompt(theme, candidates)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        # LLM never names a template — it returns a 1-based index into the list
        # it was shown, resolved back here (CLAUDE.md "LLM Indexed Reference").
        ref = IndexedRef(str(i) for i in range(len(candidates)))

        last_exc: BaseException | None = None
        for attempt in range(BUILD_ATTEMPTS):
            try:
                response = await self._llm_router.complete(
                    # reason ≤60 chars (~90 tok) + index (~3) + JSON structure (~10);
                    # estimate ~105 tok.
                    LLMScene.WORLD_TEMPLATE_SELECTION, messages,
                    max_tokens=output_budget(105), json_mode=True,
                )
                payload = extract_json_object(response.content)
                resolved = ref.resolve([payload.get("index")]) if payload else []
                if resolved:
                    chosen = candidates[int(resolved[0])]
                    logger.info(
                        "world_template_selected",
                        extra={
                            "theme": theme,
                            "template": chosen.to_runtime_context().get("template"),
                            "reason": coerce_str(payload.get("reason"), fallback=""),
                            "candidates": len(candidates),
                        },
                    )
                    return chosen
                last_exc = ValueError(
                    f"LLM returned no usable template index (attempt {attempt + 1}): "
                    f"{response.content[:200]}"
                )
                logger.warning(
                    "template_selection_attempt_failed",
                    extra={"theme": theme, "attempt": attempt + 1, "reason": str(last_exc)},
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "template_selection_attempt_failed",
                    extra={"theme": theme, "attempt": attempt + 1, "error": str(exc)},
                )
            if attempt < BUILD_ATTEMPTS - 1:
                await asyncio.sleep(1.0)
        raise ValueError(
            f"TemplateSelector failed for theme '{theme}' after {BUILD_ATTEMPTS} attempts"
        ) from last_exc

    def _build_user_prompt(self, theme: str, candidates: Sequence[WorldConfig]) -> str:
        """Volatile half of the prompt: the map menu, then this call's theme.

        Candidates first because the installed set rarely changes between calls
        — keeping it ahead of the theme lengthens the cacheable prefix.
        """
        lines: list[str] = ["【候选地图】"]
        for i, candidate in enumerate(candidates, start=1):
            context = candidate.to_runtime_context()
            era = coerce_str(context.get("era_name"), fallback="")
            name = coerce_str(context.get("world_name"), fallback="") or f"地图{i}"
            lines.append(f"#{i} {name}{f'（{era}）' if era else ''}")
            lines.append(f"   {candidate.get_world_description()}")
        lines.append("")
        lines.append("【用户主题】")
        lines.append(theme)
        lines.append("")
        lines.append("按上面说定的格式输出 JSON。")
        return "\n".join(lines)
