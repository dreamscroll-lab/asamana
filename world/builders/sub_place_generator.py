"""Sub-place generation: the theme's named spots, each assigned to the map place it belongs to."""

from __future__ import annotations

from core.coerce import coerce_mapping_list, coerce_str
from core.context import annotate_call
from core.interfaces.llm import (
    IndexedRef,
    LLMMessage,
    LLMResponse,
    LLMRouter,
    LLMScene,
    extract_json_object,
    output_budget,
)
from core.interfaces.place import SubPlace
from core.interfaces.world_config import WorldConfig
from core.logging import get_logger

from world.builders import BuildReplyRejected, complete_with_build_retry
from world.models import SUB_PLACE_LIMIT, name_key
from world.setup import MapPlaces, warn_if_over_sub_place_limit

logger = get_logger(__name__)

# Most sub-places one world gets: the array cap the output budget is computed from.
_SUB_PLACE_TOTAL_LIMIT = 20

_SUB_PLACE_SYSTEM = f"""\
【角色】
你根据一个故事主题和一张现成的地图，给出这个故事可能需要的「子地点」，并为每个子地点指定它所属的地点。地图上已有的地点本身不改动。

【任务】
读下面给出的地图地点与故事主题，分两步：
1. 先想清这个故事可能需要哪些子地点。子地点不是必需的：有的地点也可以没有子地点，只有这个故事的情节、事件或人物确实可能涉及的地方才算。说不出这个故事为什么用得到它的，就不写；一个子地点都不写，也完全合理。
2. 再为每一个子地点，从下面的地点里选出它所属的那一个：照各地点的说明，按地理、身份与常识判断它最可能属于哪一处。不属于其中任何一处的，就不写它，不要硬归到某个地点。

【子地点与地点】
- 地图上的每个地点是一大片地方。子地点是属于某个地点的一处建筑、住宅、某个具体的商店、设施、活动场所或区域等等，地图上没有单独标出它。
- 两者是归属关系：一个子地点属于且只属于一个地点，位于这个地点的范围之内；人在子地点里，就是在它所属的地点里。
- 子地点有名字，众人皆知。
- 比如：一大片地方里人人都知道的那座主楼，一片住处里某户人家的住所等等。

【约束】
- 所选的地点必须和 reason 里说的位置一致：reason 说它在哪里、是谁的、在哪里发生过什么，place_index 就选那一处；对不上的，以 reason 为准改选，改不了就不写它。
- 子地点不要小到房间、器物或随口一指的位置（某间屋子、桌前、柱子后头）。
- 子地点不要写暗室、密道、藏身处这类只有少数人知道的地方：子地点人人都知道。
- 不要把下面列出的地点再写成子地点，也不要起和它们重名的名字。
- 不要为凑数而写：数量上限只是上限，不是要写满的数；不要因为某个地点还没有子地点，就给它补一个。
- 不要写与这个世界的时代、风貌不相符，主题无关的子地点。

【输出】
严格输出以下 JSON，不要任何多余内容；故事用不到任何子地点时，sub_places 给空数组：
{{"sub_places": [{{"reason": "≤40字：故事里为什么用得到这一处，它为什么属于这个地点", "name": "子地点名，≤8字", "place_index": 它所属地点的序号（整数），须与 reason 里说的位置一致}}]}}
- sub_places 最多 {_SUB_PLACE_TOTAL_LIMIT} 个，属于同一个地点的最多 {SUB_PLACE_LIMIT} 个。
"""


class SubPlaceGenerator:
    """Name the specific spots a theme needs and assign each to the map place it belongs to.

    Which spots a story uses and where they sit is a reading of the theme (CLAUDE.md §5), so the
    LLM decides it. Each entry is reason → name → place_index (§5b): why the story needs the spot
    and why it sits where it does come first, so the name and the place follow from them rather
    than being justified after. Runs before theme analysis, which then places people and things
    by these names.
    """

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def generate(self, theme: str, world_config: WorldConfig) -> tuple[SubPlace, ...]:
        """The theme's sub-places, or none when it needs none. An unusable reply is retried once,
        then raises (Rule 2)."""
        places = world_config.get_places()
        place_ids = list(places)
        taken = MapPlaces.of(world_config).names
        messages = [
            LLMMessage(role="system", content=_SUB_PLACE_SYSTEM),
            LLMMessage(role="user", content=self._build_user_prompt(theme, world_config, place_ids)),
        ]
        ref = IndexedRef(place_ids)

        async def call() -> LLMResponse:
            # Sub-place generation and theme analysis share a scene; the tag tells their traces apart.
            with annotate_call(builder="sub_places"):
                return await self._llm_router.complete(
                    # per sub-place: reason ≤40 chars (~60 tok) + name ≤8 chars (~12) + place_index (~3)
                    # + 3 fields' structure (~15) ≈ 90; × 20 ≈ 1800 + wrapper ~10; estimate ~1810 tok.
                    LLMScene.WORLD_BUILDING, messages,
                    max_tokens=output_budget(1810), json_mode=True,
                )

        def parse(content: str) -> tuple[SubPlace, ...]:
            payload = extract_json_object(content)
            if payload is None:
                raise BuildReplyRejected(
                    f"LLM returned no JSON object for sub-places: {content[:200]}", "unusable_payload",
                )
            kept: list[SubPlace] = []
            names: set[str] = set()
            per_place: dict[str, int] = {}
            for item in coerce_mapping_list(payload.get("sub_places")):
                name = coerce_str(item.get("name")).strip()
                resolved = ref.resolve([item.get("place_index")])
                # Mechanical repairs only: an unplaced or unnamed entry, a name a place or an
                # earlier sub-place already answers to (one name, one referent).
                if not name or not resolved:
                    reason = "unplaced" if name else "unnamed"
                elif name_key(name) in taken or name_key(name) in names:
                    reason = "name_taken"
                else:
                    reason = ""
                if reason:
                    logger.debug("sub_place_dropped", extra={"sub_place": name, "reason": reason})
                    continue
                names.add(name_key(name))
                per_place[resolved[0]] = per_place.get(resolved[0], 0) + 1
                kept.append(SubPlace(location_id=resolved[0], name=name))
            # The caps are the prompt's; past them the reply is kept whole and only flagged:
            # dropping the tail would cut spots the reply meant (CLAUDE.md: no code truncation).
            warn_if_over_sub_place_limit(kept)
            if len(kept) > _SUB_PLACE_TOTAL_LIMIT:
                logger.warning(
                    "sub_places_over_total_limit",
                    extra={"count": len(kept), "limit": _SUB_PLACE_TOTAL_LIMIT},
                )
            logger.info(
                "sub_places_generated",
                extra={"theme": theme, "count": len(kept), "places": len(per_place)},
            )
            return tuple(kept)

        return await complete_with_build_retry(
            call,
            parse,
            attempt_failed_event="sub_place_generation_attempt_failed",
            log_extra={"theme": theme},
            failure=f"SubPlaceGenerator failed for theme '{theme}'",
        )

    @staticmethod
    def _build_user_prompt(theme: str, world_config: WorldConfig, place_ids: list[str]) -> str:
        """The map (the same for every theme built on it) first, then this call's theme."""
        places = world_config.get_places()
        era = coerce_str(world_config.to_runtime_context().get("era_name"), fallback="")
        lines = ["【地图】", world_config.get_world_description()]
        if era:
            lines.append(f"时代：{era}")
        lines.append("")
        lines.append("【地图上的地点】")
        for index, place_id in enumerate(place_ids, start=1):
            place = places[place_id]
            name = place.name.strip() or "某地"
            lines.append(f"#{index} {name}：{place.description}" if place.description else f"#{index} {name}")
        lines.append("")
        lines.append("【故事主题】")
        lines.append(theme)
        lines.append("")
        lines.append("按上面说定的格式输出 JSON。")
        return "\n".join(lines)
