"""Theme analysis: the LLM call that turns a theme into the world's cast, places, relations and seeds."""

from __future__ import annotations

import asyncio
import json
from typing import AbstractSet, Any, Mapping, Sequence

from agent.personality import SECRET_LABEL
from agent.relation import NEUTRAL_AFFECTION, NEUTRAL_TRUST
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json_object, output_budget
from core.logging import get_logger
from core.prompts import MEMORY_IMPORTANCE_SCORE_DEFINITION, RELATION_LABEL_FORMAT
from core.coerce import (
    coerce_float, coerce_int, coerce_mapping_list, coerce_optional_float, coerce_str,
    coerce_str_list,
)
from engine.clock import SECONDS_PER_HOUR, resolve_step_seconds

from world.models import (
    AgentTier, HistoricalEventSeed, LocationSeed, NpcSeed, RelationSeed, ThemeAnalysis,
    ThemeFigure, WorldEntitySeed,
)
from world.builders import BUILD_ATTEMPTS

logger = get_logger(__name__)


_THEME_PROMPT_SCHEMA = {
    "reason": (
        "先分析后决策（本字段排在最前，是 core_tension / narrative_theme / narrative_pitch 的推导依据）："
        "用一两句话分析这个主题意味着什么样的处境、人物关系与潜在张力，并据此判断最贴合的结构性张力形态"
        "（可不在下方菜单内，可叠加）。只分析、不抄主题原文，≤80字。"
    ),
    "era_description": "时代背景（2-3句、≤90字，描述时代氛围与历史特征）",
    "core_tension": (
        "一句话、≤100字陈述驱动这个故事的结构性未解张力——什么样的拉力使现状无法静止、给人物以行动理由。"
        "**只回答「什么在驱动」，不回答「关于什么」**（后者属于 narrative_theme）。"
        "**不要求必须是对立**。可以是对立、渴望、暧昧、不可调和、生成、必逝、信息不对称等任一形态或多种叠加。"
        "陈述结构性 standing 状态，不写具体当下事件。"
    ),
    "narrative_theme": (
        "一句话、≤120字简述这是一个什么样的故事——它「关于什么」。"
        "平实陈述即可，不要写成论文摘要式的抽象长句。"
    ),
    "narrative_pitch": (
        "基于 core_tension 与 narrative_theme，一句话、≤70字陈述 step 0 此刻叙事所处的状态。"
        "**禁止使用未来时词**：将/终将/最后/最终/演变/必然/势必/注定。"
        "描述当下情境，不预测未来走向，不替人物决定接下来要发生的事。"
    ),
    # Comes after core_tension / narrative_theme / narrative_pitch: the world name is the story's
    # title, and the story has to exist before it can be named. Placed earlier, the model hasn't
    # written what the story is yet and grabs the only name at hand, the map's name in the runtime
    # context, so every world gets called "长安" or "现代都市".
    "world_name": (
        "这个**故事**的名字（2-10字）：从你上面刚写定的 core_tension 与 narrative_theme 里长出来，"
        "让人一眼看出这是关于什么的故事。四条禁止："
        "① 不要拿地名充数——一个地方的名字说不出那里正在发生什么；"
        "② 不要照抄或截取用户主题原文；"
        "③ 不要用那类换个故事照样成立的体裁套词——名字要只配得上这一个故事；"
        "④ 不要剧透结局或预告后续走向——命名此刻，不是命名终局。"
    ),
    "world_time_config": {
        "era_name": (
            "<纪元 / 年号的名字本身（≤6字），**不含年数**：年数写进下面的 start_year，"
            "因为它要能随世界推进而增长，写死在标签里就永远不会变。"
            "不以纪元纪年的世界留空字符串>"
        ),
        "start_year": (
            "<整数 ≥1：开局年份，必填。以纪元纪年的世界填该纪元内的第几年；"
            "不以纪元纪年的世界填绝对年份>"
        ),
        "start_month": "<整数 1-12：开局月份，与主题季节相称>",
        "start_day": "<整数 1-30：开局日期>",
        "start_hour": "<整数 0-23：开局时辰，与开局情境相称（如紧张夜戏取深夜、晨间集会取早时）>",
        "hours_per_step": (
            "<整数 1-24：这个故事每推进一步跨过多少小时。判据见上方约束；由你据 core_tension 的"
            "时间尺度定，不要照抄默认值>"
        ),
    },
    "key_figures": [
        {
            "name": "人物姓名（≤10字）",
            "role": f"<角色定位 2-4 字，当前阶段的角色定位，不要预测最终的角色定位。这是别人眼里他的身份，{SECRET_LABEL}归 secret>",
            "importance": "main 或 background（判定标准见提示）",
            # Before brief: the present motive depends on what is being hidden.
            "secret": "<可选，≤40字，用第一人称：他不想别人知道的真实身份定位。打算、经历过的事、处境、心事都不算。大多数人没有，没有就给空字符串，不要胡编乱造。>",
            "brief": "一句话、≤50字人物简介：描述其当前处境与主观动机。只写当下，不要预测人物结局或命运走向，不要用作者/旁观视角描述其叙事功能",
            "age": "<整数，0-120：人物年龄；须与人物间的辈分/长幼关系一致，见提示>",
            "gender": "男 / 女",
        }
    ],
    "initial_relations": [
        {
            # Field order = generation order: first the qualitative (what relationship these two have), then
            # the quantitative that follows from it (trust/affection values). With numbers first, they have no
            # anchor and the labels can only bend to fit numbers already written.
            "from": "关系发起方人物姓名（A）",
            "to": "关系接收方人物姓名（B）",
            "labels": [
                f"<A 对 B 的关系标签，{RELATION_LABEL_FORMAT}"
                "角色一定要与 key_figures 中的 age 在逻辑上一致，比如兄弟关系A的年龄小于B的年龄，则A是B的弟弟，B是A的哥哥。这个很重要，一定要遵守！"
                "可多个标签并存。血缘、配偶、师徒等结构性绑定一旦设定不可删改。"
                "同一世界内同一类型请使用一致措辞，不要用近义词。至多 3 个。>"
            ],
            "trust": "<浮点数 0.0-1.0：A 对 B 的信任度，0=完全不信任，0.5=中立，1=完全信任。先看上面写定的 labels，这个数值要与那层关系相称>",
            "affection": "<浮点数 -1.0-1.0：A 对 B 的好感度，-1=憎恶，0=平淡，1=深爱。同样要与上面写定的 labels 相称>",
            "reverse_labels": [
                "<B 对 A 的关系标签数组，必填不可省略。结构同 labels，role 是 B 在该关系中的位置。"
                "非对称关系务必反转角色：labels=[\"父子:儿子\"]（A 是儿子）则 reverse_labels=[\"父子:父亲\"]（B 是父亲）；"
                "对称关系（如「相识」「同事」）则原样重复同一标签。至多 3 个。>"
            ],
            "reverse_trust": "<浮点数 0.0-1.0：B 对 A 的信任度。必填，即使与 trust 相同也要写出；与上面写定的 reverse_labels 相称>",
            "reverse_affection": "<浮点数 -1.0-1.0：B 对 A 的好感度。必填，即使与 affection 相同也要写出；与上面写定的 reverse_labels 相称>",
        }
    ],
    "historical_events": [
        {
            "event": "历史事件描述（一句话，≤50字，需要注意时间顺序，这是已经发生过的事情，如果内容中提到时间，那时间必须是在开局时间（world_time_config）之前。）",
            "hours_before_start": "<正整数，事件发生于故事开始前多少小时，如48表示两天前>",
            "related_figures": ["涉及的人物姓名，至多 4 人"],
            "importance": MEMORY_IMPORTANCE_SCORE_DEFINITION,
        }
    ],
    "world_entity_seeds": [
        {
            "name": "实体名称（2-8字）",
            "entity_type": (
                "item（一个人就能拿起来带走、会随人移动、会易主的东西）"
                "或 landmark（固着于某个地点、拿不走、但状态可变的东西）"
                "——判据只有一条：**它能不能被一个人带走**。"
                "这个分类同时决定了它日后能不能被夺走，按东西本身如实归类"
            ),
            "description": (
                "一句话、≤80字描述这个 entity 本身：它是什么、外观或形态、有什么显著特性、有什么用途等等。"
                "这是**任何人一眼能看到的**：如果它上面写了什么、记了什么，一概不写、也不概括，那归 content 字段写。"
                "对 landmark：人物的位置会变，描述里不要写某人此刻在此。"
            ),
            "initial_state": "初始状态词（≤8字，intact/sealed/active/covert 等）",
            "location_name": "所在地点名称（必须是可用地点列表中的 name 字段值）",
            "content": (
                "≤80字：它**上面写着、记着的东西本身**——纸上写的内容、账上的数、碑上的字；"
                "拿到它的人读到的就是这一栏，写实写全，不写「一封密信」这种概括。"
                "不能直接点出任何人的 secret。"
                "它长什么样归 description；没有写字记事的东西内容给空字符串"
            ),
        }
    ],
    "npcs": [
        {
            "name": (
                "称谓（2-5字）：写他**是哪一类人**，不写具体的人名，由它的特征定义。"
            ),
            "gender": "男 / 女",
            "age": "<整数，0-120>",
            "description": (
                "≤30字：他的定位与鲜明特点——他是做什么的，有什么长处、有什么短处"
                "（力大 / 跑得快 / 眼尖 / 认得路 / 腿脚不便 之类，按这个世界自己的样子写）。"
                "**每个人要各有各的特点**，不要写成一批可以互换的人；点到为止，不必量化。"
                "只写**外在看得出的**：他没有内心，所以不写性格、心情、立场与心事等等"
            ),
            "location_name": "所在地点名称（必须是可用地点列表中的 name 字段值，同时需要结合description进行选定，不要乱选。）",
        }
    ],
}

# Non-exhaustive menu of tension forms. Injected into the ThemeAnalyzer prompt
# to defeat the LLM's default tendency to assume "opposition" whenever it sees
# the word "tension" or "conflict". The list is not a closed enumeration —
# multiple forms can coexist in a single world.
_TENSION_FORMS_MENU = (
    "张力可能的形态（非穷尽，可叠加；不限于以下）：\n"
    "- 对立：A 与 B 想要相互排斥的东西\n"
    "- 渴望 / 缺位：X 在等 / 找 / 渴慕 Y，Y 是否到来未定\n"
    "- 暧昧：真相 / 身份 / 选择本身未明\n"
    "- 不可调和：X 必须同时是 A 与 B，无法两全\n"
    "- 生成：X 正在变成不同的自己\n"
    "- 时间 / 必逝：某事终将到来 / 某物已悄然失去\n"
    "- 信息不对称：谁知道 / 谁不知道的差异本身就是叙事动力"
)

# What ThemeAnalyzer is told about the map it is writing onto — an ALLOWLIST, in render
# order, each with its plain meaning.
#
# An allowlist, NOT a dump of to_runtime_context(): that dict is assembled for the runtime
# and carries fields this prompt must not see. Two in particular:
#   - `world_name` (the MAP's name) sits next to a request for a name, and the model copies
#     it, so every world ends up named "长安" / "现代都市". Do not show it; a warning telling
#     the model to ignore a value we handed it loses often enough.
#   - `template` ("changan_iso") is a code-layer id, meaningless in a prompt.
# What the analyzer genuinely needs reaches it by a meaningful channel: the era bracket and
# clock anchors below, and "可用地点" for the place.
_RUNTIME_CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("era_name", "地图所属的大致时代区间，供你从中特化出具体纪元/纪年"),
    (
        "calendar",
        "该世界的历法："
        "classical_cn = 以纪元纪年（era_name 填纪元名、start_year 填纪元内第几年）；"
        "modern = 不用纪元（era_name 留空、start_year 填绝对年份）",
    ),
    ("start_month", "开局月份（1-12）"),
    ("start_day", "开局日期（1-30）"),
    ("start_hour", "开局时刻（0-23 时）"),
)


def _format_world_context(world_context: Mapping[str, object]) -> str:
    """Render the allowed slice of runtime context as labeled lines with per-field meaning.

    Fields absent from the context are skipped; fields absent from the allowlist are not
    rendered at all (see ``_RUNTIME_CONTEXT_FIELDS`` for why that is the point).
    """
    lines = [
        f"- {key} = {json.dumps(world_context[key], ensure_ascii=False)}：{desc}"
        for key, desc in _RUNTIME_CONTEXT_FIELDS
        if key in world_context
    ]
    if not lines:
        return "运行时上下文：（空）"
    return "\n".join(["运行时上下文（各字段含义如下，在有帮助时用作默认值）：", *lines])


# ThemeAnalyzer's output budget. It isn't a constant: it grows with max_agents, quadratically. The
# constraints ask initial_relations to "include every structural bond possible", whose realistic
# upper bound is C(N,2) pairs. A hard-coded number would let a tunable max_agents in config
# silently decide whether this build gets truncated.
#
# Estimated the CLAUDE.md way (Chinese 1.5 tok/char, ~3 tok per enum/number, ~5 tok structure per field):
#   fixed part ~820 = reason ≤80 chars (120) + world_name ≤10 chars (20) + era_description ≤90 chars (135)
#                  + core_tension ≤100 chars (150) + narrative_theme ≤120 chars (180) + narrative_pitch ≤70 chars (105)
#                  + world_time_config (54: 5 fields + hours_per_step integer 3+5) + 11 top-level keys (55)
#   key_figures        200/person × N        (name ≤10 chars 15 + role 6 + importance 3 + secret ≤40 chars 60
#                                             + brief ≤50 chars 75 + age 3 + gender 3 + 7 fields 35)
#   initial_relations  122/pair × C(N,2)     (bidirectional: labels/reverse_labels 29 each + 4 numbers 12
#                                             + from/to 12 + 8 fields 40)
#   historical_events  124/entry × 6         (explicit prompt cap "至多 6 条")
#   world_entity_seeds 320/item × 10         (explicit prompt cap "至多 10 个"; description ≤80 chars 120
#                                             + content ≤80 chars 120 + name/type/state/location ~40
#                                             + 7 fields 40)
_THEME_TOKENS_FIXED = 820
_THEME_TOKENS_PER_FIGURE = 200
_THEME_TOKENS_PER_RELATION = 122
_THEME_TOKENS_HISTORICAL = 124 * 6
_THEME_TOKENS_SEEDS = 320 * 10
#   npcs               98/body × cap (stated explicitly in the prompt, see _npc_count_constraint)
#     name 5 chars ≈ 8 + gender ≈ 3 + age ≈ 3 + description ≤30 chars ≈ 45 + location_name ≈ 12
#     + 5 fields of structure ≈ 25 → ~96, rounded to 98.
_THEME_TOKENS_PER_NPC = 98
# With max_agents=None the prompt gives no cast-size constraint and the LLM realistically produces 5–8; estimate at the upper bound of 8.
_THEME_FIGURES_WHEN_UNBOUNDED = 8


def _cast_size_constraint(min_agents: int | None, max_agents: int | None) -> str:
    """The per-call cast-size line for the theme prompt, or "" when unbounded.

    The floor carries its own anti-padding clause: told only "至少 N 个", a model
    short on figures will invent filler to reach the number, and a cast member with
    no stake in the tension is worse than a smaller cast.
    """
    if min_agents is None and max_agents is None:
        return ""
    if min_agents is not None and max_agents is not None:
        size = f"总数 {min_agents}-{max_agents} 个"
    elif max_agents is not None:
        size = f"总数不超过 {max_agents} 个"
    else:
        size = f"总数不少于 {min_agents} 个"
    line = f"- 数量：key_figures {size}，优先保留主要角色。"
    if min_agents is not None:
        line += (
            "凑不满下限说明这个主题里真正牵涉其中的人你还没找齐——回到 core_tension "
            "去找谁的处境会被它改变，不要编与张力无关的角色来凑数。"
        )
    return line


def _npc_count_constraint(max_npcs: int) -> str:
    """The per-call line bounding how many mindless bodies this world gets.

    The cap has to be stated: without it the model scales up to the theme's pomp (a palace city can
    produce twenty attendants), and every one of them crowds into the main characters' co-present
    lists. 0 means this world has no such tier, and then the field isn't mentioned at all: a field
    that says "write 0" gets filled anyway.
    """
    if max_npcs <= 0:
        return "- npcs：这个世界不要这一类人，给空数组。"
    return (
        f"- npcs：至多 {max_npcs} 个。只写这个世界里**本就该有、而故事又真会碰上**的那几个；"
        "他们不参与冲突，也不该有自己的立场。宁可少写，也不要为凑数编出没人会照面的人。"
    )


def _distinct_entity_seeds(seeds: Sequence[WorldEntitySeed]) -> list[WorldEntitySeed]:
    """Keep one thing per id: of two seeds with the same id, only the later one stays in the world.

    Decide on the derived entity_id, not name: ``Letter`` and ``letter`` differ as names but share
    a slug. The prompt doesn't forbid duplicate names, and a silently missing catalyst is impossible
    to spot: both entity_seed_instantiated logs are there.
    """
    kept: list[WorldEntitySeed] = []
    seen: set[str] = set()
    for seed in seeds:
        if seed.entity_id in seen:
            logger.warning(
                "entity_seed_duplicate_id",
                extra={"seed_name": seed.name, "entity_id": seed.entity_id},
            )
            continue
        seen.add(seed.entity_id)
        kept.append(seed)
    return kept


def _distinct_npc_seeds(
    seeds: Sequence[NpcSeed], figure_names: AbstractSet[str]
) -> list[NpcSeed]:
    """Keep one body per appellation, and none that answers to a key figure's name.

    This tier's name is an appellation, so both rules are the same thing: one form of address can
    only refer to one group in the world. Colliding with a key_figure is worse: one name would cover
    two bodies that don't even think the same way (one weighs things, one only runs errands), and
    since agents refer to each other by name, downstream can never tell them apart. Dropping the
    mindless one is the only fix.

    The prompt already forbids both, but when that fails, the world pays for it every step.
    """
    kept: list[NpcSeed] = []
    seen: set[str] = set()
    for seed in seeds:
        if seed.name in figure_names:
            logger.warning("npc_seed_shadows_key_figure", extra={"npc_name": seed.name})
            continue
        if seed.name in seen:
            logger.warning("npc_seed_duplicate_name", extra={"npc_name": seed.name})
            continue
        seen.add(seed.name)
        kept.append(seed)
    return kept


def _distinct_figures(figures: Sequence[ThemeFigure]) -> list[ThemeFigure]:
    """Keep only the first of duplicate names. Agents refer to each other by name, so two agents with
    the same name can never be told apart downstream; if that leaves too few, the caller's cast-size
    check triggers a retry."""
    kept: list[ThemeFigure] = []
    seen: set[str] = set()
    for figure in figures:
        if figure.name in seen:
            logger.warning("key_figure_duplicate_name", extra={"figure_name": figure.name})
            continue
        seen.add(figure.name)
        kept.append(figure)
    return kept


def _theme_max_tokens(max_agents: int | None, max_npcs: int = 0) -> int:
    """max_tokens for the world-building call: the answer estimate grows with cast size."""
    n = max_agents if max_agents is not None and max_agents > 0 else _THEME_FIGURES_WHEN_UNBOUNDED
    answer = (
        _THEME_TOKENS_FIXED
        + _THEME_TOKENS_PER_FIGURE * n
        + _THEME_TOKENS_PER_RELATION * (n * (n - 1) // 2)
        + _THEME_TOKENS_HISTORICAL
        + _THEME_TOKENS_SEEDS
        + _THEME_TOKENS_PER_NPC * max(max_npcs, 0)
    )
    return output_budget(answer)


class ThemeAnalyzer:
    """Analyze a user theme into structured world knowledge."""

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def analyze(
        self,
        theme: str,
        *,
        world_context: Mapping[str, object] | None = None,
        min_agents: int | None = None,
        max_agents: int | None = None,
        max_npcs: int = 0,
        available_locations: Sequence[LocationSeed] | None = None,
    ) -> ThemeAnalysis:
        """Return structured analysis for a theme input.

        ``min_agents`` / ``max_agents`` bound the cast, and a cast outside them is a
        failed attempt — retried once, then raised (Rule 2). Neither side is repaired
        in code. Inventing the missing figures grounds nobody in the theme; and
        dropping the extra ones discards people this same analysis wrote the tension
        around — the cast, its relations and its historical events are composed as one
        set, so a cast cut down afterwards is no longer the world the model wrote.
        """

        runtime_context = dict(world_context or {})
        required = max(1, min_agents or 1)
        ceiling = max_agents
        if ceiling is not None and ceiling < required:
            # An unsatisfiable range is a config/caller error that retrying won't fix → fail right away instead of burning two build calls (Rule 3).
            raise ValueError(
                f"Unsatisfiable cast bounds: min_agents={required} > max_agents={ceiling}"
            )
        system_prompt, user_prompt = self._build_prompt(
            theme,
            runtime_context,
            min_agents=min_agents,
            max_agents=max_agents,
            max_npcs=max_npcs,
            available_locations=available_locations,
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        last_exc: BaseException | None = None
        for attempt in range(BUILD_ATTEMPTS):
            try:
                response = await self._llm_router.complete(
                    # Everything comes out in one call. The budget varies with cast size, so _theme_max_tokens computes it (see its derivation).
                    # (Truncation → invalid JSON → futile retry → Rule 2 raise, so give it enough. max_tokens is a cap, not billed.)
                    LLMScene.WORLD_BUILDING, messages,
                    max_tokens=_theme_max_tokens(max_agents, max_npcs),
                    json_mode=True,
                )
                payload = extract_json_object(response.content)
                if payload is not None:
                    analysis = self._analysis_from_payload(
                        theme, payload, runtime_context, response.content
                    )
                    figure_count = len(analysis.key_figures)
                    if required <= figure_count and (ceiling is None or figure_count <= ceiling):
                        logger.info(
                            "world_theme_analyzed",
                            extra={
                                "theme": theme,
                                "world_name": analysis.world_name,
                                "figure_count": len(analysis.key_figures),
                                "location_count": len(analysis.key_locations),
                            },
                        )
                        return analysis
                    # A dedicated event name + structured counts so cast-size mismatches can be aggregated (the
                    # attempt_failed below is shared by every failure mode and only has the numbers in free text).
                    # world_id is set into the log context by build(), so it needn't be passed here.
                    logger.warning(
                        "cast_size_rejected",
                        extra={
                            "theme": theme,
                            "attempt": attempt + 1,
                            "figure_count": figure_count,
                            "min_agents": required,
                            "max_agents": ceiling,
                        },
                    )
                    last_exc = ValueError(
                        f"LLM returned {figure_count} key_figures for theme '{theme}', "
                        f"outside the required {required}-{ceiling if ceiling else '∞'}"
                    )
                else:
                    last_exc = ValueError(
                        f"LLM returned no parseable JSON for theme '{theme}' "
                        f"(attempt {attempt + 1}): {response.content[:200]}"
                    )
                logger.warning(
                    "theme_analysis_attempt_failed",
                    extra={"theme": theme, "attempt": attempt + 1, "reason": str(last_exc)},
                )
                if attempt < BUILD_ATTEMPTS - 1:
                    await asyncio.sleep(1.0)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "theme_analysis_attempt_failed",
                    extra={"theme": theme, "attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < BUILD_ATTEMPTS - 1:
                    await asyncio.sleep(1.0)
        raise ValueError(
            f"ThemeAnalyzer failed for theme '{theme}' after {BUILD_ATTEMPTS} attempts"
        ) from last_exc

    def _build_prompt(
        self,
        theme: str,
        world_context: Mapping[str, object],
        *,
        min_agents: int | None = None,
        max_agents: int | None = None,
        max_npcs: int = 0,
        available_locations: Sequence[LocationSeed] | None = None,
    ) -> tuple[str, str]:
        """Return ``(system, user)`` split so DeepSeek/OpenAI prefix caching can
        hit the invariant scaffolding.

        ``system`` = existing role one-liner + fully-static task / tension menu /
        constraints / output schema — byte-identical across every ThemeAnalyzer
        call. ``user`` = volatile 【输入】 (theme + runtime context +
        available_locations) + per-call max_agents constraint + short output
        reminder.
        """

        # ── system: fully-invariant scaffolding (§3: role + critical rules at start,
        # output schema at end; the max_agents constraint is per-call so it stays in user).
        system_sections: list[str] = [
            "你负责生成初始化叙事世界所需的紧凑 JSON 数据。只返回 JSON，不包含任何其他内容。",
        ]

        # 【任务】: generation order — settle the engine first, then the theme and angle, then derive the step-0 state from them.
        system_sections.append(
            "【任务】\n"
            "根据主题，构建一个连贯的叙事世界开局；在有帮助时，用运行时上下文作为默认值。"
            "按以下顺序生成（先想后做：先在 reason 里分析，再据此写结论）：\n"
            "0. reason：先分析这个主题意味着什么样的处境与张力，权衡最贴合的张力形态——后续 core_tension 由它推出，而非凭空一猜。\n"
            "1. core_tension：基于 reason 的分析，写出什么样的结构性张力在驱动这个故事。下方【张力形态参考】只是参考样本，"
            "不是可选项的全集——若你的主题适合的张力不在其中，请直接写出来，也可叠加多种形态。"
            "只答「什么在驱动」，不答「关于什么」。\n"
            "2. narrative_theme：一句话简述这是一个什么样的故事（它「关于什么」），平实陈述即可。\n"
            "3. narrative_pitch：基于 core_tension 与 narrative_theme，step 0 此刻叙事处于什么状态。"
            "只描述当下，不预测未来。\n"
            "4. 在以上三者的共同框架下产出 key_figures / initial_relations / historical_events / "
            "world_entity_seeds —— 四者必须彼此自洽。\n"
            "5. npcs：等 key_figures 写定之后再写——这一档取的是它的补集："
            "谁在这个世界里做事，却不在故事里占位置。"
        )

        # 【张力形态参考】: the candidates are already a list; placed in the middle (used by task step 1).
        system_sections.append(_TENSION_FORMS_MENU)

        # 【约束】: one item per constraint so none gets buried in a paragraph and skipped. Only the 4 static
        # constraints unrelated to max_agents live here; the count constraint carries the per-call max_agents value and stays in user.
        constraint_lines = ["【约束】"]
        constraint_lines.append(
            "- 角色重要度（importance）：main = 在 core_tension 中具备独立判断与抉择能力、"
            "其决定能改变张力走向的角色（包括握有裁决权、否决权或最终决断权的角色，即使其"
            "当前态度消极或被动）；background = 主要服从他人意志、认知自主性低、只需构成可信"
            "世界背景的角色。判定依据是叙事中心性与认知自主性，而非角色是否激进活跃。"
            "角色不必都有「对立面」——根据 core_tension 的形态，角色可以是寻找者、等待者、"
            "改变中的人、目击者、抉择者等任一形态。"
        )
        constraint_lines.append(
            "initial_relations中尽可能包含所有的结构性绑定关系，比如血亲，亲戚关系等等；"
            "- 年龄一致性：key_figures 的 age 必须与人物之间的辈分、长幼、资历关系一致：长辈年龄"
            "大于晚辈、兄姊大于弟妹；角色 role 或关系标签隐含的资历次序也应在年龄上体现；"
            "同辈非孪生者不应同岁。"
            "- 性别一致性：key_figures 的 gender 必须与人物之间分性别的称谓一致："
            "关系标签里指明性别的那些（父子/父女、兄弟/兄妹、夫/妻之类）所指的性别，"
            "与该人物的 gender 不得矛盾；role 本身隐含性别时亦然。"
        )
        constraint_lines.append(
            "- 前史：生成至多 6 条 historical_events。它们都是**已经发生过**的事，"
            "所以内容里若提到时间，那个时间必须早于开局时间（world_time_config）。"
        )
        constraint_lines.append(
            "- 叙事催化物：生成至多 10 个 world_entity_seeds —— 承载或激化 core_tension 的关键物品"
            "（item）或影响格局的据点标记（landmark）。每个实体必须与 core_tension 直接相关——"
            "不一定是「武器/争夺物」，也可以是信物、遗物、未拆的信、未送出的礼、未公开的记录等。"
            "对于location_name的填写一定要符合常理，如果确实不知道或者不能确认要填写什么，直接留空。"
            "如果有content字段并涉及到时间，那时间尽量给具体的时间点，比如6月1日，六月初六等等，不要给相对时间，比如明日，三日后。注意时间不能乱填，需要结合开局时间（world_time_config）确定，同时，时间格式要符合主题的时间格式。"
        )
        constraint_lines.append(
            "- 开局时间（world_time_config）：时间不是装饰，它是整个仿真历法的锚点——start_year "
            "决定纪年、start_month 决定季节、start_hour 决定昼夜时段。四点要求："
            "(a) 纪年拆成两个字段：era_name 只写纪元的名字，年数写进 start_year；运行时上下文给的 "
            "era_name 是**时代区间**，你要在该区间内特化到具体纪元，不要照抄区间标签；"
            "不以纪元纪年的世界 era_name 留空、start_year 填绝对年份；"
            "(b) start_month 与 start_hour 彼此自洽、并与纪年及开局情境相称（紧张夜戏取深夜、"
            "晨间集会取早时）；(c) 运行时上下文给出的 start_month / start_day / start_hour 是该世界的"
            "既定常量，应优先沿用为锚点；(d) 仅当主题明确要求不同季节或时刻时才偏离常量，"
            "且偏离后仍须保持 (b) 的自洽。"
        )
        # The criterion only says what to judge by, with no genre lookup table: given genre examples, the
        # model looks up by subject keywords instead of reading the time scale of the core_tension it just
        # wrote (Rule 7 + creative output leads with negative examples).
        constraint_lines.append(
            "- 每步时长（world_time_config.hours_per_step）：一步是故事推进的最小一块，整数 1-24 "
            "小时。先想你上面写定的 core_tension 整个走完大约要多久——一夜、数日、还是一季；再把"
            "这段时间切成一块块推进，一块就是这个值。切得太粗，一场耳语和一次远行会记成同样长短，"
            "人物只剩下大动作可做；切得太细，故事在原地空耗、走不到头。三条禁忌：(a) 不要按题材/"
            "年代查表（「古代所以慢」「现代所以快」都是错的判法，同一座城既能跑一夜之变也能跑十年"
            "之衰）；(b) 不要习惯性取 1 或 24 两个极值，绝大多数故事落在中间；"
            "(c) 这个值一经写定即冻结、贯穿整个故事，不是开局的临时值。"
        )
        # Criteria for the npcs tier. The count constraint is per call (stays in user); what this tier's
        # people look like is constant and goes in the system prefix. Mostly negative examples: positive
        # ones get copied, and the main risk for this tier is a batch of interchangeable people (creative output → negative constraints first).
        constraint_lines.append(
            "- npcs（这一档人是谁）：每个npc要的是**一类人**，不是某一个人。他被手上那份特征定义，"
            "换一个人来做也照样成立；所以 name 写的是**称谓**（他是哪一类人），"
            "他这一类人的特征写进 description。五条禁忌："
            "(a) name 不要写具体的人名，可以写成一类人的统称，比如服务员，打工人，司机，护卫等等"
            "(b) npc是毫无认知的（没记忆，没有心智，没有立场，没有思考能力等等），就是一个工具人，它们存在的宗旨帮助key_figures。"
            "(c) 不要与 key_figures 是同一个人：既不许重名，也不许换个说法把同一个人写两遍；"
            "(d) 不要给他立场、心事或阵营——他不参与冲突，不选边，也没有自己要达成的事；"
            "(e) 不要把同一类人铺成好几个：一类只留一个，各自的本事要真的不一样。"
        )
        system_sections.append("\n".join(constraint_lines))

        # 【输出】: the schema sits at the end of system, next to the generation point (§3: output format last),
        # and since it's byte-fixed it's cached along with the whole system prefix.
        system_sections.append(
            "【输出】\n"
            "只返回符合以下 Schema 的紧凑 JSON，不要任何多余内容：\n"
            f"{json.dumps(_THEME_PROMPT_SCHEMA, ensure_ascii=False)}"
        )
        system_prompt = "\n\n".join(system_sections)

        # ── user: per-call inputs + this call's max_agents count constraint + a reminder at the generation point.
        user_sections: list[str] = []

        input_lines = [
            "【输入】",
            f"- 主题：{theme}",
            _format_world_context(world_context),
        ]
        if available_locations:
            input_lines.append(
                "- 可用地点列表（world_entity_seeds 的 location_name 必须从中选取一个 name 值，"
                "不要自创地点名）："
                + json.dumps(
                    [location.as_dict() for location in available_locations],
                    ensure_ascii=False,
                )
            )
        user_sections.append("\n".join(input_lines))

        # This call's count constraint (varies with the cast size, so it can't go in the cached system prefix).
        count_line = _cast_size_constraint(min_agents, max_agents)
        npc_line = _npc_count_constraint(max_npcs)
        if count_line or npc_line:
            user_sections.append(
                "【本次约束】\n" + "\n".join(x for x in (count_line, npc_line) if x)
            )

        # A one-line output reminder near the generation point (the full schema is in the system prefix).
        user_sections.append("严格按上述 JSON 输出，不写多余内容。")

        return system_prompt, "\n\n".join(user_sections)

    def _analysis_from_payload(
        self,
        theme: str,
        payload: Mapping[str, Any],
        world_context: Mapping[str, object],
        source_material: str,
    ) -> ThemeAnalysis:
        world_time_defaults = {
            "era_name": str(world_context.get("era_name", "")),
            "start_month": coerce_int(world_context.get("start_month"), default=1, minimum=1, maximum=12),
            "start_day": coerce_int(world_context.get("start_day"), default=1, minimum=1, maximum=30),
            "start_hour": coerce_int(world_context.get("start_hour"), default=6, minimum=0, maximum=23),
        }
        payload_time_config = payload.get("world_time_config")
        world_time_config = dict(payload_time_config) if isinstance(payload_time_config, Mapping) else {}
        merged_time_config = {
            **world_time_defaults,
            **world_time_config,
        }
        # The starting year has no fallback: maps carry no year (WorldConfig contract), and an arbitrary
        # one would put the wrong year on every timestamp ("1年9月1日"). Per Rule 2, fail and retry
        # instead; analyze() has one retry.
        start_year = coerce_int(merged_time_config.get("start_year"), default=0, minimum=0)
        if start_year < 1:
            raise ValueError(
                f"start_year missing or invalid for theme '{theme}': LLM must date the world "
                "(era-reckoned worlds give the ordinal year, others the absolute year); "
                "no generic fallback is provided."
            )
        merged_time_config["start_year"] = start_year
        # Step duration: set by this analysis together with core_tension; the world's only source of time
        # pricing. Quantized here at the LLM boundary: the prompt speaks hours per step, and everything past
        # this line sees only seconds_per_step, so changing precision means changing only that LLM field.
        merged_time_config.pop("hours_per_step", None)
        try:
            authored_seconds = int(world_time_config["hours_per_step"]) * SECONDS_PER_HOUR
        except (KeyError, TypeError, ValueError):
            authored_seconds = None  # missing or invalid — let resolve_step_seconds fall back to the default
        merged_time_config["seconds_per_step"] = resolve_step_seconds(authored_seconds)

        figures = [
            ThemeFigure(
                name=coerce_str(item.get("name"), fallback="Unknown"),
                role=coerce_str(item.get("role")),
                importance=_normalize_importance(item.get("importance")),
                brief=coerce_str(item.get("brief")),
                age=coerce_int(item.get("age"), default=30, minimum=1, maximum=120),
                gender=_normalize_gender(item.get("gender")),
                secret=coerce_str(item.get("secret")),
            )
            for item in coerce_mapping_list(payload.get("key_figures"))
            if coerce_str(item.get("name"))
        ]
        figures = _distinct_figures(figures)
        relations = [
            RelationSeed(
                source_name=coerce_str(item.get("from")),
                target_name=coerce_str(item.get("to")),
                trust=coerce_float(item.get("trust"), default=NEUTRAL_TRUST, minimum=0.0, maximum=1.0),
                affection=coerce_float(item.get("affection"), default=NEUTRAL_AFFECTION, minimum=-1.0, maximum=1.0),
                labels=coerce_str_list(item.get("labels")),
                reverse_trust=coerce_optional_float(item.get("reverse_trust"), minimum=0.0, maximum=1.0),
                reverse_affection=coerce_optional_float(item.get("reverse_affection"), minimum=-1.0, maximum=1.0),
                reverse_labels=coerce_str_list(item.get("reverse_labels")),
            )
            for item in coerce_mapping_list(payload.get("initial_relations"))
            if coerce_str(item.get("from")) and coerce_str(item.get("to"))
        ]
        # Backstory "hours ago" → steps uses the step duration just set here: both ends are narrative
        # durations and the conversion is pure code-layer quantization, with no outside seconds value borrowed from world_context.
        seconds_per_step = int(merged_time_config["seconds_per_step"])
        historical_events = [
            HistoricalEventSeed(
                event=coerce_str(item.get("event")),
                step_offset=-max(1, round(
                    coerce_int(item.get("hours_before_start"), default=6, minimum=1)
                    * SECONDS_PER_HOUR
                    / seconds_per_step
                )),
                related_figures=coerce_str_list(item.get("related_figures")),
                importance=coerce_float(
                    item.get("importance"), default=0.5, minimum=0.0, maximum=1.0
                ),
            )
            for item in coerce_mapping_list(payload.get("historical_events"))
            if coerce_str(item.get("event"))
        ]

        entity_seeds = _distinct_entity_seeds([
            WorldEntitySeed(
                name=coerce_str(item.get("name")),
                entity_type=coerce_str(item.get("entity_type"), fallback="item"),
                description=coerce_str(item.get("description")),
                initial_state=coerce_str(item.get("initial_state"), fallback="intact"),
                location_name=coerce_str(item.get("location_name")),
                content=coerce_str(item.get("content")),
            )
            for item in coerce_mapping_list(payload.get("world_entity_seeds"))
            if coerce_str(item.get("name"))
        ])

        npc_seeds = _distinct_npc_seeds(
            [
                NpcSeed(
                    name=coerce_str(item.get("name")),
                    gender=_normalize_gender(item.get("gender")),
                    age=coerce_int(item.get("age"), default=30, minimum=1, maximum=120),
                    description=coerce_str(item.get("description")),
                    location_name=coerce_str(item.get("location_name")),
                )
                for item in coerce_mapping_list(payload.get("npcs"))
                if coerce_str(item.get("name"))
            ],
            {figure.name for figure in figures},
        )

        if not figures:
            return ThemeAnalysis(
                theme_input=theme,
                world_name=coerce_str(payload.get("world_name"), fallback=""),
                era_description="",
                core_tension="",
                narrative_theme="",
                narrative_pitch="",
                key_figures=[],
                world_entity_seeds=entity_seeds,
                npc_seeds=npc_seeds,
            )

        core_tension_value = coerce_str(payload.get("core_tension"))
        if not core_tension_value:
            raise ValueError(
                f"core_tension missing for theme '{theme}': LLM must produce a one-sentence "
                "structural-tension statement; no generic fallback is provided."
            )
        narrative_theme_value = coerce_str(payload.get("narrative_theme"))
        if not narrative_theme_value:
            raise ValueError(
                f"narrative_theme missing for theme '{theme}': LLM must produce a one-sentence "
                "tonal statement; no generic keyword fallback is provided."
            )
        narrative_pitch_value = coerce_str(payload.get("narrative_pitch"))
        if not narrative_pitch_value:
            raise ValueError(
                f"narrative_pitch missing for theme '{theme}': LLM must produce a one-sentence "
                "step-0 situational statement; no generic fallback is provided."
            )
        # If omitted, leave it empty: no fallback, no retry (same as AgentGenerator for core_traits/core_values).
        # A fallback doesn't give a slightly worse world: a row of neutral "acquaintance" labels would erase
        # structural labels like kinship or lord-and-vassal that can never be rewritten once set, and a
        # template backstory would be seeded as every character's persistent memory and recalled from the
        # vector store again and again. A world with these empty still runs; relations grow at runtime.
        missing = [
            key for key, value in (
                ("initial_relations", relations), ("historical_events", historical_events),
            ) if not value
        ]
        if missing:
            logger.warning(
                "theme_analysis_fields_missing",
                extra={"theme": theme, "fields": missing},
            )
        return ThemeAnalysis(
            theme_input=theme,
            world_name=coerce_str(payload.get("world_name"), fallback=f"{theme}"),
            era_description=coerce_str(
                payload.get("era_description"),
                fallback=f"以「{theme}」为背景的叙事世界。",
            ),
            core_tension=core_tension_value,
            narrative_theme=narrative_theme_value,
            narrative_pitch=narrative_pitch_value,
            world_time_config=merged_time_config,
            key_figures=figures,
            initial_relations=relations,
            historical_events=historical_events,
            key_locations=[],
            world_entity_seeds=entity_seeds,
            npc_seeds=npc_seeds,
            source_material=source_material,
        )




def _normalize_importance(value: Any) -> str:
    lowered = coerce_str(value).lower()
    return AgentTier.MAIN.value if lowered == AgentTier.MAIN.value else AgentTier.BACKGROUND.value


_GENDER_SYNONYMS: dict[str, str] = {
    "男": "男", "男性": "男", "male": "男", "m": "男",
    "女": "女", "女性": "女", "female": "女", "f": "女",
}


def _normalize_gender(value: Any) -> str:
    """The prompt's 「男 / 女」, canonical: these two values are what every consumer may compare
    against (the frontend picks a body sheet by it). Anything else becomes "" — unknown, left out
    of prompts, never guessed."""
    return _GENDER_SYNONYMS.get(coerce_str(value).lower(), "")


