"""Persona generation: one LLM call per figure, parsed into an ``AgentDefinition``."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Sequence

from agent.need import NeedState, NeedType, build_innate_needs
from agent.personality import (
    SECRET_LABEL,
    EmotionState,
    EmotionType,
    SoulLayer,
    parse_emotion_type,
)
from core.context import annotate_call, observe_stage
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json_object, output_budget
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.text import slugify
from core.prompts import (
    EMOTION_INTENSITY_DEFINITION,
    EMOTION_VALENCE_DEFINITION,
    NEED_INTENSITY_DEFINITION, NEED_TYPE_DEFINITION, NEED_WEIGHT_DEFINITION, relation_legend,
)
from core.coerce import (
    coerce_float, coerce_mapping_list, coerce_optional_str, coerce_str, coerce_str_list,
)

from world.identity_color import PALETTE_CHOICES, resolve_identity_colors
from world.models import (
    AgentDefinition, AgentTier, CastDesign, FigureCastRole, HistoricalEventSeed, RelationSeed,
    ThemeAnalysis, ThemeFigure,
)
from world.builders import BUILD_ATTEMPTS

logger = get_logger(__name__)


class AgentGenerator:
    """Generate runtime-facing agent definitions from theme analysis."""

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def generate(
        self,
        analysis: ThemeAnalysis,
        cast_design: CastDesign | None = None,
    ) -> list[AgentDefinition]:
        """Build agent definitions for each key figure."""

        used_ids: set[str] = set()
        task_contexts: list[tuple[ThemeFigure, str]] = []
        tasks = []
        for figure in analysis.key_figures:
            agent_id = _unique_agent_id(figure.name, used_ids)
            cast_role = cast_design.role_for(figure.name) if cast_design else None
            task_contexts.append((figure, agent_id))
            tasks.append(self._generate_definition(figure, analysis, agent_id, cast_role=cast_role))

        if not tasks:
            return []
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        definitions: list[AgentDefinition] = []
        failed: list[str] = []
        for (figure, agent_id), result in zip(task_contexts, raw_results):
            if isinstance(result, BaseException):
                logger.error(
                    "agent_definition_failed",
                    extra={"figure_name": figure.name, "error": str(result)},
                )
                failed.append(figure.name)
            else:
                definitions.append(result)
        if failed:
            raise ValueError(
                f"AgentGenerator failed for figures: {', '.join(failed)}"
            )
        # Assign fixed identity colours across the WHOLE cast at once — distinctness is
        # a set-level constraint, so it runs after every persona is generated.
        definitions = resolve_identity_colors(definitions)
        logger.info(
            "world_agent_definitions_generated",
            extra={
                "world_name": analysis.world_name,
                "agent_count": len(definitions),
                "main_count": sum(1 for d in definitions if d.is_main_character),
            },
        )
        return definitions

    async def _generate_definition(
        self,
        figure: ThemeFigure,
        analysis: ThemeAnalysis,
        agent_id: str,
        *,
        cast_role: FigureCastRole | None = None,
    ) -> AgentDefinition:
        system_prompt, user_prompt = self._build_prompt(
            figure, analysis, cast_role=cast_role
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        last_exc: BaseException | None = None
        for attempt in range(BUILD_ATTEMPTS):
            try:
                # Tag the trace with who is being generated: agent_id lets the dev tools identify and filter;
                # the name is recorded separately because a build that fails at this step has no snapshot yet, so
                # the dev tools can't look the name up from the id. Set it inside this coroutine, not before
                # gather, or the characters' ids would get mixed up across each other's calls.
                with observe_stage(Stage.WORLD_INIT, agent_id=agent_id), \
                        annotate_call(agent_name=figure.name):
                    response = await self._llm_router.complete(
                        # reason ≤80 chars ≈ 125, plus one character's full persona: background (≤140 chars ≈ 210) + self_image/life_goal (≤50 chars each ≈ 75)
                        # + appearance (≤40 chars ≈ 60) + long_term_goals (3 × ≤40 chars ≈ 180) + hard_constraints (3 × ≤40 chars ≈ 180)
                        # + 5 × initial_needs (label ≤60 chars ≈ 90 + 3 numbers + structure ≈ 110) + traits/values/location/emotion ≈ 200;
                        # estimate ~1650 tok.
                        LLMScene.PERSONA_GENERATION, messages, max_tokens=output_budget(1650),
                        json_mode=True,
                    )
                payload = extract_json_object(response.content)
                if payload is not None:
                    return self._definition_from_payload(figure, analysis, payload, agent_id)
                last_exc = ValueError(
                    f"No parseable JSON for figure '{figure.name}' (attempt {attempt + 1})"
                )
            except Exception as exc:
                last_exc = exc
            logger.warning(
                "agent_definition_attempt_failed",
                extra={
                    "figure_name": figure.name,
                    "attempt": attempt + 1,
                    "reason": str(last_exc),
                },
            )
            if attempt < BUILD_ATTEMPTS - 1:
                await asyncio.sleep(1.0)
        raise ValueError(
            f"AgentGenerator failed for figure '{figure.name}' after {BUILD_ATTEMPTS} attempts"
        ) from last_exc

    def _build_prompt(
        self,
        figure: ThemeFigure,
        analysis: ThemeAnalysis,
        *,
        cast_role: FigureCastRole | None = None,
    ) -> tuple[str, str]:
        """Return ``(system, user)`` split for prefix-cache hit rate.

        ``system`` = existing role one-liner + fully-static 【任务】/【约束】
        blocks — byte-identical across every AgentGenerator call, so DeepSeek /
        OpenAI-compatible providers cache the whole prefix.

        ``user`` orders volatile input so the N-figures loop for one world shares
        as long a prefix as possible:
          1. WORLD-SHARED context (world_name / core_tension / narrative_theme /
             narrative_pitch / available locations) FIRST — identical across all figures of
             this build, so ``system + world-shared`` becomes the shared prefix
             across all N calls.
          2. PER-FIGURE data AFTER (figure profile, cast_role, related relations).
          3. Schema at the end, next to the generation point + one short output
             reminder.
        """

        # ── system: fully-invariant scaffolding.
        system_sections: list[str] = [
            "你负责为叙事角色生成紧凑的 JSON 人物种子数据。只返回 JSON，不包含任何其他内容。",
        ]

        system_sections.append(
            "【任务】\n"
            "为该人物生成紧凑的人物种子数据。先在 reason 里想清楚这个人是什么样的，再据此写其余字段。三类不变量的塑造分工——（只是参考，不用生搬硬套）\n"
            "- 核心张力 决定该角色的 background / core_traits / core_values / hard_constraints / "
            "life_goal：这些是角色在张力中长期形成的身份层。\n"
            "- 叙事主题 决定 core_values 的意义指向、trait 与措辞的色彩气质，"
            "以及 initial_emotion.primary 倾向的情感谱系（主题决定纹理与指向，不决定具体值）。\n"
            "- 开局情境聚焦 决定 initial_emotion 的 intensity / valence / triggered_by、"
            "以及 hidden_needs 的当前激活度——它只影响 step 0 当下的状态，不影响身份。\n"
            "- initial_needs 要分清两层：weight 由核心张力 / 身份决定——它是这个角色【一生】的结构性"
            "需求优先级（典型 1.0，只有当某需求确实是他长期最看重 / 最不在意的，才偏离）；intensity 由"
            "开局情境决定——它是该需求【此刻】的紧迫度（当下状态），允许被开局危机抬高。【开局危机只抬高 "
            "intensity，绝不写进 weight】：别把『这阵子的处境』固化成『他一生的性格优先级』。例如危局可让某需求"
            "此刻 intensity 很高，但若它本不是该角色一生所重，weight 仍应保持中性。\n"
            "- 角色不必有「对立面」——根据核心张力的形态，可以是寻找者、等待者、改变中的人、"
            "目击者、抉择者、守护者等任一定位。"
        )

        system_sections.append(
            "【约束】\n"
            "- initial_needs 和 hidden_needs 的内容不可以重复，即 type 和 label 不可以相同：如果 "
            "initial_needs 已经有了，那 hidden_needs 就不可以再出现，宁愿为空也不可以重复。"
        )
        system_prompt = "\n\n".join(system_sections)

        # ── user: world-shared context (identical across all N figures of this build, placed first to
        # lengthen the shared prefix across figures) → per-figure data → schema → output reminder.
        # Don't have the model echo name / role / agent_id: identity is fixed in code and echoes aren't
        # trusted, and copying a random id invites copying it wrong.
        schema = {
            # First: every field below is derived from this read of the character (CLAUDE.md §5).
            "reason": "动笔前先想清楚这个人是什么样的（≤80字）。只分析，不复述输入原文。",
            "core_traits": ["性格特点（1-2词），共3-5个"],
            "core_values": ["核心价值观（1-2词），共2-4个"],
            "background": "人物背景（2-3句、≤140字）：描述成长经历、出身与长期形成的处世方式。只写长期稳定的身份事实，不要写此刻在哪、在做什么，也不要预测未来。用第三人称。如果人物有不想让别人知道的事，不建议写到这里。",
            "self_image": f"以第一人称视角描述自我认知（一句话、≤50字，如：我是...）。人物有「{SECRET_LABEL}」时，他们的描述应该是自洽的。",
            "appearance": "外在形象（一句话，≤40字）：体貌、衣着或气度的标志性特征。用第三人称，只写长期稳定的外在特征，不写此刻的动作或情绪。",
            "signature_color": f"最能代表该角色气质的颜色，从以下选项中选一个填入（只填英文名）：{PALETTE_CHOICES}。凭其身份、性格与处世气度择一最贴切者。",
            "life_goal": "人生终极目标（一句话、≤50字），可选，也可以没有，不用硬编。用第一人称。",
            "long_term_goals": ["当前阶段的长期目标（一句话、每条≤40字，最多3条）。长期目标是要走很长一段路才能推进或守住的【方向】：比手头的小事远得多、又不像 life_goal 那样遥不可及，需持续奋斗、不会轻易到手，且是个能长出多种走法的方向，而非某一桩具体的事或一次具体行动。以角色自身当下的视角表述意图，不得引用任何事件的史称或事后命名，不得把尚未发生的事件当作已确定的未来，这个很重要！"],
            "hard_constraints": ["<一句话、≤40字不可逾越的底线，描述该角色绝对不会做或绝对会坚守的事，至多3条，也可以一条都没有>"],
            "initial_location": "初始地点（使用可用地点列表中的 name 字段值），需要结合角色本身和开局情景选择地点，不要乱选。",
            "initial_emotion": {
                "primary": f"情绪类型（{EmotionType.prompt_list()} 之一）",
                "intensity": EMOTION_INTENSITY_DEFINITION,
                "valence": EMOTION_VALENCE_DEFINITION,
            },
            "initial_needs": [
                {
                    "type": NEED_TYPE_DEFINITION,
                    "label": "一句话、≤60字描述该需求对这个角色的主观意义：他为何在意、如何体验这种持久的内在驱动力。不要写具体目标、时间节点或行动计划——那是目标层的内容，请用第一人称。",
                    "intensity": NEED_INTENSITY_DEFINITION,
                    "weight": NEED_WEIGHT_DEFINITION,
                    "is_hidden": False,
                }
            ],
            "hidden_needs": "格式同initial_needs，is_hidden固定为true。与 initial_needs 的区别是【觉知度】，不是【种类】：initial_needs 是人物自己【意识得到、会据以解释和行动】的驱动；hidden_needs 是他【未意识到、却在暗中左右行为】的深层驱动（被压抑/不愿承认的渴望），它只微弱地偏置行为、不会成为他明面上的主导，常造成『嘴上要A、实则被B推着走』的自我欺骗。仅在该角色确有这种潜意识张力时才写，否则空列表[]。",
        }
        related = [
            relation.as_dict()
            for relation in analysis.initial_relations
            if relation.source_name == figure.name or relation.target_name == figure.name
        ]

        # 【输入】: world-shared lines first (identical across all figures),
        # then per-figure lines (profile / narrative role / related relations).
        input_lines = [
            "【输入】",
            # ---- WORLD-SHARED (identical across all N figures of this build) ----
            f"- 世界：{analysis.world_name}",
            f"- 核心张力（引擎，预先存在）：{analysis.core_tension}",
            f"- 叙事主题：{analysis.narrative_theme}",
            f"- 开局情境聚焦（step 0 当下）：{analysis.narrative_pitch}",
            f"- 可用地点：{json.dumps([location.as_dict() for location in analysis.key_locations], ensure_ascii=False)}",
            # ---- PER-FIGURE (varies each call, so placed AFTER the shared prefix) ----
            f"- 人物：{figure.as_prompt()}",
        ]
        if cast_role is not None:
            input_lines.append(f"- 叙事角色：{cast_role.narrative_role}")
            input_lines.append(f"- 人物弧：{cast_role.arc_summary}")
            if cast_role.key_relationships:
                input_lines.append(f"- 主要关系人物：{', '.join(cast_role.key_relationships)}")
        if related:
            input_lines.append(f"- 相关关系字段说明：{relation_legend()}")
        input_lines.append(f"- 相关关系：{json.dumps(related, ensure_ascii=False)}")

        # 【输出】: schema at the end of user, next to the generation point (§3).
        output_block = (
            "【输出】\n"
            "只返回符合以下 Schema 的紧凑 JSON，不要任何多余内容：\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )
        user_prompt = "\n\n".join([
            "\n".join(input_lines),
            output_block,
            "严格按上述 JSON 输出，不写多余内容。",
        ])
        return system_prompt, user_prompt

    def _definition_from_payload(
        self,
        figure: ThemeFigure,
        analysis: ThemeAnalysis,
        payload: Mapping[str, Any],
        agent_id: str,
    ) -> AgentDefinition:
        _active = _need_states_from_payload(payload.get("initial_needs")) or _default_need_states(figure)
        _hidden = _need_states_from_payload(payload.get("hidden_needs"), default_hidden=True)
        # If the LLM omits them, leave them empty and don't retry: any fallback trait/value words would push the character toward one narrative type.
        # Deduplicated in order: the model sometimes repeats a word (['体面', '体面']).
        core_traits = list(dict.fromkeys(coerce_str_list(payload.get("core_traits"))))
        core_values = list(dict.fromkeys(coerce_str_list(payload.get("core_values"))))
        missing = [k for k, v in (("core_traits", core_traits), ("core_values", core_values)) if not v]
        if missing:
            logger.warning(
                "agent_definition_fields_missing",
                extra={"figure_name": figure.name, "fields": missing},
            )
        soul = SoulLayer(
            # figure.name is the join key fixed by the ThemeAnalyzer call: initial_relations'
            # source_name/target_name and historical_events' related_figures all join on it (see the
            # initializer's name_to_id). Don't trust the name echoed by this call instead: if the model adds
            # an honorific, all of that character's seed relations and backstory memories silently miss while
            # the build still reports success.
            name=figure.name,
            role=figure.role,
            agent_id=agent_id,
            age=figure.age,
            gender=figure.gender,
            core_traits=core_traits,
            core_values=core_values,
            # Fallbacks must stay timeless: figure.brief is a step-0 present-moment
            # seed, and self_image/background are frozen soul fields fed into every
            # runtime cognition prompt — seeding them from brief would freeze a
            # step-0 situation (and any predicted future) permanently into identity.
            self_image=coerce_str(payload.get("self_image"), fallback=f"我是{figure.name}"),
            background=coerce_str(payload.get("background"), fallback=""),
            appearance=coerce_str(payload.get("appearance"), fallback=""),
            # soul.color (final hex) is assigned cast-wide AFTER all figures are
            # generated (see resolve_identity_colors) — distinctness is a set-level
            # constraint an isolated per-figure call can't satisfy. The LLM's colour
            # NAME rides in metadata as the resolver's input.
            # No fallback — the prompt says life_goal is optional, so absent stays
            # absent: one shared placeholder would give every agent who lacks one the
            # same life goal, and this field is an identity anchor fed into runtime
            # cognition prompts. For the same reason it must never be backfilled from
            # core_tension — that is a build-time invariant, and putting it on a soul
            # leaks it into runtime emergence.
            life_goal=coerce_optional_str(payload.get("life_goal")),
            # From ThemeAnalyzer, not this call: only that call sees the whole cast, so only it can
            # keep cross-figure facts (how many hold which secret) consistent.
            secret=figure.secret,
            hard_constraints=coerce_str_list(payload.get("hard_constraints")),
            innate_needs=build_innate_needs(_active, _hidden),
        )
        return AgentDefinition(
            agent_id=agent_id,
            name=soul.name,
            tier=figure.tier,
            soul=soul,
            # Keep the LLM's text as-is (it may be a name, alias or id) and don't resolve it here. Text →
            # location id has one owner, ``WorldConfig.resolve_location_id`` (accepts ids, aliases and display
            # names), called by world/builder.py at assembly. Slugifying here first would grind "东宫" into a
            # nonexistent id the real resolver can't recognize, and every character would fall back to the
            # same default location.
            initial_location=(
                coerce_str(payload.get("initial_location"))
                or _pick_initial_location(figure, analysis)
            ),
            initial_emotion=_emotion_from_payload(payload.get("initial_emotion"), fallback=_default_emotion(figure)),
            long_term_goals=coerce_str_list(payload.get("long_term_goals")),
            initial_relations=_relations_for_figure(analysis.initial_relations, figure.name),
            historical_memories=_historical_memories_for_figure(analysis.historical_events, figure.name),
            metadata={
                "brief": figure.brief,
                "source": "llm",
                # Transient: the LLM's narrative colour name, consumed by
                # resolve_identity_colors after the whole cast is generated.
                "signature_color": coerce_str(payload.get("signature_color"), fallback=""),
            },
        )


def _unique_agent_id(name: str, used_ids: set[str]) -> str:
    base = slugify(name, fallback_prefix="agent")
    candidate = base
    counter = 2
    while candidate in used_ids:
        candidate = f"{base}-{counter}"
        counter += 1
    used_ids.add(candidate)
    return candidate



def _pick_initial_location(figure: ThemeFigure, analysis: ThemeAnalysis) -> str:
    return analysis.key_locations[0].name if analysis.key_locations else "origin"


def _default_emotion(figure: ThemeFigure) -> EmotionState:
    """Tension-neutral fallback emotion when the LLM omits initial_emotion.

    Uses a flat NEUTRAL emotion with zero valence — pre-tuned positive /
    negative valence would presuppose a particular narrative posture and
    bias subsequent perception. Tier shapes intensity only (main characters
    enter with slightly more affective charge than background agents).
    """
    intensity = 0.4 if figure.tier == AgentTier.MAIN else 0.3
    return EmotionState(
        primary=EmotionType.NEUTRAL,
        intensity=intensity,
        valence=0.0,
        triggered_by="世界初始化",
    )


def _emotion_from_payload(value: Any, *, fallback: EmotionState) -> EmotionState:
    if isinstance(value, Mapping):
        raw_primary = coerce_str(value.get("primary"), fallback=fallback.primary.value)
        return EmotionState(
            primary=parse_emotion_type(raw_primary),
            intensity=coerce_float(value.get("intensity"), default=fallback.intensity, minimum=0.0, maximum=1.0),
            valence=coerce_float(value.get("valence"), default=fallback.valence, minimum=-1.0, maximum=1.0),
            triggered_by=coerce_optional_str(value.get("triggered_by")) or fallback.triggered_by,
        )
    label = coerce_str(value)
    if not label:
        return fallback
    emotion_type = parse_emotion_type(label)
    if emotion_type in {EmotionType.ANGER, EmotionType.FRUSTRATION, EmotionType.FEAR, EmotionType.DISGUST, EmotionType.SHAME, EmotionType.CONTEMPT, EmotionType.SADNESS, EmotionType.JEALOUSY}:
        return EmotionState(primary=emotion_type, intensity=0.6, valence=-0.45, triggered_by="人物生成")
    if emotion_type in {EmotionType.JOY, EmotionType.ANTICIPATION, EmotionType.TRUST, EmotionType.PRIDE}:
        return EmotionState(primary=emotion_type, intensity=0.45, valence=0.15, triggered_by="人物生成")
    return EmotionState(primary=emotion_type, intensity=0.3, valence=0.0, triggered_by="人物生成")


def _need_states_from_payload(value: Any, *, default_hidden: bool = False) -> list[NeedState]:
    states: list[NeedState] = []
    for item in coerce_mapping_list(value):
        need_type = _coerce_need_type(item.get("type"))
        if need_type is None:
            continue
        states.append(
            NeedState(
                type=need_type,
                label=coerce_str(item.get("label"), fallback=_default_need_label(need_type)),
                intensity=coerce_float(item.get("intensity"), default=0.4, minimum=0.05, maximum=1.0),
                weight=coerce_float(item.get("weight"), default=1.0, minimum=0.1, maximum=2.0),
                is_hidden=coerce_bool(item.get("is_hidden"), default_hidden),
            )
        )
    return states


def _default_need_states(figure: ThemeFigure) -> list[NeedState]:
    if figure.tier == AgentTier.MAIN:
        return [
            NeedState(NeedType.SAFETY, "减少不确定性", intensity=0.70, weight=1.0),
            NeedState(NeedType.ESTEEM, "维护地位", intensity=0.62, weight=1.0),
            NeedState(NeedType.SOCIAL, "维系关键盟友", intensity=0.58, weight=1.0),
            NeedState(NeedType.SELF_ACTUALIZATION, "推进长远布局", intensity=0.66, weight=1.0),
        ]
    return [
        NeedState(NeedType.SAFETY, "保持安全", intensity=0.64, weight=1.0),
        NeedState(NeedType.SOCIAL, "与盟友保持一致", intensity=0.48, weight=1.0),
        NeedState(NeedType.ESTEEM, "避免受辱", intensity=0.42, weight=1.0),
    ]


def _coerce_need_type(value: Any) -> NeedType | None:
    text = coerce_str(value).lower()
    if not text:
        return None
    try:
        return NeedType(text)
    except ValueError:
        return None


def _default_need_label(need_type: NeedType) -> str:
    mapping = {
        NeedType.PHYSIOLOGICAL: "维持健康与体力",
        NeedType.SAFETY: "保持安全、降低风险",
        NeedType.SOCIAL: "维系联系与信任",
        NeedType.ESTEEM: "维护地位与尊严",
        NeedType.SELF_ACTUALIZATION: "推进长远目标",
    }
    return mapping[need_type]


def _relations_for_figure(relations: Sequence[RelationSeed], figure_name: str) -> list[RelationSeed]:
    return [
        relation
        for relation in relations
        if relation.source_name == figure_name
    ]


def _historical_memories_for_figure(
    events: Sequence[HistoricalEventSeed],
    figure_name: str,
) -> list[HistoricalEventSeed]:
    return [
        event
        for event in events
        if figure_name in event.related_figures
    ]
