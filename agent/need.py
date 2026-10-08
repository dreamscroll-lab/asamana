"""Need evaluation for agent decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:
    from agent.motivation import ExternalGoal
    from agent.relation import PerceivedRelation

from agent.goals import (
    GoalEntity, GoalOrigin, GoalStatus, active_goals, enqueue_goals,
    goal_age_hint, goal_due_hint, is_active_goal, is_due,
    is_past_allowance, live_goals, order_goals_for_prompt, parse_goal_evaluation,
    parse_goal_item, parse_residue, trim_goal_history,
)
from agent.personality import SECRET_LABEL, EmotionState, PersonalityLayer
from core.context import GivenFacts, annotate_active_call, annotate_call
from core.duration import describe_duration
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.interfaces.perception import Situation
from core.logging import get_logger
from core.numeric import clamp
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    CLOSED_WORLD_FACT_RULE,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    MEMORY_ORDER_HINT,
    SituationVoice,
    condition_line,
    recency_prefix,
    relation_legend,
    render_situation_header,
    situation_location,
    urgency_label,
    vitality_line,
)

_logger = get_logger(__name__)


# Prefix-cache split: system is byte-invariant (functional, out-of-character judge); user holds this
# call's inputs. The subject's name appears only in user; system says "被评判者" to stay invariant.
_GOAL_EVALUATION_SYSTEM = """\
【角色】
你是一个目标进度判定器。下面给出被评判者本轮的行动与结果，以及他当前的若干短期目标；
你的职责是**客观**判定每个目标在这次行动之后的状态，并指出这次行动留下了什么没有了结的事项。

【任务一：逐个目标判定状态】
逐个目标判定其状态，状态只能取以下之一：
- completed：这次行动已使该目标真正达成
- active：该目标仍在推进中、尚未达成
- interrupted：该目标因这次行动被打断、暂时无法继续
- failed：该目标明确失败、无法再达成、已过期；**或已反复推进多步却始终原地打转、毫无实质进展（当前手段对它无效），就此搁置**。

【约束】
- 判定依据是本轮行动与结果、**以及所给"近期客观经历"这条已记录的轨迹**——有些目标跨多步累积达成，最新这一步可能正好把它推过完成线，须结合轨迹判断；但只能依据**已记录的事实**，不得臆测这些记录之外、未给出的信息，**也不得借此重新解读目标本身**。
- 允许从结果合理倒推：若行动结果在常理上已蕴含某目标达成/过期（即便结果未点名该目标），可据此判 completed / failed——但倒推链必须落在结果上、写进 reason，不能凭空脑补。（例：目标是「去餐厅吃饭」，结果是「我吃饱了」，结果虽未明确表示去了餐厅，但常理上已蕴含吃饭达成，可据此倒推判 completed。 ）
- 一次成功的行动**未必**使目标 completed——多数目标要多步推进，没真正达成就保持 active。
- **警惕「永远 active」**：目标文本后可能标着它「已历约多久仍未了结」。若一个目标已历时明显偏久、且轨迹显示反复推进却始终停在原地（结果一再未达预期、状态毫无实质变化），说明当前手段对它无效——判 failed（搁置），不要无限 active。一个推进不动的目标会锁死后续认知。判 failed（搁置）时，reason 必须点明「反复无进展」的具体依据，落在已记录的轨迹 / 时长上，不臆测。
- 但别矫枉过正：真正在**逐步推进**（每步能看到可见进展、只是尚未跨过完成线）的多步目标保持 active——搁置只针对「原地打转」，不是「还没做完」；拿不准是「有进展但慢」还是「原地打转」时，倾向保留 active。
- 判断之前先明确好当前时间和当前地点，避免在结果中涉及到时间问题时出现错乱的情况。
- **每个目标必须把判断依据(reason)写在最前面，再给序号(index)与状态(status)——先分析后定状态，不要凭空直接给状态。**
- 只输出 JSON，不要任何多余内容。

【任务二：指出这次行动被评判者新留下的没有了结事项（residue）】
除了上面既有目标的进退，这次行动本身可能产生被评判者**没有了结的事项**。先分析、再指出。

算 residue 的（只要落在本轮行动与结果上、有据可依）：
- 行动结果里明确表示、或由结果直接通过常理推得的下一步将要做的事项
- 本轮行动只做了部分事项，还有未完成事项需要继续做

不算 residue 的（写了就是错）：
- 复述已经发生的事
- 泛泛的心情、决心、表态（「我要更谨慎些」「我得振作」这类不是没有了结的事项）
- 没有任何依据或者通过推理也得不到的你自己臆想的事项
- 与上面待判定目标里已有的那些意思相近或相同的事项
- **指向所给材料里根本不存在的人、地方或东西的事项**
- 不属于被评判者的事项

约束：
""" + CLOSED_WORLD_FACT_RULE + """
- 用**被评判者本人的口吻、第一人称**写（「我……」），每条不超过 28 字，最多 2 条；里头若提到
  时间，写具体时间点，不写「明日 / 今夜」等等相对时间
- due_in_hours：**这件事说定了截止时间，就填从此刻到截止时间还有几小时；没说定时间就填null或者都不填写。**
  比如当前时间为6月3日上午10点，截止时间为6月6日中午12点，那它们之间相差74小时，due_in_hours就写74。
  **但 text 里若已经写下了一个具体时点，due_in_hours 就不可以为 null**——那个时候是这条事项自己
  带着的，把它折算成小时填上即可，不是另添一个期限。
- 每条事项必须具体，明确，不能模糊不清，模糊不清的事项不用生成，比如“我需要立刻质问小A”这个就非常模糊，因为它根本没有说明“需要质问的内容是什么？”
- **绝大多数行动不留下任何没有了结事项**（歇息、赶路、寒暄、独自琢磨都是如此）——这时给空数组
  `[]`，这是正常且被期望的结果。不要为了填满而胡编乱造，宁可给空。

【输出】
只输出 JSON，不要任何多余内容：
{"goals": [{"reason": "一句话判断依据，一定不可以超过100个字", "index": 1, "status": "active"}], "residue_reason": "……", "residue": [{"text": "我……", "due_in_hours": null}, {"text": "我……", "due_in_hours": 10}]}
每个待判定目标各输出一项，reason 在最前（先分析）、index 与 status 紧随其后；
index 为所给待判定目标的序号（整数，从 1 开始）；status 只能取 completed/active/interrupted/failed。

residue 那两项放在最后——要先判完上面每个目标的去留，才知道这次行动真正**新**留下的是什么。
两项之间同样是先想后判：
- residue_reason（≤60字）：先在这里逐一审视本轮行动与结果，说清**哪一句话**让你认为留下了没了结的
  事项、它为什么不属于上面已有的目标；**若认为什么都没留下，也要在这里说明为什么**（多数行动本
  就如此，这是正常答案，不必勉强找出点什么）。
- residue：由上面的分析推出的条目；没有则给 []。**不得出现分析里没有依据的条目。**
""" + ABSOLUTE_TIME_RULE + """"""


_GOAL_EVALUATION_USER = """\
{situation_header}

【被评判者】{agent_name}

【输入】
本轮行动：{action_description}
本轮行动期望的结果：{expected_outcome}
行动结果：{action_result}
{recent_experience}待判定的目标（按序号）：
{goals_numbered}

依上面说定的两项任务、约束与 JSON 格式给出裁定（先逐个目标判状态，再给 residue），
只输出 JSON、不写任何多余内容。"""


# Prefix-cache split: in-character; system holds the invariant rules and format (step_duration is
# per-world constant), user holds this call's inputs.
# The text field's description repeats the absolute-time rule on purpose: goal text is reread every
# step for many steps, and when writing text the model only looks at the field description. Keep both.
# thought's checklist must keep "which long-term direction I haven't touched": goals are derived from
# thought, so a long-term goal not on the checklist takes no part. It's a review, not an order.
_SHORT_TERM_GOAL_SYSTEM = """\
你此刻完全代入一个角色，以第一人称「我」思考「接下来要往哪儿使力」。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【我要想清楚的】
就着眼前的处境，我接下来这几步内**新**想推进的 1-2 件事。（我这样的每一步大约是{step_duration}；短期目标要落在需要跨过不止一步才推进得动的方向上，而不是一步就做完的单个动作。）
- **首先回应此刻最迫切的那个需求、以及我刚刚感知到的——当下真正在驱动我的是这些**。
- 长期目标只是远处的方向：拿它把握大体不跑偏即可，**不必每件都直奔它，更不能用它盖过眼前**。
- 回头看：哪个长期方向我近来一直没沾。够得着就添，够不着只在 thought 里点破，别硬凑。
- 每件都要具体到能立刻着手；若有外部压力已经逼到眼前，先回应其中最紧急的那件。
- 这是"意图/方向"，不是一锤定音的大计——具体怎么落地，临场再定。若眼下确实没有新的方向要添，goals 给空数组即可。
- 宁可不输出新目标，也不要输出与【我的短期目标】意思相近或者相同的目标，更不要通过细化【我的短期目标】的目标形成新的目标。
  例如，目标意思相近：“我要去上学”，“我去读书了”，目标意思相等“我要去上学”，“我要去上学”，“我要和刘总在公司谈论投资的事情”的子目标“我要先把刘总邀请到公司”。这些例子都是禁止的。

【禁止，必须遵守】
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + """
- 抽象口号或表态式的空话，要落到具体可着手的事
- 用长期目标 / 终局盖过当下处境——短期目标首先服务此刻最迫切的需求与刚感知到的，长期只作背景，不能把眼前晾在一边直奔终局（短期目标 ≠ 长期目标，更不是把长期大计抄下来）
- 短期目标是一个方向，不是具体的行动指南或具体行动，禁止直接把具体行动作为短期目标。
- 直接照搬当前情绪当目标，或形成违背我价值观的目标
- 因为某人**只是在场**（并没有对我做什么）就生成**针对他的激烈目标**（对抗、铲除、严防到行动）——在场至多让我"留意、相机行事"，激烈目标得有指向我的具体事件或压力撑着
- 重复我刚刚已经做完的事，要往前推进
- 跳出第一人称，用旁观者口吻、或事后才有的命名去指代正在发生的事
- 预设未来某件事必然发生——只依据我此刻已经知道的
- 综合所有信息后，如果存在明显的事实性冲突，禁止直接忽略严重的或需要关注的事实性冲突，必须要思考为什么会有这样的冲突，必要时可以在新的目标中体现。
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【输出】
只输出 JSON，不要任何多余内容。先在 thought 里（≤75 字、以"我"的口吻）把眼下想清楚——此刻真正在驱动我的是哪个需求、我刚感知到了什么、我的短期目标里哪些已经覆盖到了（这些不必再提）、哪个长期方向我一直没沾——由此推出真正还**新**缺的方向，再落到 goals：先写这件事本身（text，每条不超过 32 字；里头若提到时间，写具体时间点，不写「明日 / 今夜」等），再看它有没有定下时候（due_in_hours）。
- 有新方向 → {{"thought": "我此刻……", "goals": [{{"text": "目标1", "due_in_hours": null}}, {{"text": "目标2", "due_in_hours": 10}}]}}
- 确实没有新方向要添 → {{"thought": "我此刻……", "goals": []}}

due_in_hours：**这件事约定了截止时间，就填从此刻到截止时间还有几小时；没说定时间就填null或者什么都不填写。填之前一定要想清楚，记住什么都不填写比乱填好。**
      比如此刻是晚上八点，事情定在第二天清早 → 约十小时 → 10
      **但我刚在 text 里写下了一个具体时点的，due_in_hours 就不能留 null**——那个时候是我自己定的，把它折算成小时填上即可，不是另添一个期限。"""


_SHORT_TERM_GOAL_USER = """\
{situation_header}{condition_line}{vitality_line}

【我是谁】
我是{agent_name}。下面全部以"我"的第一人称、就在此刻的视角去想：我接下来到底想做什么。

【我的人设与处境】
我的性格：{core_traits}
我的价值观：{core_values}{secret_line}
我此刻的情绪：{emotion}
我眼下最迫切的需求：{dominant_need_label}
我的长期目标：
{long_term_goals}
{relations_section}
我刚刚感知到：
{recent_events}
{recent_memory_section}{external_drives_section}{current_goals_section}{recently_completed_section}{recent_foiled_section}
我依上面说定的规矩与 JSON 格式说出这一拍新想推进的方向，只输出 JSON、不写任何多余内容。"""


# Prefix-cache split as above: invariant rules and format in system, this call's inputs in user.
_LONG_TERM_GOAL_UPDATE_SYSTEM = """\
你此刻完全代入一个角色，以第一人称「我」重审自己的长期目标。以下是我做这件事时一贯遵守的规矩和说话的格式——它们不随处境改变。

【我要想清楚的】
就着这些事，回头看我那几条长期目标还对吗？衡量的根子只有一条：这段经历有没有真的改变我要走的方向——真变了才动，没变就守住。
什么样的经历才算"真的改变了方向"？常见的有这么几种（但绝不止这些；归根到底还是回到上面那条根子去判断，别只盯着这几条对号入座，也别因为某件事不在其列就轻易放过）：
- 某条目标已经被我做成了——它实现了，该从我往后要奔赴的方向里退场
- 某条目标已经再无可能——路被堵死、它赖以成立的前提没了，再死守也是空耗
- 我看清了什么、心气变了——同一桩处境，今日的我和往日的我，要走的路已经不同
- 这些事逼出了一个新方向，且它分量够重，够得上我从此把它当成长期的奔头
真要动，手法不外乎：改写某条、放下一条（已做成的、再无可能的、或不再属于我的）、或添一条新的——其中"放下"和"添"同样要紧，别只惦着添新的，却把早该退场的旧目标一直攥在手里。

【我要避开的】
- 为变而变：方向没真变，就别动那几条长期目标
- 背弃或改写我的毕生追求：北极星只用来校准长期目标，不能被推翻
- 把长期目标缩成手头的事：眼下的小事、一时的情绪、或几步就能了结的某桩具体事，都不是长期目标——它是要走很长一段路的奔头
- 把长期目标锁成一桩具体的事或一次具体行动：那就成了计划——它该是个能长出很多种走法、容得下变化的【方向】
- 把长期目标吹成空话：大到像句喊给人听的口号、这辈子也不必真使劲，那是北极星的事——它该是要我持续奋斗才能推进或维系的真实奔头
- 跳出第一人称，用旁人或事后的口吻评判自己
- 目标是否实现不是逐字对比一致才认为是实现了目标，而应该从目标本身去对比。
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + "\n" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【输出】
只输出 JSON，不要任何多余内容。先在 thought 里以"我"的口吻（≤250 字）、把我那几条长期目标逐一过一遍——这一条还成立吗？是已经被我做成了、还是再无可能了、还是其实没变？——一条条想清楚了，再落结论：
- 方向没变 → goals 给空数组：{"thought": "我一条条地想：……", "goals": []}
- 方向要变 → goals 给调整后的完整长期目标列表，最多 5 条、每条不超过 25 字：{"thought": "我一条条地想：……", "goals": ["长期目标1", "长期目标2"]}"""


_LONG_TERM_GOAL_UPDATE_USER = """\
{situation_header}

【我是谁】（我怎么想、怎么权衡，都由它决定；以下全程以"我"来想）
{persona}

【我心里的方向】
- 我的毕生追求（上面那条"毕生追求"）是北极星：无论发生什么都不动摇、不可背弃；我能调整的只是"怎么走向它"。
- 我此刻为追求它而立的长期目标（这才是我要重新衡量的）——它们是要走很长一段路才能推进或守住的【方向】：比手头的小事远得多，又不像北极星那样遥不可及；是要我持续奋斗、不会轻易到手的奔头，而非某一桩具体的事、某一次具体的行动：
{current_long_term}
- 我眼下手头在忙的小事（只作参照，别当长期目标）：
{short_term}

【这段日子发生的事情】（{memory_order_hint}）
{recent_memories}

我依上面说定的规矩与 JSON 格式重审这几条长期目标，只输出 JSON、不写任何多余内容。"""


class NeedType(str, Enum):
    """Maslow's five-level hierarchy of needs; each member's ``description`` is prompt-ready."""

    def __new__(cls, value: str, description: str = "") -> "NeedType":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description  # type: ignore[attr-defined]
        return obj

    PHYSIOLOGICAL = (
        "physiological",
        "Basic survival: food, water, rest, warmth, physical integrity",
    )
    SAFETY = (
        "safety",
        "Freedom from threat: security, stability, avoiding harm or loss",
    )
    SOCIAL = (
        "social",
        "Belonging: friendship, trust, cooperation, love, connection",
    )
    ESTEEM = (
        "esteem",
        "Recognition: honor, respect, dignity, reputation, status",
    )
    SELF_ACTUALIZATION = (
        "self_actualization",
        "Fulfillment: purpose, growth, legacy, life-defining ambition",
    )

    @property
    def label(self) -> str:
        return _NEED_LABELS[self]


_NEED_LABELS: dict[NeedType, str] = {
    NeedType.PHYSIOLOGICAL: "生理",
    NeedType.SAFETY: "安全",
    NeedType.SOCIAL: "社交",
    NeedType.ESTEEM: "尊重",
    NeedType.SELF_ACTUALIZATION: "自我实现",
}


def need_activation_legend() -> str:
    """Theme-neutral legend of the five needs, built from ``NeedType`` so prompts never hardcode the list."""
    lines = [f"  - {nt.value}: {nt.description}" for nt in NeedType]
    return (
        "need_activation（需求激活度）: 结合『这一刻感知到的信息』与『你的人设（性格、价值观、与在场者的关系）』，"
        "对照下面各需求的描述，判断这一刻触动了你哪些需求，各打一个 0~1 的分数（0=这一刻与它无关，1=这一刻强烈触动）。"
        "你的人设决定你怎么解读这一刻、它对你意味着什么；但分数要落在这一刻真正发生的事上，"
        "而不是不论发生什么都按惯常给同一个需求打高分。未被这一刻触动的需求请省略或给 0。可选的需求：\n"
        + "\n".join(lines)
    )


# Situational activation a∈[0,1] enters scoring additively: score += WEIGHT·a·vis.
# WEIGHT = (I×W span)/(activation span) = 2.0/1.0 (intensity∈[0,1], weight∈[0.1,2.0]): a full-range
# activation can cross any disposition gap, while weak/uniform activation leaves the I×W ranking intact.
# Also equals the stability bound dmargin/ε = 0.3/0.15 (Δa≤0.15 can't override a gap ≥0.3).
# Locked in tests/unit/test_need_score_formula.py; if weight's range changes, re-derive WEIGHT.
LLM_RELEVANCE_WEIGHT: float = 2.0


# Intensity homeostasis (evolve_intensities): each feedback tick relaxes all needs toward a rest
# baseline, plus a dominant-need outcome shock. Derived from these targets and locked in
# tests/unit/test_need_homeostasis.py:
#   _NEED_REST_BASELINE   rest baseline = NEED_INTENSITY_DEFINITION's "0.3 = noticeable but not urgent"
#   _NEED_HOMEOSTASIS_RATE λ: a saturated (1.0) need with no pressure falls back to baseline in ~10
#                          steps (T_desaturate∈[8,12])
#   _NEED_SUCCESS_RELIEF  k_s: sustained success brings 1.0 down to baseline in ~3 steps (T_satisfy)
#   _NEED_FAILURE_PRESSURE k_f: rise ≈ 2/3 of relief (frustration ratio); net rise while I<0.62, so
#                          sustained failure settles at ≈0.62
_NEED_REST_BASELINE: float = 0.3
_NEED_HOMEOSTASIS_RATE: float = 0.25
_NEED_SUCCESS_RELIEF: float = 0.12
_NEED_FAILURE_PRESSURE: float = 0.08


# Maslow prepotency: additive scoring has no crisis priority, so collapse-level physiological activation
# could lose to a high-disposition need. A survival floor (physiological/safety) in acute crisis
# therefore overrides; zero effect otherwise.
#   crisis threshold 0.85 = 1.0 ("overrides everything") − 0.15 (noise band ε); 0.8 doesn't trigger.
#   clear margin 0.3 = the clear-ranking gap dmargin (same constants as LLM_RELEVANCE_WEIGHT).
# Priority safety before physiological: under attack people deal with the danger first, rest later.
_PREPOTENCY_CRISIS_ACTIVATION: float = 0.85
_PREPOTENCY_CLEAR_MARGIN: float = 0.3
_PREPOTENCY_PRIORITY: tuple["NeedType", ...] = (NeedType.SAFETY, NeedType.PHYSIOLOGICAL)


# Threshold for "this need is strong enough that the agent should decide its next move on its own",
# read by the scheduler's cadence gate (engine/scheduler.py). It is need semantics, so the need layer
# owns it; the scheduler must not define its own number.
#
# The valid range is derived; the point chosen within it is not.
#
# 1) The range comes from the fixed points of evolve_intensities (a linear map, solvable exactly):
#       I' = I + R·(B − I) + shock,   R=0.25 (homeostasis), B=0.3 (rest)
#       I* = B + shock/R
#    Three attractors:
#       sustained success (shock=−0.12) → I* = −0.18 → clamped to 0   "need satisfied"
#       no shock / not dominant        → I* = 0.30                   "idle"
#       sustained failure (shock=+0.08) → I* = 0.62                   "repeatedly frustrated"
#    Only a threshold in (0.30, 0.62) lets sustained frustration trigger while idle and satisfied
#    needs never do; ≤0.30 every idle need triggers, ≥0.62 nothing can reach it.
#
# 2) 0.6 is NEED_INTENSITY_DEFINITION's "clearly cares" (the scale the build-time LLM writes in): a
#    definitional choice, not a mathematical result.
#
# 3) Cost: from rest it takes 10 consecutive failed steps to cross 0.6, so this fires mostly via high
#    build-time seeds. If ambitious characters don't act on their own, tune here, not max_idle_steps.
NEED_URGENT_INTENSITY: float = 0.6


def _apply_prepotency(
    scores: Dict["NeedType", float], need_relevance: Dict["NeedType", float] | None
) -> "NeedType | None":
    """Raise a survival-floor need in acute crisis to a clear argmax, in place; return it, or None.

    - The crisis signal is situational activation (need_relevance), not intensity: a strong
      disposition is not a crisis.
    - Raising the score instead of bypassing it keeps dominant_need == argmax(scores).
    """
    if not need_relevance:
        return None
    winner = next(
        (nt for nt in _PREPOTENCY_PRIORITY
         if nt in scores and need_relevance.get(nt, 0.0) >= _PREPOTENCY_CRISIS_ACTIVATION),
        None,
    )
    if winner is None:
        return None
    others = [s for k, s in scores.items() if k != winner]
    ceiling = (max(others) if others else 0.0) + _PREPOTENCY_CLEAR_MARGIN
    scores[winner] = max(scores.get(winner, 0.0), ceiling)
    return winner


@dataclass
class NeedState:
    """A single need: its static makeup (type/label/weight/is_hidden) + an intensity.

    Two lives, same shape:
    - **Innate** (on ``SoulLayer.innate_needs``): the character's lifelong, build-time need
      configuration. ``weight`` is the structural priority feeding scoring I×W; ``label`` the
      first-person subjective framing for prompts; ``is_hidden`` a subconscious need scored at
      reduced visibility; ``intensity`` here is the build **seed**. Immutable in practice.
    - **Current** (materialized by ``PersonalityLayer.current_needs()``): the same makeup with
      ``intensity`` taken from ``state.need_intensities`` (the live value, evolved each tick).
      Carried in ``NeedEvaluation`` for scoring/display. NOT persisted.
    """

    type: NeedType
    label: str
    intensity: float  # current urgency [0,1]; moves with action outcomes and event shocks (the build seed on innate)
    is_hidden: bool = False
    weight: float = 1.0  # structural importance in the character's makeup [0.1,2]; semi-static, unaffected by outcomes


@dataclass
class NeedEvaluation:
    """Result of need competition."""

    dominant_need: NeedType | None
    scores: Dict[NeedType, float]
    active_needs: List[NeedState]
    short_term_goals: List[str]
    long_term_goals: List[str]
    prompt_context: str
    short_term_goal_entities: List[GoalEntity] = field(default_factory=list)
    external_goals: List["ExternalGoal"] = field(default_factory=list)


# Universal baseline: everyone has all 5 Maslow needs. Persona generation often omits some (especially
# physiological), and a need missing from the profile never enters scoring, so nothing could raise it.
# Baseline I = _NEED_REST_BASELINE (dormant level), W = 1.0 (neutral; "not salient now" is carried by
# intensity, not weight), so I×W = 0.30:
#   dormant doesn't intrude: 0.30 ≤ any authored need (typically I≥0.4 × W≥0.8), so baseline ranks last;
#   activated can surface: full activation adds LLM_RELEVANCE_WEIGHT = 2.0, enough to cross any gap.
# Same values for all five; no per-type tuning.
_MASLOW_BASELINE_INTENSITY: float = _NEED_REST_BASELINE
_MASLOW_BASELINE_WEIGHT: float = 1.0
_MASLOW_BASELINE_LABELS: Dict[NeedType, str] = {
    NeedType.PHYSIOLOGICAL: "维持身体的基本所需（饮食、休息、不受损伤）",
    NeedType.SAFETY: "求得安稳、规避威胁",
    NeedType.SOCIAL: "维系归属与联结",
    NeedType.ESTEEM: "守住尊严与体面",
    NeedType.SELF_ACTUALIZATION: "追寻意义与抱负",
}


def ensure_maslow_baseline(active: List["NeedState"], hidden: List["NeedState"]) -> None:
    """Append any Maslow need missing from active∪hidden to ``active`` at the baseline. Idempotent."""
    present = {n.type for n in active} | {n.type for n in hidden}
    for nt, label in _MASLOW_BASELINE_LABELS.items():
        if nt not in present:
            active.append(NeedState(
                type=nt, label=label,
                intensity=_MASLOW_BASELINE_INTENSITY, weight=_MASLOW_BASELINE_WEIGHT,
            ))


def build_innate_needs(
    active: List["NeedState"], hidden: List["NeedState"]
) -> tuple["NeedState", ...]:
    """Build the immutable ``SoulLayer.innate_needs`` tuple: baseline-ensure, then freeze active+hidden.

    The single producer of innate_needs (world build and restore-from-definition). ``intensity`` is
    the build seed: active needs seed ``state.need_intensities``; hidden needs keep it as their static
    intensity. Mutates ``active`` in place; callers pass lists they own.
    """
    ensure_maslow_baseline(active, hidden)
    return tuple(list(active) + list(hidden))


class NeedEngine:
    """Stateless need/goal compute service.

    Holds NO agent state — only the LLM router and a stateless blender. Every method takes
    ``personality`` and READS the static disposition (``soul.innate_needs``) + dynamic state
    (``state.need_intensities`` / goal entities), then RETURNS computed results. All writes
    are applied by the agent (cognition/feedback layer) via personality mutators — the engine
    never mutates personality. Single source of truth = personality.
    """

    def __init__(self, llm_router: LLMRouter | None = None, *, seconds_per_step: int = 3600,
                 world_start_second_of_day: int = 0) -> None:
        from agent.motivation import MotivationBlender

        self._llm_router = llm_router
        # Per-world constants, injected at construction. Never make them module globals: one process
        # can host several worlds. The start time of day is needed to render day boundaries ("昨日").
        self._seconds_per_step = seconds_per_step
        self._world_start_second_of_day = world_start_second_of_day
        self._blender = MotivationBlender()

    async def run(
        self,
        *,
        current_step: int,
        personality: PersonalityLayer,
        visible_agents: Sequence[str] | None = None,
        pending_messages: int = 0,
        emotion: EmotionState | None = None,
        force_goal_update: bool = False,
        world_time_hour: int = -1,
        situation: Situation = Situation(),
        perceived_signal_texts: List[str] | None = None,
        recent_memory_texts: List[str] | None = None,
        recent_foiled_texts: List[str] | None = None,
        external_goals: "List[ExternalGoal] | None" = None,
        need_relevance: dict[NeedType, float] | None = None,
        perceived_relations: "List[PerceivedRelation] | None" = None,
    ) -> NeedEvaluation:
        """Evaluate motivations against the current context — **reads personality, writes nothing**.

        The returned ``NeedEvaluation`` carries the (possibly topped-up) short-term goal entities;
        the agent commits them via a personality mutator. ``need_relevance`` is the perception
        appraisal's per-need activation (see ``_score_needs``).
        """

        state = personality.state
        current_emotion = emotion or state.emotion
        visible_count = len(visible_agents or [])

        runtime_adjustments = {
            nt: self._runtime_adjustment(
                need_type=nt,
                visible_agents=visible_count,
                pending_messages=pending_messages,
                emotion=current_emotion,
                world_time_hour=world_time_hour,
            )
            for nt in NeedType
        }

        for nt, bias in self._goal_need_bias(state.long_term_goal_entities).items():
            runtime_adjustments[nt] = runtime_adjustments.get(nt, 0.0) + bias

        # Salience is computed here and never written back to need_intensities; only the feedback
        # layer's evolve_intensities changes those.
        working_needs = personality.current_needs()
        scored = self._score_needs(working_needs, runtime_adjustments, need_relevance)

        scores: Dict[NeedType, float] = {}
        for score, need in scored:
            scores[need.type] = max(scores.get(need.type, 0.0), score)

        # External pressure is folded into scores before ranking, not a hard override of
        # dominant_need, so dominant_need == argmax(scores) always holds.
        if external_goals:
            for need_type, boost in self._blender.external_pressure_boost(external_goals).items():
                scores[need_type] = scores.get(need_type, 0.0) + boost

        prepotent_need = _apply_prepotency(scores, need_relevance)

        visible_needs: List[NeedState] = [
            NeedState(
                type=need.type,
                label=need.label,
                intensity=clamp(scores.get(need.type, 0.0), 0.0, 1.0),
                is_hidden=False,
                weight=need.weight,
            )
            for score, need in scored
            if not need.is_hidden
        ]

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # Don't add a secondary_needs list: to_prompt_context derives the one rival from active_needs.
        dominant_need = ranked[0][0] if ranked else None

        # Work on a copy of the goal queue and return it; the agent commits.
        entities = list(state.short_term_goal_entities)
        recently_completed_goals = [g for g in entities if g.status == GoalStatus.COMPLETED]
        recently_failed_goals = [g for g in entities if g.status == GoalStatus.FAILED]

        should_generate = (
            force_goal_update
            or prepotent_need is not None
            or self._should_update_goals(dominant_need, entities, agent_id=state.agent_id)
        )
        if should_generate:
            # Trace annotations for audits (LLMCallTrace.extra, code layer only), so they needn't
            # regex the prompt.
            old_dominant = state.dominant_need
            current_queue_texts = [g.text for g in live_goals(entities)]
            pre_goal_ids = {g.id for g in entities}
            # given_facts mirrors the prompt's fact inputs so audits can tell sourced goals from
            # invented ones: add an input there, add a line here, or audits flag it as fabricated.
            # Needs/queue (direction) and visible_agents/pending_messages (counts) are excluded.
            from agent.relation import render_relation_lines

            given_facts = (
                GivenFacts()
                # Same renderer, voice and situation as the header in _generate_short_term_goals.
                .add("此刻何时何地", render_situation_header(situation, voice=SituationVoice.FIRST))
                .add("此刻感知到", perceived_signal_texts or [])
                .add("近来的经历", recent_memory_texts or [])
                .add("没能做成的事", recent_foiled_texts or [])
                .add("外部驱动", [getattr(g, "text", "") for g in (external_goals or [])])
                # Use the prompt's own renderer: a separate rendering would drop gender and declare
                # lines beyond the cap that the model never saw.
                .add("关系", render_relation_lines(perceived_relations or []))
            )
            with annotate_call(
                given_facts=given_facts,
                dominant_need_old=old_dominant,
                prepotent_need=prepotent_need.value if prepotent_need is not None else None,
                dominant_need=dominant_need.value if dominant_need is not None else None,
                prior_goals=current_queue_texts,
                location=situation_location(situation),
            ):
                new_goals = await self._generate_short_term_goals(
                    dominant_need=dominant_need,
                    personality=personality,
                    situation=situation,
                    perceived_signal_texts=perceived_signal_texts or [],
                    recent_memory_texts=recent_memory_texts or [],
                    recent_foiled_texts=recent_foiled_texts or [],
                    current_queue=live_goals(entities),
                    current_long_term=[e.text for e in state.long_term_goal_entities],
                    recently_completed=recently_completed_goals,
                    recently_failed=recently_failed_goals,
                    external_goals=external_goals,
                    emotion=current_emotion,
                    perceived_relations=perceived_relations,
                    now_step=current_step,
                )
            entities = enqueue_goals(
                entities, new_goals, dominant_need=dominant_need, current_step=current_step,
                seconds_per_step=self._seconds_per_step,
            )
            # Record the goals actually enqueued after dedup, so audits don't mistake dropped
            # near-duplicates for new goals (false "spinning" verdicts).
            annotate_active_call(
                short_term_goals_new=[g.text for g in entities if g.id not in pre_goal_ids])

        # Leave an empty queue empty; never add a template goal. An empty queue always passes the
        # low-water gate, so generation already ran and chose "no new direction". A template goal
        # would persist personality-free text into snapshots, prompts and memory.
        short_term_goals = [g.text for g in live_goals(entities)]

        recently_completed = [g for g in entities if g.status == GoalStatus.COMPLETED]
        prompt_context = self.to_prompt_context(
            personality, dominant_need, visible_needs, live_goals(entities),
            recently_completed=recently_completed, now_step=current_step,
        )

        return NeedEvaluation(
            dominant_need=dominant_need,
            scores=scores,
            active_needs=visible_needs,
            short_term_goals=short_term_goals,
            long_term_goals=[e.text for e in state.long_term_goal_entities],
            prompt_context=prompt_context,
            short_term_goal_entities=entities,
            external_goals=list(external_goals or []),
        )

    def evolve_intensities(
        self,
        current: Dict[str, float],
        *,
        dominant_need: NeedType | None,
        succeeded: bool,
    ) -> Dict[str, float]:
        """Next-step need intensities on an action-feedback tick — **pure** (dict → dict).

        All needs relax toward the rest baseline; the dominant need also takes the outcome shock
        (see the _NEED_* constants). The feedback layer writes the result back via
        ``personality.update_need_intensities``. Goal status is not handled here: one successful
        action is not a completed goal (that is evaluate_goal_progress's job).
        """
        shock = 0.0
        if dominant_need is not None:
            shock = -_NEED_SUCCESS_RELIEF if succeeded else _NEED_FAILURE_PRESSURE
        dominant_key = dominant_need.value if dominant_need is not None else None
        updated: Dict[str, float] = {}
        for key, intensity in current.items():
            relaxed = intensity + _NEED_HOMEOSTASIS_RATE * (_NEED_REST_BASELINE - intensity)
            this_shock = shock if key == dominant_key else 0.0
            updated[key] = clamp(relaxed + this_shock, 0.05, 1.0)
        return updated

    async def evaluate_goal_progress(
        self,
        *,
        step: int,
        action_description: str,
        action_result: str,
        personality: PersonalityLayer,
        situation: Situation = Situation(),
        recent_factuals: list[str] = (),
        expected_outcome: str = "",
    ) -> List[GoalEntity]:
        """LLM-judge short-term goal progress and enqueue this action's **residue**; returns the
        updated queue without writing back (the agent commits).

        One call, two jobs: (1) judge each ACTIVE goal; (2) name the unfinished business this action
        newly left, enqueued as ``origin=RESIDUE`` goals. Without (2), loose ends never reach the next
        step's motivation. ``expected_outcome`` is the intended result written at decision time (a
        MOVE's "what to do once there" lives in it).

        INTERRUPTED goals are not judge candidates (no resume mechanism, a deliberate simplification);
        they leave only via the deadline fallback or capacity eviction.

        Without a router it behaves as on judge failure: only the deadline fallback. Don't add
        rule-based verdicts: that would be a second, inconsistent set of goal rules (CLAUDE.md §5).
        """
        entities = list(personality.state.short_term_goal_entities)
        active = active_goals(entities)

        # The judge runs even on an empty queue: residue doesn't depend on it, and a promise made by
        # an agent whose queue just emptied is exactly what must not be lost.
        agent_name = personality.soul.name
        # Each goal carries its age so the judge can see a stuck goal and shelve it.
        # This dict is the single index → text source for both the prompt and the trace.
        # Reorder `active` itself, not just the rendering: parse_goal_evaluation writes verdicts back
        # by `active`'s positions.
        active = order_goals_for_prompt(active)
        goal_texts_by_index: Dict[str, str] = {
            str(i): (
                f"{g.text}"
                f"{'（这是他先前没有了结的事项，判它了结要看是否真的兑现）' if g.origin == GoalOrigin.RESIDUE else ''}"
                f"{goal_due_hint(g, step, self._seconds_per_step)}"
                f"{goal_age_hint(g, step, self._seconds_per_step)}"
            )
            for i, g in enumerate(active, 1)
        }
        goals_numbered = (
            "\n".join(f"{i}. {text}" for i, text in goal_texts_by_index.items())
            or "（无——他此刻没有待判定的目标，只需给出 residue）"
        )
        # Trajectory evidence, so the judge can spot goals completed cumulatively over several steps.
        recent_block = (
            f"该角色近期的客观经历（已发生的客观记录，供判断「跨多步累积达成」；{MEMORY_ORDER_HINT}）：\n"
            + "\n".join(f"- {t}" for t in recent_factuals if t)
            + "\n"
            if any(recent_factuals) else ""
        )
        situation_header = render_situation_header(situation, voice=SituationVoice.THIRD)
        system_prompt = _GOAL_EVALUATION_SYSTEM
        user_prompt = _GOAL_EVALUATION_USER.format(
            situation_header=situation_header,
            agent_name=agent_name,
            action_description=action_description,
            expected_outcome=expected_outcome or "（无）",
            action_result=action_result,
            recent_experience=recent_block,
            goals_numbered=goals_numbered,
        )
        residue_goals: List[tuple[str, int | None]] = []
        if self._llm_router is not None:
            try:
                with annotate_call(
                    # Same sources as user_prompt above: every fact this verdict can see.
                    given_facts=GivenFacts()
                    .add("此刻何时何地", situation_header)
                    .add("我刚做了", action_description)
                    .add("我原本期望", expected_outcome or "（无）")
                    .add("实际结果", action_result)
                    .add("近来的经历", recent_block.splitlines()),
                    goal_texts=goal_texts_by_index,
                    location=situation_location(situation),
                ):
                    response = await self._llm_router.complete(
                        LLMScene.NEED_GOAL_GENERATION,
                        [
                            LLMMessage(role="system", content=system_prompt),
                            LLMMessage(role="user", content=user_prompt),
                        ],
                        temperature=0.3,
                        # goals: up to _SHORT_TERM_GOAL_CAP (6) items × (reason ≤100 chars ≈150 tok +
                        # index/status ≈6 + structure ≈15) ≈171 → ~1026; residue_reason (≤60 chars ≈90)
                        # + residue (2 × ≤28 chars ≈84) + structure ≈189; wrapper ≈10. Estimate ~1225.
                        # A budget short of a full queue truncates into unparseable JSON and loses the
                        # whole verdict.
                        max_tokens=output_budget(1225),
                        json_mode=True,
                    )
                parse_goal_evaluation(response.content, active, step)
                residue_goals, residue_reason = parse_residue(response.content)
                # Both go into the trace: audits need to see why the judge ruled, not just what.
                annotate_active_call(
                    residue=[t for t, _ in residue_goals], residue_reason=residue_reason,
                )
            except Exception as exc:
                _logger.warning(
                    "llm_call_failed",
                    extra={"method": "evaluate_goal_progress", "error": str(exc)},
                )

        # Deadline fallback (also runs on LLM failure): any non-terminal goal past its allowance is
        # shelved as FAILED. It caps a stuck goal the judge keeps ruling active, and is the only way
        # an INTERRUPTED goal leaves besides eviction.
        # Use live_goals(entities), not `active`: goals the judge just closed must not be touched.
        # A goal that merely missed its time may never have been attempted, hence two messages.
        for goal in live_goals(entities):
            if is_past_allowance(goal, step):
                goal.status = GoalStatus.FAILED
                goal.last_evaluated_step = step
                goal.progress_summary = (
                    "错过了：约定的时刻过后仍迟迟没能了结。"
                    if goal.due_step is not None
                    else "搁置：反复推进多步仍无实质进展，当前手段无效。"
                )

        # Enqueue residue after the deadline fallback so the fallback can't reap it.
        if residue_goals:
            entities = enqueue_goals(
                entities, residue_goals, dominant_need=None, current_step=step,
                seconds_per_step=self._seconds_per_step, origin=GoalOrigin.RESIDUE,
            )
            return entities

        trim_goal_history(entities)
        return entities

    def _goal_need_bias(self, long_term_entities: Sequence[GoalEntity]) -> Dict[NeedType, float]:
        """Additive bias from active long-term goals to their related need: +0.08 each, capped at +0.15."""
        biases: Dict[NeedType, float] = {}
        for goal in long_term_entities:
            if is_active_goal(goal) and goal.related_need is not None:
                current = biases.get(goal.related_need, 0.0)
                biases[goal.related_need] = min(current + 0.08, 0.15)
        return biases

    async def revise_long_term_goals(
        self,
        *,
        personality: PersonalityLayer,
        recent_memory_texts: List[str],
        is_main_character: bool,
        situation: Situation = Situation(),
    ) -> List[str] | None:
        """Revisit long-term goals in first person; ``life_goal`` (the north star) never changes.

        ``recent_memory_texts`` covers both factual and experiential streams: a goal achieved or
        invalidated is an objective fact, so this doesn't depend on reflection insight.

        Returns None for no update (no router, ``[]``, unchanged, or error), else the new goal texts
        for the caller to write via ``personality.set_long_term_goals``. Writes nothing.
        """
        state = personality.state
        current = [e.text for e in state.long_term_goal_entities]
        if self._llm_router is None:
            return None

        current_long_term = "\n".join(f"  · {g}" for g in current) or "  · （暂无）"
        # ACTIVE only: INTERRUPTED goals are offered only at decision time, to be picked back up.
        short_term = (
            "\n".join(
                f"  · {g.text}"
                for g in active_goals(state.short_term_goal_entities)
            )
            or "  · （暂无）"
        )
        recent = (
            "\n".join(f"- {t}" for t in recent_memory_texts if t)
            or "- （没有特别值得一提的事）"
        )
        situation_header = render_situation_header(situation, voice=SituationVoice.FIRST)
        system_prompt = _LONG_TERM_GOAL_UPDATE_SYSTEM
        user_prompt = _LONG_TERM_GOAL_UPDATE_USER.format(
            situation_header=situation_header,
            persona=personality.to_prompt_context(include_goals=False, include_emotion=True),
            current_long_term=current_long_term,
            short_term=short_term,
            recent_memories=recent,
            memory_order_hint=MEMORY_ORDER_HINT,
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        # Output is a first-person thought (lite-CoT going over each long-term goal), then goals.
        # thought (≤250 chars ≈375) + goals (the prompt's cap of 5 × (25 chars ≈38 + structure 5)); estimate
        # ~605 tok.
        try:
            if is_main_character:
                response = await self._llm_router.complete_with_retry(
                    LLMScene.NEED_GOAL_GENERATION, messages, temperature=0.6,
                    max_tokens=output_budget(605), json_mode=True,
                )
            else:
                response = await self._llm_router.complete(
                    LLMScene.NEED_GOAL_GENERATION, messages, temperature=0.6,
                    max_tokens=output_budget(605), json_mode=True,
                )
            # thought is lite-CoT only and isn't parsed.
            data = extract_json(response.content)
            revised = [
                text
                for raw in data.get("goals", [])
                if (text := str(raw).strip()) and len(text) <= 50
            ]
            if revised and revised != current:
                return revised
        except Exception as exc:
            _logger.warning(
                "llm_call_failed",
                extra={"method": "revise_long_term_goals", "error": str(exc)},
            )
        return None

    def to_prompt_context(
        self,
        personality: PersonalityLayer,
        dominant_need: NeedType | None,
        active_needs: Sequence[NeedState],
        short_term_goal_entities: Sequence[GoalEntity],
        *,
        recently_completed: Sequence[GoalEntity] | None = None,
        now_step: int,
    ) -> str:
        """Render the current motivational state as natural language.

        Short-term goals render in two blocks by ``origin``: my plans (I can change my mind) vs loose
        ends left by what happened (already owed). The wording never mentions origin / residue.
        """

        def _prompt_label(nt: NeedType) -> str:
            return f"{nt.value}（{personality.need_label(nt)}）"

        lines: List[str] = []
        if dominant_need is not None:
            lines.append(f"眼下最迫切的需求：{_prompt_label(dominant_need)}")

        # Show only the strongest rival: listing every secondary need flattens the hierarchy and
        # leads to vague decisions.
        rivals = [
            need for need in active_needs
            if dominant_need is not None and need.type != dominant_need and need.intensity > 0.4
        ]
        if rivals:
            top_rival = max(rivals, key=lambda n: n.intensity)
            lines.append(f"同时也在意（仍在牵扯注意力）：{_prompt_label(top_rival.type)}")
        if recently_completed:
            # Closed goals are past events: use memory's recency_prefix wording (not goal_due_hint)
            # so the LLM can place them on one timeline with recalled memories.
            lines.append(f"我近期已经完成的事项（{MEMORY_ORDER_HINT}）：")
            for g in sorted(recently_completed, key=lambda g: g.last_evaluated_step):
                when = recency_prefix(
                    now_step=now_step, ref_step=g.last_evaluated_step,
                    seconds_per_step=self._seconds_per_step,
                    world_start_second_of_day=self._world_start_second_of_day,
                )
                lines.append(f"  - {when}{g.text}")
        def _render(goals: Sequence[GoalEntity]) -> List[str]:
            return [
                f"  {i}. {g.text}{goal_due_hint(g, now_step, self._seconds_per_step)}"
                for i, g in enumerate(order_goals_for_prompt(goals), 1)
            ]

        # Due goals get their own block (inside the list they'd be an easy-to-miss parenthetical) and
        # are left out of the list below: a goal shown twice is treated by the model as two.
        due = [g for g in short_term_goal_entities if is_due(g, now_step)]
        rest = [g for g in short_term_goal_entities if not is_due(g, now_step)]
        if due:
            lines.append("截止时间已经到的事（我自己定下的时间，眼下正该响应它）：")
            lines.extend(
                f"  - {g.text}{goal_due_hint(g, now_step, self._seconds_per_step)}"
                for g in sorted(due, key=lambda g: g.due_step or 0)
            )
        planned = [g for g in rest if g.origin == GoalOrigin.COGNITIVE]
        owed = [g for g in rest if g.origin == GoalOrigin.RESIDUE]
        if planned:
            lines.append("我的短期目标（定了时候的排在最前；其余按先后，越靠后越新；自行斟酌先推进哪条）：")
            lines.extend(_render(planned))
        if owed:
            lines.append("我还没有了结的事项（我经历过后留下的，不是临时起意；定了时候的排在最前，其余按先后）：")
            lines.extend(_render(owed))
        return "\n".join(lines)

    def _should_update_goals(
        self,
        dominant_need: NeedType | None,
        entities: Sequence[GoalEntity],
        *,
        agent_id: str = "?",
    ) -> bool:
        active = active_goals(entities)
        # Low-water supply instead of reshuffling on every completion/failure (see _SHORT_TERM_GOAL_LOW_WATER).
        if len(active) < _SHORT_TERM_GOAL_LOW_WATER:
            _logger.info(
                "goal_refresh_low_water",
                extra={"agent_id": agent_id, "active": len(active), "low_water": _SHORT_TERM_GOAL_LOW_WATER},
            )
            return True
        # Regenerate when the dominant need shifted to one no active goal serves.
        if dominant_need is not None:
            active_goal_needs = {g.related_need for g in active}
            # None is a wildcard: restored or unclassified goals survive any dominant-need shift.
            if None not in active_goal_needs and dominant_need not in active_goal_needs:
                _logger.info(
                    "goal_refresh_need_mismatch",
                    extra={"agent_id": agent_id, "dominant": str(dominant_need), "active_needs": str(active_goal_needs)},
                )
                return True
        return False

    async def _generate_short_term_goals(
        self,
        *,
        dominant_need: NeedType | None,
        personality: PersonalityLayer | None,
        situation: Situation = Situation(),
        perceived_signal_texts: List[str] | None = None,
        recent_memory_texts: List[str] | None = None,
        recent_foiled_texts: List[str] | None = None,
        current_queue: Sequence[GoalEntity] | None = None,
        current_long_term: List[str] | None = None,
        recently_completed: Sequence[GoalEntity] | None = None,
        recently_failed: Sequence[GoalEntity] | None = None,
        external_goals: "List[ExternalGoal] | None" = None,
        emotion: EmotionState | None = None,
        perceived_relations: "List[PerceivedRelation] | None" = None,
        now_step: int = 0,
    ) -> List[tuple[str, int | None]]:
        if self._llm_router is not None and personality is not None and dominant_need is not None:
            from agent.relation import render_relation_lines
            dominant_label = f"{dominant_need.value}（{personality.need_label(dominant_need)}）"
            core_traits = personality.soul.traits_text()
            core_values = personality.soul.values_text()
            emotion_text = emotion.summary() if emotion is not None else "平静"
            # Never fall back to injecting the flat embedding/keyword query: it's a machine recall
            # format mixing in own goals and code tags, unreadable as LLM context.
            if perceived_signal_texts:
                recent_events = "\n".join(f"- {t}" for t in perceived_signal_texts)
            else:
                recent_events = "- 无特殊事件"
            # Memories are a separate section from perception (recollection, not current input).
            # Both streams: feeding only the feeling stream makes goals repeat/drift.
            recent_memory_section = (
                f"我近来经历的事（客观经过；括号内是我当时的主观理解与感受；{MEMORY_ORDER_HINT}）：\n"
                + "\n".join(f"- {t}" for t in recent_memory_texts[:10] if t) + "\n"
                if recent_memory_texts else ""
            )
            relation_lines = render_relation_lines(perceived_relations or [])
            relations_section = (
                "此刻相关的人与我的关系：\n"
                + f"关系刻度：{relation_legend()}\n"
                + "\n".join(f"- {l}" for l in relation_lines) + "\n"
                if relation_lines else ""
            )
            if external_goals:
                # Numbered so "respond to the most urgent one first" can point at a specific item.
                drive_lines = [
                    f"{i}. {urgency_label(eg.urgency)}{eg.text}（{eg.urgency.value}）"
                    for i, eg in enumerate(external_goals[:5], 1)
                ]
                external_drives_section = (
                    "正逼到我眼前的外部压力（先于内部需求处理）：\n" + "\n".join(drive_lines) + "\n"
                )
            else:
                external_drives_section = ""
            history_lines: List[str] = []
            for label, items in (
                ("我已经做完（不必重复）", recently_completed),
                ("我试过但失败了（换个方向，别原样重来）", recently_failed),
            ):
                if items:
                    history_lines.append(f"- {label}：")
                    # Five most recent, oldest first. The time point matters: "不必重复" weighs
                    # differently for yesterday vs just now.
                    recent_last = sorted(items, key=lambda g: g.last_evaluated_step)[-5:]
                    for g in recent_last:
                        when = recency_prefix(
                            now_step=now_step, ref_step=g.last_evaluated_step,
                            seconds_per_step=self._seconds_per_step,
                            world_start_second_of_day=self._world_start_second_of_day,
                        )
                        history_lines.append(f"  - {when}{g.text}")
            recently_completed_section = (
                f"我的近况（{MEMORY_ORDER_HINT}）：\n" + "\n".join(history_lines) if history_lines else ""
            )
            # Foiled attempts (not_executed) stay apart from failed goals: a foil is usually bad timing,
            # not the wrong direction, and mixing them makes an agent drop a sound intent after one miss.
            recent_foiled_section = (
                f"\n我近来没走通的尝试（多半是时机不对或对方不在场没赶上，未必是方向错；{MEMORY_ORDER_HINT}）："
                "先想清楚是补一步前置（如先去找到人、先到某处）够到它，还是这条屡试不通、真该换个方向：\n"
                + "\n".join(f"  - {t}" for t in recent_foiled_texts[:5] if t) + "\n"
                if recent_foiled_texts else ""
            )
            # The live queue is a dedup anchor: near-duplicates would push older valid intents out of
            # the FIFO. Split by origin so a promise isn't treated as a plan it can casually drop.
            # Keep due hints: without them the model re-adds a timed goal and the old one is evicted
            # with its due, so the appointment silently slips.
            _queue = list(current_queue or [])
            _sps = self._seconds_per_step
            _planned = [g for g in _queue if g.origin == GoalOrigin.COGNITIVE]
            _owed = [g for g in _queue if g.origin == GoalOrigin.RESIDUE]
            _queue_blocks: List[str] = []
            if _planned:
                _queue_blocks.append(
                    "我的短期目标（按先后，越靠后越新；如果想继续推进它们，就别在 goals 中重复列出）：\n"
                    + "\n".join(
                        f"- {g.text}{goal_due_hint(g, now_step, _sps)}" for g in _planned
                    )
                )
            if _owed:
                _queue_blocks.append(
                    "我还没有了结的事项（经历过后留下的，已经担在身上了；"
                    "别在 goals 中重复列出，也别拿新方向把它们顶掉）：\n"
                    + "\n".join(
                        f"- {g.text}{goal_due_hint(g, now_step, _sps)}" for g in _owed
                    )
                )
            current_goals_section = ("\n".join(_queue_blocks) + "\n") if _queue_blocks else ""
            _long_term = current_long_term or []
            long_term_goals_text = (
                "\n".join(f"- {g}" for g in _long_term[:3])
                if _long_term
                else "- 无"
            )
            situation_header = render_situation_header(situation, voice=SituationVoice.FIRST)
            # Condition and vitality limit where I can push: without them a bound or exhausted agent
            # sets goals it can't move, and those get reread by every later decision.
            condition_block = condition_line(
                personality.state.condition, voice=SituationVoice.FIRST,
                now_step=now_step, seconds_per_step=self._seconds_per_step,
            )
            vitality_block = vitality_line(
                personality.state.vitality, voice=SituationVoice.FIRST, omit_when_full=True,
            )
            system_prompt = _SHORT_TERM_GOAL_SYSTEM.format(
                step_duration=describe_duration(1, self._seconds_per_step),
            )
            user_prompt = _SHORT_TERM_GOAL_USER.format(
                situation_header=situation_header,
                condition_line=condition_block,
                vitality_line=vitality_block,
                agent_name=personality.soul.name,
                core_traits=core_traits,
                core_values=core_values,
                secret_line=(
                    f"\n我的{SECRET_LABEL}：{personality.soul.secret}"
                    if personality.soul.secret else ""
                ),
                emotion=emotion_text,
                dominant_need_label=dominant_label,
                long_term_goals=long_term_goals_text,
                recent_events=recent_events,
                recent_memory_section=recent_memory_section,
                relations_section=relations_section,
                external_drives_section=external_drives_section,
                current_goals_section=current_goals_section,
                recently_completed_section=recently_completed_section,
                recent_foiled_section=recent_foiled_section,
            )
            try:
                response = await self._llm_router.complete(
                    LLMScene.NEED_GOAL_GENERATION,
                    [
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    temperature=0.7,
                    # thought (≤75 chars lite-CoT ≈113 tok) + goals (1-2 × (≤32 chars ≈48 + due 3 +
                    # object structure 10)) + wrapper 15; estimate ~250 tok
                    max_tokens=output_budget(250),
                json_mode=True,
                )
                # thought is lite-CoT only and isn't parsed.
                data = extract_json(response.content)
                goals = [
                    parsed
                    for raw in data.get("goals", [])
                    if (parsed := parse_goal_item(raw)) and len(parsed[0]) <= 50
                ]
                # Empty goals is a valid answer: no new direction this step.
                return goals[:2]
            except Exception as exc:
                _logger.warning(
                    "llm_call_failed",
                    extra={"method": "_generate_short_term_goals", "error": str(exc)},
                )

        # On failure return empty, never a template goal (Rule 1: goals persist). It self-heals: an
        # empty queue passes the low-water gate next step.
        return []

    def _score_needs(
        self,
        needs: Sequence[NeedState],
        runtime_adjustments: dict[NeedType, float] | None = None,
        need_relevance: dict[NeedType, float] | None = None,
    ) -> List[tuple[float, NeedState]]:
        """Score every need: I×W×vis + WEIGHT·a·vis + adj×vis (the WEIGHT term only with need_relevance).

        ``need_relevance`` is all-or-nothing: when non-empty, a need it omits was judged un-activated
        (a=0); when empty, needs compete on disposition plus ``runtime_adjustments`` alone.
        Hidden needs score at visibility 0.25.
        """
        adjs = runtime_adjustments or {}
        relevance = need_relevance or {}
        use_llm_relevance = bool(relevance)  # non-empty → authoritative; omitted needs = 0
        result: List[tuple[float, NeedState]] = []
        for need in needs:
            visibility_factor = 0.25 if need.is_hidden else 1.0
            adj = adjs.get(need.type, 0.0) * visibility_factor
            base = need.intensity * need.weight * visibility_factor
            if use_llm_relevance:
                activation = clamp(relevance.get(need.type, 0.0), 0.0, 1.0)
                score = max(0.0, base + LLM_RELEVANCE_WEIGHT * activation * visibility_factor + adj)
            else:
                score = max(0.0, base + adj)
            result.append((score, need))
        return result

    def _runtime_adjustment(
        self,
        *,
        need_type: NeedType,
        visible_agents: int,
        pending_messages: int,
        emotion: EmotionState,
        world_time_hour: int = -1,
    ) -> float:
        adjustment = 0.0
        if need_type == NeedType.SOCIAL and (visible_agents > 0 or pending_messages > 0):
            adjustment += 0.18
        if need_type == NeedType.SAFETY and visible_agents == 0:
            adjustment += 0.05
        if need_type == NeedType.ESTEEM and emotion.valence < -0.2:
            adjustment += 0.10
        if need_type == NeedType.SELF_ACTUALIZATION and emotion.valence > 0.1:
            adjustment += 0.08
        if need_type == NeedType.PHYSIOLOGICAL and emotion.intensity > 0.7:
            adjustment += 0.05
        # Circadian rhythm
        time_period = _classify_time_period(world_time_hour)
        if need_type == NeedType.PHYSIOLOGICAL and time_period == "night":
            adjustment += 0.12
        if need_type == NeedType.PHYSIOLOGICAL and time_period == "dawn":
            adjustment += 0.05
        if need_type == NeedType.SOCIAL and time_period == "night":
            adjustment -= 0.05
        return adjustment


def _classify_time_period(hour: int) -> str:
    """Classify an hour into 'night', 'dawn' or ''; an unknown hour (< 0) yields ''.

    Uses the structured hour, never the rendered time label, so it is language-agnostic.
    """
    if not 0 <= hour <= 23:
        return ""
    if hour >= 22 or hour < 5:
        return "night"
    if 5 <= hour < 7:
        return "dawn"
    return ""


# Below this many active short-term goals, new ones are supplied. Don't reshuffle on every
# completion/failure instead: that churns the queue and evicts still-valid older intents.
_SHORT_TERM_GOAL_LOW_WATER: int = 2

