"""One uniform LLM-as-judge for every audit scope (deduction, rationale-first).

Every scope shares ``judge_scope``: input = the rendered data + that scope's metrics; output =
rationale first, then a score matrix of unit (stage / step) × metric (metric-only when
unit="none"). The LLM doesn't compute the total: ``weighted_total`` weights and normalizes it in
code (no arithmetic hallucination). The judge is a functional, out-of-character third party,
deduction-based (full marks by default, only citable problems penalized, nothing deducted for
deviating from a reference answer), and doesn't raise on failure (Rule 1).

Scopes differ only in a context-rendering function (laying the reconstructed view out as text).
Ids are fine here: this is pure code-layer offline analysis.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from agent.decision import _ACTION_SPACE
from agent.goals import GOAL_TIME_MARK_DEFINITION
from agent.need import need_activation_legend
from agent.personality import EmotionType
from agent.relation import relation_legend
from core.duration import describe_duration
from core.interfaces.urgency import Urgency
from core.prompts import (
    EMOTION_INTENSITY_DEFINITION,
    EXTERNAL_DRIVE_TYPE_DEFINITION,
    EMOTION_VALENCE_DEFINITION,
    NEED_INTENSITY_DEFINITION,
    NEED_TYPE_DEFINITION,
    NEED_WEIGHT_DEFINITION,
    PHYSICAL_DEEDS,
    urgency_label,
)
from core.logging import get_logger

from tuning.audit_metrics import (
    AuditScope,
    clamp,
    output_schema,
    render_metric_block,
    weighted_total,
)
from tuning.audit_reconstruct import (
    _DISCARDED_VERDICTS,
    _TARGET_SLOTS,
    AgentStepView,
    StageCall,
    WorldAuditView,
    _menu_lookup,
)

logger = get_logger(__name__)

_SYSTEM = (
    "你是叙事仿真的**质量评审**(第三方裁定,非扮演角色)。你收到某个已跑完世界的一段 trace 的结构化"
    "摘要,请按给定指标逐项审查。\n"
    "评分为**扣分制、满分 100**:每个维度默认 100 分,只在命中其『判分依据』时扣分——\n"
    "· 轻微问题：每条扣 5–10 分\n"
    "· 明显问题：每条扣 15–25 分\n"
    "· 严重问题（尤其『基础』维度：矛盾 / 臆造 / 失忆 / 断链 / 格式失败）：每条扣 30–50 分\n"
    "多条命中**累计扣分**,最低 0;无命中即 100。**不要因为输出与你预想的『标准答案』不同就扣分**。\n"
    "先写 rationale(点出被扣分处与扣了多少,**≤500 字**),再给分数。只输出 JSON。"
)


def _max_tokens(scope: AuditScope, n_units: int) -> int:
    """Token budget for this verdict. rationale ≤500 chars → 750 tok (1.5 tok/char); each unit ×
    metric cell is a 0-100 integer, ~8 tok, plus JSON structure, so 10 per cell.

    500 chars is needed, not slack: the rationale has to point out each deduction, more of them
    as worlds grow.
    """
    cells = max(1, n_units) * len(scope.metrics)
    return output_budget(750 + cells * 10)


def _normalize(data: Any, scope: AuditScope) -> dict[str, Any]:
    """Normalize rationale + score matrix (clamp to 0-100, keep known metrics only), then compute the weighted total.

    Treat missing keys and nulls differently: the prompt asks for every dimension to appear (null when
    it can't be judged), so a dimension missing entirely means the judge dropped the field, not that it
    abstained. Both show up as no score, but a missing key is a defect to fix in the prompt while
    abstaining is by design; lumping them together lets the defect persist. This only covers the
    current call: an old file on disk missing a dimension usually means the scoring version then
    didn't include it, which ``scope_digest``'s fingerprint identifies; it's not a missing key.
    """
    if not isinstance(data, dict):
        return {"rationale": "judge 输出非对象", "scores": {}, "total": None}
    raw = data.get("scores")
    ids = {m.id for m in scope.metrics}
    dropped: set[str] = set()

    def _cell(cell: Any) -> dict[str, int]:
        cell = cell if isinstance(cell, dict) else {}
        dropped.update(ids - set(cell))
        return {mid: s for mid in ids if (s := clamp(cell.get(mid))) is not None}

    if scope.unit == "none":
        scores: dict[str, Any] = _cell(raw)
    else:
        scores = {}
        for unit, cell in (raw or {}).items():
            if isinstance(cell, dict) and (got := _cell(cell)):
                scores[str(unit)] = got
    if dropped:
        logger.warning("audit_judge_dropped_metrics",
                       extra={"scope": scope.key, "dropped": sorted(dropped)})
    return {
        "rationale": str(data.get("rationale", "")),
        "scores": scores,
        "dropped": sorted(dropped),
        "total": weighted_total(scores, scope.metrics, scope.unit),
    }


def _empty(note: str) -> dict[str, Any]:
    return {"rationale": note, "scores": {}, "dropped": [], "total": None}


async def judge_scope(
    router: LLMRouter,
    scope: AuditScope,
    *,
    header: str,
    context_block: str,
    n_units: int = 1,
    det_findings: Any = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Run one scope's judge call. Never raises.

    Returns {rationale, scores, total, _prompt, _response} — ``_prompt`` / ``_response`` carry
    the exact LLM call (messages + raw response) for web display; the orchestrator strips them
    out of the scored entry and files them under audit_calls.json.
    """
    det_block = ""
    if det_findings:
        det_block = ("\n═══ 确定性预检（供参考，非终裁）═══\n"
                     f"{json.dumps(det_findings, ensure_ascii=False)}\n")
    unit_hint = {
        "stage": "对**每一拍**分别",
        "step": "对**每个步骤**分别",
        "none": "",
    }[scope.unit]
    # Sectioned layout (CLAUDE.md prompt principles): task + key constraints first, bulk data in the middle, output format last;
    # sections separated by ═══, discrete items as lists, no undifferentiated blob.
    prompt = (
        "═══ 任务 ═══\n"
        f"{header}\n"
        f"请{unit_hint}按下方【评分维度】给每个维度打 0-100 分（满分 100，扣分制：默认 100，"
        "命中『判分依据』才按严重程度扣分，轻微 −5~10／明显 −15~25／严重 −30~50，多条累计）。\n"
        "**若某维度在本片段无从判断（如这是中途切片而非完整结局、或窗口内无重大变故可考察成长/弧线），"
        "给 null，不要硬扣分。**\n"
        "**每个维度都必须在输出里出现**：有判断给分数，无从判断给 null —— 不要省略任何一个键。\n"
        "\n═══ 评分对象 ═══\n"
        f"{context_block}\n"
        f"{det_block}"
        "\n═══ 评分维度 ═══\n"
        "（『基础』= 叙事底线：矛盾 / 臆造 / 断链 / 失忆，命中扣得更重；"
        "『提升』= 底线之上，只罚『全程平直』『俗套到底』这类更极端的失败——"
        "**不评好到什么程度，没出彩不是问题**）\n"
        f"{render_metric_block(scope.metrics)}\n"
        "\n═══ 输出格式 ═══\n"
        "严格输出以下 JSON，不要多余内容（rationale 在前、分数在后；总分由系统计算，不必给）：\n"
        f"{output_schema(scope.metrics, scope.unit)}"
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    try:
        response = await router.complete(
            judge_scene,
            [LLMMessage(role=m["role"], content=m["content"]) for m in messages],
            temperature=0.2, max_tokens=_max_tokens(scope, n_units),
        )
    except Exception as exc:  # noqa: BLE001 — a judge failure must not abort the audit
        logger.warning("audit_judge_failed", extra={"scope": scope.key, "error": str(exc)})
        out = _empty(f"judge 调用失败: {exc}")
        return {**out, "_prompt": messages, "_response": f"[调用失败] {exc}"}
    try:
        out = _normalize(extract_json(response.content), scope)
    except (json.JSONDecodeError, ValueError):
        out = _empty("judge 输出非合法 JSON")
    # Attach this LLM call (prompt + raw response) so the web can show each audit call; the orchestrator strips it out of the score files.
    return {**out, "_prompt": messages, "_response": response.content}


# ---------------------------------------------------------------------------
# Context-block builders (one per scope)
# ---------------------------------------------------------------------------

def _fmt_output(out: Any) -> str:
    if isinstance(out, str):
        return out
    return json.dumps(out, ensure_ascii=False)


# Uniform visual separators: heavy rule for steps, light rule for stages/characters, so the LLM sees boundaries at a glance (CLAUDE.md sectioning).
def _step_sep(step: Any, world_time: str) -> str:
    return f"\n━━━━━━━━━━ 第 {step} 步（{world_time}）━━━━━━━━━━"


def _sub_sep(label: str) -> str:
    return f"\n──── {label} ────"


def _indent(text: Any, pad: str = "    ") -> str:
    body = str(text).strip()
    return "\n".join(pad + ln for ln in body.splitlines()) if body else f"{pad}（无）"


def _join(xs: Any) -> str:
    return "、".join(str(x) for x in xs) if isinstance(xs, (list, tuple)) else str(xs or "")


def _bullets(xs: Any) -> str:
    """Several sentence-level discrete items → a list (- prefix) so the LLM can check them one by one (CLAUDE.md §4).

    An item can itself span lines (an errand-runner's report carries location/present/things lines).
    Continuation lines must be indented too: prefixing only the first line drops the rest back to
    column one, where they read like separate signals and the judge can no longer check item by item.
    """
    items = xs if isinstance(xs, (list, tuple)) else [xs]
    out = []
    for x in items:
        head, *rest = str(x).splitlines() or [""]
        out.append(f"    - {head}")
        out.extend(f"      {ln.strip()}" for ln in rest if ln.strip())
    return "\n".join(out)


def _goal_lines(label: str, goals: Any) -> list[str]:
    """Short-term goal rendering: 0 → empty; 1 → inline; ≥2 → list (§4: the judge compares goals one by one for spinning in place or regressing)."""
    gs = [str(g) for g in goals if str(g).strip()] if isinstance(goals, (list, tuple)) else (
        [str(goals)] if goals else [])
    if not gs:
        return []
    if len(gs) == 1:
        return [f"  {label}：{gs[0]}"]
    return [f"  {label}：", _bullets(gs)]


_GOAL_STATUS_LABELS = {"completed": "已完成", "active": "推进中", "interrupted": "被打断", "failed": "已失败/搁置"}


def _goal_progress_lines(goals: Any, tag: str = "") -> list[str]:
    """Goal progress rendering: #index → goal text, English status → Chinese. 0 → empty; 1 → inline; ≥2 → list.
    Only the status is given; the goal-progress judge's reason is not injected. Status plus the elapsed
    time suffix on the goal text is enough to judge progress vs idling; adding another LLM's reasoning
    is redundant and would anchor the audit judge to its conclusion (judge independence)."""
    items: list[str] = []
    for g in goals if isinstance(goals, (list, tuple)) else []:
        if not isinstance(g, dict):
            continue
        st = _GOAL_STATUS_LABELS.get(g.get("status"), str(g.get("status") or "?"))
        text = g.get("text") or f"目标#{g.get('index')}"
        items.append(f"「{text}」→{st}")
    if not items:
        return []
    if len(items) == 1:
        return [f"  目标进展{tag}：{items[0]}"]
    return [f"  目标进展{tag}：", _bullets(items)]


def _action_input_line(r: dict, name: str, said: str) -> str:
    """What was handed to the executor: which action, which words, who was present.

    Same treatment as every other stage: each stage lays out its own input, and these three are the
    action stage's input. They're laid out even when identical to this step's decision: what the
    executor receives isn't necessarily what the decision wrote (a conscripted agent gets the
    initiator's framing), and steps without a decision (a multi-step action finishing) depend on it
    to say whose action the result belongs to. When the initiator isn't this agent, name the
    initiator: the description is in the initiator's first person and, unnamed, reads as "he did it
    himself". The exact words (a letter sent / a message carried by an errand-runner) also go here,
    not in the decision output: they're the content handed over to execute.
    """
    by, did = r.get("initiator"), r["did"]
    who = r.get("participants") or []
    parts = [did if not by or by == name else f"这一动由 {by} 发起：{did}"]
    if said and said not in did:
        # TALK folds the words into action_description (see decision._bind_talk); listing them again would repeat them.
        parts.append(f"原话:{said}")
    if len(who) > 1:
        parts.append(f"参与:{_join(who)}")
    return "  行动｜输入：" + "｜".join(parts)


def _action_output_lines(ar: dict, actor: str, target: Any, tag: str = "") -> list[str]:
    """Action output rendering: lay out field by field, never dump raw JSON. Key names (fact / achieved
    / detected) are code-layer identity; the judge should read "own memory", "was it detected".

    outcome (third-person, observable) and fact (first-person memory) are two channels, each on its
    own line and never merged into one sentence (see the CLAUDE.md narrative layer boundary). In sams,
    TALK's turn-by-turn dialogue shows only the turn count and the two parties; the full dialogue
    belongs at sass's single-step granularity (a deliberate decision).
    """
    if not isinstance(ar, dict):
        return [f"  行动｜输出(结果){tag}：{_fmt_output(ar)}"]
    succ = ar.get("success") if "success" in ar else ar.get("achieved")
    mark = "[成功] " if succ is True else ("[失败] " if succ is False else "")
    out = [f"  行动｜输出(结果){tag}：{mark}{ar.get('outcome') or ''}".rstrip()]
    if ar.get("fact"):
        out.append(f"    记忆(己方)：{ar['fact']}")  # actor's first person, with the other party's view removed
    dlg = ar.get("dialogue")
    if isinstance(dlg, list) and dlg:
        out.append(f"    对话：{len(dlg)} 轮（{actor}↔{str(target) if target else '对方'}）")
    if (det := ar.get("detected")) is not None:
        out.append(f"    {'被人察觉了' if det else '未被察觉'}")
    return out


def _needs(ns: Any) -> str:
    """Render as a list (one need per line, - prefix): discrete items are easier for the LLM to check one by one than a single packed sentence (CLAUDE.md §4)."""
    out = []
    for n in ns or []:
        if isinstance(n, dict):
            lab = f"·{n['label']}" if n.get("label") else ""
            out.append(f"    - {n.get('type')}{lab}（强度{n.get('intensity')}，权重{n.get('weight')}）")
    return "\n".join(out)


def _emotion_legend() -> str:
    """Emotion field definitions (authoritative, from core/prompts): kind + intensity + valence, the yardstick for judging whether an emotion is reasonable or abrupt."""
    return (f"primary（情绪种类）∈ {{{EmotionType.prompt_list()}}}；"
            f"{EMOTION_INTENSITY_DEFINITION}；{EMOTION_VALENCE_DEFINITION}")


def _action_type_legend() -> list[str]:
    """Action type semantics (single source of truth agent/decision.py:_ACTION_SPACE): sams decisions
    keep only the [TYPE] tag and lose the 【我能做的】 menu from sass's decision prompt. The judge needs
    to know what each type actually does (sync/async, changes state / only observes) to judge
    action_efficacy (idling vs really changing the world) and whether the action fits the decision."""
    return [f"{c.action_type.value.upper()}：{c.description}" for c in _ACTION_SPACE]


def _need_legend() -> str:
    """Need scale + the five category definitions + activation, all taken from authoritative constants (core/prompts, agent/need).

    The yardstick for judging whether need types are canonical, whether intensities / weights are out
    of range or unreasonable, and whether they fit the situation. A hand copy would drift: the anchors
    (0.3 = noticeable but not urgent / 0.8 = dominates thinking) are the only basis for judging whether
    a value is reasonable, and losing them makes the dimension unjudgeable. Activation and intensity are
    different quantities; reading one on the other's scale can't tell right from wrong.
    """
    activation = need_activation_legend().split("可选的需求：")[0].strip()
    return (f"{NEED_TYPE_DEFINITION}\n  {NEED_INTENSITY_DEFINITION}\n  {NEED_WEIGHT_DEFINITION}\n"
            f"  {activation}")


def _time_legend(sps: Any) -> str:
    """Two time scales: the world time in the bracket at the start of each step, and the due markers on goals.

    Judging whether an agreed time was honored needs both. Unless it's said that the marker is stamped
    by the engine, the judge will treat "（约定时刻已过约8小时）" as a line of narrative rather than a fact
    to score against. The wording comes from agent/goals' authoritative definition, not copied here.
    """
    per = f"一步是{describe_duration(1, sps)}；" if sps else ""
    return (f"每步开头括号里是该步的世界时点（按这个世界的历法）；{per}"
            f"{GOAL_TIME_MARK_DEFINITION}")


def _pressure_legend() -> str:
    """Two scales for external pressure: category + urgency. A judge reading "外部压力（threat，紧迫度【紧急】）"
    without knowing what threat means or which of the four levels 【紧急】 is can't judge whether it
    should override the dominant need.

    English enum values and Chinese level names are given side by side: rendered text uses the Chinese
    name (【紧急】) while raw output uses the enum value (``"urgency": "high"``). Give only one and the
    judge has to guess the other."""
    tiers = " | ".join(
        f"{u.value}{urgency_label(u)}={desc}" for u, desc in (
            (Urgency.LOW, "可留意（背景信号）"),
            (Urgency.NORMAL, "值得响应（需排入计划）"),
            (Urgency.HIGH, "应当优先（其它事都先放）"),
            (Urgency.CRITICAL, "必须立即响应（生死攸关 / 不可错过）"),
        )
    )
    return f"{EXTERNAL_DRIVE_TYPE_DEFINITION}；紧迫度 4 档：{tiers}"


# Scales are defined in one place and requested by name from each renderer. Rendering a scale
# without its definition wastes it: the judge sees the value and can still score, but the score has
# no basis. Conversely, a definition for a scale that isn't rendered is noise, so pick by name
# rather than dumping the whole block. All definitions come from authoritative constants
# (core/prompts, agent/*), not hand copies.
_SCALES: dict[str, Any] = {
    "关系": lambda sps: relation_legend(),
    "情绪": lambda sps: _emotion_legend(),
    # Needs keep only the bare type name (safety/social…): each type's meaning + Maslow level is needed to judge whether values are reasonable.
    "需求": lambda sps: _need_legend(),
    "外部压力": lambda sps: _pressure_legend(),
    "时间": lambda sps: _time_legend(sps),
    # Goal statuses in raw output are English enums (sams/mass already translate them; sass lays out the raw text, so it needs the mapping).
    "目标状态": lambda sps: "、".join(f"{k}={v}" for k, v in _GOAL_STATUS_LABELS.items()),
    # The relation field in adjudication output: the actor's sentiment toward the others involved (output schema of engine/executors/*).
    "关系走向": lambda sps: "裁决输出的 relation 字段 —— positive=观感转好、negative=观感转坏、"
                            "neutral=没有变化",
}


def _legend_lines(*names: str, sps: Any = None, extra: list[str] | None = None) -> list[str]:
    """The "field definitions" block: name the scales needed (legends go first, CLAUDE.md §3)."""
    lines = [_sub_sep("字段说明（对照打分）")]
    lines += [f"  {n}：{_SCALES[n](sps)}" for n in names]
    lines += extra or []
    return lines


def _beat_legend() -> list[str]:
    """How to read a beat's heading. Beat names are code-layer stage and scene identifiers; without explanation the judge can only guess.

    Scenes aren't translated one by one: they're an open set (each new call path adds one), so a
    translation table needs upkeep and every miss is another unreadable symbol. It's enough to say
    they only distinguish calls within the same beat: the judge rates the output, not where it came from.
    """
    return [
        "  拍的标题（认知环五拍，按这一步真实的先后铺开）：",
        _bullets([
            "perception 感知：此刻看到 / 听到什么，由此生出什么情绪",
            "motivation 动机：眼下最迫切的需求是什么，据此定下要往哪儿去",
            "decision 决策：这一拍做什么",
            "action 行动：世界如何裁定这一动的结果",
            "feedback 反馈：结果落地后他的反应，以及目标推进到哪一步",
        ]),
        "  （拍名后缀是这次调用的 scene，只用来区分同一拍里的不同调用（如情绪自评与目标进展），"
        "是代码层标记，不必解读；「承上·」前缀表示它属于步首那段了结，不是新起的一轮）",
    ]


def _action_type_block() -> list[str]:
    """Decisions keep only the [TYPE] tag (sass has raw output, not the full menu here), so explain what each type actually does."""
    return ["  行动类型（决策的 [类型] 标签含义）：", _bullets(_action_type_legend())]


def _deed_block() -> list[str]:
    """``deed`` in PHYSICAL adjudication output: what the hands-on beat is actually doing. Only sass lays
    out raw output, so only it encounters the word; taken from core.prompts, the same source as the adjudication prompt."""
    return ["  动手的名目（PHYSICAL 裁决 deed 字段的取值）：",
            _bullets([f"{k}：{v}" for k, v in PHYSICAL_DEEDS.items()])]


def _cell_legend_lines(sps: Any) -> list[str]:
    """Scales for reading a ``_step_lines`` cell.

    They follow the renderer, not the scope: sams and mass are two transpositions of the same cell
    (one agent by step / one step by agent), so laying out the same content needs the same scales.
    """
    return _legend_lines("关系", "情绪", "需求", "外部压力", "时间",
                         sps=sps, extra=_action_type_block())


def init_block(view: WorldAuditView) -> str:
    """Initialization audit context: full character profiles + relation descriptions with ages/trust inline + historical events.

    persona_worldview / causal_coherence judge whether a character's setup is off or detached and
    whether age matches form of address, which needs everything: personality / values / background /
    ambitions, plus both sides' ages and labels, or the judge has nothing to compare against.
    """
    init = view.init
    kf = {str(f.get("name")): f for f in init.get("key_figures", []) if isinstance(f, dict) and f.get("name")}
    personas = {str(p.get("name")): p for p in init.get("personas", []) if isinstance(p, dict) and p.get("name")}
    cast = {str(c.get("name")): c for c in init.get("cast_roles", []) if isinstance(c, dict) and c.get("name")}
    era = (init.get("world_time_config") or {}).get("era_name")

    lines = [f"世界：{init.get('world_name')}" + (f"（时代：{era}）" if era else ""),
             f"核心张力：{init.get('core_tension')}",
             f"主题：{init.get('narrative_theme')}",
             _sub_sep("角色档案（判『设定是否违和 / 游离』『动机背景是否服务同一处境』）"),
             f"  （需求刻度与类型：{_need_legend()}）"]
    # key_figures is the primary key: it's the authoritative cast. Going by personas, anyone the persona
    # step failed to generate would vanish from the initialization audit, and "a character is missing"
    # is exactly what this audit should catch, not filter out first.
    for name in list(kf) + [n for n in personas if n not in kf]:
        f, p, c = kf.get(name, {}), personas.get(name, {}), cast.get(name, {})
        # Give the tier (main/background): this audit judges whether there are detached, interchangeable
        # characters unrelated to the core tension, and a background character needn't carry the main plot.
        # Without the tier, the judge measures everyone with the same ruler.
        tier = "主角" if f.get("importance") == "main" else "背景"
        # Where they stand at the start: like the placement of things and errand-runners, a build-time
        # opening fact. Judging whether placement fits their situation, and whether the factions start
        # crowded together or each in their own place, needs it; without it the initialization audit sees
        # where things and errand-runners are but not where the people are.
        where = f"　在 {p['initial_location']}" if p.get("initial_location") else ""
        lines.append(f"\n【{name}】（{tier}）{f.get('role') or p.get('role') or ''}　"
                     f"{f.get('age')}岁　{f.get('gender') or ''}{where}")
        if not p:
            lines.append("  ⚠ 这个角色没有生成人设(persona)——下面的性格/背景/需求整块缺失。")
        if c.get("narrative_role") or c.get("arc_summary"):
            lines.append(f"  叙事角色：{c.get('narrative_role', '')}　弧线：{c.get('arc_summary', '')}")
        if p.get("core_traits"):
            lines.append(f"  性格：{_join(p['core_traits'])}")
        if p.get("core_values"):
            lines.append(f"  价值观：{_join(p['core_values'])}")
        if p.get("self_image"):
            lines.append(f"  自我认知：{p['self_image']}")
        if p.get("background"):
            lines.append(f"  背景：{p['background']}")
        if p.get("life_goal"):
            lines.append(f"  毕生追求：{p['life_goal']}")
        if p.get("long_term_goals"):
            lines.append("  长期目标：")
            lines.append(_bullets(p["long_term_goals"]))
        if p.get("hard_constraints"):
            lines.append("  硬约束（绝不为之）：")
            lines.append(_bullets(p["hard_constraints"]))
        if p.get("initial_needs"):
            lines.append("  初始需求：")
            lines.append(_needs(p["initial_needs"]))
        if p.get("hidden_needs"):
            lines.append("  隐藏需求：")
            lines.append(_needs(p["hidden_needs"]))

    lines.append(_sub_sep("初始关系（就地带年龄 + 双向 label + 信任/好感；判 年龄↔称谓 与 正反向对应）"))
    # Inject the project's own relation_legend() at the start of the relation definitions (authoritative,
    # single source of truth): it states the label's role is the initiator's position in the relation
    # ("父子:儿子" = A is B's son), which keeps the seniority direction from being read backwards.
    lines.append(f"  {relation_legend()}")
    lines.append("  （据上对照年龄判长幼：『兄弟:弟弟』的发起方应比对方年幼，反之矛盾）")
    for r in init.get("initial_relations", []):
        if not isinstance(r, dict):
            continue
        a, b = str(r.get("from", "")), str(r.get("to", ""))
        aa = (kf.get(a) or {}).get("age", "?")
        ab = (kf.get(b) or {}).get("age", "?")
        lines.append(
            f"- {a}({aa}岁) — {b}({ab}岁)：\n"
            f"    {a} 自居为 {b} 的「{_join(r.get('labels'))}」（信任{r.get('trust')}／好感{r.get('affection')}）\n"
            f"    {b} 自居为 {a} 的「{_join(r.get('reverse_labels'))}」（信任{r.get('reverse_trust')}／好感{r.get('reverse_affection')}）")

    lines.append(_sub_sep("历史事件（距开局小时数；判时间线是否倒错）"))
    for h in init.get("historical_events", []):
        if isinstance(h, dict):
            lines.append(f"- [开局前 {h.get('hours_before_start')}h] {h.get('event')}（涉及 {_join(h.get('related_figures'))}）")

    # Opening things and errand-runners: like personas, fixed at build time for the whole run. They are
    # the affordances the world gives its characters: things to seize or hand over, people to send or ask.
    # Without this block the judge sees half the world and can't judge what this opening allows besides
    # talking. Placement (where) must be included: whether things and people sit where the conflict is
    # is something this audit can judge.
    entities = init.get("world_entity_seeds") or []
    npcs = init.get("npcs") or []
    if entities or npcs:
        lines.append(_sub_sep("开局的物与差役（判是否与核心张力咬合、落位是否自洽；纯装饰即为问题）"))
    for e in entities:
        if isinstance(e, dict):
            where = f"　在 {e.get('location_name')}" if e.get("location_name") else ""
            lines.append(f"- [物] {e.get('name')}（{e.get('entity_type')}，初始状态 {e.get('initial_state')}）"
                         f"{where}：{e.get('description') or ''}")
    for n in npcs:
        if isinstance(n, dict):
            where = f"　在 {n.get('location_name')}" if n.get("location_name") else ""
            lines.append(f"- [差役] {n.get('name')}（{n.get('gender') or ''}{n.get('age')}岁）"
                         f"{where}：{n.get('description') or ''}")
    return "\n".join(lines)


# sass audits only the five cognition-loop beats (CLAUDE.md: perception → motivation → decision → action → feedback).
# It's a fixed set that doesn't float with whichever stages happen to declare facts, so the report
# shape is stable and worlds stay comparable. Calls outside the loop (interrupt / memory /
# reflection / relation_evolution / long_term_goals) aren't audited here: memory will get its own
# audit (see the TODO below); the others don't yet declare the facts they receive, and judging
# "added from nowhere" against incomplete evidence would flag sourced statements as fabricated.
# Add them once their given_facts are complete.
# TODO(memory-audit): a separate memory audit scope with its own dimensions (fidelity / importance calibration / voice),
#   sampling K entries from real traces is enough; the narrative consequences of bad memories (amnesia/contradiction) stay covered by sams' causal_coherence.
_SASS_STAGES = ("perception", "motivation", "decision", "action", "feedback")


class SassUnit(NamedTuple):
    """One beat that sass scores individually.

    ``label`` is both the key the judge scores under and the rendered heading; they must be the same
    string, or the keys the judge copies back won't match any beat. ``None`` = this call has no
    judgeable output (the call failed / JSON didn't parse): it gets a one-line note and no score.
    Those problems are reported by the deterministic pre-check and don't cost judge money.
    """

    label: str | None
    carried: bool           # True = part of the step-start settlement segment, not the new round started after it
    call: StageCall


def sass_units(sv: AgentStepView) -> list[SassUnit]:
    """Spread a cell into beats in their real order. Rendering and n_units share this; it isn't computed twice.

    A beat can land more than once in a cell, for three reasons, and each occurrence must be its own
    beat with its own score:
    1. Two things under the same beat by nature: ``feedback`` has both the emotion self-assessment
       and the goal progress verdict (different scenes), and a TALK in ``action`` yields the dialogue
       plus each side's first-person memory.
    2. At step start the engine first settles the action in hand (an interrupt tears it down on the
       spot / a multi-step action finishes), then runs a new round of cognition, so action / feedback
       each appear twice. Which segment a call belongs to is stated by the engine
       (``settles_prior_action``), not guessed from arrival order: memory goes through an async
       queue and arrives late, so order-based rules break.
    3. A decision ruled infeasible gets re-chosen.

    Merging them under one key would make the judge fold two things' fidelity into one score; one
    clean and one fabricated average out invisibly. So the label is (phase, beat, scene, ordinal), which
    keeps it unique.
    """
    units: list[SassUnit] = []
    for carried in (True, False):
        phase = [c for c in sv.calls
                 if bool(c.extra.get("settles_prior_action")) is carried and c.stage in _SASS_STAGES]
        seen: dict[tuple[str, str], int] = {}
        for stage in _SASS_STAGES:
            for c in [x for x in phase if x.stage == stage]:
                if not c.ok or c.parse_ok is False:
                    units.append(SassUnit(None, carried, c))
                    continue
                key = (c.stage, c.scene)
                n = seen[key] = seen.get(key, 0) + 1
                label = f"{'承上·' if carried else ''}{c.stage}·{c.scene}" + (f"#{n}" if n > 1 else "")
                units.append(SassUnit(label, carried, c))
    return units


def _given_facts(c: StageCall) -> list[str]:
    """Every fact this call received, each with its channel prefix; the engine declares them in ``given_facts`` for all five beats.

    The audit reads only this key. It doesn't split the prompt or know any slot names or lists: how the
    prompt is laid out, which candidate lists exist and how system/user are split are tuned for the
    model and the prefix cache, and changing them shouldn't affect the audit. The engine declares
    "what I gave it"; the audit judges "did the output go beyond it". Each side of this membrane
    evolves on its own.
    """
    return [str(t) for t in (c.extra.get("given_facts") or []) if str(t).strip()]


def _index_gloss(c: StageCall) -> list[str]:
    """Translate indices in the output into names. The judge can't tell what `#2` is and so can't
    judge the beat; decisions almost always carry ``selected_index``, and leaving it untranslated
    means scoring a string of numbers.

    Translations come from the engine's declared candidate maps (``action_menu`` / ``*_candidates`` /
    ``goal_texts`` / ``action_participants``), not by stuffing whole lists into the prompt: the lists
    are prompt structure, and "is the choice in the list" belongs to the deterministic pre-check (see
    ``audit_checks.check_bindings``). This only answers "who/what did it pick".
    """
    out = c.output if isinstance(c.output, dict) else {}
    if not out:
        return []
    pairs: list[str] = []
    if (t := _menu_lookup(c.extra.get("action_menu"), out.get("selected_index"))):
        pairs.append(f"selected_index #{out['selected_index']} = {t}")
    for slot, menu_key, _prefix in _TARGET_SLOTS:
        raw = out.get(slot)
        for idx in (raw if isinstance(raw, list) else ([] if raw is None else [raw])):
            if name := _menu_lookup(c.extra.get(menu_key), idx):
                pairs.append(f"{slot} #{idx} = {name}")
    # WORK's updated_index points to which of the things in hand (0 = none of them); the list is declared by that adjudication point.
    if (ui := out.get("updated_index")) is not None and "updated_index" in str(out):
        name = _menu_lookup(c.extra.get("item_candidates"), ui)
        pairs.append(f"updated_index #{ui} = {name or '（一件也没动到）'}"
                     if c.extra.get("item_candidates") is not None else "")
    for g in out.get("goals") or []:
        if isinstance(g, dict) and (txt := _menu_lookup(c.extra.get("goal_texts"), g.get("index"))):
            pairs.append(f"目标 #{g['index']} = {txt}")
    # A dialogue speaker is 1/2, not a name: the two participants in the order set in the prompt (initiator first).
    who = c.extra.get("action_participants") or []
    if isinstance(out.get("dialogue"), list) and len(who) == 2:
        pairs.append(f"对白 speaker 1 = {who[0]}、2 = {who[1]}")
    pairs = [p for p in pairs if p]
    return [f"    序号对照：{'｜'.join(pairs)}"] if pairs else []


def agent_step_block(view: WorldAuditView, aid: str, sv: AgentStepView) -> str:
    """Single agent · single step: for each beat in this cell, what it received → what it output (for per-beat scoring).

    sass audits fidelity: did the output go beyond its own input. So what's laid out here is the
    input the engine declared, not the rendered prompt text. The persona baseline is given once at the
    top: it's an input to every beat too, and repeating it per beat is just noise.
    """
    traj = view.per_agent.get(aid)
    name = traj.name if traj else view.agent_names.get(aid, aid)
    kf = {str(f.get("name")): f for f in view.init.get("key_figures", []) if isinstance(f, dict)}
    sps = view.init.get("seconds_per_step")
    lines = [f"角色：{name}　｜　第 {sv.step} 步（{sv.world_time}）",
             "（逐拍审查「这一拍拿到了什么 → 它输出了什么」；只判忠实度，不判故事好坏。"
             "每一拍各自打分，键用它的「拍：」标题原样抄）",
             # Output often mentions times and places like "明日" or "玄武门"; judging them needs a reference,
             # which is that beat's own "where and when it is now". Don't hoist it to the cell header: time and
             # place aren't constant within a step. After a journey, the action beat reads the new location
             # (the executor takes the live environment), and the third-party adjudication beats use
             # third-person wording. One value for the whole cell would judge correct output against the wrong reference.
             f"这一格是第 {sv.step} 步{f'（一步{describe_duration(1, sps)}）' if sps else ''}。"
             "每一拍自己的「此刻何时何地」才是它的时空——输出里提到的时刻与地点以**那一拍的**为准。"]
    lines.extend(_legend_lines("关系", "情绪", "需求", "外部压力", "时间", "目标状态", "关系走向",
                               sps=sps, extra=_action_type_block() + _deed_block() + _beat_legend()))
    lines.append(_sub_sep("人设基线（每一拍共同的输入；据此判自称的身份 / 关系有没有出处）"))
    lines.extend(_persona_baseline_lines(name, kf.get(name, {}), _personas_by_id(view).get(aid) or {}))
    units = sass_units(sv)
    shown_carried = False
    for unit in units:
        if unit.carried and not shown_carried:
            shown_carried = True
            lines.append(_sub_sep("承上：这一步开头，他手上那件事先了结了（当场被中断，或跨多步的那一动走完）"))
        elif shown_carried and not unit.carried:
            shown_carried = False
            lines.append(_sub_sep("了结之后，这一步他身上接着发生的"))
        c = unit.call
        if unit.label is None:
            lines.append(f"\n· {c.stage}／{c.scene}  ⚠该调用失败或输出无法解析，没有可判的产出")
            continue
        lines.append(_sub_sep(f"拍：{unit.label}"))
        # "Didn't enter the world" has two independent engine channels, and both must be stated:
        #   adopted=False — this output itself was discarded (unusable parse / the engine had it re-choose)
        #   verdict=conscripted·yielded — the output was adopted as an intent, but the world didn't take it up
        # Missing either, the judge goes looking for a result that doesn't exist. Unlike the narrative
        # scopes (which drop them, or they'd be compared with someone else's result), both are still
        # audited: an intent that adds a person from nowhere is a hallucination even if never executed.
        if c.adopted is False:
            why = f"（引擎的内部标记：{c.reject_reason}）" if c.reject_reason else ""
            lines.append(f"  【这一份没有进入世界】引擎未采纳{why}——照常审它的忠实度，"
                         "但别去找它的结果，它没有结果。")
        elif (verdict := c.extra.get("verdict")) in _DISCARDED_VERDICTS:
            what = {"conscripted": "他被并进了别人的行动", "stood_down": "世界没有接下它"}[verdict]
            lines.append(f"  【这一份没有进入世界】{what}——照常审它的忠实度，"
                         "但别去找它的结果，它没有结果。")
        facts = _given_facts(c)
        lines.append("  ▸ 它拿到的事实（这一拍全部的输入；不在这里的都算凭空）：")
        lines.append(_bullets(facts) if facts else "    （这一拍没有申报它拿到的事实——"
                     "无从判它有没有凭空添，grounding 给 null）")
        # The mapping goes before the raw output: on reading `selected_index: 2` you need to already know
        # who #2 is; looking back means reading twice. The output is still laid out in full (judging
        # whether fields agree with each other requires what it actually emitted).
        lines.append("  ▸ 它输出了：")
        lines.extend(_index_gloss(c))
        lines.append(_indent(_fmt_output(c.output)))
    return "\n".join(lines)


def _personas_by_id(view: WorldAuditView) -> dict[str, dict]:
    return {p["agent_id"]: p for p in view.init.get("personas", [])
            if isinstance(p, dict) and p.get("agent_id")}


def agent_trajectory_block(view: WorldAuditView, aid: str) -> str:
    """Single agent · multi step: a compact summary of the agent's full cognition chain + state changes per step (scored as a whole).

    Each step gives perception → motivation → (interrupt) → decision → action → feedback + changes in
    emotion / short-term goals / need activation, plus any relation evolution / long-term goal
    adjustment triggered that step, so the judge can assess cross-step causal continuity (amnesia /
    abrupt shifts), emotional and need development, growth and arc. A persona baseline (personality /
    long-term ambitions) comes first as the reference for character_growth / convergence.
    Compact rather than every field: sams is one call across all steps, and every field would blow
    up the tokens (unlike sass's per-stage full fields).
    """
    traj = view.per_agent[aid]
    kf = {str(f.get("name")): f for f in view.init.get("key_figures", []) if isinstance(f, dict)}
    persona = _personas_by_id(view).get(aid) or {}
    k = kf.get(traj.name, {})
    sps = view.init.get("seconds_per_step")  # convert estimated_steps to natural durations (the narrative layer has no "step")
    # How many steps the world ran vs which steps he had cognition/actions in: both numbers are needed.
    # With only the latter, someone idle most of the time reads as "the window was this short", and
    # arc / pacing / repetition get misjudged (in the missing steps he was neither scheduled nor had an
    # action settle; nothing happened, it's not missing data).
    span = len(view.world_sequence) or len(traj.steps)
    lines = [f"角色：{traj.name}　｜　世界共 {span} 步，其中他有认知 / 行动的是下列 "
             f"{len(traj.steps)} 步（整段审查，逐步小结；未列出的步他既未被调度、也无行动结算）"]
    lines.extend(_cell_legend_lines(sps))
    # Persona baseline (full): identity + personality/values/self-image/life goal/background/hard constraints/long-term ambitions, to judge drift from persona and growth.
    lines.append(_sub_sep("人设基线（逐步若无『长期目标调整』则不变；据此判是否偏离人设、有无成长）"))
    lines.extend(_persona_baseline_lines(traj.name, k, persona))

    # Iterate per step over the same per-(agent, step) rendering (shared with mass via _step_lines).
    for sv in traj.steps:
        lines.append(_step_sep(sv.step, sv.world_time))
        lines.extend(_step_lines(sv, name=traj.name, sps=sps))
    return "\n".join(lines)


def _persona_baseline_lines(name: str, k: dict[str, Any], persona: dict[str, Any]) -> list[str]:
    """Persona baseline: identity (name/tier/role/age/gender) + personality/values/self-image/life goal/background/hard constraints/long-term ambitions.

    Single source of truth: rendered once at the top for sams (single agent) and once per character
    for mass (multi agent). The judge uses it to assess drift from persona / OOC / whether a relation
    reversal is reasonable (whether "persecuting a teammate" is abrupt depends on each agent's stance
    and faction). Shared by both, so it can't drift.
    """
    tier = "主角" if k.get("importance") == "main" else "背景"
    lines = [f"  {name}（{tier}）　{k.get('role') or persona.get('role') or ''}　"
             f"{k.get('age')}岁　{k.get('gender') or ''}"]
    if persona.get("core_traits"):
        lines.append(f"  性格：{_join(persona['core_traits'])}")
    if persona.get("core_values"):
        lines.append(f"  价值观：{_join(persona['core_values'])}")
    if persona.get("self_image"):
        lines.append(f"  自我认知：{persona['self_image']}")
    if persona.get("life_goal"):
        lines.append(f"  毕生追求：{persona['life_goal']}")
    if persona.get("background"):
        lines.append(f"  背景：{persona['background']}")
    if persona.get("hard_constraints"):
        lines.append("  硬约束（绝不为之）：")
        lines.append(_bullets(persona["hard_constraints"]))
    ltg = persona.get("long_term_goals") or ([persona["life_goal"]] if persona.get("life_goal") else [])
    if ltg:
        lines.append("  长期目标：")
        lines.append(_bullets(ltg))
    return lines


def _numbered(items: list, render) -> list[str]:
    """Number items when more than one lands in a step. A single one isn't numbered: a lone "①" is just noise."""
    if len(items) == 1:
        return render(items[0], "")
    out: list[str] = []
    for i, item in enumerate(items, 1):
        out.extend(render(item, f"（其{i}）"))
    return out


def _phases(s: dict[str, Any]) -> list[dict[str, Any]]:
    """Phase order within a cell: the engine settles the action in hand first and then starts a new
    round, so with carry-over there are two segments, carry-over first (see audit_reconstruct.summarize).

    This contract is read only here. sams/mass and mams are two renderers with their own wording and
    indentation, but "a cell may have two segments, and which comes first" is the same fact. Reading it
    in each would let one side miss it, and that side would attach the new decision to the interrupted
    action's result.
    """
    carried = s.get("carried")
    return [carried, s] if carried else [s]


def _step_lines(sv: AgentStepView, *, name: str, sps: Any) -> list[str]:
    """Everything for one (agent, step), laid out in the step's real order.

    Single source of truth: sams iterates by step and mass by character, both through this, so a fix in one place applies to both.
    """
    phases = _phases(sv.summary)
    lines: list[str] = []
    for i, phase in enumerate(phases):
        if len(phases) > 1:
            lines.append("  ── 承上：这一步开头，他手上那件事先了结了（当场被中断，或跨多步的那一动走完）──"
                         if i == 0 else "  ── 了结之后，这一步他身上接着发生的 ──")
        lines.extend(_phase_lines(phase, name=name, sps=sps))
    return lines


def _phase_lines(s: dict[str, Any], *, name: str, sps: Any) -> list[str]:
    """One phase's content, laid out along the cognition loop: interrupt → perception → motivation → decision → action → feedback → memory → step end.

    Two things can still land in one phase (pulled into someone else's action in the step it
    settles); they're then numbered and listed. Merging would pair A's result with B's inputs, and
    dropping one leaves the judge unable to tell anything is missing.
    """
    lines: list[str] = []
    itr = s.get("interrupt") or {}
    if itr.get("doing") or itr.get("trigger"):
        ip = [x for x in (f"正在做:{itr['doing']}" if itr.get("doing") else "",
                          f"突发:{itr['trigger']}" if itr.get("trigger") else "") if x]
        lines.append("  中断｜输入：" + "｜".join(ip))
    if itr.get("thought") is not None or itr.get("interrupt") is not None:
        lines.append(f"  中断｜输出：{itr.get('thought', '')}（interrupt={itr.get('interrupt')}）")
    if s.get("location"):
        # Where he is: the premise for judging "conflicting actions in the same place" and "did he show up
        # when the agreement fell due". Location only: time is one per step and already in the step
        # separator, so repeating it here would duplicate it for every person every step.
        lines.append(f"  他在：{s['location']}")
    if s.get("prior_mood"):
        lines.append(f"  感知前心境：{s['prior_mood']}")
    if s.get("relations"):  # Relations (now): list + direction (who is what to whom); can change per step, so relation development is visible
        lines.append("  关系(此刻)：")
        lines.extend(
            f"    - {name} 自居为 {r['name']} 的「{r['labels'] or '未明确'}」"
            f"（信任{_num(r['trust'])}/好感{_num(r['affection'])}）" for r in s["relations"])
    if s.get("perceived"):
        lines.append("  感知｜输入(此刻感知到)：")
        lines.append(_bullets(s["perceived"]))
    emo = s.get("emotion") or {}
    po = [x for x in (
        f"触动：{s['perception_reason']}" if s.get("perception_reason") else "",
        f"情绪 {emo.get('primary')}(强度{emo.get('intensity')},效价{emo.get('valence')})"
        if emo.get("primary") else "",
        f"激活需求 {_fmt_output(s['need_activation'])}" if s.get("need_activation") else "",
    ) if x]
    if po:
        lines.append("  感知｜输出：" + "｜".join(po))
    if s.get("dominant_need"):
        lines.append(f"  动机｜输入：主导需求 {s['dominant_need']}")
    lines.extend(_goal_lines("  现有短期目标", s.get("prior_goals")))
    if s.get("motivation_thought"):
        lines.append(f"  动机｜输出：思考:{s['motivation_thought']}")
    lines.extend(_goal_lines("  新短期目标", s.get("short_term_goals")))
    lines.extend(_decision_lines(s, sps))
    lines.extend(_action_lines(s, name))
    for ap in s.get("appraisal") or []:
        before = emo.get("primary")
        arrow = f"{before}→{ap['emotion']}" if before and ap.get("emotion") else ap.get("emotion")
        lines.append(f"  反馈｜输出：情绪 {arrow}(强度{ap.get('intensity')},效价{ap.get('valence')})")
    lines.extend(_numbered(s.get("goal_progress") or [],
                           lambda g, tag: _goal_progress_lines(g, tag)))
    for res in s.get("residue") or []:
        lines.extend(_goal_lines("新留下的未了事项", res))
    for mem in s.get("memory") or []:
        lines.append(f"  记忆(主观感受)：{mem}")
    lines.extend(_step_level_lines(s))
    return lines


def _decision_lines(s: dict[str, Any], sps: Any) -> list[str]:
    """Decision. A discarded intent isn't laid out at all, just one line on what happened to it: laying
    out an intent that never executed would only get it compared with the (often someone else's) action
    result below and flagged as "decision inconsistent with action"."""
    if (why := s.get("intent_dropped")):
        # The action he was folded into doesn't necessarily settle this step: a multi-step interaction only
        # yields a result when it finishes, and until then he has no action under his name. Pointing to an
        # absent "action below" sends the judge looking for nothing, and it reads as a gap in this step's record.
        where = ("（见下方「行动」）" if (s.get("results") or s.get("joined"))
                 else "（那一动这一步还没走完，结果要到它走完那一步才有）")
        return [{
            "conscripted": f"  【本步他没有自己的行动】他被并进了别人的行动{where}，"
                           "这一步他自己决定的那件事与随之生成的短期目标都已作废、未进入世界——"
                           "属正常调度机制，勿判断链；这一意图若在后续步重现，那是被顺延的尝试，不算重复空转。",
            "stood_down": "  【本步他没有出手】他这一步的意图没有被世界接下（未被调度 / 想过了不出手 / "
                          "决策不可用），未进入世界——不是重复空转，也没有可评判的行动。",
        }[why]]
    if s.get("decision_discarded"):
        return [f"  决策｜输出：（本步决策被引擎丢弃：{s['decision_discarded']}，未进入世界——"
                "该角色这一步什么都没做，既非重复空转，也没有可评判的行动）"]
    dec = s.get("decision") or {}
    if not (dec.get("action_description") or dec.get("selected_index") is not None):
        return []
    at = f"[{dec['action_type']}] " if dec.get("action_type") else (
        f"选#{dec.get('selected_index')} " if dec.get("selected_index") is not None else "")
    parts = [f"{at}{dec.get('action_description')}"]
    if dec.get("target"):
        parts.append(f"对象:{dec['target']}")
    if dec.get("inner_monologue"):
        parts.append(f"为何:{dec['inner_monologue']}")
    if dec.get("expected_outcome"):
        parts.append(f"预期:{dec['expected_outcome']}")
    out = ["  决策｜输出：" + "｜".join(parts)]
    # Multi-step actions are marked only at the start. Later steps render as usual; it's the same action continuing, which readers can see. The audit doesn't chain across steps.
    if (n := dec.get("spans_steps")) and sps:
        out.append(f"  【这一动跨多步】要占 {describe_duration(n, sps)}，结果在它走完的那一步才有；"
                   "此后几步他仍在做这件事。")
    if dec.get("rejected"):
        out.append("  【这一动被判不可行】仲裁当场否了它（同伴毫无回应 / 正忙于他事），"
                   "当步即以失败结算——理由见下方结果。")
    return out


def _action_lines(s: dict[str, Any], name: str) -> list[str]:
    """Action: what actually happened to him this step. Being invited in / acted upon must be stated: those aren't actions he decided on."""
    lines: list[str] = []
    for ci in s.get("joined") or []:
        lines.append(f"  行动｜被邀入：{ci}"
                     "（他被他人邀入此互动，执行的不是自己决定的动作——属正常调度机制）")
    for au in s.get("acted_upon") or []:
        lines.append(f"  行动｜被施加：{au}"
                     "（他是他人行动的对象，下方结果是被施加之后的反应，不是他自己决定或执行的动作）")
    results = s.get("results") or []
    peer = s.get("peer") or (s.get("decision") or {}).get("target")
    shown: set[str] = set()  # several results from the same action (dialogue + own memory) share one input; report it once

    said = (s.get("decision") or {}).get("message_content") or ""
    said_done = False

    def _one(r: Any, tag: str) -> list[str]:
        nonlocal said_done
        head = ""
        if isinstance(r, dict) and r.get("did") and r["did"] not in shown:
            shown.add(r["did"])
            head = _action_input_line(r, name, "" if said_done else said)
            said_done = said_done or bool(said)
        return ([head] if head else []) + _action_output_lines(r, name, peer, tag)

    lines.extend(_numbered(results, _one))
    if said and not said_done:
        # An action that spans steps from the start (an errand-runner's message, a letter) has no result this
        # step, but its exact words still belong on the starting step: the settling step has no decision to take them from.
        lines.append(f"  行动｜输入：原话:{said}")
    return lines


def _step_level_lines(s: dict[str, Any]) -> list[str]:
    """Output of the step-end maintenance phase: runs after every execution has settled, at most once per step, and belongs to no action.

    Laid out as readable lines, never raw JSON: the IndexedRef indices in it have no list to check
    against in the audit. Relation evolution's index is translated into who the other party is
    (attached via extra.relation_targets); reflection's indices are only its own footnotes, and the
    judge rates the insight itself, so with nothing to look them up in, they're omitted.
    """
    lines: list[str] = []
    ltg = s.get("long_term_goals")
    if isinstance(ltg, dict):
        # Whether long-term goals changed, and to what, is the most direct cell for judging growth / arc.
        # Lay out what the engine finally adopted, not what the LLM wrote: the latter is discarded whole in
        # two cases (empty array, identical to the old), and laying it out would report a change that never happened.
        lines.append(f"  ⟳长期目标重审：{ltg.get('thought') or ''}".rstrip())
        applied = ltg.get("applied") or []
        lines.extend(f"    - 改为：{g}" for g in applied)
        if not applied:
            lines.append("    - 方向未变（他逐条审过，没有改动）")
    elif ltg:
        lines.append(f"  ⟳长期目标重审：{_fmt_output(ltg)}")
    rel = s.get("relation_evolution")
    for u in (rel.get("updates") or []) if isinstance(rel, dict) else []:
        if not isinstance(u, dict):
            continue
        labels = f"「{_join(u.get('labels'))}」" if u.get("labels") else ""
        tail = f"（{u['rationale']}）" if u.get("rationale") else ""
        lines.append(f"  ⟳关系演化：对 {u.get('target') or '某人'} {labels}{u.get('summary') or ''}{tail}")
    ref = s.get("reflection")
    for ins in (ref.get("insights") or []) if isinstance(ref, dict) else []:
        if isinstance(ins, dict) and ins.get("text"):
            lines.append(f"  ⟳反思（重要度{ins.get('importance')}）：{ins['text']}")
    return lines


def _num(v: Any) -> str:
    """Relation values: strip floating-point noise (0.08800000000000002 → 0.088), or the judge reads false precision."""
    try:
        return f"{round(float(v), 3):g}"
    except (TypeError, ValueError):
        return str(v)


def cross_agent_block(view: WorldAuditView, step: int) -> str:
    """Multi agent · single step: the characters' interactions within one step — who does what to whom, whether information flows, the relation matrix.

    The dimension is action_efficacy (does interaction really change relations / information / the
    world, or is it idle small talk and formality). Each character reuses the same ``_step_lines``
    rendering as sams (relation matrix / perception / decision [type] + target / result viewpoint /
    dialogue summary…), keeping both scopes structurally consistent. The only difference: sams
    iterates by step with step separators, mass by character with character separators. Field
    definitions/legends go first (§3); action types/targets resolve to names (no leaked indices/ids).
    """
    cells = view.cells_at(step)
    wt = next((e.get("world_time", "") for e in view.world_sequence if e.get("step") == step), "")
    names = "、".join(name for _, name, _ in cells) or "（无）"
    lines = [f"第 {step} 步（{wt}）·在场角色：{names}",
             "（审查同一步内各角色**之间**的交互是否真正改变关系/信息/世界状态，还是空转寒暄、流于形式）"]
    sps = view.init.get("seconds_per_step")
    lines.extend(_cell_legend_lines(sps))
    kf = {str(f.get("name")): f for f in view.init.get("key_figures", []) if isinstance(f, dict)}
    personas = _personas_by_id(view)
    for aid, name, sv in cells:
        lines.append(_sub_sep(f"角色 {name}"))
        # Persona baseline (shared with sams): the judge needs each agent's stance/faction/personality to judge whether interactions are OOC and relation reversals abrupt.
        lines.extend(_persona_baseline_lines(name, kf.get(name, {}), personas.get(aid) or {}))
        lines.extend(_step_lines(sv, name=name, sps=sps))  # shares the sams rendering
    return "\n".join(lines)


_GOAL_SUFFIXES = ("（已历", "（约定", "（这是他先前")


def _goal_gist(text: Any) -> str:
    """Strip the time/provenance suffix need.py appends to goal text, leaving only the goal."""
    out = str(text or "")
    for suffix in _GOAL_SUFFIXES:
        out = out.split(suffix)[0]
    return out.strip() or str(text or "")


def _emo(pri: Any, i: Any, v: Any) -> str:
    tail = "，".join(x for x in ((f"强度{i}" if i is not None else ""),
                                 (f"效价{v}" if v is not None else "")) if x)
    return f"{pri}（{tail}）" if tail else str(pri)


def _mams_lines(name: str, s: dict[str, Any], *, sps: Any) -> list[str]:
    """mams renders each (step, agent) as a labeled multi-line block: every field on its own labeled
    line, not crammed into one line with "｜" (relation labels contain "|" too, so one line would be
    ambiguous). The director has to tell "intended" from "did" at a glance.

    It shares the same summary as sams (phase order also via ``_phases``), just laid out thinner: no
    cognition details (motivation monologue / memory prose are sams-level).

    Returns empty when nothing landed in the cell: a block with just a name tells the director nothing.
    Whether it's empty is decided by the rendering itself, not by a separate list of "keys that count
    as content": that list would need updating with every new field, and a miss would swallow a person
    with real content.
    """
    dec = s.get("decision") or {}
    results = s.get("results") or []
    relations = s.get("relations")
    at = dec.get("action_type") or ""
    t = dec.get("target")
    rel = ""
    if t:
        tc = t.lstrip("→")
        for r in relations or []:
            if r.get("name") == tc:
                lb = f"{r.get('labels')}，" if r.get("labels") else ""
                rel = f"（{lb}信任{_num(r.get('trust'))}/好感{_num(r.get('affection'))}）"
                break
    tgt = ((t if t.startswith("→") else f"→{t}") + rel) if t else ""
    has_act = bool(at or dec.get("action_description") or results
                   or s.get("joined") or s.get("acted_upon"))
    # "做了" = a result actually landed this step; "想做" = only an intent, with the result still to come.
    kind = ("【行动（做了，已经发生）】" if results else
            "【决策（决定要做，实际还没发生）】") if has_act else ""
    phases = _phases(s)
    body: list[str] = []
    for i, phase in enumerate(phases):
        if len(phases) > 1:
            body.append("      · ── 承上：这一步开头，他手上那件事先了结了 ──" if i == 0
                        else "      · ── 了结之后，这一步他身上接着发生的 ──")
        body.extend(_mams_phase_lines(phase, name, sps=sps))
    return [f"  ● {name}{kind}{at}{tgt}".rstrip(), *body] if body else []


def _mams_phase_lines(s: dict[str, Any], name: str, *, sps: Any) -> list[str]:
    """One phase's body in mams: interrupt → what he does → result → emotion → goal progress → long-term goal review."""
    lines: list[str] = []
    dec = s.get("decision") or {}
    results = s.get("results") or []
    itr = s.get("interrupt") or {}
    if s.get("location"):
        lines.append(f"      · 他在：{s['location']}")
    if itr.get("interrupt"):
        ctx = "｜".join(x for x in ((f"正在做:{itr['doing']}" if itr.get("doing") else ""),
                                    (f"突发:{itr['trigger']}" if itr.get("trigger") else "")) if x)
        lines.append(f"      · 【中断】他当场放下了手上那件事（{ctx}）：{itr.get('thought') or ''}")
    if (why := s.get("intent_dropped")):
        lines.append({
            "conscripted": "      · 【本步他没有自己的行动】被并进了别人的行动，自己决定的那件事与"
                           "随之生成的短期目标都已作废——属正常调度机制，勿判断链；这一意图若在后续步"
                           "重现，那是被顺延的尝试，不算重复空转。",
            "stood_down": "      · 【本步他没有出手】意图没有被世界接下，未进入世界——不是重复空转。",
        }[why])
    for ci in s.get("joined") or []:
        lines.append(f"      · 【被邀入】{ci}（执行的不是他自己决定的动作）")
    for au in s.get("acted_upon") or []:
        lines.append(f"      · 【被施加】{au}（下方结果是被施加之后的反应，非他自己所决策 / 所执行）")
    # "What he does" prefers this phase's decision; phases without one (a multi-step action finishing,
    # settling an interrupted action, or being folded into someone else's action) fall back to the
    # landed action's own description, but only one he initiated; otherwise the initiator's first person would be pinned on him.
    body = dec.get("action_description") or next(
        (r["did"] for r in results
         if isinstance(r, dict) and r.get("did") and r.get("initiator") in (None, name)), "")
    # The exact words sent travel with the action, not on a separate line: split up, the director would
    # have to match the two lines back to the same action across the result and emotion between them.
    # TALK's words are already folded into action_description (decision._bind_talk), so they aren't repeated.
    said = dec.get("message_content") or ""
    did_parts = [x for x in (body, f"原话:{said}" if said and said not in body else "") if x]
    if did_parts:
        lines.append("      · 做什么：" + "｜".join(did_parts))
    if (n := dec.get("spans_steps")) and sps:
        lines.append(f"      · 【跨多步】这一动要占 {describe_duration(n, sps)}，结果在它走完那一步才有。")
    if dec.get("rejected"):
        lines.append("      · 【被判不可行】仲裁当场否了它，当步以失败结算。")
    # One phase can hold both results he initiated and results others applied to him. When the initiator
    # isn't him, name the initiator (same as _action_input_line): unnamed it reads as "he did it himself",
    # and someone else's [success] under "his action was ruled infeasible" reads as a contradiction.
    for i, r in enumerate(results, 1):
        by = r.get("initiator")
        marks = ([f"其{i}"] if len(results) > 1 else []) + ([f"由 {by} 发起"] if by and by != name else [])
        tag = f"（{'，'.join(marks)}）" if marks else ""
        succ = r.get("success") if "success" in r else r.get("achieved")
        mark = "[成功] " if succ is True else ("[失败] " if succ is False else "")
        if oc := (r.get("outcome") or r.get("fact")):
            lines.append(f"      · 结果{tag}：{mark}{oc}")
    # Emotion has two sources, and which one must be stated: perception emotion is how he sees the
    # situation; feedback emotion is his reaction once the result lands. A bare emotion is read as a
    # reaction to the result, which this step may not have (conscripted / interrupted means no result,
    # or the result landed without a feedback emotion). Say "no reaction to a result", not "no result":
    # the latter would contradict a "结果：…" that may already be laid out above.
    e = s.get("emotion") or {}
    before = e.get("primary")
    afters = [a for a in (s.get("appraisal") or []) if a.get("emotion")]
    last = afters[-1] if afters else None
    if before and last:
        lines.append(f"      · 情绪：{before}→{_emo(last['emotion'], last.get('intensity'), last.get('valence'))}")
    elif last:
        lines.append(f"      · 情绪（对结果的反应）：{_emo(last['emotion'], last.get('intensity'), last.get('valence'))}")
    elif before:
        lines.append(f"      · 情绪（他此刻所感；本步还没有结果的反应）：{_emo(before, e.get('intensity'), e.get('valence'))}")
    # Goal status transitions: only completed / failed move the plot (in progress = nothing happened; laying it out is noise).
    # Goal text isn't truncated: a half sentence can't say which goal was completed, and that's this cell's only information.
    trans = [f"{_goal_gist(g.get('text'))}→{_GOAL_STATUS_LABELS.get(g.get('status'), g.get('status'))}"
             for batch in (s.get("goal_progress") or []) for g in batch
             if isinstance(g, dict) and g.get("status") in ("completed", "failed")]
    if trans:
        lines.append("      · 目标进展：")
        lines.extend(f"          - {x}" for x in trans)
    # Long-term goal review: at most a few per run, but the most direct evidence of growth / arc; the director's view can't see only short-term goals.
    # Conclusion only (changed to what / unchanged); the item-by-item thought belongs to sams.
    ltg = s.get("long_term_goals")
    if isinstance(ltg, dict):
        applied = ltg.get("applied") or []
        if applied:
            lines.append("      · ⟳长期目标重审，改为：")
            lines.extend(f"          - {g}" for g in applied)
        else:
            lines.append("      · ⟳长期目标重审：方向未变")
    return lines


def _mams_script_lines(actor: str, s: dict[str, Any]) -> list[str]:
    """God's-eye script: TALK's actual dialogue (speaker 1 = initiator, 2 = other party, resolved to names).
    mams is the director's view and needs to read what was actually said to judge the narrative, so
    there's no viewpoint filtering here; every line is laid out. Words sent out (letters / messages
    carried by errand-runners) aren't here; they travel with "what he does".

    Both phases are searched: an interrupted conversation settles at step start, so its dialogue belongs
    to the carry-over segment. The other party is resolved from that phase's own participants/targets,
    never borrowed across phases.
    """
    for phase in (s.get("carried"), s):
        if not isinstance(phase, dict):
            continue
        dlg = next((r.get("dialogue") for r in (phase.get("results") or []) if r.get("dialogue")), None)
        if isinstance(dlg, list) and dlg:
            target = (phase.get("decision") or {}).get("target") or phase.get("peer")
            break
    else:
        dlg = None
    if isinstance(dlg, list) and dlg:
        who = {1: actor, 2: str(target) if target else "对方"}
        out = ["      · 对话："]
        for turn in dlg:
            if not isinstance(turn, dict):
                continue
            sp = who.get(turn.get("speaker"), f"角色{turn.get('speaker')}")
            out.append(f"          {sp}：{str(turn.get('line', '')).strip()}")
        return out
    return []


def _rel_deltas(name: str, relations: Any, prior: dict[str, dict[str, tuple[float, float]]]) -> list[str]:
    """Changes between agents: compare this step's relation matrix for name with the last one seen, reporting only |Δtrust| or |Δaffection| ≥ 0.1 (cross-step development = relation arc).
    ``prior`` is updated in place to the latest values. Relations' trust/affection are strings, converted to float safely."""
    out: list[str] = []
    for r in relations or []:
        tgt = r.get("name")
        try:
            tr, af = float(r.get("trust")), float(r.get("affection"))
        except (TypeError, ValueError):
            continue
        prev = prior.setdefault(name, {}).get(tgt)
        if prev is not None and (abs(tr - prev[0]) >= 0.1 or abs(af - prev[1]) >= 0.1):
            lb = f"「{r.get('labels')}」" if r.get("labels") else ""
            out.append(f"{name}→{tgt}{lb} 信任{_num(prev[0])}→{_num(tr)} 好感{_num(prev[1])}→{_num(af)}")
        prior[name][tgt] = (tr, af)
    return out


def world_block(view: WorldAuditView) -> str:
    """Multi agent · multi step: the holistic narrative — persona factions + the evolution sequence across steps × all agents.

    Dimensions: narrative_convergence (converging on the theme / long-term goals rather than everyone
    going their own way) / dramatic_arc (the conflict-emotion curve) / emergent_novelty (misunderstanding
    / reversal / scheming) / action_efficacy (actions drive the world, not a log of events) /
    persona_worldview (the dead reappearing / timeline / OOC). So it borrows mass's resolved interaction
    edges (type + target) + persona baseline and sams' cross-step sequence, but compresses each
    (step, agent) to one line (across all steps × agents, full blocks would blow up the tokens).
    Long-term ambitions = the convergence target (in the persona baseline).
    """
    init = view.init
    era = (init.get("world_time_config") or {}).get("era_name")
    lines = [f"世界：{init.get('world_name')}" + (f"（时代：{era}）" if era else "")
             + f"｜角色：{'、'.join(view.agent_names.values())}",
             f"核心张力：{init.get('core_tension')}",
             f"主题：{init.get('narrative_theme')}"]
    # Field definitions/legends go first (§3): relations (trust/affection scale + label meaning), emotion (for arcs), action types ([type] tag meaning).
    lines.extend(_legend_lines("关系", "情绪", "时间", sps=init.get("seconds_per_step"),
                               extra=_action_type_block()))
    # Persona baseline (shared with sams/mass): to judge OOC / the dead reappearing / everyone going their own way; long-term ambitions = convergence target.
    kf = {str(f.get("name")): f for f in init.get("key_figures", []) if isinstance(f, dict)}
    personas = _personas_by_id(view)
    lines.append(_sub_sep("人设基线（判是否 OOC / 死者复现 / 各行其是；长期志向=朝其收敛的靶子）"))
    for i, (aid, name) in enumerate(view.agent_names.items()):
        if i:  # dotted separator between agents' personas so boundaries are clear (this section has no character header; without it they run together)
            lines.append("  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈")
        lines.extend(_persona_baseline_lines(name, kf.get(name, {}), personas.get(aid) or {}))
    # Cross-step evolution sequence: per step, events + one block per agent (action/outcome/result/own changes) + relation changes (between agents).
    lines.append(_sub_sep("跨步演化序列（审查整体叙事：收敛 / 弧线 / 涌现 / 有效性 / 世界观）"))
    sps = view.init.get("seconds_per_step")  # convert the decision's estimated duration (the narrative layer has no "step")
    prior_rel: dict[str, dict[str, tuple[float, float]]] = {}  # track relation values across steps to diff out changes between agents
    for entry in view.world_sequence:
        lines.append(_step_sep(entry["step"], entry.get("world_time")))
        # World pressure (meta, a scheduling signal) isn't injected: it isn't something that happens in the world, and it distracts the director (a deliberate decision).
        for ev in entry.get("events", []):  # events = external variables actually injected into the world (e.g. a secret letter); narrative-level, so make them prominent
            if isinstance(ev, dict) and (ev.get("message") or ev.get("narrative_desc")):
                lines.append(f"  ⚡事件：{ev.get('narrative_desc') or ev.get('message')}")
        rel_changes: list[str] = []
        # Same cell path as mass (view.cells_at), just laid out step × character here.
        for _aid, nm, sv in view.cells_at(entry["step"]):
            if block := _mams_lines(nm, sv.summary, sps=sps):  # empty cells produce no block; the rendering decides
                lines.extend(block)
                lines.extend(_mams_script_lines(nm, sv.summary))  # god's-eye: what was actually said
            rel_changes.extend(_rel_deltas(nm, sv.summary.get("relations"), prior_rel))
        # Changes between agents: this step's relation matrix vs the last time it was seen. Relations are
        # only visible on steps where the agent perceived, and steps without perception aren't compared, so
        # a change may have accumulated over several steps and is recorded on the step it was seen. State
        # that, or the judge reads it as "these changes happened this step". ≥2 → list (§4; labels contain |, hard to read on one line)
        if rel_changes:
            if len(rel_changes) == 1:
                lines.append(f"  关系变动（较上次见到该角色的关系时；中间无感知的步不计）：{rel_changes[0]}")
            else:
                lines.append("  关系变动（较上次见到该角色的关系时；中间无感知的步不计）：")
                lines.extend(f"    - {c}" for c in rel_changes)
    return "\n".join(lines)
