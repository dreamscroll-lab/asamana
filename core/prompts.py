"""Prompt scales and shared renderers — the single source of numeric semantics and text shapes in prompts.

Without a scale the LLM guesses whether `intensity=0.85` is high; with one per prompt, wording
drifts. Hence:

- Numeric fields: on the output side (LLM produces a number) inject ``*_DEFINITION`` /
  ``*_SCALE_DESCRIPTION``; on the consumer side (LLM reads a number) the same, or translate it to
  natural language with ``*_label()``. A new numeric field gets a constant here first, not an
  inline description in a prompt.
- Shared renderers: the prompt shape of people / places / things / memories / conditions /
  situation headers is defined only here (``person_referent`` / ``render_location`` /
  ``render_entity`` / ``render_memory`` / ``condition_line`` / ``render_situation_header``), so a
  format change happens in one place.
- Shared rules: cross-prompt writing discipline (closed world, absolute time points, …) lives
  here, each in a functional and an in-character version (see the constants).
"""

from __future__ import annotations

from enum import Enum

from core.duration import describe_duration
from core.interfaces.phenomenon import Phenomenon
from core.interfaces.urgency import Urgency

# ============================================================================
# Emotion (scales for emotion fields)
# ============================================================================

EMOTION_INTENSITY_DEFINITION: str = (
    "intensity（情绪强度）: 浮点数 [0.0, 1.0]，情绪被激活的程度，与情绪种类无关。"
    "0.0=平静，0.2=轻微，0.5=明显，0.8=强烈，1.0=极度激烈"
)

EMOTION_VALENCE_DEFINITION: str = (
    "valence（情绪基调）: 浮点数 [-1.0, 1.0]，情绪的正负质感。"
    ">0 愉快/积极，<0 痛苦/消极，≈0 中性"
)


# ============================================================================
# Vitality damage (PHYSICAL adjudication output; same scale as agent.personality.vitality)
# ============================================================================

#: Cause-agnostic, like VITALITY_RELIEF_DEFINITION: otherwise exhaustion would have nowhere to land.
VITALITY_DAMAGE_DEFINITION: str = (
    "damage（生命力损耗）: 浮点数 [0.0, 1.0]，一次行动耗去对方多少生命力"
    "——不论损在哪里（负伤、力竭、精神耗尽等等皆可）。"
    "0.0=分毫未损 | 0.3=损耗明显 | 0.7=危及生命 | 1.0=致命"
)

#: Cap on vitality restored on the spot by one action, shared by the scale text, the schema and
#: the parse clamp; don't write the number anywhere else. It caps an amount, not a rate: rescue
#: outpacing REST is intended (it needs someone present with the means, and costs their beat).
#: Relief out of thin air is stopped by the means gate; overshoot is clamped by apply_vitality_damage.
VITALITY_RELIEF_CAP: float = 0.2

#: Cause-agnostic: degree, not cause. Tied to healing wounds, feeding someone starving would score 0.
VITALITY_RELIEF_DEFINITION: str = (
    f"relief（生命力挽回）: 浮点数 [0.0, {VITALITY_RELIEF_CAP}]，一次行动为对方挽回多少生命力"
    f"——不论他亏在哪里（负伤、力竭、饥渴、寒冷等等皆可）。"
    f"0.0=没帮上什么 | 0.05=稍微缓解一点点 | {VITALITY_RELIEF_CAP}=从濒死边上被拉了回来。"
    f"它只管当场挽回，不管整个生命力是否真的完全挽回。"
)


#: The ``deed`` of a PHYSICAL adjudication — the act, not what it achieved. The judge prompt offers
#: a subset by target kind; the audit reads values back into Chinese from the same table.
PHYSICAL_DEEDS: dict[str, str] = {
    "strike": "施加暴力伤害",
    "restrain": "动手但不为伤人：制止/擒抱/推搡/搀扶/救治",
    "seize": "把它拿到手上带走",
    "relinquish": "把它交出手：放到此地，或交到某人手上",
    # operate vs destroy turns on whether the thing still exists afterward, not how violent the act
    # looks. Don't put destruction verbs in operate's description, or destroying never reaches destroy.
    "operate": "就地操作它：开启/使用/改变其状态——事后它仍在",
    "destroy": "把它毁掉：烧毁/砸碎/撕毁——事后它不复存在",
}


def deed_options(*names: str) -> str:
    """Selected deeds → "strike（施加暴力伤害）、restrain（…）"."""
    return "、".join(f"{n}（{PHYSICAL_DEEDS[n]}）" for n in names)


#: Lower bound of "充沛", shared by vitality_label and vitality_line; don't write the number elsewhere.
VITALITY_FULL: float = 0.7


def vitality_label(vitality: float) -> str:
    """The level for vitality [0,1], injected into prompts as "how is my strength right now".

    Same scale as VITALITY_DAMAGE_DEFINITION (vitality = 1 − accumulated loss: 1 = full …
    0 = spent), thresholds 0.7/0.3. Neutral wording; ``vitality_line`` adds the per-person prefix.
    """
    if vitality <= 0:
        return "无生命力"
    if vitality > VITALITY_FULL:
        return "充沛"
    if vitality > 0.3:
        return "不济"
    return "将竭"


def render_condition(
    condition,
    *,
    now_step: int = 0,
    seconds_per_step: int = 3600,
    with_duration: bool = False,
) -> str:
    """The only renderer for conditions in prompts (duck-typed: needs .description/.since_step).

    No condition → ``""``, and the caller drops the whole line rather than render "condition: none".

    - ``with_duration=False`` → just the description, for roster marks.
    - ``with_duration=True`` → ``"<description>（已持续<duration>）"``, for profiles and adjudication,
      where the duration itself is evidence. Never step counts.
    """
    if condition is None:
        return ""
    description = (getattr(condition, "description", "") or "").strip()
    if not description:
        return ""
    if not with_duration:
        return description
    elapsed = max(now_step - int(getattr(condition, "since_step", 0) or 0), 0)
    if elapsed <= 0:
        return description
    return f"{description}（已持续{describe_duration(elapsed, seconds_per_step)}）"


# ============================================================================
# Memory Importance (build and runtime share one importance modality)
# ============================================================================
#
# Threshold anchors (see agent.memory_types.importance_level):
#   ≥ 0.9  → CRITICAL  (foundational event; never decays, never randomly forgotten)
#   ≥ 0.65 → HIGH      (markedly shapes judgment/attitude; gets recalled)
#   ≥ 0.4  → MEDIUM    (matters for a while; can be displaced by new experience)
#   else   → LOW       (atmosphere; may be forgotten)

MEMORY_IMPORTANCE_SCORE_DEFINITION: str = (
    "importance（记忆/事件的内在重量）: 浮点数 [0.0, 1.0]。"
    "0.0=完全无关琐事 | 0.4=一段时间内有影响,可被新经历替换 | "
    "0.65=显著影响判断与态度,会被回想数次 | 0.9=奠基性事件,多年后仍会被回想 | "
    "1.0=此生难忘的转折。"
)


# ============================================================================
# Need (need types / intensity / weight scales)
# ============================================================================

NEED_TYPE_DEFINITION: str = (
    "需求类型（Maslow 改编五档）:\n"
    "- physiological: 维持生理存续（进食/休息/治愈）\n"
    "- safety: 减少不确定性、规避风险、维持掌控感\n"
    "- social: 维系连接、归属、被理解、被在意\n"
    "- esteem: 维护尊严、地位、被认可的能力\n"
    "- self_actualization: 推进长远目标、表达真实自我、实现潜能"
)

NEED_INTENSITY_DEFINITION: str = (
    "intensity（需求紧迫度）: 浮点数 [0.0, 1.0]，该需求当前的内在驱力强度，"
    "与该 agent 是否能立即行动无关。"
    "0.0=已满足/无感 | 0.3=可感知但不紧迫 | 0.6=明显在意 | 0.8=主导思维 | 1.0=压倒一切"
)

NEED_WEIGHT_DEFINITION: str = (
    "weight（需求固有权重）: 浮点数 [0.1, 2.0]，该需求在该 agent 一生中的固有重要程度，"
    "塑造了 agent 是什么样的人。"
    "1.0=典型重要 | <1.0=该 agent 不太在意此类 | >1.0=该 agent 特别看重此类。"
    "intensity 反映此刻状态，weight 反映长期倾向。"
)


# ============================================================================
# Relation (relation field scales)
# ============================================================================

TRUST_RANGE: tuple[float, float] = (0.0, 1.0)
AFFECTION_RANGE: tuple[float, float] = (-1.0, 1.0)

TRUST_SCALE_DESCRIPTION: str = (
    "0.0=完全不信任 | 0.3=警惕 | 0.5=中立 | 0.7=信任 | 1.0=完全信任"
)

AFFECTION_SCALE_DESCRIPTION: str = (
    "-1.0=憎恶 | -0.3=反感 | 0.0=平淡 | 0.3=亲近 | 1.0=深爱"
)

RELATION_SCALE_LEGEND: str = (
    f"（信任度 trust ∈ [{TRUST_RANGE[0]}, {TRUST_RANGE[1]}]：{TRUST_SCALE_DESCRIPTION}；"
    f"好感度 affection ∈ [{AFFECTION_RANGE[0]}, {AFFECTION_RANGE[1]}]：{AFFECTION_SCALE_DESCRIPTION}）"
)

# The semantic core of labels (format + meaning + examples), shared by producers and readers.
# Producer-only constraints (age consistency / immutability / consistent wording / reverse flip)
# are appended by each schema after referencing this constant, not put here.
RELATION_LABEL_FORMAT: str = (
    "「关系类型:角色」格式——关系类型是结构性绑定（血缘 / 配偶 / 师徒 / 同事 / 邻居 / 同乡 等），"
    "角色是 A（发起方 from）在该关系中的位置；如「父子:儿子」表示 A 是 B（接收方 to）的儿子。"
    "无具体角色时只写关系类型（如「相识」「同事」「曾共事」），表示双方仅相识、在该关系中无具体角色。"
)

RELATION_LABEL_DESCRIPTION: str = f"labels（关系标签）: {RELATION_LABEL_FORMAT}"


def relation_legend() -> str:
    """Full relation field definitions (trust / affection / labels), injected when a prompt reads existing relation data."""
    return f"{RELATION_SCALE_LEGEND}；{RELATION_LABEL_DESCRIPTION}"


# ============================================================================
# External drive (ExternalGoal's two axes: drive_type category + urgency)
# ============================================================================
#
# urgency is the 4-level enum `core.interfaces.urgency.Urgency` (anchors in its module docstring).

EXTERNAL_DRIVE_TYPE_DEFINITION: str = (
    "drive_type（外部驱力类别）: "
    "threat=威胁安全 | authority=权威命令 | obligation=义务责任 | event=世界事件"
)

URGENCY_SCALE_DESCRIPTION: str = (
    "urgency（紧迫度）: 4 档枚举（字符串）。"
    "low=可留意（背景信号） | normal=值得响应（需排入计划） | "
    "high=应当优先（其它事都先放） | critical=必须立即响应（生死攸关 / 不可错过）"
)

# urgency → signal_strength: used when PerceptionLayer folds heterogeneous inputs into [0,1] salience
URGENCY_TO_STRENGTH: dict[Urgency, float] = {
    Urgency.LOW:      0.20,
    Urgency.NORMAL:   0.45,
    Urgency.HIGH:     0.75,
    Urgency.CRITICAL: 0.90,
}


def urgency_label(urgency: Urgency) -> str:
    """Translate an Urgency into its Chinese label for display in prompts (HIGH → "【紧急】")."""
    return _URGENCY_LABELS[urgency]


_URGENCY_LABELS: dict[Urgency, str] = {
    Urgency.LOW:      "【留意】",
    Urgency.NORMAL:   "【一般】",
    Urgency.HIGH:     "【紧急】",
    Urgency.CRITICAL: "【危急】",
}


# ============================================================================
# Severity (a Broadcast's objective event scale; bridged to the Urgency scale)
# ============================================================================

SEVERITY_SCALE_DESCRIPTION: str = (
    "severity（事件规模）: 枚举 {high, medium, low}，标注这件事客观上波及多大范围。"
    "high=全域显著事件 | medium=区域可感知事件 | low=背景级提示"
)


# ============================================================================
# Visible phenomena (phenomenon)
# ============================================================================

#: Shared by the world's two authors (EventSystem and DirectorChannel), whose broadcasts pass the
#: same ``parse_broadcast_spec`` filter: diverging wording would show flames for one author and
#: silently none for the other. Names come from the ``Phenomenon`` enum, never hand-written.
PHENOMENON_DEFINITION: str = (
    f"phenomenon（可见现象）: 取 {Phenomenon.prompt_choices()} 之一，"
    f"指这件事**当场看得见**的自然现象。它与 severity 正交——severity 说这事多大，"
    f"phenomenon 说它看上去什么样。**多数事情没有可见现象，none 是常态**，"
    f"只有当这件事本身就带着看得见的天气现象时才填，不要为了更生动硬填写一个。"
    f"{Phenomenon.sited_choices()} 必然发生在某一处，用它们就必须同时给出 location_scope；"
    f"只有 {Phenomenon.ambient_choices()} 才可以笼罩全域。"
)

#: Closed-world rule — must be injected into every prompt that produces assertions about what
#: happened in the world.
#:
#: The trap: pretraining knowledge overrides simulation state. A historical theme invites stating
#: real history as fact (two people executed) while the simulation has them alive; once embedded,
#: the whole cast reasons from a false premise.
#:
#: All three clauses are needed: a closed world (not "facts come from the context", which leaves
#: room for "I roughly know"); naming familiarity itself as the trap; and "I don't know" as a valid
#: way out, since a ban alone can't fill a gap. Keep it theme-neutral and position-neutral (it
#: lands in system or user, so "what you're given", not "above").
#: The in-character version rephrases it as a writing discipline: for the character, "you only
#: know this" is false. Don't mix the two.
CLOSED_WORLD_FACT_RULE: str = (
    "- 禁止捏造客观事实。**你所知的一切，只有本次给你的那些**；除此之外你一无所知。\n"
    "- **这个局面若让你熟悉，这个本身就是要防的**：无论它像你见过的哪段故事、"
    "哪类情节，都与此处无关。这里发生过什么、没发生过什么，只由所给的上下文决定。\n"
    "- 上下文没说的，就是**你不知道**。直言不知、含糊带过或避而不谈，都好过替世界补一个"
    "事实——**凡是别人听了会当真、会照着去做的断言，都属此列**。"
)

#: The in-character (first-person "I") version of the same rule — see CLOSED_WORLD_FACT_RULE.
CLOSED_WORLD_FACT_RULE_FIRST_PERSON: str = (
    "- 我只说所给材料里确有的事，**不替世界补一个客观事实**。\n"
    "- **这一幕若让我觉得很熟悉——像某段听过的故事、某类见过的情节——那份眼熟不作数**："
    "这里发生过什么、没发生过什么，只由所给的材料决定。\n"
    "- 材料没说的，我就是不知道。宁可照实说不知道、含糊带过或索性不提，"
    "也不编一个**别人听了会当真、会照着去做**的说法。"
)


# ── Future time points must be absolute ─────────────────────────────────────
#: A deictic time word ("明晨", "三日后") is persisted without its anchor, so each later reader
#: re-resolves it against their own "now" and an appointment never comes due. Resolve deixis at
#: write time. Only for scheduled future time points; plain description ("windy tonight") stays
#: as is, or the few appointments that matter get diluted.
#: This variant is for sites without a current time (compression summaries): it only keeps
#: existing dates. Don't add anything needing a "now" to it: a processor with no clock makes one up.
KEEP_ABSOLUTE_TIME_RULE: str = (
    "- 所给材料里已经写明日期的，照抄那个日期，不要改写成「次日 / 明晨 / 今夜」这类相对说法。"
)

#: Functional version. Inject it only where the user section gives the current world time, or the
#: model invents a date. Absolute time only: don't allow "明晨（date）", whose surviving deictic
#: word misreads when copied on that day.
#:
#: One constant serves traditional and modern calendars (Rule 7): the examples show one of each and
#: defer to the current-time line's format (WorldTime.time_label). The calendar line exists because
#: the model can't know DAYS_PER_MONTH and would guess from the real calendar at month end.
ABSOLUTE_TIME_RULE: str = """\
- 说到**一件约好的事在什么时候**（约见、期限、打算何时做某事），把那个**具体时间点**写出来，
  不要写「明晨 / 今夜 / 三日后」等等这类相对日期的说法，要说具体时间点。
- **具体时间点** = 所给的此刻时间里那套年月日的说法，加上一天里的时候。照着这样换：
      此刻那句写作「…六月初三，晚上八点」 → ✗ 明晨去见他 ✓ 六月初四清早去见他
      此刻那句写作「…3月4日，晚上八点」   → ✗ 三日后动手 ✓ 3月7日动手
  （只是示范怎么换。这两行说的是同一件事：年月日照所给的此刻时间那套写法来写，
    它怎么称呼日子你就怎么称呼，别换成另一套历法的写法。）
- 日子由所给的此刻时间推算出来，不凭印象另写一个；万一没有给出此刻时间，宁可不写日子，
  也绝不编一个。
- 单纯描摹时候、并没有约定挂在上面的（「今夜风大」），照常自然写，不必写日子。
""" + KEEP_ABSOLUTE_TIME_RULE + """
- 历法：一个月三十天，一年十二个月。"""

#: The in-character version of ABSOLUTE_TIME_RULE, phrased as a note-keeping habit.
ABSOLUTE_TIME_RULE_FIRST_PERSON: str = """\
- 说到**一件约好的事在什么时候**（约见、期限、打算何时做某事），我把那个**具体时间点**写出来，
  不要写「明晨 / 今夜 / 三日后」等等这类相对日期的说法，要说具体时间点。
- **具体时间点** = 所给的此刻时间里那套年月日的说法，加上一天里的时候。照着这样换：
      此刻那句写作「…六月初三，晚上八点」 → ✗ 明晨去见他 ✓ 六月初四清早去见他
      此刻那句写作「…3月4日，晚上八点」   → ✗ 三日后动手 ✓ 3月7日动手
  （只是示范怎么换。这两行说的是同一件事：年月日照所给的此刻时间那套写法来写，
    它怎么称呼日子我就怎么称呼，不改用另一套历法的写法。）
- 日子照所给的此刻时间推出来，不凭印象另写一个；万一没给此刻时间，我宁可不写日子，
  也绝不编一个。
- 单纯说到时候、并没有约定挂在上面的（「今夜风大」），照常自然说，不必写日子。
""" + KEEP_ABSOLUTE_TIME_RULE + """
- 历法：一个月三十天，一年十二个月。"""


# The severity→urgency translation membrane lives with the Severity enum in
# core.interfaces.severity (a type bridge, not a prompt scale).


def emotion_legend() -> str:
    """Full emotion field definitions, injected at once (for emotion outputs)."""
    return f"  - {EMOTION_INTENSITY_DEFINITION}\n  - {EMOTION_VALENCE_DEFINITION}"


# ============================================================================
# Text clipping (general prompt assembly)
# ============================================================================

#: Sentence breaks: ``clip_text`` backs off to the last one within the limit.
_SENTENCE_BREAKS = "。！？；;!?\n，,、"


def clip_text(text: str, limit: int) -> str:
    """Clip text to at most ``limit`` characters, preferring a sentence break; shorter text is returned as is.

    A half sentence reads like corrupted data. If backing off would discard more than half,
    hard-cut instead.

    Only for text with no declared limit (e.g. hand-written template descriptions). Don't clip
    fields whose length a prompt already fixes: that's a second source of truth.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(ch) for ch in _SENTENCE_BREAKS)
    return (head[:cut] if cut >= limit // 2 else head).rstrip(_SENTENCE_BREAKS + " ")


def strip_end_punct(text: str) -> str:
    """Strip trailing punctuation before embedding text in a template, so it doesn't end up doubled ("。。")."""
    return (text or "").strip().rstrip("。．.，,、；;！!？?")


# ============================================================================
# Perception signal rendering (one format for injecting "what I perceive right now")
# ============================================================================

# Per-channel cap on items injected into a prompt, so one channel can't crowd out the others.
# Every perception renderer reads it, including the decision prompt's own format.
SIGNAL_CAPS = {"visible": 10, "ambient": 10, "inbox": 20, "broadcast": 5}


# The decision prompt quotes it literally, so change it only here.
DECEASED_MARK = "已死亡"


def person_referent(name: str, gender: str = "", *marks: str) -> str:
    """The single renderer for one person in a roster: ``name（gender，marks…）``; empty parts are omitted.

    Gender must be present (see CLAUDE.md on gender): a model without it guesses, and a wrong guess
    in embedded memory never heals. ``marks`` share gender's bracket, never "李世民（男）（在场）".
    The profile form is ``SoulLayer.identity_text()``; choose by output shape.
    """
    parts = [p for p in (gender, *marks) if p]
    display = name or "某人"
    return f"{display}（{'，'.join(parts)}）" if parts else display


def render_npc(npc, *, index: int | None = None) -> str:
    """The single renderer for an Npc: ``name（gender，age，marks…）：description``, prefixed ``#N`` when indexed.

    Duck-typed like ``PerceivedNpc``. The head goes through ``person_referent``: in a roster an
    Npc is a person, not a thing. Pass ``index`` only under an IndexedRef contract; elsewhere a
    stray #N invites a made-up index.
    """
    who = getattr(npc, "identity", None)
    age = getattr(who, "age", None)
    marks = [f"{age}岁"] if age else []
    if getattr(npc, "busy", False):
        marks.append("正在忙")
    if condition := (getattr(npc, "condition", "") or "").strip():
        marks.append(condition)
    head = person_referent(getattr(who, "name", "") or "", getattr(who, "gender", "") or "", *marks)
    description = (getattr(who, "description", "") or "").strip()
    body = f"{head}：{description}" if description else head
    return f"#{index} {body}" if index is not None else body


def render_location(location) -> str:
    """Render a location (LocationView, duck-typed) as in-character text "name——description" (name alone if no description).

    None or an empty name renders as "某处", never the location_id.
    """
    name = ((getattr(location, "name", "") or "").strip()) if location is not None else ""
    if not name:
        return "某处"
    desc = (getattr(location, "description", "") or "").strip()
    rendered = f"{name}——{desc}" if desc else name
    # A perceived fact, not a bar on this person (see LocationView.is_public).
    return rendered if getattr(location, "is_public", True) else f"{rendered}（不对外开放但可前往）"


def render_entity(
    entity, *, owner_name: str | None = None, content: str = "", held_by_viewer: bool = False,
) -> str:
    """Render a world entity / item for a prompt: "name（attrs…）：description，内容：「…」".

    The single renderer for entities in prompts. Only non-default attributes appear. An empty name
    renders as "某物", never the entity_id. The caller passes ``owner_name`` and ``content``:
    readability depends on viewpoint (``WorldEntity.readable_by``).

    Takeable reads by relation: own hands "可交出", unowned "可取", someone else's only "由X持有"
    ("可取" would let the judge treat taking it as picking up an unowned item). ``held_by_viewer``
    is the reader in first person, the actor in adjudication.
    """
    name = ((getattr(entity, "name", "") or "").strip()) if entity is not None else ""
    if not name:
        return "某物"
    attrs: list[str] = []
    state = (getattr(entity, "state", "") or "").strip()
    if state and state != "intact":
        attrs.append(f"状态：{state}")
    if owner_name:
        attrs.append(f"由{owner_name}持有")
    if getattr(entity, "is_takeable", False):
        if held_by_viewer:
            attrs.append("可交出")
        elif not owner_name:
            attrs.append("可取")
    head = f"{name}（{'；'.join(attrs)}）" if attrs else name
    desc = (getattr(entity, "description", "") or "").strip()
    text = f"{head}：{desc}" if desc else head
    content = content.strip()
    return f"{text}，内容：「{content}」" if content else text


# ── Memory recency (gives the LLM a sense of how long ago something happened) ──
# Two axes that must not be mixed:
# - Duration words ("刚刚" / "几小时前") measure elapsed world time.
# - Calendar words ("今日" / "昨日" / "前日") count day boundaries crossed, which duration can't
#   answer (21 hours can already be yesterday). Derived from duration, "tomorrow" in a memory
#   slips a day and the appointment never comes due.
# Together they must read monotonically in one list. Don't use a calendar phrase that covers a
# duration word ("within today"): it would rank older yet include "hours ago", breaking
# MEMORY_ORDER_HINT. Buckets are coarse on purpose; don't fit the thresholds to one scenario.
_SUBDAY_BUCKETS: tuple[tuple[float, str], ...] = (
    (3600.0,     "刚刚"),    # < 1 hour
    (3600.0 * 6, "几小时前"),  # < 6 hours
)
# (max day boundaries crossed, label). Don't add vague buckets like "a day or two ago": they
# would unanchor "tomorrow" in memory text again.
_DAY_LABELS: tuple[tuple[int, str], ...] = (
    (0,  "今日早些时候"),
    (1,  "昨日"),
    (2,  "前日"),
    (6,  "数日前"),
    (30, "多日前"),
)
_RECENCY_REMOTE = "许久之前"


def _recency_label(elapsed_seconds: int, day_delta: int) -> str:
    """Below 6 hours duration words, above that day boundaries, so labels only ever age."""
    for threshold, label in _SUBDAY_BUCKETS:
        if elapsed_seconds < threshold:
            return label
    for max_delta, label in _DAY_LABELS:
        if day_delta <= max_delta:
            return label
    return _RECENCY_REMOTE


def render_memory(memory, *, now_step: int, seconds_per_step: int, world_start_second_of_day: int = 0) -> str:
    """Render one memory for a prompt; event memories get a coarse recency prefix.

    - Only event memories get the prefix: insights span time and summaries cover a period.
    - Render time only, never written back to stored_content (timestamps are embedding noise).
      No leading "- " or index; the caller adds those.

    ``world_start_second_of_day`` (seconds since midnight at world start) says which step
    midnight falls on; default 0. The start date cancels out, so the calendar stays out of here.
    """
    content = getattr(memory, "stored_content", "") or ""
    if getattr(memory, "kind", "event") != "event":
        return content
    created = getattr(memory, "created_step", now_step)
    return recency_prefix(
        now_step=now_step, ref_step=created, seconds_per_step=seconds_per_step,
        world_start_second_of_day=world_start_second_of_day,
    ) + content


def recency_prefix(
    *, now_step: int, ref_step: int, seconds_per_step: int, world_start_second_of_day: int = 0,
) -> str:
    """Render how long ago ``ref_step`` was as a bracketed recency word, e.g. "（昨日）".

    One vocabulary for everything past (memories, closed goals), so the LLM can place them on one
    timeline.
    """
    sps = max(1, seconds_per_step)
    now_seconds = world_start_second_of_day + now_step * sps
    elapsed_seconds = max(0, now_step - ref_step) * sps
    day_delta = now_seconds // 86400 - (now_seconds - elapsed_seconds) // 86400
    return f"（{_recency_label(elapsed_seconds, day_delta)}）"


# Ordering contract: memories arrive ordered by relevance, so every path that injects a memory list
# sorts it with order_memories_chrono and adds MEMORY_ORDER_HINT to the section header.
MEMORY_ORDER_HINT = "按发生先后排列，越靠后越近"


def order_memories_chrono(memories, *, key=None):
    """Stable sort by created_step ascending (oldest first); the one ordering point before injection.

    ``key`` extracts the memory from an item (e.g. ``(factual, experiential)`` pairs). None sorts
    first.
    """
    def _step(item):
        obj = key(item) if key is not None else item
        return getattr(obj, "created_step", 0) if obj is not None else 0

    return sorted(memories, key=_step)


def render_memory_lines(
    memories, *, now_step: int, seconds_per_step: int, world_start_second_of_day: int = 0,
) -> list[str]:
    """Render a memory list as chronological prompt lines; pair with MEMORY_ORDER_HINT."""
    return [
        render_memory(
            m, now_step=now_step, seconds_per_step=seconds_per_step,
            world_start_second_of_day=world_start_second_of_day,
        )
        for m in order_memories_chrono(memories)
    ]


class SituationVoice(str, Enum):
    """Grammatical person of the situation header, matching the role split in CLAUDE.md Prompt Design §1. Don't hard-code the strings at call sites."""

    FIRST  = "first"   # in-character first person (decision / emotion / goals / dialogue)
    SECOND = "second"  # in-character second person (the target's reaction in physical: "what you went through")
    THIRD  = "third"   # functional third-party judging / extraction


_CONDITION_LABEL: dict[SituationVoice, str] = {
    SituationVoice.FIRST:  "我此刻的处境",
    SituationVoice.SECOND: "你此刻的处境",
    SituationVoice.THIRD:  "处境",
}


def condition_line(
    condition,
    *,
    voice: SituationVoice,
    now_step: int = 0,
    seconds_per_step: int = 3600,
    lead: str = "\n",
) -> str:
    """The full "<person>此刻的处境：X" line; the only place a condition enters a prompt.

    No condition → "" including ``lead``, so callers needn't check. Pass ``lead=""`` for a
    standalone block.
    """
    text = render_condition(
        condition, now_step=now_step, seconds_per_step=seconds_per_step, with_duration=True,
    )
    return f"{lead}{_CONDITION_LABEL[voice]}：{text}" if text else ""


_VITALITY_PREFIX: dict[SituationVoice, str] = {
    SituationVoice.FIRST:  "我此刻的体力",
    SituationVoice.SECOND: "你此刻的体力",
    SituationVoice.THIRD:  "体力状况",
}


def vitality_line(
    vitality: float,
    *,
    voice: SituationVoice,
    lead: str = "\n",
    omit_when_full: bool = False,
) -> str:
    """The full "<person>此刻的体力：X" line; the only place vitality enters a prompt.

    Only cognition should set ``omit_when_full`` (a constant "full" is noise); the judge must
    always see the actor's vitality.
    """
    if omit_when_full and vitality > VITALITY_FULL:
        return ""
    return f"{lead}{_VITALITY_PREFIX[voice]}：{vitality_label(vitality)}"


def render_situation_header(situation, *, voice: SituationVoice) -> str:
    """Situation header: one line saying when and where it is, injected at the start of cognition and judge prompts.

    Goes first in the user content, not in system: it changes every step and would break the
    cacheable prefix. Either part may be missing and is then left out, never invented.
    """
    time_label = situation.time_label
    if situation.location_view is None:
        if not time_label:
            return ""
        return f"此刻是{time_label}。" if voice == SituationVoice.FIRST else f"当前时间：{time_label}。"
    place = render_location(situation.location_view)
    if voice == SituationVoice.FIRST:
        return f"我此刻在{place}，当前时间为{time_label}。" if time_label else f"我此刻在{place}。"
    return f"当前时间：{time_label}；当前地点：{place}。" if time_label else f"当前地点：{place}。"


def situation_location(situation) -> str:
    """The situation's location name for traces / audit; narrative text uses
    ``render_situation_header``."""
    view = getattr(situation, "location_view", None)
    return (view.name or "") if view is not None else ""


def render_perceived_signals(*, spatial, inbox, broadcasts) -> list[str]:
    """Render current perception channels as lines labeled by channel/source (theme-neutral).

    No leading `- `; the caller handles bullets. External pressure (ExternalGoal) isn't rendered
    here; callers append it.
    """
    lines: list[str] = []
    if getattr(spatial, "visible_agent_ids", None):
        people = []
        for aid in spatial.visible_agent_ids[:SIGNAL_CAPS["visible"]]:
            presence = spatial.visible_agents.get(aid)
            who = presence.identity if presence else None
            # Without the condition, someone bound and kneeling reads as an ordinary bystander.
            people.append(person_referent(
                getattr(who, "name", ""), getattr(who, "gender", ""),
                presence.condition if presence else "",
            ))
        lines.append(f"同处一地的人：{', '.join(people)}")
    for ev in getattr(spatial, "ambient_events", [])[:SIGNAL_CAPS["ambient"]]:
        lines.append(f"环境观察：{ev.content}")
    for msg in (inbox or [])[:SIGNAL_CAPS["inbox"]]:
        # The narrator's sender_name is already a narrative "unknown source"; no special case.
        sender = getattr(msg, "sender_name", None) or "某人"
        lines.append(f"收到来自{sender}的消息：{msg.content}")
    for bc in (broadcasts or [])[:SIGNAL_CAPS["broadcast"]]:
        lines.append(f"世界广播：{bc.content}")
    return lines
