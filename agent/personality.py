"""Personality models for Asamana agents."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Sequence

from agent.goals import GoalEntity, GoalStatus, is_live_goal, text_to_long_term_goal_entity
from core.interfaces.action import ActionType
from core.interfaces.condition import BodyCondition

if TYPE_CHECKING:
    from agent.need import NeedState, NeedType


class EmotionType(str, Enum):
    """Ekman-based emotion vocabulary covering common human and agent emotional states."""

    # Baseline
    NEUTRAL = "neutral"

    # Ekman 6 basic emotions
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"

    # Ekman extended
    CONTEMPT = "contempt"

    # Narrative-relevant additions
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    PRIDE = "pride"
    SHAME = "shame"
    JEALOUSY = "jealousy"
    FRUSTRATION = "frustration"

    @classmethod
    def prompt_list(cls) -> str:
        """Return slash-separated canonical values for use in LLM prompts."""
        return "/".join(e.value for e in cls)

    @property
    def label(self) -> str:
        return _EMOTION_LABELS[self]


_EMOTION_LABELS: dict[EmotionType, str] = {
    EmotionType.NEUTRAL: "平静",
    EmotionType.JOY: "喜悦",
    EmotionType.SADNESS: "悲伤",
    EmotionType.ANGER: "愤怒",
    EmotionType.FEAR: "恐惧",
    EmotionType.DISGUST: "厌恶",
    EmotionType.SURPRISE: "惊讶",
    EmotionType.CONTEMPT: "蔑视",
    EmotionType.ANTICIPATION: "期待",
    EmotionType.TRUST: "信任",
    EmotionType.PRIDE: "自豪",
    EmotionType.SHAME: "羞愧",
    EmotionType.JEALOUSY: "嫉妒",
    EmotionType.FRUSTRATION: "烦躁",
}


_ACTIVITY_STATUS_LABELS: dict[str, str] = {
    "idle": "空闲", "moving": "移动中", "talking": "交谈中",
    "resting": "休息中", "working": "行动中",
    "covert": "秘密行动中",
}


class AgentActivityStatus(str, Enum):
    """High-level activity modes used by the cognition loop."""

    IDLE = "idle"
    MOVING = "moving"
    TALKING = "talking"
    RESTING = "resting"
    WORKING = "working"
    COVERT = "covert"

    @property
    def label(self) -> str:
        return _ACTIVITY_STATUS_LABELS[self.value]


def activity_status_for(action_type: ActionType) -> AgentActivityStatus:
    """What an agent is visibly doing while an action of this type runs."""

    if action_type == ActionType.TALK:
        return AgentActivityStatus.TALKING
    if action_type == ActionType.REST:
        return AgentActivityStatus.RESTING
    if action_type == ActionType.MOVE:
        return AgentActivityStatus.MOVING
    if action_type == ActionType.COVERT:
        return AgentActivityStatus.COVERT
    # ERRAND lands here: status says what he is doing right now, just saying a few words; someone
    # else does the running. Don't give it its own status.
    return AgentActivityStatus.WORKING


class ActionStatus(str, Enum):
    """Lifecycle state for the current action."""

    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


from core.numeric import clamp  # noqa: E402


# Emotion intensity band edges, the single source (``summary()`` wording derives from them).
# Unlike needs, emotion has no decay dynamics (the LLM rewrites it every perception/feedback), so
# these are just wording bands for human readers, not derived thresholds.
#
# The scheduler's cadence gate reads EMOTION_STRONG_INTENSITY, not PRONOUNCED: LLM-written emotion
# skews dramatic and most agent-steps reach "明显", so gating on it would leave the gate always open.
EMOTION_STRONG_INTENSITY: float = 0.75      # ≥ this: "强烈"
EMOTION_PRONOUNCED_INTENSITY: float = 0.45  # ≥ this: "明显"; below: "轻微"


@dataclass
class EmotionState:
    """Structured emotional state used across cognition systems."""

    primary: EmotionType = EmotionType.NEUTRAL  # which emotion
    intensity: float = 0.2  # activation [0,1]: how strong, independent of kind
    valence: float = 0.0  # tone [-1,1]: >0 pleasant, <0 painful, ≈0 neutral
    triggered_by: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.primary, EmotionType):
            self.primary = parse_emotion_type(self.primary)

    def summary(self) -> str:
        if self.intensity >= EMOTION_STRONG_INTENSITY:
            strength = "强烈"
        elif self.intensity >= EMOTION_PRONOUNCED_INTENSITY:
            strength = "明显"
        else:
            strength = "轻微"
        if self.valence < -0.25:
            tone = "偏负面"
        elif self.valence > 0.25:
            tone = "偏正面"
        else:
            tone = "平稳"
        label = self.primary.label
        return f"{strength}的{label}，情绪基调{tone}"


# The one label for SoulLayer.secret in every prompt.
SECRET_LABEL = "秘密"


@dataclass(frozen=True)
class SoulLayer:
    """Stable identity traits that constrain all other cognition."""

    name: str
    role: str = ""
    agent_id: str = ""
    age: int = 0
    gender: str = ""
    core_traits: tuple[str, ...] = field(default_factory=tuple)
    core_values: tuple[str, ...] = field(default_factory=tuple)
    self_image: str = ""
    background: str = ""
    appearance: str = "" # outward appearance (build/dress/bearing); generated at build from the persona, fixed at runtime
    # Identity color (#RRGGBB), unique across the cast, frozen at build. Only the presentation
    # layer reads it; the engine never does.
    color: str = ""
    life_goal: str | None = None
    # A fact about this person that they keep from everyone else; usually empty. Fixed at build.
    # Rendered wherever their full profile is (to_prompt_context, the author layer), never in
    # perception, the directory, death notices or a bystander's brief.
    secret: str = ""
    hard_constraints: tuple[str, ...] = field(default_factory=tuple)
    # Innate needs, immutable identity: each need's lifelong weight, label and is_hidden. Their
    # ``intensity`` is only the build-time seed; the live value is StateLayer.need_intensities,
    # combined by PersonalityLayer.current_needs().
    innate_needs: tuple["NeedState", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_traits", _strings_to_tuple(self.core_traits))
        object.__setattr__(self, "core_values", _strings_to_tuple(self.core_values))
        object.__setattr__(self, "hard_constraints", _strings_to_tuple(self.hard_constraints))
        object.__setattr__(self, "innate_needs", tuple(self.innate_needs))

    def traits_text(self) -> str:
        """Render core traits as canonical prompt text. ``无`` when empty."""

        return "、".join(self.core_traits) or "无"

    def values_text(self) -> str:
        """Render core values as canonical prompt text. ``无`` when empty."""

        return "、".join(self.core_values) or "无"

    def constraints_text(self) -> str:
        """Render hard constraints as canonical prompt text. ``无`` when empty."""

        return "、".join(self.hard_constraints) or "无"

    def identity_text(self) -> str:
        """The one renderer for an identity header: ``名字，N岁，性别`` (empty fields omitted).

        Every full character profile goes through here, or fields like gender go missing. The
        split with ``core.prompts.person_referent`` follows output shape: a profile header comes
        from here, a roster line from there, even when the roster's caller holds the soul.
        """
        parts = [self.name or "某人"]
        if self.age:
            parts.append(f"{self.age}岁")
        if self.gender:
            parts.append(self.gender)
        return "，".join(parts)


@dataclass
class StateLayer:
    """Dynamic personality state updated by the cognition loop."""

    agent_id: str = ""
    step: int = 0
    emotion: EmotionState = field(default_factory=EmotionState)
    active_needs: list[str] = field(default_factory=list)
    dominant_need: str | None = None
    long_term_goals: list[str] = field(default_factory=list)
    short_term_goals: list[str] = field(default_factory=list)
    current_location: str = "unknown"
    activity_status: AgentActivityStatus = AgentActivityStatus.IDLE
    activity_target: str | None = None
    action_status: ActionStatus = ActionStatus.IDLE
    current_action: str | None = None
    action_remaining_steps: int = 0
    last_action: str | None = None
    last_action_result: str | None = None
    last_action_succeeded: bool | None = None
    short_term_goal_entities: list[GoalEntity] = field(default_factory=list)
    long_term_goal_entities: list[GoalEntity] = field(default_factory=list)
    need_intensities: dict[str, float] = field(default_factory=dict)
    vitality: float = 1.0
    # A condition persisting across steps (hands tied, drugged); see core.interfaces.condition.
    # Not merged into vitality: being restrained is not being injured, and they change independently.
    condition: "BodyCondition | None" = None
    # 0 = never; the scheduler's cadence gate treats that as starved. See mark_decided.
    last_decision_step: int = 0


class PersonalityLayer:
    """Stable and dynamic personality layers."""

    def __init__(self, soul: SoulLayer, state: StateLayer | None = None) -> None:
        self._soul = soul
        self._state = _copy_state(state or StateLayer(agent_id=soul.agent_id))
        if not self._state.agent_id:
            self._state.agent_id = soul.agent_id

    @property
    def soul(self) -> SoulLayer:

        return self._soul

    @property
    def innate_needs(self) -> tuple["NeedState", ...]:
        """Innate static needs (type/label/weight/is_hidden + seed intensity). Read-only."""

        return self._soul.innate_needs

    def current_needs(self) -> list["NeedState"]:
        """Full current needs: innate identity combined with ``state.need_intensities`` (hidden
        needs, absent there, keep their seed). The only entry point; scoring and display both
        read here instead of reassembling. Returns new objects."""
        from agent.need import NeedState  # local import: need.py imports personality at module load

        out: list["NeedState"] = []
        for innate in self._soul.innate_needs:
            out.append(NeedState(
                type=innate.type,
                label=innate.label,
                intensity=self._state.need_intensities.get(innate.type.value, innate.intensity),
                is_hidden=innate.is_hidden,
                weight=innate.weight,
            ))
        return out

    def need_label(self, need_type: "NeedType") -> str:
        """What the need means to this character, falling back to the generic description.
        In-character prompts always use this, so agents aren't flattened into one Maslow entry."""
        for innate in self._soul.innate_needs:
            if innate.type == need_type and innate.label:
                return innate.label
        return need_type.description

    @property
    def state(self) -> StateLayer:
        """Return a defensive snapshot of dynamic state."""

        return _copy_state(self._state)

    def restore_state(self, state: StateLayer) -> None:
        """Replace the internal state (used only for session restore)."""
        self._state = _copy_state(state)

    def to_prompt_context(self, *, include_goals: bool = True, include_emotion: bool = True) -> str:
        """Render the persona for a prompt, impersonally (no "你" / "我"): the caller's section
        header sets the grammatical person, so one block fits prompts of either person.

        ``include_goals=False`` when the caller supplies fresher goals (e.g.
        NeedEvaluation.prompt_context), so stale ones don't appear beside them.
        ``include_emotion=False`` for stages that generate the emotion: seeding them with the
        standing emotion is circular and makes the fresh reaction echo prior mood.
        """

        soul = self._soul
        state = self._state
        lines = [soul.identity_text() + "。"]
        if soul.role:
            lines.append(f"身份：{soul.role}。")
        if soul.background:
            lines.append(f"背景：{soul.background}")
        if soul.secret:
            lines.append(f"{SECRET_LABEL}：{soul.secret}")
        if soul.core_traits:
            lines.append(f"核心性格：{soul.traits_text()}。")
        if soul.core_values:
            lines.append(f"核心价值观：{soul.values_text()}。")
        if soul.self_image:
            lines.append(f"自我认知：{soul.self_image}")
        if soul.life_goal:
            lines.append(f"毕生追求：{soul.life_goal}")
        if include_emotion and state.emotion.intensity > 0.1:
            lines.append(f"此刻情绪：{state.emotion.summary()}。")
        if include_goals:
            recently_completed = [
                g.text for g in state.short_term_goal_entities if g.status == GoalStatus.COMPLETED
            ]
            if recently_completed:
                # Not "just now": the window is trimmed by count, so items can be many steps old,
                # and this caller has no world clock to say when (see NeedEngine.to_prompt_context).
                lines.append("我近期已经完成的事项：")
                lines.extend(f"  - {g}" for g in recently_completed)
            if state.short_term_goals:
                lines.append("短期目标（按先后，越靠后越新）：")
                lines.extend(f"  {i}. {g}" for i, g in enumerate(state.short_term_goals, 1))
        return "\n".join(lines)

    def apply_need_state(
        self,
        *,
        step: int,
        active_needs: Sequence[object],
        dominant_need: str | None,
        long_term_goals: Sequence[str],
        short_term_goal_entities: Sequence[GoalEntity],
    ) -> None:
        """Write active/dominant need + goals. Not intensities: their single owner is
        ``update_need_intensities``."""

        self.update_needs(step=step, active_needs=active_needs, dominant_need=dominant_need)
        self.set_short_term_goal_entities(short_term_goal_entities, step=step)
        self.set_long_term_goals(long_term_goals)

    def update_need_intensities(self, values: dict[str, float]) -> None:
        """Replace need intensities, the single source of need strength. Owned by the feedback
        layer (``NeedEngine.evolve_intensities``); separate from ``apply_need_state`` so it
        needn't rewrite dominant_need / goals."""
        self._state.need_intensities = dict(values)

    def mark_decided(self, step: int) -> None:
        """Record entry to the decision loop. The scheduler's cadence gate reads it to find who is
        starved, so write it for exactly the agents admitted to planning, not those that merely
        got a stub plan."""
        self._state.last_decision_step = step

    def set_short_term_goal_entities(self, entities: "Sequence[GoalEntity]", *, step: int | None = None) -> None:
        """Replace the short-term goal queue, the single source of truth. The ``short_term_goals``
        text view is derived from the live entities so the two never drift."""
        if step is not None:
            self._state.step = step
        self._state.short_term_goal_entities = list(entities)
        self._state.short_term_goals = [g.text for g in entities if is_live_goal(g)]

    def set_long_term_goals(self, texts: Sequence[str]) -> None:
        """Replace long-term goals from text; an unchanged text keeps its existing GoalEntity."""
        existing = {e.text: e for e in self._state.long_term_goal_entities}
        self._state.long_term_goal_entities = [
            existing[t] if t in existing else text_to_long_term_goal_entity(t)
            for t in texts
        ]
        self._state.long_term_goals = [e.text for e in self._state.long_term_goal_entities]

    def begin_action(
        self,
        *,
        step: int,
        description: str,
        activity_status: AgentActivityStatus,
        target: str | None = None,
        estimated_steps: int = 1,
    ) -> None:

        self._state.step = step
        self._state.activity_status = activity_status
        self._state.activity_target = target
        self._state.current_action = description
        self._state.action_status = ActionStatus.IN_PROGRESS
        self._state.action_remaining_steps = max(0, estimated_steps - 1)

    def complete_action(
        self,
        *,
        step: int,
        description: str,
        result_summary: str,
        succeeded: bool,
        emotion: EmotionState | None,
    ) -> None:
        """Apply the post-action state transition.

        ``emotion=None`` means the appraisal failed: keep the current mood, don't synthesize one
        (fallback tier 1). The action result is a known fact and is recorded as usual.
        """

        self.update_action_result(
            step=step,
            action=description,
            result=result_summary,
            succeeded=succeeded,
        )
        if emotion is not None:
            self._state.emotion = _copy_emotion(emotion)

    def apply_vitality_damage(self, delta: float) -> None:
        """Apply vitality change. Positive delta = damage; negative = recovery. Clamped [0, 1]."""
        self._state.vitality = max(0.0, min(1.0, self._state.vitality - delta))

    @property
    def is_alive(self) -> bool:
        return self._state.vitality > 0.0

    def set_condition(self, condition: "BodyCondition") -> None:
        """Impose or replace the condition (single slot, not a stack).

        Exception: a self-expiring condition can't displace one that needs outside help to remove.
        Otherwise the scrape noted after a failed struggle would replace "still tied up" and, on
        expiring, free him.
        """
        standing = self._state.condition
        if (
            standing is not None
            and standing.until_step is None
            and condition.until_step is not None
        ):
            return
        self._state.condition = condition

    def clear_condition(self) -> None:
        """Clear the condition (released by someone, broke free, or expired)."""
        self._state.condition = None

    def update_emotion(
        self,
        *,
        primary: EmotionType | str,
        intensity: float,
        valence: float,
        triggered_by: str | None = None,
    ) -> None:

        self._state.emotion = EmotionState(
            primary=primary,
            intensity=clamp(intensity, 0.0, 1.0),
            valence=clamp(valence, -1.0, 1.0),
            triggered_by=triggered_by,
        )

    def update_needs(
        self,
        *,
        step: int,
        active_needs: Sequence[object],
        dominant_need: object | None,
    ) -> None:
        """Apply the active and dominant needs."""

        self._state.step = step
        self._state.active_needs = [_need_value(need) for need in active_needs]
        self._state.dominant_need = _need_value(dominant_need) if dominant_need is not None else None

    def update_location(
        self,
        *,
        step: int | None = None,
        location: str,
    ) -> None:
        """Apply the agent's location; arriving somewhere leaves the agent idle there."""

        if step is not None:
            self._state.step = step
        self._state.current_location = location
        self._state.activity_status = AgentActivityStatus.IDLE
        self._state.activity_target = None

    def update_action_status(
        self,
        *,
        step: int | None = None,
        status: ActionStatus,
        current_action: str | None = None,
        remaining_steps: int = 0,
    ) -> None:
        """Apply the action lifecycle state."""

        if step is not None:
            self._state.step = step
        self._state.action_status = status
        self._state.current_action = current_action
        self._state.action_remaining_steps = max(0, remaining_steps)

    def update_action_result(
        self,
        *,
        step: int,
        action: str,
        result: str,
        succeeded: bool,
    ) -> None:
        """Apply the last action's result."""

        self._state.step = step
        self._state.last_action = action
        self._state.last_action_result = result
        self._state.last_action_succeeded = succeeded
        self._state.current_action = None
        self._state.action_remaining_steps = 0
        self._state.action_status = ActionStatus.COMPLETED if succeeded else ActionStatus.FAILED
        self._state.activity_status = AgentActivityStatus.IDLE
        self._state.activity_target = None


def parse_emotion_type(raw: str | EmotionType) -> EmotionType:
    """Normalize an emotion label to EmotionType (in ``EmotionState.__post_init__`` and at LLM
    output boundaries); NEUTRAL if unrecognized."""
    if isinstance(raw, EmotionType):
        return raw
    lowered = str(raw).lower().strip()
    try:
        return EmotionType(lowered)
    except ValueError:
        pass
    _synonyms: dict[str, EmotionType] = {
        # Chinese labels
        "平静": EmotionType.NEUTRAL, "镇定": EmotionType.NEUTRAL, "沉着": EmotionType.NEUTRAL, "冷静": EmotionType.NEUTRAL,
        "专注": EmotionType.NEUTRAL, "专心": EmotionType.NEUTRAL,
        "喜悦": EmotionType.JOY, "满足": EmotionType.JOY, "愉悦": EmotionType.JOY,
        "投入": EmotionType.JOY, "热情": EmotionType.JOY, "兴奋": EmotionType.JOY,
        "悲伤": EmotionType.SADNESS, "沮丧": EmotionType.SADNESS, "压抑": EmotionType.SADNESS,
        "愤怒": EmotionType.ANGER, "怒": EmotionType.ANGER, "愤": EmotionType.ANGER,
        "恐惧": EmotionType.FEAR, "害怕": EmotionType.FEAR, "惊恐": EmotionType.FEAR,
        "警惕": EmotionType.FEAR, "担忧": EmotionType.FEAR, "焦虑": EmotionType.FEAR, "忧虑": EmotionType.FEAR,
        "厌恶": EmotionType.DISGUST, "反感": EmotionType.DISGUST,
        "惊讶": EmotionType.SURPRISE, "震惊": EmotionType.SURPRISE,
        "蔑视": EmotionType.CONTEMPT, "轻蔑": EmotionType.CONTEMPT, "鄙视": EmotionType.CONTEMPT,
        "期待": EmotionType.ANTICIPATION, "憧憬": EmotionType.ANTICIPATION,
        "信任": EmotionType.TRUST, "信赖": EmotionType.TRUST,
        "自豪": EmotionType.PRIDE, "骄傲": EmotionType.PRIDE,
        "羞愧": EmotionType.SHAME, "惭愧": EmotionType.SHAME, "羞耻": EmotionType.SHAME,
        "嫉妒": EmotionType.JEALOUSY, "妒忌": EmotionType.JEALOUSY,
        "烦躁": EmotionType.FRUSTRATION,
        # English synonyms
        "calm": EmotionType.NEUTRAL, "steady": EmotionType.NEUTRAL, "focused": EmotionType.NEUTRAL,
        "happy": EmotionType.JOY, "content": EmotionType.JOY, "pleased": EmotionType.JOY,
        "satisfied": EmotionType.JOY, "engaged": EmotionType.JOY,
        "sad": EmotionType.SADNESS, "melancholy": EmotionType.SADNESS, "depressed": EmotionType.SADNESS,
        "rage": EmotionType.ANGER, "furious": EmotionType.ANGER, "angry": EmotionType.ANGER,
        "afraid": EmotionType.FEAR, "scared": EmotionType.FEAR, "apprehensive": EmotionType.FEAR,
        "fearful": EmotionType.FEAR, "guarded": EmotionType.FEAR, "anxious": EmotionType.FEAR,
        "revolted": EmotionType.DISGUST, "disgusted": EmotionType.DISGUST,
        "shocked": EmotionType.SURPRISE, "astonished": EmotionType.SURPRISE, "shock": EmotionType.SURPRISE,
        "grateful": EmotionType.TRUST, "gratitude": EmotionType.TRUST,
        "relieved": EmotionType.JOY, "relief": EmotionType.JOY,
        "warm": EmotionType.TRUST, "warmth": EmotionType.TRUST,
        "disdainful": EmotionType.CONTEMPT, "scornful": EmotionType.CONTEMPT,
        "hopeful": EmotionType.ANTICIPATION, "wistful": EmotionType.ANTICIPATION, "optimistic": EmotionType.ANTICIPATION,
        "trusting": EmotionType.TRUST,
        "proud": EmotionType.PRIDE,
        "ashamed": EmotionType.SHAME, "embarrassed": EmotionType.SHAME,
        "envious": EmotionType.JEALOUSY, "envy": EmotionType.JEALOUSY, "jealous": EmotionType.JEALOUSY,
        "frustrated": EmotionType.FRUSTRATION,
    }
    return _synonyms.get(lowered, EmotionType.NEUTRAL)


def _strings_to_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values if str(value))


def _copy_emotion(emotion: EmotionState) -> EmotionState:
    return EmotionState(
        primary=emotion.primary,
        intensity=emotion.intensity,
        valence=emotion.valence,
        triggered_by=emotion.triggered_by,
    )


def _copy_state(state: StateLayer) -> StateLayer:
    return StateLayer(
        agent_id=state.agent_id,
        step=state.step,
        emotion=_copy_emotion(state.emotion),
        active_needs=list(state.active_needs),
        dominant_need=state.dominant_need,
        long_term_goals=list(state.long_term_goals),
        short_term_goals=list(state.short_term_goals),
        current_location=state.current_location,
        activity_status=state.activity_status,
        activity_target=state.activity_target,
        action_status=state.action_status,
        current_action=state.current_action,
        action_remaining_steps=state.action_remaining_steps,
        last_action=state.last_action,
        last_action_result=state.last_action_result,
        last_action_succeeded=state.last_action_succeeded,
        # Copy the entities, not just the list: NeedEngine mutates what it reads from .state.
        short_term_goal_entities=[replace(g) for g in state.short_term_goal_entities],
        long_term_goal_entities=[replace(g) for g in state.long_term_goal_entities],
        need_intensities=dict(state.need_intensities),
        vitality=state.vitality,
        # Frozen, so aliasing is fine. Don't drop this line: .state goes through here, and without
        # it the field silently reads None everywhere.
        condition=state.condition,
        last_decision_step=state.last_decision_step,
    )


def _need_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)
