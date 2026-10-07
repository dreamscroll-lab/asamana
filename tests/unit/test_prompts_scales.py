"""Unit tests for core/prompts.py, where LLM prompt scale legends are centralized.

Ensures:
- the centralized constants carry the expected scale content
- urgency_label's level thresholds are correct
- re-exports from agent/personality.py + agent/relation.py still work
- the urgency scale is injected into world_pressure._SYSTEM_PROMPT from the shared constant
"""

from __future__ import annotations


from core.interfaces.urgency import Urgency
from core.prompts import (
    clip_text,
    AFFECTION_SCALE_DESCRIPTION,
    EMOTION_INTENSITY_DEFINITION,
    EMOTION_VALENCE_DEFINITION,
    RELATION_SCALE_LEGEND,
    TRUST_SCALE_DESCRIPTION,
    URGENCY_SCALE_DESCRIPTION,
    URGENCY_TO_STRENGTH,
    emotion_legend,
    urgency_label,
)


# ---------------------------------------------------------------------------
# Constant content
# ---------------------------------------------------------------------------


def test_emotion_intensity_definition_contains_anchor_scale() -> None:
    """EMOTION_INTENSITY_DEFINITION must include the 5 anchor levels (0.0/0.2/0.5/0.8/1.0)."""
    for anchor in ("0.0", "0.2", "0.5", "0.8", "1.0"):
        assert anchor in EMOTION_INTENSITY_DEFINITION


def test_emotion_valence_definition_describes_polarity() -> None:
    assert "[-1.0, 1.0]" in EMOTION_VALENCE_DEFINITION
    assert "愉快" in EMOTION_VALENCE_DEFINITION
    assert "痛苦" in EMOTION_VALENCE_DEFINITION


def test_trust_scale_covers_full_range() -> None:
    for anchor in ("0.0", "0.3", "0.5", "0.7", "1.0"):
        assert anchor in TRUST_SCALE_DESCRIPTION


def test_affection_scale_covers_negative_to_positive() -> None:
    for anchor in ("-1.0", "-0.3", "0.0", "0.3", "1.0"):
        assert anchor in AFFECTION_SCALE_DESCRIPTION


def test_relation_scale_legend_includes_both() -> None:
    assert "信任度" in RELATION_SCALE_LEGEND
    assert "好感度" in RELATION_SCALE_LEGEND
    assert TRUST_SCALE_DESCRIPTION in RELATION_SCALE_LEGEND
    assert AFFECTION_SCALE_DESCRIPTION in RELATION_SCALE_LEGEND


def test_urgency_scale_describes_4_enum_levels() -> None:
    """The urgency description must contain the 4 enum labels (low/normal/high/critical)."""
    for label in ("low", "normal", "high", "critical"):
        assert label in URGENCY_SCALE_DESCRIPTION


def test_urgency_to_strength_anchors_aligned() -> None:
    """URGENCY_TO_STRENGTH anchors must match the explanation in URGENCY_SCALE_DESCRIPTION."""
    assert URGENCY_TO_STRENGTH[Urgency.LOW]      == 0.20
    assert URGENCY_TO_STRENGTH[Urgency.NORMAL]   == 0.45
    assert URGENCY_TO_STRENGTH[Urgency.HIGH]     == 0.75
    assert URGENCY_TO_STRENGTH[Urgency.CRITICAL] == 0.90


# ---------------------------------------------------------------------------
# urgency_label(): mapping onto the four enum levels
# ---------------------------------------------------------------------------


def test_urgency_label_maps_each_enum_to_chinese_tag() -> None:
    assert urgency_label(Urgency.LOW)      == "【留意】"
    assert urgency_label(Urgency.NORMAL)   == "【一般】"
    assert urgency_label(Urgency.HIGH)     == "【紧急】"
    assert urgency_label(Urgency.CRITICAL) == "【危急】"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_emotion_legend_includes_both_fields() -> None:
    legend = emotion_legend()
    assert EMOTION_INTENSITY_DEFINITION in legend
    assert EMOTION_VALENCE_DEFINITION in legend


# ---------------------------------------------------------------------------
# Integration check: world_pressure._SYSTEM_PROMPT includes the shared urgency scale
# ---------------------------------------------------------------------------


def test_world_pressure_system_prompt_contains_urgency_anchors() -> None:
    """When the LLM outputs urgency it must see the 4-level enum legend, or it can
    only guess from priors."""
    from engine.world_pressure import _SYSTEM_PROMPT
    # The urgency description must list the 4 enum labels
    label_count = sum(1 for label in ("low", "normal", "high", "critical") if label in _SYSTEM_PROMPT)
    assert label_count >= 3, f"_SYSTEM_PROMPT 应含至少 3 档 urgency 枚举标签;实际仅 {label_count}"
    # The whole URGENCY_SCALE_DESCRIPTION is injected
    assert URGENCY_SCALE_DESCRIPTION in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# clip_text: the length guard for long text entering prompts
# ---------------------------------------------------------------------------


def test_clip_text_leaves_short_text_alone() -> None:
    assert clip_text("李渊次子，少年统军。", 100) == "李渊次子，少年统军。"
    assert clip_text("  两头有空白  ", 100) == "两头有空白"
    assert clip_text("", 10) == ""


def test_clip_text_breaks_on_a_sentence_not_mid_word() -> None:
    """A hard cut leaves half a sentence ("…视为眼中钉，"), which reads like corrupted data."""
    text = "李渊次子，自幼随父起兵，少年统军平定四方。半生戎马换来不世之功。"

    clipped = clip_text(text, 20)

    assert len(clipped) <= 20
    assert clipped == "李渊次子，自幼随父起兵"      # cut at the second comma, without keeping the comma
    assert not clipped.endswith(("，", "。", "、"))


def test_clip_text_falls_back_to_a_hard_cut_when_the_break_is_too_early() -> None:
    """Backing off too far leaves only the opening, less information than half a sentence; hard-cut
    instead."""
    text = "甲，" + "无标点的一长段文字" * 5

    clipped = clip_text(text, 20)

    assert len(clipped) == 20
    assert clipped.startswith("甲，无标点")


# ---------------------------------------------------------------------------
# render_situation_header: either axis may be missing; a missing axis isn't rendered at all
# ---------------------------------------------------------------------------


def _situation(*, place: str | None, time_label: str = ""):
    from core.interfaces.perception import LocationView, Situation

    return Situation(
        location_view=LocationView(name=place, description="") if place else None,
        time_label=time_label,
    )


def test_situation_header_renders_both_axes() -> None:
    from core.prompts import SituationVoice, render_situation_header

    both = _situation(place="皇城", time_label="武德九年，六月初三，凌晨零点")
    assert render_situation_header(both, voice=SituationVoice.FIRST) == (
        "我此刻在皇城，当前时间为武德九年，六月初三，凌晨零点。"
    )
    assert render_situation_header(both, voice=SituationVoice.THIRD) == (
        "当前时间：武德九年，六月初三，凌晨零点；当前地点：皇城。"
    )


def test_situation_header_with_time_only_does_not_invent_a_place() -> None:
    """Prompts like reflection have only the time axis (looking back on recent events has nothing to
    do with where you are).

    Going through render_location(None) would render "我此刻在某处", inventing a location, and that
    sentence would feed an in-character prompt told "don't fabricate".
    """
    from core.prompts import SituationVoice, render_situation_header

    time_only = _situation(place=None, time_label="武德九年，六月初三，凌晨零点")

    first = render_situation_header(time_only, voice=SituationVoice.FIRST)
    third = render_situation_header(time_only, voice=SituationVoice.THIRD)

    assert first == "此刻是武德九年，六月初三，凌晨零点。"
    assert third == "当前时间：武德九年，六月初三，凌晨零点。"
    assert "某处" not in first and "某处" not in third


def test_situation_header_with_place_only_does_not_invent_a_time() -> None:
    from core.prompts import SituationVoice, render_situation_header

    place_only = _situation(place="皇城")

    assert render_situation_header(place_only, voice=SituationVoice.FIRST) == "我此刻在皇城。"
    assert render_situation_header(place_only, voice=SituationVoice.THIRD) == "当前地点：皇城。"


def test_situation_header_is_empty_when_both_axes_are_unknown() -> None:
    """The extreme state before perceive: don't fabricate a header."""
    from core.prompts import SituationVoice, render_situation_header

    assert render_situation_header(_situation(place=None), voice=SituationVoice.FIRST) == ""
    assert render_situation_header(_situation(place=None), voice=SituationVoice.THIRD) == ""


# ---------------------------------------------------------------------------
# vitality_line: the single place vitality enters a prompt
# ---------------------------------------------------------------------------


def test_vitality_line_uses_the_voice_of_the_prompt_it_lands_in() -> None:
    from core.prompts import SituationVoice, vitality_line

    assert vitality_line(0.5, voice=SituationVoice.FIRST, lead="") == "我此刻的体力：不济"
    assert vitality_line(0.5, voice=SituationVoice.SECOND, lead="") == "你此刻的体力：不济"
    assert vitality_line(0.5, voice=SituationVoice.THIRD, lead="") == "体力状况：不济"


def test_vitality_line_omits_a_full_tank_only_where_the_caller_asked() -> None:
    """Not omitted by default, so nobody casually hides vitality from the judge."""
    from core.prompts import SituationVoice, vitality_line

    assert vitality_line(1.0, voice=SituationVoice.FIRST, omit_when_full=True) == ""
    assert "充沛" in vitality_line(1.0, voice=SituationVoice.THIRD)          # not omitted by default
    # "不济" / "将竭" / "无生命力" always appear, even with omit on.
    for v in (0.5, 0.1, 0.0):
        assert vitality_line(v, voice=SituationVoice.FIRST, omit_when_full=True)


def test_vitality_line_lead_is_dropped_together_with_the_line() -> None:
    """When the whole line disappears, lead goes with it; otherwise the call site gets a blank
    line."""
    from core.prompts import SituationVoice, vitality_line

    assert vitality_line(1.0, voice=SituationVoice.FIRST, lead="\n", omit_when_full=True) == ""
    assert vitality_line(0.5, voice=SituationVoice.FIRST, lead="\n").startswith("\n")


def test_vitality_full_threshold_is_shared_by_label_and_line() -> None:
    """Both places share one number: change the tier in only one and a "不济" (flagging) person loses
    this line."""
    from core.prompts import VITALITY_FULL, SituationVoice, vitality_label, vitality_line

    assert vitality_label(VITALITY_FULL) != "充沛"          # the boundary isn't full
    assert vitality_line(VITALITY_FULL, voice=SituationVoice.FIRST, omit_when_full=True)
    assert vitality_label(VITALITY_FULL + 0.01) == "充沛"
    assert vitality_line(VITALITY_FULL + 0.01, voice=SituationVoice.FIRST, omit_when_full=True) == ""
