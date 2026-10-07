"""How condition reaches the three prompts — and where it must NOT reach.

The field's value lies entirely in propagation: however correctly it's written, if the decision
prompt and the ruling scene can't read it, the captor will tie the same person up again a dozen
steps later (which is why it exists). Propagated too far, it becomes omniscience — so both sides
are locked here: it must reach where it should and must not reach where it shouldn't.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from agent.personality import PersonalityLayer, SoulLayer
from core.interfaces.condition import BodyCondition
from core.interfaces.perception import (
    LocationView,
    PerceivedIdentity,
    PerceivedPresence,
    SpatialPerception,
)

BOUND = BodyCondition(description="双手被反绑", source_agent_id="a1", since_step=10)


def _spatial(*, visible: dict[str, str], conditions: dict[str, str], step: int = 34) -> SpatialPerception:
    return SpatialPerception(
        location_id="loc",
        location_view=LocationView(name="玄武门", description="宫城北门"),
        world_time_label="六月初四，凌晨",
        current_step=step,
        visible_agents={
            aid: PerceivedPresence(
                PerceivedIdentity(name=n, gender="男"), conditions.get(aid, ""),
            )
            for aid, n in visible.items()
        },
    )


# ---------------------------------------------------------------------------
# Perception packet: derived every step, never accumulated
# ---------------------------------------------------------------------------

def test_condition_never_enters_the_persistent_name_cache() -> None:
    """Condition deliberately does NOT hang on PerceivedIdentity; it's the other half of
    PerceivedPresence.

    ``Agent.remember_agent`` accumulates PerceivedIdentity into the persistent name cache
    ``_known_agents`` (name and gender are fixed identity, so caching them is right). Condition is
    mutable — riding along would turn it into a never-updated false belief: the person was untied
    long ago, yet whoever remembers him sees him bound forever.

    Nesting (presence.identity / presence.condition) writes this boundary into the types: the cache
    takes ``PerceivedIdentity``, so condition simply can't get in.
    """
    assert not hasattr(PerceivedIdentity(name="李元吉"), "condition")

    sp = _spatial(visible={"a2": "李元吉"}, conditions={"a2": "双手被反绑"})
    presence = sp.visible_agents["a2"]
    assert presence.condition == "双手被反绑"
    assert presence.identity == PerceivedIdentity(name="李元吉", gender="男")


def test_spatial_perception_defaults_to_nobody_present() -> None:
    """Empty by default — presence is expressed by key existence, so "nobody" is an empty dict."""
    sp = SpatialPerception(
        location_id="loc", location_view=LocationView(name="玄武门", description=""),
        world_time_label="", current_step=1,
    )
    assert sp.visible_agents == {}
    assert sp.visible_agent_ids == []


def test_visible_agent_ids_is_a_derived_view_not_a_second_field() -> None:
    """``visible_agent_ids`` is a read-only derived view of ``visible``.

    Having no setter is deliberate: "who is here" can only be replaced wholesale (reset_visible +
    attach_presence), never half-changed while the identity/condition halves stay on the old set —
    exactly the drift that separate parallel fields keep producing.
    """
    sp = _spatial(visible={"a2": "李元吉", "a3": "常何"}, conditions={})
    assert sp.visible_agent_ids == ["a2", "a3"]          # key order = presence order
    with pytest.raises(AttributeError):
        sp.visible_agent_ids = ["a9"]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Decision prompt: own condition + list tags + information asymmetry boundary
# ---------------------------------------------------------------------------

def _decision_prompt(personality, packet) -> str:
    """Take only the user part. The system part is the fixed role frame and references these
    section titles by name ("动手前我先把【我所处的现实】里那几份名单看清楚"), so splitting on the
    title would cut there."""
    from agent.decision import DecisionEngine
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    engine = DecisionEngine(
        LLMRouter({scene: MockLLMProvider() for scene in LLMScene}),
    )
    system, user, _facts = engine._build_decision_prompt(personality, packet, [])
    return user


def _packet(spatial):
    """Minimal perception packet — shaped like _make_packet in tests/unit/test_decision.py."""
    from agent.need import NeedEvaluation, NeedState, NeedType
    from agent.perception import InternalContext, PerceptionPacket
    from agent.personality import EmotionState

    need_eval = NeedEvaluation(
        dominant_need=NeedType.SAFETY, scores={NeedType.SAFETY: 1.0},
        active_needs=[NeedState(NeedType.SAFETY, "safety", 1.0)],
        short_term_goals=[], long_term_goals=[], prompt_context="",
    )
    return PerceptionPacket(
        agent_id="a2", step=spatial.current_step, spatial=spatial, inbox=[], broadcasts=[],
        internal_context=InternalContext(
            emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
            dominant_need=NeedType.SAFETY,
            active_needs=list(need_eval.active_needs),
            short_term_goals=[], long_term_goals=[],
            factual_memories=[], experiential_memories=[], relevant_relations=[],
            need_evaluation=need_eval,
        ),
    )


def test_my_own_condition_leads_the_reality_section() -> None:
    """The first-person condition goes on the first line of 【我所处的现实】 (§3 strong position):
    it's the hardest constraint on "what can I do", and reframes every list read before it.
    Deliberately not in 【我是谁】 — that's the persona, and mixing it in would make "tied up" read
    like a personality trait."""
    p = PersonalityLayer(soul=SoulLayer(name="李元吉", gender="男", agent_id="a2"))
    p.set_condition(BOUND)
    prompt = _decision_prompt(p, _packet(_spatial(visible={}, conditions={})))

    # It's the very next line of that section — not "somewhere in that section".
    section = prompt.split("【我所处的现实】", 1)[1].splitlines()
    assert section[1] == "我此刻的处境：双手被反绑（已持续约1天）"
    # Not in the persona section: condition is condition, not character.
    persona = prompt.split("【我是谁】", 1)[1].split("【", 1)[0]
    assert "双手被反绑" not in persona


def test_a_present_person_wears_their_condition_in_the_roster() -> None:
    p = PersonalityLayer(soul=SoulLayer(name="李世民", gender="男", agent_id="a1"))
    prompt = _decision_prompt(
        p, _packet(_spatial(visible={"a2": "李元吉"}, conditions={"a2": "双手被反绑"})),
    )
    assert "李元吉（男，在场，双手被反绑）" in prompt


def test_an_absent_person_carries_no_condition_mark() -> None:
    """Information asymmetry boundary: whether he's still bound after I leave can only be a belief
    in my memory, not an always-fresh omniscient field. visible_conditions naturally only holds
    co-located people, so absent people on the list carry no tag."""
    p = PersonalityLayer(soul=SoulLayer(name="李世民", gender="男", agent_id="a1"))
    sp = _spatial(visible={}, conditions={})
    packet = _packet(sp)
    packet.internal_context.relevant_relations = []
    prompt = _decision_prompt(p, packet)
    assert "双手被反绑" not in prompt


# ---------------------------------------------------------------------------
# Read model
# ---------------------------------------------------------------------------

def test_read_model_carries_only_the_description() -> None:
    """The frontend only needs one line, "what he's like right now"; since_step / until_step /
    source are code-layer coordinates."""
    from interaction.models import _agent_states_from_snapshot
    from core.interfaces.snapshot import WorldSnapshot

    record = {
        "agent_id": "a2", "agent_name": "李元吉", "location_name": "玄武门",
        "condition": {"description": "双手被反绑", "source_agent_id": "a1",
                      "since_step": 10, "until_step": None},
    }
    snapshot = WorldSnapshot(
        world_id="w", step=34, timestamp=datetime(2026, 1, 1), agent_states={"a2": record},
    )
    states = _agent_states_from_snapshot(snapshot, actions=[])
    assert states["a2"].condition == "双手被反绑"


@pytest.mark.parametrize("record_condition", [None, {}, "双手被反绑", 42])
def test_read_model_degrades_to_empty_not_crash(record_condition: object) -> None:
    """Records on the replay path have been through a json.dumps(default=str) round-trip, so any
    shape can arrive."""
    from interaction.models import _agent_states_from_snapshot
    from core.interfaces.snapshot import WorldSnapshot

    snapshot = WorldSnapshot(
        world_id="w", step=1, timestamp=datetime(2026, 1, 1),
        agent_states={"a2": {"agent_name": "李元吉", "condition": record_condition}},
    )
    states = _agent_states_from_snapshot(snapshot, actions=[])
    assert states["a2"].condition == ""


def test_snapshot_payload_stays_json_native() -> None:
    """A BodyCondition instance must never appear in the snapshot payload — it feeds both
    persistence and the read model."""
    from agent.agent import snapshot_agent_state

    class _A:
        agent_id = "a2"
        is_active = True
        is_main_character = False
        personality = PersonalityLayer(soul=SoulLayer(name="李元吉", agent_id="a2"))

    a = _A()
    a.personality.set_condition(BOUND)
    payload = snapshot_agent_state(a)
    assert isinstance(payload["condition"], dict)
    json.dumps(payload, ensure_ascii=False)   # passes if it doesn't raise


# ---------------------------------------------------------------------------
# Perceived signals: the co-located list fed to perception emotion and short-term goal generation
# ---------------------------------------------------------------------------

def test_the_perceived_signal_roster_shows_a_standing_condition() -> None:
    """``render_perceived_signals`` is the fourth render point of the co-located list, and the
    easiest to miss — it isn't on any obvious decision/ruling/pressure path, yet it feeds both
    perception emotion and short-term goal generation.

    Perception emotion is grounding-first (emotion must rest on signals actually perceived right
    now, no imagining), so if the most conspicuous thing in the room isn't on the list, the model
    either ignores it or imagines it — both bad.
    """
    from core.prompts import render_perceived_signals

    lines = render_perceived_signals(
        spatial=_spatial(visible={"a2": "李元吉"}, conditions={"a2": "双手被反绑"}),
        inbox=[], broadcasts=[],
    )
    assert lines == ["同处一地的人：李元吉（男，双手被反绑）"]


def test_the_perceived_signal_roster_stays_bare_when_nobody_is_held() -> None:
    from core.prompts import render_perceived_signals

    lines = render_perceived_signals(
        spatial=_spatial(visible={"a2": "李元吉"}, conditions={}), inbox=[], broadcasts=[],
    )
    assert lines == ["同处一地的人：李元吉（男）"]


# ---------------------------------------------------------------------------
# Full list of injection sites: the list itself is the contract
# ---------------------------------------------------------------------------

def test_every_prompt_that_narrates_a_body_is_told_its_condition() -> None:
    """The full list of condition injection sites, pinned as one assertion.

    The field's value isn't in being written correctly but in being visible to every prompt that
    writes or rules on "what this body did". Missing one is a hole: that site writes a man tied
    behind his back sneaking or bursting out of the palace gate, text that is persisted, embedded,
    recalled for many steps, and can't heal itself.

    The list groups sites by their reason for injection; place new prompts accordingly. If none
    fits, it most likely shouldn't inject.
    """
    expected = {
        # ① Writes first- or third-person narrative in which the body's state is visible
        "agent/memory.py":                 "体验记忆散文（被 embedding）",
        "engine/executors/social.py":      "对白转录 + 两处会话记忆",
        "engine/executors/work.py":        "劳作自评 + 被打断的记忆",
        "engine/event.py":                 "事件编辑的人物简报（与体力并列）",
        "engine/executors/physical.py":    "物理裁决双方 + 承受方第一人称反应",
        # ② Rules on whether this body can do it
        "engine/executors/covert.py":      "潜行/窥探的可行性",
        "engine/scene.py":                 "裁决现场的在场者（四个 executor 共用）",
        "engine/world_pressure.py":        "外部压力评估的目标与同处者",
        "engine/presence.py":              "感知包的身份+处境唯一装配点",
        # ③ Chooses what to do next based on it
        "agent/decision.py":               "行动选择",
        "agent/need.py":                   "短期目标生成",
        "agent/agent.py":                  "中断权衡 + 反馈情绪",
        "agent/perception_emotion.py":     "感知情绪",
        "engine/director.py":              "导演人物菜单",
    }
    missing = [
        f"{path}（{why}）" for path, why in expected.items()
        if not _grep(("condition_line", "render_condition", "visible_conditions"), path)
    ]
    assert missing == [], "这些 prompt 该看见处境却看不见:\n" + "\n".join(missing)


def test_the_condition_line_has_exactly_one_definition() -> None:
    """The label and the "no condition omits the whole line" rule are defined once, in
    core/prompts.condition_line.

    This guards "change one place, change everywhere": hand-built f-strings across a dozen-plus
    injection sites and four persons would diverge.
    """
    # Look for the label literal (followed by a quote), not comments and docstrings that mention
    # these words in prose.
    definitions = _grep('此刻的处境"', "agent/", "engine/", "world/", "interaction/", "core/")
    non_core = [ln for ln in definitions if not ln.startswith("core/prompts.py:")]
    assert non_core == [], "标签在 core/prompts.py 之外被重新定义:\n" + "\n".join(non_core)


REPO = Path(__file__).resolve().parents[2]


def _grep(patterns, *paths) -> list[str]:
    """Same as test_condition_boundaries._grep: one -e per pattern (BSD grep's | is not
    alternation)."""
    pats = (patterns,) if isinstance(patterns, str) else patterns
    args = ["grep", "-rn", "--include=*.py"]
    for pat in pats:
        args += ["-e", pat]
    proc = subprocess.run([*args, *paths], cwd=REPO, capture_output=True, text=True)
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]
