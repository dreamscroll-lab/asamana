"""The relation legend is a constant: every prompt that shows relations gives it once in its system
message, in a 【说明】 section before the output, so it stays in the cached prefix instead of being
resent beside the per-call relation lines."""

from __future__ import annotations

import pytest

from agent.need import NeedType, NeedEngine, _SHORT_TERM_GOAL_SYSTEM
from agent.perception_emotion import build_perception_emotion_prompt
from agent.personality import PersonalityLayer
from agent.relation import PerceivedRelation
from core.prompts import relation_legend
from engine.executors.social import _DIALOGUE_SYSTEM, SocialExecutor
from engine.world_pressure import _SYSTEM_PROMPT as _PRESSURE_SYSTEM
from tests.unit.test_condition_reaches_every_prompt import _agent, _Capture, _directory
from tests.unit.test_perception_emotion_prompt import _personality_stub, _rel, _spatial
from tests.unit.test_soul_secret import _soul
from tests.unit.test_world_pressure_prompt import _ev, _msg, _stub_agent

LEGEND = relation_legend()


def _explained_before_output(system: str) -> bool:
    explained = system.index("【说明】")
    return explained < system.index(LEGEND) and "【输出】" not in system[:explained]


def _relation() -> PerceivedRelation:
    return PerceivedRelation(trust=0.2, affection=-0.3, target_agent_id="a3", target_agent_name="李世民",
                             target_agent_gender="男", labels=["兄弟:哥哥"])


@pytest.mark.asyncio
async def test_short_term_goals() -> None:
    llm = _Capture('{"goals": []}')
    await NeedEngine(llm_router=llm)._generate_short_term_goals(  # type: ignore[arg-type]  # noqa: SLF001
        dominant_need=NeedType.SAFETY, personality=PersonalityLayer(soul=_soul()),
        perceived_relations=[_relation()],
    )
    system, user = (m.content for m in llm.messages[-1])

    assert LEGEND in _SHORT_TERM_GOAL_SYSTEM
    assert LEGEND in system and _explained_before_output(system)
    assert LEGEND not in user and "李世民" in user


def test_perception_emotion() -> None:
    built = build_perception_emotion_prompt(
        personality=_personality_stub(), spatial=_spatial(visible={"a1": "长孙无忌"}),
        inbox=[], broadcasts=[], external_goals=[],
        perceived_relations=[_rel("a1", "长孙无忌", trust=0.9, affection=0.8, labels=["挚友"])],
    )
    assert built is not None
    assert LEGEND in built.system and "【说明】" in built.system
    assert LEGEND not in built.user and "长孙无忌" in built.user


def test_world_pressure() -> None:
    ev = _ev()
    signals = ev._signals(  # noqa: SLF001
        inbox=[_msg("速来东宫", sender_id="a_shimin", sender_name="李世民")],
        broadcasts=[], ambient=[],
        sender_relations={"a_shimin": "[兄弟:敌对]，信任度0.10、好感度-0.60"}, visible_ids=set(),
    )
    user = ev._build_agent_prompt(  # noqa: SLF001
        _stub_agent(name="李建成", role="太子", location="donggong"),
        signals=signals, situation_header="当前时间：辰时；当前地点：东宫。",
    )
    assert LEGEND in _PRESSURE_SYSTEM and "【说明】" in _PRESSURE_SYSTEM
    assert LEGEND not in user and "信任度0.10" in user


@pytest.mark.asyncio
async def test_narrated_dialogue() -> None:
    llm = _Capture('{"dialogue":[{"speaker":1,"line":"…"}],"observation":""}')
    ctx = {"relation": _relation(), "about_other": [], "about_topic": [], "recent": []}
    await SocialExecutor(llm, _directory(), seconds_per_step=3600)._llm_full_dialogue(  # noqa: SLF001
        initiator=_agent("a1", "李世民", condition=None), target=_agent("a2", "李元吉", condition=None),
        initiator_name="李世民", target_name="李元吉", purpose="逼问东宫往来", expected_outcome="", turns=2,
        initiator_ctx=dict(ctx), target_ctx=dict(ctx), scene="", now_step=24,
    )
    system, user = (m.content for m in llm.messages[-1])

    assert system == _DIALOGUE_SYSTEM and _explained_before_output(system)
    assert LEGEND not in user and "信任度" in user
