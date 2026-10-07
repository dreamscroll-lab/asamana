"""Every prompt that should see condition really does — verified by rendering, not by the file
mentioning it.

Why a separate file: condition only matters if every prompt that writes or rules on "what this body
did" sees it (see test_condition_propagation.test_every_prompt_that_narrates_a_body_is_told_its_condition).

These cases deliberately render the real prompt and then assert, rather than grepping for
`condition_line`: grep stays green when the variable is still defined but the interpolation slot
has been deleted.
"""

from __future__ import annotations

import pytest

from agent.personality import PersonalityLayer, SoulLayer
from core.interfaces.condition import BodyCondition

BOUND = BodyCondition(description="双手被反绑", source_agent_id="a1", since_step=0)
MARK = "双手被反绑"


class _Capture:
    """Records what it was asked and answers with a fixed line."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.messages: list = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001
        self.messages.append(messages)

        class _R:
            content = self.payload

        return _R()

    async def complete_with_retry(self, scene, messages, **kwargs):  # noqa: ANN001
        return await self.complete(scene, messages, **kwargs)

    def text(self) -> str:
        return "\n".join(m.content for m in self.messages[-1])


class _Rel:
    async def load_existing(self, _):  # noqa: ANN001
        return type("R", (), {"labels": [], "trust_objective": 0.5, "affection_objective": 0.0})()

    async def perceive_existing(self, *_a, **_kw):
        return type("P", (), {"labels": [], "trust": 0.5, "affection": 0.0})()


def _agent(agent_id: str = "a2", name: str = "李元吉", *, condition=BOUND):
    class _A:
        def __init__(self) -> None:
            self.agent_id = agent_id
            self.is_main_character = False
            self.is_active = True
            self.personality = PersonalityLayer(
                soul=SoulLayer(name=name, gender="男", agent_id=agent_id),
            )
            if condition is not None:
                self.personality.set_condition(condition)
            self.relation_system = _Rel()

    return _A()


def _directory():
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    return LiveWorldDirectory.from_agents({}, EnvironmentSystem())


# ---------------------------------------------------------------------------
# Rules on whether this body can do it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_covert_judge_knows_the_sneaker_is_bound() -> None:
    """"Acting unnoticed" depends most on bodily state: a bound man sneaking or an immobilized one
    spying simply doesn't work."""
    from engine.executors.covert import CovertExecutor

    llm = _Capture('{"reason":"x","achieved":false,"detected":true,"outcome":"o","fact":"f"}')
    ex = CovertExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._judge(  # noqa: SLF001
        _agent(), purpose="潜到廊下偷听", duration_label="约1小时",
        expected_outcome="", scene="", now_step=24,
    )
    assert MARK in llm.text()


# ---------------------------------------------------------------------------
# Writes narrative text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_self_appraisal_knows_the_worker_is_bound() -> None:
    """WORK's self-assessment produces fact / outcome / observation text, all persisted."""
    from engine.executors.work import WorkExecutor

    llm = _Capture('{"fact":"f","success":true,"outcome":"o","observation":"ob","why":""}')
    ex = WorkExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._generate_outcome(  # noqa: SLF001
        agent=_agent(), purpose="誊抄名册", duration_label="约2小时", now_step=24,
    )
    assert MARK in llm.text()


@pytest.mark.asyncio
async def test_social_memory_summary_knows_the_speaker_is_bound() -> None:
    """First-person memory after a conversation — embedded and recalled repeatedly later."""
    from engine.executors.social import SocialExecutor

    llm = _Capture('{"fact":"f","success":true,"relation":"neutral","why":""}')
    ex = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._memory_summary(  # noqa: SLF001
        agent=_agent(), other_id="a1", other_name="李世民", transcript="……",
        purpose="求他放我一条生路", fallback_fact="谈了一场。", is_initiator=False, now_step=24,
    )
    assert MARK in llm.text()


@pytest.mark.asyncio
async def test_physical_target_reaction_knows_the_target_was_already_bound() -> None:
    """The recipient's first-person fact goes straight into his memory.

    The condition read here is the one he already had when the blow landed — any new condition
    from this action lands later, in the feedback layer.
    """
    from engine.executors.physical import PhysicalExecutor

    llm = _Capture(
        '{"fact":"f","emotion_type":"anger","emotion_intensity":0.8,'
        '"emotion_valence":-0.7,"relation":"negative"}'
    )
    ex = PhysicalExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._llm_target_reaction(  # noqa: SLF001
        target_agent=_agent(), actor_id="a1", actor_name="李世民",
        event_line="李世民揪住尉迟恭的衣领，将他按在墙上", succeeded=True, target_damage=0.0, target_relief=0.0, step=24,
    )
    text = llm.text()
    assert MARK in text
    # Second-person framing; must not slip into first person.
    assert "你此刻的处境" in text


@pytest.mark.asyncio
async def test_experiential_memory_prose_knows_the_writer_is_bound(container) -> None:
    """Experiential memory is first-person prose that gets embedded and recalled countless times.

    Without condition, a bound man writes bodily actions he can't do, like "我攥紧了拳", and that
    memory stays forever.
    """
    from agent.memory import MemorySystem
    from core.interfaces.llm import LLMRouter, LLMScene
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response="我心里明白，这一步再没有回头路了。")
    router = LLMRouter({scene: provider for scene in LLMScene})
    memory = MemorySystem(
        router, container.embedding, container.vector_store,
        world_id="w", agent_id="a2", seconds_per_step=3600,
    )
    personality = PersonalityLayer(soul=SoulLayer(name="李元吉", gender="男", agent_id="a2"))
    personality.set_condition(BOUND)
    await memory._write_experiential(  # noqa: SLF001
        raw_content="李世民命人将他双手反绑。", personality=personality, now_step=24,
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    assert MARK in prompt


# ---------------------------------------------------------------------------
# Designs what happens to this person based on it
# ---------------------------------------------------------------------------


def test_event_brief_lists_condition_beside_vitality() -> None:
    """The event editor's character briefing is about externally observable scalars — who this is,
    where, how much life is left — and "hands tied behind the back" is the most conspicuous of
    them. Without it the editor designs events where a bound man bursts out of the palace gate, and
    that text is broadcast, perceived, and written into memory."""
    from engine.event import EventSystem

    main = _agent("a2", "李元吉")
    main.is_main_character = True
    brief = EventSystem._brief_characters(  # noqa: SLF001
        type("S", (), {"directory": _directory()})(), {"a2": main},
    )
    assert MARK in brief
    assert "体力" in brief  # alongside it, same granularity


def test_event_brief_omits_condition_when_unencumbered() -> None:
    from engine.event import EventSystem

    main = _agent("a2", "李元吉", condition=None)
    main.is_main_character = True
    brief = EventSystem._brief_characters(  # noqa: SLF001
        type("S", (), {"directory": _directory()})(), {"a2": main},
    )
    assert "处境" not in brief


@pytest.mark.asyncio
async def test_social_interrupt_summary_knows_the_speaker_is_bound() -> None:
    """In an interrupted conversation, each participant writes a first-person memory — also stored
    and embedded."""
    from engine.executors.social import SocialExecutor

    llm = _Capture('{"fact":"f"}')
    ex = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._interrupt_summary(  # noqa: SLF001
        agent=_agent(), other_name="李世民", purpose="替自己分辩", thought="我说不下去了",
        cause="", elapsed_label="约1小时", is_triggered=True, is_initiator=False, scene="",
        now_step=24,
    )
    assert MARK in llm.text()


@pytest.mark.asyncio
async def test_dialogue_transcript_knows_which_speaker_is_bound() -> None:
    """The dialogue transcript is written by a third-person narrator: both people's bodily state
    must be present, or it writes a bound man slamming the table and standing up. One line each;
    giving only one side isn't enough."""
    from engine.executors.social import SocialExecutor

    llm = _Capture('{"dialogue":[{"speaker":1,"line":"…"}],"observation":"两人谈了一场。"}')
    ex = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    empty = {"relation": "", "about_other": "", "about_topic": ""}
    # Each person has a different condition: checking only one stays green even if the other's
    # interpolation slot is deleted.
    limping = BodyCondition(description="右腿有伤", since_step=0)
    await ex._llm_full_dialogue(  # noqa: SLF001
        initiator=_agent("a1", "李世民", condition=limping),
        target=_agent("a2", "李元吉"),
        initiator_name="李世民", target_name="李元吉",
        purpose="逼问东宫往来", expected_outcome="", turns=2,
        initiator_ctx=dict(empty), target_ctx=dict(empty), scene="", now_step=24,
    )
    text = llm.text()
    assert MARK in text and "右腿有伤" in text


@pytest.mark.asyncio
async def test_dialogue_leaves_a_free_speaker_unmarked() -> None:
    """The side with no condition omits the whole line — otherwise every conversation hangs a line
    of noise on everyone."""
    from engine.executors.social import SocialExecutor

    llm = _Capture('{"dialogue":[{"speaker":1,"line":"…"}],"observation":"两人谈了一场。"}')
    ex = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    empty = {"relation": "", "about_other": "", "about_topic": ""}
    await ex._llm_full_dialogue(  # noqa: SLF001
        initiator=_agent("a1", "李世民", condition=None),
        target=_agent("a2", "李元吉", condition=None),
        initiator_name="李世民", target_name="李元吉",
        purpose="议事", expected_outcome="", turns=2,
        initiator_ctx=dict(empty), target_ctx=dict(empty), scene="", now_step=24,
    )
    assert "此刻的处境" not in llm.text()


@pytest.mark.asyncio
async def test_work_interrupt_memory_knows_the_worker_is_bound() -> None:
    """Interrupted work also leaves a first-person memory — this and _generate_outcome are two
    prompts with separate interpolation slots, and deleting either doesn't trip the other's test."""
    from engine.executors.base import ActionExecutionState
    from engine.executors.work import WorkExecutor
    from core.interfaces.action import ActionType

    llm = _Capture('{"fact":"我正誊到一半，被人打断了。"}')
    ex = WorkExecutor(llm, _directory(), seconds_per_step=3600)
    state = ActionExecutionState(
        execution_id="e1", action_type=ActionType.WORK, initiator_id="a2",
        participant_ids=["a2"], started_step=20, estimated_steps=4, remaining_steps=2,
        purpose="誊抄名册",
    )
    await ex.interrupt(state, 24, agents={"a2": _agent()}, thought="我停下了", cause="外头喧哗")
    assert MARK in llm.text()
