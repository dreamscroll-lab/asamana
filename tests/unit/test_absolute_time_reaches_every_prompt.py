"""The absolute-future-time rule reaches every prompt it should, and only those.

A deictic time word ("明晨" tomorrow morning) is persisted without the moment it was said, so
each later reader resolves it against their own "now": an appointment slides a day per reading
and never comes due. Every output channel that misses the rule keeps pouring such words into the
narrative substrate. (A separate file for the same reason as test_condition_reaches_every_prompt.)

Three layers of assertion, all required:
1. Present: the rendered prompt actually contains it (not just a file that mentions the constant).
2. Paired: a prompt carrying the "compute a date" rule must also give the current time;
   otherwise the model invents "now", which is as wrong as deixis drift and harder to see.
3. Bounded: prompts on the exclusion list really don't have it. physical / covert adjudication
   outputs are short past-tense "what just happened" and already say "不要带时间或地点前缀";
   pure-judgment prompts (importance, should_inject) never reach the narrative substrate. Adding
   it there only makes the model waver between two instructions.
"""

from __future__ import annotations

import re

import pytest

from agent.personality import PersonalityLayer, SoulLayer
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    KEEP_ABSOLUTE_TIME_RULE,
)

#: The phrase both person variants share; detects the rule regardless of variant.
MARK = "具体时间点"
#: Only the "compute a date" rule carries this; KEEP-only sites don't.
COMPUTE_MARK = "此刻时间"

WORLD_NOW = "武德九年，六月初三，凌晨零点"


def _assert_rule_is_the_last_word(system: str) -> None:
    """The rule must be the last block of system, next to the generation point (§3): buried
    mid-list it is ignored, and a stronger model doesn't fix that.

    Assert "is it the last block", not a position percentage, which shifts with prompt length.
    """
    tail = system.rstrip()
    assert tail.endswith(ABSOLUTE_TIME_RULE_FIRST_PERSON.rstrip()) or \
        tail.endswith(ABSOLUTE_TIME_RULE.rstrip())


def test_the_worked_example_is_labelled_and_points_back_to_the_given_clock() -> None:
    """The example hard-codes a specific day on purpose: an undefined slot like "〈日期〉" makes
    the model guess, and in practice it stops writing dates.

    One line per calendar style, so it isn't tied to one dating system. Era names are elided to
    "…"; a real one would bind the template to that theme (Rule 7).
    """
    ERA = re.compile(r"武德|贞观|开元|洪武|康熙")     # a real era name = bound to a specific world
    for rule in (ABSOLUTE_TIME_RULE, ABSOLUTE_TIME_RULE_FIRST_PERSON):
        flat = re.sub(r"\s+", "", rule)
        assert "只是示范怎么换" in flat
        assert "照所给的此刻时间那套写法来写" in flat
        assert "✗" in flat and "✓" in flat          # it demonstrates a substitution: both wrong and right sides must appear
        # One line per dating style; neither dominates
        assert re.search(r"[正一二三四五六七八九十冬腊]月初[一二三四五六七八九十]", flat)
        assert re.search(r"\d+月\d+日", flat)
        assert not ERA.search(rule), "例子里出现了真年号——模板绑死到某个主题上了"


#: The composite form "relative phrase + bracketed absolute time", e.g. "明晨（日期）".
_COMPOSITE_FORM = re.compile(r"(明晨|明早|明日|明天|今夜|今晚|后日|次日|三日后)（")


def test_the_rule_never_teaches_the_composite_relative_plus_absolute_form() -> None:
    """Appointments are referenced by absolute time only; the rule must not demonstrate a form
    like "明晨（日期）".

    Keeping the deictic word fails either way: copied forward to that day it reads "tomorrow =
    today", and recomputing the date to keep "明晨" true pushes the appointment back a day. An
    absolute time stays correct whichever step or agent copies it.
    """
    for rule in (ABSOLUTE_TIME_RULE, ABSOLUTE_TIME_RULE_FIRST_PERSON, KEEP_ABSOLUTE_TIME_RULE):
        found = _COMPOSITE_FORM.findall(rule)
        assert not found, f"规则里示范了复合形式 {found}——约定只用绝对时点指代"


def test_the_rule_keeps_its_time_of_day_wording_calendar_neutral() -> None:
    """Time-of-day wording must not be tied to one calendar either.

    "寅时" belongs only to the era-name side; a modern world says "早上六点". clock.py doesn't use
    the two-hour shichen at all (too coarse; narrative time would stall). Putting it in the rule
    would mis-teach half the worlds.
    """
    for rule in (ABSOLUTE_TIME_RULE, ABSOLUTE_TIME_RULE_FIRST_PERSON):
        assert not re.search(r"[子丑寅卯辰巳午未申酉戌亥]时", rule)


def test_the_keep_only_rule_never_asks_for_a_clock_it_does_not_have() -> None:
    """KEEP is only for sites with no current time (compression summaries): asking a processor
    without a clock for "now" makes it invent one. The stricter rule lives in the two full variants.
    """
    assert COMPUTE_MARK not in KEEP_ABSOLUTE_TIME_RULE
    assert "当前时间" not in KEEP_ABSOLUTE_TIME_RULE


def test_decision_prompt_tells_me_to_answer_a_goal_whose_hour_has_come() -> None:
    """The instruction must name the block heading the renderer actually emits, or the model
    has nothing to match. Due goals get their own block ahead of the list rather than a
    parenthetical buried in it.
    """
    from agent.decision import _ACTION_SPACE, DecisionEngine
    from tests.unit.test_decision import _make_packet, _make_personality

    system, _, _facts = DecisionEngine(None)._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )

    assert "「截止时间已经到的事」" in system
    # Sits with the date rule at the tail of system (§3): this kind of instruction is barely
    # followed when buried mid-list.
    assert system.index("截止时间已经到的事") / len(system) > 0.8


def test_both_person_variants_carry_the_shared_mark() -> None:
    """Every assertion in this file relies on the two MARKs above; check both variants carry them."""
    for rule in (ABSOLUTE_TIME_RULE, ABSOLUTE_TIME_RULE_FIRST_PERSON):
        assert MARK in rule
        assert COMPUTE_MARK in rule
        # Calendar constant: the model can't know a month has thirty days; month rollover
        # depends on it.
        assert "一个月三十天" in rule


class _Capture:
    """Record what was asked; reply with a fixed line."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.messages: list = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001
        self.messages.append(messages)

        class _R:
            content = self.payload
            model = "mock"

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


def _agent(agent_id: str = "a2", name: str = "长孙无忌"):
    class _A:
        def __init__(self) -> None:
            self.agent_id = agent_id
            self.is_main_character = False
            self.personality = PersonalityLayer(
                soul=SoulLayer(name=name, gender="男", agent_id=agent_id),
            )
            self.relation_system = _Rel()

    return _A()


def _environment_at(location_id: str = "palace", *, agent_ids=("a2",)):
    """A real EnvironmentSystem that has already run begin_step. Executors take their time and
    place anchor from it.

    The clock starts at WORLD_NOW, so assertions can look for that exact calendar label rather
    than a weak "some time is present" marker.
    """
    from core.interfaces.place import Place
    from engine.clock import WorldTime, WorldTimeConfig
    from engine.environment import EnvironmentSystem

    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id=location_id, name="皇城", description="宫城之南", connections={}, is_public=True,
    ))
    for aid in agent_ids:
        env.place_agent(agent_id=aid, location_id=location_id)
    config = WorldTimeConfig(
        era_name="武德", start_year=9, start_month=6, start_day=3, start_hour=0,
    )
    env.begin_step(step=0, world_time=WorldTime.from_step(0, config))
    return env


def _scene_module():
    return __import__("engine.scene", fromlist=["x"])


def _directory():
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    return LiveWorldDirectory.from_agents({}, EnvironmentSystem())


# ---------------------------------------------------------------------------
# Tier 1: prompts that can invent a future time point
# ---------------------------------------------------------------------------


def test_decision_prompt_carries_the_rule_and_the_clock() -> None:
    """Decision writes action_description / message_content, which is where a letter asking to
    meet "tomorrow morning" comes from."""
    from agent.decision import _ACTION_SPACE, DecisionEngine
    from tests.unit.test_decision import _make_packet, _make_personality

    engine = DecisionEngine(None)
    system, user, _facts = engine._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )
    assert MARK in system
    # Paired: the current time is in the situation header at the top of user.
    assert "时间为" in user
    _assert_rule_is_the_last_word(system)


def test_decision_prompt_also_tells_me_to_read_the_clock_before_i_write_it() -> None:
    """Reading and writing are two rules, not one: the write-side rule only bites when a future
    time point is actually written.

    Choosing on an assumed hour (planning "趁今夜无人打扰" at noon) writes no time point, so the
    write side can't reach it. Hence a separate "先认准此刻是什么时候", which must come before the
    write-side rule (read before write).
    """
    from agent.decision import _ACTION_SPACE, DecisionEngine
    from tests.unit.test_decision import _make_packet, _make_personality

    engine = DecisionEngine(None)
    system, _user, _facts = engine._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )
    assert "先认准此刻是什么时候" in system
    assert system.index("先认准此刻是什么时候") < system.index(MARK)   # read before write


def test_short_term_goal_prompt_carries_the_rule() -> None:
    """Short-term goals drift worst: rendered with no time anchor, "review before tomorrow
    morning" never comes due."""
    from agent.need import _SHORT_TERM_GOAL_SYSTEM, _SHORT_TERM_GOAL_USER

    assert MARK in _SHORT_TERM_GOAL_SYSTEM
    assert "{situation_header}" in _SHORT_TERM_GOAL_USER  # paired


def test_long_term_goal_prompt_carries_the_rule() -> None:
    from agent.need import _LONG_TERM_GOAL_UPDATE_SYSTEM, _LONG_TERM_GOAL_UPDATE_USER

    assert MARK in _LONG_TERM_GOAL_UPDATE_SYSTEM
    assert "{situation_header}" in _LONG_TERM_GOAL_UPDATE_USER  # paired


def test_world_pressure_prompt_carries_the_rule() -> None:
    """External-drive text can promise something for "tomorrow morning", and it is fed into
    decisions for several steps in a row."""
    from engine.world_pressure import _SYSTEM_PROMPT

    assert MARK in _SYSTEM_PROMPT


def test_director_prompt_carries_the_rule() -> None:
    """A director line lands in the recipient's inbox and memory, indistinguishable from what
    the agent wrote itself."""
    from engine.director import _SYSTEM_PROMPT

    assert MARK in _SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_interrupt_prompt_carries_the_rule_and_the_clock(container: object) -> None:
    """evaluate_interrupt's thought directly produces persistent narrative text and is easy
    to miss.

    Executors like MOVE and REST have no memory-writing LLM; the thought is appended verbatim
    to factual_memory via format_interrupt_thought (engine/narration.py, called from movement.py
    and simple.py). It isn't an intermediate value; it is the memory itself.
    """
    from core.interfaces.perception import LocationView, Situation
    from providers.llm.mock import MockLLMProvider
    from tests.unit.test_agent_interrupt import _make_agent

    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    agent._situation = Situation(  # noqa: SLF001
        location_view=LocationView(name="皇城", description="宫城之南"), time_label=WORLD_NOW,
    )
    await agent.evaluate_interrupt(
        step=1, reason="宫门方向骤起兵刃相击",
        current_action_desc="推演明日面谈的论据", intent="", progress_hint="进行了一会儿",
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    assert MARK in prompt
    assert WORLD_NOW in prompt  # paired


def test_goal_progress_residue_prompt_carries_the_rule() -> None:
    """Residue becomes an origin=RESIDUE GoalEntity, the same persistent channel as short-term
    goals.

    It renders in the decision prompt's "我还没有了结的事项" with no time anchor, the same gap
    as short-term goals.
    """
    from agent.need import _GOAL_EVALUATION_SYSTEM, _GOAL_EVALUATION_USER

    assert MARK in _GOAL_EVALUATION_SYSTEM
    assert "{situation_header}" in _GOAL_EVALUATION_USER  # paired


@pytest.mark.asyncio
async def test_dialogue_prompt_carries_the_rule_and_the_clock() -> None:
    """A "明晨" agreed in dialogue goes into both parties' memories, and from then on their
    readings of the same appointment diverge."""
    from engine.executors.social import SocialExecutor

    llm = _Capture('{"dialogue":[{"speaker":1,"line":"好"}],"observation":"o"}')
    ex = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    env = _environment_at(agent_ids=("a2", "a3"))
    await ex._llm_full_dialogue(  # noqa: SLF001
        initiator=_agent("a2", "长孙无忌"), target=_agent("a3", "李世民"),
        initiator_name="长孙无忌", target_name="李世民",
        initiator_ctx={}, target_ctx={}, purpose="储位之事",
        turns=2, now_step=0, expected_outcome="",
        scene=_scene_module().assemble_scene_context(
            "a2", environment=env, directory=_directory(),
            visibility=_scene_module().SceneVisibility.GOD,
        ).text,
    )
    text = llm.text()
    assert MARK in text
    assert WORLD_NOW in text  # paired
    _assert_rule_is_the_last_word(llm.messages[-1][0].content)


# ---------------------------------------------------------------------------
# Tier 2: prompts that can paraphrase an existing absolute date back into a relative one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_work_self_appraisal_carries_the_rule_and_the_clock() -> None:
    """WORK's fact goes straight into memory, and it can mention "tomorrow's meeting"."""
    from core.interfaces.perception import Situation
    from core.prompts import SituationVoice, render_situation_header
    from engine.executors.work import WorkExecutor
    from core.interfaces.perception import LocationView

    llm = _Capture('{"fact":"f","success":true,"outcome":"o","observation":"ob","why":""}')
    ex = WorkExecutor(llm, _directory(), seconds_per_step=3600)
    header = render_situation_header(
        Situation(location_view=LocationView(name="皇城", description=""), time_label=WORLD_NOW),
        voice=SituationVoice.FIRST,
    )
    await ex._generate_outcome(  # noqa: SLF001
        agent=_agent(), purpose="推演明日面谈的论据", duration_label="约8小时",
        now_step=0, situation_header=header,
    )
    text = llm.text()
    assert MARK in text
    assert WORLD_NOW in text  # paired


@pytest.mark.asyncio
async def test_social_memory_summary_carries_the_rule_and_the_clock() -> None:
    from engine.executors.social import SocialExecutor

    llm = _Capture('{"fact":"f","success":true,"relation":"neutral","why":""}')
    ex = SocialExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._memory_summary(  # noqa: SLF001
        agent=_agent(), other_id="a3", other_name="李世民", transcript="……",
        purpose="约定面谈", fallback_fact="谈了一场。", is_initiator=True, now_step=0,
        environment=_environment_at(agent_ids=("a2", "a3")),
    )
    text = llm.text()
    assert MARK in text
    assert WORLD_NOW in text  # paired
    _assert_rule_is_the_last_word(llm.messages[-1][0].content)


@pytest.mark.asyncio
async def test_experiential_memory_prose_carries_the_rule(container) -> None:
    """Experiential monologue is embedded and recalled for many later steps; a "明晨" written
    into it can never be removed."""
    from agent.memory import MemorySystem
    from core.interfaces.llm import LLMRouter, LLMScene
    from core.interfaces.perception import LocationView, Situation
    from providers.llm.mock import MockLLMProvider

    provider = MockLLMProvider(fixed_response="我心里明白，这一步再没有回头路了。")
    router = LLMRouter({scene: provider for scene in LLMScene})
    memory = MemorySystem(
        router, container.embedding, container.vector_store,
        world_id="w", agent_id="a2", seconds_per_step=3600,
    )
    personality = PersonalityLayer(soul=SoulLayer(name="长孙无忌", gender="男", agent_id="a2"))
    await memory._write_experiential(  # noqa: SLF001
        raw_content="我传讯请求明晨面谈。", personality=personality, now_step=0,
        situation=Situation(
            location_view=LocationView(name="皇城", description=""), time_label=WORLD_NOW,
        ),
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    assert MARK in prompt
    assert WORLD_NOW in prompt  # paired


def test_period_summary_carries_only_the_keep_clause(container) -> None:
    """Compression summaries carry dates from the records they restate, but they only have a
    time range, not "now".

    So they get only the "copy existing dates" clause; the "compute a date" clause would force
    computing from an anchor they don't have.
    """
    from agent.memory import MemorySystem
    from agent.memory_types import MemoryStream

    memory = MemorySystem(
        None, container.embedding, container.vector_store,
        world_id="w", agent_id="a2", seconds_per_step=3600,
    )
    system, _ = memory._build_summary_prompt(  # noqa: SLF001
        MemoryStream.FACTUAL, time_range="六月初一 至 六月初三", contents="- 甲做了乙。",
    )
    assert "照抄那个日期" in system
    assert COMPUTE_MARK not in system  # no "now", so it must not be asked to compute


def test_reflection_prompt_carries_the_rule_and_the_clock() -> None:
    """Reflection needs its time anchor before it can be asked to write dates."""
    from agent.memory_types import Memory
    from agent.reflection import ReflectionEngine
    from agent.memory_types import MemoryStream

    engine = ReflectionEngine(
        None, None, PersonalityLayer(soul=SoulLayer(name="长孙无忌", agent_id="a2")),
        agent_id="a2", time_label_for=lambda _step: WORLD_NOW,
    )
    candidate = Memory(
        id="m1", agent_id="a2", stream=MemoryStream.EXPERIENTIAL,
        raw_content="x", stored_content="我约了明晨面谈。", created_step=0, kind="event",
    )
    system, user = engine._build_prompt([candidate], current_step=1)  # noqa: SLF001
    assert MARK in system
    assert WORLD_NOW in user  # paired
    assert "某处" not in user  # time without a place: must not invent a place


def test_reflection_prompt_degrades_without_a_clock() -> None:
    """With no clock wired (tests/offline) the block is omitted: no raise, no empty anchor."""
    from agent.memory_types import Memory
    from agent.reflection import ReflectionEngine
    from agent.memory_types import MemoryStream

    engine = ReflectionEngine(
        None, None, PersonalityLayer(soul=SoulLayer(name="长孙无忌", agent_id="a2")),
        agent_id="a2",
    )
    candidate = Memory(
        id="m1", agent_id="a2", stream=MemoryStream.EXPERIENTIAL,
        raw_content="x", stored_content="我约了明晨面谈。", created_step=0, kind="event",
    )
    _system, user = engine._build_prompt([candidate], current_step=1)  # noqa: SLF001
    assert "此刻是" not in user
    assert user.startswith("【这是我】")


# ---------------------------------------------------------------------------
# Bounded: the exclusion list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_covert_judge_stays_out_of_scope() -> None:
    """Covert adjudication writes traces and findings, both told "不要带时间前缀"; the system
    stamps time and place itself."""
    from engine.executors.covert import CovertExecutor

    llm = _Capture('{"reason":"x","achieved":false,"detected":true,"outcome":"o","fact":"f"}')
    ex = CovertExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._judge(  # noqa: SLF001
        _agent(), purpose="潜到廊下偷听", duration_label="约1小时",
        expected_outcome="", scene="", now_step=0,
    )
    assert MARK not in llm.text()


@pytest.mark.asyncio
async def test_physical_target_reaction_stays_out_of_scope() -> None:
    """The target-reaction prompt has no time anchor; adding "compute a date" would force it
    to invent one."""
    from engine.executors.physical import PhysicalExecutor

    llm = _Capture(
        '{"fact":"f","emotion_type":"anger","emotion_intensity":0.8,'
        '"emotion_valence":-0.7,"relation":"negative"}'
    )
    ex = PhysicalExecutor(llm, _directory(), seconds_per_step=3600)
    await ex._llm_target_reaction(  # noqa: SLF001
        target_agent=_agent(), actor_id="a3", actor_name="李世民",
        event_line="李世民揪住尉迟恭的衣领，将他按在墙上", succeeded=True, target_damage=0.0, target_relief=0.0, step=0,
    )
    assert MARK not in llm.text()


def test_importance_evaluator_stays_out_of_scope() -> None:
    """Pure judgment: outputs a score plus a short reason, never enters the narrative substrate."""
    import inspect

    import agent.importance_evaluator as mod

    assert "ABSOLUTE_TIME_RULE" not in inspect.getsource(mod)


def test_relation_summary_stays_out_of_scope() -> None:
    """Relation evolution's summary is a <=25-char retrospective trend
    ("近期数次交锋后信任明显下降").

    A time-worded line from it ("他是今夜最可靠的执行者") already breaks the "summarize the
    trend" constraint, and RelationEvolution has no world clock (its constructor takes only
    lookback_steps), so "compute a date" would force it to invent one. A deliberate trade-off.
    """
    import inspect

    import agent.relation_evolution as mod

    assert "ABSOLUTE_TIME_RULE" not in inspect.getsource(mod)


def test_world_building_stays_out_of_scope() -> None:
    """Build time produces background/history (past) and long-term goals (direction).

    Long-term goals already have a stronger guard, "不得把尚未发生的事件当作已确定的未来",
    which rules out a concrete future appointment at the source. And when ThemeAnalyzer runs the
    world start time doesn't exist yet (it is deciding world_time_config), so there is no "now".
    """
    import inspect

    import world.builder
    import world.builders.agent_generator
    import world.builders.cast_designer
    import world.builders.template_selector
    import world.builders.theme_analyzer

    build_src = "".join(inspect.getsource(m) for m in (
        world.builder, world.builders.agent_generator, world.builders.cast_designer,
        world.builders.template_selector, world.builders.theme_analyzer,
    ))
    assert "ABSOLUTE_TIME_RULE" not in build_src
    assert "不得把尚未发生的事件当作已确定的未来" in inspect.getsource(world.builders.agent_generator)  # that existing guard is still in place
