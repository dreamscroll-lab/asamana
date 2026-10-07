"""SoulLayer.secret: a fact about a person that only they know.

It reaches every place that renders that person's own full profile and the author layer, and none
that shows the person to someone else.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.need import NeedEngine, NeedType
from agent.personality import SECRET_LABEL, PersonalityLayer, SoulLayer
from core.interfaces.llm import LLMResponse, LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider
from world.models import AgentDefinition, ThemeFigure

SECRET = "我把小队的行动透露给了外人"


def _soul(secret: str = SECRET) -> SoulLayer:
    return SoulLayer(name="顾湘", gender="女", agent_id="a1", role="外勤", secret=secret)


def _agent(secret: str = SECRET, agent_id: str = "a1", *, main: bool = True):
    class _A:
        def __init__(self) -> None:
            self.agent_id = agent_id
            self.is_main_character = main
            self.is_active = True
            self.personality = PersonalityLayer(soul=_soul(secret))

    return _A()


def _directory():
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    return LiveWorldDirectory.from_agents({}, EnvironmentSystem())


class _Capture:
    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list = []

    async def complete(self, scene, messages, **_kw):  # noqa: ANN001
        self.messages.append(messages)
        return LLMResponse(content=self._content, input_tokens=0, output_tokens=0, model="test")

    def text(self) -> str:
        return "\n".join(m.content for m in self.messages[-1])


# --- the person's own profile ------------------------------------------------


def test_profile_renders_the_secret_only_when_there_is_one() -> None:
    assert f"{SECRET_LABEL}：{SECRET}" in PersonalityLayer(soul=_soul()).to_prompt_context()
    assert SECRET_LABEL not in PersonalityLayer(soul=_soul("")).to_prompt_context()


@pytest.mark.asyncio
async def test_short_term_goal_prompt_carries_the_secret() -> None:
    """Goals can be revised away during the run; the secret can't, so it is fed in directly."""
    llm = _Capture('{"goals": []}')
    await NeedEngine(llm_router=llm)._generate_short_term_goals(  # type: ignore[arg-type]  # noqa: SLF001
        dominant_need=NeedType.SAFETY, personality=PersonalityLayer(soul=_soul()),
    )
    assert f"我的{SECRET_LABEL}：{SECRET}" in llm.text()


@pytest.mark.asyncio
async def test_inner_monologue_carries_the_secret(container) -> None:
    from agent.memory import MemorySystem

    provider = MockLLMProvider(fixed_response="我得先稳住，不能让人看出来。")
    memory = MemorySystem(
        LLMRouter({scene: provider for scene in LLMScene}), container.embedding,
        container.vector_store, world_id="w", agent_id="a1", seconds_per_step=3600,
    )
    await memory._write_experiential(  # noqa: SLF001
        raw_content="老潘说要逐个面谈。", personality=PersonalityLayer(soul=_soul()), now_step=3,
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    assert f"我的{SECRET_LABEL}：{SECRET}" in prompt


# --- the author layer ---------------------------------------------------------


def test_event_brief_and_director_line_carry_the_secret() -> None:
    """The author layer plants clues and resolves "the traitor", so it must know the truth."""
    from engine.director import DirectorChannel
    from engine.event import EventSystem

    brief = EventSystem._brief_characters(  # noqa: SLF001
        SimpleNamespace(directory=_directory()), {"a1": _agent()},
    )
    assert SECRET in brief

    channel = DirectorChannel.__new__(DirectorChannel)
    channel._directory = _directory()  # noqa: SLF001
    assert SECRET in channel._agent_line(1, _agent())  # noqa: SLF001


# --- views of the person from outside ----------------------------------------


def test_a_bystander_brief_never_carries_the_secret() -> None:
    from engine.scene import _person_background

    agent = _agent()
    agent.personality = PersonalityLayer(soul=SoulLayer(
        name="顾湘", background="入队三年。", secret=SECRET,
    ))
    brief = _person_background(agent, "顾湘")  # type: ignore[arg-type]
    assert "入队三年" in brief  # the brief itself is there
    assert SECRET not in brief


def test_directory_does_not_expose_the_secret() -> None:
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem

    directory = LiveWorldDirectory.from_agents({"a1": _agent()}, EnvironmentSystem())
    assert directory.describe("a1") is not None
    assert SECRET not in repr(directory.describe("a1"))
    assert SECRET not in repr(directory.agent_identity_map(["a1"]))


@pytest.mark.asyncio
async def test_world_pressure_shows_the_target_s_secret_but_not_a_co_present_person_s() -> None:
    """The assessment's result reaches the target's goals, so it may weigh only what the target
    knows."""
    from engine.world_pressure import WorldPressureEvaluator

    class _Store:
        async def load_relation(self, *_a):  # noqa: ANN002
            return None

    target = _agent("我欠了外人一大笔钱", agent_id="t")
    target.world_id = "w"
    target.agent_store = _Store()
    other = _agent(agent_id="o")
    ev = WorldPressureEvaluator(llm_router=None)
    co_located = await ev._co_located(target, {"t": target, "o": other}, None, ["o"])  # noqa: SLF001
    prompt = ev._build_agent_prompt(  # noqa: SLF001
        target, inbox=[], broadcasts=[], ambient=[], co_located=co_located,
        sender_relations={}, situation_header="",
    )
    assert "我欠了外人一大笔钱" in prompt
    assert co_located  # the co-present person is rendered
    assert SECRET not in prompt


# --- build -------------------------------------------------------------------


def _theme_payload(figures: list[dict]) -> str:
    return json.dumps({
        "world_name": "队内有鬼", "era_description": "e", "core_tension": "t",
        "narrative_theme": "n", "narrative_pitch": "p",
        "world_time_config": {"start_year": 2026}, "key_figures": figures,
    }, ensure_ascii=False)


def test_theme_analyzer_parses_the_secret_and_leaves_it_empty_when_absent() -> None:
    from world.builders.theme_analyzer import ThemeAnalyzer

    payload = _theme_payload([
        {"name": "顾湘", "role": "外勤", "importance": "main", "brief": "b", "secret": SECRET},
        {"name": "老潘", "role": "队长", "importance": "main", "brief": "b"},
    ])
    router = LLMRouter({scene: MockLLMProvider(fixed_response=payload) for scene in LLMScene})
    analysis = asyncio.run(ThemeAnalyzer(router).analyze("any theme"))
    assert [f.secret for f in analysis.key_figures] == [SECRET, ""]


def test_figure_line_shows_the_secret_only_when_there_is_one() -> None:
    assert SECRET in ThemeFigure(name="顾湘", secret=SECRET).as_prompt()
    assert SECRET_LABEL not in ThemeFigure(name="老潘").as_prompt()


def test_the_soul_takes_the_secret_from_the_figure_and_keeps_it_through_a_round_trip() -> None:
    """ThemeAnalyzer is its only author: the per-figure call can't see how many others hold one."""
    from tests.unit.test_world_builder import _analysis
    from world.builders.agent_generator import AgentGenerator

    figure = ThemeFigure(name="顾湘", role="外勤", importance="main", secret=SECRET)
    definition = AgentGenerator.__new__(AgentGenerator)._definition_from_payload(  # noqa: SLF001
        figure, _analysis(), {"secret": "别的事"}, "a1",
    )
    assert definition.soul.secret == SECRET
    assert AgentDefinition.from_dict(definition.as_dict()).soul.secret == SECRET
