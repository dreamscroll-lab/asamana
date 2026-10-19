"""What the first-person records of a conversation are given about the other people.

Each record becomes an embedded memory written in pronouns; given only a name, the model guesses
the gender, and a wrong guess (「约他」 for a woman) stays in memory for good.
"""

from __future__ import annotations

import pytest

from agent.personality import PersonalityLayer, SoulLayer


class _Capture:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.messages: list = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001
        self.messages.append(messages)

        class _R:
            content = self.payload

        return _R()

    def user(self) -> str:
        return self.messages[-1][-1].content


class _Rel:
    async def perceive_existing(self, *_a, **_kw):
        return type("P", (), {"labels": ["相识"], "trust": 0.5, "affection": 0.0})()


def _agent(agent_id: str, name: str, gender: str):
    class _A:
        def __init__(self) -> None:
            self.agent_id = agent_id
            self.is_main_character = False
            self.is_active = True
            self.personality = PersonalityLayer(
                soul=SoulLayer(name=name, gender=gender, agent_id=agent_id),
            )
            self.relation_system = _Rel()

    return _A()


def _executor(llm: _Capture, *agents):  # noqa: ANN002
    from engine.directory import LiveWorldDirectory
    from engine.environment import EnvironmentSystem
    from engine.executors.social import SocialExecutor

    directory = LiveWorldDirectory.from_agents({a.agent_id: a for a in agents}, EnvironmentSystem())
    return SocialExecutor(llm, directory, seconds_per_step=3600)


@pytest.mark.asyncio
async def test_the_addressee_s_record_knows_who_came_to_talk() -> None:
    llm = _Capture('{"fact":"f","success":true,"relation":"neutral","why":""}')
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    await _executor(llm, sun, zhao)._memory_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", transcript="赵六：孙姐。",
        purpose="问分工", fallback_fact="谈了一场。", is_initiator=False,
    )
    assert "赵六（女）来找我谈" in llm.user()


@pytest.mark.asyncio
async def test_an_interrupted_record_knows_the_other_party() -> None:
    llm = _Capture('{"fact":"f"}')
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    await _executor(llm, sun, zhao)._interrupt_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", purpose="问分工", thought="",
        cause="", elapsed_label="约1小时", is_triggered=False, is_initiator=True, scene="",
    )
    assert "是我主动找赵六（女）谈的" in llm.user()


@pytest.mark.asyncio
async def test_a_listener_s_record_knows_both_speakers() -> None:
    llm = _Capture('{"fact":"f"}')
    wang, sun, zhao = _agent("a3", "王五", "男"), _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    await _executor(llm, wang, sun, zhao)._listener_memory(  # noqa: SLF001
        agent=wang, initiator_id="a2", target_id="a1", transcript="赵六：孙姐。",
    )
    assert "说话的是赵六（女）与孙七（女）" in llm.user()


@pytest.mark.asyncio
async def test_present_relations_name_their_holder_and_gender() -> None:
    from core.interfaces.place import Place
    from engine.environment import EnvironmentSystem

    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="med", name="医疗室", description="", connections={}, is_public=True,
    ))
    for a in (sun, zhao):
        env.place_agent(agent_id=a.agent_id, location_id="med")
    block = await _executor(_Capture(""), sun, zhao)._present_relations_block(sun, env)  # noqa: SLF001
    assert "- 对在场的赵六（女）：" in block


@pytest.mark.asyncio
async def test_the_relation_legend_comes_before_the_rows_it_explains() -> None:
    """The legend is static and goes in system, read before the rows in user; trailing them, it
    comes after the model has already read each label by its own prior."""
    from core.interfaces.place import Place
    from engine.environment import EnvironmentSystem

    marker = "「父子:父亲」表示对方是持有者的父亲"
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="med", name="医疗室", description="", connections={}, is_public=True,
    ))
    for a in (sun, zhao):
        env.place_agent(agent_id=a.agent_id, location_id="med")
    for record in ("summary", "interrupt"):
        llm = _Capture('{"fact":"f","success":true,"relation":"neutral","why":""}')
        ex = _executor(llm, sun, zhao)
        if record == "summary":
            await ex._memory_summary(  # noqa: SLF001
                agent=sun, other_id="a2", other_name="赵六", transcript="赵六：孙姐。",
                purpose="问分工", fallback_fact="谈了一场。", is_initiator=False, environment=env,
            )
        else:
            await ex._interrupt_summary(  # noqa: SLF001
                agent=sun, other_id="a2", other_name="赵六", purpose="问分工", thought="",
                cause="", elapsed_label="约1小时", is_triggered=False, is_initiator=True, scene="",
                environment=env,
                relation_block=await ex._present_relations_block(sun, env),  # noqa: SLF001
            )
        system, user = (m.content for m in llm.messages[-1])
        assert system.count(marker) == 1, record
        assert "【我对在场者的关系】" in system and "我是持有者" in system, record
        assert marker not in user and "- 对在场的赵六（女）：" in user, record
