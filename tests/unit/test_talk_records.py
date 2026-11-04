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


def _agent(agent_id: str, name: str, gender: str):
    class _A:
        def __init__(self) -> None:
            self.agent_id = agent_id
            self.is_main_character = False
            self.is_active = True
            self.personality = PersonalityLayer(
                soul=SoulLayer(name=name, gender=gender, agent_id=agent_id),
            )
            self.relation_context_asked: list[list[str]] = []

        async def relation_context(self, agent_ids) -> str:  # noqa: ANN001
            """Stands in for the canonical rows: one acquaintance per id asked about."""
            from agent.relation import render_relation_context
            from core.interfaces.relation import AgentRelation
            from core.prompts import person_referent

            ids = list(agent_ids)
            self.relation_context_asked.append(ids)
            return render_relation_context([
                (person_referent("赵六", "女"), AgentRelation(
                    world_id="w", from_id=agent_id, to_id=aid, trust_objective=0.5,
                    affection_objective=0.0, updated_step=0, labels=["相识"],
                ))
                for aid in ids
            ], include_legend=False)

    return _A()


def _env():
    from engine.environment import EnvironmentSystem

    return EnvironmentSystem()


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
        purpose="问分工", is_initiator=False, environment=_env(),
    )
    assert "赵六（女）来找我谈" in llm.user()
    # The purpose is the initiator's own; the addressee knows the talk by what was said.
    assert "问分工" not in llm.user()


@pytest.mark.asyncio
async def test_an_interrupted_record_gives_the_topic_only_to_the_initiator() -> None:
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    for is_initiator in (True, False):
        llm = _Capture('{"fact":""}')
        fallback = await _executor(llm, sun, zhao)._interrupt_summary(  # noqa: SLF001
            agent=sun, other_id="a2", other_name="赵六", purpose="问分工", thought="",
            cause="", elapsed_label="约1小时", is_triggered=False, is_initiator=is_initiator, scene="",
            environment=_env(),
        )
        assert ("问分工" in llm.user()) is is_initiator
        assert ("问分工" in fallback) is is_initiator


@pytest.mark.asyncio
async def test_the_addressee_knows_the_topic_stated_up_front_and_nothing_more() -> None:
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    llm = _Capture('{"fact":"f","success":true,"relation":"neutral","why":""}')
    await _executor(llm, sun, zhao)._memory_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", transcript="赵六：孙姐。",
        purpose="套她的话", talk_topic="问分工", is_initiator=False, environment=_env(),
    )
    assert "来意是「问分工」" in llm.user() and "套她的话" not in llm.user()

    llm = _Capture('{"fact":""}')
    fallback = await _executor(llm, sun, zhao)._interrupt_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", purpose="套她的话", talk_topic="问分工",
        thought="", cause="", elapsed_label="约1小时", is_triggered=False, is_initiator=False, scene="",
        environment=_env(),
    )
    assert "问分工" in llm.user() and "套她的话" not in llm.user()
    assert "问分工" in fallback and "套她的话" not in fallback


@pytest.mark.asyncio
async def test_an_interrupted_record_knows_the_other_party() -> None:
    llm = _Capture('{"fact":"f"}')
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    await _executor(llm, sun, zhao)._interrupt_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", purpose="问分工", thought="",
        cause="", elapsed_label="约1小时", is_triggered=False, is_initiator=True, scene="",
        environment=_env(),
    )
    assert "是我主动找赵六（女）谈的" in llm.user()


@pytest.mark.asyncio
async def test_a_listener_s_record_knows_both_speakers() -> None:
    llm = _Capture('{"fact":"f"}')
    wang, sun, zhao = _agent("a3", "王五", "男"), _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    await _executor(llm, wang, sun, zhao)._listener_memory(  # noqa: SLF001
        agent=wang, initiator_id="a2", target_id="a1", transcript="赵六：孙姐。", environment=_env(),
    )
    assert "说话的是赵六（女）与孙七（女）" in llm.user()
    assert "来意" not in llm.user()


@pytest.mark.asyncio
async def test_a_listener_knows_the_topic_stated_up_front() -> None:
    llm = _Capture('{"fact":"f"}')
    wang, sun, zhao = _agent("a3", "王五", "男"), _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    await _executor(llm, wang, sun, zhao)._listener_memory(  # noqa: SLF001
        agent=wang, initiator_id="a2", target_id="a1", transcript="赵六：孙姐。", talk_topic="问分工",
        environment=_env(),
    )
    assert "赵六（女）发起的对话，来意是「问分工」" in llm.user()


@pytest.mark.asyncio
async def test_present_relations_are_the_agents_own_block_for_who_is_here() -> None:
    """The block is the agent's canonical one (``Agent.relation_context``), asked about exactly
    the people present; the executor renders no relations by a rule of its own."""
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
    assert sun.relation_context_asked == [["a2"]]
    assert "赵六（女）：[相识]" in block


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
                purpose="问分工", is_initiator=False, environment=env,
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
        assert marker not in user and "赵六（女）：[相识]" in user, record


@pytest.mark.asyncio
async def test_a_record_without_its_verdict_claims_no_success() -> None:
    """The talk happened and its record came back, so it is kept; a missing verdict claims the
    least rather than a success nobody read off the record."""
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    llm = _Capture('{"fact":"我和赵六谈了一场。","relation":"neutral"}')
    summary = await _executor(llm, sun, zhao)._memory_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", transcript="赵六：孙姐。",
        purpose="问分工", is_initiator=True, environment=_env(),
    )
    assert summary is not None
    fact, succeeded, _, _ = summary
    assert fact == "我和赵六谈了一场。"
    assert succeeded is False


@pytest.mark.asyncio
async def test_a_record_without_its_fact_is_a_null_step() -> None:
    """No record, no memory: a stand-in would remember a talk nobody recorded."""
    sun, zhao = _agent("a1", "孙七", "女"), _agent("a2", "赵六", "女")
    llm = _Capture('{"success":true,"relation":"neutral"}')
    summary = await _executor(llm, sun, zhao)._memory_summary(  # noqa: SLF001
        agent=sun, other_id="a2", other_name="赵六", transcript="赵六：孙姐。",
        purpose="问分工", is_initiator=True, environment=_env(),
    )
    assert summary is None
