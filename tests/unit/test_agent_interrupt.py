"""Tests for Agent.evaluate_interrupt() and finalize_ongoing_action(is_interrupt=True)."""

from __future__ import annotations

import pytest

from agent.agent import Agent
from agent.decision import DecisionEngine
from agent.memory import MemorySystem
from agent.need import NeedEngine
from agent.personality import PersonalityLayer, SoulLayer, StateLayer, EmotionState
from agent.relation import RelationSystem
from core.interfaces.llm import LLMRouter, LLMScene
from core.prompts import MEMORY_ORDER_HINT
from providers.llm.mock import MockLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_personality(*, agent_id: str = "agent-1") -> PersonalityLayer:
    soul = SoulLayer(
        name="Test",
        agent_id=agent_id,
        role="advisor",
        core_traits=["pragmatic"],
        core_values=["loyalty"],
    )
    state = StateLayer(
        step=1,
        emotion=EmotionState(primary="neutral", intensity=0.3, valence=0.0),
        current_location="palace",
        dominant_need="social",
    )
    return PersonalityLayer(soul=soul, state=state)


def _make_agent(
    container: object,
    *,
    agent_id: str = "agent-1",
    is_main: bool = False,
    llm_provider: object | None = None,
) -> Agent:
    if llm_provider is None:
        llm_provider = MockLLMProvider()
    router = LLMRouter({scene: llm_provider for scene in LLMScene})  # type: ignore[arg-type]
    return Agent(
        world_id="world-1",
        agent_id=agent_id,
        personality=_make_personality(agent_id=agent_id),
        decision_engine=DecisionEngine(router),
        llm_router=router,
        memory_system=MemorySystem(
            router,
            container.embedding,  # type: ignore[attr-defined]
            container.vector_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id=agent_id,
        ),
        need_engine=NeedEngine(),
        relation_system=RelationSystem(
            container.agent_store,  # type: ignore[attr-defined]
            world_id="world-1",
            agent_id=agent_id,
        ),
        agent_store=container.agent_store,  # type: ignore[attr-defined]
        is_main_character=is_main,
    )


# ---------------------------------------------------------------------------
# evaluate_interrupt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_main_character_interrupt_llm_decision_yes(container: object) -> None:
    provider = MockLLMProvider(fixed_response='{"thought": "火比奏折要紧，我得立刻去看", "interrupt": true}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    should_interrupt, thought = await agent.evaluate_interrupt(
        step=1,
        reason="走廊失火",
        current_action_desc="阅读奏折",
        intent="",
        progress_hint="已进行了约2小时，已过了大半",
    )
    assert should_interrupt is True
    assert thought == "火比奏折要紧，我得立刻去看"
    assert len(provider.call_history) == 1


@pytest.mark.asyncio
async def test_interrupt_prompt_is_first_person_without_step_leak(container: object) -> None:
    """The decision prompt is in-character (first-person) and carries progress as
    narrative text — no step integers leak into it (time-axis boundary)."""
    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    await agent.evaluate_interrupt(
        step=1,
        reason="外面有动静",
        current_action_desc="读书",
        intent="",
        progress_hint="才刚开始",
    )
    # The call sends [system, user]; join both for the substring/order assertions (mirrors
    # test_decision._decision_prompt).
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    assert "【我是谁】" in prompt          # first-person role header
    assert "我不会这样做" in prompt        # negative constraints lead
    assert "才刚开始" in prompt            # narrative progress, not a step count
    assert "步" not in prompt              # no code-layer step leak
    # thought field must precede the interrupt verdict (think-then-decide)
    assert prompt.index("thought") < prompt.index("interrupt")


@pytest.mark.asyncio
async def test_interrupt_prompt_includes_situation_header(container: object) -> None:
    """Situation anchoring: the interrupt prompt's first line injects "我此刻在{地点},
    当前时间为{时间}" (first-person). The situation comes from the cached _situation of the last
    perceive_step; runtime doesn't need to pass it through."""
    from core.interfaces.perception import LocationView, Situation
    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    # Simulate the situation runtime caches in perceive_step (no need to actually run
    # perceive_step here; set _situation directly).
    agent._situation = Situation(
        location_view=LocationView(name="书斋", description="安静的内书房"),
        time_label="戌时",
    )
    await agent.evaluate_interrupt(
        step=1,
        reason="窗外有马蹄声",
        current_action_desc="读奏折",
        intent="",
        progress_hint="进行了一会儿",
    )
    # The call sends [system, user]; join both for the substring/order assertions (mirrors
    # test_decision._decision_prompt).
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    # first-person situation header, before the persona (at the start)
    assert "我此刻在书斋" in prompt and "时间为戌时" in prompt
    assert prompt.index("我此刻在书斋") < prompt.index("【我是谁】")
    # no leak of the code-layer location_id
    assert "study" not in prompt


@pytest.mark.asyncio
async def test_interrupt_prompt_carries_my_own_intent(container: object) -> None:
    """"What I'm after" must be present: seeing only the action and not its aim, the model can't
    tell whether the current task already leads to what the sudden signal wants (e.g. someone
    heading somewhere who receives "come here quickly" would abandon the trip)."""
    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    await agent.evaluate_interrupt(
        step=1,
        reason="收到李世民的急讯：速来布政坊。",
        current_action_desc="从崇仁坊前往布政坊",
        intent="我打算「抵达布政坊，面见秦王听候议事」。",
        progress_hint="已进行了约8小时，已过了大半",
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    assert "我打算「抵达布政坊，面见秦王听候议事」。" in prompt
    # Right inside the 【我正在做的事】 section, not a separate section
    assert prompt.index("从崇仁坊前往布政坊") < prompt.index("我打算「抵达布政坊")
    assert prompt.index("我打算「抵达布政坊") < prompt.index("【突然发生了】")


@pytest.mark.asyncio
async def test_interrupt_prompt_omits_intent_when_absent(container: object) -> None:
    """Recruited / never wrote an expectation → omit the whole line: no blank line and no
    half-finished "我打算"."""
    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    await agent.evaluate_interrupt(
        step=1, reason="某事", current_action_desc="读书", intent="", progress_hint="才刚开始",
    )
    user = provider.call_history[-1][-1].content
    assert "我打算" not in user
    assert "「读书」，才刚开始\n\n【突然发生了】" in user


@pytest.mark.asyncio
async def test_interrupt_prompt_states_what_dropping_costs(container: object) -> None:
    """Dropping it = the task is abandoned and the effort wasted. Without stating the cost, the
    model reads interrupt as "hurry over". It's a rule that doesn't vary across calls → must be in
    system (prefix cache)."""
    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    await agent.evaluate_interrupt(
        step=1, reason="某事", current_action_desc="读书", intent="", progress_hint="才刚开始",
    )
    system = provider.call_history[-1][0].content
    assert "到此作罢" in system and "白费" in system
    assert "本来就通向" in system


@pytest.mark.asyncio
async def test_interrupt_prompt_without_cached_spatial_omits_header(container: object) -> None:
    """Edge case: perceive_step hasn't run yet (_situation is an empty Situation()) → the header is
    omitted entirely, no exception."""
    from core.interfaces.perception import Situation
    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    assert agent._situation == Situation()
    await agent.evaluate_interrupt(
        step=1,
        reason="某事", current_action_desc="读书", intent="", progress_hint="才刚开始",
    )
    # The call sends [system, user]; join both for the substring/order assertions (mirrors
    # test_decision._decision_prompt).
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    # No header injected, but the prompt is still valid
    assert "时间为" not in prompt
    assert "【我是谁】" in prompt
    # Header omitted entirely → the user message's first block is 【我是谁】 directly (not the
    # header's "我此刻在…，当前时间为…" / "我此刻在某处"). The startswith assertion targets only the
    # user message; the constant system prefix comes before it and doesn't matter here.
    user_content = provider.call_history[-1][-1].content
    assert user_content.lstrip().startswith("【我是谁】")


@pytest.mark.asyncio
async def test_main_character_interrupt_missing_field_defaults_to_no_interrupt(
    container: object,
) -> None:
    """Valid JSON that omits the interrupt verdict → don't interrupt (couldn't
    determine a real decision; the conservative value is non-destructive)."""
    provider = MockLLMProvider(fixed_response='{"thought": "我在想"}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    should_interrupt, thought = await agent.evaluate_interrupt(
        step=1,
        reason="某事",
        current_action_desc="读书",
        intent="",
        progress_hint="进行了一会儿",
    )
    assert should_interrupt is False


@pytest.mark.asyncio
async def test_main_character_interrupt_llm_decision_no(container: object) -> None:
    provider = MockLLMProvider(fixed_response='{"thought": "不值得为这点动静停下", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    should_interrupt, thought = await agent.evaluate_interrupt(
        step=1,
        reason="外面有喧哗声",
        current_action_desc="冥思苦想",
        intent="",
        progress_hint="已进行了约1小时，进行了一会儿",
    )
    assert should_interrupt is False
    assert len(provider.call_history) == 1


@pytest.mark.asyncio
async def test_main_character_interrupt_llm_parse_failure_defaults_to_no_interrupt(
    container: object,
) -> None:
    provider = MockLLMProvider(fixed_response="这不是有效的JSON")
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    should_interrupt, thought = await agent.evaluate_interrupt(
        step=1,
        reason="未知事件",
        current_action_desc="写信",
        intent="",
        progress_hint="就快收尾了",
    )
    # Error / can't decide → don't interrupt (conservative, non-destructive): interrupting could
    # wrongly wreck an in-progress action, while the signal has been perceived and the external
    # goal stays, so the next re-plan handles it. Returns (False, "").
    assert should_interrupt is False
    assert thought == ""


@pytest.mark.asyncio
async def test_interrupt_prompt_injects_recent_factual_not_short_term_goals(container: object) -> None:
    """The interrupt decision weighs the sudden event against recent OBJECTIVE context,
    not the tactical short-term goal queue. The queue is redundant with the current action
    and stale here (no fresh re-derivation); recent factual memory situates whether the
    event escalates the local arc. Deep stakes (life-goal/values) stay via persona."""
    from types import SimpleNamespace

    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    # A distinctive short-term goal that must NOT appear (include_goals=False).
    agent.personality.state.short_term_goals = ["去偏殿清点府库"]
    # Two recent factual events, oldest first after chrono sort.
    agent.memory_system.sample_recent_events = lambda step, top_k=5: [  # type: ignore[method-assign]
        (SimpleNamespace(stored_content="我在朝堂上听见李建成与人低语", created_step=1, kind="event"), None),
        (SimpleNamespace(stored_content="我瞥见东宫侍卫悄然调动", created_step=3, kind="event"), None),
    ]

    await agent.evaluate_interrupt(
        step=4,
        reason="有人来报，李建成在东宫设宴",
        current_action_desc="批阅奏折",
        intent="",
        progress_hint="进行了一会儿",
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    # Recent factual memory is injected, chronological, under the memory-order contract.
    assert "东宫侍卫悄然调动" in prompt
    assert "李建成与人低语" in prompt
    assert MEMORY_ORDER_HINT in prompt
    assert prompt.index("李建成与人低语") < prompt.index("东宫侍卫悄然调动")  # oldest first
    # The tactical short-term goal queue is left out.
    assert "去偏殿清点府库" not in prompt
    assert "短期目标" not in prompt


# ---------------------------------------------------------------------------
# Own condition: how a bound person weighs an interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_prompt_states_my_own_standing_condition(container: object) -> None:
    """A bound person weighing "should I drop what I'm doing" must know he can't move.

    This thought goes through engine/narration.format_interrupt_reason_3p into the outcome, which
    is persisted and embedded. Without this line the model plans to lunge or rush out: a fabricated
    fact about his own bound body.

    The condition is read from ``self.personality.state``, like _situation; no perception packet is
    needed.
    """
    from core.interfaces.condition import BodyCondition
    from core.interfaces.perception import LocationView, Situation

    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    agent._situation = Situation(
        location_view=LocationView(name="玄武门", description="宫城北门"),
        time_label="寅时",
    )
    agent.personality.set_condition(
        BodyCondition(description="双手被反绑", source_agent_id="other", since_step=10),
    )
    await agent.evaluate_interrupt(
        step=34,
        reason="宫门方向骤起兵刃相击",
        current_action_desc="屏息养力",
        intent="",
        progress_hint="进行了一会儿",
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    # Same label and same duration-bearing form as the first line of decision's 【我所处的现实】.
    assert "我此刻的处境：双手被反绑（已持续约1天）" in prompt
    # Right after the space-time anchor and before the persona — "where I am, what state I'm in"
    # is one block.
    assert prompt.index("我此刻在玄武门") < prompt.index("我此刻的处境") < prompt.index("【我是谁】")
    # Narrative-layer text: no step numbers, no ids.
    assert "34" not in prompt and "10" not in prompt and "other" not in prompt


@pytest.mark.asyncio
async def test_interrupt_prompt_omits_the_line_when_unencumbered(container: object) -> None:
    """No condition is the norm for most people most of the time — omit the whole line rather than
    render noise like "处境：无"."""
    from core.interfaces.perception import LocationView, Situation

    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    agent._situation = Situation(
        location_view=LocationView(name="玄武门", description="宫城北门"), time_label="寅时",
    )
    await agent.evaluate_interrupt(
        step=34, reason="某事", current_action_desc="读书", intent="", progress_hint="才刚开始",
    )
    prompt = "\n".join(m.content for m in provider.call_history[-1])
    # Assert on the label itself, not the word "处境" — the constant system text contains
    # "它们不随处境改变".
    assert "我此刻的处境" not in prompt
    assert "我此刻在玄武门" in prompt


@pytest.mark.asyncio
async def test_condition_alone_still_anchors_the_prompt(container: object) -> None:
    """Edge case: condition present but no space-time anchor (the extreme state before perceive) →
    still emit the condition line; don't collapse to an empty block or raise."""
    from core.interfaces.condition import BodyCondition
    from core.interfaces.perception import Situation

    provider = MockLLMProvider(fixed_response='{"thought": "...", "interrupt": false}')
    agent = _make_agent(container, is_main=True, llm_provider=provider)
    assert agent._situation == Situation()
    agent.personality.set_condition(BodyCondition(description="昏迷不醒", since_step=0))
    await agent.evaluate_interrupt(
        step=2, reason="某事", current_action_desc="读书", intent="", progress_hint="才刚开始",
    )
    user = provider.call_history[-1][-1].content
    assert user.lstrip().startswith("我此刻的处境：昏迷不醒")
    assert "【我是谁】" in user
