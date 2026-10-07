"""How a physical ruling declares a condition — and how it declares nothing when the ruling never
happened.

A condition persists and keeps entering prompts, so it tolerates fabrication even less than a
one-off outcome: an invented "hands tied behind his back" makes everyone weigh it on every later
step, and no step ever corrects it. So both sides are locked here: what the judge says lands, and
when the judge couldn't speak, nothing lands.
"""

from __future__ import annotations

import json

import pytest

from agent.personality import PersonalityLayer, SoulLayer
from core.interfaces.action import Deed
from core.interfaces.condition import BodyCondition
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.executors.physical import PhysicalExecutor

BOUND = BodyCondition(description="双手被反绑", source_agent_id="a1", since_step=10)


class _ScriptedLLM:
    """Replies from a script and records what it was asked."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.messages: list = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001
        self.messages.append(messages)

        class _R:
            content = self.payload

        return _R()


class _Rel:
    async def load_existing(self, _):  # noqa: ANN001
        return type("R", (), {"labels": [], "trust_objective": 0.5, "affection_objective": 0.0})()


class _Agent:
    def __init__(self, agent_id: str, name: str, condition=None) -> None:
        self.agent_id = agent_id
        self.personality = PersonalityLayer(soul=SoulLayer(name=name, gender="男", agent_id=agent_id))
        if condition is not None:
            self.personality.set_condition(condition)
        self.relation_system = _Rel()


def _executor(payload: str) -> tuple[PhysicalExecutor, _ScriptedLLM]:
    llm = _ScriptedLLM(payload)
    ex = PhysicalExecutor(
        llm, LiveWorldDirectory.from_agents({}, EnvironmentSystem()), seconds_per_step=3600,
    )
    return ex, llm


def _verdict_payload(**over) -> str:
    base = {
        "reason": "他已被缚，无从抵抗", "deed": "restrain", "success": True,
        "outcome": "甲把乙的双腿也捆上。", "fact": "我把乙的双腿也捆上。", "why": "",
        "relation": "negative", "target_damage": 0.0, "actor_damage": 0.0,
    }
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


async def _judge(payload: str, *, is_person: bool = True, step: int = 34):
    ex, llm = _executor(payload)
    verdict = await ex._judge(
        _Agent("a1", "甲"), description="再捆他的腿", expected_outcome="", scene="",
        target_block="乙（人）", is_person=is_person, has_entity=False,
        seize_available=False, actor_holds=False,
        needs_relation=is_person, recipient_line="", step=step,
    )
    return verdict, llm


# ---------------------------------------------------------------------------
# Ruling input: both parties' conditions must be present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_target_block_states_the_target_condition_with_duration() -> None:
    """The judge must know the target is already restrained, or it weighs his resistance as a free
    man's; duration is a real basis for the ruling ("tied for three days, the guards should be
    slacking")."""
    ex, _ = _executor(_verdict_payload())
    block = await ex._person_target_block(
        _Agent("a1", "甲"), _Agent("a2", "乙", BOUND), "a1", "a2", step=34,
    )
    assert "处境：双手被反绑（已持续约1天）" in block


@pytest.mark.asyncio
async def test_the_actor_block_states_the_actor_condition() -> None:
    """Basis for the escape valve: a bound person choosing PHYSICAL is struggling — the judge must
    know what he's struggling against and for how long to rule whether he breaks free."""
    ex, llm = _executor(_verdict_payload())
    await ex._judge(
        _Agent("a2", "乙", BOUND), description="奋力挣脱绳索", expected_outcome="", scene="",
        target_block="绳索", is_person=False, has_entity=False, seize_available=False,
        actor_holds=False, needs_relation=False, recipient_line="", step=34,
    )
    user = llm.messages[0][1].content
    assert "处境：双手被反绑（已持续约1天）" in user


@pytest.mark.asyncio
async def test_an_unconditioned_party_adds_no_line_at_all() -> None:
    """No condition omits the whole line — "处境：无" would add a line of noise to every ruling
    for everyone."""
    ex, llm = _executor(_verdict_payload())
    await ex._judge(
        _Agent("a1", "甲"), description="推他一把", expected_outcome="", scene="",
        target_block="乙（人）", is_person=True, has_entity=False, seize_available=False,
        actor_holds=False, needs_relation=True, recipient_line="", step=3,
    )
    assert "处境：" not in llm.messages[0][1].content


@pytest.mark.asyncio
async def test_the_step_unit_is_taught_inside_the_field_that_needs_it() -> None:
    """The LLM may output step counts only when the prompt explicitly teaches the unit
    (controlled upstream channel).

    The unit must live inside the *_condition_steps bullet itself: an isolated bullet floating
    above the field list can only say "the step mentioned above…" while the field is below it — a
    dead reference pointing the wrong way. Each group has its own steps and teaches it once.
    """
    _, llm = await _judge(_verdict_payload())
    system = llm.messages[0][0].content
    for field in ("target_condition_steps", "actor_condition_steps"):
        bullet = [ln for ln in system.splitlines() if ln.startswith(f"- {field}：")]
        assert len(bullet) == 1
        assert "一步约1小时" in bullet[0]
        assert "上面" not in bullet[0]


@pytest.mark.asyncio
async def test_each_condition_field_gets_its_own_line() -> None:
    """One bullet per field, same shape as - outcome / - fact / - deed / - why (§4: discrete items
    are individually addressable).

    Don't squeeze them into one bullet with `·` sub-items: sub-items are soft-wrapped for source
    readability, and those wraps land verbatim in the prompt.
    """
    _, llm = await _judge(_verdict_payload())
    section = llm.messages[0][0].content.split("【输出】")[1].split('{"reason"')[0]
    lines = [ln for ln in section.splitlines() if ln.strip()]
    for field in ("target_condition", "target_condition_steps", "frees_target",
                  "actor_condition", "actor_condition_steps", "frees_actor"):
        assert sum(ln.startswith(f"- {field}：") for ln in lines) == 1
    assert not any(ln.startswith(("  ", "\t", "·")) or ln.lstrip().startswith("·") for ln in lines), \
        "输出段里出现了续行/子项——这一段每条都该是独立单行"


# ---------------------------------------------------------------------------
# Ruling output: parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_declared_condition_reaches_the_verdict() -> None:
    verdict, _ = await _judge(
        _verdict_payload(target_condition="手脚俱被反绑", target_condition_steps=0)
    )
    assert verdict.target_condition_desc == "手脚俱被反绑"
    assert verdict.target_condition_steps == 0
    assert verdict.frees_target is False
    assert verdict.deed == Deed.RESTRAIN


@pytest.mark.asyncio
async def test_the_actor_condition_is_a_slot_of_its_own() -> None:
    """A ruling's consequence can land on the actor (a bystander pins him down on the spot) — that
    isn't the target's condition. With only one subject-less slot, the judge can only put it in the
    target column, so A's condition gets permanently stuck on B."""
    verdict, _ = await _judge(
        _verdict_payload(actor_condition="被反剪双臂压在地上", actor_condition_steps=0)
    )
    assert verdict.actor_condition_desc == "被反剪双臂压在地上"
    assert verdict.target_condition_desc == ""


@pytest.mark.asyncio
async def test_the_actor_axis_survives_an_entity_verdict() -> None:
    """What lands on the actor himself doesn't depend on whether he acted on a person or a thing
    (caught in a door while prying it, slipping out of ropes — neither goes through a target) — so
    this group is always present. It's also the escape valve's only landing point."""
    verdict, llm = await _judge(
        _verdict_payload(actor_condition="右手被门夹伤", frees_actor=True), is_person=False,
    )
    assert verdict.actor_condition_desc == "右手被门夹伤"
    assert verdict.frees_actor is True
    assert '"actor_condition"' in llm.messages[0][0].content


@pytest.mark.asyncio
async def test_a_silent_verdict_declares_nothing() -> None:
    """Most physical actions leave nothing behind — the default must read as "unchanged", not
    "clear"."""
    verdict, _ = await _judge(_verdict_payload())
    assert verdict.target_condition_desc == ""
    assert verdict.actor_condition_desc == ""
    assert verdict.frees_target is False
    assert verdict.frees_actor is False


@pytest.mark.asyncio
async def test_an_entity_verdict_never_carries_a_target_condition() -> None:
    """The thing branch goes through EntityStateChange. Even if the LLM emits extra, discard it, so
    "the door was smashed open" isn't taken as someone's condition; the schema shouldn't ask for it
    either."""
    verdict, llm = await _judge(
        _verdict_payload(target_condition="门被撞开了", frees_target=True), is_person=False,
    )
    assert verdict.target_condition_desc == ""
    assert verdict.frees_target is False
    assert '"target_condition"' not in llm.messages[0][0].content


@pytest.mark.asyncio
async def test_a_long_condition_is_not_re_cut_by_code() -> None:
    """The prompt already says ≤15 characters; cutting again in code only produces broken sentences
    and creates a second source of truth."""
    long_text = "双手被反绑并以铁链锁在庭中石柱之上动弹不得"
    verdict, _ = await _judge(_verdict_payload(target_condition=long_text))
    assert verdict.target_condition_desc == long_text


# ---------------------------------------------------------------------------
# Failure floor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_unparseable_verdict_writes_no_condition() -> None:
    """When the ruling never happened at all (Rule 1 tier-1), invent no condition — fabricating a
    persistent condition that gets read again and again is far worse than nothing happening this
    step."""
    verdict, _ = await _judge("这不是 JSON")
    assert verdict is None
