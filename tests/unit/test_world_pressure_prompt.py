"""Unit tests for the per-agent WorldPressureEvaluator.

The evaluator builds one prompt per agent containing ONLY that agent's
perceivable signals, so information boundaries (message recipient, broadcast
location) are enforced structurally rather than by instruction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.interfaces.llm import IndexedRef, LLMResponse
from core.interfaces.perception import (
    AmbientEvent,
    Broadcast,
    BroadcastType,
    LocationView,
    PerceivedIdentity,
    PerceivedPresence,
    SpatialPerception,
)
from core.interfaces.urgency import Urgency
from engine.world_pressure import (
    _GOAL_TEXT_MAX_CHARS,
    _MAX_TOKENS,
    _REASON_MAX_CHARS,
    _SYSTEM_PROMPT,
    WorldPressureEvaluator,
)


class _StubStore:
    """Agent store stub: no relations (load_relation → None)."""
    async def load_relation(self, world_id, from_id, to_id):  # noqa: ANN001, D102
        return None


def _stub_agent(
    *,
    name: str,
    role: str,
    location: str,
    activity_label: str = "空闲",
    core_traits: tuple[str, ...] = (),
    core_values: tuple[str, ...] = (),
    background: str = "",
    gender: str = "男",
    agent_id: str | None = None,
    condition=None,
    is_active: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        world_id="w",
        agent_id=agent_id or name,
        is_active=is_active,
        agent_store=_StubStore(),
        personality=SimpleNamespace(
            soul=SimpleNamespace(
                name=name, role=role, gender=gender, core_traits=tuple(core_traits),
                core_values=tuple(core_values), background=background, secret="",
            ),
            state=SimpleNamespace(
                current_location=location,
                activity_status=SimpleNamespace(label=activity_label),
                # Match the real StateLayer: the field always exists and defaults to None. A stub missing a field
                # would force production code into getattr guards, bending product code to fit a test double.
                condition=condition,
            ),
        ),
    )


def _spatial(*, location: str, visible: dict[str, str] | None = None, ambient: list[str] | None = None) -> SpatialPerception:
    visible = visible or {}
    return SpatialPerception(
        location_id=location,
        location_view=LocationView(name=location, description=""),
        world_time_label="辰时",
        current_step=1,
        visible_agents={k: PerceivedPresence(PerceivedIdentity(name=v)) for k, v in sorted(visible.items())},
        ambient_events=[AmbientEvent(content=text) for text in (ambient or [])],
    )


def _msg(
    content: str, *, sender_id: str = "x", sender_name: str = "某人",
    sender_is_agent: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, sender_id=sender_id, sender_name=sender_name,
        sender_is_agent=sender_is_agent,
    )


def _bc(content: str, *, location_scope: str | None = None, severity: str = "low") -> Broadcast:
    return Broadcast(
        content=content, source="system", broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1, location_scope=location_scope, severity=severity,
    )


def _ev() -> WorldPressureEvaluator:
    return WorldPressureEvaluator(llm_router=None)


# ---------------------------------------------------------------------------
# _build_agent_prompt
# ---------------------------------------------------------------------------

def test_agent_prompt_separates_reference_from_perceived_and_carries_relations() -> None:
    ev = _ev()
    agent = _stub_agent(name="李世民", role="秦王", location="xuanwu", core_traits=("果决",), background="战功赫赫。")
    co_located = [{"id": "a2", "name": "李建成", "role": "太子", "gender": "男",
                   "traits": ["隐忍", "权谋"], "background": "正统储君。",
                   "trust": 0.1, "affection": -0.6, "labels": ["兄弟:敌对"]}]
    prompt = ev._build_agent_prompt(
        agent, inbox=[_msg("速来", sender_name="长孙无忌")],
        broadcasts=[_bc("喊杀骤起", severity="high")], ambient=[AmbientEvent(content="地上有血迹")],
        co_located=co_located, sender_relations={}, situation_header="当前时间：辰时；当前地点：玄武门——皇城北门。",
    )
    # target info is framed as reference, not pressure source
    assert "【目标角色（仅供识人参考，不是压力来源）】" in prompt
    # The current time and place come from the shared THIRD header; the location isn't inlined into the character descriptor (that would render it twice)
    assert "当前时间：辰时；当前地点：玄武门——皇城北门。" in prompt
    # Gender shares the identity parenthesis: the third-party assessment writes about the pressure he or she is under, and gender decides the pronouns and forms of address.
    assert "李世民（男，秦王，" in prompt and "位于" not in prompt
    assert "性格：果决" in prompt
    # perceived content is framed as the only pressure source
    assert "【该角色此刻感知到的内容（外部压力的唯一来源）】" in prompt
    # The severity code enum is translated into a natural word; "severity=high" must not leak
    assert "广播[全局，重大] 喊杀骤起" in prompt
    assert "severity=" not in prompt
    assert "收到来自长孙无忌的消息：速来" in prompt
    assert "观察到：地上有血迹" in prompt
    # co-located: role + relation (via shared format_relation_block) + personality + background
    assert "#1 李建成（男，太子）" in prompt
    assert "[兄弟:敌对]" in prompt and "信任度0.10" in prompt and "好感度-0.60" in prompt
    assert "其性格：隐忍、权谋" in prompt
    assert "其背景：正统储君。" in prompt


def test_agent_prompt_shows_target_reference_and_empty_perceived() -> None:
    ev = _ev()
    agent = _stub_agent(name="路人", role="无", location="街市")
    prompt = ev._build_agent_prompt(
        agent, inbox=[], broadcasts=[], ambient=[], co_located=[], sender_relations={},
        situation_header="当前时间：辰时；当前地点：街市。",
    )
    # target reference block is always present (identity is reference context)
    assert "【目标角色（仅供识人参考，不是压力来源）】" in prompt
    # with no perceivable signals the perceived section is "（无）"
    assert "【该角色此刻感知到的内容（外部压力的唯一来源）】" in prompt
    assert "（无）" in prompt
    assert "同处一地" not in prompt


def test_agent_prompt_renders_remote_sender_relation_inline() -> None:
    """A message sender's relation is shown inline so the pressure can be judged by who sent it;
    the relation legend appears even with no co-located block."""
    ev = _ev()
    agent = _stub_agent(name="李建成", role="太子", location="donggong")
    prompt = ev._build_agent_prompt(
        agent, inbox=[_msg("速来东宫", sender_id="a_shimin", sender_name="李世民")],
        broadcasts=[], ambient=[], co_located=[],
        sender_relations={"a_shimin": "[兄弟:敌对]，信任度0.10、好感度-0.60"},
        situation_header="当前时间：辰时；当前地点：东宫。",
    )
    assert "收到来自李世民的消息：速来东宫" in prompt
    assert "该角色与李世民的关系：[兄弟:敌对]，信任度0.10、好感度-0.60" in prompt
    # legend shows even though there is no co-located block
    assert "涉及的关系字段含义" in prompt
    assert "同处一地" not in prompt


class _RelStore:
    """Agent store stub returning a fixed objective relation per target id (None = unknown)."""
    def __init__(self, rels: dict[str, SimpleNamespace]) -> None:
        self._rels = rels

    async def load_relation(self, world_id, from_id, to_id):  # noqa: ANN001, D102
        return self._rels.get(to_id)


def _rel(*, trust: float, affection: float, labels: list[str]) -> SimpleNamespace:
    return SimpleNamespace(trust_objective=trust, affection_objective=affection, labels=labels)


@pytest.mark.asyncio
async def test_sender_relations_resolves_remote_skips_non_agents_and_colocated() -> None:
    """Only senders one can form a relation with are resolved, judged by the message's own ``sender_is_agent``, not an id list.

    Narration, the director and bodies without cognition each have their own ids. With a list, any
    id it misses renders as "关系未明确", asking the LLM to weigh a closeness that by design never exists.
    """
    ev = _ev()
    agent = _stub_agent(name="李建成", role="太子", location="donggong")
    agent.agent_store = _RelStore({"a_shimin": _rel(trust=0.1, affection=-0.6, labels=["兄弟:敌对"])})
    inbox = [
        _msg("速来", sender_id="a_shimin", sender_name="李世民"),    # remote agent → resolved
        _msg("传旨", sender_id="narrator", sender_name="旁白", sender_is_agent=False),
        _msg("他在城西", sender_id="director", sender_name="不知来源", sender_is_agent=False),
        _msg("话带到了", sender_id="npc_1", sender_name="禁军屯卫", sender_is_agent=False),
        _msg("近前", sender_id="a_here", sender_name="侍卫"),         # co-located → excluded
    ]
    out = await ev._sender_relations(agent, inbox, exclude_ids={"a_here"})
    assert set(out.keys()) == {"a_shimin"}
    assert "信任度0.10" in out["a_shimin"] and "[兄弟:敌对]" in out["a_shimin"]


@pytest.mark.asyncio
async def test_sender_relations_unknown_sender_renders_undefined() -> None:
    ev = _ev()
    agent = _stub_agent(name="李建成", role="太子", location="donggong")
    agent.agent_store = _RelStore({})  # no relation known for the sender
    out = await ev._sender_relations(
        agent, [_msg("你是谁", sender_id="stranger", sender_name="陌生人")], exclude_ids=set(),
    )
    assert out["stranger"] == "关系未明确"


# ---------------------------------------------------------------------------
# _parse_agent_goals
# ---------------------------------------------------------------------------

def test_one_field_one_cap() -> None:
    """One limit per field. Writing one number in the field description and another in the output
    schema makes two sources of truth, and the model only obeys the one nearest the generation point
    (§3). The budget is then estimated from the other number, and truncation hits exactly when
    pressure is highest. Both places interpolate the same constant.
    """
    import re

    assert re.findall(r"≤(\d+)字", _SYSTEM_PROMPT) == [
        str(_GOAL_TEXT_MAX_CHARS),   # text in 【字段说明】
        str(_REASON_MAX_CHARS),      # reason in the schema
        str(_GOAL_TEXT_MAX_CHARS),   # text in the schema; must use the same constant as the first
    ]


def test_the_token_budget_covers_the_caps_the_prompt_declares() -> None:
    """Raising a limit without raising the budget is the classic truncation trap, and truncation yields unparseable JSON (all pressure lost).

    Compute it the CLAUDE.md way: Chinese at 1.5 tok/char, ~3 per number/enum, ~5 structure per
    field, then ×3 (50% buffer, ×2 for thinking). Changing any limit makes this fire.
    """
    reason = _REASON_MAX_CHARS * 1.5
    per_goal = _GOAL_TEXT_MAX_CHARS * 1.5 + 4 * 3 + 5 * 5   # text + 4 numbers/enums + 5 fields of structure
    estimate = reason + 3 * per_goal + 10                   # goals capped at 3 + outer wrapper
    assert _MAX_TOKENS >= 3 * estimate, f"预估 {estimate:.0f} tok,需 ≥{3 * estimate:.0f}"


def test_parse_agent_goals_basic_and_source() -> None:
    ev = _ev()
    ref = IndexedRef(["a2", "a3"])  # co-located others
    raw = ('{"goals": [{"text": "迎战", "urgency": "high", "drive_type": "threat", '
           '"related_need": "safety", "source": 1}]}')
    goals = ev._parse_agent_goals(raw, source_ref=ref)
    assert len(goals) == 1
    assert goals[0].text == "迎战" and goals[0].urgency == Urgency.HIGH
    assert goals[0].source_id == "a2"  # source=1 → ref[0]


def test_parse_agent_goals_source_zero_is_world() -> None:
    ev = _ev()
    raw = '{"goals": [{"text": "应对", "urgency": "normal", "drive_type": "event", "related_need": null, "source": 0}]}'
    goals = ev._parse_agent_goals(raw, source_ref=IndexedRef([]))
    assert goals[0].source_id == "world"


def test_parse_agent_goals_fallbacks() -> None:
    from agent.motivation import ExternalDriveType
    ev = _ev()
    raw = '{"goals": [{"text": "做事", "urgency": "very-high", "drive_type": "unknown_xyz", "related_need": null, "source": 0}]}'
    goals = ev._parse_agent_goals(raw, source_ref=IndexedRef([]))
    assert goals[0].urgency == Urgency.NORMAL  # unknown urgency → NORMAL
    assert goals[0].drive_type == ExternalDriveType.EVENT  # unknown drive → EVENT


def test_parse_agent_goals_self_actualization() -> None:
    from agent.need import NeedType
    ev = _ev()
    raw = '{"goals": [{"text": "成就大业", "urgency": "high", "drive_type": "event", "related_need": "self_actualization", "source": 0}]}'
    goals = ev._parse_agent_goals(raw, source_ref=IndexedRef([]))
    assert goals[0].related_need == NeedType.SELF_ACTUALIZATION


def test_a_goal_written_as_a_bare_string_is_kept_not_dropped() -> None:
    """The model shortens a pressure item to a bare string (perfectly legal under json_mode). That string is the required text; the rest has defaults.

    Dropping it drops a real signal: "设法离开东宫" is the pressure itself, not format noise.
    """
    ev = _ev()
    raw = '{"reason": "局势紧张", "goals": ["设法离开东宫", "去见秦王"]}'
    goals = ev._parse_agent_goals(raw, source_ref=IndexedRef([]))
    assert [g.text for g in goals] == ["设法离开东宫", "去见秦王"]
    assert all(g.urgency == Urgency.NORMAL and g.source_id == "world" for g in goals)


def test_one_unparseable_item_does_not_take_the_whole_agent_s_pressure_with_it(caplog) -> None:
    """Per-item skipping must catch everything: the bad item is dropped, the good ones kept.

    This loop feeds _evaluate_one's return, which is outside its try. An escaping exception is
    recorded by gather as the whole agent's evaluation failing, so losing one item becomes losing a
    round, and pressure feeds the dominant_need override and interrupt decisions. So the catch
    can't enumerate exception types: missing one type has exactly that cost.
    """
    ev = _ev()
    raw = ('{"goals": [{"text": "守住玄武门"}, 123, null, "去见秦王", {"no_text": 1}]}')
    with caplog.at_level("WARNING", logger="engine.world_pressure"):
        goals = ev._parse_agent_goals(raw, source_ref=IndexedRef([]), agent_id="a1")
    assert [g.text for g in goals] == ["守住玄武门", "去见秦王"]
    # A drop must be logged and say whose item it was; skipping silently means nobody notices when the model systematically changes its format.
    dropped = [r for r in caplog.records if r.message == "world_pressure_goal_dropped"]
    assert [r.reason for r in dropped] == ["not_an_object", "not_an_object", "unparseable"]
    assert all(r.agent_id == "a1" for r in dropped)


def test_parse_agent_goals_invalid_or_non_list() -> None:
    ev = _ev()
    assert ev._parse_agent_goals("not json", source_ref=IndexedRef([])) == []
    assert ev._parse_agent_goals('{"pressures": []}', source_ref=IndexedRef([])) == []  # old format → no "goals"


def test_parse_agent_goals_no_truncation() -> None:
    ev = _ev()
    goals_json = ", ".join(
        f'{{"text": "g{i}", "urgency": "normal", "drive_type": "event", "related_need": null, "source": 0}}'
        for i in range(5)
    )
    goals = ev._parse_agent_goals(f'{{"goals": [{goals_json}]}}', source_ref=IndexedRef([]))
    assert len(goals) == 5


# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT
# ---------------------------------------------------------------------------

def test_system_prompt_single_agent_and_theme_neutral() -> None:
    assert "单个目标角色" in _SYSTEM_PROMPT  # per-agent, third-party assessment
    assert "第三者" in _SYSTEM_PROMPT
    assert '"goals"' in _SYSTEM_PROMPT  # per-agent output schema
    assert "pressures" not in _SYSTEM_PROMPT  # no holistic multi-agent array
    # Think before acting (CLAUDE.md §5, functional judge): reason is the schema's first key, before goals.
    assert '"reason"' in _SYSTEM_PROMPT
    assert _SYSTEM_PROMPT.index('"reason"') < _SYSTEM_PROMPT.index('"goals"')
    # pressure source = perceived content; agent info is reference only
    assert "本身不是压力来源" in _SYSTEM_PROMPT
    for token in ["皇帝", "统帅", "守将", "太子", "秦王", "玄武", "长安", "大唐"]:
        assert token not in _SYSTEM_PROMPT


def test_system_prompt_related_need_tokens_and_descriptions() -> None:
    from agent.need import NeedType
    for need in NeedType:
        assert f'"{need.value}"' in _SYSTEM_PROMPT
        assert need.description in _SYSTEM_PROMPT
    assert "autonomy" not in _SYSTEM_PROMPT  # phantom value stays absent


# ---------------------------------------------------------------------------
# evaluate() — per-agent gating + structural boundaries
# ---------------------------------------------------------------------------

class _RecordingLLM:
    def __init__(self, response: str = '{"goals": []}') -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self._response = response

    async def complete(self, scene, messages, temperature=0.3, max_tokens=600,
        **kwargs,
    ) -> LLMResponse:
        self.calls += 1
        self.prompts.append(messages[-1].content)
        return LLMResponse(content=self._response, input_tokens=0, output_tokens=0, model="test")


@pytest.mark.asyncio
async def test_evaluate_calls_once_per_signal_bearing_agent() -> None:
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall")}
    # both perceive a global broadcast → 2 calls
    await ev.evaluate(agents=agents, agent_inboxes={"a1": [], "a2": []},
                      broadcasts=[_bc("天下大乱", severity="high")], world_time_label="辰时",
                      agent_spatials={"a1": _spatial(location="hall"), "a2": _spatial(location="hall")})
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_evaluate_skips_dead_agent_even_on_global_broadcast() -> None:
    """World-wide broadcasts (e.g. a death notice) ignore location: unless the dead are filtered out, they'd be assessed a pressure nobody can bear."""
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall", is_active=False)}
    result = await ev.evaluate(agents=agents, agent_inboxes={"a1": [], "a2": []},
                               broadcasts=[_bc("丙已身亡", severity="high")], world_time_label="辰时",
                               agent_spatials={"a1": _spatial(location="hall"), "a2": _spatial(location="hall")})
    assert llm.calls == 1
    assert "a2" not in result


@pytest.mark.asyncio
async def test_evaluate_skips_pure_co_location() -> None:
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall")}
    spatials = {"a1": _spatial(location="hall", visible={"a2": "乙"}),
                "a2": _spatial(location="hall", visible={"a1": "甲"})}
    result = await ev.evaluate(agents=agents, agent_inboxes={"a1": [], "a2": []}, broadcasts=[],
                               world_time_label="辰时", agent_spatials=spatials)
    assert result == {} and llm.calls == 0


@pytest.mark.asyncio
async def test_evaluate_ambient_only_triggers_that_agent() -> None:
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall")}
    await ev.evaluate(agents=agents, agent_inboxes={"a1": []}, broadcasts=[], world_time_label="辰时",
                      agent_spatials={"a1": _spatial(location="hall", ambient=["地上有血迹"])})
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_evaluate_location_broadcast_only_reaches_in_location() -> None:
    """Structural location boundary: an agent NOT at the scope never sees the broadcast."""
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hallA"),
              "a2": _stub_agent(name="乙", role="无", location="hallB")}
    spatials = {"a1": _spatial(location="hallA"), "a2": _spatial(location="hallB")}
    await ev.evaluate(agents=agents, agent_inboxes={"a1": [], "a2": []},
                      broadcasts=[_bc("仅A厅事件", location_scope="hallA", severity="high")],
                      world_time_label="辰时", agent_spatials=spatials)
    assert llm.calls == 1  # only a1 (at hallA) perceives it; a2 skipped
    assert "仅A厅事件" in llm.prompts[0]


@pytest.mark.asyncio
async def test_evaluate_does_not_leak_other_agents_private_message() -> None:
    """Structural recipient boundary: a co-located non-recipient's prompt omits the message."""
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall")}
    spatials = {"a1": _spatial(location="hall", visible={"a2": "乙"}),
                "a2": _spatial(location="hall", visible={"a1": "甲"})}
    # a1 gets a private message; both perceive a global broadcast (so both get evaluated).
    await ev.evaluate(
        agents=agents,
        agent_inboxes={"a1": [_msg("只给甲的密信", sender_name="信使")], "a2": []},
        broadcasts=[_bc("全局大事", severity="high")], world_time_label="辰时", agent_spatials=spatials,
    )
    assert llm.calls == 2
    a2_prompt = next(p for p in llm.prompts if "\n乙（" in p)  # "乙" is the target in this prompt
    assert "只给甲的密信" not in a2_prompt  # private message must not leak to "乙"


@pytest.mark.asyncio
async def test_evaluate_attributes_each_call_to_its_agent() -> None:
    """Concurrent per-agent assessments each carry agent_id and the pressure stage, so the trace can attribute them per agent."""
    from core.context import get_log_context

    seen: list[tuple[str, str]] = []

    class _ContextLLM(_RecordingLLM):
        async def complete(self, scene, messages, temperature=0.3, max_tokens=600, **kwargs):
            ctx = get_log_context()
            seen.append((ctx.get("agent_id", ""), ctx.get("stage", "")))
            return await super().complete(scene, messages, temperature, max_tokens, **kwargs)

    ev = WorldPressureEvaluator(llm_router=_ContextLLM())
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall")}
    await ev.evaluate(agents=agents, agent_inboxes={"a1": [], "a2": []},
                      broadcasts=[_bc("天下大乱", severity="high")], world_time_label="辰时",
                      agent_spatials={"a1": _spatial(location="hall"), "a2": _spatial(location="hall")})
    assert sorted(seen) == [("a1", "pressure"), ("a2", "pressure")]
