"""Unit tests for the per-agent WorldPressureEvaluator.

The evaluator builds one prompt per agent containing ONLY what that agent knows (its signals,
its own memories, those present as it perceives them), so information boundaries (message
recipient, broadcast location) are enforced structurally rather than by instruction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.interfaces.llm import LLMResponse
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
from engine.signals import perceived_signals
from engine.world_pressure import (
    _GOAL_TEXT_MAX_CHARS,
    _MAX_TOKENS,
    _REASON_MAX_CHARS,
    _SYSTEM_PROMPT,
    WorldPressureEvaluator,
)


class _StubRelations:
    """Relation system stub: no relations (load_existing → None)."""
    async def load_existing(self, target_agent_id):  # noqa: ANN001, D102
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
    reads_world_pressure: bool = True,
    recent_past: tuple[str, ...] = (),
) -> SimpleNamespace:
    cues_seen: list[list[str]] = []
    present_seen: list[list[str]] = []

    async def pressure_recent_past(step, *, cues, present=()):  # noqa: ANN001, ANN202
        cues_seen.append(list(cues))
        present_seen.append(list(present))
        return list(recent_past)

    return SimpleNamespace(
        world_id="w",
        agent_id=agent_id or name,
        is_active=is_active,
        reads_world_pressure=reads_world_pressure,
        relation_system=_StubRelations(),
        pressure_recent_past=pressure_recent_past,
        cues_seen=cues_seen,
        present_seen=present_seen,
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


def _spatial(
    *, location: str, visible: dict[str, str] | None = None, ambient: list[str] | None = None,
    ambient_doers: tuple[str, ...] = (),
) -> SpatialPerception:
    visible = visible or {}
    return SpatialPerception(
        location_id=location,
        location_view=LocationView(name=location, description=""),
        world_time_label="辰时",
        current_step=1,
        visible_agents={k: PerceivedPresence(PerceivedIdentity(name=v, gender="男")) for k, v in sorted(visible.items())},
        ambient_events=[
            AmbientEvent(content=text, doer_ids=ambient_doers, doer_agent_ids=ambient_doers)
            for text in (ambient or [])
        ],
    )


def _msg(
    content: str, *, sender_id: str = "x", sender_name: str = "某人",
    sender_is_agent: bool = True, anonymous: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, sender_id=sender_id, sender_name=sender_name,
        sender_is_agent=sender_is_agent, anonymous=anonymous,
    )


def _bc(content: str, *, location_scope: str | None = None, severity: str = "low") -> Broadcast:
    return Broadcast(
        content=content, source="system", broadcast_type=BroadcastType.WORLD_EVENT,
        deliver_step=1, location_scope=location_scope, severity=severity,
    )


def _ev() -> WorldPressureEvaluator:
    return WorldPressureEvaluator(llm_router=None)


def _prompt(
    ev: WorldPressureEvaluator, agent, *, inbox=(), broadcasts=(), ambient=(), acted_upon=(),  # noqa: ANN001
    co_located=(), sender_relations=None, situation_header="", recent_past=(), doing="",
    visible_ids=frozenset(),
) -> str:
    signals = ev._signals(
        inbox=list(inbox), broadcasts=list(broadcasts), ambient=list(ambient),
        acted_upon=list(acted_upon), sender_relations=sender_relations or {}, visible_ids=set(visible_ids),
    )
    return ev._build_agent_prompt(
        agent, signals=signals, co_located=list(co_located), recent_past=recent_past, doing=doing,
        situation_header=situation_header,
    )


# ---------------------------------------------------------------------------
# _build_agent_prompt
# ---------------------------------------------------------------------------

def test_agent_prompt_separates_reference_from_signals_and_carries_relations() -> None:
    ev = _ev()
    agent = _stub_agent(name="李世民", role="秦王", location="xuanwu", core_traits=("果决",), background="战功赫赫。")
    co_located = [{"id": "a2", "name": "李建成", "gender": "男", "condition": "",
                   "trust": 0.1, "affection": -0.6, "labels": ["兄弟:敌对"]}]
    prompt = _prompt(
        ev, agent, inbox=[_msg("速来", sender_name="长孙无忌")],
        broadcasts=[_bc("喊杀骤起", severity="high")], ambient=[AmbientEvent(content="地上有血迹")],
        co_located=co_located, situation_header="当前时间：辰时；当前地点：玄武门——皇城北门。",
    )
    # target info is framed as reference, not pressure source
    assert "【该角色（仅供识人参考，不是压力来源）】" in prompt
    # The current time and place come from the shared THIRD header; the location isn't inlined into the character descriptor (that would render it twice)
    assert "当前时间：辰时；当前地点：玄武门——皇城北门。" in prompt
    # Gender shares the identity parenthesis: the third-party assessment writes about the pressure he or she is under, and gender decides the pronouns and forms of address.
    assert "李世民（男，秦王，" in prompt and "位于" not in prompt
    assert "性格特点：果决" in prompt
    # Those present are reference, with his own relation: unnumbered, before the signals.
    assert "【同处一地的人（该角色眼中的他们、该角色对其的关系与过往；仅供读懂信号，在场本身不是压力来源）】" in prompt
    assert "- 李建成（男） — 关系：" in prompt and "同处一地的人：" not in prompt
    assert "[兄弟:敌对]" in prompt and "信任度0.10" in prompt and "好感度-0.60" in prompt
    # Signals are numbered, the only pressure source, and come last (nearest the output).
    assert prompt.index("【同处一地的人") < prompt.index("【此刻感知到的信号（外部压力的唯一来源）】")
    # The severity code enum is translated into a natural word; "severity=high" must not leak
    assert "#1 全局广播（重大）：喊杀骤起" in prompt
    assert "severity=" not in prompt
    assert "#2 收到来自长孙无忌的消息：速来" in prompt
    assert "#3 环境观察：地上有血迹" in prompt


def test_agent_prompt_shows_target_reference_and_empty_perceived() -> None:
    ev = _ev()
    agent = _stub_agent(name="路人", role="无", location="街市")
    prompt = _prompt(ev, agent, situation_header="当前时间：辰时；当前地点：街市。")
    # target reference block is always present (identity is reference context)
    assert "【该角色（仅供识人参考，不是压力来源）】" in prompt
    # with no perceivable signals the signal section is "（无）"
    assert "【此刻感知到的信号（外部压力的唯一来源）】" in prompt
    assert "（无）" in prompt
    assert "同处一地" not in prompt
    # Nothing remembered, nothing under way: no empty sections.
    assert "近来的经历" not in prompt and "正在做的事" not in prompt


def test_agent_prompt_carries_his_own_memories_and_what_he_is_doing_as_reference() -> None:
    """Whether a signal answers something he started, is already being dealt with, or repeats
    one he let pass turns on what he has done; that is his own memory, so the boundary holds."""
    ev = _ev()
    agent = _stub_agent(name="李世民", role="秦王", location="xuanwu")
    prompt = _prompt(
        ev, agent, inbox=[_msg("东西收到了，子时老地方见", sender_name="尉迟恭")],
        recent_past=["（昨日）我托人给尉迟恭送去一封密信"], doing="赶往玄武门",
    )
    assert "正在做的事：「赶往玄武门」" in prompt
    assert "【该角色近来的经历（该角色自己的记忆，「我」即该角色；按发生先后排列，越靠后越近；仅供读懂信号，不是压力来源）】" in prompt
    assert "- （昨日）我托人给尉迟恭送去一封密信" in prompt
    # Memories are reference: unnumbered, before the signals, never citable as a source.
    assert prompt.index("近来的经历") < prompt.index("【此刻感知到的信号")
    assert "#1 收到来自尉迟恭的消息" in prompt and "#2" not in prompt


def test_agent_prompt_renders_remote_sender_relation_inline() -> None:
    """A message sender's relation is shown inline so the pressure can be judged by who sent it."""
    ev = _ev()
    agent = _stub_agent(name="李建成", role="太子", location="donggong")
    prompt = _prompt(
        ev, agent, inbox=[_msg("速来东宫", sender_id="a_shimin", sender_name="李世民")],
        sender_relations={"a_shimin": "[兄弟:敌对]，信任度0.10、好感度-0.60"},
        situation_header="当前时间：辰时；当前地点：东宫。",
    )
    assert "收到来自李世民的消息：速来东宫" in prompt
    assert "该角色对李世民的关系：[兄弟:敌对]，信任度0.10、好感度-0.60" in prompt
    # The legend is in the system prompt (test_relation_legend_in_system), not beside the relations.
    assert "涉及的关系字段含义" not in prompt
    assert "同处一地" not in prompt


def test_each_signal_carries_the_source_it_names() -> None:
    """A goal's source names a person only where the signal itself does. An anonymous letter
    stays "world": binding it to its sender would reveal what the letter withholds."""
    ev = _ev()
    signals = ev._signals(
        inbox=[_msg("速来", sender_id="a_shimin", sender_name="李世民"),
               _msg("小心", sender_id="a_shimin", sender_name="匿名", anonymous=True)],
        broadcasts=[_bc("宫门失火")],
        ambient=[AmbientEvent(content="乙拔刀", doer_agent_ids=("a2",)),
                 AmbientEvent(content="远处有人拔刀", doer_agent_ids=("a9",)),
                 AmbientEvent(content="地上有血迹")],
        acted_upon=["乙一把将我推开"],
        sender_relations={}, visible_ids={"a2"},
    )
    assert [s.person_id for s in signals] == [None, "a_shimin", None, "a2", None, None, None]


def test_someone_present_shows_his_past_with_them_as_reference() -> None:
    """What he knows of a person is what lets a signal be read against them: the relation's
    history rides on their reference line, which no goal can cite."""
    ev = _ev()
    agent = _stub_agent(name="甲", role="无", location="hall")
    prompt = _prompt(
        ev, agent, ambient=[AmbientEvent(content="乙拔刀", doer_agent_ids=("a2",))],
        co_located=[{"id": "a2", "name": "乙", "gender": "男", "condition": "",
                     "trust": None, "affection": None, "labels": ["旧识"],
                     "history": "乙曾当众折辱该角色"}],
        visible_ids={"a2"},
    )
    assert "- 乙（男） — 关系：[旧识]；与该角色的过往：乙曾当众折辱该角色" in prompt
    assert "#1 环境观察：乙拔刀" in prompt and "#2" not in prompt


def test_a_pressure_several_signals_add_up_to_is_one_goal_sourced_from_everyone_they_name() -> None:
    """A joint pressure cites every signal behind it and keeps every person they name, in
    signal order and once each; signals naming no one leave it with no source."""
    ev = _ev()
    people = ["a2", None, "a3", None, "a2"]

    def sources(cited: str) -> tuple[str, ...]:
        raw = f'{{"goals": [{{"signals": {cited}, "text": "设法脱身", "urgency": "high", "drive_type": "threat", "related_need": "safety"}}]}}'
        [goal] = ev._parse_agent_goals(raw, signal_people=people)
        return goal.source_ids

    assert sources("[1, 2]") == ("a2",)
    assert sources("[3, 1, 5]") == ("a3", "a2")
    assert sources("[2, 4]") == ()
    # A bare index, not wrapped in a list, still anchors the goal.
    assert sources("3") == ("a3",)


def test_the_system_prompt_keeps_those_present_out_of_the_sources() -> None:
    from engine.world_pressure import _SYSTEM_PROMPT

    assert "（收到的消息 / 可感知的广播 / 环境观察 / " in _SYSTEM_PROMPT
    assert "同处一地的人作为参考信息，自己本身不是外界压力的源头" in _SYSTEM_PROMPT
    assert "未给出的信息，一律视为该角色不知道，不得揣测" in _SYSTEM_PROMPT


class _Relations:
    """Relation system stub returning a fixed objective relation per target id (None = unknown)."""
    def __init__(self, rels: dict[str, SimpleNamespace]) -> None:
        self._rels = rels

    async def load_existing(self, target_agent_id):  # noqa: ANN001, D102
        return self._rels.get(target_agent_id)


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
    agent.relation_system = _Relations({"a_shimin": _rel(trust=0.1, affection=-0.6, labels=["兄弟:敌对"])})
    inbox = [
        _msg("速来", sender_id="a_shimin", sender_name="李世民"),    # remote agent → resolved
        _msg("传旨", sender_id="narrator", sender_name="旁白", sender_is_agent=False),
        _msg("他在城西", sender_id="director", sender_name="匿名", sender_is_agent=False, anonymous=True),
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
    agent.relation_system = _Relations({})  # no relation known for the sender
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
    per_goal = _GOAL_TEXT_MAX_CHARS * 1.5 + 4 * 3 + 5 * 5   # text + signal/urgency/drive/need + 5 fields of structure
    estimate = reason + 3 * per_goal + 10                   # goals capped at 3 + outer wrapper
    assert _MAX_TOKENS >= 3 * estimate, f"预估 {estimate:.0f} tok,需 ≥{3 * estimate:.0f}"


def test_parse_agent_goals_basic_and_source() -> None:
    ev = _ev()
    people = [None, "a2"]  # signal #1 a broadcast, #2 a2's letter
    raw = ('{"goals": [{"signals": [2], "text": "迎战", "urgency": "high", "drive_type": "threat", '
           '"related_need": "safety"}]}')
    goals = ev._parse_agent_goals(raw, signal_people=people)
    assert len(goals) == 1
    assert goals[0].text == "迎战" and goals[0].urgency == Urgency.HIGH
    assert goals[0].source_ids == ("a2",)  # signals=[2] → people[1]


def test_parse_agent_goals_fallbacks() -> None:
    from agent.motivation import ExternalDriveType
    ev = _ev()
    raw = '{"goals": [{"signals": [1], "text": "做事", "urgency": "very-high", "drive_type": "unknown_xyz", "related_need": null}]}'
    goals = ev._parse_agent_goals(raw, signal_people=[None])
    assert goals[0].urgency == Urgency.NORMAL  # unknown urgency → NORMAL
    assert goals[0].drive_type == ExternalDriveType.EVENT  # unknown drive → EVENT


def test_parse_agent_goals_self_actualization() -> None:
    from agent.need import NeedType
    ev = _ev()
    raw = '{"goals": [{"signals": [1], "text": "成就大业", "urgency": "high", "drive_type": "event", "related_need": "self_actualization"}]}'
    goals = ev._parse_agent_goals(raw, signal_people=[None])
    assert goals[0].related_need == NeedType.SELF_ACTUALIZATION


def test_a_goal_citing_no_signal_is_dropped(caplog) -> None:
    """The signal anchor is what keeps a memory or a co-present person alone from becoming
    pressure. A goal without one (missing, out of range, a bare string) has no anchor and must
    not fall back to "world"."""
    ev = _ev()
    raw = ('{"goals": [{"signals": [1], "text": "守住玄武门"}, {"text": "去报旧仇"}, '
           '{"signals": [5], "text": "去见秦王"}, "设法离开东宫"]}')
    with caplog.at_level("WARNING", logger="engine.world_pressure"):
        goals = ev._parse_agent_goals(raw, signal_people=[None, "a2"], agent_id="a1")
    assert [g.text for g in goals] == ["守住玄武门"]
    dropped = [r for r in caplog.records if r.message == "world_pressure_goal_dropped"]
    assert [r.reason for r in dropped] == ["no_signal", "no_signal", "not_an_object"]


def test_one_unparseable_item_does_not_take_the_whole_agent_s_pressure_with_it(caplog) -> None:
    """Per-item skipping must catch everything: the bad item is dropped, the good ones kept.

    This loop feeds _evaluate_one's return, which is outside its try. An escaping exception is
    recorded by gather as the whole agent's evaluation failing, so losing one item becomes losing a
    round, and pressure feeds the dominant_need override and interrupt decisions. So the catch
    can't enumerate exception types: missing one type has exactly that cost.
    """
    ev = _ev()
    raw = ('{"goals": [{"signals": [1], "text": "守住玄武门"}, 123, null, '
           '{"signals": [1], "text": "去见秦王"}, {"signals": [1], "no_text": 1}]}')
    with caplog.at_level("WARNING", logger="engine.world_pressure"):
        goals = ev._parse_agent_goals(raw, signal_people=[None], agent_id="a1")
    assert [g.text for g in goals] == ["守住玄武门", "去见秦王"]
    # A drop must be logged and say whose item it was; skipping silently means nobody notices when the model systematically changes its format.
    dropped = [r for r in caplog.records if r.message == "world_pressure_goal_dropped"]
    assert [r.reason for r in dropped] == ["not_an_object", "not_an_object", "unparseable"]
    assert all(r.agent_id == "a1" for r in dropped)


def test_parse_agent_goals_invalid_or_non_list() -> None:
    ev = _ev()
    assert ev._parse_agent_goals("not json", signal_people=[]) == []
    assert ev._parse_agent_goals('{"pressures": []}', signal_people=[]) == []  # no "goals"


def test_parse_agent_goals_no_truncation() -> None:
    ev = _ev()
    goals_json = ", ".join(
        f'{{"signals": [1], "text": "g{i}", "urgency": "normal", "drive_type": "event", "related_need": null}}'
        for i in range(5)
    )
    goals = ev._parse_agent_goals(f'{{"goals": [{goals_json}]}}', signal_people=[None])
    assert len(goals) == 5


# ---------------------------------------------------------------------------
# _SYSTEM_PROMPT
# ---------------------------------------------------------------------------

def test_system_prompt_single_agent_and_theme_neutral() -> None:
    assert "给定单个角色（下文称\"该角色\"）" in _SYSTEM_PROMPT  # per-agent, third-party assessment
    assert "第三者" in _SYSTEM_PROMPT
    assert '"goals"' in _SYSTEM_PROMPT  # per-agent output schema
    assert "pressures" not in _SYSTEM_PROMPT  # no holistic multi-agent array
    # Think before acting (CLAUDE.md §5, functional judge): reason is the schema's first key, before goals.
    assert '"reason"' in _SYSTEM_PROMPT
    assert _SYSTEM_PROMPT.index('"reason"') < _SYSTEM_PROMPT.index('"goals"')
    # The signals are the goal's generation premise (§5b): they come before the text they shape.
    assert _SYSTEM_PROMPT.index('"signals"') < _SYSTEM_PROMPT.index('"text"')
    assert '"source"' not in _SYSTEM_PROMPT
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


async def _evaluate(ev, *, agents, agent_inboxes, broadcasts, world_time_label, agent_spatials, acted_upon=None):
    """The evaluator fed what reached each agent, built as the runtime builds it."""
    signals = {
        aid: perceived_signals(
            location_id=agent_spatials[aid].location_id,
            inbox=agent_inboxes.get(aid, []),
            broadcasts=broadcasts,
            ambient=agent_spatials[aid].ambient_events,
            acted_upon=(acted_upon or {}).get(aid, ()),
        )
        for aid in agents
    }
    return await ev.evaluate(
        agents=agents, signals=signals, world_time_label=world_time_label, step=1,
        agent_spatials=agent_spatials,
    )


@pytest.mark.asyncio
async def test_evaluate_calls_once_per_signal_bearing_agent() -> None:
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall")}
    # both perceive a global broadcast → 2 calls
    await _evaluate(ev, agents=agents, agent_inboxes={"a1": [], "a2": []},
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
    result = await _evaluate(ev, agents=agents, agent_inboxes={"a1": [], "a2": []},
                               broadcasts=[_bc("丙已身亡", severity="high")], world_time_label="辰时",
                               agent_spatials={"a1": _spatial(location="hall"), "a2": _spatial(location="hall")})
    assert llm.calls == 1
    assert "a2" not in result


@pytest.mark.asyncio
async def test_evaluate_skips_a_character_whose_motivation_is_its_actors() -> None:
    """External goals are motivation, which an actor supplies for their character."""
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall"),
              "a2": _stub_agent(name="乙", role="无", location="hall", reads_world_pressure=False)}
    result = await _evaluate(ev, agents=agents, agent_inboxes={"a1": [], "a2": []},
                               broadcasts=[_bc("宫门失火", severity="high")], world_time_label="辰时",
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
    result = await _evaluate(ev, agents=agents, agent_inboxes={"a1": [], "a2": []}, broadcasts=[],
                               world_time_label="辰时", agent_spatials=spatials)
    assert result == {} and llm.calls == 0


@pytest.mark.asyncio
async def test_evaluate_ambient_only_triggers_that_agent() -> None:
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall")}
    await _evaluate(ev, agents=agents, agent_inboxes={"a1": []}, broadcasts=[], world_time_label="辰时",
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
    await _evaluate(ev, agents=agents, agent_inboxes={"a1": [], "a2": []},
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
    await _evaluate(ev, 
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
    await _evaluate(ev, agents=agents, agent_inboxes={"a1": [], "a2": []},
                      broadcasts=[_bc("天下大乱", severity="high")], world_time_label="辰时",
                      agent_spatials={"a1": _spatial(location="hall"), "a2": _spatial(location="hall")})
    assert sorted(seen) == [("a1", "pressure"), ("a2", "pressure")]


@pytest.mark.asyncio
async def test_an_act_landing_on_an_agent_reaches_it_as_pressure() -> None:
    """Being acted upon is a signal like any other: it brings the agent before the judgment, and
    the prompt says what landed, as his own memory has it."""
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall")}
    await _evaluate(ev, agents=agents, agent_inboxes={"a1": []}, broadcasts=[], world_time_label="辰时",
                    agent_spatials={"a1": _spatial(location="hall")},
                    acted_upon={"a1": ("乙一把将我推开",)})
    assert llm.calls == 1
    assert "落到该角色身上的事（该角色自己的记忆，「我」即该角色）：乙一把将我推开" in llm.prompts[0]


@pytest.mark.asyncio
async def test_co_present_people_appear_only_as_he_perceives_them() -> None:
    """The result reaches his goals, so the prompt may hold only what he knows of those present:
    what perception shows and his own relation, never their profile (role, traits, background)."""
    llm = _RecordingLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {
        "a1": _stub_agent(name="甲", role="无", location="hall"),
        "a2": _stub_agent(name="乙", role="暗探", location="hall", core_traits=("阴鸷",),
                          core_values=("效忠敌营",), background="暗中投靠了敌营。"),
    }
    spatials = {"a1": _spatial(location="hall", visible={"a2": "乙"}),
                "a2": _spatial(location="hall", visible={"a1": "甲"})}
    await _evaluate(ev, agents=agents, agent_inboxes={"a1": [_msg("速来", sender_name="信使")]},
                    broadcasts=[], world_time_label="辰时", agent_spatials=spatials)
    [prompt] = llm.prompts
    assert "- 乙（男） — 关系：关系未明确" in prompt
    assert "#1 收到来自信使的消息：速来" in prompt and "#2" not in prompt
    # His past with each person present is recalled by identity, not by the signals' cues.
    assert agents["a1"].present_seen == [["a2"]]
    assert agents["a1"].cues_seen == [["信使：速来"]]
    for private in ("暗探", "阴鸷", "效忠敌营", "暗中投靠了敌营"):
        assert private not in prompt


@pytest.mark.asyncio
async def test_his_memories_are_recalled_by_the_signals_and_his_action_reaches_the_prompt() -> None:
    from core.context import get_call_annotations

    declared: list[list[str]] = []

    class _FactsLLM(_RecordingLLM):
        async def complete(self, scene, messages, temperature=0.3, max_tokens=600, **kwargs):
            declared.append(list(get_call_annotations().get("given_facts") or []))
            return await super().complete(scene, messages, temperature, max_tokens, **kwargs)

    llm = _FactsLLM()
    ev = WorldPressureEvaluator(llm_router=llm)
    agent = _stub_agent(name="甲", role="无", location="hall", recent_past=("（昨日）我给乙送去一封信",))
    await ev.evaluate(
        agents={"a1": agent},
        signals={"a1": perceived_signals(
            location_id="hall", inbox=[_msg("信收到了", sender_name="乙")], broadcasts=[],
            ambient=[], acted_upon=("乙一把将我推开",),
        )},
        world_time_label="辰时", step=3, agent_spatials={"a1": _spatial(location="hall")},
        doings={"a1": "赶往城门"},
    )
    assert agent.cues_seen == [["乙：信收到了", "乙一把将我推开"]]
    [prompt] = llm.prompts
    assert "- （昨日）我给乙送去一封信" in prompt
    assert "正在做的事：「赶往城门」" in prompt
    # The call declares what it held, from the same sources, so reviews read it instead of
    # rebuilding it (a second recall could differ).
    assert declared == [[
        "此刻感知到的信号：收到来自乙的消息：信收到了（该角色对乙的关系：关系未明确）",
        "此刻感知到的信号：落到该角色身上的事（该角色自己的记忆，「我」即该角色）：乙一把将我推开",
        "近来的经历：（昨日）我给乙送去一封信",
        "正在做的事：赶往城门",
    ]]


@pytest.mark.asyncio
async def test_a_goal_lands_with_the_source_of_the_signal_it_cites() -> None:
    llm = _RecordingLLM('{"reason": "乙拔刀相向", "goals": [{"signals": [1], "text": "躲开乙的刀", '
                        '"urgency": "high", "drive_type": "threat", "related_need": "safety"}]}')
    ev = WorldPressureEvaluator(llm_router=llm)
    agents = {"a1": _stub_agent(name="甲", role="无", location="hall")}
    result = await _evaluate(
        ev, agents=agents, agent_inboxes={}, broadcasts=[], world_time_label="辰时",
        agent_spatials={"a1": _spatial(location="hall", visible={"a2": "乙"},
                                       ambient=["乙拔刀"], ambient_doers=("a2",))},
    )
    assert [(g.text, g.source_ids) for g in result["a1"]] == [("躲开乙的刀", ("a2",))]
