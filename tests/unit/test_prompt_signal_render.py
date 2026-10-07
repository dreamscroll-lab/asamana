"""Unit tests for the shared signal / relation renderers (DRY across prompts)."""

from __future__ import annotations

from types import SimpleNamespace

from agent.relation import PerceivedRelation, render_relation_lines
from core.interfaces.perception import (
    AmbientEvent,
    Broadcast,
    BroadcastType,
    LocationView,
    PerceivedIdentity,
    PerceivedPresence,
    SpatialPerception,
)
from core.prompts import SIGNAL_CAPS, render_entity, render_memory, render_perceived_signals


def _mem(*, content="某事发生了", created_step=0, kind="event"):
    return SimpleNamespace(stored_content=content, created_step=created_step, kind=kind)


def test_render_memory_event_prepends_relative_recency() -> None:
    """Event memories get a relative-recency prefix so the LLM has a sense of time: durations within
    6 hours, days after."""
    sps = 3600  # 1 step = 1 hour; day boundaries align with step 0
    def pre(created, now):
        return render_memory(_mem(created_step=created), now_step=now, seconds_per_step=sps)
    assert pre(10, 10).startswith("（刚刚）")
    assert pre(7, 10).startswith("（几小时前）")    # 3h
    assert pre(0, 10).startswith("（今日早些时候）")  # 10h, same day
    assert pre(0, 30).startswith("（昨日）")      # 30h, one day boundary crossed
    assert pre(0, 60).startswith("（前日）")      # 60h, two crossed
    assert pre(0, 120).startswith("（数日前）")   # 5d
    assert pre(0, 240).startswith("（多日前）")   # 10d
    assert render_memory(
        _mem(content="我看见血迹", created_step=10), now_step=10, seconds_per_step=sps,
    ).endswith("我看见血迹")


def test_render_memory_day_labels_count_midnights_not_hours() -> None:
    """Today/yesterday labels count midnights crossed, not hours elapsed; mixing them names the
    wrong day.

    1 a.m. and 10 a.m. on the same day are 30h and 21h back; by duration the latter reads as today,
    shifting any "tomorrow" in its text by a day. By day difference both are "昨日" (yesterday).
    """
    from engine.clock import WorldTime, WorldTimeConfig

    cfg = WorldTimeConfig(  # 3 hours per step, opening at 4 a.m. → midnight falls between steps
        era_name="武德", start_year=9, start_month=6, start_day=1,
        start_hour=4, seconds_per_step=10800,
    )
    def label(created, now):
        return render_memory(
            _mem(created_step=created), now_step=now, seconds_per_step=cfg.seconds_per_step,
            world_start_second_of_day=WorldTime.from_step(0, cfg).elapsed_seconds,
        )
    assert label(15, 25).startswith("（昨日）")    # day 3 01:00 → day 4 07:00, 30h
    assert label(18, 25).startswith("（昨日）")    # day 3 10:00 → day 4 07:00, 21h
    assert label(12, 25).startswith("（前日）")    # day 2 16:00
    assert label(23, 25).startswith("（今日早些时候）")  # day 4 01:00, same day
    assert label(24, 25).startswith("（几小时前）")  # 3h; below a day boundary, only duration


def test_render_memory_recency_never_gets_younger_as_time_passes() -> None:
    """As now advances, a memory only looks older, never newer; monotonic on both sides of the
    6-hour crossover."""
    order = ["刚刚", "几小时前", "今日早些时候", "昨日", "前日", "数日前", "多日前", "许久之前"]
    seen = [
        order.index(render_memory(
            _mem(created_step=0), now_step=n, seconds_per_step=3600,
            world_start_second_of_day=23 * 3600,  # opening at 11 p.m. → midnight is the next step
        ).split("）")[0][1:])
        for n in range(0, 24 * 40, 3)
    ]
    assert seen == sorted(seen)


def test_render_memory_insight_and_summary_have_no_recency_prefix() -> None:
    """insight (cross-time belief) / summary (time span) get no time prefix; selective by kind."""
    assert render_memory(_mem(content="他在躲我", kind="insight"), now_step=99, seconds_per_step=3600) == "他在躲我"
    assert render_memory(_mem(content="那段日子人心惶惶", kind="summary"), now_step=99, seconds_per_step=3600) == "那段日子人心惶惶"


def test_render_memory_never_leaks_step_number() -> None:
    """Membrane guard: step is a code-layer coordinate and never appears in rendered text."""
    out = render_memory(_mem(content="正文", created_step=3), now_step=47, seconds_per_step=3600)
    assert "47" not in out and "step" not in out.lower()


def _entity(*, name="天子剑", state="intact", description="", is_takeable=False):
    return SimpleNamespace(name=name, state=state, description=description, is_takeable=is_takeable)


def test_render_entity_bare_name_when_all_default() -> None:
    """An ordinary item that's intact, not takeable and undescribed → name only, no
    default-attribute noise."""
    assert render_entity(_entity(name="石桌")) == "石桌"


def test_render_entity_non_default_attrs_and_description() -> None:
    """Non-default attributes go in brackets (state≠intact / takeable); the description follows a
    colon."""
    out = render_entity(_entity(name="木门", state="broken", description="被劈开的门", is_takeable=False))
    assert out == "木门（状态：broken）：被劈开的门"
    out2 = render_entity(_entity(name="匕首", state="intact", description="寒光凛冽", is_takeable=True))
    assert out2 == "匕首（可取）：寒光凛冽"


def test_render_entity_owner_translated_name_not_id() -> None:
    """owner_name arrives already translated by the caller (membrane), rendered as "由X持有"; omitted
    when unowned."""
    out = render_entity(_entity(name="令牌", is_takeable=True), owner_name="李世民")
    assert "由李世民持有" in out and "可取" not in out
    assert "持有" not in render_entity(_entity(name="令牌"))


def test_render_entity_says_what_the_reader_can_do_with_what_he_holds() -> None:
    """In the reader's hand: marked "可交出"; unowned on the ground: "可取"; in someone else's hand: only
    the holder, since it has to be taken from them. Nothing untakeable is marked."""
    token = _entity(name="令牌", is_takeable=True)
    mine = render_entity(token, owner_name="我", held_by_viewer=True)
    assert "可交出" in mine and "可取" not in mine
    theirs = render_entity(token, owner_name="李世民")
    assert theirs == "令牌（由李世民持有）"
    assert render_entity(token) == "令牌（可取）"
    assert "可交出" not in render_entity(_entity(name="石桌"), held_by_viewer=True)


def test_render_entity_renders_self_possession_in_first_person() -> None:
    """First-person channel: the decision side passes "我" for its own items, rendered as "由我持有".

    One renderer serves both the god's-eye judge (passes a name) and first-person decisions (passes
    "我"); the caller picks the holder's referent by viewpoint, and the renderer doesn't and
    shouldn't know who's looking.
    """
    assert "由我持有" in render_entity(_entity(name="虎符", is_takeable=True), owner_name="我")


def test_render_entity_empty_name_falls_back_to_descriptive_not_id() -> None:
    """Empty name → "某物" (something), never entity_id (the narrative layer has no ids)."""
    assert render_entity(_entity(name="")) == "某物"
    assert render_entity(None) == "某物"


def _spatial(*, visible=None, ambient=None) -> SpatialPerception:
    visible = visible or {}
    return SpatialPerception(
        location_id="loc",
        location_view=LocationView(name="loc", description=""),
        world_time_label="辰时", current_step=1,
        visible_agents={k: PerceivedPresence(PerceivedIdentity(name=v)) for k, v in sorted(visible.items())},
        ambient_events=[AmbientEvent(content=c) for c in (ambient or [])],
    )


def _msg(sender_id, sender_name, content):
    return SimpleNamespace(sender_id=sender_id, sender_name=sender_name, content=content)


def _bc(content: str) -> Broadcast:
    return Broadcast(content=content, source="system", broadcast_type=BroadcastType.WORLD_EVENT, deliver_step=1)


def test_render_perceived_signals_labels_each_channel() -> None:
    lines = render_perceived_signals(
        spatial=_spatial(visible={"a": "甲"}, ambient=["远处有动静"]),
        inbox=[_msg("b", "乙", "速来")],
        broadcasts=[_bc("城门已闭")],
    )
    joined = "\n".join(lines)
    assert "同处一地的人：甲" in joined   # stative, so not read as an entrance
    assert "环境观察：远处有动静" in joined
    assert "收到来自乙的消息：速来" in joined
    assert "世界广播：城门已闭" in joined


def test_render_perceived_signals_narrator_message_labels_unknown_source() -> None:
    """Narrator is like a message relayed from an unknown source: rendered as usual as
    "收到来自不知来源的消息"; system concepts (the narrative engine) never enter in-character text."""
    narrator = SimpleNamespace(
        sender_id="narrator", sender_name="不知来源",
        content="李世民心头突然涌起一阵莫名的不安", metadata={"narrative": True},
    )
    lines = render_perceived_signals(spatial=_spatial(), inbox=[narrator], broadcasts=[])
    joined = "\n".join(lines)
    assert "收到来自不知来源的消息：李世民心头突然涌起一阵莫名的不安" in joined
    assert "叙事引擎" not in joined and "narrator" not in joined


def test_render_perceived_signals_empty_when_no_signal() -> None:
    assert render_perceived_signals(spatial=_spatial(), inbox=[], broadcasts=[]) == []


def test_render_perceived_signals_caps_ambient() -> None:
    """Ask ``SIGNAL_CAPS`` for the count; don't copy a number here, that would be a second gate
    table."""
    cap = SIGNAL_CAPS["ambient"]
    lines = render_perceived_signals(
        spatial=_spatial(ambient=[f"事件{i}" for i in range(cap + 4)]), inbox=[], broadcasts=[],
    )
    amb = [l for l in lines if l.startswith("环境观察")]
    assert len(amb) == cap


def test_render_relation_lines_includes_labels_and_numbers() -> None:
    rels = [
        PerceivedRelation(trust=0.9, affection=0.8, target_agent_id="b", target_agent_name="长孙无忌",
                          labels=["挚友"], history_summary="玄武门并肩"),
        PerceivedRelation(trust=0.5, affection=0.0, target_agent_id="c", target_agent_name="陌生人", labels=[]),
    ]
    lines = render_relation_lines(rels)
    # label (relation type) + current affect numbers + history
    assert lines[0] == "长孙无忌：[挚友]，信任度0.90、好感度0.80（玄武门并肩）"
    assert lines[1] == "陌生人：[未明确]，信任度0.50、好感度0.00"
    # no source id
    assert all("id" not in l for l in lines)


def test_render_relation_lines_empty() -> None:
    assert render_relation_lines([]) == []


# ---------------------------------------------------------------------------
# Unified chronological ordering for memory injection
# ---------------------------------------------------------------------------

def test_order_memories_chrono_sorts_by_created_step_ascending() -> None:
    """Uniform ordering before prompt injection: created_step ascending (oldest first, newest last),
    stable."""
    from core.prompts import order_memories_chrono

    a = _mem(content="最旧", created_step=1)
    b = _mem(content="居中", created_step=5)
    c = _mem(content="最新", created_step=9)
    # Input out of order (ranked by relevance/importance); output in ascending time
    assert [m.stored_content for m in order_memories_chrono([c, a, b])] == ["最旧", "居中", "最新"]


def test_order_memories_chrono_key_supports_pairs_and_none() -> None:
    """The key extractor supports (factual, experiential) pairs; None elements count as oldest and
    go first."""
    from core.prompts import order_memories_chrono

    p_new = (_mem(content="新事实", created_step=8), None)
    p_old = (_mem(content="旧事实", created_step=2), _mem(created_step=2))
    p_none = (None, _mem(content="仅解读", created_step=0))
    ordered = order_memories_chrono([p_new, p_old, p_none], key=lambda pair: pair[0] or pair[1])
    # created_step: p_none(0) < p_old(2) < p_new(8)
    assert [(f or e).stored_content for f, e in ordered] == ["仅解读", "旧事实", "新事实"]


def test_render_memory_lines_orders_then_renders() -> None:
    """render_memory_lines = uniform ordering + per-item render_memory; callers only add
    bullets/headers."""
    from core.prompts import render_memory_lines

    newest = _mem(content="后发生", created_step=10)
    oldest = _mem(content="先发生", created_step=1)
    lines = render_memory_lines([newest, oldest], now_step=10, seconds_per_step=3600)
    assert lines[0].endswith("先发生") and lines[-1].endswith("后发生")  # oldest first


def test_render_memory_lines_labels_read_monotonically() -> None:
    """A chronologically ordered list of memories must read progressively more recent; calendar
    words must not out-rank duration words.

    For example: on the same day, 8 hours ago printed "今日内" and 4 hours ago "几小时前";
    the former covers the latter yet sits in the older slot, so the list reads out of order and
    MEMORY_ORDER_HINT is broken on the spot.
    """
    from core.prompts import render_memory_lines

    order = ["刚刚", "几小时前", "今日早些时候", "昨日", "前日", "数日前", "多日前", "许久之前"]

    def ranks(steps, *, now, sps, start_h):
        return [
            order.index(line.split("）")[0][1:])
            for line in render_memory_lines(
                [_mem(content=f"e{s}", created_step=s) for s in steps],
                now_step=now, seconds_per_step=sps, world_start_second_of_day=start_h * 3600)
        ]

    # 4 hours per step, opening at 4 a.m.: two pre-history items + two from 8 hours ago same day +
    # one from 4 hours ago
    assert ranks([-180, -12, 1, 1, 2], now=3, sps=4 * 3600, start_h=4) == [6, 4, 2, 2, 1]
    # Monotonic for any step length and opening hour: the 6-hour crossover must not slip a newer
    # tier into the middle
    for sps in (1800, 3600, 10800, 21600, 43200):
        for start_h in (0, 4, 11, 23):
            got = ranks(range(-40, 61), now=60, sps=sps, start_h=start_h)
            assert got == sorted(got, reverse=True)  # oldest first → tiers go old to new


def test_memory_order_hint_is_single_sourced() -> None:
    """The chronology rule text has one source (a constant), reused by every header instead of
    hard-coded."""
    from core.prompts import MEMORY_ORDER_HINT

    assert "先后" in MEMORY_ORDER_HINT and "越靠后越近" in MEMORY_ORDER_HINT


def test_visible_people_carry_gender_alongside_the_name() -> None:
    """Co-present people carry gender: this signal feeds emotion and goal generation, which must
    write he/she."""
    lines = render_perceived_signals(
        spatial=SpatialPerception(
            location_id="loc", location_view=LocationView(name="玄武门"),
            world_time_label="辰时", current_step=1,
            visible_agents={
                "a1": PerceivedPresence(PerceivedIdentity(name="长孙无垢", gender="女")),
                "a2": PerceivedPresence(PerceivedIdentity(name="尉迟恭")),  # no gender: name only
            },
        ),
        inbox=[], broadcasts=[],
    )
    assert "同处一地的人：长孙无垢（女）, 尉迟恭" in lines[0]


def test_every_signal_channel_is_gated_by_the_same_table() -> None:
    """Only one table answers "how many items can a channel inject".

    The decision prompt's signal blocks keep their own format, but the cap isn't format: two answers
    to it mean nobody can say which is right.
    """
    import ast
    import pathlib

    source = pathlib.Path("agent/decision.py").read_text()
    assert "SIGNAL_CAPS" in source, "决策 prompt 必须引用同一张闸门表"

    # Every `lines.extend(... for x in <channel>)` must be sliced; nothing may be laid in unbounded.
    ungated: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "extend" or not node.args:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.GeneratorExp):
            continue
        for comp in arg.generators:
            rendered = ast.unparse(comp.iter)
            if any(ch in rendered for ch in ("ambient_events", "broadcasts", "inbox")):
                if "SIGNAL_CAPS" not in rendered:
                    ungated.append(rendered)
    assert ungated == [], ungated
