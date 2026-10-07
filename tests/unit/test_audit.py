"""Unit tests for the post-hoc trace audit engine (tuning/audit_*).

Deterministic, no real LLM: a synthetic trace dir feeds reconstruction + the
structural relation check; a schema-aware mock provider drives the uniform judge
and the orchestrator. Covers extraction, weighted normalization, judge
normalise/fallback, and run_audit's five-scope file output.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.interfaces.llm import LLMProvider, LLMResponse, LLMRouter, LLMScene
from tuning.audit_checks import check_relations
from tuning.audit_metrics import SCOPES, weighted_total
from tuning.audit_reconstruct import reconstruct_world
from tuning.judge_audit import judge_scope, init_block, agent_step_block
from tuning.trace import InMemoryTraceSink

WORLD = "w-test"


# ---------------------------------------------------------------------------
# Synthetic trace fixtures
# ---------------------------------------------------------------------------

def _rec(**over) -> dict:
    base = {
        "kind": "llm_call", "world_id": WORLD, "stage": "world_init", "scene": "world_building",
        "prompt_messages": [{"role": "user", "content": "p"}], "response_content": "{}",
        "temperature": 0.2, "max_tokens": 100, "input_tokens": 1, "output_tokens": 1,
        "model": "mock", "latency_ms": 1.0, "timestamp": "2026-07-01T00:00:00",
        "agent_id": None, "step": None, "ok": True, "parse_ok": True, "call_id": "c",
    }
    base.update(over)
    return base


def _write(dir_: Path, name: str, records: list[dict]) -> None:
    p = dir_ / WORLD / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


@pytest.fixture()
def trace_dir(tmp_path: Path) -> str:
    # A non-reciprocal relation category ("政敌" declared one way only) — the structural,
    # theme/language-neutral tripwire. Age/label semantics are the judge's job.
    world_building = {
        "world_name": "测试世界", "core_tension": "争位", "narrative_theme": "夺权",
        "key_figures": [
            {"name": "甲", "role": "长", "age": 30, "gender": "男", "brief": "b"},
            {"name": "乙", "role": "次", "age": 20, "gender": "男", "brief": "b"},
        ],
        "initial_relations": [
            {"from": "甲", "to": "乙", "labels": ["同宗:兄长", "政敌"], "reverse_labels": ["同宗:幼弟"]},
        ],
        "historical_events": [{"event": "结盟", "hours_before_start": 12, "related_figures": ["甲"]}],
        "world_entity_seeds": [
            {"name": "虎符", "entity_type": "item", "initial_state": "intact",
             "location_name": "北营", "description": "调兵之信物"},
        ],
        "npcs": [{"name": "营门卒", "gender": "男", "age": 28,
                  "location_name": "北营", "description": "守营门，认符不认人"}],
    }
    _write(Path(tmp_path), "build.jsonl", [
        _rec(scene="world_building", response_content=json.dumps(world_building, ensure_ascii=False)),
        _rec(scene="cast_design", response_content=json.dumps({"roles": []}, ensure_ascii=False)),
        _rec(scene="persona_generation",
             response_content=json.dumps({"agent_id": "agent-x", "name": "甲"}, ensure_ascii=False)),
        _rec(scene="persona_generation",
             response_content=json.dumps({"agent_id": "agent-y", "name": "乙"}, ensure_ascii=False)),
    ])
    perc = {"emotion": "anger", "intensity": 0.7, "valence": -0.5, "need_activation": {"safety": 0.6}, "reason": "r"}
    moti = {"thought": "t", "goals": ["巩固势力"]}
    deci = {"inner_monologue": "m", "act": "TALK", "action_description": "召集心腹", "selected_index": 1}
    _write(Path(tmp_path), "step_000001.jsonl", [
        _rec(stage="perception", scene="agent_decision_main", agent_id="agent-x", step=1,
             response_content=json.dumps(perc, ensure_ascii=False)),
        _rec(stage="motivation", scene="need_goal_generation", agent_id="agent-x", step=1,
             response_content=json.dumps(moti, ensure_ascii=False)),
        _rec(stage="decision", scene="agent_decision_main", agent_id="agent-x", step=1,
             response_content=json.dumps(deci, ensure_ascii=False)),
        _rec(stage="memory", scene="memory_summarization", agent_id="agent-x", step=1,
             response_content="我召集了心腹，密议已定。", parse_ok=None),
        _rec(stage="perception", scene="agent_decision_main", agent_id="agent-y", step=1,
             response_content=json.dumps(perc, ensure_ascii=False)),
        _rec(stage="pressure", scene="world_pressure", agent_id=None, step=1,
             response_content=json.dumps({"reason": "紧张", "goals": []}, ensure_ascii=False)),
        {"kind": "step", "world_id": WORLD, "step": 1, "world_time": {"label": "测试·清晨", "hour": 6, "minute": 0},
         "wall_ms": 1.0, "timestamp": "2026-07-01T00:00:01"},
    ])
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Schema-aware mock providers
# ---------------------------------------------------------------------------

class _SchemaMock(LLMProvider):
    """Reads the metric ids out of the output-schema in the prompt and scores each 80.

    Emits a nested {unit:{...}} matrix when the schema shows a unit placeholder,
    else a flat {id:score} object — so one provider serves every scope."""

    model = "mock-judge"

    async def complete(self, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ) -> LLMResponse:
        user = messages[-1].content
        ids = re.findall(r'"([a-z_]+)":\s*\d', user)  # matches "id": 0-100 / "id": 1-5
        cell = {i: 80 for i in ids}
        if "<拍的标题" in user or "<步号>" in user:
            scores: dict = {"u1": dict(cell), "u2": dict(cell)}
        else:
            scores = dict(cell)
        return LLMResponse(content=json.dumps({"rationale": "r", "scores": scores}, ensure_ascii=False),
                           input_tokens=1, output_tokens=1, model=self.model)

    async def stream(self, messages, temperature=0.7) -> AsyncIterator[str]:
        yield "x"


class _FailProvider(LLMProvider):
    async def complete(self, messages, temperature=0.7, max_tokens=1000,
        **kwargs,
    ) -> LLMResponse:
        raise RuntimeError("boom")

    async def stream(self, messages, temperature=0.7) -> AsyncIterator[str]:
        yield "x"


def _router(provider: LLMProvider) -> LLMRouter:
    return LLMRouter({s: provider for s in LLMScene}, trace_sink=InMemoryTraceSink())


# ---------------------------------------------------------------------------
# Reconstruction + structural check
# ---------------------------------------------------------------------------

def test_reconstruct_extracts_init_and_trajectories(trace_dir: str) -> None:
    view = reconstruct_world(WORLD, trace_dir=trace_dir)
    assert view.init["world_name"] == "测试世界"
    assert view.agent_names == {"agent-x": "甲", "agent-y": "乙"}
    step1 = view.per_agent["agent-x"].steps[0]
    assert step1.step == 1
    d = step1.summary
    assert d["emotion"]["primary"] == "anger"
    assert d["short_term_goals"] == ["巩固势力"]
    # The decision carries selected_index (the candidate actually chosen); judging whether the
    # choice was reasonable needs it.
    assert d["decision"]["selected_index"] == 1
    assert d["decision"]["action_description"] == "召集心腹"
    assert d["memory"] == ["我召集了心腹，密议已定。"]
    # World pressure stays out of this view. Each pressure call evaluates one agent but carries no
    # agent_id, so treating the first one as "this step's world pressure" would assign one person's
    # pressure to the whole step. Nothing consumes it, so leave it out.
    assert "pressure" not in view.world_sequence[0]


def test_persona_identity_comes_from_trace_not_echo(tmp_path: Path) -> None:
    """An agent_id / name the model echoes back wrong must not become an identity in review."""
    _write(tmp_path, "build.jsonl", [
        _rec(scene="persona_generation", agent_id="agent-real", extra={"agent_name": "张老师"},
             response_content=json.dumps({"agent_id": "agent-typo", "name": "张老师（班主任）"},
                                         ensure_ascii=False)),
    ])
    view = reconstruct_world(WORLD, trace_dir=str(tmp_path))
    assert view.agent_names == {"agent-real": "张老师"}
    assert [(p["agent_id"], p["name"]) for p in view.init["personas"]] == [("agent-real", "张老师")]


def test_cast_roles_reach_review_under_the_figure_their_number_names(tmp_path: Path) -> None:
    """CastDesigner names each role's figure by roster number; review must read it the same way,
    or every narrative role and arc silently drops out of the initialization view."""
    roster = {"key_figures": [{"name": "甲"}, {"name": "乙"}]}
    roles = {"roles": [{"index": 2, "narrative_role": "次子", "arc_summary": "弧"}]}
    _write(tmp_path, "build.jsonl", [
        _rec(scene="world_building", response_content=json.dumps(roster, ensure_ascii=False)),
        _rec(scene="cast_design", response_content=json.dumps(roles, ensure_ascii=False)),
    ])
    view = reconstruct_world(WORLD, trace_dir=str(tmp_path))
    assert [(r["name"], r["narrative_role"]) for r in view.init["cast_roles"]] == [("乙", "次子")]


def test_init_shows_the_world_s_things_and_errand_hands(trace_dir: str) -> None:
    """Items and errand-runners are part of the opening, fixed at build time and unchangeable.
    Without them the judge sees only half the world and can't judge what the opening allows beyond
    talking. Placement is needed too: where things and people are is how the judge tells whether
    they engage with the conflict."""
    from tuning.judge_audit import init_block

    view = reconstruct_world(WORLD, trace_dir=trace_dir)
    block = init_block(view)
    assert "[物] 虎符（item，初始状态 intact）　在 北营：调兵之信物" in block
    assert "[差役] 营门卒（男28岁）　在 北营：守营门，认符不认人" in block


def test_reconstruct_steps_filter(trace_dir: str) -> None:
    view = reconstruct_world(WORLD, trace_dir=trace_dir, steps=[99])
    assert view.per_agent == {}
    assert view.init["world_name"] == "测试世界"


def test_relation_check_fires_on_non_reciprocal(trace_dir: str) -> None:
    view = reconstruct_world(WORLD, trace_dir=trace_dir)
    findings = check_relations(view)
    assert {f["kind"] for f in findings} == {"category_not_reciprocal"}
    assert any("政敌" in f["detail"] for f in findings)


# ---------------------------------------------------------------------------
# Weighted normalization
# ---------------------------------------------------------------------------

def test_criteria_loaded_from_yaml() -> None:
    # default bundled yaml populates criteria on the module-level SCOPES.
    m = {x.id: x for x in SCOPES["single_agent_single_step"].metrics}
    assert set(m) == {"grounding", "instruction_adherence"}
    assert any("凭空" in c for c in m["grounding"].criteria)


def _complete_criteria_yaml(*, persona: str = "红旗", drop: tuple[str, str] | None = None) -> str:
    """Builds a YAML with every (scope, dimension) from _SCOPE_DEFS filled in; ``drop`` deliberately
    leaves out the given cell."""
    from tuning.audit_metrics import _SCOPE_DEFS

    lines: list[str] = []
    for key, _label, _purpose, _unit, dim_ids in _SCOPE_DEFS:
        lines.append(f"{key}:")
        for did in dim_ids:
            if drop == (key, did):
                continue
            text = persona if (key, did) == ("initialization", "persona_worldview") else "红旗"
            lines.append(f"  {did}:\n    criteria:\n      - \"{text}\"")
    return "\n".join(lines) + "\n"


def test_build_scopes_reads_criteria_from_yaml(tmp_path: Path) -> None:
    """criteria come from YAML; weights don't (they're fixed in DIMENSIONS)."""
    from tuning.audit_metrics import build_scopes

    yaml_path = tmp_path / "crit.yaml"
    yaml_path.write_text(_complete_criteria_yaml(persona="自定义红旗"), encoding="utf-8")
    pw = next(m for m in build_scopes(yaml_path)["initialization"].metrics
              if m.id == "persona_worldview")
    assert pw.criteria == ["自定义红旗"] and pw.weight == 20


def test_a_dimension_without_criteria_is_a_hard_error(tmp_path: Path) -> None:
    """criteria are the deduction checklist ("deduct on hit"). Without them the judge scores by
    feel, producing numbers that can't tell anything apart, so a missing tier or item raises.
    scope→dimension lives in Python and criteria in YAML, so changing one and forgetting the other
    lands here.
    """
    from tuning.audit_metrics import build_scopes

    partial = tmp_path / "partial.yaml"
    partial.write_text(_complete_criteria_yaml(drop=("initialization", "causal_coherence")),
                       encoding="utf-8")
    with pytest.raises(ValueError, match=r"initialization\.causal_coherence"):
        build_scopes(partial)

    with pytest.raises(ValueError, match="missing"):
        build_scopes(tmp_path / "nope.yaml")

    broken = tmp_path / "broken.yaml"
    broken.write_text("initialization:\n  - [unbalanced\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_scopes(broken)


def test_every_shipped_dimension_carries_its_failure_modes() -> None:
    """Deduction scoring assumes the judge's checklist isn't empty. All 15 (scope, dimension) pairs
    in the production config need criteria."""
    from tuning.audit_metrics import SCOPES

    assert all(m.criteria for sc in SCOPES.values() for m in sc.metrics)


def test_dimension_weights_sum_to_100() -> None:
    from tuning.audit_metrics import DIMENSIONS
    assert sum(d.weight for d in DIMENSIONS.values()) == 100
    by_id = {m.id: m for m in SCOPES["multi_agent_multi_step"].metrics}
    assert (by_id["narrative_convergence"].category, by_id["narrative_convergence"].weight) == ("basic", 15)
    assert (by_id["dramatic_arc"].category, by_id["dramatic_arc"].weight) == ("quality", 15)
    assert (by_id["emergent_novelty"].category, by_id["emergent_novelty"].weight) == ("quality", 5)


def test_weighted_total_none_unit() -> None:
    sc = SCOPES["initialization"]  # persona_worldview 20, causal_coherence 20 (equal)
    # (20*100 + 20*60) / 40 = 80.0
    assert weighted_total({"persona_worldview": 100, "causal_coherence": 60}, sc.metrics, "none") == 80.0


def test_weighted_total_matrix_respects_weights() -> None:
    sc = SCOPES["single_agent_multi_step"]  # only four of its dimensions here: causal 20 / action_efficacy 20 / growth 5 / arc 15
    scores = {
        "1": {"causal_coherence": 100, "action_efficacy": 100, "character_growth": 0, "dramatic_arc": 0},
        "2": {"causal_coherence": 100, "action_efficacy": 100, "character_growth": 0, "dramatic_arc": 0},
    }
    # per cell num=20*100+20*100+5*0+15*0=4000, den=60 ; two cells → 8000/120 = 66.7 (weighted > unweighted 50)
    assert weighted_total(scores, sc.metrics, "step") == 66.7


def test_clamp_and_missing_cells() -> None:
    sc = SCOPES["initialization"]
    # 150 clamps to 100; missing metric excluded from the weighted mean.
    assert weighted_total({"persona_worldview": 150, "causal_coherence": 60}, sc.metrics, "none") == 80.0
    assert weighted_total({"persona_worldview": 100}, sc.metrics, "none") == 100.0


# ---------------------------------------------------------------------------
# Uniform judge: normalise + fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_none_unit_scope(trace_dir: str) -> None:
    view = reconstruct_world(WORLD, trace_dir=trace_dir)
    res = await judge_scope(_router(_SchemaMock()), SCOPES["initialization"],
                            header="h", context_block=init_block(view),
                            det_findings=check_relations(view))
    assert res["total"] == 80.0
    assert res["scores"]["persona_worldview"] == 80
    assert res["rationale"] == "r"


def _sum(calls, name="甲"):
    from tuning.audit_reconstruct import summarise
    return summarise(calls, name)


def _call(stage, scene="agent_decision_main", out=None, extra=None, adopted=None, rej=""):
    from tuning.audit_reconstruct import StageCall
    return StageCall(stage=stage, scene=scene, prompt=[{"role": "user", "content": ""}],
                     output=out if out is not None else {}, parse_ok=True, ok=True,
                     adopted=adopted, reject_reason=rej, extra=extra or {})


def test_a_conscripted_step_drops_the_intent_and_keeps_what_actually_happened() -> None:
    """On a step where he was recruited, what he decided himself never happened, and the
    short-term goals generated with it are void too. Neither is injected. What's injected is what
    actually happened to him: what he was pulled into, the result, the feedback.

    Showing an intent that never executed makes the judge compare it with the result below (often
    of an action someone else started) and rule "decision and action don't match / fabricated /
    broken chain". The basis is the verdict arbitration stamped, not inference from side signals.
    """
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    calls = [
        _call("perception", out={"emotion": "anger", "intensity": 0.6, "valence": -0.7}),
        _call("motivation", "need_goal_generation", {"thought": "我得找丙"},
              {"short_term_goals_new": ["去找丙商议"]}, adopted=False, rej="conscripted"),
        _call("decision", out={"selected_index": 0, "action_description": "我去找丙商议对策"},
              extra={"action_menu": {"0": "TALK"}, "verdict": "conscripted"}),
        _call("action", "agent_action_narration", {"fact": "乙拉着我谈了半天", "success": False},
              {"talk_role": "addressee", "action_owner": "甲",
               "conscripted_input": "乙来找我谈，话题是「借一步说话」"}),
        _call("feedback", out={"emotion": "frustration", "intensity": 0.7, "valence": -0.6}),
    ]
    s = _sum(calls)
    assert "decision" not in s and "short_term_goals" not in s   # intents and goals that never happened aren't injected
    assert s["intent_dropped"] == "conscripted"
    assert s["joined"] == ["乙来找我谈，话题是「借一步说话」"]
    assert s["results"] == [{"fact": "乙拉着我谈了半天", "success": False}]
    assert s["appraisal"][0]["emotion"] == "frustration"

    block = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=s),
                                  name="甲", sps=3600))
    assert "我去找丙商议对策" not in block and "去找丙商议" not in block
    assert "【本步他没有自己的行动】" in block and "行动｜被邀入：乙来找我谈" in block
    assert "乙拉着我谈了半天" in block


def test_the_conscripted_note_only_points_at_an_action_that_is_in_this_step() -> None:
    """An action someone was merged into usually settles only when it finishes, and until then he
    has no action of his own. Pointing to an "action below" that isn't there makes the judge read
    this step's record as missing a piece."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    def _block(calls):
        return "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=_sum(calls)),
                                     name="甲", sps=3600))

    dec = _call("decision", out={"selected_index": 0, "action_description": "我去找丙"},
                extra={"action_menu": {"0": "TALK"}, "verdict": "conscripted"})
    still_running = _block([dec])
    assert "见下方" not in still_running and "这一步还没走完" in still_running
    landed = _block([dec, _call("action", "agent_action_narration",
                                {"fact": "乙拉着我谈了半天", "success": True},
                                {"talk_role": "addressee", "action_owner": "甲",
                                 "conscripted_input": "乙来找我谈，话题是「借一步说话」"})])
    assert "见下方「行动」" in landed


def test_a_step_that_settles_then_starts_again_keeps_the_two_apart_and_in_order() -> None:
    """Within a step, the engine first settles the action in hand (interrupt teardown + feedback),
    then runs a new cognition cycle. The two parts must stay separate with the carried-over part
    first. Mixed together, the new decision gets paired with the interrupted action's result, and
    both emotions share one "before" state.
    """
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    prior = {"settles_prior_action": True}  # the marker the engine stamps at those three points at step start
    calls = [
        _call("interrupt", "agent_interrupt_decision", {"thought": "此事更急", "interrupt": True},
              {**prior, "interrupt_doing": "在府中清点人手", "interrupt_trigger": "宫中急报"}),
        _call("action", "agent_action_narration", {"outcome": "他悄然离开", "success": True}, prior),
        _call("feedback", out={"emotion": "anticipation", "intensity": 0.7, "valence": -0.2},
              extra=prior),
        _call("perception", out={"emotion": "frustration", "intensity": 0.65, "valence": -0.55}),
        _call("decision", out={"selected_index": 0, "action_description": "回府报信"},
              extra={"action_menu": {"0": "TALK"}, "verdict": "executed"}),
        # Memory writes go through an async queue, so the memory prose of the settled action
        # arrives late, after the new cycle. It still belongs to the carried-over part.
        _call("memory", "memory_summarization", extra=prior),
        _call("action", "agent_action_narration", {"fact": "我把消息带到了", "success": True}),
        _call("feedback", out={"emotion": "trust", "intensity": 0.7, "valence": 0.4}),
    ]
    calls[5].output = "我脱身了。"
    s = _sum(calls)
    carried = s["carried"]
    assert [r.get("outcome") for r in carried["results"]] == ["他悄然离开"]
    assert [a["emotion"] for a in carried["appraisal"]] == ["anticipation"]
    assert carried["memory"] == ["我脱身了。"] and "memory" not in s
    assert [r.get("fact") for r in s["results"]] == ["我把消息带到了"]
    assert [a["emotion"] for a in s["appraisal"]] == ["trust"]

    block = "\n".join(_step_lines(AgentStepView(step=4, world_time="t", summary=s),
                                  name="甲", sps=3600))
    assert block.index("他悄然离开") < block.index("回府报信") < block.index("我把消息带到了")
    assert "中断｜输入：正在做:在府中清点人手｜突发:宫中急报" in block
    # The "before" state belongs only to this cycle's emotion. The carried-over part has no
    # perception to pair with and must not borrow this cycle's frustration.
    assert "反馈｜输出：情绪 anticipation" in block and "情绪 frustration→trust" in block


def test_a_landing_step_says_which_action_settled_and_who_was_in_it() -> None:
    """A multi-step action settles on the step it finishes, and that step has no decision. Unless
    it says which action settled and who was there, the judge reads a result with no owner. The
    input travels with the result (the extra attached at the adjudication point), which also lets a
    two-person action identify the other side of the conversation."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    whose = {"action_description": "我转向乙，问他对局势的看法",
             "action_initiator": "甲", "action_participants": ["甲", "乙"]}
    s = _sum([_call("action", "agent_action_narration", {"dialogue": [{"speaker": 1, "line": "如何"}]},
                    whose),
              _call("action", "agent_action_narration", {"fact": "我与乙谈过了", "success": True},
                    {**whose, "talk_role": "initiator", "action_owner": "甲"})])
    block = "\n".join(_step_lines(AgentStepView(step=6, world_time="t", summary=s),
                                  name="甲", sps=3600))
    assert "行动｜输入：我转向乙，问他对局势的看法｜参与:甲、乙" in block
    assert block.count("行动｜输入") == 1        # two results of the same action share one input line
    assert "对话：1 轮（甲↔乙）" in block          # the participant list names the other side directly instead of "对方"
    assert "记忆(己方)：我与乙谈过了" in block


def test_two_participants_sharing_a_name_leave_the_peer_unnamed() -> None:
    """When two people share a name, "the other side" can't be identified from the list. Leave it
    blank rather than crash the whole review."""
    s = _sum([_call("action", "agent_action_narration", {"fact": "我喝问了他", "success": True},
                    {"action_initiator": "甲", "action_participants": ["甲", "甲"]})])
    assert "peer" not in s
    assert [r.get("fact") for r in s["results"]] == ["我喝问了他"]


def test_the_action_input_is_spelled_out_even_when_it_echoes_the_decision() -> None:
    """The action stage shows its own input, consistent with other stages. What the executor got
    isn't necessarily what the decision wrote, and without it the judge can't tell "not given" from
    "same as above"."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    s = _sum([_call("decision", out={"selected_index": 0, "action_description": "我转向乙，问他对局势的看法"},
                    extra={"action_menu": {"0": "TALK"}, "verdict": "executed"}),
              _call("action", "agent_action_narration", {"fact": "我与乙谈过了", "success": True},
                    {"action_description": "我转向乙，问他对局势的看法（我打算说：如何）",
                     "action_initiator": "甲", "action_participants": ["甲", "乙"]})])
    block = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=s),
                                  name="甲", sps=3600))
    assert "行动｜输入：我转向乙，问他对局势的看法（我打算说：如何）｜参与:甲、乙" in block


def test_mams_keeps_the_settled_action_apart_from_the_round_that_followed() -> None:
    """The director view has to separate them too: the action settled first in a step and the new
    cycle he starts afterwards each carry their own result and emotion. Merged, the new decision
    gets the interrupted action's result (reading as "he decided to go east, yet ended up stopped
    in the west")."""
    from tuning.judge_audit import _mams_lines

    prior = {"settles_prior_action": True}
    s = _sum([
        _call("interrupt", "agent_interrupt_decision", {"thought": "护主要紧", "interrupt": True},
              {**prior, "interrupt_doing": "赶往北营", "interrupt_trigger": "主公独自出行"}),
        _call("feedback", out={"emotion": "frustration", "intensity": 0.6, "valence": -0.5},
              extra={**prior, "feedback_action_desc": "赶往北营",
                     "feedback_action_actual": "我走到半路折返了"}),
        _call("perception", out={"emotion": "fear", "intensity": 0.7, "valence": -0.6}),
        _call("decision", out={"selected_index": 2, "action_description": "跟上主公"},
              extra={"action_menu": {"2": "MOVE"}, "verdict": "executed"}),
        _call("action", "agent_action_narration", {"fact": "我跟上了主公", "success": True}),
        _call("feedback", out={"emotion": "trust", "intensity": 0.6, "valence": 0.3}),
    ])
    block = "\n".join(_mams_lines("甲", s, sps=3600))
    assert "【中断】他当场放下了手上那件事（正在做:赶往北营｜突发:主公独自出行）：护主要紧" in block
    assert block.index("我走到半路折返了") < block.index("跟上主公") < block.index("我跟上了主公")
    # The carried-over part has no perception to pair with, so it can't borrow this cycle's fear
    # for an arrow; only this cycle's emotion is fear→trust.
    assert "情绪（对结果的反应）：frustration" in block and "情绪：fear→trust" in block


def test_a_settled_multi_step_action_stays_apart_from_a_conscription_that_followed() -> None:
    """The same boundary covers this case: at step start his multi-step action lands, and then he's
    pulled into someone else's activity. He ran no cognition this step, but they're still two
    separate things. The engine's markers decide ownership, not whether a new cognition cycle ran."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    prior = {"settles_prior_action": True}
    s = _sum([
        _call("action", "agent_action_narration", {"fact": "我把卷宗核完了", "success": True},
              {**prior, "action_description": "核对卷宗", "action_initiator": "甲",
               "action_participants": ["甲"]}),
        _call("feedback", out={"emotion": "pride", "intensity": 0.6, "valence": 0.5}, extra=prior),
        _call("action", "agent_action_narration", {"fact": "乙拉着我谈了半天", "success": True},
              {"talk_role": "addressee", "action_owner": "甲",
               "conscripted_input": "乙来找我谈，话题是「借一步说话」"}),
        _call("feedback", out={"emotion": "frustration", "intensity": 0.7, "valence": -0.6}),
    ])
    assert [a["emotion"] for a in s["carried"]["appraisal"]] == ["pride"]
    assert [a["emotion"] for a in s["appraisal"]] == ["frustration"]
    block = "\n".join(_step_lines(AgentStepView(step=5, world_time="t", summary=s),
                                  name="甲", sps=3600))
    assert block.index("我把卷宗核完了") < block.index("乙来找我谈")
    assert "行动｜输入：核对卷宗" in block and "行动｜被邀入：乙来找我谈" in block


def test_mams_says_which_emotion_it_is_showing() -> None:
    """On the step he's recruited he has no result yet, so what's shown is his perception emotion.
    Showing a bare emotion makes the judge treat it as a reaction to a result and go looking for a
    result that doesn't exist."""
    from tuning.judge_audit import _mams_lines

    s = _sum([_call("perception", out={"emotion": "fear", "intensity": 0.8, "valence": -0.7}),
              _call("decision", out={"selected_index": 0, "action_description": "我去找丙"},
                    extra={"action_menu": {"0": "TALK"}, "verdict": "conscripted"})])
    block = "\n".join(_mams_lines("甲", s, sps=3600))
    assert "情绪（他此刻所感；本步还没有结果的反应）：fear（强度0.8，效价-0.7）" in block


def test_a_long_term_goal_revision_shows_what_the_engine_kept() -> None:
    """Long-term goal reviews happen only a few times per world and are the most direct evidence
    for growth / arc, so both sams and mams need to see them. What's shown must be what the engine
    finally adopted: two kinds of LLM output are discarded wholesale (an empty array means no
    change of direction, or identical to the previous goals), and showing them would present a
    change that never happened."""
    from tuning.judge_audit import _mams_lines, _step_lines
    from tuning.audit_reconstruct import AgentStepView

    def _blocks(out, extra):
        s = _sum([_call("long_term_goals", "long_term_goal_revision", out, extra)])
        return ("\n".join(_step_lines(AgentStepView(step=9, world_time="t", summary=s),
                                      name="甲", sps=3600)),
                "\n".join(_mams_lines("甲", s, sps=3600)))

    changed = {"thought": "兄弟已无退路", "goals": ["夺取储位", "保全天策府旧属"]}
    for block in _blocks(changed, {"long_term_goals_new": changed["goals"]}):
        assert "夺取储位" in block and "保全天策府旧属" in block
        assert "方向未变" not in block and '{"' not in block
    # The LLM wrote the same two goals back → the engine sees no update and leaves state alone, so
    # review must not say he changed them.
    for block in _blocks(changed, {"long_term_goals_new": []}):
        assert "方向未变" in block and "夺取储位" not in block


def test_a_participant_block_never_claims_someone_elses_deed() -> None:
    """Someone merged into another's action also has a result (his own first-person memory). The
    description is in the initiator's first person, so without naming the initiator it reads as
    "he did it himself"; mams doesn't use it for "what he did" at all."""
    from tuning.judge_audit import _mams_lines, _step_lines
    from tuning.audit_reconstruct import AgentStepView

    s = _sum([_call("action", "agent_action_narration", {"fact": "甲拉着我谈了半天", "success": True},
                    {"action_description": "我转向乙，问他对局势的看法",
                     "action_initiator": "甲", "action_participants": ["甲", "乙"],
                     "talk_role": "addressee", "action_owner": "乙",
                     "conscripted_input": "甲来找我谈，话题是「局势」"})], name="乙")
    block = "\n".join(_step_lines(AgentStepView(step=6, world_time="t", summary=s),
                                  name="乙", sps=3600))
    assert "行动｜输入：这一动由 甲 发起：我转向乙" in block
    assert "\n".join(_mams_lines("乙", s, sps=3600)).count("做什么") == 0


def test_the_words_actually_sent_are_an_input_to_the_action() -> None:
    """The exact words are the content handed to execution (an input, like the description and the
    participants), so they don't get a separate line. Shown apart, the reader has to match two lines
    back to one action, and in mams the result and emotion sit between them."""
    from tuning.judge_audit import _mams_lines, _step_lines
    from tuning.audit_reconstruct import AgentStepView

    s = _sum([_call("decision", out={"selected_index": 1, "action_description": "遣人送信给乙",
                                     "message_content": "速来"},
                    extra={"action_menu": {"1": "SEND_MESSAGE"}, "verdict": "executed"})])
    block = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=s),
                                  name="甲", sps=3600))
    assert "决策｜原话" not in block and "原话" not in block.split("\n")[0]
    assert "  行动｜输入：原话:速来" in block
    assert "      · 做什么：遣人送信给乙｜原话:速来" in "\n".join(_mams_lines("甲", s, sps=3600))


def test_a_multi_step_action_is_marked_only_at_its_opening() -> None:
    """A multi-step action is flagged as multi-step only on the step it starts; the remaining step
    count comes from the execution body, not the decision's self-reported estimated_steps. Later
    steps render normally, whether interrupted or finished, and the reader can see it's the same
    action continuing. Review doesn't chain across steps (that would mean guessing, or machinery
    that exists only for this)."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    opening = _sum([_call("decision", out={"selected_index": 3, "action_description": "闭门批阅文书"},
                          extra={"action_menu": {"3": "WORK"}, "verdict": "executed", "spans_steps": 4})])
    block = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=opening),
                                  name="甲", sps=3600))
    assert "【这一动跨多步】" in block and "约4小时" in block
    # The continuation step: no new decision, so no multi-step marker, only its own result.
    later = _sum([_call("action", "agent_action_narration", {"outcome": "文书批完了", "success": True})])
    b2 = "\n".join(_step_lines(AgentStepView(step=4, world_time="t", summary=later),
                               name="甲", sps=3600))
    assert "跨多步" not in b2 and "文书批完了" in b2


def test_a_rejected_action_says_the_world_refused_it() -> None:
    """Actions arbitration rejected on the spot (partner unresponsive / busy with something else)
    must be marked. Otherwise the judge reads "he decided X and it failed" as his own incompetence,
    when the world never took the action up."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    s = _sum([_call("decision", out={"selected_index": 0, "action_description": "找乙商议"},
                    extra={"action_menu": {"0": "TALK"}, "verdict": "rejected"}),
              _call("action", "agent_action_narration",
                    {"fact": "我本想找乙商议，却因乙正忙于他事未能如愿", "success": False})])
    block = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=s), name="甲", sps=None))
    assert "【这一动被判不可行】" in block and "找乙商议" in block


def test_an_agent_acted_upon_says_so_and_does_not_claim_the_deed() -> None:
    """Acted upon: he is the target of someone else's action. The result is his reaction to it, not
    an action he decided on or performed. Without saying so, the judge compares it to his own
    decision and rules a broken chain."""
    from tuning.judge_audit import _step_lines
    from tuning.audit_reconstruct import AgentStepView

    s = _sum([_call("decision", out={"selected_index": 5, "action_description": "隐蔽身形暗中观察"},
                    extra={"action_menu": {"5": "COVERT"}, "verdict": "executed"}),
              _call("action", "agent_action_narration", {"fact": "乙猛然扑上试图将我制服"},
                    {"action_owner": "甲", "acted_upon": "乙对我做了：趁其不备猛然扑上"})],
             "甲")
    block = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=s), name="甲", sps=None))
    assert "行动｜被施加：乙对我做了：趁其不备猛然扑上" in block
    assert "决策｜输出：[COVERT] 隐蔽身形暗中观察" in block   # his intent is still shown separately


def test_motivation_new_goals_read_from_stored_post_dedup_field() -> None:
    """New short-term goals = what the engine actually queued after dedup
    (extra.short_term_goals_new). Review reads that directly and doesn't redo the dedup. Near-duplicate
    goals the engine dropped aren't in it, so they aren't mistaken for new goals (and misjudged as
    repetition / idling). Without that field, fall back to the raw goals."""
    from tuning.audit_reconstruct import StageCall

    got = _sum([StageCall(
        stage="motivation", scene="need_goal_generation",
        prompt=[{"role": "user", "content": ""}],
        output={"goals": ["挺枪逼近御座，以武力威慑逼李渊交出兵权", "遣心腹联络东宫旧部以备援手"]},
        parse_ok=True, ok=True,
        extra={"short_term_goals_new": ["遣心腹联络东宫旧部以备援手"]})])
    assert got["short_term_goals"] == ["遣心腹联络东宫旧部以备援手"]  # read only what was actually queued

    legacy = _sum([StageCall(
        stage="motivation", scene="need_goal_generation",
        prompt=[{"role": "user", "content": ""}],
        output={"goals": ["甲目标", "乙目标"]}, parse_ok=True, ok=True)])
    assert legacy["short_term_goals"] == ["甲目标", "乙目标"]


def test_resolve_targets_reads_candidate_maps_from_extra() -> None:
    """Decision target index → name, translated via the candidate maps the decision call attaches
    to extra (people / locations / items), without parsing the prompt. Index base matches what
    decision.py prints: people / locations / items are 1-based. JSON persistence turns int keys into
    str, so the maps use string keys."""
    from tuning.audit_reconstruct import _resolve_targets

    extra = {
        "person_candidates": {"1": "尉迟恭", "2": "魏征"},
        "destination_candidates": {"1": "崇仁坊"},
        "item_candidates": {"1": "太子印信"},
    }
    out = {"person_indices": [2], "destination_index": 1, "physical_entity_index": 1}
    assert _resolve_targets(extra, out) == "魏征、→崇仁坊、[物]太子印信"


def test_resolve_targets_physical_on_person_uses_person_roster() -> None:
    """PHYSICAL on a person: physical_person_index points into "我能触及的人" (person_candidates)
    and is a person, not a thing, so it renders as the person's name without "[物]". Each channel
    binds its own list and index 1 means different things in each. If review resolved it as an
    item, "按住李元吉" would show as "[物]皇帝密诏" (wrong entity and wrong kind)."""
    from tuning.audit_reconstruct import _resolve_targets

    extra = {
        "person_candidates": {"1": "李元吉", "2": "长孙无忌"},
        "item_candidates": {"1": "皇帝密诏"},  # an item at the same index 1, showing the channels must be kept apart
    }
    out = {"selected_index": 4, "physical_person_index": 1}
    assert _resolve_targets(extra, out) == "李元吉"  # not "[物]皇帝密诏"


def test_resolve_targets_covers_errand_and_carry_slots() -> None:
    """ERRAND / MOVE-carry / hand-over each have their own candidate list. Missing a channel means
    that kind of action has no target in review, and action_efficacy judges whether it really
    changed the world by its target."""
    from tuning.audit_reconstruct import _resolve_targets

    extra = {
        "person_candidates": {"1": "李建成", "2": "魏征"},
        "destination_candidates": {"1": "崇仁坊", "2": "十六王宅"},
        "item_candidates": {"1": "密信"},
        "npc_candidates": {"1": "传旨内侍"},
    }
    errand = {"errand_npc_index": 1, "errand_destination_index": 2,
              "errand_item_index": 1, "errand_recipient_index": 2}
    assert _resolve_targets(extra, errand) == "遣:传旨内侍、→十六王宅、[物]密信、说与:魏征"
    assert _resolve_targets(extra, {"destination_index": 1, "move_carry_indices": [1]}) \
        == "→崇仁坊、带走:李建成"
    assert _resolve_targets(extra, {"physical_entity_index": 1, "physical_recipient_index": 1}) \
        == "[物]密信、交予:李建成"


def test_feedback_goals_resolve_text_from_extra() -> None:
    """Feedback goal progress: translate index back to text via the index → goal text map the
    feedback call attaches at extra.goal_texts; otherwise render degrades to "目标#N". The map uses
    string keys (JSON persistence) and output indices are ints, so look up with str(index)."""
    from tuning.audit_reconstruct import StageCall

    calls = [StageCall(
        stage="feedback", scene="need_goal_generation",
        prompt=[{"role": "user", "content": ""}],
        output={"goals": [{"index": 1, "status": "completed"},
                          {"index": 2, "status": "active"},
                          {"index": 3, "status": "active"}]},
        parse_ok=True, ok=True,
        extra={"goal_texts": {
            "1": "传令禁军封锁太极宫各门（已历约1小时仍未了结）",  # the duration hint is kept too
            "2": "安排亲信暗中监控偏殿",
            "3": "梳理指控细节，预设证据矛盾点"}})]
    goals = _sum(calls)["goal_progress"][0]
    texts = {g["index"]: g.get("text") for g in goals}
    assert texts[1] == "传令禁军封锁太极宫各门（已历约1小时仍未了结）"
    assert texts[2] == "安排亲信暗中监控偏殿"
    assert texts[3] == "梳理指控细节，预设证据矛盾点"


def test_a_step_cell_is_resolved_in_one_place() -> None:
    """Who is present and which view their cell is: step-based scopes always ask ``cells_at``. Names
    come from the trajectory (keying on display names would silently collide characters with the
    same name), and people with no cell this step don't appear."""
    from tuning.audit_reconstruct import AgentStepView, AgentTrajectory, WorldAuditView

    here = [AgentStepView(step=1, world_time="t")]
    view = WorldAuditView(
        world_id="w", init={}, per_step={1: {"a1": [], "a2": [], "elsewhere": []}},
        per_agent={"a1": AgentTrajectory("a1", "李恪", here),
                   "a2": AgentTrajectory("a2", "李恪", here),
                   "elsewhere": AgentTrajectory("elsewhere", "丙", [AgentStepView(step=9, world_time="t")])},
        world_sequence=[{"step": 1, "world_time": "t"}], agent_names={})
    assert [(aid, nm) for aid, nm, _ in view.cells_at(1)] == [("a1", "李恪"), ("a2", "李恪")]


def test_mams_keeps_a_cell_whose_only_news_is_a_goal_finishing() -> None:
    """Rendering decides for itself whether to emit a block. A separate list of "keys that count as
    content" would need updating with every new field, and missing one would swallow a person who
    really has content."""
    from tuning.judge_audit import _mams_lines

    block = "\n".join(_mams_lines(
        "甲", {"goal_progress": [[{"status": "completed", "text": "取兵符"}]]}, sps=None))
    assert "取兵符→已完成" in block
    assert _mams_lines("甲", {}, sps=None) == []      # nothing landed = no block; don't leave a bare name


def test_mams_goal_progress_is_not_truncated_mid_sentence() -> None:
    """Goal progress shows only status transitions, and the text isn't truncated: a cut-off half
    sentence can't tell which goal completed, which is the only information in this cell. The
    duration / origin suffix need.py appends is stripped."""
    from tuning.judge_audit import _mams_lines

    summary = {"results": [{"outcome": "成了"}], "goal_progress": [[
        {"status": "completed", "text": "六月初一上午应付传旨内侍，如实回答太子秦王动向（约定就在此刻）"},
        {"status": "active", "text": "另一条还在推进的目标"}]]}
    block = "\n".join(_mams_lines("甲", summary, sps=None))
    assert "六月初一上午应付传旨内侍，如实回答太子秦王动向→已完成" in block
    assert "另一条还在推进" not in block  # only completed / failed transitions are reported


# The sass header needs the persona baseline (the shared input to every beat); the rendering itself
# only uses init + agent_names.
from tuning.audit_reconstruct import AgentTrajectory as _Traj, WorldAuditView as _View

_VIEW_FOR_SASS = _View(
    world_id="w", per_step={}, world_sequence=[], agent_names={"a1": "丙"},
    per_agent={"a1": _Traj(agent_id="a1", name="丙", steps=[])},
    init={"key_figures": [{"name": "丙", "importance": "main", "age": 30, "role": "将军"}],
          "personas": [{"name": "丙", "agent_id": "a1", "core_traits": ["果决"]}]},
)


def test_sass_reads_what_the_engine_declared_not_the_prompt() -> None:
    """sass shows the facts and lists the engine declared, not prompt text. Parsing the prompt would
    make review shift whenever wording changes or system/user is split, and both of those are tuned
    for the model and the prefix cache."""
    from tuning.audit_reconstruct import AgentStepView, StageCall

    sv = AgentStepView(step=1, world_time="t", calls=[
        StageCall(stage="decision", scene="agent_decision_main",
                  prompt=[{"role": "system", "content": "你是某角色，遵守设定"},
                          {"role": "user", "content": "这段 prompt 文本不该出现在审查里"}],
                  output={"selected_index": 0, "action_description": "找甲说话",
                          "person_indices": [1]},
                  parse_ok=True, ok=True,
                  extra={"given_facts": ["客观经历：昨夜敌军压境"],
                         "action_menu": {"0": "TALK"}, "person_candidates": {"1": "甲"}})])
    block = agent_step_block(_VIEW_FOR_SASS, "a1", sv)

    assert "客观经历：昨夜敌军压境" in block          # declared facts
    assert "找甲说话" in block                       # raw output shown as-is
    assert "拍：decision·agent_decision_main" in block
    assert "这段 prompt 文本不该出现在审查里" not in block
    assert "遵守设定" not in block
    # Indices must be translated: the judge can't tell what #0 / #1 are, so the beat can't be judged.
    assert "selected_index #0 = TALK" in block and "person_indices #1 = 甲" in block
    # But the full candidate list isn't shown. The list is prompt structure, and "is the choice on
    # the list" belongs to the deterministic precheck.
    assert "能触及的人" not in block and "可前往的地方" not in block


def test_sass_still_judges_an_output_the_engine_threw_away() -> None:
    """Fidelity asks whether the output went beyond its own input, regardless of whether the engine
    used it. An intent that invents a person is a hallucination by this machine even if it never
    reached the world.

    Narrative scopes must drop it (it would be compared against someone else's action result); sass
    isn't bound by that since it doesn't show results at all. The re-selected attempt and the
    discarded one are separate beats, scored separately, or one clean and one fabricated would
    average into a single score.
    """
    from tuning.audit_reconstruct import AgentStepView, StageCall
    from tuning.judge_audit import sass_units

    sv = AgentStepView(step=1, world_time="t", calls=[
        StageCall(stage="decision", scene="agent_decision_main",
                  prompt=[], output={"action_description": "找丙说话"}, parse_ok=True, ok=True,
                  adopted=False, reject_reason="talk_target_not_present",
                  extra={"given_facts": ["丢弃那次的情境：可触及的人有甲乙"]}),
        StageCall(stage="decision", scene="agent_decision_main",
                  prompt=[], output={"action_description": "找甲说话"}, parse_ok=True, ok=True,
                  adopted=True, extra={"given_facts": ["重选那次的情境：可触及的人有甲乙"]}),
    ])
    block = agent_step_block(_VIEW_FOR_SASS, "a1", sv)

    assert [u.label for u in sass_units(sv)] == [
        "decision·agent_decision_main", "decision·agent_decision_main#2"]
    for text in ("丢弃那次的情境", "找丙说话", "重选那次的情境", "找甲说话"):
        assert text in block
    # But say clearly that it never reached the world, so the judge doesn't look for a nonexistent
    # result.
    assert "这一份没有进入世界" in block and "talk_target_not_present" in block


def test_a_step_whose_every_decision_was_discarded_is_told_as_a_void_beat() -> None:
    """Nothing adopted = a null step, and the narrative view must say so. An unexplained "没有决策"
    reads as a missed capture or mechanical repetition, when actually nothing happened this beat
    (the direct target of action_efficacy's idling check)."""
    from tuning.audit_reconstruct import AgentStepView, StageCall
    from tuning.judge_audit import _step_lines

    calls = [StageCall(stage="decision", scene="agent_decision_main",
                       prompt=[], output={"action_description": "找丙说话", "selected_index": 0},
                       parse_ok=True, ok=True,
                       adopted=False, reject_reason="talk_target_not_present")]
    s = _sum(calls, "丙")
    assert "decision" not in s                         # a decision that never existed stays out of the summary
    assert s["decision_discarded"] == "talk_target_not_present"
    text = "\n".join(_step_lines(AgentStepView(step=1, world_time="t", summary=s),
                                  name="丙", sps=3600))
    assert "本步决策被引擎丢弃" in text and "找丙说话" not in text


@pytest.mark.asyncio
async def test_judge_stage_unit_scope(trace_dir: str) -> None:
    view = reconstruct_world(WORLD, trace_dir=trace_dir)
    sv = view.per_agent["agent-x"].steps[0]
    res = await judge_scope(_router(_SchemaMock()), SCOPES["single_agent_single_step"],
                            header="h", context_block=agent_step_block(view, "agent-x", sv), n_units=3)
    assert res["total"] == 80.0
    # nested matrix: unit → {metric: score}
    assert all(isinstance(v, dict) for v in res["scores"].values())
    assert res["scores"]["u1"]["grounding"] == 80


@pytest.mark.asyncio
async def test_judge_fallback_on_failure(trace_dir: str) -> None:
    view = reconstruct_world(WORLD, trace_dir=trace_dir)
    res = await judge_scope(_router(_FailProvider()), SCOPES["initialization"],
                            header="h", context_block=init_block(view))
    assert res["total"] is None and res["scores"] == {}
    assert "失败" in res["rationale"]


# ---------------------------------------------------------------------------
# Orchestrator end-to-end (mock judge, real reconstruction + files)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_audit_writes_five_scopes(trace_dir: str) -> None:
    from tuning.audit import run_audit

    config = SimpleNamespace(engine=SimpleNamespace(max_concurrent_llm=4))
    summary = await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir)

    assert summary["world_score"] == 80.0
    assert set(summary["scope_totals"]) == set(SCOPES)
    assert all(v == 80.0 for v in summary["scope_totals"].values())
    # Everything run → every dimension's evidence shares are all present.
    assert summary["coverage"] == 1.0
    assert all(d["coverage"] == 1.0 for d in summary["dimensions"].values())
    assert summary["fidelity_score"] == 80.0     # sass stands apart, reported alongside the world score
    assert summary["mixed_provenance"] is False
    # 7-dimension scorecard: every dimension scored (mock=80), weights sum to 100.
    from tuning.audit_metrics import DIMENSIONS
    assert set(summary["dimensions"]) == set(DIMENSIONS)
    assert all(d["score"] == 80.0 for d in summary["dimensions"].values())
    # Each dimension reports who supplied its score. Shares come from the weight table, not sample
    # counts.
    assert set(summary["dimensions"]["causal_coherence"]["sources"]) == {
        "single_agent_multi_step", "multi_agent_single_step", "initialization"}
    assert set(summary["scope_provenance"]) == set(SCOPES)

    out = Path(trace_dir) / WORLD / "audit"
    for name in ("summary.json", "initialization.json", "multi_agent_multi_step.json",
                 "multi_agent_single_step.json", "single_agent_multi_step.json",
                 "single_agent_single_step.json"):
        assert (out / name).exists(), name

    sass = json.loads((out / "single_agent_single_step.json").read_text(encoding="utf-8"))
    assert "agent-x#1" in sass and sass["agent-x#1"]["step"] == 1
    # scored entries must NOT carry the raw prompt/response (those live in audit_calls.json).
    assert "_prompt" not in sass["agent-x#1"] and "_response" not in sass["agent-x#1"]
    sams = json.loads((out / "single_agent_multi_step.json").read_text(encoding="utf-8"))
    assert set(sams) == {"agent-x", "agent-y"}

    # every audit judge call is persisted with its prompt + raw response, keyed by scope→entry.
    calls = json.loads((out / "audit_calls.json").read_text(encoding="utf-8"))
    assert calls["single_agent_single_step"]["agent-x#1"]["prompt"][0]["role"] == "system"
    assert calls["single_agent_single_step"]["agent-x#1"]["response"]
    assert calls["initialization"]["_"]["prompt"]


@pytest.mark.asyncio
async def test_run_audit_scope_and_agent_selection(trace_dir: str) -> None:
    from tuning.audit import run_audit

    config = SimpleNamespace(engine=SimpleNamespace(max_concurrent_llm=4))
    s = await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir,
                        scopes=["single_agent_single_step"], agents=["甲"])  # name → agent-x
    assert s["ran_scopes"] == ["single_agent_single_step"]
    assert s["scope_totals"]["single_agent_single_step"] == 80.0
    assert s["scope_totals"]["multi_agent_multi_step"] is None  # not run
    # Only sass run: fidelity has a score, but no narrative dimension has evidence → world score is
    # null; an engineering score doesn't stand in for it.
    assert s["fidelity_score"] == 80.0
    assert s["world_score"] is None and s["coverage"] == 0.0
    out = Path(trace_dir) / WORLD / "audit"
    sass = json.loads((out / "single_agent_single_step.json").read_text(encoding="utf-8"))
    assert set(sass) == {"agent-x#1"}                      # only "甲", step 1
    assert not (out / "multi_agent_multi_step.json").exists()  # skipped scope not written


@pytest.mark.asyncio
async def test_run_audit_incremental_merge(trace_dir: str) -> None:
    from tuning.audit import run_audit

    config = SimpleNamespace(engine=SimpleNamespace(max_concurrent_llm=4))
    out = Path(trace_dir) / WORLD / "audit"
    await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir,
                    scopes=["single_agent_single_step"], agents=["agent-x"])
    await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir,
                    scopes=["single_agent_single_step"], agents=["agent-y"])
    sass = json.loads((out / "single_agent_single_step.json").read_text(encoding="utf-8"))
    assert set(sass) == {"agent-x#1", "agent-y#1"}  # y merged in, x preserved

    s3 = await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir,
                         scopes=["initialization"])
    # Scope totals are all recomputed from disk: the one not rerun stays, the newly run one is added.
    assert s3["scope_totals"]["initialization"] == 80.0
    assert s3["scope_totals"]["single_agent_single_step"] == 80.0
    # Opening evidence can't stand alone: dimensions scored only by init count as lacking evidence,
    # so the world score isn't propped up by one build-time score.
    assert s3["world_score"] is None
    assert s3["dimensions"]["persona_worldview"]["score"] is None
    assert s3["dimensions"]["persona_worldview"]["coverage"] == 0.2


@pytest.mark.asyncio
async def test_run_audit_rejects_unknown_scope(trace_dir: str) -> None:
    from tuning.audit import run_audit

    with pytest.raises(ValueError):
        await run_audit(None, SimpleNamespace(engine=SimpleNamespace(max_concurrent_llm=4)),
                        WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir, scopes=["nope"])


def test_cli_parse_scopes_aliases() -> None:
    from tuning.cli import _parse_scopes

    assert _parse_scopes("sass,mams") == ["single_agent_single_step", "multi_agent_multi_step"]
    assert _parse_scopes("initialization") == ["initialization"]
    assert _parse_scopes(None) is None
    with pytest.raises(SystemExit):
        _parse_scopes("bogus")


def test_every_scale_the_audit_renders_has_its_definition_in_the_same_block() -> None:
    """Every scale in the review context must have its definition in the same block, or the judge
    scores values with no basis. Definitions come from core/prompts, not hand copies. A new field
    that renders a scale with no definition fails here.
    """
    from tuning.judge_audit import (
        _cell_legend_lines, _emotion_legend, _need_legend, _pressure_legend)
    from tuning.audit_reconstruct import AgentStepView, AgentTrajectory, WorldAuditView
    from tuning.judge_audit import agent_trajectory_block, cross_agent_block

    sv = AgentStepView(step=1, world_time="t", summary={
            "relations": [{"name": "乙", "labels": "君臣:君主", "trust": 0.7, "affection": 0.3}],
            "emotion": {"primary": "anger", "intensity": 0.8, "valence": -0.9},
            "perceived": ["外部压力（threat，紧迫度【紧急】）：提防东宫"],
            "need_activation": {"safety": 0.9},
            "dominant_need": "safety",
            "prior_goals": ["去北营取兵符（约定时刻已过约4小时）"],
            "decision": {"action_type": "TALK", "action_description": "找乙说话"},
        })
    view = WorldAuditView(
        world_id="w", init={"key_figures": [], "personas": [], "seconds_per_step": 7200},
        per_agent={"a1": AgentTrajectory(agent_id="a1", name="甲", steps=[sv])},
        per_step={1: {"a1": []}}, world_sequence=[{"step": 1, "world_time": "t"}],
        agent_names={"a1": "甲"})

    # (token appearing in the rendering, definition snippet that must appear in the same block)
    scales = [
        ("信任0.7", "0.7=信任"),                    # trust
        ("好感0.3", "0.3=亲近"),                    # affection
        ("「君臣:君主」", "「关系类型:角色」格式"),      # labels
        ("强度0.8", "0.8=强烈"),                    # emotion intensity
        ("效价-0.9", "<0 痛苦/消极"),                # emotion valence
        ("紧迫度【紧急】", "【紧急】=应当优先"),         # urgency
        ("（threat，", "threat=威胁安全"),            # drive_type
        ("激活需求", "need_activation（需求激活度）"),  # activation
        ("主导需求 safety", "safety: 减少不确定性"),    # need type
        ("[TALK]", "TALK："),                       # action type menu
        ("（约定时刻已过约4小时）", "「（约定时刻已过…）」= 时刻过了仍未了结"),  # goal overdue marker
        ("第 1 步（t）", "一步是约2小时"),             # world time at step start + step length
    ]
    # sass has to pass this too: its facts include trust / affection / urgency / overdue markers,
    # and its output includes emotion intensity, valence, and need activation.
    sass = agent_step_block(view, "a1", sv)
    sams, mass = agent_trajectory_block(view, "a1"), cross_agent_block(view, 1)
    for block in (sams, mass):
        for token, definition in scales:
            assert token in block, f"用例没渲染出 {token}，这条守卫会漏掉它"
            assert definition in block, f"渲染了 {token} 却没给它的定义（{definition}）"
    # sams and mass are two transpositions of the same cell and show the same content, so the
    # scales must be the same too.
    for line in _cell_legend_lines(7200):
        assert line in sams and line in mass
    # sass shows different things, but none of the scales it renders can be missing.
    for scale in ("关系", "情绪", "需求", "外部压力", "时间"):
        assert f"  {scale}：" in sass, f"sass 少了「{scale}」的刻度定义"
    # All three legends come from the authoritative constants, not local copies. A copy drifts (the
    # anchors go first).
    assert "0.3=可感知但不紧迫" in _need_legend()      # anchor of NEED_INTENSITY_DEFINITION
    assert "1.0=典型重要" in _need_legend()            # anchor of NEED_WEIGHT_DEFINITION
    assert "【危急】" in _pressure_legend() and "event=世界事件" in _pressure_legend()
    assert "neutral/joy/sadness" in _emotion_legend()


def test_what_the_audit_renders_has_something_to_judge_it_by() -> None:
    """Rendering a block without criteria is a wasted injection: the judge sees it but doesn't know
    what to look for (and conversely, whatever the criteria mention must actually render). This
    guards two such blocks: the opening's errand-runners and the overdue markers on goals."""
    from tuning.audit_metrics import SCOPES

    init_c = " ".join(c for m in SCOPES["initialization"].metrics for c in m.criteria)
    assert "差役" in init_c                       # init_block shows the errand-runner list
    sams_c = " ".join(c for m in SCOPES["single_agent_multi_step"].metrics for c in m.criteria)
    assert "约定时刻已过" in sams_c                # goal lines show the overdue marker
    assert "人设基线" in sams_c or "人设漂移" in sams_c   # the full persona baseline is shown at the top of the trajectory


def test_a_scopes_say_does_not_grow_with_the_number_of_calls_it_made() -> None:
    """The point of two-level aggregation: a scope's say comes from the weight table, not from how
    many calls it produced.

    mass runs once per step and sams once per person. Pooling scores flat would let "how many steps
    ran" silently reweight everything, and a 30-step world's score couldn't be compared with a
    100-step world's.
    """
    from tuning.audit_metrics import aggregate_dimensions

    few = {"causal_coherence": {"single_agent_multi_step": [100], "multi_agent_single_step": [0],
                                "initialization": [100]}}
    many = {"causal_coherence": {"single_agent_multi_step": [100],
                                 "multi_agent_single_step": [0] * 200,
                                 "initialization": [100]}}
    assert aggregate_dimensions(few)[0]["causal_coherence"]["score"] == \
        aggregate_dimensions(many)[0]["causal_coherence"]["score"] == 70.0  # 55*100 + 30*0 + 15*100


def test_missing_scopes_renormalise_and_are_declared_as_coverage() -> None:
    """Running only some scopes still produces a score (normalized among those present), but
    coverage is reported with it. Scores with different coverage aren't comparable."""
    from tuning.audit_metrics import aggregate_dimensions

    card, _world, _cov = aggregate_dimensions(
        {"persona_worldview": {"single_agent_multi_step": [60], "multi_agent_multi_step": [100]}})
    dim = card["persona_worldview"]
    assert dim["score"] == 70.0                       # 60/80 * 60 + 20/80 * 100, init absent
    assert dim["coverage"] == 0.8
    assert dim["sources"]["single_agent_multi_step"]["share"] == 0.75


def test_a_half_witnessed_dimension_pulls_half_as_hard_on_the_world_score() -> None:
    """Estimate and confidence are separate: normalization gives the best estimate, and
    ``coverage`` decides how much the dimension weighs in the world score.

    When one witness abstains and another has the dimension to itself, the answer is still the best
    estimate, but it shouldn't weigh as much as a dimension with full evidence.
    """
    from tuning.audit_metrics import aggregate_dimensions

    card, world, _cov = aggregate_dimensions({
        # Full evidence (sams+mams+init), full score
        "persona_worldview": {"single_agent_multi_step": [100], "multi_agent_multi_step": [100],
                              "initialization": [100]},
        # Half the witnesses (sams, weight 50, is absent), score 0
        "dramatic_arc": {"multi_agent_multi_step": [0]},
    })
    assert card["persona_worldview"]["effective_weight"] == 20      # 20 × 1.0
    assert card["dramatic_arc"]["effective_weight"] == 7.5          # 15 × 0.5
    assert card["dramatic_arc"]["score"] == 0.0                     # the estimate itself isn't discounted
    # (100×20 + 0×7.5) / 27.5 = 72.7; without shrinking the design weight it would be
    # (100×20 + 0×15)/35 = 57.1
    assert world == 72.7


def test_full_coverage_leaves_the_designed_weights_untouched() -> None:
    """Shrinking only applies when evidence is incomplete. With full evidence, the world score
    matches the design weights exactly."""
    from tuning.audit_metrics import DIMENSIONS, aggregate_dimensions, _EVIDENCE_WEIGHTS

    per = {did: {sk: [50.0] for sk in w} for did, w in _EVIDENCE_WEIGHTS.items()}
    card, world, cov = aggregate_dimensions(per)
    assert cov == 1.0 and world == 50.0
    assert all(c["effective_weight"] == DIMENSIONS[did].weight for did, c in card.items())


def test_build_time_evidence_cannot_carry_a_dimension_alone() -> None:
    """init can only corroborate: a dimension with only init present counts as lacking evidence.
    Otherwise normalization would inflate one build-time score into the whole dimension.

    By contrast, mass has a narrow view but reviews the same runtime object, so on its own it still
    scores (coverage shows how narrow it is).
    """
    from tuning.audit_metrics import aggregate_dimensions

    card, world, _ = aggregate_dimensions({"causal_coherence": {"initialization": [100]}})
    assert card["causal_coherence"]["score"] is None and world is None
    card, world, _ = aggregate_dimensions({"causal_coherence": {"multi_agent_single_step": [40]}})
    assert card["causal_coherence"]["score"] == 40.0 and world == 40.0


def test_build_time_evidence_never_grows_past_its_designed_share() -> None:
    """When others are absent or abstain, init's share is capped at its design value; the freed
    weight is redistributed only among runtime scopes.

    Otherwise "build-time evidence can't pass as runtime evidence" has a side door: one abstention
    by the judge on a dimension and the build-time vote gets inflated (init's share can more than
    double).
    """
    from tuning.audit_metrics import aggregate_dimensions

    # Design: sams 60 / mams 20 / init 20. sams absent → init stays at 20%, and the freed 60 all
    # goes to mams.
    card, _w, _c = aggregate_dimensions(
        {"persona_worldview": {"multi_agent_multi_step": [70], "initialization": [100]}})
    src = card["persona_worldview"]["sources"]
    assert src["initialization"]["share"] == 0.2 and src["multi_agent_multi_step"]["share"] == 0.8
    assert card["persona_worldview"]["score"] == 76.0

    # Everyone present → the shares are the design table itself.
    card, _w, _c = aggregate_dimensions(
        {"persona_worldview": {"single_agent_multi_step": [100], "multi_agent_multi_step": [100],
                               "initialization": [100]}})
    assert {k: v["share"] for k, v in card["persona_worldview"]["sources"].items()} == {
        "single_agent_multi_step": 0.6, "multi_agent_multi_step": 0.2, "initialization": 0.2}


def test_the_three_reasons_a_dimension_has_no_score_are_reported_apart() -> None:
    """Not run / criteria changed / run but unanswered all show up as a missing score, but call for
    completely different fixes, so they can't look the same."""
    from tuning.audit_metrics import aggregate_dimensions

    card, _w, _c = aggregate_dimensions(
        {"causal_coherence": {"single_agent_multi_step": [80]}},
        audited={"single_agent_multi_step", "multi_agent_single_step", "initialization"},
        stale={"initialization"})
    dim = card["causal_coherence"]
    assert dim["abstained"] == ["multi_agent_single_step"]   # asked but unanswered → check criteria / prompt
    assert dim["stale"] == ["initialization"]                # that dimension wasn't in the criteria at the time → rerun the scope
    # Neither mass nor init is in sources, but for different reasons; scopes that weren't run
    # aren't recorded under either.


def test_a_scope_digest_moves_when_the_dimension_list_does_not_only_the_criteria() -> None:
    """The fingerprint must cover the scope→dimension table. It lives in Python, so a hash of the
    criteria files can't see it.

    Otherwise, adding a dimension to a scope makes old sams results show that dimension as "no
    witness" while the criteria fingerprint stays unchanged.
    """
    from tuning.audit_metrics import build_scopes, scope_digest

    scopes = build_scopes()
    before = scope_digest("single_agent_multi_step", scopes)
    trimmed = dict(scopes)
    sc = trimmed["single_agent_multi_step"]
    trimmed["single_agent_multi_step"] = type(sc)(
        sc.key, sc.label, sc.purpose, [m for m in sc.metrics if m.id != "persona_worldview"], sc.unit)
    assert scope_digest("single_agent_multi_step", trimmed) != before


@pytest.mark.asyncio
async def test_run_audit_flags_scopes_judged_under_an_older_metric_set(trace_dir: str) -> None:
    """If scores on disk weren't produced under the current criteria, the world score stitches two
    versions together. Still compute it, but say so."""
    from tuning.audit import run_audit

    config = SimpleNamespace(engine=SimpleNamespace(max_concurrent_llm=4))
    await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir)
    meta_path = Path(trace_dir) / WORLD / "audit" / "scope_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert all(m["digest"] for m in meta.values())

    meta["single_agent_multi_step"]["digest"] = "stale00000000"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    s = await run_audit(None, config, WORLD, judge_provider=_SchemaMock(), trace_dir=trace_dir,
                        scopes=["initialization"])
    assert s["stale_scopes"] == ["single_agent_multi_step"]
    # A scope whose criteria changed produced a score: still count it, but flag every source. Only
    # missing scores go into the dimension-level stale list.
    assert s["dimensions"]["character_growth"]["sources"]["single_agent_multi_step"]["stale"] is True


def test_a_dropped_metric_key_is_recorded_not_silently_taken_as_abstention() -> None:
    """The prompt requires every dimension to appear (null when it can't judge), so one missing
    entirely is a dropped field, a defect rather than an abstention."""
    from tuning.judge_audit import _normalise

    res = _normalise({"rationale": "r", "scores": {"causal_coherence": 80}},
                     SCOPES["initialization"])
    assert res["dropped"] == ["persona_worldview"]
    assert _normalise({"rationale": "r", "scores": {"causal_coherence": 80, "persona_worldview": None}},
                      SCOPES["initialization"])["dropped"] == []   # explicit null = abstention, not an omission


def test_evidence_weights_stay_aligned_with_the_scope_table() -> None:
    """The two tables must describe the same thing. A missing cell would silently drop a line of
    evidence, so it raises at import time (Rule 3)."""
    from tuning.audit_metrics import _EVIDENCE_WEIGHTS, _SCOPE_DEFS, DIMENSIONS, _validate_evidence_weights

    assert all(sum(w.values()) == 100 for w in _EVIDENCE_WEIGHTS.values())
    assert set(_EVIDENCE_WEIGHTS) == set(DIMENSIONS)
    _validate_evidence_weights()

    saved = _EVIDENCE_WEIGHTS["character_growth"]
    _EVIDENCE_WEIGHTS["character_growth"] = {"multi_agent_multi_step": 100}
    try:
        with pytest.raises(ValueError):
            _validate_evidence_weights()
    finally:
        _EVIDENCE_WEIGHTS["character_growth"] = saved
    assert _SCOPE_DEFS  # the table is still in place


def test_the_fidelity_probe_never_reaches_the_world_score() -> None:
    """sass judges whether the machine is faithful to its input: engineering quality, not narrative
    quality. Its metrics aren't in DIMENSIONS, so structurally they can't enter the world score,
    without relying on any scope-level switch."""
    from tuning.audit_metrics import DIMENSIONS, SCOPES, aggregate_dimensions

    sass = {m.id for m in SCOPES["single_agent_single_step"].metrics}
    assert sass and not (sass & set(DIMENSIONS))
    card, world, _cov = aggregate_dimensions({mid: {"single_agent_single_step": [0.0]} for mid in sass})
    assert world is None and all(d["score"] is None for d in card.values())


def test_an_off_menu_index_is_reported_by_the_engine_not_re_derived() -> None:
    """At runtime IndexedRef silently filters out out-of-range indices, leaving no trace in the
    world. Unless the engine reports it, it can never be found.

    Review doesn't check this itself: that would need another copy of the "which slot reads which
    list" table here, whose real owner is decision.py's output schema. A copy is a second source of
    truth that goes silently stale whenever the schema changes."""
    from tuning.audit_checks import check_bindings
    from tuning.audit_reconstruct import AgentStepView, StageCall

    sv = AgentStepView(step=1, world_time="t", calls=[StageCall(
        stage="decision", scene="s", prompt=[], parse_ok=True, ok=True,
        output={"person_indices": [1, 7]},
        extra={"dropped_indices": ["person_indices 填了 [1, 7]，只有 1 个落在这份名单（共 1 项）内"],
               "person_candidates": {"1": "甲"}})])
    found = check_bindings(sv)
    assert [f["kind"] for f in found] == ["index_off_menu"]
    assert "person_indices" in found[0]["detail"]

    # Nothing declared by the engine = nothing out of range. Review doesn't count the lists itself.
    quiet = AgentStepView(step=1, world_time="t", calls=[StageCall(
        stage="decision", scene="s", prompt=[], parse_ok=True, ok=True,
        output={"person_indices": [1, 7]}, extra={"person_candidates": {"1": "甲"}})])
    assert check_bindings(quiet) == []


def test_no_bare_index_is_left_for_the_judge_to_decode() -> None:
    """sass shows raw output, and raw output is full of IndexedRef indices. Every one must be
    translated to a name.

    The judge can't tell what `selected_index: 2` means. Translations come from the candidate maps
    the engine declares, not the whole list in the prompt: the list is prompt structure, and "is the
    choice on the list" belongs to the deterministic precheck. A new indexed output without a map
    fails here.
    """
    from tuning.audit_reconstruct import StageCall
    from tuning.judge_audit import _index_gloss

    gloss = "".join(_index_gloss(StageCall(
        stage="decision", scene="s", prompt=[], parse_ok=True, ok=True,
        output={"selected_index": 2, "person_indices": [1, 2], "destination_index": 3,
                "errand_npc_index": 1, "physical_entity_index": 1, "updated_index": 1,
                "goals": [{"index": 1, "status": "completed"}],
                "dialogue": [{"speaker": 1, "line": "…"}]},
        extra={"action_menu": {"2": "MOVE"},
               "person_candidates": {"1": "甲", "2": "乙"},
               "destination_candidates": {"3": "玄武门"},
               "npc_candidates": {"1": "马夫"},
               "item_candidates": {"1": "兵符"},
               "goal_texts": {"1": "取兵符"},
               "action_participants": ["甲", "乙"]})))
    for expect in ("selected_index #2 = MOVE", "person_indices #1 = 甲", "person_indices #2 = 乙",
                   "destination_index #3 = 玄武门", "errand_npc_index #1 = 马夫",
                   "physical_entity_index #1 = 兵符", "updated_index #1 = 兵符",
                   "目标 #1 = 取兵符", "对白 speaker 1 = 甲、2 = 乙"):
        assert expect in gloss, f"序号没译出来：{expect}"

    # No indices, no line. An empty comparison column is noise.
    assert _index_gloss(StageCall(stage="perception", scene="s", prompt=[], parse_ok=True,
                                  ok=True, output={"emotion": "fear"}, extra={})) == []


def test_when_and_where_follow_one_contract_across_every_scope() -> None:
    """Time belongs to the step, place belongs to the beat.

    Time doesn't change within a step, so the step separator / cell header states it once. Place
    is per person and can change within a step (after a journey), so one label for the whole cell
    would judge correct output against the wrong reference.

    sams / mass / mams give "他在" per agent; sass gives each beat its own "此刻何时何地", as that
    prompt showed it.
    """
    from tuning.audit_reconstruct import AgentStepView, AgentTrajectory, StageCall, WorldAuditView
    from tuning.judge_audit import (
        agent_step_block, agent_trajectory_block, cross_agent_block, world_block,
    )

    summ = {"location": "东宫", "decision": {"action_type": "MOVE", "action_description": "我去玄武门"}}
    sv = AgentStepView(step=1, world_time="武德九年，六月初一，中午十二点", summary=summ, calls=[
        StageCall(stage="perception", scene="agent_decision_main", prompt=[], parse_ok=True, ok=True,
                  output={"emotion": "fear"},
                  extra={"given_facts": ["此刻何时何地：我此刻在东宫，时间为中午十二点。"]}),
        StageCall(stage="action", scene="agent_action_narration", prompt=[], parse_ok=True, ok=True,
                  output={"fact": "我到了玄武门"},
                  # After a journey, this beat's location is no longer the one at step start.
                  extra={"given_facts": ["此刻何时何地：我此刻在玄武门，时间为中午十二点。"]}),
    ])
    view = WorldAuditView(
        world_id="w", per_step={1: {"a1": []}}, world_sequence=[{"step": 1, "world_time": sv.world_time}],
        per_agent={"a1": AgentTrajectory("a1", "乙", [sv])}, agent_names={"a1": "乙"},
        init={"key_figures": [], "personas": [], "seconds_per_step": 7200})

    for name, block in (("sams", agent_trajectory_block(view, "a1")),
                        ("mass", cross_agent_block(view, 1)),
                        ("mams", world_block(view))):
        assert "他在：东宫" in block, f"{name} 没交代人在哪"
        # The time appears once, in the step separator, not repeated per person.
        assert block.count(sv.world_time) == 1, f"{name} 把时刻重复了 {block.count(sv.world_time)} 遍"

    sass = agent_step_block(view, "a1", sv)
    # Location changed within the step: each beat keeps its own time and place; they can't be
    # merged.
    assert "我此刻在东宫" in sass and "我此刻在玄武门" in sass
    assert "每一拍自己的「此刻何时何地」才是它的时空" in sass


def test_where_he_stood_survives_a_step_he_never_perceived() -> None:
    """A location declared by any beat counts, not just the perception beat.

    Scheduling doesn't guarantee everyone re-perceives every step. Steps that only settle an old
    action or only run feedback still form a cell, and the person is still somewhere. Counting only
    perception leaves such cells without "他在：", which can drop whole locations from the view,
    and judging "mutually exclusive actions in one place" or "did he show up when the appointment
    came due" depends on it.
    """
    from tuning.audit_reconstruct import (
        AgentStepView, AgentTrajectory, StageCall, WorldAuditView, summarise)
    from tuning.judge_audit import cross_agent_block

    calls = [StageCall(stage="feedback", scene="agent_decision_main", prompt=[], parse_ok=True,
                       ok=True, output={"emotion": "anger"}, extra={"location": "太极宫"})]
    sv = AgentStepView(step=1, world_time="t", summary=summarise(calls, "甲"), calls=calls)
    view = WorldAuditView(
        world_id="w", per_step={1: {"a1": []}}, world_sequence=[{"step": 1, "world_time": "t"}],
        per_agent={"a1": AgentTrajectory("a1", "甲", [sv])}, agent_names={"a1": "甲"},
        init={"key_figures": [], "personas": []})
    assert "他在：太极宫" in cross_agent_block(view, 1)

    # Complement: if no beat ever declared a location, don't invent one.
    bare = [StageCall(stage="feedback", scene="agent_decision_main", prompt=[], parse_ok=True,
                      ok=True, output={"emotion": "anger"}, extra={})]
    assert "location" not in summarise(bare, "甲")


def test_no_code_layer_token_reaches_the_judge_unexplained() -> None:
    """General guard: any code-layer symbol rendered for the judge must have its meaning in the same
    block.

    This doesn't name specific enums. It pulls English snake_case values straight out of the
    rendered output, so any new enum output, scene, or state fails here without waiting for someone
    to notice each one.

    Only two kinds are exempt: ① values whose meaning is given in this block's legend (the enum
    value appears with its explanation); ② values explicitly described as code-layer markers that
    need no interpretation (like scene suffixes or the engine's internal not-adopted marker).
    """
    import re

    from tuning.audit_reconstruct import AgentStepView, AgentTrajectory, StageCall, WorldAuditView
    from tuning.judge_audit import agent_step_block

    sv = AgentStepView(step=3, world_time="武德九年，六月初一，中午十二点", calls=[
        StageCall(stage="decision", scene="agent_decision_main", prompt=[], parse_ok=True, ok=True,
                  output={"selected_index": 0, "urgency": "high", "action_description": "找甲说话"},
                  extra={"given_facts": ["此刻何时何地：我此刻在东宫，时间为武德九年，六月初一，中午十二点。",
                                         "客观经历：昨夜甲来过"], "action_menu": {"0": "TALK"},
                         "verdict": "conscripted"}),
        StageCall(stage="feedback", scene="need_goal_generation", prompt=[], parse_ok=True, ok=True,
                  output={"goals": [{"index": 1, "status": "completed"}]},
                  extra={"given_facts": ["实际结果：成了"], "goal_texts": {"1": "取兵符"}}),
    ])
    view = WorldAuditView(
        world_id="w", per_step={}, world_sequence=[], agent_names={"a1": "乙"},
        per_agent={"a1": AgentTrajectory("a1", "乙", [sv])},
        init={"key_figures": [], "personas": [], "seconds_per_step": 21600})
    block = agent_step_block(view, "a1", sv)

    # Criterion: if a symbol appears in the body, its meaning must appear in the field-description
    # area (before the first "拍：").
    legend, _, body = block.partition("──── 拍：")
    tokens = {t for pat in (r'"[\w]+"\s*:\s*"([a-z][a-z0-9_]{2,})"', r'= ([A-Z][A-Z_]{2,})\b')
              for t in re.findall(pat, body)}
    assert tokens, "用例没渲染出任何代码层取值，这条守卫会空转"
    for tok in tokens:
        assert tok in legend, f"「{tok}」出现在判官面前，字段说明里却没有它的含义"

    # Time and place are given per beat (see test_when_and_where_follow_one_contract_across_every_scope).
    assert "每一拍自己的「此刻何时何地」才是它的时空" in block
    assert "一步" in block and "6小时" in block
    assert "此刻何时何地：我此刻在东宫，时间为武德九年，六月初一，中午十二点。" in block
    # How to read beat names + scene is only a code-layer marker
    assert "perception 感知" in block and "是代码层标记，不必解读" in block


def test_each_scope_only_carries_what_its_window_can_witness() -> None:
    """Each scope covers what only it can see. Dimension scores are averaged across scopes, so
    taking on a dimension a scope can't see well adds noise to it rather than evidence."""
    from tuning.audit_metrics import SCOPES

    dims = {k: {m.id for m in sc.metrics} for k, sc in SCOPES.items()}
    # Drift across steps is only visible over a whole trajectory, and sams shows the persona
    # baseline at the top.
    assert "persona_worldview" in dims["single_agent_multi_step"]
    # Judging whether an action worked requires seeing its result. Actions often span several steps
    # with the result landing in another step, and mass only sees one step.
    assert "action_efficacy" not in dims["multi_agent_single_step"]
    assert "action_efficacy" in dims["single_agent_multi_step"]   # sees the whole trajectory
    assert "action_efficacy" in dims["multi_agent_multi_step"]    # sees the whole world


def test_a_multiline_signal_stays_one_bullet() -> None:
    """A perceived signal can itself span several lines (an errand-runner's report has lines for
    location / who's present / items). Prefixing only the first line with "- " drops continuation
    lines back to column one, and the judge reads them as separate signals, defeating the point of
    a list that is meant to be checked item by item."""
    from tuning.judge_audit import _bullets

    report = "收到来自内侍的消息：我去了皇城一趟。\n- 皇城此处无人，话没有人听见。\n- 在场的其他人：无"
    out = _bullets([report, "另一条信号"])
    tops = [ln for ln in out.splitlines() if ln.startswith("    - ")]
    assert len(tops) == 2, "两条信号 = 两个顶层条目,续行不许冒充第三条"
    assert all(ln.startswith("      ") for ln in out.splitlines() if ln not in tops)


def test_trajectory_header_gives_the_world_span_not_just_his_own_steps() -> None:
    """Someone idle most of the time appears in only a few steps (say 8 steps with calls in a
    20-step world). If the header reports only the latter, the judge assumes an 8-step window and
    misjudges arc / pacing / repetition. Both numbers are needed."""
    from tuning.audit_reconstruct import AgentStepView, AgentTrajectory, WorldAuditView
    from tuning.judge_audit import agent_trajectory_block

    steps = [AgentStepView(step=n, world_time="t") for n in (1, 7, 20)]
    view = WorldAuditView(
        world_id="w", init={"key_figures": [], "personas": []},
        per_agent={"a1": AgentTrajectory(agent_id="a1", name="甲", steps=steps)},
        per_step={}, world_sequence=[{"step": n} for n in range(1, 21)], agent_names={"a1": "甲"})
    head = agent_trajectory_block(view, "a1").splitlines()[0]
    assert "世界共 20 步" in head and "下列 3 步" in head


def test_maintenance_products_render_readably_without_bare_indices() -> None:
    """End-of-step maintenance output isn't dumped as raw JSON, because its IndexedRef indices have
    no list to check against in review. Relation evolution's target_index is translated to who the
    other person is (judging whether a relation flip is abrupt requires knowing who it's about);
    reflection's source_indices are just its own footnotes the judge can't look up, so they're
    omitted."""
    from tuning.judge_audit import _step_level_lines

    lines = _step_level_lines({
        "relation_evolution": {"updates": [{"rationale": "他亲口允诺，默契加深", "target_index": 1,
                                            "target": "李世民", "labels": ["姻亲:妻兄", "君臣:臣属"],
                                            "summary": "信任上升"}]},
        "reflection": {"insights": [{"source_indices": [1, 2, 3], "text": "我才明白他等的不是准信。",
                                     "importance": 0.95}]},
    })
    block = "\n".join(lines)
    assert "⟳关系演化：对 李世民 「姻亲:妻兄、君臣:臣属」信任上升（他亲口允诺，默契加深）" in block
    assert "⟳反思（重要度0.95）：我才明白他等的不是准信。" in block
    assert "target_index" not in block and "source_indices" not in block   # bare indices stay out of the context
    # When the list is missing (an older trace, or the annotation wasn't attached), fall back to a
    # descriptive referent, not a bare index.
    fallback = "\n".join(_step_level_lines(
        {"relation_evolution": {"updates": [{"target_index": 2, "summary": "疏远了"}]}}))
    assert "对 某人 疏远了" in fallback


def test_init_keys_on_the_cast_and_shows_the_tier() -> None:
    """Initialization is keyed on key_figures (the authoritative cast): if persona generation missed
    someone, they'd vanish from review entirely, and "a character is missing" is exactly what this
    tier should catch. The tier is shown too: background characters don't need to carry the main
    plot."""
    from tuning.audit_reconstruct import WorldAuditView
    from tuning.judge_audit import init_block

    view = WorldAuditView(
        world_id="w", per_agent={}, per_step={}, world_sequence=[], agent_names={},
        init={"key_figures": [{"name": "甲", "importance": "main", "age": 30, "role": "王"},
                              {"name": "乙", "importance": "background", "age": 20, "role": "卒"}],
              "personas": [{"name": "甲", "agent_id": "a1", "core_traits": ["果决"]}],
              "initial_relations": [], "historical_events": [], "cast_roles": []})
    block = init_block(view)
    assert "【甲】（主角）" in block
    assert "【乙】（背景）" in block, "persona 缺失的角色仍要出现在名册里"
    assert "没有生成人设" in block   # and says what it's missing


