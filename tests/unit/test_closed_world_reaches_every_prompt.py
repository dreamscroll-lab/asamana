"""The closed-world rule really reaches every place it should — and only those places.

Why a separate file (same reason as test_absolute_time_reaches_every_prompt): the model recognizes
stories from training and adds people the world doesn't have — no id, on no list, never responding.
No goal set around such a person can finish, and the failed goal is reborn under new wording.

Three layers of assertion, all required:
1. Present — the prompts that should have it really render it.
2. Right person — first-person sites use the FIRST_PERSON version, functional sites the
   third-person version (the §1 role split).
3. No overreach — prompts on the exclusion list really do NOT have it: a pure-judgment prompt
   outputs a score/boolean + a short reason that never enters the narrative substrate, so adding
   the rule just wastes prefix.
"""

from __future__ import annotations

import inspect

from core.prompts import (
    CLOSED_WORLD_FACT_RULE,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
)

#: The sentence both person versions share — one sentence tells whether the rule is present,
#: without distinguishing which version.
MARK = "这里发生过什么、没发生过什么"
#: Only this rule treats the "the model recognizes this story" failure mode; no other rule does.
FAMILIARITY_MARK = "见过的"


def test_both_person_variants_carry_the_shared_marks() -> None:
    """The two MARKs are the pivot of every assertion in this file — first prove both versions
    really share them."""
    for rule in (CLOSED_WORLD_FACT_RULE, CLOSED_WORLD_FACT_RULE_FIRST_PERSON):
        assert MARK in rule
        assert FAMILIARITY_MARK in rule


def test_the_two_variants_never_swap_their_voice() -> None:
    """Mixing persons breaks first-person immersion on the spot, so neither version may contain the
    other's pronouns."""
    assert "我" not in CLOSED_WORLD_FACT_RULE
    assert "你" not in CLOSED_WORLD_FACT_RULE_FIRST_PERSON


# ---------------------------------------------------------------------------
# Tier 1 — where phantom people are born: goals and decisions
#
# These four sites produce things that get acted on. A person invented out of nowhere only has to
# land here once to be copied all the way down the goal queue.
# ---------------------------------------------------------------------------


def test_short_term_goal_prompt_carries_the_rule() -> None:
    """Short-term goals are the main birthplace: a goal to "summon someone who doesn't exist" gets
    queued and read on every later beat."""
    from agent.need import _SHORT_TERM_GOAL_SYSTEM

    assert MARK in _SHORT_TERM_GOAL_SYSTEM
    assert CLOSED_WORLD_FACT_RULE_FIRST_PERSON in _SHORT_TERM_GOAL_SYSTEM


def test_long_term_goal_prompt_carries_the_rule() -> None:
    """The existing guard only covers time (don't treat what hasn't happened as settled), not
    facts."""
    from agent.need import _LONG_TERM_GOAL_UPDATE_SYSTEM

    assert MARK in _LONG_TERM_GOAL_UPDATE_SYSTEM
    assert CLOSED_WORLD_FACT_RULE_FIRST_PERSON in _LONG_TERM_GOAL_UPDATE_SYSTEM


def test_goal_progress_residue_prompt_carries_the_rule() -> None:
    """Residue is queued just like short-term goals, the second door of the same birthplace; this
    site is a functional third-party ruling."""
    from agent.need import _GOAL_EVALUATION_SYSTEM

    assert MARK in _GOAL_EVALUATION_SYSTEM
    assert CLOSED_WORLD_FACT_RULE in _GOAL_EVALUATION_SYSTEM


def test_decision_prompt_carries_the_rule() -> None:
    """Decisions write action_description / message_content, which get adjudicated, perceived by
    onlookers, and written into memory."""
    from agent.decision import _ACTION_SPACE, DecisionEngine
    from tests.unit.test_decision import _make_packet, _make_personality

    engine = DecisionEngine(None)
    system, _user, _facts = engine._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )
    assert MARK in system
    assert FAMILIARITY_MARK in system


# ---------------------------------------------------------------------------
# Tier 2 — existing sites: don't drop them as a side effect of a refactor
# ---------------------------------------------------------------------------


def test_pre_existing_sites_keep_the_rule() -> None:
    """Memory writes, relation evolution, event generation, the four executors' adjudication — all
    direct entry points into the narrative substrate."""
    import agent.memory
    import agent.relation_evolution
    import engine.event
    import engine.executors.covert
    import engine.executors.physical
    import engine.executors.social
    import engine.executors.work

    for mod in (
        agent.memory,
        agent.relation_evolution,
        engine.event,
        engine.executors.covert,
        engine.executors.physical,
        engine.executors.social,
        engine.executors.work,
    ):
        assert "CLOSED_WORLD_FACT_RULE" in inspect.getsource(mod), mod.__name__


# ---------------------------------------------------------------------------
# Tier 3 — no overreach
# ---------------------------------------------------------------------------


def test_importance_evaluator_stays_out_of_scope() -> None:
    """Pure judgment: the output is a score + a short reason, never entering the narrative
    substrate."""
    import agent.importance_evaluator as mod

    assert "CLOSED_WORLD_FACT_RULE" not in inspect.getsource(mod)


def test_the_pacing_gate_stays_out_of_scope() -> None:
    """The pacing gate only outputs a boolean + a 40-character reason. In the same
    engine/event.py, the event-generation half has the rule (it writes broadcast text) and the gate
    half doesn't — so the assertion must target the gate's own system, not the whole module."""
    from engine.event import EventSystem

    src = inspect.getsource(EventSystem._passes_llm_check)
    assert "CLOSED_WORLD_FACT_RULE" not in src
