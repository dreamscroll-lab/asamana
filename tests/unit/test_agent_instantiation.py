"""WorldInitializer._instantiate_agent contract: cognition is unconditionally LLM,
narrative tier is still carried by is_main_character.

There is NO agent-level "rule cognition" mode: every agent — background included — is
wired with the full LLM cognition stack (reflection, relation evolution, LLM-backed
memory). ``is_main_character`` carries the *narrative* tier only: scheduling priority,
perception fidelity, memory write threshold, model tier / retry.

The ``tiered_cognition`` / ``use_llm_cognition`` assertions below are tripwires against
that switch growing back (CLAUDE.md §5).
"""

from __future__ import annotations

import pytest

from engine.clock import WorldTimeConfig
from world.initializer import WorldInitializer
from world.models import AgentDefinition, AgentTier, EmotionState, SoulLayer


def _definition(agent_id: str, tier: AgentTier) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        name=agent_id.upper(),
        tier=tier,
        soul=SoulLayer(name=agent_id.upper(), role="", agent_id=agent_id),
        initial_location="taiji_palace",
        initial_emotion=EmotionState(),
    )


@pytest.mark.parametrize(
    ("tier", "expect_main"),
    [(AgentTier.BACKGROUND, False), (AgentTier.MAIN, True)],
)
def test_every_agent_gets_the_full_llm_cognition_stack(
    container, tier: AgentTier, expect_main: bool
) -> None:
    """Both tiers are wired with reflection + relation evolution + memory: cognition mode is not
    a per-agent property, only the narrative tier is (CLAUDE.md §5)."""
    initializer = WorldInitializer(container)
    agent = initializer._instantiate_agent(  # noqa: SLF001
        world_id="w", definition=_definition("a", tier)
    )

    # Narrative tier is preserved...
    assert agent.is_main_character is expect_main
    # ...and it does NOT gate cognition: the LLM subsystems exist for both tiers.
    assert agent.reflection_engine is not None
    assert agent.relation_evolution is not None
    assert agent.memory_system is not None


def test_cognition_mode_is_not_a_field_on_agent(container) -> None:
    """No agent-level cognition-mode flag exists.

    A cheap guard: a reintroduced per-agent LLM-vs-rule gate would most likely be spelled one of
    these two ways, and this fails loudly.
    """
    initializer = WorldInitializer(container)
    agent = initializer._instantiate_agent(  # noqa: SLF001
        world_id="w", definition=_definition("bg", AgentTier.BACKGROUND)
    )

    assert not hasattr(agent, "use_llm_cognition")
    assert not hasattr(container, "tiered_cognition")


def test_memory_write_threshold_still_follows_narrative_tier(container) -> None:
    """is_main_character keeps this job: it is handed to MemorySystem, which
    uses it for the experiential write threshold (a narrative-fidelity knob, not a
    cognition-mode gate)."""
    initializer = WorldInitializer(container)
    main = initializer._instantiate_agent(  # noqa: SLF001
        world_id="w", definition=_definition("main", AgentTier.MAIN)
    )
    bg = initializer._instantiate_agent(  # noqa: SLF001
        world_id="w", definition=_definition("bg", AgentTier.BACKGROUND)
    )

    assert main.memory_system._is_main_character is True  # noqa: SLF001
    assert bg.memory_system._is_main_character is False  # noqa: SLF001


def test_every_memory_renderer_gets_the_world_calendar(container) -> None:
    """All four cognition subsystems that render memories must get the world's starting clock
    time, or "today / yesterday" falls back to dividing durations.

    The default 0 means "the world starts at midnight" — no error, it just silently names the
    wrong day, so missing one wiring site makes no noise at all.
    """
    initializer = WorldInitializer(container)
    agent = initializer._instantiate_agent(  # noqa: SLF001
        world_id="w",
        definition=_definition("a", AgentTier.MAIN),
        clock_config=WorldTimeConfig(start_hour=4, seconds_per_step=10800),
    )

    # The value comes from the real clock (starts at 4 o'clock), not the default 0.
    owners = {
        "agent": agent.world_start_second_of_day,
        "decision": agent.decision_engine._world_start_second_of_day,  # noqa: SLF001
        "memory": agent.memory_system._world_start_second_of_day,  # noqa: SLF001
        "reflection": agent.reflection_engine._world_start_second_of_day,  # noqa: SLF001
    }
    assert [k for k, v in owners.items() if v != 4 * 3600] == []
