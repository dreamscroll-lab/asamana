from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from config import load_config
from core.container import Container
from core.context import clear_log_context
from core.interfaces.llm import LLMRouter, LLMScene
from providers.llm.mock import MockLLMProvider

# Shared mock world-building contract for tests that need a fully built world.
# Single source of truth: schema changes to WorldBuilder/CastDesigner update here.
MOCK_WORLD_BUILDING_RESPONSE = json.dumps({
    "world_name": "武德九年：玄武门",
    "era_description": "唐朝宫廷政治激化为继承之争。",
    "core_tension": "皇位继承秩序与实权秩序难以共存，二者各有所属。",
    "narrative_theme": "权力的代价",
    "narrative_pitch": "继承之争已被双方默认逼到临界，谁先开口或动手尚未决定。",
    "world_time_config": {"era_name": "武德", "start_year": 9},
    "key_figures": [
        {"name": "Li Shimin", "role": "Prince of Qin", "importance": "main", "brief": "Battle-tested prince under threat."},
        {"name": "Li Jiancheng", "role": "Crown Prince", "importance": "main", "brief": "Heir defending his position."},
        {"name": "Emperor Gaozu", "role": "Emperor", "importance": "background", "brief": "Weary sovereign."},
    ],
    "initial_relations": [
        {"from": "Li Shimin", "to": "Li Jiancheng", "trust": 0.15, "affection": -0.55, "label": "rival brothers"},
    ],
    "historical_events": [
        {"event": "Li Shimin built prestige that threatens the succession.", "hours_before_start": 48, "related_figures": ["Li Shimin", "Li Jiancheng"], "importance": "high"},
    ],
    # Names are Chinese, so entity ids go through slugify's sha1 fallback, the same shape as
    # production. ASCII names would hide the id-stability question from tests, and restore relies
    # on stable ids to recognize the same thing.
    "world_entity_seeds": [
        {"name": "传国玉玺", "entity_type": "item", "description": "受命于天的信物", "initial_state": "intact", "location_name": "太极宫"},
        {"name": "玄武门门枢", "entity_type": "landmark", "description": "北门启闭所系", "initial_state": "intact", "location_name": "玄武门"},
    ],
    # NPCs must be present too: a world built with none would leave every NPC-related path
    # (ERRAND admission, perception lists, restore) silently untested.
    "npcs": [
        {"name": "传诏使者", "gender": "男", "age": 35, "description": "脚程快，认得宫中道路", "location_name": "太极宫"},
        {"name": "守门禁军", "gender": "男", "age": 30, "description": "披甲持戟，认得车驾", "location_name": "玄武门"},
    ],
}, ensure_ascii=False)

MOCK_CAST_DESIGN_RESPONSE = json.dumps({
    "roles": [
        {
            "index": 1,
            "narrative_role": "隐忍的挑战者",
            "arc_summary": "从被动防守走向主动出击，以玄武门之变改写自身命运。",
            "key_relationships": [2, 3],
        },
        {
            "index": 2,
            "narrative_role": "摇摇欲坠的继承人",
            "arc_summary": "权位受威胁，试图先发制人却引发覆灭。",
            "key_relationships": [1, 3],
        },
        {
            "index": 3,
            "narrative_role": "两难的父皇",
            "arc_summary": "试图平衡两子争斗，最终无力阻止悲剧收场。",
            "key_relationships": [1, 2],
        },
    ],
}, ensure_ascii=False)


@pytest.fixture()
def test_config() -> object:
    return load_config(Path("config/config.test.yaml"))


# Executor adjudication verdict served to every test that runs an executor.
#
# All four executors adjudicate through one LLM scene, AGENT_ACTION_NARRATION. The mock's
# default "mock response" isn't JSON, so without this every action would be an
# `adjudication_failed=True` null step and tests would never reach the executors' normal path.
#
# One combined payload serves all four; each parser reads only its own keys:
#   physical: reason / success / outcome / fact / relation / actor_damage / target_damage
#             / new_entity_state / deed / condition / condition_steps / frees_target
#             (the three condition keys are left out on purpose: absent means no ongoing
#             condition, which is what most physical actions leave. Condition tests supply their own.)
#   covert:   reason / detected / outcome / fact / achieved / why
#             (achieved = what the attempt uncovered, not whether it finished. The `fact` below
#             reveals nothing: don't use it as an intel sample; intel tests supply their own.)
#   work:     success / fact / outcome (3p) / why
#   social:   dialogue[{speaker:1|2, line}] + observation (onlooker view of the whole scene)
#             + fact / success / relation / why (per-participant self-assessment)
#
# The default verdict is the most conservative: success, no damage, neutral relation,
# undetected. Cases needing another verdict (failure / detected / SEIZE …) override `fixed_response`.
MOCK_ADJUDICATION_RESPONSE = json.dumps({
    "reason": "行动者从容行事，场面无人阻拦。",
    "success": True,
    "achieved": True,
    "detected": False,
    "outcome": "他把这桩事做完了。",
    "fact": "我把这桩事做完了。",
    # why is read only on failure (each executor decides by outcome), so it is empty here.
    "why": "",
    "observation": "两人说了几句便各自散了。",
    "relation": "neutral",
    "actor_damage": 0.0,
    "target_damage": 0.0,
    "new_entity_state": "被动过",
    "deed": "operate",
    "dialogue": [
        {"speaker": 1, "line": "近来可好？"},
        {"speaker": 2, "line": "尚可，你呢？"},
    ],
}, ensure_ascii=False)


@pytest.fixture()
def container(test_config: object) -> Container:
    container = Container.from_config(test_config)
    # The adjudication scene must return valid JSON (see above), or every executor action
    # becomes an adjudication_failed null step.
    container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION).fixed_response = (
        MOCK_ADJUDICATION_RESPONSE
    )
    return container


@pytest.fixture()
def mock_build_container(test_config: object) -> Container:
    """Container whose LLM router returns valid world-building JSON and a parseable decision,
    so agents act during run_step (an unparseable one would make decide() return None).

    The adjudication scene must return valid JSON too, as in `container`; otherwise every
    action is an `adjudication_failed` null step that never reaches the observer stream.
    """

    c = Container.from_config(test_config)
    providers = {scene: c.llm_router.get(scene) for scene in LLMScene}
    providers[LLMScene.WORLD_BUILDING] = MockLLMProvider(fixed_response=MOCK_WORLD_BUILDING_RESPONSE)
    providers[LLMScene.CAST_DESIGN] = MockLLMProvider(fixed_response=MOCK_CAST_DESIGN_RESPONSE)
    providers[LLMScene.PERSONA_GENERATION] = MockLLMProvider(fixed_response="{}")
    providers[LLMScene.AGENT_ACTION_NARRATION] = MockLLMProvider(
        fixed_response=MOCK_ADJUDICATION_RESPONSE
    )
    providers[LLMScene.AGENT_DECISION_MAIN] = MockLLMProvider(
        fixed_response='{"selected_index": 3, "action_description": "处理政务", "estimated_steps": 1}'
    )
    c.llm_router = LLMRouter(providers)
    return c


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    clear_log_context()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
