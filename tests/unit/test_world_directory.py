"""Unit tests for the per-world read-only identity directory."""

from __future__ import annotations

from agent.personality import SoulLayer
from core.interfaces.perception import PerceivedIdentity
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from world.models import WorldEntity, WorldEntityType
from core.interfaces.place import Place


def _make_location(entity_id: str, name: str, description: str = "") -> Place:
    return Place(
        place_id=entity_id,
        name=name,
        description=description,
        is_public=True,
    )


def _make_item(entity_id: str, name: str, description: str = "", is_takeable: bool = True) -> WorldEntity:
    return WorldEntity(
        entity_id=entity_id,
        name=name,
        entity_type=WorldEntityType.ITEM,
        description=description,
        is_takeable=is_takeable,
        is_public=True,
    )


def _make_directory(
    souls: dict[str, SoulLayer] | None = None,
    environment: EnvironmentSystem | None = None,
) -> tuple[LiveWorldDirectory, EnvironmentSystem]:
    env = environment or EnvironmentSystem()
    return LiveWorldDirectory(souls=souls or {}, environment=env), env


# ---------------------------------------------------------------------------
# agent_name / agent_identity_map / all_agent_names
# ---------------------------------------------------------------------------


def test_agent_name_resolves_known_soul() -> None:
    directory, _ = _make_directory(
        souls={"li_shimin": SoulLayer(name="李世民", role="秦王", agent_id="li_shimin")}
    )
    assert directory.agent_name("li_shimin") == "李世民"


def test_agent_name_falls_back_to_descriptive_for_unknown() -> None:
    directory, _ = _make_directory()
    # A miss on the name axis falls back to a descriptive reference, never a raw id (else the id leaks into narrative text).
    assert directory.agent_name("ghost") == "某人"


def test_agent_name_falls_back_to_descriptive_for_empty_soul_name() -> None:
    directory, _ = _make_directory(souls={"a1": SoulLayer(name="", agent_id="a1")})
    assert directory.agent_name("a1") == "某人"


def test_agent_identity_map_omits_unknown_ids() -> None:
    directory, _ = _make_directory(
        souls={"a1": SoulLayer(name="李世民", agent_id="a1")}
    )
    assert directory.agent_identity_map(["a1", "ghost"]) == {
        "a1": PerceivedIdentity(name="李世民")
    }


def test_agent_identity_map_empty_input() -> None:
    directory, _ = _make_directory(
        souls={"a1": SoulLayer(name="李世民", agent_id="a1")}
    )
    assert directory.agent_identity_map([]) == {}


def test_agent_identity_map_carries_gender() -> None:
    """Gender travels the perception channel along with the name: runtime fills SpatialPerception.visible_agents from this map."""
    directory, _ = _make_directory(
        souls={"a1": SoulLayer(name="长孙无垢", agent_id="a1", gender="女")}
    )
    assert directory.agent_identity_map(["a1"])["a1"].gender == "女"


def test_all_agent_names_returns_full_map() -> None:
    directory, _ = _make_directory(
        souls={
            "a1": SoulLayer(name="李世民", agent_id="a1"),
            "a2": SoulLayer(name="尉迟恭", agent_id="a2"),
        }
    )
    assert directory.all_agent_names() == {"a1": "李世民", "a2": "尉迟恭"}


def test_from_agents_builds_souls_from_personality() -> None:
    class _Personality:
        def __init__(self, soul: SoulLayer) -> None:
            self.soul = soul

    class _Agent:
        def __init__(self, soul: SoulLayer) -> None:
            self.personality = _Personality(soul)

    env = EnvironmentSystem()
    agents = {"a1": _Agent(SoulLayer(name="李世民", agent_id="a1"))}
    directory = LiveWorldDirectory.from_agents(agents, env)
    assert directory.agent_name("a1") == "李世民"


# ---------------------------------------------------------------------------
# location_name / entity_name
# ---------------------------------------------------------------------------


def test_location_name_delegates_to_environment() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("taiji_hall", "太极殿"))
    directory, _ = _make_directory(environment=env)
    assert directory.location_name("taiji_hall") == "太极殿"
    # On miss, fall back to a descriptive reference, never the raw id (display text
    # must not leak code-layer ids). Single resolver (narrative_location_name) → "此处".
    assert directory.location_name("nowhere") == "此处"


def test_entity_name_resolves_item_and_falls_back() -> None:
    env = EnvironmentSystem()
    env.register_entity(_make_item("seed_sword", "佩剑"))
    directory, _ = _make_directory(environment=env)
    assert directory.entity_name("seed_sword") == "佩剑"
    assert directory.entity_name("no_such_item") == "某物"


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_agent_entry() -> None:
    directory, _ = _make_directory(
        souls={"a1": SoulLayer(name="李世民", role="秦王", agent_id="a1")}
    )
    entry = directory.describe("a1")
    assert entry is not None
    assert (entry.kind, entry.name, entry.role) == ("agent", "李世民", "秦王")


def test_describe_location_entry() -> None:
    env = EnvironmentSystem()
    env.space.register_place(_make_location("taiji_hall", "太极殿", description="皇宫正殿"))
    directory, _ = _make_directory(environment=env)
    entry = directory.describe("taiji_hall")
    assert entry is not None
    assert (entry.kind, entry.name, entry.description) == ("location", "太极殿", "皇宫正殿")


def test_describe_item_entry() -> None:
    env = EnvironmentSystem()
    env.register_entity(_make_item("seed_sword", "佩剑", description="一柄旧剑", is_takeable=True))
    directory, _ = _make_directory(environment=env)
    entry = directory.describe("seed_sword")
    assert entry is not None
    assert (entry.kind, entry.name, entry.is_takeable) == ("item", "佩剑", True)


def test_describe_unknown_returns_none() -> None:
    directory, _ = _make_directory()
    assert directory.describe("nothing") is None


def test_describe_prefers_agent_over_entity_on_id_collision() -> None:
    env = EnvironmentSystem()
    env.register_entity(_make_item("dual", "同名物品"))
    directory, _ = _make_directory(
        souls={"dual": SoulLayer(name="同名角色", agent_id="dual")},
        environment=env,
    )
    entry = directory.describe("dual")
    assert entry is not None
    assert (entry.kind, entry.name) == ("agent", "同名角色")


# ---------------------------------------------------------------------------
# liveness: directory sees entities registered/restored after construction
# ---------------------------------------------------------------------------


def test_entity_registered_after_construction_is_visible() -> None:
    directory, env = _make_directory()
    assert directory.describe("late_item") is None
    env.register_entity(_make_item("late_item", "迟到的物品"))
    assert directory.entity_name("late_item") == "迟到的物品"
    entry = directory.describe("late_item")
    assert entry is not None and entry.kind == "item"


def test_entity_restored_after_construction_is_visible() -> None:
    source_env = EnvironmentSystem()
    source_env.register_entity(_make_item("seed_sword", "佩剑"))
    state = source_env.snapshot_state()

    directory, fresh_env = _make_directory()
    assert directory.describe("seed_sword") is None
    fresh_env.restore_state(state)
    assert directory.entity_name("seed_sword") == "佩剑"
