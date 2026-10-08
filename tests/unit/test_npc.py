"""Npc as a world citizen: it stands somewhere, holds things, and survives a restore.

The middle of the three tiers: a body without cognition. This file covers only its properties as
a thing in the world; how errands run belongs to ``test_npc_errand``, and the guard that it is
not an agent belongs to ``test_npc_is_not_an_agent``.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from datetime import datetime

from core.interfaces.action import ErrandOrder
from engine.npc_runner import NpcRunner
from core.interfaces.condition import BodyCondition
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from agent.personality import SoulLayer
from engine.presence import attach_presence
from world.models import ActiveErrand, EntityPresence, NpcSeed, WorldEntity, WorldEntityType
from core.interfaces.place import Place


def _location(entity_id: str, name: str, **connections: int) -> Place:
    return Place(
        place_id=entity_id, name=name, connections=dict(connections),
    )


def _world(*, max_npcs: int = 6) -> EnvironmentSystem:
    env = EnvironmentSystem(max_npcs=max_npcs)
    env.space.register_place(_location("hall", "大殿", gate=1))
    env.space.register_place(_location("gate", "宫门", hall=1))
    return env


def _seed(name: str, **kw) -> NpcSeed:
    return NpcSeed(name=name, gender=kw.get("gender", "男"), age=kw.get("age", 34),
                   description=kw.get("description", "跑得快"))


def test_a_body_without_a_mind_is_present_but_not_among_the_people() -> None:
    """It is present but not on the list of people with cognition; the split is a type-level gate,
    not caller discipline."""
    env = _world()
    assert env.spawn_npc(_seed("王二"), location_id="hall")
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    spatial = env.spatial_for(agent_id="a1")
    assert spatial.visible_agent_ids == ["a2"]
    assert len(spatial.visible_npc_ids) == 1
    # The two lists don't overlap: mix one into the other and TALK/PHYSICAL presence checks admit an
    # id with no Agent.
    assert not set(spatial.visible_agent_ids) & set(spatial.visible_npc_ids)


def test_what_it_carries_is_here_like_anything_else_here() -> None:
    """What it holds is in this room just like what's on the floor; holding changes affordance, not
    location."""
    env = _world()
    env.spawn_npc(_seed("王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    env.register_entity(WorldEntity(
        entity_id="letter", name="密信", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD, presence_ref=npc_id, is_takeable=True,
    ))
    env.place_agent(agent_id="a1", location_id="hall")

    assert "密信" in [e.name for e in env.spatial_for(agent_id="a1").visible_entities]
    # When it leaves, the item leaves with it and is out of reach for whoever stays behind.
    env.move_body(body_id=npc_id, location_id="gate")
    assert "密信" not in [e.name for e in env.spatial_for(agent_id="a1").visible_entities]


def test_the_quota_stops_the_world_from_filling_up_with_extras() -> None:
    """The cap is enforced at the single creation point. The one over the cap never lands and the
    table doesn't grow."""
    env = _world(max_npcs=2)
    assert env.spawn_npc(_seed("甲"), location_id="hall")
    assert env.spawn_npc(_seed("乙"), location_id="hall")
    assert not env.spawn_npc(_seed("丙"), location_id="hall")
    assert len(env.all_npcs()) == 2


def test_a_body_needs_real_ground_to_stand_on() -> None:
    """Don't create one that can't land in a real place: a person nobody can reach who never goes
    away is worse than none."""
    env = _world()
    assert not env.spawn_npc(_seed("王二"), location_id="nowhere")
    assert env.all_npcs() == []


def test_identity_comes_from_the_directory_and_the_rest_from_the_body() -> None:
    """Identity (immutable) goes through the directory; busy state and condition (change every step)
    through the live object. The facade must not carry mutable state."""
    env = _world()
    env.spawn_npc(_seed("王二", description="认得路"), location_id="hall")
    npc = env.all_npcs()[0]
    npc.condition = BodyCondition(description="双手被反绑", since_step=2)
    npc.errand = ActiveErrand(
        order=ErrandOrder(npc.npc_id, "gate", recipient_id="a1", message="话"),
        requester_id="a1", origin_id="hall", assigned_step=2,
    )
    env.place_agent(agent_id="a1", location_id="hall")

    directory = LiveWorldDirectory.from_agents({}, env)
    # The directory returns only immutable identity; busy state and condition are not in its return
    # value.
    identity = directory.npc_identity_map([npc.npc_id])[npc.npc_id]
    assert (identity.name, identity.gender, identity.age) == ("王二", "男", 34)
    assert identity.description == "认得路"

    spatial = env.spatial_for(agent_id="a1")
    attach_presence(spatial, directory=directory, agents={}, environment=env)
    seen = spatial.visible_npcs[npc.npc_id]
    assert seen.identity.name == "王二"
    assert seen.busy is True
    assert seen.condition == "双手被反绑"


def test_a_body_and_its_errand_survive_a_restore_whole() -> None:
    """Location, condition and the errand in hand must all be restored.

    Location especially: the agent half of ``body_locations`` is write-only (agent_store is the
    authority), but an Npc has no such record. Skip it and after restore the NPC stands in the
    wrong place, or isn't in the world at all.
    """
    env = _world()
    env.spawn_npc(_seed("王二"), location_id="hall")
    npc = env.all_npcs()[0]
    env.move_body(body_id=npc.npc_id, location_id="gate")
    npc.condition = BodyCondition(description="腿上带伤", since_step=5)
    npc.errand = ActiveErrand(
        order=ErrandOrder(npc.npc_id, "hall", recipient_id="a1", message="殿下召见"),
        requester_id="a1", origin_id="hall", outbound=False, seen="那里空无一人",
        done=("话已带到。",), assigned_step=41,
    )

    payload = env.snapshot_state()
    json.dumps(payload, ensure_ascii=False)      # payloads leaving the sim must be JSON-native

    fresh = _world()
    fresh.restore_state(payload)
    back = fresh.get_npc(npc.npc_id)
    assert back is not None
    assert fresh.get_body_location(npc.npc_id) == "gate"
    assert back.condition is not None and back.condition.description == "腿上带伤"
    assert back.errand is not None
    assert back.errand.outbound is False
    # The step the errand was assigned must come back too: lose it and he sets off on the restored
    # step, wasting the assignment step.
    assert back.errand.assigned_step == 41
    assert back.errand.seen == "那里空无一人"
    assert back.errand.done == ("话已带到。",)
    assert (back.errand.order.recipient_id, back.errand.order.message) == ("a1", "殿下召见")
    assert fresh.snapshot_state()["npc_states"] == payload["npc_states"]


def test_a_half_written_errand_is_dropped_not_thrown(caplog) -> None:
    """An errand payload missing its destination voids only that errand, not the whole restore.

    ``destination_id`` has no default and the other three axes hang off it (see the eight-cell
    space of ErrandOrder): no destination, no errand. Raising inside restore_state would fail the
    whole world's restore.
    """
    env = _world()
    payload = {
        "npc_states": {
            "npc_x": {
                "name": "王二", "gender": "男", "age": 34,
                # Only the NPC, no destination: a hand-written fixture / a truncated write / an old
                # save predating the field
                "errand": {"order": {"npc_id": "npc_x"}, "requester_id": "a1"},
            }
        },
        "body_locations": {"npc_x": "gate"},
    }

    env.restore_state(payload)          # passes if it doesn't raise

    back = env.get_npc("npc_x")
    assert back is not None             # the person is restored
    assert back.errand is None          # only the errand is gone
    assert env.get_body_location("npc_x") == "gate"


def test_restore_does_not_re_apply_the_quota() -> None:
    """Restore isn't capped: a world legitimately at the limit must not lose people on restore."""
    env = _world(max_npcs=3)
    for name in ("甲", "乙", "丙"):
        env.spawn_npc(_seed(name), location_id="hall")

    fresh = _world(max_npcs=1)
    fresh.restore_state(env.snapshot_state())
    assert len(fresh.all_npcs()) == 3


def test_an_errand_lands_on_a_free_body_and_only_one() -> None:
    """Two people send the same NPC on the same step: the first gets it, the second is honestly
    foiled; never silently replace the first."""
    env = _world()
    env.spawn_npc(_seed("王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    first = ErrandOrder(npc_id, "gate", ())
    second = ErrandOrder(npc_id, "hall", ())
    assert env.assign_errand(first, requester_id="a1")
    assert not env.assign_errand(second, requester_id="a2")
    assert env.get_npc(npc_id).errand.requester_id == "a1"


def test_a_held_body_takes_on_nothing_and_can_be_cut_short() -> None:
    """A restrained NPC can't take new errands; one in progress can be stopped where it stands."""
    from core.interfaces.action import NpcEffect

    env = _world()
    env.spawn_npc(_seed("王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    env.place_agent(agent_id="a1", location_id="hall")
    env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")

    env.apply_npc_effect(NpcEffect(
        npc_id=npc_id,
        condition_set=BodyCondition(description="被按在地上", since_step=3),
    ))
    npc = env.get_npc(npc_id)
    assert npc.condition is not None
    # The errand stays: a restrained NPC can't move, but still owes this errand. Voiding it
    # would turn an interception into a permanent cancel, and the sender would never hear back.
    assert npc.errand is not None
    # Whoever stopped it wants it to stay here, so it isn't sent back.
    assert env.get_body_location(npc_id) == "hall"
    assert not env.assign_errand(ErrandOrder(npc_id, "gate", ()), requester_id="a1")

    # Clearing: a second flat field shaped like the agent one, not "pass None".
    env.apply_npc_effect(NpcEffect(npc_id=npc_id, condition_cleared=True))
    assert env.get_npc(npc_id).condition is None
    assert env.get_npc(npc_id).errand is not None, "解开之后他接着走原来那趟"


def _runner(env: EnvironmentSystem, *, souls: dict | None = None) -> NpcRunner:
    """A runner over this world — used here only to expire conditions and compose sentences; walking
    belongs to test_npc_errand.py."""
    from engine.message_system import MessageSystem
    from providers.message.in_memory import InMemoryMessageProvider

    return NpcRunner(
        environment=env,
        message_system=MessageSystem(InMemoryMessageProvider(), world_id="w"),
        directory=LiveWorldDirectory(souls=souls or {}, environment=env),
        world_id="w",
        # One second per step: the maps here write edge walking time in steps.
        seconds_per_step=1,
    )


def test_a_condition_never_quietly_cancels_what_it_was_sent_to_do() -> None:
    """``NpcEffect`` has no "abort errand" field, deliberately; guard against it being added.

    Whether to stop is decided by condition alone (``NpcRunner`` checks it every step); there is no
    second criterion. Two criteria eventually disagree, and silently: the main character's memory
    is left holding something that will never resolve.
    """
    import dataclasses

    from core.interfaces.action import NpcEffect

    names = {f.name for f in dataclasses.fields(NpcEffect)}
    assert names == {"npc_id", "condition_set", "condition_cleared"}, names


def test_each_condition_axis_has_exactly_one_writer() -> None:
    """Condition has two axes, each with one owner; nowhere else may assign it directly.

      agents → ``agent/personality.py`` (their own state, changed by themselves)
      NPCs   → ``engine/environment.py`` (owner of ``_npcs``: apply, clear and expiry all live there)

    A direct write elsewhere opens a second writer on the same slot, and two writers eventually
    disagree. This is the same rule ``clear_errand`` states for the errand slot.
    """
    import ast
    import subprocess

    repo = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    sources = subprocess.run(
        ["git", "ls-files", "agent/*.py", "core/*.py", "engine/*.py", "world/*.py",
         "worlds/*.py", "interaction/*.py", "providers/*.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.split()
    sources = [rel for rel in sources if (pathlib.Path(repo) / rel).exists()]

    offenders: list[str] = []
    for rel in sources:
        if rel in ("engine/environment.py", "agent/personality.py"):
            continue                       # each axis's owner
        for node in ast.walk(ast.parse((pathlib.Path(repo) / rel).read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "condition":
                    offenders.append(f"{rel}: {ast.unparse(node)}")
    assert offenders == [], offenders


def test_a_self_limiting_condition_wears_off_on_its_own() -> None:
    """``until_step`` isn't write-only: a single "tripped him up" must not become indefinite.

    NPCs aren't in the runtime's ``agents`` list, so ``NpcRunner`` picks the moment each step,
    while the change itself belongs to ``EnvironmentSystem.expire_npc_condition``.
    """
    from core.interfaces.action import NpcEffect

    env = _world()
    env.spawn_npc(_seed("王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    runner = _runner(env)

    env.apply_npc_effect(NpcEffect(
        npc_id=npc_id,
        condition_set=BodyCondition(description="被绊了一下", since_step=3, until_step=5),
    ))
    asyncio.run(runner.advance(4))
    assert env.get_npc(npc_id).condition is not None, "没到期不许自己好"
    asyncio.run(runner.advance(5))
    assert env.get_npc(npc_id).condition is None


def test_a_condition_that_needs_untying_never_unties_itself() -> None:
    """``until_step is None`` = must be cleared by someone. Ropes don't untie themselves; same as
    the agent side."""
    from core.interfaces.action import NpcEffect

    env = _world()
    env.spawn_npc(_seed("王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    runner = _runner(env)

    env.apply_npc_effect(NpcEffect(
        npc_id=npc_id,
        condition_set=BodyCondition(description="被绳索捆住", since_step=3, until_step=None),
    ))
    for step in range(4, 40):
        asyncio.run(runner.advance(step))
    assert env.get_npc(npc_id).condition is not None


def test_a_body_is_named_and_described_but_never_by_its_id() -> None:
    """List rendering: name, gender, age in one bracket, blurb after; unknown NPCs fall back to a
    descriptive referent."""
    from core.interfaces.perception import NpcIdentity, PerceivedNpc
    from core.prompts import render_npc

    line = render_npc(
        PerceivedNpc(identity=NpcIdentity(name="王二", gender="男", age=34, description="跑得快")),
        index=1,
    )
    assert line == "#1 王二（男，34岁）：跑得快"
    # No numbering at the adjudication scene: there's no index back-reference contract there, and a
    # #N with nowhere to land invites made-up indices.
    assert render_npc(PerceivedNpc(identity=NpcIdentity(name="王二", gender="男"))).startswith("王二")
    assert render_npc(PerceivedNpc()) == "某人"


def test_what_he_is_up_to_reads_off_which_axes_were_filled() -> None:
    """The purpose is read from the order, not a table of action verbs; same source as the eight
    cells of ``NpcRunner._do_errand``, which describes the result once done. When adding an axis,
    change both, or this line says he's delivering a letter while he did something else.

    Only the kind of task is stated: the message belongs to the step he actually says it, not to the
    purpose repeated on every travel step. Items are named only.
    """
    env = _world()
    env.register_entity(WorldEntity(
        entity_id="e1", name="书信", entity_type=WorldEntityType.ITEM,
        presence_ref="hall", is_takeable=True,
    ))
    runner = _runner(env, souls={"a2": SoulLayer(name="阿石", gender="男", age=27)})

    def intent(item: str = "", to: str = "", message: str = "") -> str:
        return runner._intent(ErrandOrder("npc_x", "gate", item_id=item, recipient_id=to, message=message))

    assert intent(item="e1", to="a2") == "把书信交给阿石"
    assert intent(item="e1") == "把书信送去"
    assert intent(to="a2", message="快去") == "给阿石带句话"
    assert intent(message="快去") == "当众传句话"
    assert intent(to="a2") == "看看阿石在不在"
    assert intent() == "来这里看看"
    # Item and message share one addressing axis, so the recipient is named once; twice reads like
    # two people.
    assert intent(item="e1", to="a2", message="快去") == "把书信交给阿石，再带句话"
    # Unknown ids always fall back to a descriptive referent, never a bare id: this sentence is for
    # people to read.
    assert intent(item="zzz", to="qqq") == "把某物交给某人"


def test_the_observer_gets_the_sentence_the_runner_wrote_and_the_place_he_stands() -> None:
    """The read model only carries: sentences are copied as-is, location is joined from the
    body-location table; it composes nothing.

    Location isn't written into the sentence too, which would give two place names; the same-place
    invariant makes one enough.
    """
    from core.interfaces.snapshot import WorldSnapshot
    from interaction.models import _npcs_from_snapshot

    snap = WorldSnapshot(
        world_id="w", step=7, timestamp=datetime(2026, 1, 1), metadata={"environment": {
            "npc_states": {"npc_x": {"name": "王二", "gender": "男", "age": 34}},
            "body_locations": {"npc_x": "gate"},
            "location_names": {"gate": "宫门"},
            "npc_outcomes": {"npc_x": {"text": "把书信交给了阿石", "ongoing": False}},
        }},
    )
    summary = _npcs_from_snapshot(snap)[0]
    assert (summary.location, summary.outcome, summary.ongoing) == (
        "宫门", "把书信交给了阿石", False,
    )
    # Nothing happened this step: an empty sentence, while identity and location are still reported.
    snap.metadata["environment"]["npc_outcomes"] = {}
    quiet = _npcs_from_snapshot(snap)[0]
    assert (quiet.location, quiet.outcome, quiet.ongoing) == ("宫门", "", False)


def test_the_observer_is_told_which_npcs_were_displaced_this_step() -> None:
    from core.interfaces.snapshot import WorldSnapshot
    from interaction.models import _npcs_from_snapshot

    snap = WorldSnapshot(
        world_id="w", step=7, timestamp=datetime(2026, 1, 1), metadata={"environment": {
            "npc_states": {"npc_x": {"name": "王二"}, "npc_y": {"name": "阿石"}},
            "body_locations": {"npc_x": "gate", "npc_y": "gate"},
            "npc_displaced": ["npc_x"],
        }},
    )
    flags = {n.npc_id: n.displaced for n in _npcs_from_snapshot(snap)}
    assert flags == {"npc_x": True, "npc_y": False}
