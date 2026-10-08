"""Structural guard: a body without cognition must never be treated as an agent.

This file tests boundaries, not features. An Npc never asks a cognitive question, so it doesn't
split the world into two causal laws (CLAUDE.md §5); any gap that lets it into the agent path is
the entry point for a second physics.

Each assertion stands for a concrete silent failure: arbitration rejecting a whole action as "no
response", adjudication writing emotion against an id with no ``Agent``, the reachable list
accepting a sender who can't reply.
"""

from __future__ import annotations

import pathlib

import pytest

from core.interfaces.action import ActionTarget, KIND_NPC, Ref
from core.interfaces.message import Message
from core.interfaces.perception import (
    LocationView, NpcIdentity, PerceivedIdentity, PerceivedNpc, SpatialPerception,
)
from engine.environment import EnvironmentSystem
from world.models import Npc, NpcSeed
from core.interfaces.place import Place


def _action_space():
    from agent.decision import _ACTION_SPACE
    return _ACTION_SPACE


def _env() -> EnvironmentSystem:
    env = EnvironmentSystem()
    env.space.register_place(Place(
        place_id="hall", name="大殿", ))
    return env


def test_a_body_without_a_mind_carries_no_cognition_at_all() -> None:
    """Zero cognition, not cut-down cognition: this class may not have vitality / emotion / need /
    relation / memory.

    This guards the boundary itself: any field that makes its behavior depend on its own past turns
    it into a cheap agent.
    """
    forbidden = {
        "vitality", "emotion", "needs", "relations", "memory", "memory_system",
        "personality", "decision_engine", "long_term_goals", "short_term_goals",
    }
    assert forbidden.isdisjoint(Npc.__dataclass_fields__)


def test_it_never_enters_the_roster_the_whole_cognition_loop_runs_on() -> None:
    """``World.agents`` is the system's only agent roster, the entry point for scheduling,
    arbitration and death handling.

    Npcs live in the environment's own table, with no overlap with that roster, so those three paths
    never see them and need no filtering code at all.
    """
    env = _env()
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id

    agents: dict[str, object] = {"a1": object()}
    assert npc_id not in agents
    # The location table is shared (a body is a body), but "who is an Npc" has its own lookup table.
    assert npc_id in env._locations                       # noqa: SLF001
    assert env.get_npc("a1") is None


def test_the_present_roster_keeps_the_two_kinds_apart() -> None:
    """Every id in ``visible_agents`` must resolve to an ``Agent``.

    All its existing consumers (TALK / PHYSICAL-on-person presence checks, passive-join,
    arbitration) assume so. Mix in one that doesn't resolve and those paths only discover it at
    adjudication, where it becomes a silent null step.
    """
    env = _env()
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    env.place_agent(agent_id="a1", location_id="hall")
    env.place_agent(agent_id="a2", location_id="hall")

    spatial = env.spatial_for(agent_id="a1")
    assert all(env.get_npc(aid) is None for aid in spatial.visible_agent_ids)
    assert all(env.get_npc(nid) is not None for nid in spatial.visible_npc_ids)


def test_binding_never_puts_one_where_arbitration_would_look_for_an_agent() -> None:
    """It must never enter ``claims``: arbitration looks up claim ids in the agents table, and a
    miss is judged "no response", rejecting the whole action. It has no turn to spend, so it goes
    only through acts_on."""
    from agent.decision import _bind_errand, _Slots

    slots = _Slots(
        payload={"errand_npc_index": 1, "errand_destination_index": 1},
        packet=_packet(),
        visible_ids=[], roster_ids=[], reachable_ids=["gate"],
        entity_ids=[], entity_types={}, npc_ids=["npc_x"], own_item_ids=[],
    )
    bound = _bind_errand(slots)
    assert bound is not None
    assert bound.target.claims == []
    assert bound.target.reaches == []
    assert bound.target.acted_on_kind == KIND_NPC


def test_the_thing_it_carries_is_numbered_off_the_list_that_was_printed() -> None:
    """Indices may only count against the one list printed in the prompt.

    "What I hold" is a subset of it; used as a second numbering, the model's #2 would resolve to "my
    second held item" and neither side would error. So holding is validation, not a namespace:
    choosing an item not in hand rejects the whole step for a re-decision, never silently dropping
    it from the errand.
    """
    from agent.decision import _bind_errand, _Slots

    def _bind(index: int):
        return _bind_errand(_Slots(
            payload={
                "errand_npc_index": 1, "errand_destination_index": 1,
                "errand_item_index": index, "errand_recipient_index": 1,
            },
            packet=_packet(),
            visible_ids=[], roster_ids=["a2"], reachable_ids=["gate"],
            # Three items in the printed list; only the second is in my hand.
            entity_ids=["floor_lamp", "my_letter", "his_seal"],
            entity_types={}, npc_ids=["npc_x"], own_item_ids=["my_letter"],
        ))

    bound = _bind(2)                      # the second printed item = the letter in my hand
    assert bound is not None
    assert bound.errand.item_id == "my_letter"

    # Choosing an item not in hand → reject the whole step; don't silently degrade to an errand
    # carrying nothing.
    assert _bind(1) is None
    assert _bind(3) is None


def test_a_kinded_ref_does_not_answer_to_the_people_helpers() -> None:
    """``agents_of`` and ``acted_on_agents`` filter by kind; an npc ref must not mix in with
    people."""
    target = ActionTarget(acts_on=[Ref.npc("npc_x")])
    assert target.acted_on_agents == []
    assert target.single_acted_on_agent is None
    assert Ref.npc("npc_x").is_agent is False


def test_a_report_from_one_is_not_a_correspondent() -> None:
    """Its report-back is also a Message, and it can't reply, talk or act.

    Accepting it only yields a letter that can't be delivered. The criterion is the message's own
    ``sender_is_agent`` (Message's contract: the sender declares what it is), not each consumer
    re-deciding.
    """
    from agent.decision import _message_roster

    packet = _packet(
        visible_npcs={"npc_x": PerceivedNpc(identity=NpcIdentity(name="王二"))},
        inbox=[_message("npc_x", "王二", is_agent=False), _message("a3", "丙")],
    )
    roster = [aid for aid, _who in _message_roster(packet)]
    assert "npc_x" not in roster        # the one reporting back doesn't enter the reachable list
    assert "a3" in roster               # other senders still do


def test_the_read_model_does_not_call_one_an_agent() -> None:
    """The read model must not lie: it's not in ``agent_states``, so the relation graph and
    character cards never include it."""
    from core.interfaces.snapshot import WorldSnapshot
    from interaction.models import StepEvent
    from world.identity_color import _PALETTE, MINDLESS_BODY_COLOR

    env = _env()
    env.spawn_npc(
        NpcSeed(name="王二", gender="男", age=34, description="跑得快"), location_id="hall",
    )
    npc_id = env.all_npcs()[0].npc_id
    snapshot = WorldSnapshot(
        world_id="w", step=1, timestamp=None, world_time={},
        agent_states={}, metadata={"environment": env.snapshot_state()},
    )
    event = StepEvent.from_snapshot(snapshot)

    assert npc_id not in event.agent_states
    assert [n.npc_id for n in event.npcs] == [npc_id]
    seen = event.npcs[0]
    assert (seen.name, seen.gender, seen.age) == ("王二", "男", 34)
    assert seen.location == "大殿"                 # narrative name, not an id
    # This tier has no identity, so it gets the reserved value, not a slot from the character
    # palette.
    assert seen.color == MINDLESS_BODY_COLOR
    assert seen.color not in _PALETTE


def test_the_code_layer_tier_never_crosses_into_the_narrative_layer() -> None:
    """"工具人" / "NPC" name a code-layer tier; the narrative layer has no such thing, the same
    membrane as ids and steps.

    In a prompt these words shape ``action_description`` / ``inner_monologue`` / ``outcome``, which
    are written to memory, embedded and recalled for good. The distinction is stated in in-world
    terms instead: he does what he's told, doesn't decide for himself, doesn't refuse.
    """
    from engine.directory import LiveWorldDirectory
    from engine.scene import SceneVisibility, assemble_scene_context
    from agent.decision import DecisionEngine
    from world.models import NpcSeed

    env = _env()
    env.spawn_npc(
        NpcSeed(name="王二", gender="男", age=34, description="跑得快"), location_id="hall",
    )
    env.place_agent(agent_id="a1", location_id="hall")
    banned = ("NPC", "npc", "工具人")

    # 1. The scene the judge reads
    scene = assemble_scene_context(
        "a1", environment=env, directory=LiveWorldDirectory.from_agents({}, env),
        visibility=SceneVisibility.GOD,
    ).text
    assert "此处还有" in scene                      # the information that should be there is
    for word in banned:
        assert word not in scene, f"判官现场泄漏代码层档次「{word}」：{scene}"

    # 2. The first-person decision prompt, scanning the action menu and output schema too.
    menu = "\n".join(c.description for c in _action_space())
    schema = DecisionEngine.__dict__["_build_decision_prompt"].__doc__ or ""
    # Scan only the human-readable half: key names (errand_npc_index etc.) are machine contract,
    # like physical_entity_index; the model just fills them and never writes them into narrative.
    # The text after the colon does.
    from agent import decision as _d
    source_schema = "\n".join(
        ln.split(":", 1)[1] if ":" in ln else ln
        for ln in pathlib.Path(_d.__file__).read_text().splitlines()
        if ln.lstrip().startswith(('"errand_', '"physical_npc'))
    )
    for text, where in ((menu, "行动菜单"), (schema, "prompt docstring"),
                        (source_schema, "输出 schema")):
        for word in banned:
            assert word not in text, f"{where}泄漏代码层档次「{word}」"


def _packet(*, visible_npcs=None, inbox=None):
    """A minimal but real perception packet: the binder only validates when ``packet is not
    None``."""
    from agent.need import NeedEvaluation
    from agent.perception import InternalContext, PerceptionPacket

    return PerceptionPacket(
        agent_id="a1",
        step=1,
        spatial=SpatialPerception(
            location_id="hall", location_view=LocationView(name="大殿"),
            world_time_label="", current_step=1,
            visible_agents={"a2": _presence("乙")},
            visible_npcs=dict(visible_npcs or {}),
        ),
        inbox=list(inbox or []),
        broadcasts=[],
        internal_context=InternalContext(
            emotion=None, dominant_need=None, active_needs=[],
            short_term_goals=[], long_term_goals=[],
            factual_memories=[], experiential_memories=[], relevant_relations=[],
            need_evaluation=NeedEvaluation(
                dominant_need=None, scores={}, active_needs=[],
                short_term_goals=[], long_term_goals=[], prompt_context="",
            ),
        ),
    )


def _presence(name: str):
    from core.interfaces.perception import PerceivedPresence
    return PerceivedPresence(identity=PerceivedIdentity(name=name))


def _message(sender_id: str, sender_name: str, *, is_agent: bool = True) -> Message:
    return Message(
        id=f"m_{sender_id}", world_id="w", sender_id=sender_id, content="话",
        recipients=["a1"], location_scope=None, deliver_step=1, created_step=0,
        sender_name=sender_name, sender_is_agent=is_agent,
    )


# ---------------------------------------------------------------------------
# One table, two kinds of bodies: anything that does people-things with "who's standing here" must
# use agents_at

def test_the_place_tells_bodies_from_people() -> None:
    """``bodies_at`` holds both tiers; ``agents_at`` / ``npcs_at`` are its two complementary halves.

    The criterion (in ``_npcs`` or not) may live only in these two views. Filtering again anywhere
    else is a second copy, and a stale copy doesn't error.
    """
    env = _env()
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    env.place_agent(agent_id="a1", location_id="hall")

    assert set(env.bodies_at("hall")) == {npc_id, "a1"}
    assert env.agents_at("hall") == ["a1"]
    assert [n.npc_id for n in env.npcs_at("hall")] == [npc_id]


@pytest.mark.asyncio
async def test_speaking_to_a_room_of_tool_bodies_is_speaking_to_nobody() -> None:
    """When the only one in the room is a body without cognition, speaking aloud means nobody
    hears.

    It has no inbox, no perception and remembers nothing; every message delivered to it is dead
    letter, yet the system would conclude "someone's listening". The criterion has one source in two
    places (``_resolve_receivers`` and SEND_MESSAGE's audibility check), so both are tested here.
    """
    from engine.message_system import MessageSystem
    from providers.message.in_memory import InMemoryMessageProvider
    from core.interfaces.urgency import Urgency

    env = _env()
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    env.place_agent(agent_id="a1", location_id="hall")

    ms = MessageSystem(InMemoryMessageProvider(), world_id="w")
    broadcast = Message(
        id="m1", world_id="w", sender_id="a1", sender_name="李世民",
        content="都听好了", recipients=None, location_scope="hall",
        deliver_step=2, created_step=1, urgency=Urgency.NORMAL,
    )
    assert ms._resolve_receivers(broadcast, agent_ids=["a1"], environment=env) == []
    assert npc_id not in ms._resolve_receivers(broadcast, agent_ids=["a1"], environment=env)


def test_a_letter_from_a_body_without_a_mind_builds_no_relationship() -> None:
    """Its messages are heard and remembered as coming from it, but form no relation with it.

    A relation is an expectation of "how this person will treat me", and a fully obedient body
    without cognition creates none; such a relation would grow all the way into relation evolution,
    get a set of labels, and land in the relation graph.
    """
    ordinary = Message(
        id="m1", world_id="w", sender_id="a2", sender_name="李建成", content="来一趟",
        recipients=["a1"], location_scope=None, deliver_step=1, created_step=1,
    )
    from_a_tool_body = Message(
        id="m2", world_id="w", sender_id="npc_x", sender_name="王二", content="我回来了",
        recipients=["a1"], location_scope=None, deliver_step=1, created_step=1,
        sender_is_agent=False,
    )
    assert ordinary.sender_is_agent is True        # default is "a person"
    assert from_a_tool_body.sender_is_agent is False
    assert from_a_tool_body.sender_name == "王二"   # narratively still a real person


def test_a_trace_it_leaves_never_enters_the_identity_index() -> None:
    """Ambient from its errands includes it under "who did it" but not under "which people it
    concerns".

    The former is what self-exclusion needs (all bodies); the latter goes into memory's identity
    index and, via ``collect_recent_related_agents``, becomes relation-evolution candidates, so it
    may only contain beings with cognition.
    """
    env = _env()
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id
    env.place_agent(agent_id="a1", location_id="hall")

    env.record_carry_observation(
        location_id="hall", observation="在大殿，王二把木匣搁下了。",
        actor_ids=(npc_id, "a1"),
    )
    event = env._carry_annotations["hall"][0]
    assert event.actor_ids == (npc_id, "a1")       # who did it: both tiers
    assert event.agent_actor_ids == ("a1",)        # which people: only those with cognition


def _environment_functions():
    import ast as _ast
    tree = _ast.parse(pathlib.Path("engine/environment.py").read_text(encoding="utf-8"))
    return _ast, tree, [
        n for n in _ast.walk(tree)
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
    ]


def test_what_kind_of_body_it_is_is_never_read_off_the_npc_roster() -> None:
    """``_npcs`` is the data table for the Npc kind, not a lookup criterion.

    "In ``_npcs`` or not" welds "only two kinds of bodies" into every call site: a third kind would
    count as an agent and flow into TALK admission, arbitration and relation evolution, which
    silently null-step on ids that don't resolve to an ``Agent``.
    """
    _ast, _, functions = _environment_functions()

    def _is_npc_roster(node) -> bool:
        return (
            isinstance(node, _ast.Attribute) and node.attr == "_npcs"
            and isinstance(node.value, _ast.Name) and node.value.id == "self"
        )

    offenders = [
        func.name
        for func in functions
        for node in _ast.walk(func)
        if isinstance(node, _ast.Compare)
        and any(isinstance(op, (_ast.In, _ast.NotIn)) for op in node.ops)
        and any(_is_npc_roster(c) for c in node.comparators)
    ]
    # The one in spawn_npc isn't a criterion; it's the collision loop when allocating an id (is
    # this id taken).
    assert sorted(set(offenders)) == ["spawn_npc"], offenders


def test_every_read_of_a_body_kind_is_a_positive_one() -> None:
    """Ask about body kind only as "is it this kind", never "is it not that kind".

    ``is not NPC`` and ``is AGENT`` give the same answer today but won't tomorrow; that's exactly
    how the complement criterion sneaks back in a different spelling, without a single test going
    red.
    """
    _ast, _, functions = _environment_functions()

    def _reads_kind(node) -> bool:
        # self._bodies.get(x) / self._bodies[x]
        target = node.func.value if isinstance(node, _ast.Call) and isinstance(
            node.func, _ast.Attribute
        ) else getattr(node, "value", None)
        return (
            isinstance(target, _ast.Attribute) and target.attr == "_bodies"
            and isinstance(target.value, _ast.Name) and target.value.id == "self"
        )

    offenders = [
        func.name
        for func in functions
        for node in _ast.walk(func)
        if isinstance(node, _ast.Compare) and _reads_kind(node.left)
        and not all(isinstance(op, _ast.Is) for op in node.ops)
    ]
    assert offenders == [], f"这些地方在反着问身体的种类：{offenders}"


def test_a_third_kind_of_body_is_neither_an_agent_nor_an_npc() -> None:
    """When another kind of embodied thing arrives, existing checks need no change at all.

    That's the whole difference between a complement and a positive check: a complement silently
    turns it into an agent. Here a body of unknown kind stands in the room: it must be on the body
    list, off both presence lists, and out of the identity index.
    """
    env = _env()
    env.place_agent(agent_id="a1", location_id="hall")
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id

    env._place_body("beast1", "hall", "beast")   # a kind outside BodyKind

    assert set(env.bodies_at("hall")) == {"a1", npc_id, "beast1"}
    assert env.agents_at("hall") == ["a1"], "它不是人,不该进能听见、能收信、能结关系的那份"
    assert [n.npc_id for n in env.npcs_at("hall")] == [npc_id], "它也不是 Npc,差不动它"
    assert env.has_cognition("beast1") is False
    assert env._cognizant(("a1", npc_id, "beast1")) == ("a1",)

    spatial = env.spatial_for(agent_id="a1")
    assert set(spatial.visible_agents) == set()
    assert set(spatial.visible_npcs) == {npc_id}


def test_the_npc_roster_holds_exactly_the_bodies_registered_as_npcs() -> None:
    """The roster (``_npcs``) and the kind table (``_bodies``) must say the same thing.

    Both are written together only at creation and restore; writing them separately produces bodies
    "registered as Npc but with no data", which ``npcs_at`` positively selects and then KeyErrors
    on.
    """
    env = _env()
    env.place_agent(agent_id="a1", location_id="hall")
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    env.spawn_npc(NpcSeed(name="张三"), location_id="hall")

    def _invariant() -> None:
        from world.models import BodyKind
        kinded = {bid for bid, kind in env._bodies.items() if kind is BodyKind.NPC}
        assert set(env._npcs) == kinded, (set(env._npcs), kinded)

    _invariant()

    restored = _env()
    restored.restore_state(env.snapshot_state())
    from world.models import BodyKind
    assert {bid for bid, k in restored._bodies.items() if k is BodyKind.NPC} == set(restored._npcs)
    assert set(restored._npcs) == set(env._npcs), "还原过来的名册要一样"


def test_a_position_api_never_claims_to_take_an_agent() -> None:
    """Move, remove and locate work for every kind of body, so their parameters may not be named
    ``agent_id``: that signature lies without raising, and callers end up mixing agents and Npcs.

    Placement (``place_agent``) is the opposite: it accepts one kind, because placement registers
    the kind, inferred from the entry point used, never self-reported by the caller.
    """
    import inspect

    for name in ("move_body", "remove_body", "transit_origin", "get_body_location"):
        params = inspect.signature(getattr(EnvironmentSystem, name)).parameters
        assert "agent_id" not in params, f"{name} 的签名说它只收 agent,实际每种身体都收"
        assert "body_id" in params, f"{name} 少了 body_id"

    placement = inspect.signature(EnvironmentSystem.place_agent).parameters
    assert "agent_id" in placement and "kind" not in placement, (
        "落地入口必须按身体的种类分开,且不接受一个自报的 kind 参数"
    )


def test_a_body_leaving_the_map_is_not_the_same_as_leaving_the_world() -> None:
    """``remove_body`` only handles location, not the roster; they're two different things.

    Merging them would say "not anywhere" and "this person isn't in the world" are the same, which
    agents refute daily (the dead are removed from the environment but stay in ``agents_dict``).
    """
    env = _env()
    env.spawn_npc(NpcSeed(name="王二"), location_id="hall")
    npc_id = env.all_npcs()[0].npc_id

    env.remove_body(npc_id)
    assert env.get_body_location(npc_id) == "unknown"
    assert env.npcs_at("hall") == []
    assert env.get_npc(npc_id) is not None, "位置没了不等于这个人没了"


def test_one_criterion_decides_whether_a_sender_is_someone() -> None:
    """Whether a sender is someone I can form a relation with has one answer: ``sender_is_agent``
    (Message's contract: the sender declares what it is).

    Any other criterion leaks: an id whitelist only knows the ids it spells out (miss the director
    and you render a relation line for "不知来源"), and an id set built from perception requires
    having met the sender. This blocks comparing an id against a set or a literal.
    """
    import ast
    from pathlib import Path

    watched = (
        Path("agent/agent.py"), Path("agent/perception_emotion.py"), Path("agent/decision.py"),
        Path("agent/perception_layer.py"), Path("engine/world_pressure.py"),
    )
    non_agent_ids = ("narrator", "director", "world", "system")
    hits: list[str] = []
    for path in watched:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            src = ast.unparse(node)
            low = src.lower()
            if any(isinstance(op, ast.In) for op in node.ops):
                if "npc" in low and "sender" in low:
                    hits.append(f"{path}: {src}")
            if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                if any(f'"{name}"' in src or f"'{name}'" in src for name in non_agent_ids):
                    hits.append(f"{path}: {src}")
    assert not hits, "发信人判别另立了判据:\n" + "\n".join(hits)


def test_a_word_with_no_audience_is_not_shouted_but_kept_as_the_briefing() -> None:
    """Spoken words need listeners, via the same two channels as SEND_MESSAGE: a named recipient,
    or errand_announce.

    With neither, the words are the instructions given when sending the errand, so they go into
    action_description, not ``ErrandOrder``. Don't infer a public shout from an empty recipient: the
    sender can't see who's at the destination, and the briefing often must stay private. The flag
    is consumed in decide; downstream, a message always has a real listener.
    """
    from agent.decision import _bind_errand, _Slots

    def _bind(**extra):
        return _bind_errand(_Slots(
            payload={"errand_npc_index": 1, "errand_destination_index": 1, **extra},
            packet=_packet(),
            visible_ids=[], roster_ids=["a2"], reachable_ids=["gate"],
            entity_ids=[], entity_types={}, npc_ids=["npc_x"], own_item_ids=[],
        ))

    # No listener: the words aren't spoken, but they don't vanish either; they go into the
    # description
    briefing = _bind(errand_message="你去东宫外头转一圈，回来报我")
    assert briefing.errand.message == ""
    assert briefing.description_suffix == "（我吩咐他：你去东宫外头转一圈，回来报我）"

    # A named recipient: said only to him
    direct = _bind(errand_recipient_index=1, errand_message="殿下召见")
    assert (direct.errand.message, direct.errand.recipient_id) == ("殿下召见", "a2")
    assert direct.description_suffix == ""

    # Declared public: announced aloud, no recipient needed
    public = _bind(errand_announce=True, errand_message="城门今夜不开")
    assert (public.errand.message, public.errand.recipient_id) == ("城门今夜不开", "")
    assert public.description_suffix == ""
