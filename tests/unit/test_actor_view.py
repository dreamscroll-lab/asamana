"""The actor's view is the observer's read models filtered to one character: its own body and
mind, the public side of the place it ends the step in, and events whose audience includes it.
Of others, only who they are and how they look; never their inner life or narrative tier."""

from __future__ import annotations

from interaction.actor_view import graph_for, profile_for, step_view_for
from interaction.models import (
    ActionPrivateView,
    ActionSummary,
    AgentPrivateView,
    AgentProfile,
    AgentPublicView,
    AgentView,
    BroadcastSummary,
    EntityPrivateView,
    EntityView,
    GraphEdge,
    GraphNode,
    InjectionSummary,
    LocationScope,
    MessageSummary,
    NpcPrivateView,
    NpcView,
    ObservationSummary,
    ObservedView,
    ObserverView,
    StepView,
    TransitScope,
    WorldGraph,
    WorldScope,
    WorldTimeView,
)

ME = "agent-me"


def _body(agent_id: str, **private: object) -> AgentView:
    return AgentView(
        agent_id=agent_id, name=agent_id, gender="女", age=30, is_main_character=True,
        public=AgentPublicView(
            activity_status="站着", condition="手臂带伤",
            carrying={"seal": EntityView(name="印", entity_type="item", state="", private=EntityPrivateView("密文"))},
        ),
        private=AgentPrivateView(emotion="anger", long_term_goals=["夺位"], **private),  # type: ignore[arg-type]
    )


def _action(seq: int, audience: list[str], *, public: str | None = "有人动手") -> ActionSummary:
    return ActionSummary(
        seq=seq, initiator_id=audience[0] if audience else "", action_type="physical",
        public=ObservedView(public) if public is not None else None,
        private=ActionPrivateView(audience=audience, outcome="完整经过", gist="一句话"),
    )


def _letter(seq: int, audience: list[str] | None) -> MessageSummary:
    return MessageSummary(
        seq=seq, message_id=f"m{seq}", sender_id="agent-x", sender_name="某", receiver_ids=[],
        scope="direct" if audience else "place", content="信", spoken="信", audience=audience,
    )


def _step(**fields: object) -> StepView:
    return StepView(world_id="w", step=3, world_time=WorldTimeView.from_payload({}), **fields)  # type: ignore[arg-type]


def test_in_the_place_it_ends_in_the_character_sees_bodies_and_things_as_they_show() -> None:
    hidden = EntityView(name="暗格", entity_type="item", state="", is_public=False)
    letter = EntityView(name="信", entity_type="item", state="", private=EntityPrivateView("内容"))
    view = step_view_for(_step(locations={"hall": LocationScope(
        location_id="hall", name="大殿",
        agents=[_body(ME), _body("agent-b")],
        npcs=[NpcView(npc_id="n1", name="仆", private=NpcPrivateView(outcome="跑腿"))],
        entities={"hidden": hidden, "letter": letter},
    )}), ME)

    [me, other] = view.locations["hall"].agents
    assert me == _body(ME)
    assert (other.name, other.gender, other.age, other.public.condition) == ("agent-b", "女", 30, "手臂带伤")
    assert other.is_main_character is None
    assert other.private == AgentPrivateView(emotion="")
    assert other.public.carrying["seal"].private.content == ""
    assert view.locations["hall"].npcs[0].private == NpcPrivateView()
    # Seeing a thing is not reading it; a thing nobody here can perceive isn't there at all.
    assert set(view.locations["hall"].entities) == {"letter"}
    assert view.locations["hall"].entities["letter"].private.content == ""


def test_own_belongings_keep_what_is_written_on_them() -> None:
    view = step_view_for(_step(locations={"hall": LocationScope(
        location_id="hall", name="大殿", agents=[_body(ME)],
    )}), ME)

    assert view.locations["hall"].agents[0].public.carrying["seal"].private.content == "密文"


def test_events_here_show_their_public_side_unless_the_character_is_inside() -> None:
    view = step_view_for(_step(locations={"hall": LocationScope(
        location_id="hall", name="大殿", agents=[_body(ME)],
        events=[
            _action(1, ["agent-b", ME]),
            _action(2, ["agent-b"]),
            _action(3, ["agent-b"], public=None),
            _letter(4, ["agent-x", "agent-b"]),
            _letter(5, None),
            ObservationSummary(seq=6, initiator_id="agent-b", participant_ids=["agent-b"], text="他走了"),
        ],
    )}), ME)

    shown = {e.seq: e for e in view.locations["hall"].events}
    assert set(shown) == {1, 2, 5, 6}
    assert isinstance(shown[1], ActionSummary) and shown[1].private.outcome == "完整经过"
    assert isinstance(shown[2], ActionSummary) and shown[2].private.outcome == ""
    assert shown[2].public == ObservedView("有人动手")


def test_elsewhere_only_its_own_events_and_nobody_in_them() -> None:
    view = step_view_for(_step(locations={
        "hall": LocationScope(location_id="hall", name="大殿", agents=[_body(ME)]),
        "gate": LocationScope(
            location_id="gate", name="宫门", agents=[_body("agent-b")],
            events=[_letter(1, ["agent-x", ME]), _action(2, ["agent-b"]), _letter(3, None)],
        ),
        "yard": LocationScope(location_id="yard", name="内院", agents=[_body("agent-c")]),
    }), ME)

    assert set(view.locations) == {"hall", "gate"}
    gate = view.locations["gate"]
    assert (gate.agents, gate.npcs, gate.entities) == ([], [], {})
    assert [e.seq for e in gate.events] == [1]


def test_on_the_way_only_itself_and_its_own_events() -> None:
    view = step_view_for(_step(transit=TransitScope(
        agents=[_body(ME), _body("agent-b")],
        npcs=[NpcView(npc_id="n1", name="仆")],
        events=[_action(1, [ME]), _action(2, ["agent-b"])],
    )), ME)

    assert [a.agent_id for a in view.transit.agents] == [ME]
    assert view.transit.npcs == []
    assert [e.seq for e in view.transit.events] == [1]


def test_the_world_scope_reaches_everyone_and_the_observer_nobody() -> None:
    proclamation = BroadcastSummary(seq=1, content="大赦天下")
    view = step_view_for(_step(
        world=WorldScope(events=[proclamation]),
        observer=ObserverView(injections=[InjectionSummary(
            id="i1", authored_by="director", narrative="宫门失火", seq=2,
        )]),
    ), ME)

    assert view.world.events == [proclamation]
    assert view.observer == ObserverView()


def test_profiles_show_others_only_who_they_are() -> None:
    def profile(agent_id: str) -> AgentProfile:
        return AgentProfile(
            agent_id=agent_id, name=agent_id, role="宰相", age=50, gender="男", background="出身",
            appearance="清瘦", color="#123456", is_main_character=True, core_traits=["多疑"],
            core_values=["权"], self_image="能臣", life_goal="封侯", secret="私通",
        )

    assert profile_for(profile(ME), ME) == profile(ME)
    assert profile_for(profile("agent-b"), ME) == AgentProfile(
        agent_id="agent-b", name="agent-b", role="宰相", age=50, gender="男", background="出身",
        appearance="清瘦", color="#123456", is_main_character=None,
    )


def test_the_graph_is_the_characters_own_relations() -> None:
    graph = WorldGraph(
        nodes=[GraphNode(id=i, name=i, role="r", is_main_character=True) for i in (ME, "b", "c")],
        edges=[
            GraphEdge(from_id=ME, to_id="b", trust=0.1, affection=0.2),
            GraphEdge(from_id="b", to_id=ME, trust=0.3, affection=0.4),
            GraphEdge(from_id="b", to_id="c", trust=0.5, affection=0.6),
        ],
    )

    view = graph_for(graph, ME)

    assert [(e.from_id, e.to_id) for e in view.edges] == [(ME, "b")]
    assert [(n.id, n.role, n.is_main_character) for n in view.nodes] == [(ME, "r", True), ("b", "r", None)]


def test_what_arrived_as_the_step_began_follows_who_received_it() -> None:
    """A place's broadcast and announcement were heard where the character stood as the step
    began: kept after it leaves, unseen by whoever arrives by the step's end."""
    def at_gate(audience: list[str]) -> list:
        return [
            BroadcastSummary(seq=1, content="宫门走水", audience=audience),
            MessageSummary(
                seq=2, message_id="m2", sender_id="agent-x", sender_name="某", receiver_ids=audience,
                scope="place", content="肃静", spoken="肃静", audience=audience,
            ),
        ]

    left = step_view_for(_step(locations={
        "gate": LocationScope(location_id="gate", name="宫门", events=at_gate([ME])),
        "hall": LocationScope(location_id="hall", name="大殿", agents=[_body(ME)]),
    }), ME)
    arrived = step_view_for(_step(locations={
        "gate": LocationScope(location_id="gate", name="宫门", agents=[_body(ME)], events=at_gate(["agent-b"])),
    }), ME)

    assert [e.seq for e in left.locations["gate"].events] == [1, 2]
    assert arrived.locations["gate"].events == []
