"""What the actor sees: the observer's read models filtered to one character.

Every split here is one the read model already records (an event's public side and audience,
an entity's ``is_public``, a body's public and private views); nothing decides visibility a
second time. Of a step the character sees its own body and mind, the public side of the place
it ends the step in, and every event whose audience includes it.

The two follow when the character perceives. What happened in the step reaches it as the next
step begins, where it ended this one, hence the public side of that place. What arrived as this
step began (letters, a place's announcements and broadcasts) reached it where it stood then,
hence by audience, wherever it ended the step. Between two waypoints (in transit) there is no
place to watch from, so only its own body and its own events.

Of others it sees what shows at a glance (name, gender, age, appearance, condition), never
their inner life or narrative tier: ``is_main_character`` is None, "not known to this viewer".
"""

from __future__ import annotations

from dataclasses import replace

from interaction.models import (
    ActionPrivateView,
    ActionSummary,
    AgentPrivateView,
    AgentProfile,
    AgentView,
    BroadcastSummary,
    EntityPrivateView,
    EntityView,
    LocationScope,
    MessageSummary,
    NpcPrivateView,
    NpcView,
    ObserverView,
    ScopeEvent,
    StepView,
    TransitScope,
    WorldGraph,
    WorldScope,
)


def step_view_for(view: StepView, viewer_id: str) -> StepView:
    """``view`` as the character ``viewer_id`` perceived it."""
    end_place = next(
        (pid for pid, scope in view.locations.items() if any(a.agent_id == viewer_id for a in scope.agents)),
        None,
    )
    locations: dict[str, LocationScope] = {}
    for place_id, scope in view.locations.items():
        if place_id == end_place:
            locations[place_id] = replace(
                scope,
                agents=[_agent(a, viewer_id) for a in scope.agents],
                npcs=[_npc(n) for n in scope.npcs],
                entities={eid: _entity(e) for eid, e in scope.entities.items() if e.is_public},
                events=[e for e in (_public_or_own(ev, viewer_id) for ev in scope.events) if e is not None],
            )
            continue
        own = [e for e in scope.events if _in_audience(e, viewer_id)]
        if own:
            locations[place_id] = LocationScope(location_id=place_id, name=scope.name, events=own)
    return replace(
        view,
        locations=locations,
        transit=TransitScope(
            agents=[a for a in view.transit.agents if a.agent_id == viewer_id],
            events=[e for e in view.transit.events if _in_audience(e, viewer_id)],
        ),
        world=WorldScope(
            events=[e for e in (_public_or_own(ev, viewer_id) for ev in view.world.events) if e is not None],
        ),
        observer=ObserverView(),
    )


def profile_for(profile: AgentProfile, viewer_id: str) -> AgentProfile:
    """The viewer's own profile whole; anyone else's only as they look."""
    if profile.agent_id == viewer_id:
        return profile
    return AgentProfile(
        agent_id=profile.agent_id,
        name=profile.name,
        role="",
        age=profile.age,
        gender=profile.gender,
        background="",
        appearance=profile.appearance,
        color=profile.color,
        is_main_character=None,
    )


def graph_for(graph: WorldGraph, viewer_id: str) -> WorldGraph:
    """The viewer's own relations: its outgoing edges and the people they lead to."""
    edges = [e for e in graph.edges if e.from_id == viewer_id]
    ids = {viewer_id, *(e.to_id for e in edges)}
    return WorldGraph(
        nodes=[
            n if n.id == viewer_id else replace(n, role="", is_main_character=None)
            for n in graph.nodes if n.id in ids
        ],
        edges=edges,
    )


# ----- one step --------------------------------------------------------------


def _audience(event: ScopeEvent) -> list[str] | None:
    """Who the event reached or involved; None when it has no such list (public where it sits).

    For an action it is who is inside its private side; for what arrived as the step began (a
    letter, a place's announcement or broadcast) it is who received it, wherever they ended
    the step."""
    if isinstance(event, ActionSummary):
        return event.private.audience
    if isinstance(event, (MessageSummary, BroadcastSummary)):
        return event.audience
    return None


def _in_audience(event: ScopeEvent, viewer_id: str) -> bool:
    audience = _audience(event)
    return audience is not None and viewer_id in audience


def _public_or_own(event: ScopeEvent, viewer_id: str) -> ScopeEvent | None:
    """An event in the viewer's place: whole if it is inside, its public side if not."""
    if _in_audience(event, viewer_id):
        return event
    if isinstance(event, ActionSummary):
        # An act that showed nothing here (an unexposed covert act) has no public side.
        if event.public is None:
            return None
        return replace(event, private=ActionPrivateView(audience=[], outcome="", gist=""))
    if isinstance(event, (MessageSummary, BroadcastSummary)):
        return event if event.audience is None else None
    return event


def _agent(agent: AgentView, viewer_id: str) -> AgentView:
    if agent.agent_id == viewer_id:
        return agent
    return replace(
        agent,
        is_main_character=None,
        public=replace(
            agent.public, carrying={eid: _entity(e) for eid, e in agent.public.carrying.items()},
        ),
        private=AgentPrivateView(emotion=""),
    )


def _npc(npc: NpcView) -> NpcView:
    return replace(
        npc,
        public=replace(npc.public, carrying={eid: _entity(e) for eid, e in npc.public.carrying.items()}),
        private=NpcPrivateView(),
    )


def _entity(entity: EntityView) -> EntityView:
    """A thing someone else has, or that lies about: seeing it is not reading it."""
    return replace(entity, private=EntityPrivateView())
