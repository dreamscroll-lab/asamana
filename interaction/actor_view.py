"""What the actor sees: the observer's read models filtered to one character.

Every split here is one the read model already records (an event's public, shared and private
tiers, an entity's ``is_public``, a body's public and private views); nothing decides visibility
a second time. Of a step the character sees its own body and mind, the public side of the place
it ends the step in, and the shared side of every event whose audience includes it; of an
action's private parts, only its own.

The two follow when the character perceives. What happened in the step reaches it as the next
step begins, where it ended this one, hence the public side of that place. What arrived as this
step began (letters, a place's announcements and broadcasts) reached it where it stood then,
hence by audience, wherever it ended the step. Between two waypoints (in transit) there is no
place to watch from, so only its own body and its own events.

Of others it sees their public CharacterIdentity (name, gender, age, role, background, appearance,
color) and how they look now, never their inner life or narrative tier:
``is_main_character`` is None, "not known to this viewer".
"""

from __future__ import annotations

from dataclasses import replace

from interaction.models import (
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
                entities=_seen_things({eid: e for eid, e in scope.entities.items() if e.is_public}),
                events=_seen_here(scope.events, viewer_id),
            )
            continue
        if own := _own(scope.events, viewer_id):
            locations[place_id] = LocationScope(location_id=place_id, events=own)
    return replace(
        view,
        locations=locations,
        transit=TransitScope(
            agents=[a for a in view.transit.agents if a.agent_id == viewer_id],
            events=_own(view.transit.events, viewer_id),
        ),
        world=WorldScope(events=_seen_here(view.world.events, viewer_id)),
        observer=ObserverView(),
    )


def profile_for(profile: AgentProfile, viewer_id: str) -> AgentProfile:
    """The viewer's own profile whole; anyone else's only who they are."""
    if profile.agent_id == viewer_id:
        return profile
    return AgentProfile(
        agent_id=profile.agent_id,
        name=profile.name,
        role=profile.role,
        age=profile.age,
        gender=profile.gender,
        background=profile.background,
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
            n if n.id == viewer_id else replace(n, is_main_character=None)
            for n in graph.nodes if n.id in ids
        ],
        edges=edges,
    )


# ----- one step --------------------------------------------------------------


def _audience(event: ScopeEvent) -> list[str] | None:
    """Who the event reached or involved; None when it has no such list (public where it sits).

    For an action it is who knows its shared side; for what arrived as the step began (a
    letter, a place's announcement or broadcast) it is who received it, wherever they ended
    the step."""
    if isinstance(event, ActionSummary):
        return event.shared.audience if event.shared is not None else None
    if isinstance(event, (MessageSummary, BroadcastSummary)):
        return event.audience
    return None


def _in_audience(event: ScopeEvent, viewer_id: str) -> bool:
    audience = _audience(event)
    return audience is not None and viewer_id in audience


def _own(events: list[ScopeEvent], viewer_id: str) -> list[ScopeEvent]:
    """Events somewhere the viewer isn't: only those it is inside."""
    return [_inside(e, viewer_id) for e in events if _in_audience(e, viewer_id)]


def _inside(event: ScopeEvent, viewer_id: str) -> ScopeEvent:
    """An event the viewer is inside: everything but the other participants' own parts."""
    if isinstance(event, ActionSummary):
        return replace(event, private=[p for p in event.private if p.agent_id == viewer_id])
    return event


def _seen_here(events: list[ScopeEvent], viewer_id: str) -> list[ScopeEvent]:
    """Events where the viewer is: each as much of it as the viewer knows."""
    return [e for e in (_public_or_own(ev, viewer_id) for ev in events) if e is not None]


def _public_or_own(event: ScopeEvent, viewer_id: str) -> ScopeEvent | None:
    """An event in the viewer's place: as an insider if it is inside, its public side if not."""
    if _in_audience(event, viewer_id):
        return _inside(event, viewer_id)
    if isinstance(event, ActionSummary):
        # An act that showed nothing here (an unexposed covert act) has no public side.
        if event.public is None:
            return None
        return replace(event, shared=None, private=[])
    if isinstance(event, (MessageSummary, BroadcastSummary)):
        return event if event.audience is None else None
    return event


def _agent(agent: AgentView, viewer_id: str) -> AgentView:
    if agent.agent_id == viewer_id:
        return agent
    return replace(
        agent,
        is_main_character=None,
        public=replace(agent.public, carrying=_seen_things(agent.public.carrying)),
        private=AgentPrivateView(emotion=""),
    )


def _npc(npc: NpcView) -> NpcView:
    return replace(
        npc,
        public=replace(npc.public, carrying=_seen_things(npc.public.carrying)),
        private=NpcPrivateView(),
    )


def _seen_things(things: dict[str, EntityView]) -> dict[str, EntityView]:
    """Things someone else has, or that lie about: seeing them is not reading them."""
    return {eid: replace(e, private=EntityPrivateView()) for eid, e in things.items()}
