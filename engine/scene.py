"""The adjudication scene — who and what is at an actor's location, rendered for executor prompts.

Narrative-layer text only: names and descriptions, never ids. It lives outside ``engine.executors``
because modules that are not executors assemble scenes too (``npc_runner``), and importing anything
under that package runs its ``__init__``, which loads every executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from agent.personality import ActionStatus
from core.interfaces.perception import NpcIdentity, PerceivedNpc, Situation
from core.prompts import (
    SituationVoice, person_referent, render_condition, render_entity, render_npc,
    render_situation_header,
)
from engine.environment import MAX_SCENE_ENTITIES

# Entities are only what the story tracks; without this line an empty item list reads as an empty
# room. Not numbered: these are no entities, so nothing can be indexed, seized or handed over.
_ORDINARY_FIXTURES_LINE = "- 此处通常就有的：符合此时此地的基本生活用品、办公书写用品与基础设施（不单列，随手可用）"

if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.directory import WorldDirectory
    from engine.environment import EnvironmentSystem
    from world.models import Npc, WorldEntity


def situation_for(environment: "EnvironmentSystem", agent_id: str) -> Situation:
    """Build an agent's narrative Situation (location_view + world time_label) from god-view env.

    Single source for the time+place anchor across executor prompts, both THIRD and FIRST voice.
    """
    loc_id = environment.get_body_location(agent_id)
    # Go through the place roster's renderer, not a copy. ``None`` = not on any registered place.
    return Situation(
        location_view=environment.space.view_of(loc_id),
        time_label=environment.current_world_time_label,
    )


def _scene_owner_name(
    item: "WorldEntity", viewer_id: str, directory: "WorldDirectory", first_person: bool,
) -> str | None:
    """How the holder is referred to in this scene; things on the ground have no holder.

    In first person his own things read as "我" (me): his own name handed back breaks the voice.
    """
    if not item.owner_id:
        return None
    if first_person and item.owner_id == viewer_id:
        return "我"
    return directory.agent_name(item.owner_id)


def scene_npc_view(npc: "Npc") -> PerceivedNpc:
    """Live ``Npc`` → the shape the renderer consumes.

    God-view side only; the cognition side gets the ``PerceivedNpc`` assembled by perception,
    where information asymmetry is enforced.
    """
    return PerceivedNpc(
        identity=NpcIdentity(
            name=npc.name, gender=npc.gender, age=npc.age, description=npc.description,
        ),
        busy=npc.busy,
        condition=render_condition(npc.condition) if npc.condition else "",
    )


@dataclass(frozen=True)
class SceneContext:
    """The two faces of a scene: the text the LLM reads, and the things behind the numbered line.

    The list is fixed at render time: recomputed after the judge's LLM call, a snatched item
    would shift every index.
    """

    text: str
    own_entities: tuple["WorldEntity", ...] = ()


class SceneVisibility(str, Enum):
    """Whose eyes this scene is seen through; orthogonal to ``voice`` (whose person it's told in).

    Kept separate so "third person, but only what this body can see" (an errand's report) is
    expressible.

    - ``GOD`` judge view: everything in the room, hidden things included.
    - ``OWN_EYES`` its own things (even if hidden) plus public ones; same rule as
      ``EnvironmentSystem.spatial_for``.
    """

    GOD = "god"
    OWN_EYES = "own_eyes"


def assemble_scene_context(
    agent_id: str,
    *,
    environment: "EnvironmentSystem",
    directory: "WorldDirectory",
    agents: "dict[str, Agent] | None" = None,
    with_background: bool = False,
    include_header: bool = True,
    voice: SituationVoice = SituationVoice.THIRD,
    visibility: SceneVisibility,
    split_own_entities: bool = False,
    reveal_all_content: bool = False,
    reveal_intents: bool = True,
    with_ordinary_fixtures: bool = True,
) -> SceneContext:
    """Assemble the on-site situation (location / present people / items) as
    narrative-layer text for executor adjudication prompts.

    Mutable facts come from EnvironmentSystem, immutable identity from WorldDirectory; names and
    descriptions only, never ids.

    ``agents`` (pass it whenever you have it) feeds two things: the present-people line's
    **condition** mark (「双手被反绑」), safe for every caller since it is visible on sight; and
    ``with_background``, which is NOT.

    ``with_background`` adds a brief persona per present person. **FUNCTIONAL judge only**: it
    stops the referee guessing blind, but in a first-person prompt it would hand the agent a
    stranger's inner background.

    ``include_header=False`` drops the THIRD-voice time+place line, for in-character callers
    that already anchor it in FIRST voice.

    ``voice`` (in whose person it is told) governs only the items line: ``FIRST`` makes his own
    things read 「由我持有」; ``THIRD`` names every holder.

    ``visibility`` (whose eyes) is required at every call site rather than derived: a
    first-person prompt fed the god view silently hands the reader what he cannot know.

    ``split_own_entities`` moves the reader's own things into a numbered line and returns the same
    list on ``SceneContext.own_entities``, for a caller that has the LLM pick one by number (WORK).
    FIRST voice only: the line says 「我手上的」.

    ``reveal_intents=False`` keeps only whether each present person is busy. The judge needs
    intents only to rule whether people would notice; the content otherwise gets written up as if
    it had happened. Turn it off for any judge whose output others read or that enters memory.

    ``reveal_all_content`` is NOT implied by the god view: what a thing *says* stays at what
    ``agent_id`` can read unless this is set. Set it only when the verdict is the actor learning
    what something says (COVERT); a judge whose prose reaches others (PHYSICAL, TALK) would leak a
    sealed letter's words.

    ``with_ordinary_fixtures=False`` drops the ordinary-fixtures line, a premise for judges rather
    than something seen: for callers that relay the scene as someone's report (an errand).
    """
    loc_id = environment.get_body_location(agent_id)
    lines: list[str] = []

    header = (
        render_situation_header(situation_for(environment, agent_id), voice=SituationVoice.THIRD)
        if include_header else ""
    )
    if header:
        lines.append(header)

    # Agents and Npcs on separate lines: the judge must know what an Npc is for to rule on an
    # errand. The "which kind" criterion lives only in ``agents_at`` / ``npcs_at``; don't filter
    # again here. The reader may be either kind, so both halves exclude it.
    bystander_ids = [aid for aid in environment.agents_at(loc_id) if aid != agent_id]
    npcs_here = [npc for npc in environment.npcs_at(loc_id) if npc.npc_id != agent_id]
    if bystander_ids:
        people: list[str] = []
        for aid in bystander_ids:
            entry = directory.describe(aid)
            # From the live Agent: condition is mutable state the directory must not carry.
            # Without it the judge treats someone bound and kneeling as free to lunge.
            bystander = agents.get(aid) if agents else None
            condition = (
                render_condition(bystander.personality.state.condition)
                if bystander is not None else ""
            )
            # Gender: the judge writes third-person prose about present people and would
            # otherwise guess pronouns.
            people.append(person_referent(
                directory.agent_name(aid),
                entry.gender if entry is not None else "",
                *((entry.role,) if entry is not None and entry.role else ()),
                *((condition,) if condition else ()),
            ))
        lines.append("- 在场的其他人：" + "、".join(people))
        if with_background and agents:
            for aid in bystander_ids:
                agent_obj = agents.get(aid)
                name = directory.agent_name(aid)
                bg = _person_background(agent_obj, name)
                intent = _person_current_intent(agent_obj)
                # Quoted as that person's own unresolved words. An empty intent must be stated,
                # not dropped: otherwise the judge follows expected_outcome into inventing deeds
                # for someone who did nothing (see IDLE_BYSTANDER_VERDICT_RULE, reveal_intents).
                intent_clause = (
                    (f"此刻意图：「{intent}」" if intent else "此刻没有任何动作") if reveal_intents
                    else ("此刻正忙着手上的事" if intent else "此刻没有任何动作")
                )
                if bg:
                    lines.append(f"  · {bg}；{intent_clause}")
                else:
                    lines.append(f"  · {name}：{intent_clause}")
    else:
        # Absence is a key fact for adjudication: state it, never omit it.
        lines.append("- 在场的其他人：无")

    # In the world an Npc is a person; its tier is a code-layer notion, so the line only says he
    # does things when told. Not numbered: a #N with no index contract invites made-up indices.
    if npcs_here:
        lines.append("- 此处还有(听人吩咐做事的)：" + "、".join(
            render_npc(scene_npc_view(npc)) for npc in npcs_here
        ))

    # items_present_at includes things in present people's hands; owner_name marks the holder.
    # Stated even when empty.
    first_person = voice is SituationVoice.FIRST
    own_eyes = visibility is SceneVisibility.OWN_EYES
    # Ordering and truncation share ``reachable_in_order`` with the perception layer, or the judge
    # could rule on something the actor never saw. Filtering stays separate on purpose.
    present = environment.items_present_at(loc_id)
    own_entities = (
        environment.reachable_in_order(
            [item for item in present if item.owner_id == agent_id], viewer_id=agent_id,
        ) if split_own_entities else []
    )
    items = environment.reachable_in_order(
        [
            item for item in present
            if (not own_eyes or item.owner_id == agent_id or item.is_public)
            # Listed separately below: don't render twice.
            and not (split_own_entities and item.owner_id == agent_id)
        ],
        viewer_id=agent_id,
        limit=MAX_SCENE_ENTITIES,
    )
    if items:
        lines.append("- 现场物件：" + "；".join(
            render_entity(
                item,
                owner_name=_scene_owner_name(item, agent_id, directory, first_person),
                content=item.content if (reveal_all_content or item.readable_by(agent_id)) else "",
                held_by_viewer=first_person and item.owner_id == agent_id,
            )
            for item in items
        ))
    else:
        lines.append("- 现场物件：无")

    if split_own_entities:
        # IndexedRef contract: number even a single item. No holder mark: all are in his hands.
        held = "；".join(
            f"#{i} {render_entity(item, content=item.content, held_by_viewer=True)}"
            for i, item in enumerate(own_entities, 1)
        )
        lines.append(f"- 我手上的东西：{held or '无'}")
    if with_ordinary_fixtures:
        lines.append(_ORDINARY_FIXTURES_LINE)

    return SceneContext(text="\n".join(lines), own_entities=tuple(own_entities))


def _person_background(agent: "Agent | None", name: str) -> str:
    """Brief who-they-are for the functional judge, so it doesn't guess blind.

    Never the secret: a bystander's has no bearing on someone else's action, and the judge's
    prose reaches other people.
    """
    if agent is None:
        return ""
    soul = agent.personality.soul
    parts: list[str] = []
    traits = "、".join(getattr(soul, "core_traits", []) or [])
    if traits:
        parts.append(f"核心性格特点：{traits}")
    values = "、".join(getattr(soul, "core_values", []) or [])
    if values:
        parts.append(f"核心价值观：{values}")
    bg = (getattr(soul, "background", "") or "").strip()
    if bg:
        parts.append(f"背景：{bg}")
    return f"{name}：{'；'.join(parts)}" if parts else ""


def _person_current_intent(agent: "Agent | None") -> str:
    """A co-present agent's own stated intent for THIS step, for the FUNCTIONAL judge only.

    current_action is set at begin and adjudication is read-only, so this is a stable snapshot
    covering every concurrent action, single-step ones included. It is unresolved INTENT in the
    agent's own voice; the scene quotes it. An idle bystander returns "", which the caller renders
    as an explicit 「此刻没有任何动作」."""
    if agent is None:
        return ""
    state = agent.personality.state
    if state.action_status != ActionStatus.IN_PROGRESS:
        return ""
    return (state.current_action or "").strip()


def observe_location(environment: "EnvironmentSystem", agent_id: str) -> str:
    """The narrative name of the actor's current location; the single entry point for every
    executor resolving "where".

    Room name; in transit → "途中" (on the way); unknown → "此处" (here); never a bare id.
    """
    return environment.narrative_location_name(environment.get_body_location(agent_id))


#: Adjudication discipline for the 「此刻没有任何动作」 (not doing anything right now) line in
#: 【现场】. Stating the absence isn't enough: the judge must know it's an established fact, or
#: expected_outcome pulls it into inventing deeds for an idle bystander.
IDLE_BYSTANDER_VERDICT_RULE: str = """\
- 【现场】里标注「此刻没有任何动作」的人，**此刻确实什么都没做**——那不是信息缺失，是已经查明
  的事实。不得为他编造任何动作或反应。
- 若这次行动的目的正是「看某人会不会做某事」，而他标着没有任何动作，那么结果就是**什么也没发生**
  ——这是正常且常见的裁断，不要为了给出一个结果而捏造客观事实。"""
