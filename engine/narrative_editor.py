"""Sparse narrative injections: the automatic LLM narrative editor.

Design charter:
- Only pacing + channel adaptation. Never mutates agent state; agents react to the perception
  signals on their own.
- No injection type enums: the LLM declares the channels (broadcast / message / object changes)
  in its plan, possibly several at once.
- Object changes go through ``WorldMutationChannel`` as ``Author.SYSTEM``, which only allows
  putting a new entity down or changing / destroying an unheld entity on the ground. People and
  what they hold are the human director's only (``engine/director.py``): an automatic LLM
  must not overturn results an agent has a causal claim to. The permission exists so the world
  follows what the editor's broadcasts say appeared or vanished.
- Delivery goes through the ``InjectionDispatcher`` shared with the director
  (``engine/injection.py``); a new communication pattern needs its own channel module first.
- One rule path besides the LLM gate: cold places (``_MAX_COLD_HEAT``) skip the gate and the
  plan guides characters there. How often characters have been somewhere is an objective
  threshold; whom to tell and how stays the LLM's.
- Ledgers are separate from the director's, so the quota only constrains the editor and the
  editor's briefing never learns someone reached in from outside the world.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Collection, Iterable, Mapping, Sequence

from agent.personality import SECRET_LABEL
from core.context import note_active_call_adoption
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import IndexedRef, LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.interfaces.phenomenon import Phenomenon
from core.interfaces.severity import Severity
from core.interfaces.snapshot import SnapshotProvider, WorldSnapshot
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    CLOSED_WORLD_FACT_RULE,
    DECEASED_MARK,
    PHENOMENON_DEFINITION,
    RELATION_LABEL_DESCRIPTION,
    SEVERITY_SCALE_DESCRIPTION,
    URGENCY_SCALE_DESCRIPTION,
    person_referent,
    render_condition,
    render_location,
    strip_end_punct,
    vitality_label,
)
from core.logging import get_logger
from core.validate import require_at_least, require_positive
from engine.clock import WorldTime
from engine.injection import (
    Author,
    BroadcastSpec,
    CommittedInjection,
    InjectionDispatcher,
    InjectionLedger,
    MessageSpec,
    InjectionRecord,
    normalize_is_positive,
    parse_broadcast_spec,
    parse_message_spec,
)
from engine.world_mutation import (
    EntityMutation, Mutation, SpawnMutation, parse_entity_mutation, parse_spawn,
)

if TYPE_CHECKING:
    from agent.agent import Agent
    from core.interfaces.entity import WorldEntity
    from core.interfaces.place import Place
    from engine.space import SpaceManager

logger = get_logger(__name__)


# Narrative briefing bounds (the gate also consumes this briefing, so it must stay bounded).
_BRIEF_LOOKBACK_STEPS = 6        # steps the dynamic "recent events" block looks back
_BRIEF_MAX_RELATIONS = 8         # max relation pairs listed (main-character pairs first)
_BRIEF_MAX_PRIOR_INJECTIONS = 5  # max most-recent injections listed

# How to read the briefing's relation lines; a static definition, so it lives in both system prompts.
_RELATIONS_LEGEND = f"简报【当前关系】每行「甲 对 乙:标签」,甲是持有者、乙是对方;{RELATION_LABEL_DESCRIPTION}"

# What an injection is for; the gate decides by it and the plan designs to it, so both read one text.
_INJECTION_AIMS = """\
- 使叙事丰富多彩:向平淡或流水账的世界中注入可能引起故事往不同方向推进、或产生不同叙事内容的事件。
- 升华或加深叙事:把已经达到临界点的内容进一步升华或点燃。
- 重大转折叙事:注入可能引起叙事的内容或方向发生重大转折、反转的事件。
- 打破僵局:打破当下叙事原地打转、反复循环，无实质进展的局面。比如某些角色一直在反复做一类事情并且没有有效推进的情况。
以上只是几种典型情况,不限于此。
推进指事情真的变了:有人得到或失去了东西、知道了新的事、关系或处境变了。约定、答应、「到时候再说」本身不是推进;
同一件事被一再约到下一次,一直没有履行,就是原地打转。比如一方一再把说清楚的时间往后推、另一方一再追问，无实质进展,这就是打转。"""

# Unoccupied places the plan menu may list on top of the occupied ones. A few elsewhere are
# enough to pull the story out of one spot; more only dilutes the menu.
_MAX_STOCKED_PLACES = 5          # with unheld entities on the ground, most first
_MAX_BARE_PLACES = 3             # with nothing on them, a different draw each step

# A place's heat is how many characters stood there, summed over the last _HEAT_WINDOW_STEPS
# steps. A place with unheld entities on the ground and heat at most _MAX_COLD_HEAT is cold: the
# next check skips the pacing gate and the plan guides characters there. Without the threshold the
# quietest place would always be picked and guides would take the whole quota. Npcs count for
# neither heat nor worth, as everywhere in the editor: only characters perceive what it injects.
_HEAT_WINDOW_STEPS = 10
_MAX_COLD_HEAT = 2
# One guide names at most this many places, so different characters can be sent different ways;
# more won't fit the length limits of one broadcast and one message.
_MAX_GUIDED_PLACES = 3
# A place guided to stays out of the pick this long, so it isn't pressed again while people are on
# their way or have chosen not to go.
_GUIDE_REPEAT_STEPS = 20


# Not shared with the director's DirectivePlan: composing a beat and translating a sentence
# are different products, even if they look alike.


@dataclass(frozen=True)
class _InjectionPlan:
    narrative_desc: str
    is_positive: bool | None
    broadcast: BroadcastSpec | None
    message: MessageSpec | None
    spawn: SpawnMutation | None = None
    alter: EntityMutation | None = None
    destroy: EntityMutation | None = None
    # The cold places this plan guides characters to; empty on the gate's path.
    guided_place_ids: tuple[str, ...] = ()

    @property
    def mutations(self) -> list[Mutation]:
        return [m for m in (self.spawn, self.alter, self.destroy) if m is not None]


@dataclass(frozen=True)
class _Menu:
    """The places (``_menu_places``) and the unheld entities on them, rendered once per check: the
    plan picks from it by index, the gate reads the same rows to see whether anything elsewhere
    could be pulled in. A selection, not the whole map. ``place_rows`` / ``entity_rows`` follow
    ``places`` / ``entities`` order."""

    places: list["Place"]
    entities: list["WorldEntity"]
    place_rows: list[str]
    entity_rows: list[str]


@dataclass(frozen=True)
class _Guide:
    """The cold places the next plan must guide characters to, most worth going first."""

    place_ids: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeEditorSettings:
    """Pacing and narrative anchors for editing one world.

    ``narrative_theme`` goes only into the editor's briefing, never into agent cognition.
    The quota is a sliding window (``max_injections_per_window`` per ``quota_window`` steps): it bounds density, not a lifetime total that a long world would spend early.
    """

    narrative_theme: str = ""
    enabled: bool = True
    max_injections_per_window: int = 3
    quota_window: int = 20
    check_interval: int = 5

    def __post_init__(self) -> None:
        require_positive(self.check_interval, "check_interval")
        require_at_least(self.max_injections_per_window, 0, "max_injections_per_window")
        require_positive(self.quota_window, "quota_window")


def _step_deeds(snapshot: WorldSnapshot) -> list[Mapping[str, Any]]:
    """This step's deeds a main character took part in, eligible for the briefing. Each action
    event is one execution's beat already (the step record folds participants), so a dialogue
    counts once and the pacing gate sees the real density.
    """
    agents = snapshot.state.agents
    return [
        event for event in snapshot.events_of("action")
        if any(agents.get(pid, {}).get("is_main_character") for pid in event.get("participant_ids") or [])
        # Progress notes aren't deeds; they'd repeat one act per step it spans.
        and event.get("phase") != "ongoing_tick"
        # Never engaged the world (precondition failed); the editor would react to something
        # that never happened. An adjudicated failure stays: "he tried and was beaten" is a deed.
        and not event.get("not_executed")
    ]


def _reaches_someone(
    plan: "_InjectionPlan", living: Mapping[str, "Agent"], agent_locations: Mapping[str, str],
) -> bool:
    """Whether some living character perceives the plan: a message, a world-wide broadcast, or
    one sited where someone stands."""
    if plan.message is not None:
        return True
    broadcast = plan.broadcast
    if broadcast is None:
        return False
    if broadcast.location_scope is None:
        return True
    return any(agent_locations.get(aid) == broadcast.location_scope for aid in living)


def _numbered(rows: Sequence[str]) -> str:
    """Menu rows numbered (``#1 …``), for a call that selects from them."""
    return "\n".join(f"#{i + 1} {row}" for i, row in enumerate(rows)) or "（无）"


def _bulleted(rows: Sequence[str]) -> str:
    """Menu rows as plain bullets, for a call that only reads them."""
    return "\n".join(f"- {row}" for row in rows) or "（无）"


def _menu_places(
    locations: Sequence["Place"],
    headcount: Mapping[str, int],
    entity_counts: Mapping[str, int],
    rng: random.Random,
    must_include: Collection[str] = (),
) -> list["Place"]:
    """Places the plan may name, in ``locations`` order: every occupied place (the only ones a
    broadcast reaches), the unoccupied places with the most unheld entities on the ground, a few
    bare ones drawn by ``rng``, and ``must_include`` (the places a guide names). Don't pick the
    bare ones by a fixed rule such as nearness: while people stay put the same few would always
    win, and the rest of the map could never host an injection."""
    unoccupied = [loc for loc in locations if not headcount.get(loc.place_id)]
    stocked = sorted(
        (loc for loc in unoccupied if entity_counts.get(loc.place_id)),
        key=lambda loc: -entity_counts[loc.place_id],
    )[:_MAX_STOCKED_PLACES]
    bare = [loc for loc in unoccupied if not entity_counts.get(loc.place_id)]
    drawn = rng.sample(bare, min(len(bare), _MAX_BARE_PLACES))
    chosen = {loc.place_id for loc in (*stocked, *drawn)} | set(must_include)
    return [loc for loc in locations if headcount.get(loc.place_id) or loc.place_id in chosen]


class NarrativeEditor:
    """Narrative pacing controller + channel adapter (see the module docstring's charter):
    a rule gate + LLM gate decide whether to inject, the LLM writes the plan, and the shared
    dispatcher delivers it."""

    def __init__(
        self,
        *,
        llm_router: LLMRouter,
        snapshot_provider: SnapshotProvider,
        dispatcher: InjectionDispatcher,
        directory: WorldDirectory,
        settings: NarrativeEditorSettings,
    ) -> None:
        self.llm_router = llm_router
        self.snapshot_provider = snapshot_provider
        self._dispatcher = dispatcher
        self.directory = directory
        self.settings = settings
        self._ledger = InjectionLedger(Author.SYSTEM)
        # At most one background generation at a time: bounds cost and prevents quota races
        # (the ledger is written only at commit).
        self._inflight: "asyncio.Task[_InjectionPlan | None] | None" = None
        # Heat bookkeeping, not committed to the snapshot: a restore starts the window again,
        # which only delays the next guide by up to _HEAT_WINDOW_STEPS (and may name a place
        # guided to just before the restore).
        # How many characters stood in each place, one entry per step, oldest first.
        self._presence: deque[Counter[str]] = deque(maxlen=_HEAT_WINDOW_STEPS)
        self._guided_at: dict[str, int] = {}      # place id → step a guide to it was committed

    def list_injections(self) -> list[InjectionRecord]:
        return self._ledger.all()

    def restore_state(self, landed_injections: Iterable[Mapping[str, Any]]) -> None:
        """Rehydrate the quota window from persisted injections, so a restore doesn't grant a fresh
        quota. The ledger keeps only this editor's injections; the director's are filtered out."""
        self._ledger.restore(landed_injections)

    def _passes_rule_check(self, current_step: int) -> bool:
        settings = self.settings
        if not settings.enabled:
            return False
        if current_step <= 0 or current_step % settings.check_interval != 0:
            return False
        if self._ledger.count_since(current_step - settings.quota_window) >= settings.max_injections_per_window:
            return False
        return True

    async def poll_injection(
        self,
        *,
        current_step: int,
        world_time: WorldTime,
        world_id: str,
        all_agents: Mapping[str, "Agent"],
        agent_locations: Mapping[str, str],
        space: "SpaceManager",
        entities: Sequence["WorldEntity"],
    ) -> CommittedInjection | None:
        """The step loop's single injection entry point: commit a finished background generation
        and return it, or start one and return None. Never blocks on the LLM.

        ``agent_locations`` (agent id → where his body is; no entry when unplaced), ``space``
        and ``entities`` are read-only views from ``EnvironmentSystem`` (the author layer doesn't
        hold it). Places render from ``space``'s views, sub-places included, as characters see them. Where people
        are comes from ``agent_locations``, the table broadcasts are delivered by; don't read
        ``personality.state.current_location``: that is the agent's own record, which lags while
        he travels.
        """
        self._note_presence(current_step, all_agents, agent_locations)
        ready = await self._consume_ready(current_step, all_agents)
        if ready is not None:
            return ready
        self._ensure_generation(
            current_step, world_time, world_id, all_agents, agent_locations, space, entities,
        )
        return None

    def _note_presence(
        self, current_step: int, all_agents: Mapping[str, "Agent"], agent_locations: Mapping[str, str],
    ) -> None:
        """Count this step's characters by place. Called once per step."""
        self._presence.append(Counter(
            location for aid, agent in all_agents.items()
            if agent.is_active and (location := agent_locations.get(aid))
        ))

    def _cold_places(
        self,
        current_step: int,
        space: "SpaceManager",
        entities: Sequence["WorldEntity"],
    ) -> _Guide | None:
        """The cold places to guide characters to, or None: at most _MAX_GUIDED_PLACES, the ones
        with the most on the ground first, then the coldest. None until a whole window has been
        watched: heat over a shorter one would call every place cold at the start."""
        if len(self._presence) < _HEAT_WINDOW_STEPS:
            return None
        worth = Counter(e.location_id for e in entities if e.location_id is not None)
        heat = sum(self._presence, Counter())
        # Someone standing there already perceives what is on the ground, however cold the place.
        here_now = self._presence[-1]
        candidates: list[tuple[int, int, str]] = []
        for place_id in space.all_place_ids():
            if not worth[place_id] or here_now[place_id] or heat[place_id] > _MAX_COLD_HEAT:
                continue
            guided = self._guided_at.get(place_id)
            if guided is not None and current_step - guided < _GUIDE_REPEAT_STEPS:
                continue
            candidates.append((-worth[place_id], heat[place_id], place_id))
        if not candidates:
            return None
        picked = sorted(candidates)[:_MAX_GUIDED_PLACES]
        place_ids = tuple(place_id for _worth, _heat, place_id in picked)
        logger.info(
            "narrative_editor_guide",
            extra={
                "step": current_step,
                "location_ids": list(place_ids),
                "worth": [worth[place_id] for place_id in place_ids],
                "heat": [heat[place_id] for place_id in place_ids],
            },
        )
        return _Guide(place_ids=place_ids)

    def _ensure_generation(
        self,
        current_step: int,
        world_time: WorldTime,
        world_id: str,
        all_agents: Mapping[str, "Agent"],
        agent_locations: Mapping[str, str],
        space: "SpaceManager",
        entities: Sequence["WorldEntity"],
    ) -> None:
        """If eligible and nothing is in flight, start "gate + plan generation" in the
        background, off the step loop's critical path."""
        if self._inflight is not None:
            return
        if not self._passes_rule_check(current_step):
            return
        guide = self._cold_places(current_step, space, entities)
        self._inflight = asyncio.create_task(
            self._generate_plan(
                current_step, world_time, world_id, all_agents, agent_locations, space, entities,
                guide=guide,
            )
        )

    async def _consume_ready(
        self,
        current_step: int,
        all_agents: Mapping[str, "Agent"],
    ) -> CommittedInjection | None:
        """Commit a finished background generation and return it; never awaits an unfinished
        one. Runs at step start, before collect, so its broadcast is perceived this step.
        A None result (gate said no / generation failed) just frees the slot.
        """
        task = self._inflight
        if task is None or not task.done():
            return None
        self._inflight = None
        try:
            plan = task.result()
        except Exception as exc:  # noqa: BLE001 — Rule 1: a failed background generation means no injection
            logger.warning("injection_generation_failed", extra={"error": str(exc)})
            return None
        if plan is None:
            return None
        return await self._commit_plan(plan, current_step, all_agents)

    async def _generate_plan(
        self,
        current_step: int,
        world_time: WorldTime,
        world_id: str,
        all_agents: Mapping[str, "Agent"],
        agent_locations: Mapping[str, str],
        space: "SpaceManager",
        entities: Sequence["WorldEntity"],
        *,
        guide: _Guide | None,
    ) -> "_InjectionPlan | None":
        """Gate + generation, returning a plan or None. No side effects: committing lives in
        _commit_plan, keeping background execution apart from deterministic consumption.
        A ``guide`` skips the gate: the heat rule has already decided this beat happens."""
        narrative_summary = await self._build_narrative_summary(
            world_id, current_step, all_agents, agent_locations,
        )
        menu = self._menu(
            current_step, all_agents, agent_locations, space, entities,
            must_include=guide.place_ids if guide is not None else (),
        )
        if guide is None and not await self._passes_llm_check(world_time, narrative_summary, menu):
            return None
        return await self._generate_injection_plan(
            current_step=current_step,
            world_time=world_time,
            narrative_summary=narrative_summary,
            all_agents=all_agents,
            agent_locations=agent_locations,
            menu=menu,
            guide=guide,
        )

    async def _commit_plan(
        self,
        plan: "_InjectionPlan",
        current_step: int,
        all_agents: Mapping[str, "Agent"],
    ) -> CommittedInjection | None:
        """Deliver a plan through the shared dispatcher and record it. If every channel comes up
        empty nothing is recorded, so it costs no quota."""
        committed = await self._dispatcher.dispatch(
            author=Author.SYSTEM,
            step=current_step,
            agents=all_agents,
            narrative_desc=plan.narrative_desc,
            broadcast=plan.broadcast,
            message=plan.message,
            mutations=plan.mutations,
            is_positive=plan.is_positive,
        )
        if committed is not None:
            self._ledger.record(committed.record)
            for place_id in plan.guided_place_ids:
                self._guided_at[place_id] = current_step
        return committed

    async def aclose(self) -> None:
        """Cancel the in-flight generation at shutdown. A finished task is awaited too: skipping
        it on ``done()`` would lose an unread exception that explains why no injections appeared."""
        task = self._inflight
        self._inflight = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 — shutdown cleanup swallows everything
            logger.debug("injection_generation_discarded_on_close", extra={"error": str(exc)})

    async def _passes_llm_check(
        self,
        world_time: WorldTime,
        narrative_summary: str,
        menu: _Menu,
    ) -> bool:
        """Pacing gate: should the world make something happen on its own right now?

        Any gate failure means "don't inject": better to miss a beat than force in an unfounded
        injection.
        """
        system = f"""\
【角色】你是开放世界叙事的节奏编辑,从第三方视角观察故事走向,判断此刻世界该不该发生一件带来重大改变的事。

【输入说明】
- {_RELATIONS_LEGEND}
- 「各地点」是此刻有人的地方和别处的几个地方,「地上的东西」是不在任何人手里、放在这些地方的东西。

【任务】
判断此刻是否需要给这个世界注入一件带来重大改变的事。它应当带来下面这类改变:
{_INJECTION_AIMS}
若故事正自然推进、人物已在自行把事情往前推,则不必干预;宁可不注入,也不要为了填空硬塞一件进来。
对此刻的世界不痛不痒的事,就不要注入。
人都围在一两处打转、别处却放着没人碰过的东西时,这些东西可能就是能借力的地方;但它们只帮你看清有没有
可借力之处,不是注入的理由本身。

【输出】严格输出 JSON,reason 在前、inject 在后,不要任何多余内容:
{{"reason": "当前节奏处于什么状态、为何需要/不需要,不超过 40 字", "inject": true 或 false}}
"""
        user = f"""\
【输入】
当前世界时间:{world_time.time_label}

各地点:
{_bulleted(menu.place_rows)}

地上的东西:
{_bulleted(menu.entity_rows)}

{narrative_summary}

依上面说定的 JSON 格式判断（reason 在前、inject 在后），只输出 JSON、不写任何多余内容。"""
        try:
            response = await self.llm_router.complete(
                LLMScene.NARRATIVE_EDITOR,
                [
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=user),
                ],
                # reason ≤40 chars (60) + inject (3) + 2-field structure (10); ~73 tok.
                # Holds only while the prompt caps reason: truncation makes the gate silently say no.
                max_tokens=output_budget(73),
                temperature=0.3,
            json_mode=True,
            )
            data = extract_json(response.content)
        except Exception as exc:
            note_active_call_adoption(False, "unusable_payload")
            logger.warning("narrative_editor_gate_failed", extra={"error": str(exc)})
            return False
        inject = coerce_bool(data.get("inject"), False)
        if "inject" not in data:
            note_active_call_adoption(False, "missing_inject")
        logger.debug(
            "narrative_editor_gate",
            extra={
                "inject": inject,
                "reason": str(data.get("reason") or "")[:120],
            },
        )
        return inject

    async def _build_narrative_summary(
        self,
        world_id: str,
        current_step: int,
        all_agents: Mapping[str, "Agent"],
        agent_locations: Mapping[str, str],
    ) -> str:
        """The narrative briefing the editor decides from. Characters and relations use the
        latest frame; recent events and injections are a step sequence, because pacing momentum
        exists only in the time series. No step numbers or ids in the text."""
        recent_snapshots: list[WorldSnapshot] = []
        for step in range(max(0, current_step - _BRIEF_LOOKBACK_STEPS), current_step):
            snapshot = await self.snapshot_provider.load(world_id, step)
            if snapshot is not None:
                recent_snapshots.append(snapshot)

        main_ids = {aid for aid, a in all_agents.items() if a.is_main_character}
        latest = recent_snapshots[-1] if recent_snapshots else None
        blocks = [
            self._brief_premise(),
            self._brief_characters(all_agents, agent_locations),
            self._brief_relations(latest, main_ids),
            self._brief_recent(recent_snapshots, main_ids),
            self._brief_prior_injections(),
        ]
        return "\n\n".join(b for b in blocks if b)

    def _brief_premise(self) -> str:
        lines = []
        if self.settings.narrative_theme:
            lines.append(f"故事主题:{self.settings.narrative_theme}")
        if not lines:
            return ""
        return "【故事前提】\n" + "\n".join(lines)

    def _brief_characters(
        self, all_agents: Mapping[str, "Agent"], agent_locations: Mapping[str, str],
    ) -> str:
        """Main-character profiles: externally observable identity and situation only. No
        emotion / needs / goals: the editor applies pressure from outside without reading
        anyone's inner life."""
        main_lines: list[str] = []
        bg_locations: list[str] = []
        bg_count = 0
        for aid, agent in all_agents.items():
            soul = agent.personality.soul
            state = agent.personality.state
            if agent.is_main_character:
                head = soul.identity_text()
                if soul.role:
                    head += f"（{soul.role}）"
                # Situation before the profile, so a long background doesn't push it into the
                # weak-attention middle. The dead get no location / vitality / condition.
                location_id = agent_locations.get(aid)
                now: list[str] = [DECEASED_MARK] if not agent.is_active else [
                    *([f"现位于{self.directory.location_name(location_id)}"] if location_id else []),
                    f"生命值{vitality_label(state.vitality)}",
                ]
                # Without it the editor might broadcast a bound man bursting out of the gate.
                if agent.is_active and (_cond := render_condition(state.condition)):
                    now.append(f"处境{_cond}")
                # Same terms as personality.to_prompt_context / core.prompts.
                attrs: list[str] = []
                if soul.background:
                    attrs.append(f"背景{soul.background}")
                # The editor must know what is hidden to plant clues that point at it.
                if soul.secret:
                    attrs.append(f"{SECRET_LABEL}：{soul.secret}")
                if soul.core_traits:
                    attrs.append(f"性格特点{soul.traits_text()}")
                if soul.core_values:
                    attrs.append(f"价值观{soul.values_text()}")
                if soul.life_goal:
                    attrs.append(f"人生目标{soul.life_goal}")
                # Strip each field's own punctuation, or joining gives "。；".
                profile = f"{'；'.join(strip_end_punct(a) for a in attrs)}。" if attrs else ""
                main_lines.append(f"- {head}：{'；'.join(now)}。{profile}")
            elif agent.is_active:
                bg_count += 1
                # Counted but not placed when unplaced: he is alive, only nowhere to name.
                if where := agent_locations.get(aid):
                    bg_locations.append(self.directory.location_name(where))

        parts: list[str] = []
        if main_lines:
            parts.append("主角:\n" + "\n".join(main_lines))
        if bg_count:
            dist = "、".join(f"{loc}{n}人" for loc, n in Counter(bg_locations).items())
            parts.append(f"背景角色:共{bg_count}人" + (f"，分布于 {dist}" if dist else ""))
        if not parts:
            return ""
        return "【角色】\n" + "\n".join(parts)

    def _brief_relations(
        self,
        latest: WorldSnapshot | None,
        main_ids: set[str],
    ) -> str:
        """Labeled relations from the latest frame, main-character pairs first. No trust /
        affection numbers: those are first-person perception the editor shouldn't read."""
        if latest is None or not latest.state.relations:
            return ""
        scored: list[tuple[int, str]] = []
        for rel in latest.state.relations.values():
            labels = rel.get("labels") or []
            if not labels:
                continue
            from_id = str(rel.get("from_id", ""))
            to_id = str(rel.get("to_id", ""))
            from_name = self.directory.agent_name(from_id)
            to_name = rel.get("to_name") or self.directory.agent_name(to_id)
            desc = rel.get("history_summary") or ""
            line = f"- {from_name} 对 {to_name}:{'、'.join(labels)}"
            if desc:
                line += f"（{desc}）"
            priority = 0 if (from_id in main_ids or to_id in main_ids) else 1
            scored.append((priority, line))
        if not scored:
            return ""
        scored.sort(key=lambda t: t[0])
        rows = [row for _, row in scored[:_BRIEF_MAX_RELATIONS]]
        return "【当前关系】(最新关系与标签)\n" + "\n".join(rows)

    def _brief_recent(self, recent_snapshots: list[WorldSnapshot], main_ids: set[str]) -> str:
        """Recent events per step: main-character deeds (``_step_deeds``), world broadcasts, and
        environment traces not caused by a main character (those duplicate the deed's outcome).
        """
        if not recent_snapshots:
            return "【近况】\n叙事刚刚开始,尚无足够信息。"
        lines: list[str] = []
        for snapshot in recent_snapshots:
            when = snapshot.time_label
            for action in _step_deeds(snapshot):
                initiator = str(action.get("initiator_id") or "")
                # The snapshot's own name for him: it is whole, and names are frozen at build.
                name = str(snapshot.state.agents.get(initiator, {}).get("agent_name") or "某人")
                where = self.directory.location_name(str(action.get("location_id") or ""))
                # The initiator's own part speaks for the deed.
                lead = next(
                    (p for p in action.get("parts") or [] if p.get("agent_id") == initiator), {},
                )
                outcome = lead.get("outcome") or lead.get("action_description") or ""
                if outcome:
                    lines.append(f"- {when}·{where}:{name} {outcome}")
            for bc in snapshot.events_of("broadcast"):
                content = bc.get("content")
                if not content:
                    continue
                scope = bc.get("location_scope")
                where = self.directory.location_name(str(scope)) if scope else "全域"
                lines.append(f"- {when}·{where}广播:{content}")
            annotations = snapshot.engine.environment.get("step_annotations")
            for scope, scope_events in (annotations or {}).items():
                where = self.directory.location_name(str(scope)) if scope else ""
                for ev in scope_events or []:
                    if not isinstance(ev, Mapping):
                        continue
                    if any(a in main_ids for a in (ev.get("doer_ids") or [])):
                        continue
                    content = ev.get("content")
                    if content:
                        prefix = f"{when}·{where}:" if where else f"{when}·"
                        lines.append(f"- {prefix}{content}")
        if not lines:
            return "【近况】\n近期无显著进展。"
        return "【近况】\n" + "\n".join(lines)

    def _brief_prior_injections(self) -> str:
        """The last few injections with whom they touched and where, so the editor avoids
        repeating people / place / trope. Only the last few: the ledger grows without bound."""
        if not len(self._ledger):
            return ""
        rows: list[str] = []
        for record in self._ledger.recent(_BRIEF_MAX_PRIOR_INJECTIONS):
            if not record.narrative_desc:
                continue
            tags: list[str] = []
            if record.affected_names:
                tags.append(f"波及{'、'.join(record.affected_names)}")
            if record.location_label:
                tags.append(f"在{record.location_label}")
            suffix = f"（{'；'.join(tags)}）" if tags else ""
            rows.append(f"- {record.narrative_desc}{suffix}")
        if not rows:
            return ""
        return "【已注入事件】(勿对同一批人 / 同一地点 / 同类梗重复)\n" + "\n".join(rows)

    async def _generate_injection_plan(
        self,
        *,
        current_step: int,
        world_time: WorldTime,
        narrative_summary: str,
        all_agents: Mapping[str, "Agent"],
        agent_locations: Mapping[str, str],
        menu: _Menu,
        guide: _Guide | None,
    ) -> _InjectionPlan | None:
        """The LLM generates an injection and declares its channels (spawn / alter / destroy /
        broadcast / message). With a ``guide`` the beat only tells characters what is at those
        places: spawn / alter / destroy don't apply and are ignored, and a plan no character
        perceives is dropped.

        The editor sees beyond where people stand: the menu is every occupied place plus some
        places elsewhere (``_menu_places``), and the unheld entities on them. The gate reads the
        same menu (``_Menu``), so it judges what this plan can reach.
        Content is steered only by failure modes (§2), never positive prescriptions.
        """
        # Only the living: a message to the dead is dropped yet still recorded as injected.
        living = {aid: a for aid, a in all_agents.items() if a.is_active}
        if not living:
            return None

        agent_id_list = list(living.keys())
        # Gender: background characters appear only here, and messages address the recipient.
        # Location: the plan must keep who-is-where straight, and the briefing has it only for
        # main characters.
        agent_lines = "\n".join(
            f"#{i + 1} {self._person_line(a, agent_locations.get(aid))}"
            for i, (aid, a) in enumerate(living.items())
        )
        location_ref = IndexedRef(loc.place_id for loc in menu.places)
        location_lines = _numbered(menu.place_rows)
        entity_ref = IndexedRef(e.entity_id for e in menu.entities)
        entity_lines = _numbered(menu.entity_rows)

        system = f"""\
【角色】你是开放世界叙事的节奏编辑,从第三方视角设计世界此刻自己发生的一件事。它要从下方的
故事前提、人物与近况里长出来,并给这个世界带来**重大改变**:
{_INJECTION_AIMS}

【输入说明】{_RELATIONS_LEGEND}

【硬约束】(必守)
1. 你能直接改变的只有地上的东西:在某个地点放下一件新东西,或改变 / 毁掉「地上的东西」
   名单里的一件。人的生死、伤病、去向,以及任何人手里的东西,你一概不能动 —— 想表达
   「某人死亡/受伤」只能描述他人感知到此事,其在机制上仍存在。
2. 事件必须从下方简报的前提、人物、关系、近况里生长出来,不要凭空降下与本世界无关的桥段。
3. 你改变的是人物的**处境**,不是故事的**结局**——不要用事件直接了结张力或替人物做决定。
4. 引用一律用序号:recipients 用人物序号,location_scope 与 spawn.location 用地点序号,
   alter.entity 与 destroy.entity 用物件序号,不要写名字或 id。
5. 投送的事件内容要符合时间和空间逻辑常理。比如不能把两个不在同一个地方的人说成在同一个地方对峙等等。
   你看得见各处的情形,人物只知道自己身边的事;在此刻无人的地点广播,没有人感知得到。
6. 这个世界只由下方简报定义:
{CLOSED_WORLD_FACT_RULE}
7. 你写下的每一段话(各处的 content、observation、narrative_desc)都会被人记住、日后反复读到,
   所以提到将来的时点时:
{ABSOLUTE_TIME_RULE}

【任务】设计最能推动当前叙事的一件事,选用下面的通道(至少一个,可组合):
- spawn:在某个地点放下一件世界里原本没有的东西。它会真实存在于那里,在场的人看得见。
- alter:改变「地上的东西」名单里一件东西的状态、样子或上面写的内容。
- destroy:毁掉「地上的东西」名单里的一件。
- broadcast:世界级公开广播,全域或某地点的人都会感知到——可播报环境变化,也可播报
  世界级消息/动静。
- message:匿名定向投递给特定人物的内容，不会附带任何角色身份；正文不要出现编辑器或系统等幕后身份。
spawn / alter / destroy 每拍各至多一件。你投出去的话里若有一件东西出现、被改或被毁,
就用对应的一项让它真的发生 —— 否则人们会去找一件根本不存在的东西。
新东西只能落在某个地点上,不会凭空到谁手里:要让某人得到它,就放在他所在的地方,
别写成「送到他手中」。

【引导前往】输入末尾若给出「这一拍要引导人物前往的地方」(一处或几处),说明那些地方地上有东西,
最近却很少有人物去过。这一拍不设计事件,只把这些地方的情形告诉你判断合适的人物;告诉谁、怎么告知
由你定,听到之后去不去、在不在意,由他们自己决定。
- 投给谁:依据人物的处境、目标与关系,判断这些地方对谁可能有用,用 message 告诉那一个或几个人物;
  也可以用全域 broadcast(location_scope 留空)让所有人知道。几处地方可以分在两个通道里说,
  告诉不同的人。不要把 broadcast 限在这些没有人物的地方，因为即使 broadcast 谁也感知不到。
- 说什么:每一处都说到,并用上面给出的地点名称呼它 —— 是哪个地方，那里有什么，是什么情况。
  东西的样子和上面写的字照给出的原话转述，不要补上没有的内容和描述等等;不添、不改、不编,
  也不替人物判断值不值得去。
- 怎么说:像平常告知一件事情一样,不故弄玄虚,不制造悬念,不用「似乎」「据说」这类含糊的说法。
- 这一拍 spawn / alter / destroy 一律填 null;开头「带来重大改变」的要求和下面避免清单里「不痛不痒」
  那一条不适用于这一拍。

【避免】(失败模式,逐条规避)
- 笼统抽象、等于没说的事件，必须给出具体的事情、人物，事件，变化等等。
- 与故事前提 / 人物身份脱钩、凭空降下的通用桥段
- 与【已注入事件】里同一批人、同一地点或同类手法的重复
- 「地上的东西」可以直接通过变更就可以成为自己想要的东西，但却另造一件相似的新东西。
- 一锤定音直接化解张力的天降转机
- 用造出、改动或毁掉一件东西来直接了结整个叙事的张力(凭空出现的铁证、烧掉那份决定一切的文书)
- 不痛不痒、写与不写都不会给现在的世界带来实质性改变的事。
- 把伤亡写成既成的机制事实,而非他人「感知到」
- 喧宾夺主,盖过人物自身的自主行动
- message 内容用第三人称指称收件人(它是直接投递给对方的)
- broadcast 把 observation 已经说过的同一件事再说一遍(在场的人会感知两遍)

【输出】严格输出 JSON,不要任何多余内容。字段按设计成形的先后排列,照这个次序往下写:
先在 reason 里分析此刻局势、想清楚该做什么、为什么 → 定这一拍的基调(is_positive) → 先让地上的东西真的变
(spawn / alter / destroy,广播与消息往往在讲它) → **先指认投给哪里 / 投给谁(序号)、再写投出去的内容**
(定向消息是直接送到收件人面前的,不先定收件人就写不出对他说的话) → 最后回头把这一拍已经投出去的东西概括成 narrative_desc。
字段说明:is_positive 仅 true/false/null;entity_type 仅 item(拿得走的物品)或 landmark(拿不走的固着物);
observation 只写这件东西本身看得见的变化，不写哪些人看到;
{SEVERITY_SCALE_DESCRIPTION};{URGENCY_SCALE_DESCRIPTION};
{PHENOMENON_DEFINITION}
{{
  "reason": "先想清楚:此刻这个故事最需要怎样的一拍、为什么 —— 据此再设计下面的事件,不超过 60 字",
  "is_positive": true 或 false 或 null,
  "spawn": {{"location": 放在哪个地点(地点序号), "entity_type": "item 或 landmark", "name": "它叫什么,不超过 12 字", "description": "它看上去是什么样,不超过 40 字", "content": "只有它本身承载文字或记录时才填,写它上面写着 / 记着的东西,要有具体、实质内容,不超过 80 字;其余一律留空字符串", "observation": "这件东西本身的变化,不写人,不超过 40 字"}} 或 null,
  "alter": {{"entity": 物件序号, "state": "变化后的状态词,不超过 12 字,不变就留空", "description": "变化后它看上去的样子,不超过 40 字,不变就留空", "content": "改动后上面写着的全文,要有具体、实质内容,不超过 80 字,不变就留空", "observation": "这件东西本身的变化,不写人,不超过 40 字"}} 或 null,
  "destroy": {{"entity": 物件序号, "observation": "这件东西本身的变化,不写人,不超过 40 字"}} 或 null,
  "broadcast": {{"location_scope": 发生地的地点序号(笼罩全域就填 null,或整个不写这一项), "content": "广播内容,不超过 80 字", "severity": "{Severity.prompt_choices()}", "phenomenon": "{Phenomenon.prompt_choices()}"}} 或 null,
  "message": {{"recipients": [人物序号，最多10个], "content": "定向内容,不超过 80 字", "urgency": "low|normal|high|critical"}} 或 null,
  "narrative_desc": "事件总览(供回放):概括上面这一拍实际做了什么,不超过 60 字"
}}
"""
        user = f"""\
【输入】
当前世界时间:{world_time.time_label}

可选地点(location_scope 与 spawn.location 用其序号;location_scope 留空则全域):
{location_lines}

可选人物(recipients 用其序号):
{agent_lines}

地上的东西(alter / destroy 用其序号;只有这些能改、能毁):
{entity_lines}

{narrative_summary}
{self._guide_block(guide, menu)}
依上面说定的 JSON 格式设计事件（reason 在前），只输出 JSON、不写任何多余内容。"""

        try:
            response = await self.llm_router.complete(
                LLMScene.NARRATIVE_EDITOR,
                [
                    LLMMessage(role="system", content=system),
                    LLMMessage(role="user", content=user),
                ],
                # reason ≤60 chars (90) + is_positive (3)
                # + spawn{location 3 + type 3 + name ≤12 chars 18 + description ≤40 chars 60
                #   + content ≤80 chars 120 + observation ≤40 chars 60 + 6 fields 30} = 294
                # + alter{entity 3 + state ≤12 chars 18 + description 60 + content 120
                #   + observation 60 + 5 fields 25} = 286
                # + destroy{entity 3 + observation 60 + 2 fields 10} = 73
                # + broadcast{scope 3 + content ≤80 chars 120 + severity 3 + phenomenon 3
                #   + 4 fields 20} = 149
                # + message{≤10 recipients ×3 =30 + content ≤80 chars 120 + urgency 3 + 3 fields 15} = 168
                # + narrative_desc ≤60 chars (90) + top-level 8-field structure (40);
                #   ~1193 tok
                max_tokens=output_budget(1193),
                temperature=0.8,
            json_mode=True,
            )
        except Exception as exc:
            logger.warning(
                "injection_plan_llm_failed",
                extra={"step": current_step, "error": str(exc)},
            )
            return None

        try:
            data = extract_json(response.content)
        except Exception as exc:
            note_active_call_adoption(False, "invalid_json")
            logger.warning(
                "injection_plan_parse_failed",
                extra={"step": current_step, "error": str(exc)},
            )
            return None

        plan = self._parse_plan(data, agent_id_list, location_ref, entity_ref)
        if plan is None:
            note_active_call_adoption(False, "invalid_plan")
            logger.warning("injection_plan_invalid", extra={"step": current_step})
            return None
        if plan.broadcast is None and plan.message is None and not plan.mutations:
            note_active_call_adoption(False, "no_dispatchable_channel")
            logger.warning("injection_plan_no_valid_channel", extra={"step": current_step})
            return None
        if guide is None:
            return plan
        plan = replace(plan, spawn=None, alter=None, destroy=None)
        if not _reaches_someone(plan, living, agent_locations):
            note_active_call_adoption(False, "guide_unperceived")
            logger.warning(
                "injection_guide_unperceived",
                extra={"step": current_step, "location_ids": list(guide.place_ids)},
            )
            return None
        told = self._told_places(plan, guide)
        if not told:
            note_active_call_adoption(False, "guide_untold")
            logger.warning(
                "injection_guide_untold", extra={"step": current_step, "location_ids": list(guide.place_ids)},
            )
            return None
        if len(told) < len(guide.place_ids):
            logger.info(
                "injection_guide_partly_told",
                extra={
                    "step": current_step,
                    "told": list(told),
                    "untold": [p for p in guide.place_ids if p not in told],
                },
            )
        return replace(plan, guided_place_ids=told)

    def _told_places(self, plan: "_InjectionPlan", guide: _Guide) -> tuple[str, ...]:
        """The guide's places whose name appears in what the plan says.

        Only these are recorded as guided: an untold place stays in the pick, rather than being
        kept out for _GUIDE_REPEAT_STEPS though nobody heard of it. Matching is on the exact name,
        the one the plan was shown and asked to use, so it errs toward untold: an alias or
        shortening reads as untold, and the place may be named again at the next check, which
        costs one beat at most. It reads a place as told by mistake only when its name sits inside
        another place's name that was said; an LLM judge would catch both, but not worth a call
        for bookkeeping.
        """
        said = " ".join(c.content for c in (plan.message, plan.broadcast) if c is not None)
        return tuple(p for p in guide.place_ids if self.directory.location_name(p) in said)

    def _guide_block(self, guide: _Guide | None, menu: _Menu) -> str:
        """The places a guide names and what is on the ground there, for the end of the plan's
        user message; "" without a guide."""
        if guide is None:
            return ""
        index = {loc.place_id: i for i, loc in enumerate(menu.places, 1)}
        sections = []
        for place_id in guide.place_ids:
            things = [row for e, row in zip(menu.entities, menu.entity_rows) if e.location_id == place_id]
            sections.append(
                f"可选地点 #{index[place_id]} {self.directory.location_name(place_id)},地上的东西:\n"
                f"{_bulleted(things)}"
            )
        joined = "\n".join(sections)
        return f"""
这一拍要引导人物前往的地方:
{joined}
"""

    def _parse_plan(
        self,
        data: dict[str, Any],
        agent_id_list: list[str],
        location_ref: IndexedRef,
        entity_ref: IndexedRef,
    ) -> _InjectionPlan | None:
        if not isinstance(data, dict):
            return None
        narrative_desc = str(data.get("narrative_desc", "")).strip()
        if not narrative_desc:
            return None

        is_positive = normalize_is_positive(data.get("is_positive"))
        broadcast = parse_broadcast_spec(data.get("broadcast"), location_ref)
        message = parse_message_spec(data.get("message"), agent_id_list)

        return _InjectionPlan(
            narrative_desc=narrative_desc,
            is_positive=is_positive,
            broadcast=broadcast,
            message=message,
            spawn=parse_spawn(data.get("spawn"), location_ref),
            alter=parse_entity_mutation(data.get("alter"), entity_ref, destroyed=False),
            destroy=parse_entity_mutation(data.get("destroy"), entity_ref, destroyed=True),
        )

    def _person_line(self, agent: "Agent", location_id: str | None) -> str:
        soul = agent.personality.soul
        # An unplaced body has no place to name (see EnvironmentSystem.body_locations).
        where = f"在{self.directory.location_name(location_id)}" if location_id else ""
        return person_referent(soul.name, soul.gender, where)

    def _menu(
        self,
        current_step: int,
        agents: Mapping[str, "Agent"],
        agent_locations: Mapping[str, str],
        space: "SpaceManager",
        entities: Sequence["WorldEntity"],
        *,
        must_include: Collection[str] = (),
    ) -> _Menu:
        """The menu one check's gate and plan both read (see ``_Menu``)."""
        living = [aid for aid, a in agents.items() if a.is_active]
        headcount = Counter(agent_locations.get(aid) for aid in living)
        on_ground = [e for e in entities if e.location_id is not None]
        places = _menu_places(
            space.all_places(), headcount, Counter(e.location_id for e in on_ground),
            random.Random(current_step), must_include,
        )
        listed = {loc.place_id for loc in places}
        grounded = [e for e in on_ground if e.location_id in listed]
        return _Menu(
            places=places,
            entities=grounded,
            place_rows=[
                f"{render_location(space.view_of(loc.place_id))}"
                f"（{f'此刻{headcount[loc.place_id]}人' if headcount[loc.place_id] else '此刻无人'}）"
                for loc in places
            ],
            entity_rows=[self._entity_text(e) for e in grounded],
        )

    def _entity_text(self, entity: "WorldEntity") -> str:
        """One entity on the ground in the menu. Written content is shown only for fixed entities
        (as ``WorldEntity.readable_by``): otherwise the editor could broadcast words nobody
        present can read."""
        facts = [f for f in (
            f"此刻状态：{entity.state}" if entity.state else "",
            f"现于{self.directory.location_name(entity.location_id)}" if entity.location_id else "",
        ) if f]
        lines = [entity.name + (f"（{'，'.join(facts)}）" if facts else "")]
        if entity.description:
            lines.append(f"    {entity.description}")
        if entity.content and not entity.is_takeable:
            lines.append(f"    上面写着：{entity.content}")
        return "\n".join(lines)
