"""Sparse narrative event injection: the automatic LLM editor.

Design charter:
- Only pacing + channel adaptation. Never mutates agent state; agents react to the perception
  signals on their own.
- No event type enums: the LLM declares the channels (broadcast / message / object changes)
  in its plan, possibly several at once.
- Object changes go through ``WorldMutationChannel`` as ``Author.SYSTEM``, which only allows
  putting a new thing down or changing / destroying an unheld thing on the ground. People and
  what they hold are the human director's only (``engine/director.py``): an automatic LLM
  must not overturn results an agent has a causal claim to. The permission exists so the world
  follows what the editor's broadcasts say appeared or vanished.
- Delivery goes through the ``InjectionDispatcher`` shared with the director
  (``engine/injection.py``); a new communication pattern needs its own channel module first.
- Ledgers are separate from the director's, so the quota only constrains the editor and the
  editor's briefing never learns someone reached in from outside the world.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from agent.personality import SECRET_LABEL
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
    SEVERITY_SCALE_DESCRIPTION,
    URGENCY_SCALE_DESCRIPTION,
    person_referent,
    render_condition,
    render_location,
    strip_end_punct,
    vitality_label,
)
from core.logging import get_logger
from engine.clock import WorldTime
from engine.injection import (
    Author,
    BroadcastSpec,
    CommittedInjection,
    InjectionDispatcher,
    InjectionLedger,
    MessageSpec,
    WorldEvent,
    normalize_is_positive,
    parse_broadcast_spec,
    parse_message_spec,
)
from engine.world_mutation import EntityMutation, Mutation, SpawnMutation, parse_spawn

if TYPE_CHECKING:
    from agent.agent import Agent
    from world.models import WorldEntity
    from core.interfaces.place import Place

logger = get_logger(__name__)


# Narrative briefing bounds (the gate also consumes this briefing, so it must stay bounded).
_BRIEF_LOOKBACK_STEPS = 6        # steps the dynamic "recent events" block looks back
_BRIEF_MAX_RELATIONS = 8         # max relation pairs listed (main-character pairs first)
_BRIEF_MAX_PRIOR_EVENTS = 5      # max most-recent injected events listed

# Unoccupied places the plan menu may list on top of the occupied ones. A few elsewhere are
# enough to pull the story out of one spot; more only dilutes the menu.
_MAX_STOCKED_PLACES = 5          # with unheld things on the ground, most things first
_MAX_BARE_PLACES = 3             # with nothing on them, a different draw each step


# Not shared with the director's DirectivePlan: composing a beat and translating a sentence
# are different products, even if they look alike.


@dataclass(frozen=True)
class _EventPlan:
    narrative_desc: str
    is_positive: bool | None
    broadcast: BroadcastSpec | None
    message: MessageSpec | None
    spawn: SpawnMutation | None = None
    alter: EntityMutation | None = None
    destroy: EntityMutation | None = None

    @property
    def mutations(self) -> list[Mutation]:
        return [m for m in (self.spawn, self.alter, self.destroy) if m is not None]


@dataclass(frozen=True)
class EventSettings:
    """Pacing and narrative anchors for editing one world.

    ``core_tension`` / ``narrative_theme`` go only into the editor's briefing, never into agent
    cognition. The quota is a sliding window (``max_events_per_window`` per ``quota_window``
    steps): it bounds density, not a lifetime total that a long world would spend early.
    """

    core_tension: str = ""
    narrative_theme: str = ""
    enabled: bool = True
    max_events_per_window: int = 2
    quota_window: int = 20
    check_interval: int = 5

    def __post_init__(self) -> None:
        if self.check_interval <= 0:
            raise ValueError("check_interval must be positive")
        if self.max_events_per_window < 0:
            raise ValueError("max_events_per_window must be non-negative")
        if self.quota_window <= 0:
            raise ValueError("quota_window must be positive")


def _step_deeds(actions: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """This step's action records eligible for the briefing: filter out non-deeds, then fold
    a joint action's per-participant records into the initiator's (by ``execution_id``), or
    the same dialogue would double the density the pacing gate sees.

    Fold after filtering: a group whose initiator was filtered out falls back to its first
    record rather than vanishing. Fold only within this step: a multi-step execution's start
    and end beats both belong.
    """
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for index, action in enumerate(actions):
        if not action.get("is_main_character"):
            continue
        # Progress notes aren't deeds; they'd repeat one act per step it spans.
        if action.get("phase") == "ongoing_tick":
            continue
        # Never engaged the world (precondition failed); the editor would react to something
        # that never happened. An adjudicated failure stays: "he tried and was beaten" is a deed.
        if action.get("not_executed"):
            continue
        # No execution_id: its own group, never merged.
        key: Any = str(action.get("execution_id") or "") or index
        grouped.setdefault(key, []).append(action)
    return [
        next((r for r in records if r.get("agent_id") == r.get("initiator_id")), records[0])
        for records in grouped.values()
    ]


def _menu_places(
    locations: Sequence["Place"],
    headcount: Mapping[str, int],
    things: Mapping[str, int],
    rng: random.Random,
) -> list["Place"]:
    """Places the plan may name, in ``locations`` order: every occupied place (the only ones a
    broadcast reaches), the unoccupied places with the most unheld things on the ground, and a few
    bare ones drawn by ``rng``. Don't pick the bare ones by a fixed rule such as nearness: while
    people stay put the same few would always win, and the rest of the map could never host
    an event."""
    unoccupied = [loc for loc in locations if not headcount.get(loc.place_id)]
    stocked = sorted(
        (loc for loc in unoccupied if things.get(loc.place_id)),
        key=lambda loc: -things[loc.place_id],
    )[:_MAX_STOCKED_PLACES]
    bare = [loc for loc in unoccupied if not things.get(loc.place_id)]
    drawn = rng.sample(bare, min(len(bare), _MAX_BARE_PLACES))
    chosen = {loc.place_id for loc in (*stocked, *drawn)}
    return [loc for loc in locations if headcount.get(loc.place_id) or loc.place_id in chosen]


def _observation(raw: Mapping[str, Any]) -> str:
    return str(raw.get("observation", "")).strip()


def _parse_alter(raw: Any, entity_ref: IndexedRef) -> EntityMutation | None:
    """An alter that changes nothing (all three fields empty) isn't a change; drop it."""
    if not isinstance(raw, dict):
        return None
    which = entity_ref.resolve([raw.get("entity")])
    observation = _observation(raw)
    state = str(raw.get("state", "")).strip()
    description = str(raw.get("description", "")).strip()
    content = str(raw.get("content", "")).strip()
    if not which or not observation or not (state or description or content):
        return None
    return EntityMutation(
        observation=observation,
        entity_id=which[0],
        new_state=state,
        new_description=description,
        new_content=content,
    )


def _parse_destroy(raw: Any, entity_ref: IndexedRef) -> EntityMutation | None:
    if not isinstance(raw, dict):
        return None
    which = entity_ref.resolve([raw.get("entity")])
    observation = _observation(raw)
    if not which or not observation:
        return None
    return EntityMutation(observation=observation, entity_id=which[0], destroyed=True)


class EventSystem:
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
        settings: EventSettings,
    ) -> None:
        self.llm_router = llm_router
        self.snapshot_provider = snapshot_provider
        self._dispatcher = dispatcher
        self.directory = directory
        self.settings = settings
        self._ledger = InjectionLedger(Author.SYSTEM)
        # At most one background generation at a time: bounds cost and prevents quota races
        # (the ledger is written only at commit).
        self._inflight: "asyncio.Task[_EventPlan | None] | None" = None

    def list_events(self) -> list[WorldEvent]:
        return self._ledger.all()

    def restore_state(self, fired_events: Iterable[Mapping[str, Any]]) -> None:
        """Rehydrate the quota window from persisted events, so a restore doesn't grant a fresh
        quota. The ledger keeps only this system's events; the director's are filtered out."""
        self._ledger.restore(fired_events)

    def _passes_rule_check(self, current_step: int) -> bool:
        settings = self.settings
        if not settings.enabled:
            return False
        if current_step <= 0 or current_step % settings.check_interval != 0:
            return False
        if self._ledger.count_since(current_step - settings.quota_window) >= settings.max_events_per_window:
            return False
        return True

    async def poll_event(
        self,
        *,
        current_step: int,
        world_time: WorldTime,
        world_id: str,
        all_agents: Mapping[str, "Agent"],
        locations: Sequence["Place"],
        entities: Sequence["WorldEntity"],
    ) -> CommittedInjection | None:
        """The step loop's single event entry point: commit a finished background generation
        and return it, or start one and return None. Never blocks on the LLM.

        ``locations`` / ``entities`` are read-only (the author layer doesn't hold EnvironmentSystem).
        """
        ready = await self._consume_ready(current_step, all_agents)
        if ready is not None:
            return ready
        self._ensure_generation(current_step, world_time, world_id, all_agents, locations, entities)
        return None

    def _ensure_generation(
        self,
        current_step: int,
        world_time: WorldTime,
        world_id: str,
        all_agents: Mapping[str, "Agent"],
        locations: Sequence["Place"],
        entities: Sequence["WorldEntity"],
    ) -> None:
        """If eligible and nothing is in flight, start "gate + plan generation" in the
        background, off the step loop's critical path."""
        if self._inflight is not None:
            return
        if not self._passes_rule_check(current_step):
            return
        self._inflight = asyncio.create_task(
            self._generate_plan(current_step, world_time, world_id, all_agents, locations, entities)
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
            logger.warning("event_generation_failed", extra={"error": str(exc)})
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
        locations: Sequence["Place"],
        entities: Sequence["WorldEntity"],
    ) -> "_EventPlan | None":
        """Gate + generation, returning a plan or None. No side effects: committing lives in
        _commit_plan, keeping background execution apart from deterministic consumption."""
        narrative_summary = await self._build_narrative_summary(
            world_id, current_step, all_agents
        )
        should_inject = await self._passes_llm_check(world_time, narrative_summary)
        if not should_inject:
            return None
        return await self._generate_event_plan(
            current_step=current_step,
            world_time=world_time,
            narrative_summary=narrative_summary,
            all_agents=all_agents,
            locations=locations,
            entities=entities,
        )

    async def _commit_plan(
        self,
        plan: "_EventPlan",
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
            self._ledger.record(committed.event)
        return committed

    async def aclose(self) -> None:
        """Cancel the in-flight generation at shutdown. A finished task is awaited too: skipping
        it on ``done()`` would lose an unread exception that explains why no events appeared."""
        task = self._inflight
        self._inflight = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 — shutdown cleanup swallows everything
            logger.debug("event_generation_discarded_on_close", extra={"error": str(exc)})

    async def _passes_llm_check(
        self,
        world_time: WorldTime,
        narrative_summary: str,
    ) -> bool:
        """Pacing gate: should the world make something happen on its own right now?

        Any gate failure means "don't inject": better to miss a beat than force in an unfounded
        event.
        """
        system = """\
【角色】你是开放世界叙事的节奏编辑,从第三方视角观察故事走向,判断此刻世界该不该发生一件带来重大改变的事。

【任务】
判断此刻是否需要给这个世界注入一件带来重大改变的事。它应当做到下面四件事之一:
- 使叙事丰富多彩:向平淡或流水账的世界中注入可能引起故事往不同方向推进、或产生不同叙事内容的事件。
- 升华或加深叙事:把已经达到临界点的内容进一步升华或点燃。
- 重大转折叙事:注入可能引起叙事的内容或方向发生重大转折、反转的事件。
- 打破僵局:打破当下叙事原地打转、反复循环，无实质进展的局面。比如某些角色一直在反复做一类事情并且没有有效推进的情况。
若故事正自然推进、人物已在自行把事情往前推,则不必干预;宁可不注入,也不要为了填空硬塞一件进来。
对此刻的世界不痛不痒的事,就不要注入。

【输出】严格输出 JSON,reason 在前、inject 在后,不要任何多余内容:
{"reason": "当前节奏处于什么状态、为何需要/不需要,不超过 40 字", "inject": true 或 false}
"""
        user = f"""\
【输入】
当前世界时间:{world_time.time_label}

{narrative_summary}

依上面说定的 JSON 格式判断（reason 在前、inject 在后），只输出 JSON、不写任何多余内容。"""
        try:
            response = await self.llm_router.complete(
                LLMScene.EVENT_TIMING,
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
            logger.warning("event_timing_gate_failed", extra={"error": str(exc)})
            return False
        inject = coerce_bool(data.get("inject"), False)
        logger.debug(
            "event_timing_gate",
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
    ) -> str:
        """The narrative briefing the editor decides from. Characters and relations use the
        latest frame; recent and injected events are a step sequence, because pacing momentum
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
            self._brief_characters(all_agents),
            self._brief_relations(latest, main_ids),
            self._brief_recent(recent_snapshots, main_ids),
            self._brief_prior_events(),
        ]
        return "\n\n".join(b for b in blocks if b)

    def _brief_premise(self) -> str:
        lines = []
        if self.settings.core_tension:
            lines.append(f"核心张力:{self.settings.core_tension}")
        if self.settings.narrative_theme:
            lines.append(f"故事主题:{self.settings.narrative_theme}")
        if not lines:
            return ""
        return "【故事前提】\n" + "\n".join(lines)

    def _brief_characters(self, all_agents: Mapping[str, "Agent"]) -> str:
        """Main-character profiles: externally observable identity and situation only. No
        emotion / needs / goals: the editor applies pressure from outside without reading
        anyone's inner life."""
        main_lines: list[str] = []
        bg_locations: list[str] = []
        for agent in all_agents.values():
            soul = agent.personality.soul
            state = agent.personality.state
            if agent.is_main_character:
                head = soul.identity_text()
                if soul.role:
                    head += f"（{soul.role}）"
                # Situation before the profile, so a long background doesn't push it into the
                # weak-attention middle. The dead get no location / vitality / condition.
                now: list[str] = [DECEASED_MARK] if not agent.is_active else [
                    f"现位于{self.directory.location_name(state.current_location)}",
                    f"体力{vitality_label(state.vitality)}",
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
                    attrs.append(f"核心性格{soul.traits_text()}")
                if soul.core_values:
                    attrs.append(f"核心价值观{soul.values_text()}")
                if soul.life_goal:
                    attrs.append(f"毕生追求{soul.life_goal}")
                # Strip each field's own punctuation, or joining gives "。；".
                profile = f"{'；'.join(strip_end_punct(a) for a in attrs)}。" if attrs else ""
                main_lines.append(f"- {head}：{'；'.join(now)}。{profile}")
            elif agent.is_active:
                bg_locations.append(self.directory.location_name(state.current_location))

        parts: list[str] = []
        if main_lines:
            parts.append("主要人物:\n" + "\n".join(main_lines))
        if bg_locations:
            dist = "、".join(f"{loc}{n}人" for loc, n in Counter(bg_locations).items())
            parts.append(f"背景人物:共{len(bg_locations)}人，分布于 {dist}")
        if not parts:
            return ""
        return "【主要人物】\n" + "\n".join(parts)

    def _brief_relations(
        self,
        latest: WorldSnapshot | None,
        main_ids: set[str],
    ) -> str:
        """Labeled relations from the latest frame, main-character pairs first. No trust /
        affection numbers: those are first-person perception the editor shouldn't read."""
        if latest is None or not latest.agent_relations:
            return ""
        scored: list[tuple[int, str]] = []
        for rel in latest.agent_relations.values():
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
            for action in _step_deeds(snapshot.actions_this_step):
                name = action.get("agent_name") or "某人"
                where = self.directory.location_name(str(action.get("location_id") or ""))
                outcome = (
                    action.get("outcome")
                    or action.get("summary")
                    or action.get("action_description")
                    or ""
                )
                if outcome:
                    lines.append(f"- {when}·{where}:{name} {outcome}")
            for bc in snapshot.metadata.get("broadcasts", []) or []:
                if not isinstance(bc, Mapping):
                    continue
                content = bc.get("content")
                if not content:
                    continue
                scope = bc.get("location_scope")
                where = self.directory.location_name(str(scope)) if scope else "全域"
                lines.append(f"- {when}·{where}广播:{content}")
            env = snapshot.metadata.get("environment") or {}
            annotations = env.get("step_annotations") if isinstance(env, Mapping) else None
            for scope, scope_events in (annotations or {}).items():
                where = self.directory.location_name(str(scope)) if scope else ""
                for ev in scope_events or []:
                    if not isinstance(ev, Mapping):
                        continue
                    if any(a in main_ids for a in (ev.get("actor_ids") or [])):
                        continue
                    content = ev.get("content")
                    if content:
                        prefix = f"{when}·{where}:" if where else f"{when}·"
                        lines.append(f"- {prefix}{content}")
        if not lines:
            return "【近况】\n近期无显著进展。"
        return "【近况】\n" + "\n".join(lines)

    def _brief_prior_events(self) -> str:
        """The last few injections with whom they touched and where, so the editor avoids
        repeating people / place / trope. Only the last few: the ledger grows without bound."""
        if not len(self._ledger):
            return ""
        rows: list[str] = []
        for ev in self._ledger.recent(_BRIEF_MAX_PRIOR_EVENTS):
            if not ev.narrative_desc:
                continue
            tags: list[str] = []
            if ev.affected_names:
                tags.append(f"波及{'、'.join(ev.affected_names)}")
            if ev.location_label:
                tags.append(f"在{ev.location_label}")
            suffix = f"（{'；'.join(tags)}）" if tags else ""
            rows.append(f"- {ev.narrative_desc}{suffix}")
        if not rows:
            return ""
        return "【已注入事件】(勿对同一批人 / 同一地点 / 同类梗重复)\n" + "\n".join(rows)

    async def _generate_event_plan(
        self,
        *,
        current_step: int,
        world_time: WorldTime,
        narrative_summary: str,
        all_agents: Mapping[str, "Agent"],
        locations: Sequence["Place"],
        entities: Sequence["WorldEntity"],
    ) -> _EventPlan | None:
        """The LLM generates an event and declares its channels (spawn / alter / destroy /
        broadcast / message).

        The editor sees beyond where people stand: the menu is every occupied place plus some
        places elsewhere (``_menu_places``), and the unheld things on them. Only plan generation gets it; the gate judges pacing, which the map doesn't change.
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
            f"#{i + 1} {self._person_line(a)}" for i, a in enumerate(living.values())
        )
        headcount = Counter(a.personality.state.current_location for a in living.values())
        on_ground = [e for e in entities if e.location_id is not None]
        places = _menu_places(
            locations, headcount, Counter(e.location_id for e in on_ground),
            random.Random(current_step),
        )
        location_ref = IndexedRef(loc.place_id for loc in places)
        location_lines = "\n".join(
            f"#{i + 1} {render_location(loc)}"
            f"（{f'此刻{headcount[loc.place_id]}人' if headcount[loc.place_id] else '此刻无人'}）"
            for i, loc in enumerate(places)
        ) or "（无）"
        listed = {loc.place_id for loc in places}
        grounded = [e for e in on_ground if e.location_id in listed]
        entity_ref = IndexedRef(e.entity_id for e in grounded)
        entity_lines = "\n".join(
            self._entity_line(i + 1, e) for i, e in enumerate(grounded)
        ) or "（无）"

        system = f"""\
【角色】你是开放世界叙事的节奏编辑,从第三方视角设计世界此刻自己发生的一件事。它要从下方的
故事前提、人物与近况里长出来,并给这个世界带来**重大改变**:
- 把已达临界点的内容升华或点燃:向平淡或流水账的世界中注入可能引起故事往不同方向推进、或产生不同叙事内容的事件。
- 升华或加深叙事:把已经达到临界点的内容进一步升华或点燃。
- 令叙事的内容或方向重大转折:注入可能引起叙事的内容或方向发生重大转折、反转的事件。
- 打破原地打转的僵局:打破当下叙事原地打转、反复循环，无实质进展的局面。比如某些角色一直在反复做一类事情并且没有有效推进的情况。
优先从上面四个方向着手,但不限于此。

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
- message:定向投递给特定人物的内容。
spawn / alter / destroy 每拍各至多一件。你投出去的话里若有一件东西出现、被改或被毁,
就用对应的一项让它真的发生 —— 否则人们会去找一件根本不存在的东西。
新东西只能落在某个地点上,不会凭空到谁手里:要让某人得到它,就放在他所在的地方,
别写成「送到他手中」。

【避免】(失败模式,逐条规避)
- 笼统抽象、等于没说的事件，必须给出具体的事情、人物，事件，变化等等。
- 与故事前提 / 人物身份脱钩、凭空降下的通用桥段
- 与【已注入事件】里同一批人、同一地点或同类手法的重复
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
  "spawn": {{"location": 放在哪个地点(地点序号), "entity_type": "item 或 landmark", "name": "它叫什么,不超过 12 字", "description": "它看上去是什么样,不超过 40 字", "content": "只有它本身承载文字或记录时才填,写它上面写着 / 记着的东西,不超过 80 字;其余一律留空字符串", "observation": "这件东西本身的变化,不写人,不超过 40 字"}} 或 null,
  "alter": {{"entity": 物件序号, "state": "变化后的状态词,不超过 12 字,不变就留空", "description": "变化后它看上去的样子,不超过 40 字,不变就留空", "content": "改动后上面写着的全文,不超过 80 字,不变就留空", "observation": "这件东西本身的变化,不写人,不超过 40 字"}} 或 null,
  "destroy": {{"entity": 物件序号, "observation": "这件东西本身的变化,不写人,不超过 40 字"}} 或 null,
  "broadcast": {{"location_scope": 发生地的地点序号(笼罩全域就填 null,或整个不写这一项), "content": "广播内容,不超过 80 字", "severity": "{Severity.prompt_choices()}", "phenomenon": "{Phenomenon.prompt_choices()}"}} 或 null,
  "message": {{"recipients": [人物序号], "content": "定向内容,不超过 80 字", "urgency": "low|normal|high|critical"}} 或 null,
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

依上面说定的 JSON 格式设计事件（reason 在前），只输出 JSON、不写任何多余内容。"""

        try:
            response = await self.llm_router.complete(
                LLMScene.EVENT_TIMING,
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
                # + message{recipients ~6 + content ≤80 chars 120 + urgency 3 + 3 fields 15} = 144
                # + narrative_desc ≤60 chars (90) + top-level 8-field structure (40);
                #   ~1169 tok
                max_tokens=output_budget(1169),
                temperature=0.8,
            json_mode=True,
            )
        except Exception as exc:
            logger.warning(
                "event_plan_llm_failed",
                extra={"step": current_step, "error": str(exc)},
            )
            return None

        try:
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "event_plan_parse_failed",
                extra={"step": current_step, "error": str(exc)},
            )
            return None

        plan = self._parse_plan(data, agent_id_list, location_ref, entity_ref)
        if plan is None:
            logger.warning("event_plan_invalid", extra={"step": current_step})
            return None
        if plan.broadcast is None and plan.message is None and not plan.mutations:
            logger.warning("event_plan_no_valid_channel", extra={"step": current_step})
            return None
        return plan

    def _parse_plan(
        self,
        data: dict[str, Any],
        agent_id_list: list[str],
        location_ref: IndexedRef,
        entity_ref: IndexedRef,
    ) -> _EventPlan | None:
        if not isinstance(data, dict):
            return None
        narrative_desc = str(data.get("narrative_desc", "")).strip()
        if not narrative_desc:
            return None

        is_positive = normalize_is_positive(data.get("is_positive"))
        broadcast = parse_broadcast_spec(data.get("broadcast"), location_ref)
        message = parse_message_spec(data.get("message"), agent_id_list)

        return _EventPlan(
            narrative_desc=narrative_desc,
            is_positive=is_positive,
            broadcast=broadcast,
            message=message,
            spawn=parse_spawn(data.get("spawn"), location_ref),
            alter=_parse_alter(data.get("alter"), entity_ref),
            destroy=_parse_destroy(data.get("destroy"), entity_ref),
        )

    def _person_line(self, agent: "Agent") -> str:
        soul = agent.personality.soul
        where = agent.personality.state.current_location
        return person_referent(
            soul.name, soul.gender, f"在{self.directory.location_name(where)}" if where else "",
        )

    def _entity_line(self, index: int, entity: "WorldEntity") -> str:
        """One thing on the ground in the menu. Written content is shown only for fixed things
        (as ``WorldEntity.readable_by``): otherwise the editor could broadcast words nobody
        present can read."""
        facts = [f for f in (
            f"此刻状态：{entity.state}" if entity.state else "",
            f"现于{self.directory.location_name(entity.location_id)}" if entity.location_id else "",
        ) if f]
        lines = [f"#{index} {entity.name}" + (f"（{'，'.join(facts)}）" if facts else "")]
        if entity.description:
            lines.append(f"    {entity.description}")
        if entity.content and not entity.is_takeable:
            lines.append(f"    上面写着：{entity.content}")
        return "\n".join(lines)
