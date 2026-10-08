"""Npc errand runner: how a body with no cognition carries an errand through step by step.

There is no LLM here, and there must not be one: the module is deterministic and exercises no
discretion. Feasibility was already ruled on by ``ErrandExecutor`` on the assignment tick. Adding
judgment here (changing its mind, picking another route) turns it into a cheap Agent, which
defeats the reason this tier exists (CLAUDE.md §5).

Every errand has the same fixed skeleton:

    go somewhere → gather what it sees → act on the four axes → return → report to whoever sent it

Gathering and reporting are part of every errand. The middle reads the four axes of
``ErrandOrder`` (where / what to carry / whom to find / what to say); there is no dispatch table,
because what it can do comes from combining the axes, not from a list.

It is a runtime phase, not part of ErrandExecutor: that executor ends on the assignment tick,
while the trip takes several more (full argument in the ``engine.executors.errand`` docstring).

Scanning the whole (config-capped) table each step is intentional: a separate roster of active
errands would be a second source of truth, and missing a sync leaves someone who never wakes up.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from core.interfaces.action import EntityStateChange, ErrandOrder
from core.interfaces.directory import WorldDirectory
from core.interfaces.message import Message
from core.interfaces.urgency import Urgency
from core.logging import get_logger
from core.prompts import SituationVoice
from engine.environment import IN_TRANSIT, UNPLACED
from engine.narration import scene_line
from engine.scene import SceneVisibility, assemble_scene_context
from world.models import ActiveErrand

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem
    from world.models import Npc

logger = get_logger(__name__)

#: Walking time covered per tick, in steps: twice an agent's pace, since an errand body only walks
#: while an agent's MOVE keeps living, can be interrupted and rethinks on arrival.
#: - At 1x the reply arrives later than going yourself, so the errand adds nothing.
#: - From 3x up it buys nothing but flattens the map: distance stops constraining who knows what,
#:   and the body spends too few ticks anywhere to be met or stopped.
#: No config key: this is world physics, not a deployment knob.
NPC_PACE = 2


class NpcRunner:
    """Advances every Npc's errand by one hop per step. Deterministic, no LLM."""

    def __init__(
        self,
        *,
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
        directory: WorldDirectory,
        world_id: str,
        seconds_per_step: int,
    ) -> None:
        self._environment = environment
        self._messages = message_system
        self._directory = directory
        self._world_id = world_id
        self._seconds_per_step = seconds_per_step

    async def advance(self, step: int, agents: "dict[str, Agent] | None" = None) -> None:
        """Move every errand on by one step. Expired conditions are cleared first, then it walks.

        ``agents`` is passed through to ``_look``: condition only lives on the live ``Agent``, and
        without it the report (written into the sender's memory) renders a bound person as free.
        """
        for npc in self._environment.all_npcs():
            # Only decides when; the condition slot belongs to ``EnvironmentSystem``.
            self._environment.expire_npc_condition(npc.npc_id, step)
            errand = npc.errand
            if errand is None:
                continue
            if npc.condition is not None:
                # Paused, not cancelled: once untied he carries on. Cancelling would turn one
                # interception into a permanent, unseen cancellation.
                continue
            try:
                await self._advance_one(npc, errand, step, agents)
            except Exception as exc:  # noqa: BLE001
                # Guard per slot (Rule 4).
                logger.warning(
                    "npc_errand_advance_failed",
                    extra={"npc_id": npc.npc_id, "step": step, "error": str(exc)},
                )

    async def _advance_one(
        self, npc: "Npc", errand: ActiveErrand, step: int,
        agents: "dict[str, Agent] | None" = None,
    ) -> None:
        """Advance one errand by one tick.

        Same-place invariant: whatever he accomplishes this tick happens where he stands when the
        tick ends (see ``engine.narration.SAME_PLACE_VERDICT_RULE``). Walking and acting share
        code here, so it is enforced here; broken, the observation channels place one tick a
        whole leg apart.

        Pacing: 1 tick for the order + T out + T back; the tick that reaches a leg's end does that
        leg's business and doesn't move again. The order tick is spent at the start so onlookers
        who just heard the order can still stop him.
        """
        if errand.assigned_step == step:
            return

        dest = errand.order.destination_id if errand.outbound else errand.origin_id
        here = self._environment.get_body_location(npc.npc_id)
        if here != dest:
            if not self._travel(npc, errand, here, dest, step):
                # Turn back and report; don't cancel silently: the sender remembers "I sent him",
                # and a wasted trip still needs a reply.
                where = self._environment.narrative_location_name(dest)
                errand.done = errand.done + (f"到不了{where}。",)
                if not errand.outbound:
                    await self._report(npc, errand, step)
                    self._environment.note_npc_outcome(
                        npc.npc_id, f"回不去{where}，直接在这里回话了", ongoing=False,
                    )
                    self._environment.clear_errand(npc.npc_id)
                    return
                errand.outbound = False
                self._environment.note_npc_outcome(npc.npc_id, f"去不了{where}", ongoing=False)
                return
            here = self._environment.get_body_location(npc.npc_id)
            if here != dest:
                self._environment.note_npc_outcome(
                    npc.npc_id, self._on_the_way(errand), ongoing=True,
                )
                return

        if errand.outbound:
            # Look first, then act: otherwise the report describes "the thing I handed over" as
            # "I saw it there".
            errand.seen = self._look(npc, agents)
            reports, line = await self._do_errand(
                npc, errand.order, step, requester_id=errand.requester_id,
            )
            errand.done = errand.done + tuple(reports)
            errand.outbound = False
            # A destination-only errand still walked there and looked; an empty line would say
            # nothing happened this tick.
            self._environment.note_npc_outcome(npc.npc_id, line or "看了一眼", ongoing=False)
            return

        await self._report(npc, errand, step)
        self._environment.note_npc_outcome(
            npc.npc_id,
            f"回来向{self._directory.agent_name(errand.requester_id)}回了话",
            ongoing=False,
        )
        self._environment.clear_errand(npc.npc_id)

    def _on_the_way(self, errand: ActiveErrand) -> str:
        """The line for travel ticks; the way back doesn't restate the purpose."""
        if not errand.outbound:
            return "正往回走"
        where = self._environment.narrative_location_name(errand.order.destination_id)
        return f"正往{where}去{self._intent(errand.order)}"

    def _intent(self, order: "ErrandOrder") -> str:
        """What kind of business he's going about (payload × whether a person is named). Category
        only.

        The message content appears only on the tick he speaks it (``_do_errand``), not on every
        travel tick. Items are named only: what's written inside waits until it's opened.
        """
        to = self._directory.agent_name(order.recipient_id) if order.recipient_id else ""
        message = (order.message or "").strip()
        has_message = bool(message)
        # Item and message share one addressing axis: name the recipient once.
        if order.item_id:
            what = self._directory.entity_name(order.item_id)
            if has_message:
                return f"把{what}交给{to}，再带句话" if to else f"把{what}送去，再当众传句话"
            return f"把{what}交给{to}" if to else f"把{what}送去"
        if has_message:
            return f"给{to}带句话" if to else "当众传句话"
        return f"看看{to}在不在" if to else "来这里看看"

    # ------------------------------------------------------------------
    # One hop

    def _travel(
        self, npc: "Npc", errand: ActiveErrand, here: str, dest: str, step: int,
    ) -> bool:
        """Walk toward the target within this tick's pace budget; returns False if there's no way.

        Passing through emits no ambient: the channel has only 1-2 slots per step, and bodies
        running back and forth would crowd out what matters.

        It walks distance, not nodes: treating a hop as a step would run a body on different
        physics over the same graph. An unfinished edge keeps its remainder in ``leg_remaining``.

        The path is recomputed every step (cheap, cached), never stored: a stored path is a second
        source of truth that silently leads toward a phantom once the body is moved.
        """
        # Catches "never placed", not "halfway": halfway is leg_remaining while the body stays at
        # the place it left.
        if here in (IN_TRANSIT, UNPLACED):
            logger.warning(
                "npc_errand_no_ground",
                extra={"npc_id": npc.npc_id, "step": step, "at": here},
            )
            return False

        budget = NPC_PACE * self._seconds_per_step
        while budget > 0 and here != dest:
            path = self._environment.space.shortest_path(here, dest)
            if not path or len(path) < 2:
                logger.warning(
                    "npc_errand_no_path",
                    extra={"npc_id": npc.npc_id, "step": step, "from": here, "to": dest},
                )
                return False
            nxt = path[1]
            if errand.leg_remaining <= 0:
                errand.leg_remaining = self._environment.space.edge_seconds(here, nxt)
            spent = min(budget, errand.leg_remaining)
            budget -= spent
            errand.leg_remaining -= spent
            if errand.leg_remaining > 0:
                break                       # this edge isn't finished; he's still at here
            self._environment.move_body(body_id=npc.npc_id, location_id=nxt)
            here = nxt
        return True

    # ------------------------------------------------------------------
    # At the far end

    def _look(self, npc: "Npc", agents: "dict[str, Agent] | None" = None) -> str:
        """Gather what he sees: transmit, don't interpret.

        The text goes verbatim into the report. ``visibility=OWN_EYES`` + ``voice=THIRD``: an
        onlooker's voice, but only his own view.

        Bring both what's there and what just happened there: the latter is what moves things.
        Ambient comes via ``spatial_for`` (the one perception algorithm, with self-exclusion), not
        the environment's private buffer.
        """
        scene = assemble_scene_context(
            npc.npc_id, environment=self._environment, directory=self._directory,
            agents=agents, include_header=True, voice=SituationVoice.THIRD,
            visibility=SceneVisibility.OWN_EYES, with_ordinary_fixtures=False,
        ).text
        happenings = [
            ev.content
            for ev in self._environment.spatial_for(agent_id=npc.npc_id).ambient_events
            if ev.content
        ]
        if not happenings:
            # State the absence: a blank reads as "he didn't look".
            return f"   {scene}\n- 刚发生的事：无"
        # One per line: each ambient ends with its own full stop.
        lines = "\n".join(f"  · {text}" for text in happenings)
        return f"   {scene}\n   - 刚发生的事：\n{lines}"

    async def _do_errand(
        self, npc: "Npc", order: "ErrandOrder", step: int, *, requester_id: str,
    ) -> tuple[list[str], str]:
        """Carry out the order at the far end and report it either way. Returns (lines reported
        to the sender, the line onlookers see).

        The two voices can't share text, but are written side by side in each branch: reading the
        axes in two places eventually tells two stories.

        "Whom to find" is the addressing axis, shared by item and message:

        ==========  =========================  ==============================
        payload     person named               no person named
        ==========  =========================  ==============================
        item        hand it to them            leave it here
        message     tell them                  say it publicly here
        neither     check whether they're in   (just bring back what he saw)
        ==========  =========================  ==============================

        "Named a person but didn't find them": bring it back unchanged, never redirect it to
        someone else. A public message can also find nobody there to hear it.
        """
        here = self._environment.get_body_location(npc.npc_id)
        place = self._environment.narrative_location_name(here)
        bearer = npc.name or "某人"
        # Whether anyone is there to receive it is only known now; the sender may not have known.
        addressed = bool(order.recipient_id)
        to_name = self._directory.agent_name(order.recipient_id) if addressed else ""
        met = addressed and (
            self._environment.get_body_location(order.recipient_id) == here
        )
        reports: list[str] = []
        outcome_parts: list[str] = []
        # Name the recipient once: twice reads like two people.
        named = False

        # ── Only a person named: the point is "are they there". Don't collapse it into "go take
        #    a look".
        if addressed and not order.item_id and not order.message:
            reports.append(f"{to_name}在{place}。" if met else f"{to_name}不在{place}。")
            outcome_parts.append(f"看到{to_name}在这儿" if met else f"没见着{to_name}")
            named = True

        # ── Item: in someone's hands or on the ground.
        if order.item_id:
            item = self._environment.get_entity(order.item_id)
            item_name = (item.name if item is not None else "") or "某物"
            if item is None or item.owner_id != npc.npc_id:
                reports.append(f"{item_name}没能带到。")
                outcome_parts.append(f"没能把{item_name}带来")
            elif addressed and not met:
                reports.append(f"{to_name}不在{place}，{item_name}没能交出去。")
                outcome_parts.append(
                    f"{item_name}没交出去" if named
                    else f"没见着{to_name}，{item_name}没交出去"
                )
                named = True
            else:
                # Go through the environment's single convergence point: the placement
                # invariant lives only there.
                handed = f"{bearer}把{item_name}交到了{to_name}手上。"
                dropped = f"{bearer}把{item_name}放在了此处。"
                self._environment.change_entity_state(EntityStateChange(
                    entity_id=order.item_id,
                    owner_id=order.recipient_id if addressed else None,
                    location_id=None if addressed else here,
                    perception=scene_line(place, handed if addressed else dropped),
                ), acting_agent_id=npc.npc_id)
                reports.append(
                    f"{item_name}已交到{to_name}手上。" if addressed
                    else f"{item_name}已放在{place}。"
                )
                if addressed:
                    outcome_parts.append(
                        f"把{item_name}交了出去" if named else f"把{item_name}交给了{to_name}"
                    )
                    named = True
                else:
                    outcome_parts.append(f"把{item_name}放在了这儿")

        # ── Message: directed vs area broadcast are two cells of the MessageSystem delivery
        #    matrix, not a new mechanism.
        if order.message:
            # Must match ``MessageSystem._resolve_receivers`` for the public cell.
            heard = [] if addressed else [
                aid for aid in self._environment.agents_at(here) if aid != requester_id
            ]
            if addressed and not met:
                reports.append(f"{to_name}不在{place}，话没能带到。")
                outcome_parts.append("话也没带到" if named else f"没见着{to_name}，话没带到")
            elif not addressed and not heard:
                # A report saying "delivered" to an empty room would leave the sender waiting
                # forever for a reply.
                reports.append(f"{place}此处无人，话没有人听见。")
                outcome_parts.append(
                    f"{'又' if outcome_parts else ''}传了句话，但此处无人，没有人听见：{order.message}"
                )
            else:
                await self._messages.publish(Message(
                    id=str(uuid.uuid4()),
                    world_id=self._world_id,
                    sender_id=npc.npc_id,
                    sender_name=bearer,
                    content=order.message,
                    recipients=[order.recipient_id] if addressed else None,
                    location_scope=None if addressed else here,
                    # Heard next tick: this tick's recipient inboxes have already gone out.
                    deliver_step=step + 1,
                    created_step=step,
                    urgency=Urgency.NORMAL,
                    # The listener remembers who said it but forms no relation with it.
                    sender_is_agent=False,
                    # Include the dictating sender, or standing there he'd receive his own order
                    # as someone forging his words.
                    actor_ids=(npc.npc_id, requester_id),
                    metadata={"source": "errand"},
                ))
                if addressed:
                    reports.append(f"话已带到{to_name}：{order.message}")
                    outcome_parts.append(
                        f"又带了句话：{order.message}" if named
                        else f"给{to_name}带了句话：{order.message}"
                    )
                else:
                    # Name everyone who heard it: the report must later answer "who heard this".
                    listeners = "、".join(self._directory.agent_name(aid) for aid in heard)
                    reports.append(f"话已在{place}当着{listeners}的面说了：{order.message}")
                    outcome_parts.append(
                        f"{'又' if outcome_parts else ''}当众传了句话：{order.message}"
                    )
        return reports, "，".join(outcome_parts)

    # ------------------------------------------------------------------
    # Reporting back

    async def _report(self, npc: "Npc", errand: ActiveErrand, step: int) -> None:
        """Report what he saw and how the errand went to whoever sent him.

        The Message pipe's lossless transport enforces "transmit, don't interpret" in code.
        """
        errands_done = "\n".join(f" - {line}" for line in errand.done if line)
        where = self._environment.narrative_location_name(errand.order.destination_id)
        parts = [f"我去了{where}一趟，回来了。"]
        if errands_done:
            parts.append(errands_done)
        # What he says stays true; the scene is a long snapshot, stale by next tick, that would
        # crowd memory if stored verbatim. The whole thing is sent; ``spoken`` marks the split so
        # downstream doesn't guess.
        spoken = "\n".join(parts)
        if errand.seen:
            parts.append(f" 那里的情形是这样：\n    {errand.seen}")
        await self._messages.publish(Message(
            id=str(uuid.uuid4()),
            world_id=self._world_id,
            sender_id=npc.npc_id,
            sender_name=npc.name or "某人",
            content="\n".join(parts),
            recipients=[errand.requester_id],
            location_scope=None,
            deliver_step=step + 1,
            created_step=step,
            urgency=Urgency.NORMAL,
            sender_is_agent=False,
            metadata={"source": "errand_report", "spoken": spoken},
        ))
