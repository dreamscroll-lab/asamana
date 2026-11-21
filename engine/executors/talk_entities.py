"""A safety net for things a finished conversation shows changing.

Things change hands, open or break through PHYSICAL and WORK, which adjudicate and key them; the
dialogue prompts discourage a conversation from doing it. Should a transcript still show
something picked up, handed over, set down, opened, ruined or made on the spot, one functional
call reads which of that is already done; code checks each against the world and declares it
through the usual ``EntityStateChange`` / ``EntitySpawn`` channels, so the world doesn't fall
behind what everyone remembers. Its main work is the other half: the conversation's memories are
written after this and told what really changed, so a "给您" that changed nothing is remembered
as said, not done.

A live conversation is ruled the same way, without asking the actor: this is a net, not the
actor's way of acting on things, which goes through PHYSICAL and WORK like anyone's.

In play is what the dialogue's own scene puts within the speakers' reach: the things in their
hands and the things lying here. A bystander's things are not: he takes no part in the talk.

Trade-off: picking up or altering a thing on the ground lands without PHYSICAL's adjudication,
and the ground isn't among TALK's item keys (``ExecutionProcessor._execution_item_keys``), so a
same-step grab elsewhere can land on the same thing. Accepted because a conversation rarely
touches the ground, while keying it would queue every conversation at a place behind one
another. If it grows common, make ground changes fail here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.coerce import coerce_mapping_list, coerce_str
from core.context import GivenFacts, annotate_call
from core.interfaces.action import EntitySpawn, EntityStateChange
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import IndexedRef, LLMRouter, coerce_bool, output_budget
from core.interfaces.perception import DEFAULT_ENTITY_STATE
from core.logging import get_logger
from core.prompts import SituationVoice, person_referent, render_entity, render_situation_header
from engine.executors.base import complete_json
from engine.environment import MAX_SCENE_ENTITIES
from engine.narration import scene_line
from engine.scene import situation_for

if TYPE_CHECKING:
    from core.interfaces.entity import WorldEntity
    from engine.environment import EnvironmentSystem

logger = get_logger(__name__)

#: Told to each record of the conversation when nothing changed hands, changed or was made.
#: Stated rather than left out: without it a line like "给您" reads as done.
NO_ENTITY_CHANGE: str = "这场对话里，没有任何东西真的易手、改变或被做出来。"

#: The byte-identical system prefix (prefix cache). Speakers are #1 #2 and the things continue
#: from #3, so a person's number written into the entity slot falls out of range.
_SYSTEM: str = """\
【角色】
你是中立的记录者，不代入对话中的任何一方，只判断一场刚结束的对话里，东西实际起了什么变化。

【所给】
- 此刻的时间与地点。
- 说话的两个人，编号 #1、#2。
- 两人手上的东西和此处地上摆着的东西，从 #3 起编号。
- 对话记录。

【任务】
判断这场对话进行时，所给的哪件东西**当场真的**起了变化：比如被拿起、交到了对方手上、被放下、被打开、写上了字或改了样子、被毁掉等等；以及有没有当场新做出一件东西。

【约束】
- 只认已经做了的：对白里明白写出东西递了过去、被拿起、被拆开、被撕掉、被写成之类的才算；答应给、约好以后给、要对方先交出来、讨价还价之类的，都只是说。比如一方说「明天给你」，东西就还在他手上。
- 只认这场对话里当场做的：对白里说自己之前已经做过的，不是这场对话里起的变化。比如 #1 说「我昨天就取了某件东西」，这不是这场对话里当场发生的动作。
- 只认所给的东西：拿起、交出、放下、打开、毁掉等等，只能是所给编了号的那几件；对白里提到的别的东西不在所给之列，就什么也没发生。
- 新做出的东西，必须是交谈时当场做成的有形之物，材料是此时此地随手可得的，比如当场写下的一张字条；对白里说拿出了一件早就有的东西，那不是当场做成的，不能把它做出来。
- 改名是换了身份：只有它已经成了另一样东西才改名；被撕了一角、写满了字之类的，名字照旧，比如一张写了字的纸条还叫纸条。别人记得它，靠的就是这个名字。
- 拿不准就不算：漏记一件真事，下回还能再做；多记一件没发生的事，就成了假事实。

【输出】
严格输出以下 JSON，不要任何多余内容；什么也没发生就给空数组：
{"changes": [{"reason": "≤30字", "entity": 物件序号或0, "holder": 人的序号或0, "state": "≤12字；没变则空字符串", "destroyed": true或false, "name": "≤14字", "description": "≤24字", "content": "≤80字"}]}
- 每条写一件起了变化的东西在谈完之后的样子，一件东西只写一条，最多 3 条。
- reason：对白里哪句话表明这件事已经做了。
- entity：是所给哪件东西；当场新做出来的填 0。
- holder：谈完之后它在谁手上，填人的序号；放下了、或一直摆在地上没人持有，填 0。
- state：它现在的状态，没变留空；destroyed：毁掉了填 true，否则 false。
- name / description / content：谈完之后它的名字、样子、内容。新做出来的东西 name 必填；已有的东西哪一栏改了才填哪一栏，没改留空。description 是旁人一眼看得到的样子；content 是上面写着的内容本身，改了就写它现在的全部内容，不是只写新添的那一笔；没有内容的东西留空。
- 先写 reason，再写其余各栏。序号只填所给的数字，写裸整数，不加引号。

格式示意（只示意格式，不是内容）：#2 把手上的 #3 递给了 #1，#1 接了过去——
{"changes": [{"reason": "#1 接过了它", "entity": 3, "holder": 1, "state": "", "destroyed": false, "name": "", "description": "", "content": ""}]}
常见的错：#2 只说了「这就给你」，手上没动，却写成 holder 为 1——那是把说当成了做，应给空数组。
"""


@dataclass(frozen=True)
class TalkEntityChanges:
    """The changes a conversation brought about, checked against the world and ready to land,
    with one third-person line per change for the conversation's records."""

    state_changes: tuple[EntityStateChange, ...] = ()
    spawns: tuple[EntitySpawn, ...] = ()
    lines: tuple[str, ...] = ()

    def record_text(self) -> str:
        """What the records of this conversation are told really changed."""
        return "\n".join(f"- {line}" for line in self.lines) if self.lines else NO_ENTITY_CHANGE


async def judge_talk_entity_changes(
    llm: LLMRouter,
    directory: WorldDirectory,
    environment: "EnvironmentSystem",
    *,
    speaker_ids: tuple[str, str],
    transcript: str,
    location: str,
    step: int,
) -> TalkEntityChanges:
    """Read which changes the transcript shows as done, and keep those the world allows.

    Empty when the call fails (Rule 1, tier 1): nothing lands, and the records are told so,
    which is the truth of the world either way.
    """
    names = tuple(directory.agent_name(pid) for pid in speaker_ids)
    here = environment.get_body_location(speaker_ids[0])
    # Ground order and cap as the dialogue's own scene has them (``reachable_in_order``).
    in_play: list["WorldEntity"] = [
        *(e for pid in speaker_ids for e in environment.get_items_of(pid)),
        *environment.reachable_in_order(
            environment.get_items_at(here), viewer_id=speaker_ids[0], limit=MAX_SCENE_ENTITIES,
        ),
    ]
    # Nothing within reach, nothing to rule on. A thing made from nothing on the spot goes
    # unruled too: the net is for things the world has, and the records are still told truly
    # that nothing changed.
    if not in_play:
        return TalkEntityChanges()
    items = IndexedRef((e.entity_id for e in in_play), start=len(speaker_ids) + 1)
    people = IndexedRef(speaker_ids)

    roster = "\n".join(
        f"#{i} {person_referent(name, _gender(directory, pid))}"
        for i, (pid, name) in enumerate(zip(speaker_ids, names), 1)
    )
    # With its content: a change to it replaces the whole of it, so the ruling must see what is
    # there to write it out in full. Safe to show: only the ruling reads it, and the records'
    # lines never quote content.
    things = "\n".join(
        f"#{items.number(e.entity_id)} "
        + render_entity(
            e, owner_name=directory.agent_name(e.owner_id) if e.owner_id else None, content=e.content,
        )
        for e in in_play
    )
    header = render_situation_header(
        situation_for(environment, speaker_ids[0]), voice=SituationVoice.THIRD,
    )
    header_part = f"{header}\n\n" if header else ""
    user = f"""\
{header_part}【说话的两个人】
{roster}

【两人手上和此处地上的东西】
{things}

【对话记录】
{transcript}

依上面说定的 JSON 格式判断这场对话里东西的实际变化，只输出 JSON、不写任何多余内容。"""
    with annotate_call(
        given_facts=GivenFacts()  # same source as user above
        .add("此刻何时何地", header)
        .add("说话的两个人", roster)
        .add("两人手上和此处地上的东西", things)
        .add("对话记录", transcript),
        item_candidates={
            str(items.number(e.entity_id)):
                f"{e.name or '某物'}（{directory.agent_name(e.owner_id) if e.owner_id else '地上'}）"
            for e in in_play
        },
    ):
        data = await complete_json(
            llm.complete, _SYSTEM, user,
            temperature=0.2,
            # changes ≤3 × (reason ≤30 chars ≈45 + entity ~3 + holder ~3 + state ≤12 ≈18
            #   + destroyed ~3 + name ≤14 ≈21 + description ≤24 ≈36 + content ≤80 ≈120
            #   + structure for 8 fields ~40) ≈ 3 × 289 = 867, + the array's ~10; est. ~877 tok.
            max_tokens=output_budget(877),
            failure_event="talk_entity_changes_failed",
            log_extra={"speakers": list(speaker_ids), "step": step},
        )
    if data is None:
        return TalkEntityChanges()
    return _checked(
        data, speaker_ids=speaker_ids, names=names, items=items, people=people,
        environment=environment, here=here, location=location, step=step,
    )


def _gender(directory: WorldDirectory, agent_id: str) -> str:
    entry = directory.describe(agent_id)
    return entry.gender if entry is not None else ""


def _checked(
    data: dict[str, Any],
    *,
    speaker_ids: tuple[str, str],
    names: tuple[str, str],
    items: IndexedRef,
    people: IndexedRef,
    environment: "EnvironmentSystem",
    here: str,
    location: str,
    step: int,
) -> TalkEntityChanges:
    """Turn the answer into changes the world allows.

    Each entry says how one thing stands after the talk; what happened is read off it here:
    ``entity`` 0 is a thing made on the spot, a ``holder`` other than the current one is a pick-up,
    a handover or a setting-down, a new ``state`` / ``name`` / ``description`` / ``content`` or
    ``destroyed`` is an alteration. An entry the
    world doesn't allow is dropped whole, never repaired into another target: the model's
    reading stays the only source.

    ``here`` is the place id the ground is checked against; ``location`` its name for narration.
    ``holder`` is where WORK's entry has ``carried`` (``work._WorkChange``): here either speaker
    can end up holding the thing, so a number picks one; there only the doer can.
    """
    name_of = dict(zip(speaker_ids, names))
    changes: list[EntityStateChange] = []
    spawns: list[EntitySpawn] = []
    lines: list[str] = []
    settled: set[str] = set()

    def _reject(reason: str) -> None:
        logger.warning("talk_entity_change_rejected", extra={"reason": reason, "step": step})

    for entry in coerce_mapping_list(data.get("changes")):
        # A missing holder keeps the thing where it is; a number naming nobody drops the entry.
        holder_raw = coerce_str(entry.get("holder")).strip()
        holder_ids = people.resolve([holder_raw]) if holder_raw else []
        if holder_raw not in ("", "0") and not holder_ids:
            _reject("holder_not_a_speaker")
            continue
        holder_id = holder_ids[0] if holder_ids else None
        state = coerce_str(entry.get("state")).strip()
        ruined = coerce_bool(entry.get("destroyed"), False)
        name = coerce_str(entry.get("name")).strip()
        description = coerce_str(entry.get("description")).strip()
        content = coerce_str(entry.get("content")).strip()

        if coerce_str(entry.get("entity")).strip() == "0":
            if not name:
                _reject("made_without_a_name")
                continue
            # In hand it is private, like a carried WORK product; set down it is public, and seen
            # appearing.
            spawns.append(EntitySpawn(
                name=name,
                description=description,
                content=content,
                state=state or DEFAULT_ENTITY_STATE,
                holder_id=holder_id,
                is_public=holder_id is None,
                perception="" if holder_id else scene_line(
                    location, f"{names[0]}与{names[1]}交谈时，{name}被留在了此处。",
                ),
            ))
            lines.append(
                f"{name_of[holder_id]}手上多了当场做成的{name}。" if holder_id
                else f"当场做成的{name}被留在了此处。"
            )
            continue

        resolved = items.resolve([entry.get("entity")])
        entity = environment.get_entity(resolved[0]) if resolved else None
        within_reach = entity is not None and not entity.is_destroyed and (
            entity.owner_id in name_of
            or (entity.owner_id is None and entity.location_id == here)
        )
        if entity is None or not within_reach:
            _reject("entity_out_of_reach")
            continue
        if entity.entity_id in settled:
            _reject("entity_already_settled")
            continue
        owner_id = entity.owner_id    # None: it lies on the ground
        new_holder = owner_id if not holder_raw else holder_id
        moved = not ruined and new_holder != owner_id
        if moved and not entity.is_takeable:
            _reject("entity_not_takeable")
            continue
        # A value copied back unchanged is no change. A ruined thing changes nothing else: its
        # new name or words would describe nothing.
        if ruined or name == entity.name:
            name = ""
        if ruined or description == entity.description:
            description = ""
        if ruined or content == entity.content:
            content = ""
        if state == entity.state:
            state = ""
        if not (state or ruined or moved or name or description or content):
            _reject("nothing_changed")
            continue
        settled.add(entity.entity_id)

        where = f"{name_of[owner_id]}手上的{entity.name}" if owner_id else f"摆在此处的{entity.name}"
        if state and not ruined:
            lines.append(f"{where}，状态变为「{state}」。")
        if name:
            lines.append(f"{where}成了{name}。")
        if description:
            lines.append(f"{where}，样子变为「{description}」。")
        if content:
            # Not quoted: the words are for whoever holds it to read, and these lines reach
            # listeners too.
            lines.append(f"{where}写上了新的内容。")
        if ruined:
            perception = (
                f"{name_of[owner_id]}毁掉了{entity.name}。" if owner_id
                else f"{names[0]}与{names[1]}交谈时，{entity.name}被毁掉了。"
            )
            lines.append(f"{where}被毁掉了。")
        elif moved and owner_id and new_holder:
            perception = f"{name_of[owner_id]}把{entity.name}交到了{name_of[new_holder]}手上。"
            lines.append(perception)
        elif moved and owner_id:
            perception = f"{name_of[owner_id]}把{entity.name}放在了此处。"
            lines.append(perception)
        elif moved and new_holder:
            perception = f"{name_of[new_holder]}拿起了{entity.name}。"
            lines.append(perception)
        else:
            # Others see it where it was, not what changed in it.
            perception = ""
        changes.append(EntityStateChange(
            entity_id=entity.entity_id,
            new_name=name,
            new_state=state,
            new_description=description,
            new_content=content,
            destroyed=ruined,
            owner_id=new_holder if moved and new_holder else None,
            location_id=here if moved and not new_holder else None,
            perception=scene_line(location, perception) if perception else "",
        ))

    return TalkEntityChanges(state_changes=tuple(changes), spawns=tuple(spawns), lines=tuple(lines))
