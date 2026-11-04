/**
 * A body that acts but does not think, sent on an errand.
 *
 * Its identity arrives through the NPC roster; each step supplies its body state without
 * a relation or action record. The whole round trip is staged because the joints matter: setting out, the
 * handover at the far end, the answer coming back as an ordinary letter.
 */
import { ActionType, Deed, RefKind } from "../../lib/contract";
import type { LabPlaces } from "../places";
import {
  WANG, ZHANG, LI, MO, SHI, carried, did, mindless, step, type LabScene, type Room, type RunnerRecord, who,
} from "./fixtures";
import { nextSeq } from "./fixtures";

function runner(
  at: Room, outcome = "", condition = "", ongoing = false,
): RunnerRecord {
  return mindless(WANG, at, { condition, outcome, ongoing });
}

export function buildErrandScenes(places: LabPlaces): LabScene[] {
  const { pivot: HOME, ne: FAR } = places;
  // Held, so the handover is a real presence change rather than a caption.
  const LETTER: [string, string][] = [["书信", "火漆未启"]];

  return [
    {
      id: "errand-round-trip",
      group: "指派任务",
      title: "指派任务 · 执行流程",
      watch:
        `王二是没有认知的那一档：他该是统一的冷灰、不是任何人的身份色，身体按（男·34）选。` +
        `figure 上**没有**常驻标记。鼠标悬停在他身上应出现一张小卡：名字（男·34岁）、` +
        `他擅长什么、这一拍他在做什么；指针移开要收掉。` +
        `走路要沿路走、不穿墙、有行走帧。叙事流最下面应有一行「NPC …」。` +
        `**接到任务那一拍他不动身、叙事流也没有他的行**——那一拍是**指派任务的人**的行动：` +
        `叙事流里一枚青柠色的「🏃 指派任务」，地图上阿墨转向王二、说一句话的姿势，说完就散——` +
        `不该保持着交谈的姿势站在那里，执行任务的人已经走了。` +
        `地图只在他**停下来**的两拍出声（走到东北办完事、走回出发地回完话）：那一句浮在` +
        `**他头顶**、随即淡去，不挂在房间上——一间屋里站着好几个人，挂在屋上就说不清是谁办的。` +
        `赶路那两拍地图静默，只有叙事流说他往哪去、去办什么。` +
        `**办事那一拍他人就在东北**：浮字、叙事流那行的地点、和他站的房间必须是同一处——` +
        `对不上就是后端的同地不变式破了。` +
        `书信换手那一拍，东西应当**从他身上飞到阿石身上**，而不是从一个口袋瞬移到另一个。` +
        `最后一拍他回话：**头顶那只信封气泡只该有他说的那一句**，他带回的那份现场（时间、` +
        `地点、在场的人…）不该出现在地图上——那是叙事流的事。`,
      steps: [
        // Standing about: no line in the feed. The baseline both marks are read against.
        step({
          states: [who(MO, HOME), who(SHI, FAR)],
          npcs: [runner(HOME)],
          entities: carried(WANG.npc_id, LETTER),
        }),
        // Told. The beat belongs to the man who gave the order, the only act in the round trip;
        // the runner's row stays empty rather than repeat it.
        step({
          states: [who(MO, HOME), who(SHI, FAR)],
          npcs: [runner(HOME)],
          entities: carried(WANG.npc_id, LETTER),
          actions: [
            did(MO, {
              action_type: ActionType.errand,
              deed: Deed.errand,
              // The one place an `npc` kind reaches `acts_on`: it turns the teller toward the runner.
              target: { acts_on: [{ kind: RefKind.npc, id: WANG.npc_id }], claims: [], reaches: [] },
              action_description: `吩咐${WANG.name}把书信送到${FAR.name}，交到${SHI.name}手上。`,
              outcome: `在${HOME.name}，${MO.name}吩咐${WANG.name}把书信送往${FAR.name}。`,
            }),
          ],
        }),
        // On the road, still reported at home: a mover belongs to his origin until he arrives. No float.
        step({
          states: [who(MO, HOME), who(SHI, FAR)],
          npcs: [runner(HOME, `正往${FAR.name}去把书信交给阿石`, "", true)],
          entities: carried(WANG.npc_id, LETTER),
        }),
        // Arrives and does it in one beat, so the float, the feed's place and his room agree. The
        // letter's `presence_ref` is now SHI ("阿石"): it must fly from him to SHI.
        step({
          states: [who(MO, HOME), who(SHI, FAR)],
          npcs: [runner(FAR, `把书信交给了阿石`)],
          entities: carried(SHI.id, LETTER),
        }),
        // On the road home: nothing to announce.
        step({
          states: [who(MO, HOME), who(SHI, FAR)],
          npcs: [runner(FAR, `正往回走`, "", true)],
          entities: carried(SHI.id, LETTER),
        }),
        // Home and reporting in one beat, as a letter to the man who sent him. The errand's content
        // reaches the reader through the feed, never off the figure on the map.
        step({
          states: [who(MO, HOME), who(SHI, FAR)],
          npcs: [runner(HOME, `回来向${MO.name}回了话`)],
          entities: carried(SHI.id, LETTER),
          messages: [
            {
              message_id: "1b5d90a4-3c27-4f08-9ad6-6e2c85b70f13",
              sender_id: WANG.npc_id,
              anonymous: false, sender_name: WANG.name,
              receiver_ids: [MO.id],
              // The backend sends both parts: the bubble shows `spoken` (short), the feed shows
              // `content` (the full text).
              spoken: `任务报告：我去了${FAR.name}执行任务，回来了。\n - 书信已交到阿石手上。`,
              content:
                `任务报告：我去了${FAR.name}执行任务，回来了。\n - 书信已交到阿石手上。\n`
                + ` 到达时所见（六月初一，下午两点）：\n`
                + `  - 地点：${FAR.name}——百官官署所在。\n`
                + `  - 在场的其他人：阿石（男）\n  - 现场物件：无`,
              scope: "direct",
              seq: nextSeq(),
            },
          ],
        }),
      ],
    },
    {
      id: "errand-held",
      group: "指派任务",
      title: "指派任务 · 被拦下",
      watch:
        `被制住的人办不了事，但任务不作废。琥珀色的处境标记应在头边。` +
        `叙事流那行应把处境放在前面：先说他为什么没在动。`,
      steps: [
        step({
          states: [who(MO, HOME), who(SHI, HOME)],
          npcs: [runner(HOME, `正往${FAR.name}去给阿石带句话`, "", true)],
        }),
        // Nothing happened while he was held, so only the condition is shown. The errand is still
        // on, and he carries on once untied.
        step({
          states: [who(MO, HOME), who(SHI, HOME)],
          npcs: [runner(HOME, "", "被按在地上")],
        }),
      ],
    },
    {
      id: "errand-crowd",
      group: "指派任务",
      title: "指派任务 · 三人同时执行",
      watch:
        `两个在办事、一个闲着：叙事流仍只有一行（各项以 · 分隔），不是三行——这行是折叠的` +
        `状态，不是三条事件。**闲着的李四完全不出现**。身体按各自的（性别·年龄）选：` +
        `李四该是女性形象。`,
      steps: [
        step({
          states: [who(MO, HOME)],
          npcs: [
            mindless(WANG, HOME, { outcome: `正往${FAR.name}去把书信交给阿石`, ongoing: true }),
            mindless(
              ZHANG,
              FAR, { outcome: "正往回走", ongoing: true },
            ),
            mindless(
              LI,
              HOME,
            ),
          ],
        }),
      ],
    },
  ];
}
