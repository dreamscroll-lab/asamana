import { ActionType } from "../../lib/contract";
import type { EntityView } from "../../types";
import type { LabPlaces } from "../places";
import { MO, SHI, did, step, type LabScene, who } from "./fixtures";

/**
 * A thing that did not exist before, left by finished work: kept (unseen by those present) or set
 * down (seen by all). The arrival plays once, on the step that made it; every scene runs a step
 * past the making to check it then behaves as an ordinary thing.
 */

// Engine-shaped ids: runtime-made things wear a `made_` prefix, seeded ones `seed_`.
const KEPT = "made_item-6b1f90c4aa";
const SET_DOWN = "made_item-2d47ae1130";

/** A thing just written and pocketed — nobody standing there can perceive it. */
const kept = (holder: string): Record<string, EntityView> => ({
  [KEPT]: {
    name: "换防部署令",
    entity_type: "item",
    state: "intact",
    presence: "held",
    presence_ref: holder,
    description: "圈定亲信的名单",
    is_public: false,
    content: "",
    // step({ bornIds: [...] }) sets the real birth step, since only it knows the step number.
    created_step: 0,
  },
});

/** A thing built where it stands — it hides from nobody. */
const setDown = (where: string): Record<string, EntityView> => ({
  [SET_DOWN]: {
    name: "一道栅栏",
    entity_type: "item",
    state: "intact",
    presence: "at_location",
    presence_ref: where,
    description: "拦住了侧门",
    content: "",
    is_public: true,
    created_step: 0,
  },
});

export function buildMadeScenes({ open: OPEN }: LabPlaces): LabScene[] {
  return [
    {
      id: "made-kept",
      group: "造物",
      title: "造物 · 做完了收在自己身上",
      watch:
        "第二拍阿墨伏案做完一件事，一份文书**到他手上**。它没有落地、地上不该多出任何东西——" +
        "只在他头顶飘一句「换防部署令 已入手」，名字右肩的把手从无到有。" +
        "点开口袋：那一行是**空心星 ✧** 配「（不公开）」，与实心 ✦ 的寻常随身之物分得开——" +
        "旁边的阿石看不见这东西，观察界面是上帝视角才画得出来，标记就是在说这件事。" +
        "第三拍什么也没发生：飘字不该再来一次，那一行照旧留在口袋里。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          entities: kept(MO.id),
          bornIds: [KEPT],
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: ActionType.work,
              action_description: "我摊开花名册，拟定换防部署。",
              outcome: `在${OPEN.name}，阿墨写完了换防部署令。`,
              affected_entity_ids: [KEPT],
            }),
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN)], entities: kept(MO.id) }),
      ],
    },
    {
      id: "made-set-down",
      group: "造物",
      title: "造物 · 做完了留在原地",
      watch:
        "第二拍地上**长出**一件东西:marker 从无到有弹起来（不是淡入——淡入与每次重画没有区别），" +
        "同时在房间上方飘一句「一道栅栏 已出现」。这是「已损毁」的镜像:一个只报损失、" +
        "多出东西时一声不响的世界，读起来像只会衰败。" +
        "第三拍它就是一件寻常物件了，安静待在原地，不再弹、不再飘字。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [who(MO, OPEN)],
          entities: setDown(OPEN.id),
          bornIds: [SET_DOWN],
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: ActionType.work,
              action_description: "我把栅栏一根根立起来。",
              outcome: `在${OPEN.name}，阿墨筑起了一道栅栏。`,
              affected_entity_ids: [SET_DOWN],
            }),
          ],
        }),
        step({ states: [who(MO, OPEN)], entities: setDown(OPEN.id) }),
      ],
    },
    {
      id: "made-not-mine",
      group: "造物",
      title: "造物 · 早就在那儿的东西不该有出场",
      watch:
        "反例。第一拍那件东西**已经**在地上了，此后两拍谁也没造它——" +
        "所以从头到尾不该有任何弹出或飘字，它就是一件寻常物件。" +
        "出场只认「这一拍的行动点了它的名 **且** 这一拍正是它的出生步」两条同时成立；" +
        "少了前一条，回放里往前拖一步就会让沿途造出来的东西一起重演一遍出场。",
      steps: [
        step({ states: [who(MO, OPEN)], entities: setDown(OPEN.id) }),
        step({
          states: [who(MO, OPEN)],
          entities: setDown(OPEN.id),
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: ActionType.work,
              action_description: "我在栅栏旁清点绢帛。",
              outcome: `在${OPEN.name}，阿墨清点完了绢帛。`,
            }),
          ],
        }),
        step({ states: [who(MO, OPEN)], entities: setDown(OPEN.id) }),
      ],
    },
  ];
}
