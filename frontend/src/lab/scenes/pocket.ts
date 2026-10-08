import { ActionType, Deed } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { Emotion, LIN, MO, SHI, aimAgents, carried, did, nextSeq, step, type LabScene, who } from "./fixtures";

/**
 * What a person is, standing still: the condition laid on him and the things in his hands.
 *
 * Not beats, so these scenes ask whether a standing fact stays shown (through quiet steps, a
 * hand-off, wearing off) without shouting over what is happening.
 *
 * The bench's "跟随" (follow) toggle decides whether these figures are subjects or background, and
 * the pocket handle stands down on a figure nobody is watching: focus off = everyone keeps his
 * handle; focus on = they all are focused. Flip it on a still scene to see a handle come and go.
 */
export function buildPocketScenes({ open: OPEN }: LabPlaces): LabScene[] {
  return [
    {
      id: "condition-mark",
      group: "随身",
      title: "处境 · 只有一个人被制住",
      watch:
        "阿石身上有处境、阿墨没有。阿石头侧一颗琥珀小点在慢慢呼吸，阿墨身上干干净净——" +
        "标记绝不能串到旁人身上，也不能变成人人都有的装饰。**处境的文字任何时候都不上名牌**：" +
        "阿石两手空空，但他有处境，所以名字右边仍有一个可点的箭头，那句话在口袋里。" +
        "两人都还什么都没做，所以这一屏上除了名字，就只该有那一颗点和那个箭头。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN),
            who(SHI, OPEN, { condition: "双手被反绑", emotion: Emotion.fear, emotion_valence: -0.6 }),
          ],
        }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN, { condition: "双手被反绑", emotion: Emotion.fear })],
        }),
      ],
    },
    {
      id: "condition-wears-off",
      group: "随身",
      title: "处境 · 自行消退（无人宣告）",
      watch:
        "第二拍施加、第三拍还在、第四拍到期自解。消失是**静默**的——引擎不会为「他渐渐醒转」发任何一条叙事，" +
        "所以这里只该看到肩上的点没了，不该有任何飘字或特效替它宣告。反过来，点也不能赖着不走。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN, { condition: "中了软筋散，四肢发软" })],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.restrain,
              target: aimAgents(SHI.id),
              action_description: "把药粉拂进阿石鼻息之间。",
              outcome: `在${OPEN.name}，阿墨把药粉拂向阿石，阿石当即腿脚发软。`,
            }),
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN, { condition: "中了软筋散，四肢发软" })] }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
      ],
    },
    {
      id: "pocket-many",
      group: "随身",
      title: "口袋 · 一个人带五件东西",
      watch:
        "名牌右肩只有「✦5 ▸」一个把手，五件东西一个字都不该露在外面。点开：面板贴在名牌右侧，" +
        "不盖住人、不盖住脚下的结果条。**只有两行带状态**——「弓 · 弦已断」「一卷绢帛 · 封缄未拆」；" +
        "其余三件是 intact（后端的默认值），一个字都不该印出来，否则五行全以同一个英文词收尾，" +
        "真正有事的那一行反而最难找。再点一次收起，把手的箭头跟着翻向。",
      steps: [
        step({ states: [who(MO, OPEN)], entities: carried(MO.id, [["横刀", "intact"]]) }),
        step({
          states: [who(MO, OPEN)],
          entities: carried(MO.id, [
            ["横刀", "intact"],
            ["弓", "弦已断"],
            ["玉佩", "intact"],
            ["一卷绢帛", "封缄未拆"],
            ["水囊", "intact"],
          ]),
        }),
      ],
    },
    {
      id: "pocket-with-condition",
      group: "随身",
      title: "口袋 · 处境与随身之物同时",
      watch:
        "三个人各点开一次对比。琥珀那行处境在最上，金色的东西在下面——两段不同颜色、中间一道细线，" +
        "一眼分得开谁是「他被怎么了」、谁是「他拿着什么」；面板边框是各自的身份色，人多时也知道这块板子是谁的。" +
        "字号要**比名字小**：面板是名字的附注，不该比名字还响。" +
        "同一时刻只允许一个口袋开着：点第二个人，第一个自动收起；改选别人时，开着的口袋跟着那个人一起退场。",
      steps: [
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)],
          entities: {
            ...carried(MO.id, [["横刀", "intact"], ["火把", "点燃"]]),
            ...carried(LIN.id, [["药囊", "intact"]]),
          },
        }),
        step({
          states: [
            who(MO, OPEN),
            who(SHI, OPEN, { condition: "双手被反绑", emotion: Emotion.anger }),
            who(LIN, OPEN, { condition: "左腿受创，行动迟缓" }),
          ],
          entities: {
            ...carried(MO.id, [["横刀", "intact"], ["火把", "点燃"]]),
            ...carried(LIN.id, [["药囊", "intact"]]),
          },
        }),
      ],
    },
    {
      id: "pocket-live-change",
      group: "随身",
      title: "口袋 · 开着的时候东西换了手",
      watch:
        "**在第二拍之前把阿墨的口袋点开，然后一直开着看完。** 横刀转到阿石手上时，阿墨开着的面板要当场少一行、" +
        "把手从 ✦2 变 ✦1，而不是留着一件已经不在他身上的东西；阿石的把手同时从无到有。" +
        "最后一拍阿墨手上空了：面板自己消失（空盒子不该留在屏幕上），阿石那边照旧。",
      steps: [
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          entities: carried(MO.id, [["横刀", "intact"], ["水囊", "intact"]]),
        }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          entities: {
            ...carried(MO.id, [["横刀", "intact"]]),
            ...carried(SHI.id, [["水囊", "intact"]]),
          },
        }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          entities: carried(SHI.id, [["水囊", "intact"], ["横刀", "intact"]]),
        }),
      ],
    },
    {
      id: "pocket-vs-post",
      group: "随身",
      title: "口袋 · 与收信徽标互斥",
      watch:
        "同一个人同时有随身之物和一封刚到的信。两个控件各在各的位置——✉ 在名牌之上（这一步发生的事），" +
        "口袋把手在名字右肩（一直为真的事）——不许挤在一起。开信时开着的口袋要自动收起，反过来也一样：" +
        "一个人头上不该同时浮着两块面板。",
      steps: [
        step({
          states: [who(MO, OPEN), who(LIN, OPEN)],
          entities: carried(LIN.id, [["药囊", "intact"], ["短刃", "intact"]]),
        }),
        step({
          states: [who(MO, OPEN), who(LIN, OPEN, { emotion: Emotion.anticipation })],
          entities: carried(LIN.id, [["药囊", "intact"], ["短刃", "intact"]]),
          messages: [
            {
              message_id: "3d71c0a4-9e52-47bb-8f10-6c2ad4915b7e",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [LIN.id],
              perceived_summary: "阿墨要你今夜之前把药送到坊门。",
              spoken: "阿墨要你今夜之前把药送到坊门。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        step({
          states: [who(MO, OPEN), who(LIN, OPEN, { emotion: Emotion.anticipation })],
          entities: carried(LIN.id, [["药囊", "intact"], ["短刃", "intact"]]),
        }),
      ],
    },
  ];
}
