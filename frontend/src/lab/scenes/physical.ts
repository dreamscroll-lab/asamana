import { ActionType, Deed, RefKind } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { CRATE, Emotion, GATE, LIN, MINDLESS_INK, MO, SHI, aimAgents, aimThing, carried, crate, did, gate, mindless, step, type LabScene, who } from "./fixtures";

/** The body with no mind an act can land on — the other end of `physical_npc_index`. */
const PORTER = {
  npc_id: "npc_lu-liu",
  name: "陆六",
  color: MINDLESS_INK,
  gender: "男",
  age: 29,
  description: "脚夫，替人扛货过市",
};

/** PHYSICAL's seven verbs — the one action type whose deed says nothing about its intent. */
export function buildPhysicalScenes({ open: OPEN }: LabPlaces): LabScene[] {
  return [
    // ---- PHYSICAL's seven verbs ------------------------------------------------
    {
      id: "strike",
      group: "动手",
      title: "strike · 击（成功）",
      watch:
        "attack 三帧挥击 + 受击者 hurt 后仰；命中特效与朝向应先转身再出手。" +
        "关键看顺序：阿石的血条（掉到 0.55）必须等刀落下之后才出现并变短，不能人还没挥手血就先掉了。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.anger, emotion_valence: -0.6, emotion_intensity: 0.9 }),
            who(SHI, OPEN, { vitality: 0.55, emotion: Emotion.fear, emotion_valence: -0.7 }),
          ],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.strike,
              target: aimAgents(SHI.id),
              action_description: "挥拳打向阿石的肩头。",
              outcome: `在${OPEN.name}，阿墨一拳打中阿石的肩头。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "strike-miss",
      group: "动手",
      title: "strike · 击（落空）",
      watch: "判定为「做了但没成」——仍要有完整的挥击姿势，只是特效与 ✗ 标记不同。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.frustration, emotion_valence: -0.4 }), who(SHI, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.strike,
              succeeded: false,
              target: aimAgents(SHI.id),
              action_description: "挥拳打向阿石的肩头。",
              outcome: `在${OPEN.name}，阿墨挥拳打向阿石，被闪开了。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "restrain",
      group: "动手",
      title: "restrain · 擒",
      watch: "shove（推按）而非挥击——按住人和打人必须看得出区别；对方 duck。",
      steps: [
        step({ states: [who(MO, OPEN), who(LIN, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(LIN, OPEN, { emotion: Emotion.fear, emotion_valence: -0.6 })],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.restrain,
              target: aimAgents(LIN.id),
              action_description: "上前按住阿霖的手腕，不让他走。",
              outcome: `在${OPEN.name}，阿墨扣住了阿霖的手腕。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "condition",
      group: "动手",
      title: "restrain · 留下持续处境",
      watch:
        "处境是**状态**不是一拍：第 2 拍捆住之后，名牌上出现一行琥珀色「双手被反绑」，" +
        "并且**第 3、4 拍它必须还在**（那两拍阿霖什么也没做，也没有任何卡片再提起这件事——" +
        "这正是本条要验的：它不靠事件维持）。第 5 拍有人解开，这一行才消失。" +
        "三条线要分得清：携带行是金色 ✦，处境行是暗琥珀，死亡的 OVER 是红色徽章；" +
        "处境**不得**让人像变暗或变灰——那是死亡独占的表达。",
      steps: [
        step({ states: [who(MO, OPEN), who(LIN, OPEN)] }),
        step({
          states: [
            who(MO, OPEN),
            who(LIN, OPEN, { emotion: Emotion.fear, emotion_valence: -0.6, condition: "双手被反绑" }),
          ],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.restrain,
              target: aimAgents(LIN.id),
              action_description: "用绳索将阿霖双手反绑。",
              outcome: `在${OPEN.name}，阿墨将阿霖的双手反绑起来。`,
            }),
          ],
        }),
        // Two beats with no events: the condition persists on its own.
        step({ states: [who(MO, OPEN), who(LIN, OPEN, { condition: "双手被反绑" })] }),
        step({ states: [who(MO, OPEN), who(LIN, OPEN, { condition: "双手被反绑" })] }),
        step({
          states: [who(MO, OPEN), who(LIN, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.restrain,
              target: aimAgents(LIN.id),
              action_description: "解开阿霖手上的绳索。",
              outcome: `在${OPEN.name}，阿墨解开了阿霖手上的绳索。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "seize",
      group: "动手",
      title: "seize · 取物",
      watch:
        "hold（伸手取）+ 货箱从地面消失、转入名牌上的携带行（✦ 货箱）。" +
        "关键看顺序：货箱必须等到伸手那一下之后才离地，不能人还没动就已经在名牌上了。",
      steps: [
        step({ states: [who(MO, OPEN)], entities: crate(OPEN.id) }),
        step({
          states: [who(MO, OPEN)],
          entities: crate(OPEN.id, { presence: "held", presence_ref: MO.id }),
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.seize,
              target: aimThing(CRATE),
              affected_entity_ids: [CRATE],
              action_description: "俯身取走地上的货箱。",
              outcome: `在${OPEN.name}，阿墨拿起了货箱。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "operate",
      group: "动手",
      title: "operate · 摆弄",
      watch:
        "interact 两帧循环（就地操作），不是挥击；坊门状态从 sealed 翻到 open。" +
        "关键看顺序：状态徽标与「坊门：sealed→open」浮字必须在手推到之后才变，不能先变后推。",
      steps: [
        step({ states: [who(MO, OPEN)], entities: gate(OPEN.id) }),
        step({
          states: [who(MO, OPEN)],
          entities: gate(OPEN.id, { state: "open" }),
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.operate,
              target: aimThing(GATE, RefKind.landmark),
              affected_entity_ids: [GATE],
              action_description: "推开紧闭的坊门。",
              outcome: `在${OPEN.name}，阿墨推开了坊门。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "destroy",
      group: "动手",
      title: "destroy · 毁物",
      watch: "同一个挥击动作，但余波不同（碎裂特效）；货箱应在挥击落下之后才消失。",
      steps: [
        step({ states: [who(MO, OPEN)], entities: crate(OPEN.id) }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.anger, emotion_valence: -0.7 })],
          entities: crate(OPEN.id, { presence: "destroyed", presence_ref: null, state: "destroyed" }),
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.destroy,
              target: aimThing(CRATE),
              affected_entity_ids: [CRATE],
              action_description: "抡起木杠砸向货箱。",
              outcome: `在${OPEN.name}，阿墨将货箱砸得粉碎。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "exert",
      group: "动手",
      title: "exert · 无对象发力",
      watch: "无人无物，仍是完整的挥击——朝向应保持在原来的朝向上，不乱转。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.anticipation, emotion_intensity: 0.6 })],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.exert,
              action_description: "奋力去搬路旁的石墩。",
              outcome: `在${OPEN.name}，阿墨憋足了力气，把石墩挪开半尺。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "relinquish",
      group: "动手",
      title: "relinquish · 交物给人",
      watch:
        "夺取的反面，也是 deed 表里最后一个动作。要看的是**东西真的易主**：" +
        "①阿墨摆出 `show` 的姿势——把东西递出去，不是攥住(`hold`)、更不是挥击；" +
        "②货箱应当**从阿墨身上飞到阿石身上**，而不是在一个人身上消失、在另一个人身上出现；" +
        "③飞完之后两人的随身清单各自对上：阿墨空了，阿石多了一件。" +
        "④这一动做在**那件东西**上（`acts_on` 是货箱），阿石是被波及的受事（`reaches`）——" +
        "所以叙事流里**不该**画出「阿墨 → 阿石」那道箭头：箭头说的是「做在他身上」。" +
        "这一幕是 `reaches` 在场景台里唯一一次被喂上。",
      steps: [
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          entities: carried(MO.id, [["货箱", "完好"]]),
        }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.trust, emotion_valence: 0.3 }),
            who(SHI, OPEN, { emotion: Emotion.anticipation }),
          ],
          entities: carried(SHI.id, [["货箱", "完好"]]),
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.relinquish,
              // acts_on is the item, since it changes hands. The recipient goes in reaches, not
              // claims, because receiving something doesn't use up his turn. See the PHYSICAL
              // binding in agent/decision.py.
              target: aimThing(`seed_seed-${MO.id.slice(-8)}0`, RefKind.item, [SHI.id]),
              affected_entity_ids: [`seed_seed-${MO.id.slice(-8)}0`],
              action_description: `把货箱交到${SHI.name}手上。`,
              outcome: `在${OPEN.name}，${MO.name}把货箱交到了${SHI.name}手上。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "strike-mindless",
      group: "动手",
      title: "动手 · 对象是没有认知的人",
      watch:
        "决策层允许对**听人吩咐做事的那些人**动手（拦下、按住、夺他手里的东西，见 " +
        "`physical_npc_index`），所以 `acts_on` 里是一个 npc 而不是 agent。" +
        "要看的就是这一条：**转身和挨打必须是同一个人**——阿墨转向陆六了，那么受力的也该是陆六，" +
        "他身上要有制住的压迫与余波，而不是阿墨对着空气使一下劲。" +
        "陆六不摆受击姿势是对的（不给没有认知的身体摆姿势），但**挨没挨到是世界的事实，与他有没有心智无关**。" +
        "若看到的是使空，那不是这一幕写错了，是渲染只认 agent、不认 npc。",
      steps: [
        step({ states: [who(MO, OPEN)], npcs: [mindless(PORTER, OPEN)] }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.anger, emotion_valence: -0.5, emotion_intensity: 0.8 })],
          npcs: [mindless(PORTER, OPEN, { condition: "被按在地上" })],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.restrain,
              target: { acts_on: [{ kind: RefKind.npc, id: PORTER.npc_id }], claims: [], reaches: [] },
              action_description: `拦下${PORTER.name}，不许他把货抬走。`,
              outcome: `在${OPEN.name}，${MO.name}一把按住${PORTER.name}，货没能抬出市口。`,
            }),
          ],
        }),
      ],
    },
  ];
}
