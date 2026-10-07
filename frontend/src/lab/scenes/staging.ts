import { ActionType, Deed, RefKind } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { CHE, CRATE, LAN, LIN, MO, SHI, YAN, aimAgents, aimThing, carried, crate, did, step, type LabScene, who } from "./fixtures";
import type { GraphEdge } from "../../types";

/** A pair who feel something about each other, as the graph route answers it. */
const bond = (a: string, b: string, v: number): GraphEdge[] => [
  { from_id: a, to_id: b, trust: v, affection: v, labels: [], interaction_count: 4, history_summary: "" },
  { from_id: b, to_id: a, trust: v, affection: v, labels: [], interaction_count: 4, history_summary: "" },
];

/** Where bodies end up standing — when the room runs out of room, and when they have
 *  opinions about each other. */
export function buildStagingScenes({ tight: TIGHT, open: OPEN }: LabPlaces): LabScene[] {
  return [
    {
      id: "crowded-room",
      group: "站位",
      title: "拥挤 · 六人同室，两人交谈、一人开箱",
      watch:
        "第一拍六人无事，散开站；第二拍阿墨与阿霖交谈、阿石开箱。要看的是**挤不挤得散**：" +
        "① 阿墨与阿霖必须相邻（挨着，或实在没地方时叠在一格），不能被另外四个人挤到两头；" +
        "② 阿石必须站到货箱旁边、并转向它，不能隔着人堆伸手；" +
        "③ 另外三人被挤开是对的，但不该把上面两组拆散。" +
        "人少时怎么摆都对——这一场存在的理由就是把地方站满，看约束还成不成立。",
      steps: [
        step({
          states: [
            who(MO, TIGHT), who(SHI, TIGHT), who(LIN, TIGHT),
            who(LAN, TIGHT), who(YAN, TIGHT), who(CHE, TIGHT),
          ],
          entities: crate(TIGHT.id),
        }),
        step({
          states: [
            who(MO, TIGHT), who(SHI, TIGHT), who(LIN, TIGHT),
            who(LAN, TIGHT), who(YAN, TIGHT), who(CHE, TIGHT),
          ],
          entities: crate(TIGHT.id),
          actions: [
            // A joint TALK emits one record per participant (see fixtures.ts); the layout needs only
            // one, but both are written to match the wire.
            did(MO, {
              action_type: ActionType.talk, deed: Deed.talk,
              target: aimAgents(LIN.id),
              action_description: "压低声音同阿霖商议。",
              outcome: `在${TIGHT.name}，阿墨与阿霖低声商议。`,
            }),
            did(LIN, {
              action_type: ActionType.talk, deed: Deed.talk,
              target: aimAgents(MO.id),
              action_description: "参与阿墨发起的行动，行动意图：压低声音同阿霖商议。",
              outcome: `在${TIGHT.name}，阿墨与阿霖低声商议。`,
            }),
            // He has to reach the crate, and the crate cannot step aside for him.
            did(SHI, {
              action_type: ActionType.physical, deed: Deed.operate,
              target: aimThing(CRATE), affected_entity_ids: [CRATE],
              action_description: "掀开货箱查看。",
              outcome: `在${TIGHT.name}，阿石掀开了货箱。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "converging",
      group: "站位",
      title: "围拢 · 三人围着一人、两人围着一物",
      watch:
        "第一拍六人无事；第二拍阿墨与阿霖交谈（阿岚在旁听）、阿石把布包交到阿霖手上、" +
        "阿岩与阿澈一起翻看货箱。要看的是**一个对象被几个人同时对付时站不站得拢**：" +
        "① 阿墨、阿岚、阿石三人都要挨着阿霖站，不能只有先排上的那一个挨着、其余被挤到别处；" +
        "② 阿石交物时要转向阿霖——这一动做在布包上，阿霖只是收物人（`reaches`），但仍是当面交；" +
        "③ 阿岩、阿澈都站在货箱旁边并朝向它；" +
        "④ 对话时说话的人与他回应的人相对，旁听的阿岚跟着看说话的人。",
      steps: [
        step({
          states: [
            who(MO, TIGHT), who(SHI, TIGHT), who(LIN, TIGHT),
            who(LAN, TIGHT), who(YAN, TIGHT), who(CHE, TIGHT),
          ],
          entities: { ...crate(TIGHT.id), ...carried(SHI.id, [["布包", "完好"]]) },
        }),
        step({
          states: [
            who(MO, TIGHT), who(SHI, TIGHT), who(LIN, TIGHT),
            who(LAN, TIGHT), who(YAN, TIGHT), who(CHE, TIGHT),
          ],
          entities: { ...crate(TIGHT.id), ...carried(LIN.id, [["布包", "完好"]]) },
          actions: [
            did(MO, {
              action_type: ActionType.talk, deed: Deed.talk,
              target: {
                acts_on: [{ kind: RefKind.agent, id: LIN.id }],
                claims: [{ kind: RefKind.agent, id: LIN.id }],
                reaches: [{ kind: RefKind.agent, id: LAN.id }],
              },
              action_description: "问阿霖那批货几时能到。",
              outcome: `在${TIGHT.name}，阿墨问阿霖货期，阿岚在旁听着。`,
              dialogue: [
                { speaker_id: MO.id, speaker: MO.name, line: "那批货，到底几时能到？" },
                { speaker_id: LIN.id, speaker: LIN.name, line: "最迟后日，路上耽搁了。" },
                { speaker_id: LAN.id, speaker: LAN.name, line: "后日？那就赶不上了。" },
              ],
            }),
            did(LIN, {
              action_type: ActionType.talk, deed: Deed.talk,
              target: {
                acts_on: [{ kind: RefKind.agent, id: MO.id }],
                claims: [{ kind: RefKind.agent, id: LIN.id }],
                reaches: [{ kind: RefKind.agent, id: LAN.id }],
              },
              action_description: "参与阿墨发起的行动，行动意图：问阿霖那批货几时能到。",
              outcome: `在${TIGHT.name}，阿墨问阿霖货期，阿岚在旁听着。`,
            }),
            did(SHI, {
              action_type: ActionType.physical, deed: Deed.relinquish,
              target: aimThing(`seed_seed-${SHI.id.slice(-8)}0`, RefKind.item, [LIN.id]),
              affected_entity_ids: [`seed_seed-${SHI.id.slice(-8)}0`],
              action_description: "把布包交到阿霖手上。",
              outcome: `在${TIGHT.name}，阿石把布包交到了阿霖手上。`,
            }),
            did(YAN, {
              action_type: ActionType.physical, deed: Deed.operate,
              target: aimThing(CRATE), affected_entity_ids: [CRATE],
              action_description: "掀开货箱点数。",
              outcome: `在${TIGHT.name}，阿岩掀开货箱点数。`,
            }),
            did(CHE, {
              action_type: ActionType.physical, deed: Deed.operate,
              target: aimThing(CRATE), affected_entity_ids: [CRATE],
              action_description: "帮着把货箱里的东西翻出来。",
              outcome: `在${TIGHT.name}，阿澈帮着把货箱里的东西翻出来。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "affinity-layout",
      group: "站位",
      title: "亲疏 · 交好的与交恶的站在一处",
      watch:
        "六个人同在一处空地，无人行动——这一幕**只在考站位**。阿墨/阿石/阿霖三人交好，" +
        "阿岚/阿岩/阿澈三人交好，而两伙人彼此交恶。应当看到**两簇**：各自三人凑近，两簇之间拉开距离，" +
        "而不是六个人均匀散成一圈。\n" +
        "这是场景台里唯一带关系边的一幕：亲疏随每一步的 relations 下发，" +
        "落到渲染层就是歇脚位置上的引力与斥力。没有这一幕，整个场景台都在" +
        "「所有人彼此无感」这一种关系下作画，而真实世界从来不是。",
      steps: [
        step({
          relations: [
            ...bond(MO.id, SHI.id, 0.8),
            ...bond(MO.id, LIN.id, 0.8),
            ...bond(SHI.id, LIN.id, 0.8),
            ...bond(LAN.id, YAN.id, 0.8),
            ...bond(LAN.id, CHE.id, 0.8),
            ...bond(YAN.id, CHE.id, 0.8),
            ...bond(MO.id, LAN.id, -0.9),
            ...bond(MO.id, YAN.id, -0.9),
            ...bond(MO.id, CHE.id, -0.9),
            ...bond(SHI.id, LAN.id, -0.9),
            ...bond(SHI.id, YAN.id, -0.9),
            ...bond(SHI.id, CHE.id, -0.9),
            ...bond(LIN.id, LAN.id, -0.9),
            ...bond(LIN.id, YAN.id, -0.9),
            ...bond(LIN.id, CHE.id, -0.9),
          ],
          states: [
            who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN),
            who(LAN, OPEN), who(YAN, OPEN), who(CHE, OPEN),
          ],
        }),
      ],
    },
  ];
}
