import { ActionType, Deed, Phase } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { Emotion, LIN, MO, Need, SHI, aimPlace, aimTalk, did, execId, nextSeq, seenBy, step, type LabCharacter, type LabScene, type Room, walking, who } from "./fixtures";

function headingScene(id: string, dir: string, from: Room, to: Room, watch: string): LabScene {
  return {
    id,
    group: "移动",
    title: `移动 · ${dir}`,
    watch,
    steps: [
      step({ states: [who(MO, from)] }),
      step({
        states: [
          who(MO, from, {
            transit: walking([from, to], 1, 1),
            activity_status: "moving",
            dominant_need: Need.selfActualization,
          }),
        ],
        actions: [
          did(MO, {
            action_type: ActionType.move,
            deed: Deed.move,
            target: aimPlace(to.id),
            phase: Phase.begin,
            elapsed_steps: 1,
            total_steps: 1,
            duration_label: "约2小时",
            action_description: `离开${from.name}，向${dir}行去。`,
            outcome: `${MO.name}从${from.name}动身前往${to.name}，路程约2小时。`,
          }),
        ],
      }),
      step({ states: [who(MO, to)] }),
    ],
  };
}

/**
 * Walking, and the four ways to do it.
 *
 * THE HEADINGS. On a 2:1 isometric grid the four ways to walk project to the
 * four screen diagonals, and which one a figure takes is read off its real
 * movement (skins.dirOf), not declared. So a heading is demonstrated by picking
 * two rooms whose centers differ along ONE grid axis — +gx is SE, +gy is SW, and
 * the negatives are their opposites.
 */
export function buildMovementScenes({ pivot: HUB, ne: NE, se: SE, far: FAR, tight: TIGHT, next: NEXT }: LabPlaces): LabScene[] {
  return [
    {
      id: "actor-presence",
      group: "移动",
      title: "演员视角 · 人物进出",
      viewer: MO.id,
      watch: "留在原地的人先站定；来者从可通行的地点边缘淡入走近，离者走向边缘再淡出。" +
        "进出途中不能点选人物。重新进入不留残影；切换到最后一拍时，新地点的人直接站定。",
      steps: seenBy(MO.id, [
        step({ states: [who(MO, HUB)] }),
        step({ states: [who(MO, HUB), who(SHI, HUB)] }),
        step({ states: [who(MO, HUB)] }),
        step({ states: [who(MO, HUB), who(SHI, HUB)] }),
        step({ states: [who(MO, NEXT), who(SHI, NEXT)] }),
      ]),
    },
    // ---- headings ------------------------------------------------------------
    headingScene(
      "walk-ne",
      "东北 NE",
      HUB,
      NE,
      "背对镜头远去：应看到 walkNE 的背面四帧，而不是正面倒着走。",
    ),
    headingScene("walk-sw", "西南 SW", NE, HUB, "面朝镜头走来：walkSW，脸可见。"),
    headingScene("walk-se", "东南 SE", HUB, SE, "向右下：walkSE，身体朝屏幕右侧。"),
    headingScene("walk-nw", "西北 NW", SE, HUB, "向左上：walkNW，背面偏左。"),
    {
      id: "walk-onestep",
      group: "移动",
      title: "移动 · 一步到位（无 transit）",
      watch:
        "他必须绕着走过去，不能斜穿两点之间的建筑与墙体。这一幕是这一整类的唯一覆盖：" +
        "sim 把图上相邻的两处之间的移动记成一步走完，落到 payload 里就是 location 已是终点、transit 为 null——" +
        "没有 transit 的移动最容易被渲成一条直线补间、直接穿墙。其余移动幕全都带 transit，测不到它。",
      steps: [
        step({ states: [who(MO, TIGHT)] }),
        step({
          // location is already the destination and transit is null, which is how the engine
          // reports a move finished in one step. The two places are directly connected on this
          // map's graph; places.ts picks them (see LabPlaces.tight/next).
          states: [who(MO, NEXT, { activity_status: "idle", dominant_need: Need.selfActualization })],
          actions: [
            did(MO, {
              action_type: ActionType.move,
              deed: Deed.move,
              target: aimPlace(NEXT.id),
              action_description: `从${TIGHT.name}前往${NEXT.name}`,
              outcome: `在${NEXT.name}，${MO.name}自${TIGHT.name}行至${NEXT.name}。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "walk-loop",
      group: "移动",
      title: "移动 · 四向连走",
      watch: "一口气 NE→SW→SE→NW：看转向的瞬间步态是否连续（不应每次转向都从第 0 帧重迈）。",
      steps: [
        step({ states: [who(MO, HUB)] }),
        step({ states: [who(MO, HUB, { transit: walking([HUB, NE], 1, 1), activity_status: "moving" })] }),
        step({ states: [who(MO, NE)] }),
        step({ states: [who(MO, NE, { transit: walking([NE, HUB], 1, 1), activity_status: "moving" })] }),
        step({ states: [who(MO, HUB)] }),
        step({ states: [who(MO, HUB, { transit: walking([HUB, SE], 1, 1), activity_status: "moving" })] }),
        step({ states: [who(MO, SE)] }),
        step({ states: [who(MO, SE, { transit: walking([SE, HUB], 1, 1), activity_status: "moving" })] }),
        step({ states: [who(MO, HUB)] }),
      ],
    },
    {
      id: "walk-long",
      group: "移动",
      title: "移动 · 多段长途（分 3 步）",
      watch:
        "一趟跨地点的长路，path 里带着中途经过的地点——看每段接缝处是否连续、有没有回跳或变速。" +
        "最后一拍是关键：引擎的 transit 只在「在途」时存在（transit_view: for an in-flight MOVE, else None），" +
        "抵达那一步 transit 就没了、location 直接是终点。所以在途只到 2/3，剩下的三分之一路是靠一个无 transit 的位移走完的。" +
        "顺带看虚实：走到被树或楼盖住的那截路上时，人应当淡下去（读作「在那后面」），" +
        "走出来再实回来；淡入淡出要看不出突跳，且再淡也得看得清是谁、朝哪走。",
      steps: [
        step({ states: [who(MO, HUB)] }),
        // Transit only goes up to total-1. elapsed_steps = estimated - remaining, and when
        // remaining reaches zero the action completes and its execution state is dropped, so the
        // engine never sends elapsed == total. Don't send 3/3: he would reach the destination a
        // beat early, and the fact that the arrival beat has no transit would go untested.
        ...[1, 2].map((n) =>
          step({
            states: [
              who(MO, HUB, {
                transit: walking([HUB, NE, FAR], n, 3),
                activity_status: "moving",
                dominant_need: Need.selfActualization,
              }),
            ],
            actions:
              n === 1
                ? [
                    did(MO, {
                      action_type: ActionType.move,
                      deed: Deed.move,
                      target: aimPlace(FAR.id),
                      phase: Phase.begin,
                      elapsed_steps: 1,
                      total_steps: 3,
                      duration_label: "约6小时",
                      action_description: `离开${HUB.name}，取道${NE.name}，前往${FAR.name}。`,
                      outcome: `阿墨从${HUB.name}动身前往${FAR.name}，路程约6小时。`,
                    }),
                  ]
                : [],
          }),
        ),
        step({ states: [who(MO, FAR)] }),
      ],
    },
    carriedScene(HUB, FAR),
    ...beforeWalkScenes(TIGHT, NEXT),
  ];
}

/**
 * A momentary act and a walk in one step: the letter or the word goes out where he stood, then he
 * leaves. The engine files the act under the room it was issued in and the walk under where it
 * ends, so the map plays the act there first (playsBeforeWalk).
 *
 * The walk is one step between rooms the map's graph joins, as the wire carries one: no transit,
 * location already the destination, the route on the MOVE's closing beat. The letter is
 * delivered the step after, as in the send_message scenes.
 */
function beforeWalkScenes(from: Room, to: Room): LabScene[] {
  const moveTo = (claims: string[] = []) => ({
    action_type: ActionType.move,
    deed: Deed.move,
    target: aimPlace(to.id, claims),
    route: walking([from, to], 1, 1),
    location_id: to.id,
  });
  const letter = (id: string, content: string, scope: "direct" | "place", receivers: LabCharacter[]) => ({
    message_id: id,
    sender_id: MO.id,
    anonymous: false,
    sender_name: MO.name,
    receiver_ids: receivers.map((r) => r.id),
    content,
    spoken: content,
    scope,
    ...(scope === "place" ? { location_id: from.id } : {}),
    seq: nextSeq(),
  });
  const carryId = execId("move", SHI, 1);
  const carryIntent = `拽着${MO.name}去${to.name}`;
  return [
    {
      id: "walk-after-letter",
      group: "移动",
      title: "移动 · 先发信再走",
      watch:
        `同一步里阿墨在${from.name}发信给阿霖，然后走去${to.name}。` +
        `应先在${from.name}原地做递信的姿势、信封飞向阿霖，**做完才起步**；` +
        `不能边走边递，也不能走到${to.name}之后才递。` +
        "下一拍信在阿霖头上打开。",
      steps: [
        step({ states: [who(MO, from), who(LIN, from)] }),
        step({
          states: [who(MO, to, { dominant_need: Need.social }), who(LIN, from)],
          actions: [
            did(MO, {
              action_type: ActionType.send_message,
              deed: Deed.send_message,
              target: aimTalk(LIN.id),
              location_id: from.id,
              action_description: "写一封短笺留给阿霖，约他明日再见。",
              outcome: `在${from.name}，阿墨遣人把书信送往阿霖处。`,
            }),
            did(MO, {
              ...moveTo(),
              action_description: `从${from.name}前往${to.name}`,
              outcome: `在${to.name}，${MO.name}自${from.name}行至${to.name}。`,
            }),
          ],
        }),
        step({
          states: [who(MO, to), who(LIN, from, { emotion: Emotion.anticipation })],
          messages: [letter("6f5a7182-93a4-4b5c-8d7e-f9a0b1c2d3e4", "阿墨约你明日辰时在坊门相见。", "direct", [LIN])],
        }),
        step({ states: [who(MO, to), who(LIN, from)] }),
      ],
    },
    {
      id: "walk-after-announce",
      group: "移动",
      title: "移动 · 当众传话后离开",
      watch:
        `阿墨在${from.name}当众传话，同一步走去${to.name}。` +
        `传话的涟漪应在${from.name}、从他原地站的位置扩散开，**扩散完才起步**；` +
        `涟漪不能跟着他走，也不能出现在${to.name}。` +
        `下一拍告示绑在${from.name}上，阿墨已经不在那里，牌子照常出现。`,
      steps: [
        step({ states: [who(MO, from), who(SHI, from), who(LIN, from)] }),
        step({
          states: [who(MO, to), who(SHI, from), who(LIN, from)],
          actions: [
            did(MO, {
              action_type: ActionType.send_message,
              deed: Deed.send_message,
              location_id: from.id,
              action_description: "临走前高声传话，让众人都听见。",
              outcome: `在${from.name}，阿墨当众传话。`,
            }),
            did(MO, {
              ...moveTo(),
              action_description: `从${from.name}前往${to.name}`,
              outcome: `在${to.name}，${MO.name}自${from.name}行至${to.name}。`,
            }),
          ],
        }),
        step({
          states: [who(MO, to), who(SHI, from), who(LIN, from)],
          messages: [letter("7a6b8293-a4b5-4c6d-9e8f-a0b1c2d3e4f5", "今夜坊门提前落锁，各家早些回去。", "place", [SHI, LIN])],
        }),
        step({ states: [who(MO, to), who(SHI, from), who(LIN, from)] }),
      ],
    },
    {
      id: "walk-carried-after-letter",
      group: "移动",
      title: "移动 · 发信时被人拽走",
      watch:
        `同一步里阿墨在${from.name}发信给阿霖，阿石拽着他去${to.name}。` +
        "信没有被拽走打断：应先看到阿墨原地递信、信封飞出，**然后两人一起起步**。" +
        "阿石要等阿墨递完才走，不能先走开再让阿墨隔空追上。",
      steps: [
        step({ states: [who(MO, from), who(SHI, from), who(LIN, from)] }),
        step({
          states: [who(MO, to), who(SHI, to), who(LIN, from)],
          actions: [
            did(MO, {
              action_type: ActionType.send_message,
              deed: Deed.send_message,
              target: aimTalk(LIN.id),
              location_id: from.id,
              action_description: "匆匆写下几句，托人送给阿霖。",
              outcome: `在${from.name}，阿墨遣人把书信送往阿霖处。`,
            }),
            did(SHI, {
              ...moveTo([MO.id]),
              execution_id: carryId,
              initiator_id: SHI.id,
              action_description: carryIntent,
              outcome: `在${to.name}，${SHI.name}拽着${MO.name}自${from.name}行至${to.name}。`,
            }),
            did(MO, {
              ...moveTo([MO.id]),
              execution_id: carryId,
              initiator_id: SHI.id,
              action_description: `参与${SHI.name}发起的行动，行动意图：${carryIntent}`,
              outcome: `${MO.name}被${SHI.name}强制拉入「${carryIntent}」`,
            }),
          ],
        }),
        step({
          states: [who(MO, to), who(SHI, to), who(LIN, from, { emotion: Emotion.fear })],
          messages: [letter("8b7c93a4-b5c6-4d7e-8f9a-b1c2d3e4f5a6", "有人逼我离开，速去报官。", "direct", [LIN])],
        }),
        step({ states: [who(MO, to), who(SHI, to), who(LIN, from)] }),
      ],
    },
  ];
}

/**
 * Taking someone along: the joint form of MOVE, as two records sharing one execution_id.
 *
 * A move is aimed at a place; the person taken along only has his turn claimed
 * (ActionTarget.claims). Both records therefore have the place in `acts_on` and the companion
 * in `claims`. Nobody is acted on, so the feed draws no arrow. The companion's
 * `inner_monologue` is empty because he made no decision, and his outcome is the engine's
 * participant line "被…强制拉入".
 *
 * The companion's `location_id` is an empty string, as on the wire: someone in transit is in
 * no room.
 */
function carriedScene(from: Room, to: Room): LabScene {
  const eid = execId("move", MO, 1);
  const intent = `从${from.name}前往${to.name}`;
  const shared = {
    action_type: ActionType.move,
    deed: Deed.move,
    execution_id: eid,
    initiator_id: MO.id,
    phase: Phase.begin,
    elapsed_steps: 1,
    total_steps: 3,
    duration_label: "约6小时",
  } as const;
  return {
    id: "walk-carried",
    group: "移动",
    title: "移动 · 带着人走",
    watch:
      "叙事流那半边:名字行只该有发起者,被带的人在「带着 X」那枚淡 chip 里。" +
      "两条记录都是 acts_on=地点、claims=被带的人,叙事流据此判定没有箭头 —— " +
      "把他摆到「A → B」的右边等于把一次同行报成一次针对他的行动。" +
      "地图那半边:transit 只发给发起者,所以只有他会走 —— 那是当前的真实行为,不是这幕写漏了。",
    steps: [
      step({ states: [who(MO, from), who(SHI, from)] }),
      step({
        states: [
          who(MO, from, {
            transit: walking([from, to], 1, 3),
            activity_status: "moving",
            dominant_need: Need.selfActualization,
          }),
          who(SHI, from, { location: "", location_id: "", activity_status: "moving" }),
        ],
        actions: [
          did(MO, {
            ...shared,
            target: aimPlace(to.id, [SHI.id]),
            action_description: `携${SHI.name}即刻赶往${to.name}，务必在天黑前抵达。`,
            inner_monologue: `${SHI.name}就在眼前，此去${to.name}路远，独自走不如把他一并带上。`,
            outcome: `${MO.name}带着${SHI.name}从${from.name}动身前往${to.name}，路程约6小时。`,
          }),
          did(SHI, {
            ...shared,
            target: aimPlace(to.id, [SHI.id]),
            action_description: `参与${MO.name}发起的行动，行动意图：${intent}`,
            outcome: `${SHI.name}被${MO.name}强制拉入「${intent}」`,
          }),
        ],
      }),
      step({ states: [who(MO, to), who(SHI, to)] }),
    ],
  };
}
