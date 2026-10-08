import { ActionType, Deed, Phase } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { Emotion, MO, Need, SHI, aimAgents, aimTalk, did, step, type LabScene, who } from "./fixtures";

/** Acts that outlast a step: their opening, their ticks, their close — and being cut short. */
export function buildOngoingScenes({ open: OPEN }: LabPlaces): LabScene[] {
  return [
    {
      id: "work",
      group: "长活",
      title: "work · 多步长活（开始→进行中→完成·判否）",
      watch:
        "interact 姿势要「跨步保持」，不能每步闪回站立；中间拍是进度条、不重放开场；收场拍计数归 0/0（引擎的写法），只报结果。" +
        "**这一幕的收场是判否的**——账点完了但对不上，所以标签走失败样式、写的是 `failure_reason`「账目与实物对不上」。" +
        "判是判否由裁判决定、默认判是（`work.py`），所以这不是收场拍的通例：成功收场去看「talk · 交谈」与「rest · 歇息」。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [who(MO, OPEN, { activity_status: "working", dominant_need: Need.esteem })],
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: Deed.work,
              phase: Phase.begin,
              elapsed_steps: 1,
              total_steps: 3,
              duration_label: "约6小时",
              action_description: "清点仓中的绢帛，逐匹核对数目。",
              outcome: `在${OPEN.name}，阿墨着手清点仓中的绢帛。`,
            }),
          ],
        }),
        ...[2, 3].map((n) =>
          step({
            states: [who(MO, OPEN, { activity_status: "working", dominant_need: Need.esteem })],
            actions: [
              did(MO, {
                action_type: ActionType.work,
                deed: Deed.work,
                phase: Phase.ongoing_tick,
                elapsed_steps: n,
                total_steps: 3,
                action_description: "清点仓中的绢帛，逐匹核对数目。",
                outcome: `在${OPEN.name}，阿墨正进行「清点仓中的绢帛，逐匹核对数目。」，已持续约${n * 2}小时。`,
              }),
            ],
          }),
        ),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.frustration, emotion_valence: -0.3 })],
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: Deed.work,
              phase: Phase.ongoing_complete,
              succeeded: false,
              action_description: "清点仓中的绢帛，逐匹核对数目。",
              outcome: `在${OPEN.name}，阿墨点完了仓中绢帛，短了三十匹。`,
              // Its own field: the feed prints it on its own line and the chip shows it instead of
              // a truncated narration. Never parsed out of `outcome`.
              failure_reason: "账目与实物对不上",
            }),
          ],
        }),
      ],
    },
    {
      id: "interrupt",
      group: "长活",
      title: "中断 · 长活被打断",
      watch:
        "报告的是「被打断」这件事本身：琥珀色的 ✕ 浮字升起后散去，不重演被打断的那件事，也不在结果位上留一枚常驻标签——中断是一个瞬间，不是这一步的结果。心声在 outcome 换行之后，不该进浮字。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN, { activity_status: "working" }), who(SHI, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: Deed.work,
              phase: Phase.begin,
              elapsed_steps: 1,
              total_steps: 2,
              duration_label: "约4小时",
              action_description: "搬石补砌坍了半边的坊墙。",
              outcome: `在${OPEN.name}，阿墨着手修补坊墙。`,
            }),
          ],
        }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.frustration }), who(SHI, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.work,
              deed: Deed.work,
              phase: Phase.interrupt,
              succeeded: false,
              // An agent ID, as the engine sends it — never a name.
              interrupted_by: SHI.id,
              action_description: "搬石补砌坍了半边的坊墙。",
              outcome: `在${OPEN.name}，阿墨修补坊墙的活计被阿石打断。\n阿墨当时的心思：他这时候来找我，必不是为了墙。`,
              gist: `在${OPEN.name}，阿墨修补坊墙的活计被阿石打断。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "not-executed",
      group: "长活",
      title: "未成事 · 意图从未落地",
      watch:
        "地图上应当「什么都不演」（没有姿势、没有特效、没有失败标记）——这一条是看「没有发生」。注意 outcome 是「本想…却因…」句式，没有地点前缀。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [who(MO, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.talk,
              deed: Deed.talk,
              not_executed: true,
              succeeded: false,
              target: aimTalk(SHI.id),
              action_description: "找阿石问清昨日货栈的账目。",
              outcome: `阿墨本想找阿石问清昨日货栈的账目。，却因阿石不在${OPEN.name}未能如愿。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "foiled-physical",
      group: "长活",
      title: "未成事 · 动手却扑空（deed 为空）",
      watch:
        "裁决压根没发生，所以引擎给的 deed 是空串——渲染器必须据此「不画任何动作」，而不是回退成挥空拳。这是线上真实世界里唯一出现过的 physical。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.anger, emotion_valence: -0.5 })],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: "", // adjudication never happened
              not_executed: true,
              succeeded: false,
              target: aimAgents(SHI.id),
              action_description: "冲上去拦住阿石，夺下他手里的东西。",
              outcome: `在${OPEN.name}，阿墨本想做「冲上去拦住阿石，夺下他手里的东西。」，却因阿石不在${OPEN.name}，无法对其施加物理行动未能做成。`,
            }),
          ],
        }),
      ],
    },
    {
      id: "rest",
      group: "长活",
      title: "rest · 歇息（多步：坐下→歇着→起身）",
      watch:
        "歇息是**状态**，不是一拍：第二拍蹲下之后，中间每一拍都得**保持**蹲着，" +
        "不能每步闪回站立；只有收场那一拍才起身。这一场是多步的，因为引擎本来就把 REST " +
        "按估算时长铺开跑（engine/executors/simple.py），单步版验不到「跨步保持」这件唯一会坏的事。" +
        "姿势用的是蹲姿——这套人物没有坐姿帧，而唯一「躺下」的帧是倒地死亡帧，" +
        "拿来当歇息会让活人和尸体共用一个身体。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, {
              activity_status: "resting",
              emotion: Emotion.joy,
              emotion_valence: 0.4,
              dominant_need: Need.safety,
              vitality: 0.999,
            }),
          ],
          actions: [
            did(MO, {
              action_type: ActionType.rest,
              deed: Deed.rest,
              phase: Phase.begin,
              elapsed_steps: 1,
              total_steps: 3,
              duration_label: "约6小时",
              action_description: "在市角寻个背风处坐下歇脚。",
              outcome: `在${OPEN.name}，阿墨在市角坐下歇脚。`,
            }),
          ],
        }),
        ...[2, 3].map((n) =>
          step({
            states: [
              who(MO, OPEN, {
                activity_status: "resting",
                emotion: Emotion.joy,
                emotion_valence: 0.4,
                dominant_need: Need.safety,
                vitality: 0.999,
              }),
            ],
            actions: [
              did(MO, {
                action_type: ActionType.rest,
                deed: Deed.rest,
                phase: Phase.ongoing_tick,
                elapsed_steps: n,
                total_steps: 3,
                action_description: "在市角寻个背风处坐下歇脚。",
                outcome: `在${OPEN.name}，阿墨仍在市角歇着，已歇了约${n * 2}小时。`,
              }),
            ],
          }),
        ),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.joy, emotion_valence: 0.5, vitality: 1 })],
          actions: [
            did(MO, {
              action_type: ActionType.rest,
              deed: Deed.rest,
              phase: Phase.ongoing_complete,
              // REST cannot fail on the wire: `simple.py` hardcodes succeeded=True.
              action_description: "在市角寻个背风处坐下歇脚。",
              outcome: `在${OPEN.name}，阿墨歇够了，起身拍了拍衣上的尘土。`,
            }),
          ],
        }),
        // One empty beat after he gets up, to show he stays up rather than sliding back into a crouch.
        step({ states: [who(MO, OPEN)] }),
      ],
    },
  ];
}
