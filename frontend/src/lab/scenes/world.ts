import { ActionType, Deed, Phase } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { CHE, Emotion, LAN, LIN, MO, Need, SHI, YAN, aimAgents, aimNothing, did, nextSeq, step, type LabScene, who } from "./fixtures";

/** The channels nobody owns: seeding, injected events, broadcasts, deaths, the clock. */
export function buildWorldScenes({ se: SE, far: FAR, open: OPEN, tight: TIGHT }: LabPlaces): LabScene[] {
  return [
    // ---- world-level channels ------------------------------------------------
    {
      id: "initialization",
      group: "世界",
      title: "初始化 · 第 0 步就位",
      watch:
        "step 0 的播种记录：action_type 与 deed 都是空串，outcome 恒为「初始化完成」。就位不是一件行动，" +
        "所以地图对它一拍都不演——不出气泡、不出结果标签、不摆姿态，人只是站在那里，画面定格约 1.8 秒供人看清谁在哪。",
      steps: [
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, SE)],
          actions: [MO, SHI].map((a) =>
            did(a, {
              action_type: "",
              deed: "",
              phase: Phase.initialization,
              target: aimNothing(),
              action_description: `在 ${OPEN.name} 就位`,
              outcome: "初始化完成",
            }),
          ),
          hour: 6,
        }),
      ],
    },
    {
      id: "death",
      group: "世界",
      title: "死亡 · 先行动后倒下",
      watch:
        "致命一击必须「先演完」，受击者才转灰、才挂上红色 OVER；顺序反了就是「人先死后行动」。" +
        "死讯是无发送者的世界广播，且落在下一拍——死讯在执行阶段才发布，deliver_step=次步（death_handler.py）。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.anger, emotion_valence: -0.8, emotion_intensity: 1 }),
            who(SHI, OPEN, { vitality: 0, is_active: false, emotion: Emotion.fear }),
          ],
          actions: [
            did(MO, {
              action_type: ActionType.physical,
              deed: Deed.strike,
              target: aimAgents(SHI.id),
              action_description: "拔刀刺向阿石。",
              outcome: `在${OPEN.name}，阿墨一刀刺中阿石的要害。`,
            }),
          ],
        }),
        // The news comes a step later: the engine publishes deaths with deliver_step=step+1.
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.frustration, emotion_valence: -0.5 }),
            who(SHI, OPEN, { vitality: 0, is_active: false, emotion: Emotion.fear }),
          ],
          broadcasts: [
            {
              content: "阿石(市井行商)被利刃刺中要害而亡。",
              broadcast_type: "world_event",
              location_scope: null,
              location_name: "",
              severity: "high",
              phenomenon: "none",
              seq: nextSeq(),
            },
          ],
        }),
        // The engine never drops the dead from agent_states, so the payload keeps saying he's dead
        // and the map must keep not showing him.
        step({ states: [who(MO, OPEN), who(SHI, OPEN, { vitality: 0, is_active: false })] }),
        step({
          states: [
            who(MO, OPEN),
            who(LIN, OPEN),
            who(SHI, OPEN, { vitality: 0, is_active: false }),
          ],
        }),
      ],
    },
    {
      id: "death-offscreen",
      group: "世界",
      title: "死亡 · 倒在原地",
      watch:
        "阿石无声无息地死去（没有任何行动指向他），同拍阿霖走进同一处——屋里的人一变，" +
        "resting slot 就会重排。阿石必须**倒在他站着的那一格**，不许先走到新位子上再倒下；" +
        "一具会自己挪窝的尸体就是这条错了。下一步他消失，且不再回来。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN),
            who(LIN, OPEN, { emotion: Emotion.fear }),
            who(SHI, OPEN, { vitality: 0, is_active: false }),
          ],
          actions: [
            did(LIN, {
              action_type: ActionType.work,
              deed: Deed.exert,
              action_description: "四下张望。",
              outcome: `在${OPEN.name}，阿霖四下张望。`,
            }),
          ],
        }),
        step({
          states: [
            who(MO, OPEN),
            who(LIN, OPEN),
            who(SHI, OPEN, { vitality: 0, is_active: false }),
          ],
          broadcasts: [
            {
              content: "阿石(市井行商)油尽灯枯。",
              broadcast_type: "world_event",
              location_scope: null,
              location_name: "",
              severity: "high",
              phenomenon: "none",
              seq: nextSeq(),
            },
          ],
        }),
        step({
          states: [
            who(MO, OPEN),
            who(LIN, OPEN),
            who(SHI, OPEN, { vitality: 0, is_active: false }),
          ],
        }),
      ],
    },
    {
      id: "world-event",
      group: "世界",
      title: "全城通告（事件不上图）",
      watch:
        "应当只看到一块公告——全城通告的 HUD。这一步同时带着一条 world_event，" +
        "但地图刻意不画它：EventSystem 只在投递成功后才记事件，narrative_desc 是对刚投出去那条广播的概括，" +
        "画上去就是同一件事被演两遍。事件的地点/波及面也只有叙事层指称（名字、地名），" +
        "地图要的几何在广播的 location_scope 上，不在事件上。" +
        "HUD 本身在缩放/拖动时应保持在屏幕上不变形。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.frustration }),
            who(SHI, OPEN, { emotion: Emotion.frustration }),
          ],
          // As in the engine, one LLM call wrote both: narrative_desc summarizes the broadcast
          // below, not a second happening.
          events: [
            {
              id: "2eafb783-33a0-44a2-89ba-bba9a1efd579",
              authored_by: "system",
              narrative: `京兆府出榜示禁，${OPEN.name}摊贩闻之仓皇收货。`,
              affected_names: [],
              location_label: null,
              is_positive: null,
              directive_text: "",   // event editor output doesn't grow from a directive; only director injections carry one
              receipt: null,
              seq: nextSeq(),
            },
          ],
          broadcasts: [
            {
              content: "京兆府出榜：申时闭坊门，违者论罪。",
              broadcast_type: "world_event",
              location_scope: null,
              location_name: "",
              severity: "medium",
              phenomenon: "none",
              seq: nextSeq(),
            },
          ],
          hour: 14,
        }),
      ],
    },
    {
      id: "place-broadcast",
      group: "世界",
      title: "地点传讯（只惊动一处）",
      watch:
        "同一条广播带了 location_scope 就不再上顶部横幅，而是落在那个地点上：字幕贴在地点名之上、地面荡开涟漪。" +
        `看字幕与「${OPEN.name}」二字是否互不挤压——它们是两样东西，不该叠在一起。`,
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.fear }),
            who(SHI, OPEN, { emotion: Emotion.fear }),
          ],
          broadcasts: [
            {
              content: `${OPEN.name}南墙塌了一角，尘土扑面，摊贩四散。`,
              broadcast_type: "world_event",
              location_scope: OPEN.id,
              location_name: OPEN.name,
              severity: "medium",
              phenomenon: "none",
              seq: nextSeq(),
            },
          ],
        }),
      ],
    },
    {
      id: "narrator-message",
      group: "世界",
      title: "无来源的密报",
      watch:
        "事件注入的消息没有真正的发送者：sender_id 是 narrator，名字是「不知来源」。地图上没有任何人做出「递出」的动作——" +
        "信封 ✉？ 从世界之外飞进来落在收信人身上，再打开成正文，标题「✉ 不知来源」用中性石板色（不属于任何人的身份色）。" +
        "「看得见它来了，也看得见你看不见它从哪来」——飞在空中和摊开在手上，必须是同一种颜色。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.fear, emotion_valence: -0.5, dominant_need: Need.safety }),
            who(SHI, OPEN),
          ],
          messages: [
            {
              message_id: "1a3d2719-d81e-4389-b41b-11b4b8532b99",
              sender_id: "narrator",
              sender_name: "不知来源",
              receiver_ids: [MO.id],
              perceived_summary: `急报：${OPEN.name}有人在打听你的行踪，speak 慎言。`,
              spoken: `急报：${OPEN.name}有人在打听你的行踪，speak 慎言。`,
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
      ],
    },
    {
      id: "night",
      group: "世界",
      title: "昼夜 · 同一场戏的五个时段",
      watch:
        "日夜色温随「机器时钟」（world_time.hour）推移，而不是叙事标签。五个时段各一拍，覆盖 setWorldTime 的全部分支。关掉跟随、按「全图」缩到整城再看：" +
        "①色温层是屏幕空间的，缩放下必须仍然盖满视口，缺一角就是它又退化成世界矩形了；" +
        "②入夜后只有「有人站着」的三处亮灯——空城其余各坊一律漆黑，这是「零星」的全部含义；" +
        "③人多的那处灯更大更亮；④正午灯全灭；⑤任何时段坊名的颜色都不该变（灯是加光，不是把字调暗）。",
      // Hour 0: the darkest band (alpha 0.36), where a tint covering only part of the screen is
      // easiest to spot. People at three densities because lights follow where people are.
      steps: [0, 6, 12, 18, 23].map((h) =>
        step({
          states: [
            who(MO, OPEN), who(SHI, OPEN),
            who(LIN, FAR),
            who(LAN, TIGHT), who(YAN, TIGHT), who(CHE, TIGHT),
          ],
          hour: h,
        }),
      ),
    },
    {
      id: "director-event-landed",
      group: "世界",
      title: "导演注入 · 有人因此改了行止",
      watch:
        "导演的注入和世界自己的波折共用一张卡，但**绝不该长得一样**——分不清哪几拍是自己造成的，" +
        "这条流就不能当实验记录看。要看的是：①靛蓝底、🎬、以及「导演」那枚角标；" +
        "②注入的**原话**单独一行（斜体「导演说：…」），压在引擎改写出的那句之下——" +
        "这两句能对照，才看得出是话没说清还是世界不理会；" +
        "③回执四行**各占一行**（送达／压力／因此决策／打断），标签一列、名字一列——" +
        "挤成一段话时「打断」二字会落在折行中间，根本找不着。" +
        "④出现在压力里却不在决策里的人是有意义的空缺，别把四行合并。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.fear, emotion_valence: -0.5 }),
            who(SHI, OPEN, { emotion: Emotion.anticipation }),
          ],
          events: [
            {
              id: "0f3c81d5-9a27-4d10-8f6b-5c2ad7ee1b44",
              authored_by: "director",
              narrative: `一队甲士自北面开进${OPEN.name}，当街封了出口。`,
              affected_names: [MO.name, SHI.name],
              location_label: OPEN.name,
              is_positive: false,
              directive_text: "让禁军封锁市口",
              receipt: {
                delivered_to: [
                  { agent_id: MO.id, name: MO.name },
                  { agent_id: SHI.id, name: SHI.name },
                ],
                pressure: [{ agent_id: MO.id, name: MO.name, urgency: "high" }],
                decided: [{ agent_id: MO.id, name: MO.name }],
                interrupted: [{ agent_id: MO.id, name: MO.name }],
              },
              seq: nextSeq(),
            },
          ],
          broadcasts: [
            {
              content: `甲士封了${OPEN.name}的出口。`,
              broadcast_type: "world_event",
              location_scope: OPEN.id,
              location_name: OPEN.name,
              severity: "high",
              phenomenon: "none",
              seq: nextSeq(),
            },
          ],
        }),
      ],
    },
    {
      id: "director-event-ignored",
      group: "世界",
      title: "导演注入 · 送到了，无人理会",
      watch:
        "**这一幕才是回执存在的理由**：注入被解析、被接受、准确送到了两个人的感官里，" +
        "而压力评估判定它与谁都不相干——于是世界照旧。要看的是那句否定的交代" +
        "（「无人为此改变行止」）确实出现，且它是**回执的判词、不是第五列名字**：" +
        "它该自成一行、不带标签。没有它，「送达了但没人动」和「注入根本没生效」" +
        "在这条流里长得一模一样，而这两件事一个是世界的回答、一个是 bug。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          events: [
            {
              id: "7b1e4c06-2d88-4f35-91aa-30ec5b9f7c21",
              authored_by: "director",
              narrative: "坊墙外传来一阵不知何人的哭声。",
              affected_names: [MO.name, SHI.name],
              location_label: OPEN.name,
              is_positive: null,
              directive_text: "让他们听见有人在哭",
              receipt: {
                delivered_to: [
                  { agent_id: MO.id, name: MO.name },
                  { agent_id: SHI.id, name: SHI.name },
                ],
                pressure: [],
                decided: [],
                interrupted: [],
              },
              seq: nextSeq(),
            },
          ],
          broadcasts: [
            {
              content: "坊墙外有人在哭。",
              broadcast_type: "world_event",
              location_scope: OPEN.id,
              location_name: OPEN.name,
              severity: "low",
              phenomenon: "none",
              seq: nextSeq(),
            },
          ],
        }),
      ],
    },
  ];
}
