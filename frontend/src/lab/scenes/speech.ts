import { ActionType, Deed, Phase } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { CHE, Emotion, LAN, LIN, MO, Need, SHI, YAN, aimNothing, aimTalk, did, execId, nextSeq, step, type LabActor, type LabScene, type Room, who } from "./fixtures";

/**
 * A conversation as the engine emits one: two records per beat sharing one `execution_id` (each
 * participant writes memory off his own record); the initiator's carries the deed and targets.
 *
 * The transcript lands on the closing beat, identically on both records and inside `outcome` after a
 * newline; the renderer's `seenDialogue` guard keeps the exchange from playing twice.
 */
function talkBeats(
  a: LabActor,
  b: LabActor,
  room: Room,
  lines: [LabActor, string][],
  overheard: LabActor[] = [],
  // The judge's verdict and, on failure, its reason; both ride on the closing beat only.
  succeeded = true,
  failure = "",
) {
  const eid = execId("talk", a, 1);
  const transcript = lines.map(([who, line]) => `${who.name}：${line}`).join("\n");
  const dialogue = lines.map(([who, line]) => ({ speaker_id: who.id, speaker: who.name, line }));
  const open = [
    did(a, {
      action_type: ActionType.talk,
      deed: Deed.talk,
      execution_id: eid,
      initiator_id: a.id,
      target: aimTalk(b.id),
      phase: Phase.begin,
      elapsed_steps: 1,
      total_steps: 2,
      duration_label: "约4小时",
      action_description: `上前向${b.name}攀谈，问${room.name}近日的绢价。`,
      outcome: `在${room.name}，${a.name}与${b.name}开始聊了起来。`,
    }),
    did(b, {
      action_type: ActionType.talk,
      deed: Deed.talk,
      execution_id: eid,
      initiator_id: a.id,
      target: aimNothing(), // the participant's opening record names nobody
      phase: Phase.begin,
      elapsed_steps: 1,
      total_steps: 2,
      duration_label: "约4小时",
      action_description: `上前向${b.name}攀谈，问${room.name}近日的绢价。`,
      outcome: `${b.name}被${a.name}邀入「上前向${b.name}攀谈，问${room.name}近日的绢价。」的交谈。`,
    }),
  ];
  // Closing beat: counters reset to 0/0, no duration, and the judge's verdict (`social.py`:
  // coerce_bool(success, True)), which the chip, card and pose all read.
  const close = [a, b].map((speaker) =>
    did(speaker, {
      action_type: ActionType.talk,
      deed: Deed.talk,
      phase: Phase.ongoing_complete,
      execution_id: eid,
      initiator_id: a.id,
      succeeded,
      ...(failure ? { failure_reason: failure } : {}),
      target: aimTalk(speaker === a ? b.id : a.id),
      action_description: `上前向${b.name}攀谈，问${room.name}近日的绢价。`,
      outcome: `在${room.name}，${a.name}与${b.name}的交谈：\n${transcript}`,
      gist: `在${room.name}，${a.name}与${b.name}交谈。`,
      dialogue,
      // Only on the initiator's record, as in the engine (completion_record reads it from the
      // initiator's applied effects); on both it would read as overheard twice.
      ...(speaker === a && overheard.length ? { overheard_by: overheard.map((o) => o.id) } : {}),
    }),
  );
  return { open, close };
}

/** What people say to each other, and what they send when they cannot. */
export function buildSpeechScenes({ se: SE, open: OPEN }: LabPlaces): LabScene[] {
  const TALK = talkBeats(MO, SHI, OPEN, [
    [MO, `这几日${OPEN.name}的绢价，涨得没有道理。`],
    [SHI, "涨的不是绢，是人心。"],
    [MO, "那你我便趁人心未定，先去看看货。"],
  ]);
  // The talk breaks down: same shape, opposite verdict. Its own scene because it examines how a
  // refusal is reported without narrating it twice.
  const BROKEN = talkBeats(
    MO,
    LIN,
    OPEN,
    [
      [MO, "这批绢，你我五五分。"],
      [LIN, "分不得。这本就是官中的东西。"],
      [MO, "官中的东西，也得有人经手。"],
      [LIN, "经手的是我，不是你。"],
    ],
    [],
    false,
    "对方不肯松口",
  );
  const OVERHEARD = talkBeats(
    MO,
    SHI,
    OPEN,
    [
      [MO, "东市那批货，明日三更进坊门。"],
      [SHI, "小声些。方才那位一直没走。"],
      [MO, "他听见便听见了，横竖拦不住。"],
    ],
    [LIN],
  );

  return [
    // ---- the other action types ---------------------------------------------
    {
      id: "talk",
      group: "言语",
      title: "talk · 交谈（联合行动，两拍）",
      watch:
        "开场一拍两人走到一起并起势；收场一拍才出对白。引擎给两人各发一份「相同」的 dialogue —— 对白应只播一遍，不是两遍。talk 两帧口型随轮次交替。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, SE)] }),
        step({
          states: [
            who(MO, OPEN, { activity_status: "talking", dominant_need: Need.social }),
            who(SHI, OPEN, { activity_status: "talking", dominant_need: Need.social }),
          ],
          actions: TALK.open,
        }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.trust, emotion_valence: 0.4 }),
            who(SHI, OPEN, { emotion: Emotion.trust, emotion_valence: 0.3 }),
          ],
          actions: TALK.close,
        }),
      ],
    },
    {
      id: "talk-broken",
      group: "言语",
      title: "talk · 谈崩了（收场判否）",
      watch:
        "和上一幕同一个形状,只有收场那一拍的判定相反。要看的是**失败怎么被报出来**:" +
        "①地图上那枚行动标签应当是失败样式,且写的是 `failure_reason`「对方不肯松口」——" +
        "**不是**把 outcome 截一段;②叙事流里失败的缘由单独一行(✗),对白照常全播——" +
        "谈崩了不等于没谈,那四句话都说出口了;③两人仍是一次联合行动,对白只播一遍。" +
        "拿它和上一幕对照着看:除了判定与那一行缘由,别处都不该有差别——" +
        "尤其**姿势和气泡不该因为判否就变**,他确实谈了。",
      steps: [
        step({ states: [who(MO, OPEN), who(LIN, SE)] }),
        step({
          states: [
            who(MO, OPEN, { activity_status: "talking", dominant_need: Need.esteem }),
            who(LIN, OPEN, { activity_status: "talking", dominant_need: Need.safety }),
          ],
          actions: BROKEN.open,
        }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.frustration, emotion_valence: -0.4 }),
            who(LIN, OPEN, { emotion: Emotion.anger, emotion_valence: -0.3 }),
          ],
          actions: BROKEN.close,
        }),
      ],
    },
    {
      id: "talk-overheard",
      group: "言语",
      title: "talk · 旁边有人听着",
      watch:
        "阿墨与阿石交谈，阿霖在同一处**听着但没插话**——他一回合都没花，所以两处都不该把他画成参与者。" +
        "①叙事流：卡片头部多一个暗色 chip「👂 阿霖 在旁听着」，名字染阿霖自己的身份色；" +
        "他**不进头像栈**、也**不进「阿墨 → 阿石」那行名字**——多一张脸就读成他参与了。" +
        "②地图：阿霖**转过身面向说话的人**，然后就没有别的了——没有气泡、不摆交谈姿势、没有连线。" +
        "那个「什么都没有」就是渲染本身。" +
        "③转身是**单向的**：阿墨和阿石在对谈，他们只互相看着，不该有谁转过来看阿霖。" +
        "④对白仍只播一遍（两条 close 记录同一份 dialogue），旁听不该让它播两次。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { activity_status: "talking", dominant_need: Need.social }),
            who(SHI, OPEN, { activity_status: "talking", dominant_need: Need.social }),
            who(LIN, OPEN),
          ],
          actions: OVERHEARD.open,
        }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.trust, emotion_valence: 0.3 }),
            who(SHI, OPEN, { emotion: Emotion.trust, emotion_valence: 0.3 }),
            who(LIN, OPEN, { emotion: Emotion.anticipation }),
          ],
          actions: OVERHEARD.close,
        }),
      ],
    },
    {
      id: "send-message",
      group: "言语",
      title: "send_message · 传信（定向）",
      watch:
        "一封信两拍，隔一步，各说各的事：第 2 拍是 show（递出）姿势 + 信封飞向阿霖，**不带字**（他此刻还不知道）；" +
        "第 3 拍才是「读到」——信在阿霖头上打开，标题 ✉ 阿墨、穿阿墨的身份色（和信封火漆同色），正文是信的内容。" +
        "两拍不该重复：第 2 拍不出字，第 3 拍不再飞一次。",
      steps: [
        step({ states: [who(MO, OPEN), who(LIN, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(LIN, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.send_message,
              deed: Deed.send_message,
              target: aimTalk(LIN.id),
              action_description: "写一封短笺，托人送给阿霖。",
              outcome: `在${OPEN.name}，阿墨遣人把书信送往阿霖处。`,
            }),
          ],
        }),
        step({
          states: [who(MO, OPEN), who(LIN, OPEN, { emotion: Emotion.anticipation })],
          messages: [
            {
              message_id: "9f2c41e0-5a7b-4d16-8c33-2b1f7a0e94dd",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [LIN.id],
              perceived_summary: "阿墨约你明日辰时在坊门相见。",
              spoken: "阿墨约你明日辰时在坊门相见。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        // A beat after the delivery: the "送达" (delivered) bubble is under examination, and a last
        // beat can be cut short when the scene ends.
        step({ states: [who(MO, OPEN), who(LIN, OPEN, { emotion: Emotion.anticipation })] }),
      ],
    },
    {
      id: "mail-many-readers",
      group: "言语",
      title: "传信 · 一封信两个收件人",
      watch:
        "一封信同时指名阿石和阿霖。**两个人都要持有**——两颗头顶各挂一个 ✉ 徽标，各自点开都是同一封信的全文。" +
        "指名私信的每一个收件人都是真的私下收到了，地图不得替其中一个判定「你没真收到」（「别把一封公告变成 N 封私信」" +
        "那条只管 place/world 广播，广播的受众是一个房间）。" +
        "但**只有一个会自动展开**（有聚焦时是被聚焦者，无聚焦时是收信最多的那个），另一个留徽标待点——拥挤靠「展开是稀缺的」来管，不靠扣着不给。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, SE)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, SE)],
          messages: [
            {
              message_id: "6f5a7182-93a4-b5c6-d7e8-f9a0b1c2d3e4",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [SHI.id, LIN.id],
              perceived_summary: "今夜三更，坊门东侧见，两位都来。",
              spoken: "今夜三更，坊门东侧见，两位都来。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, SE)] }),
      ],
    },
    {
      id: "mail-to-the-dead",
      group: "言语",
      title: "传信 · 收信人已倒下",
      watch:
        "信寄到时阿石已经倒下。**他不该挂 ✉ 徽标，也不该展开气泡**——读信是活人才做的事，" +
        "一具尸体顶着徽标、在自己遗体上摊开一封信，是地图在断言一次没发生过的认知。" +
        "同一封信也寄给了活着的阿霖，他照常收到：这一拍应当**只有阿霖**有动静。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)] }),
        step({
          states: [
            who(MO, OPEN),
            who(SHI, OPEN, { vitality: 0, is_active: false }),
            who(LIN, OPEN),
          ],
          messages: [
            {
              message_id: "7a6b8293-a4b5-c6d7-e8f9-a0b1c2d3e4f5",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [SHI.id, LIN.id],
              perceived_summary: "事情有变，你们两个都先别动。",
              spoken: "事情有变，你们两个都先别动。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN, { vitality: 0, is_active: false }), who(LIN, OPEN)],
        }),
      ],
    },
    {
      id: "mail-overflow",
      group: "言语",
      title: "传信 · 五封一起到",
      watch:
        "五个人同一步写到阿霖手上。气泡**只列前三封**，剩下的收成一行「…另 2 封」——" +
        "过了三行，要紧的已经不是「哪几封」而是「有多少封」，而数目用报的比用列的说得清（全文在叙事流里）。" +
        "徽标应显示「✉ 5」。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LAN, OPEN), who(YAN, OPEN), who(CHE, OPEN), who(LIN, SE)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LAN, OPEN), who(YAN, OPEN), who(CHE, OPEN), who(LIN, SE)],
          messages: [MO, SHI, LAN, YAN, CHE].map((who_, i) => ({
            message_id: `8b7c93a4-b5c6-d7e8-f9a0-b1c2d3e4f5${String(i).padStart(2, "0")}`,
            sender_id: who_.id,
            sender_name: who_.name,
            receiver_ids: [LIN.id],
            perceived_summary: `${who_.name}让你尽快回话。`,
            spoken: `${who_.name}让你尽快回话。`,
            scope: "direct" as const,
            place: "",
            place_id: "",
            seq: nextSeq(),
          })),
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LAN, OPEN), who(YAN, OPEN), who(CHE, OPEN), who(LIN, SE)] }),
      ],
    },
    {
      id: "mail-off-map",
      group: "言语",
      title: "传信 · 收信人不在图上",
      watch:
        "阿墨写给一个不在这张图上的人。信**离开**：沿弧线飞出世界的边缘、消失在视野外（`departingMessage`），" +
        "而不是原地什么都不画。信寄出去了，就要看见它走——去了哪里则正确地看不见。" +
        "下一拍没有任何送达，因为没有落点。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.send_message,
              deed: Deed.send_message,
              target: aimTalk("agent-not-on-this-map"),
              action_description: "写信托人捎往城外。",
              outcome: `在${OPEN.name}，阿墨遣人把书信送出城去。`,
            }),
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
      ],
    },
    {
      id: "mail-pile",
      group: "言语",
      title: "传信 · 同一步收到三封",
      watch:
        "三个人在同一步写到阿霖手上。①头顶挂一个「✉ 3」徽标——**不是红点**（红色在这张图上只归死亡标记，且红点意味「未读」这个本图不保存的状态）；" +
        "②三封都落在阿霖一个人手里，所以**只开一个气泡**、不是弹三次：标题条是**阿霖自己的名字和身份色**" +
        "（框挂在谁头上、尾巴指着谁，标题条就是谁的——和动作气泡同一条规矩），" +
        "正文每人一段，段首是**染了发信人身份色的名字**，段内是内容；" +
        "③徽标是这个框的**收起态**，两者不同时出现：展开时徽标收起（它正好落在气泡尾巴上，同时显示会读成尾巴脏了），淡出后回来，点它可重开；" +
        "④徽标随步清除，不许留到下一步。" +
        "要看的是它读起来是一件事——「他这一刻被三方同时找上」——而不是同一个位置连弹三次「有一封信」。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LAN, OPEN), who(LIN, SE)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LAN, OPEN), who(LIN, SE, { emotion: Emotion.fear, emotion_valence: -0.4 })],
          messages: [
            {
              message_id: "1a0b2c3d-4e5f-6071-8293-a4b5c6d7e8f9",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [LIN.id],
              perceived_summary: "阿墨要你今夜之前给个准话。",
              spoken: "阿墨要你今夜之前给个准话。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
            {
              message_id: "2b1c3d4e-5f60-7182-93a4-b5c6d7e8f9a0",
              sender_id: SHI.id,
              sender_name: SHI.name,
              receiver_ids: [LIN.id],
              perceived_summary: "阿石劝你别答应，那件事有诈。",
              spoken: "阿石劝你别答应，那件事有诈。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
            {
              message_id: "3c2d4e5f-6071-8293-a4b5-c6d7e8f9a0b1",
              sender_id: LAN.id,
              sender_name: LAN.name,
              receiver_ids: [LIN.id],
              perceived_summary: "阿岚说坊门已经有人守着了。",
              spoken: "阿岚说坊门已经有人守着了。",
              scope: "direct",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LAN, OPEN), who(LIN, SE)] }),
      ],
    },
    {
      id: "announce",
      group: "言语",
      title: "send_message · 传信（当众）",
      watch:
        "一条规则贯穿三种 scope：**送出画空间、不带字；送达画内容、带字**。" +
        "第 2 拍（送出）没有具名收信人，所以是向四周扩散的告示动效——涟漪说的是「他的声音传出去多远」，不出字；" +
        "第 3 拍（送达）内容才落地：place scope **绑在那个地点上**——地点名之上的通告，牌子顶上多一行眉标「◈ 阿墨 ◈」染他的身份色，两行共用一块牌（不是两块牌叠着），" +
        "world scope 走顶部横幅、眉标改成「◈ 发信人 传告 ◈」并染他的身份色。对照「全城通告（事件不上图）」那一场：**无来源**的地点广播不该有眉标（没有可指认的人，而坊名就在下面锚着），那条路径应一点没变。" +
        "**不该给每个听众头顶挂 ✉ 徽标**——徽标只属于 direct，公告的受众是一个地方、不是一串人头。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.send_message,
              deed: Deed.send_message,
              action_description: "在市中高声传话，让众人都听见。",
              outcome: `在${OPEN.name}，阿墨当众传话。`,
            }),
          ],
        }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)],
          messages: [
            {
              message_id: "4d3e5f60-7182-93a4-b5c6-d7e8f9a0b1c2",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [SHI.id, LIN.id],
              perceived_summary: "西市的绢价今日起按新章程走，各家自便。",
              spoken: "西市的绢价今日起按新章程走，各家自便。",
              scope: "place",
              place: OPEN.name,
              place_id: OPEN.id,
              seq: nextSeq(),
            },
          ],
        }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)],
          messages: [
            {
              message_id: "5e4f6071-8293-a4b5-c6d7-e8f9a0b1c2d3",
              sender_id: MO.id,
              sender_name: MO.name,
              receiver_ids: [SHI.id, LIN.id],
              perceived_summary: "自今日起，坊门宵禁提前一个时辰。",
              spoken: "自今日起，坊门宵禁提前一个时辰。",
              scope: "world",
              place: "",
              place_id: "",
              seq: nextSeq(),
            },
          ],
        }),
        step({ states: [who(MO, OPEN), who(SHI, OPEN), who(LIN, OPEN)] }),
      ],
    },
  ];
}
