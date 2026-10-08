import { SITED_PHENOMENA } from "../../phaser/weatherPlan";
import type { LabPlaces } from "../places";
import { type LabScene, MO, SHI, nextSeq, step, who } from "./fixtures";

// One scene for each entry in the engine's closed vocabulary except none. This table is the
// checklist: when the engine adds a phenomenon, add a row here and the renderer gets a check
// for it. Each phenomenon runs in both scopes, except sited ones (see SCOPES).
const PHENOMENA = [
  { key: "rain", label: "落雨", severity: "medium" },
  { key: "snow", label: "落雪", severity: "medium" },
  { key: "wind", label: "起风", severity: "low" },
  { key: "fire", label: "起火", severity: "high" },
  { key: "smoke", label: "浓烟", severity: "medium" },
  { key: "quake", label: "地动", severity: "high" },
  { key: "dark", label: "天昏", severity: "high" },
] as const;

// Two scopes: local (the ward's outline) and world-wide (the whole map). Same code path, but areas
// an order of magnitude apart fail differently: local tends to ignore scope (whole-camera shake,
// full-screen veil), world-wide tends to thin out until invisible.
//
// Sited phenomena (fire, smoke, quake) have no world-wide version: the engine lowers such a
// broadcast's phenomenon to none at the Broadcast boundary (core/interfaces/phenomenon.py).
const SCOPES = [
  { scoped: true, tag: "限于一地" },
  { scoped: false, tag: "全域" },
] as const;

/** Weather ("天象"): the only visual the engine names, and the only one it names from a closed set. */
export function buildWeatherScenes({ open: OPEN }: LabPlaces): LabScene[] {
  return [
    // A phenomenon is a fact from a closed vocabulary (`core/interfaces/phenomenon.py`), never a
    // rendering instruction. It rides the broadcast, the senderless place-scoped channel and the
    // only one the map resolves to coordinates.
    ...PHENOMENA.flatMap(({ key, label, severity }) =>
      // Sited phenomena only get the local scope; see the comment on SCOPES.
      SCOPES.filter(({ scoped }) => scoped || !SITED_PHENOMENA.includes(key)).map(({ scoped, tag }) => ({
      id: `weather-${key}-${scoped ? "local" : "global"}`,
      group: "天象",
      title: `${label}（${tag}）`,
      watch:
        `应当看到${label}${scoped ? `落在${OPEN.name}那片地界之内——是它在地图上的真实轮廓,不是套在外面的方框` : "铺满**整张地图**,而不是恰好铺满你此刻看到的这一块"}。` +
        "**粒子和世界一起缩放**:拉远变小、拉近变大,它是这个世界里的东西,不是贴在屏幕上的一层滤镜;" +
        "但拉到极远也不该细到消失——那时你该看见的是一片雨雾,不是一颗颗雨滴。" +
        "**而且不该有字幕板和地面波纹**:天象已经把这条消息说尽了,再叠一层广播表现就是同一件事演两遍。" +
        "**它应当在这一步的人动起来之前就已经在下,并一直持续到下一步开始**——天象不是对这一步的评论," +
        "而是这一步发生所在的条件,人是在雨里行动的;所以不该是「演完再下、下一小会儿就散」。" +
        "**最后一步是晴天,专看它怎么停**:不该一帧全没,而是不再落新的、天上的那些各自落完," +
        "暗幕与火光缓缓退去——尾巴压到最后一步里是对的,一场雨本来就不在两帧之间结束。" +
        (scoped
          ? "拖动地图时它跟着那片地界走(它在那儿发生),不会黏在屏幕上。"
          : "拖动地图时它跟着地图走:落在地图之外的地方不该有雨——「全域」说的是世界,不是你的视野。") +
        "数量由 severity 决定——引擎已有的「这事多大」刻度,渲染层不另造一把尺。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          broadcasts: [
            {
              content: `${label}。`,
              broadcast_type: "world_event",
              location_scope: scoped ? OPEN.id : null,
              location_name: scoped ? OPEN.name : "",
              severity,
              phenomenon: key,
              seq: nextSeq(),
            },
          ],
        }),
        // Clear sky: shows how the previous step's weather clears.
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
      ],
      })),
    ),

    {
      id: "weather-unknown-phenomenon",
      group: "天象",
      title: "不认识的天象（应当什么都不画）",
      watch:
        "应当**退回普通广播的表现**(字幕板 + 地面波纹),一个粒子都不该有。" +
        "引擎日后新增一种现象时,旧渲染器会原样收到那个词——凭空画一种谁也没报告过的天气很糟," +
        "但让广播跟着一起沉默更糟:那条消息就彻底没人看见了。所以让位只在**真画出了东西**时发生。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        step({
          states: [who(MO, OPEN)],
          broadcasts: [
            {
              content: "天边出现了某种从未见过的景象。",
              broadcast_type: "world_event",
              location_scope: OPEN.id,
              location_name: OPEN.name,
              severity: "high",
              phenomenon: "aurora",   // not in the closed vocabulary
              seq: nextSeq(),
            },
          ],
        }),
      ],
    },

    {
      id: "weather-severity-scale",
      group: "天象",
      title: "同一场雨的三种规模",
      watch:
        "三步:小雨 → 中雨 → 大雨,同一个 phenomenon、只有 severity 在变。" +
        "要看的不只是**变密**:雨还应当**更急**(落得更快、雨丝被拉长)、**更粗**、**更亮**。" +
        "严重程度是个四维的小向量(多少/多急/多大/多显),不是一个数量旋钮——只改根数会读作" +
        "「同一场雨多了几根」。对照「落雪」:雪下得大只会更密更大,**不会更快**,否则它就成了白色的雨。" +
        "顺带看每一步的雨都活满整步,以及**三场雨是怎么接上的**:上一场不会在下一场开始的那一帧" +
        "凭空消失,而是一边收尾一边被下一场盖过去。三档之间因此是连续变密,不是三次「清空重下」。",
      steps: [
        step({ states: [who(MO, OPEN)] }),
        ...(["low", "medium", "high"] as const).map((severity) =>
          step({
            states: [who(MO, OPEN)],
            broadcasts: [
              {
                content: `雨势${severity === "low" ? "初起" : severity === "medium" ? "渐密" : "如注"}。`,
                broadcast_type: "world_event",
                location_scope: OPEN.id,
                location_name: OPEN.name,
                severity,
                phenomenon: "rain",
                seq: nextSeq(),
              },
            ],
          }),
        ),
      ],
    },
  ];
}
