/**
 * "身形" (build): which body each (gender, age) resolves to in this template's cast art, the whole
 * table at once. Bodies arrive as every figure's do (`setCast` → `CharacterSet.bodyFor`); nothing
 * here selects a sheet, since that is the answer being checked.
 */

import type { LabPlaces } from "../places";
import { BODIES, type LabScene, step, walking, who } from "./fixtures";

export function buildBodyScenes({ pivot: HUB, ne: NE, open: OPEN }: LabPlaces): LabScene[] {
  return [
    {
      id: "bodies-standing",
      group: "身形",
      title: "身形 · 八种身形同框（站）",
      watch:
        "男女各四档年龄一次全在场，站位从左到右就是 男童/男青/男壮/男老 · 女童/女青/女壮/女老。" +
        "要看的是这套人物资产**分不分得开**：孩童该明显矮小、老者该有别于壮年、男女该一眼可辨。" +
        "**两个挨着的看起来一模一样是合法结果**——说明这套资产把两档年龄映到了同一张图，" +
        "那是这份 characters 清单的事实，不是渲染错了。真正的错是错位（点名要壮年却画出孩童）" +
        "或者集体退回同一张兜底图。顺带看**染色**：八个人各有身份色，身上该染上各自的颜色。",
      steps: [step({ states: BODIES.map((a) => who(a, OPEN)) })],
    },
    {
      id: "bodies-walking",
      group: "身形",
      title: "身形 · 八种身形同框（走）",
      watch:
        "同样八个人，一起走同一段路。要看的是**每种身形都有自己的走路循环**：" +
        "谁要是原地滑行（贴图不换帧），就是这套资产缺了那一档的 walk 帧、退回了静止图。" +
        "步幅与身高该相称——孩童迈小步、老者慢，而不是八个人套同一套脚步。" +
        "左上角的帧读数是判据：看那一栏的帧名是 walk 还是 idle，别靠肉眼猜。",
      steps: [
        step({ states: BODIES.map((a) => who(a, HUB)) }),
        step({
          states: BODIES.map((a) =>
            who(a, HUB, { transit: walking([HUB, NE], 1, 1), activity_status: "moving" }),
          ),
        }),
        step({ states: BODIES.map((a) => who(a, NE)) }),
      ],
    },
  ];
}
