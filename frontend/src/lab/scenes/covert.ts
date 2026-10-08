import { ActionType, Deed } from "../../lib/contract";
import type { LabPlaces } from "../places";
import { type LabScene, Emotion, MO, Need, SHI, did, step, who } from "./fixtures";

export function buildCovertScenes({ open: OPEN }: LabPlaces): LabScene[] {
  return [
    {
      id: "covert",
      group: "潜行",
      title: "covert · 潜行",
      watch: "duck（伏低）并且从这一拍起就变淡（shroud 0.42）+ 收拢的暗环；第三拍他什么也没做，伏低应当解除。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [
            who(MO, OPEN, { activity_status: "covert", emotion: Emotion.anticipation, dominant_need: Need.safety }),
            who(SHI, OPEN),
          ],
          actions: [
            did(MO, {
              action_type: ActionType.covert,
              deed: Deed.covert,
              action_description: "伏在货堆之后，窥探阿石的动静。",
              outcome: `在${OPEN.name}，阿墨伏身隐入货堆之后。`,
            }),
          ],
        }),
        // The crouch is sustained, and is released here by the figure doing nothing (see deedPose.ts).
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
      ],
    },
    {
      id: "covert-spotted",
      group: "潜行",
      title: "covert · 潜行被发觉",
      watch: "同样伏低，但「不变淡」（被看见就等于没藏住），暗环转红。",
      steps: [
        step({ states: [who(MO, OPEN), who(SHI, OPEN)] }),
        step({
          states: [who(MO, OPEN, { emotion: Emotion.fear, emotion_valence: -0.5 }), who(SHI, OPEN)],
          actions: [
            did(MO, {
              action_type: ActionType.covert,
              deed: Deed.covert,
              succeeded: false,
              // Independent of `succeeded`: the veil colors off `detected` alone, so a clean
              // getaway that found nothing stays purple.
              detected: true,
              action_description: "伏在货堆之后，窥探阿石的动静。",
              outcome: `在${OPEN.name}，阿墨刚一伏身便被阿石看破。`,
              failure_reason: "阿石早有察觉",
            }),
          ],
        }),
      ],
    },
  ];
}
