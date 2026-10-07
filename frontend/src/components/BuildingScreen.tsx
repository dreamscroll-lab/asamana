// The world-building wait. Its own component so /lab/loading can show it without
// paying for a build.
import { useEffect, useState } from "react";

import EmergingWorld from "./EmergingWorld";

const STATUS = [
  "随便走进一家街角的小店，点一杯热咖啡感受烫手的温度……",
  "喝一口冰可乐，体验一下什么叫真正冰爽与快乐……",
  "找一个公园的躺椅，或者干脆是一片草地，瘫躺着什么都不干……",
  "看着头顶的树叶被风吹得沙沙作响，任凭思维像断了线的风筝一样乱飘……",
  "走在人潮拥挤的街道上，去观察每一个擦肩而过的人……",
  "在熙熙攘攘的人群中，看他们的眼神、听他们的笑声……",
  "去蹦极，感受那种自由落地的刺激……",
];
const STATUS_MS = 2500;

export default function BuildingScreen() {
  const [stage, setStage] = useState(0);

  // Pure theatre: the build job reports only building/ready/failed, so there is no
  // real progress to show.
  useEffect(() => {
    const t = setInterval(() => setStage((s) => (s + 1) % STATUS.length), STATUS_MS);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 text-center relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[680px] h-[680px] bg-indigo-500/15 rounded-full blur-3xl animate-pulse pointer-events-none" />
      {/* Not a spinner. The wait is a world being built, so it is drawn as one:
          see EmergingWorld for why it turns slowly and fills up rather than loops. */}
      <div className="relative mb-10">
        <EmergingWorld />
      </div>
      {/* Wide enough for the longest line to stay on one line (each is read once, in 2.5s):
          at text-lg a full-width glyph is ~18px plus tracking, so max-w-2xl (672px) fits
          ~36 characters. A longer line needs this widened or the line shortened. */}
      <p className="text-lg text-indigo-200/90 italic max-w-2xl tracking-wide">
        {STATUS[stage]}
      </p>
    </div>
  );
}
