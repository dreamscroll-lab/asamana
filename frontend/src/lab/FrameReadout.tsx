import { useEffect, useState } from "react";

import type { TiledWorldScene } from "../phaser/TiledWorldScene";

/** Poll faster than the walk cycle (8fps ≈ 125ms) so no frame goes unseen. */
const SAMPLE_MS = 80;

/** One figure on the map, as the readout names it: the step's own name and colour. */
export interface Figure {
  id: string;
  name: string;
  color: string;
}

interface Row extends Figure {
  dir: string;
  anim: string | null;
  frame: string;
}

/**
 * What each figure is actually drawing: its heading, the animation playing, and the atlas frame.
 *
 * At 96px a still of `walkSE1` and of `idleSE` look alike, but their frame names don't, so the
 * bench reads the name off the sprite: a wrong heading, or a pose falling back to a still because
 * the manifest lacks an entry, shows up in words.
 *
 * Takes the figures from the scene rather than a fixed cast, since scenes stage extra bodies
 * (an errand's runners). Polled, not subscribed: Phaser has no per-frame hook out to React. Its own
 * component so the 80ms ticks re-render only this strip.
 */
export default function FrameReadout({
  scene,
  figures,
}: {
  scene: TiledWorldScene | null;
  figures: Figure[];
}) {
  const [rows, setRows] = useState<Row[]>([]);

  useEffect(() => {
    if (!scene) {
      setRows([]);
      return;
    }
    const id = setInterval(() => {
      setRows(
        figures.flatMap((who) => {
          const tok = scene.token(who.id);
          if (!tok) return [];
          return [
            {
              ...who,
              dir: tok.dir,
              anim: tok.sprite.anims.currentAnim?.key ?? null,
              frame: String(tok.sprite.frame.name),
            },
          ];
        }),
      );
    }, SAMPLE_MS);
    return () => clearInterval(id);
  }, [scene, figures]);

  if (!rows.length) return null;

  return (
    <div className="absolute top-2 left-2 pointer-events-none rounded-md bg-slate-950/80 border border-slate-800 px-2 py-1.5 font-mono text-[11px] leading-relaxed">
      {rows.map((r) => (
        <div key={r.id} className="flex items-center gap-2 whitespace-nowrap">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: r.color }} />
          <span className="text-slate-300 w-8">{r.name}</span>
          <span className="text-indigo-300 w-7">{r.dir}</span>
          {/* No anim key = a still: no cycle was registered for this pose. */}
          <span className="text-amber-200">{r.frame}</span>
          <span className="text-slate-600">{r.anim ? "▶" : "■"}</span>
        </div>
      ))}
    </div>
  );
}
