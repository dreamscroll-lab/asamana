import { useState } from "react";

import MapStage from "../components/MapStage";
import type { MapSource } from "../phaser/mapSource";
import type { TiledWorldScene } from "../phaser/TiledWorldScene";

/** One button press, as a zoom factor — about three wheel detents. */
const ZOOM_STEP = 1.4;

const button =
  "text-[11px] whitespace-nowrap px-2 py-0.5 rounded-md border transition bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white disabled:opacity-30";

/**
 * The map bench: the template's ground and place names with no step fed, i.e. exactly what
 * a world built on it starts from. Mounted through `MapStage`, so zoom and drag match the
 * observation view.
 */
export default function MapBench({ source }: { source: MapSource }) {
  const [scene, setScene] = useState<TiledWorldScene | null>(null);

  return (
    <main className="flex-1 min-w-0 h-full flex flex-col">
      <div className="relative flex-1 min-h-0">
        <MapStage source={source} onScene={setScene} />
      </div>
      <div className="shrink-0 flex items-center gap-1 px-3 py-2 border-t border-slate-800/70 bg-slate-950/50">
        <span className="text-[10px] whitespace-nowrap text-slate-500 mr-auto">
          Scroll to zoom · drag to pan
        </span>
        <button
          onClick={() => scene?.zoomBy(1 / ZOOM_STEP)}
          disabled={!scene}
          title="Zoom out"
          className={button}
        >
          −
        </button>
        <button
          onClick={() => scene?.zoomBy(ZOOM_STEP)}
          disabled={!scene}
          title="Zoom in"
          className={button}
        >
          +
        </button>
        <button
          onClick={() => scene?.resetView()}
          disabled={!scene}
          title="Fit the whole map"
          className={button}
        >
          Fit map
        </button>
      </div>
    </main>
  );
}
