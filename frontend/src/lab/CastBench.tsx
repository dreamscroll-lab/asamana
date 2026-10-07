import { useEffect, useMemo, useState } from "react";

import { loadSheet, type FrameRect, type Sheet } from "../lib/atlasFrames";
import { bodyCells, frameStyle, poseRow, unusedFrames, type Cell } from "./castSheet";
import { type CharacterSet, DIRECTIONS, POSE_NAMES } from "../phaser/skins";

/** Display names for the body brackets, where they differ from the manifest keys. */
const BRACKET_LABEL: Record<string, string> = { middle: "adult" };
const GENDER_LABEL: Record<string, string> = { male: "M", female: "F" };

/**
 * The cast bench: the whole delivered pose table (11 poses × 4 headings, see
 * CHARACTER_ASSET_SPEC.md §3) on screen at once, for acceptance checks done by eye. A running
 * world shows only the poses its cast happens to choose.
 *
 * Drawn by clipping the sheet with CSS, not through the renderer: the question is what is in
 * this rectangle of the image. Only which frames a pose resolves to comes from the render path
 * (see castSheet).
 *
 * Every frame is the same size and anchored at the foot (§3.3), so fixed, bottom-aligned cells
 * make an anchor that drifts between poses visible — the most common delivery defect.
 */
export default function CastBench({ cast }: { cast: CharacterSet }) {
  const cells = useMemo(() => bodyCells(cast), [cast]);
  const [body, setBody] = useState<string | null>(null);
  const [zoom, setZoom] = useState(0.5);
  const [playing, setPlaying] = useState(true);
  const [tick, setTick] = useState(0);
  const [stack, setStack] = useState(false);
  const [outline, setOutline] = useState(true);
  // null = still loading; a null entry = that one body's art could not be read.
  const [sheets, setSheets] = useState<Record<string, Sheet | null> | null>(null);

  useEffect(() => {
    let live = true;
    setSheets(null);
    Promise.all(
      cast.atlases.map(
        async (a) => [a.key, await loadSheet(a.image, a.atlas).catch(() => null)] as const,
      ),
    ).then((pairs) => live && setSheets(Object.fromEntries(pairs)));
    return () => {
      live = false;
    };
  }, [cast]);

  // One clock for the whole table, so a walk cycle can be compared across headings
  // instead of four cells drifting apart.
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setTick((t) => t + 1), 1000 / cast.walkFps);
    return () => clearInterval(timer);
  }, [playing, cast.walkFps]);

  const chosen = body ?? cells.find((c) => c.atlas)?.atlas ?? null;
  const art = cast.atlases.find((a) => a.key === chosen);
  const sheet = (chosen && sheets ? sheets[chosen] : null) ?? null;
  const frames = sheet?.frames;

  const rows = useMemo(
    () =>
      chosen && frames
        ? POSE_NAMES.map((pose) => ({ pose, row: poseRow(cast, chosen, pose, frames) }))
        : [],
    [cast, chosen, frames],
  );
  const tally = useMemo(() => {
    const flat = rows.flatMap(({ row }) => Object.values(row));
    return {
      unmapped: flat.filter((c) => c.kind === "unmapped").length,
      missing: flat.filter((c) => c.kind === "missing").length,
      unused: chosen && frames ? unusedFrames(cast, chosen, frames).length : 0,
    };
  }, [rows, cast, chosen, frames]);

  const zooms: [string, number][] = [
    [`In game ${cast.scale}×`, cast.scale],
    ["0.5×", 0.5],
    ["Actual 1×", 1],
    ["2×", 2],
  ];
  const chip = (on: boolean) =>
    `text-[11px] whitespace-nowrap px-2 py-0.5 rounded-md border transition ${
      on
        ? "bg-indigo-600 border-indigo-500 text-white"
        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
    }`;

  return (
    <>
      {/* ---- which body ---------------------------------------------------- */}
      <aside className="w-72 shrink-0 h-full flex flex-col border-r border-slate-800/70 bg-slate-950">
        <div className="shrink-0 p-4 border-b border-slate-800/60">
          <h2 className="text-[11px] font-medium uppercase tracking-widest text-slate-500 mb-3">
            Bodies
          </h2>
          <div className="grid grid-cols-4 gap-1">
            {cells.map((c) => {
              const on = c.atlas !== null && c.atlas === chosen;
              const portrait = !!cast.atlases.find((a) => a.key === c.atlas)?.portrait;
              return (
                <button
                  key={`${c.gender}/${c.bracket}`}
                  onClick={() => c.atlas && setBody(c.atlas)}
                  title={c.atlas ? `${c.atlas}${portrait ? "" : " (no portrait)"}` : "No asset for this bracket in the manifest"}
                  className={`relative text-[11px] py-1 rounded-md border transition ${
                    c.atlas === null
                      ? "border-rose-800/70 bg-rose-950/30 text-rose-300 cursor-not-allowed"
                      : on
                        ? "bg-indigo-600 border-indigo-500 text-white"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                  }`}
                >
                  {GENDER_LABEL[c.gender]} {BRACKET_LABEL[c.bracket] ?? c.bracket}
                  {c.atlas && !portrait && (
                    <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-amber-400" />
                  )}
                </button>
              );
            })}
          </div>
          {/* True once, then noise — it explains the table rather than reporting on it. */}
          <details className="mt-3 text-[11px] text-slate-600">
            <summary className="cursor-pointer hover:text-slate-400 transition">How to read these eight cells</summary>
            <p className="mt-1.5 leading-relaxed">
              Two cells pointing at the same atlas is valid: the asset set has no separate character for those two brackets. A red cell means the manifest has no character asset for that bracket at all. A yellow dot in the corner means that body has no portrait.
            </p>
          </details>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto p-4">
          <h2 className="text-[11px] font-medium uppercase tracking-widest text-slate-500 mb-3">
            This atlas
          </h2>
          <div className="font-mono text-[11px] text-slate-400 break-all mb-3">{chosen ?? "—"}</div>
          {frames && (
            <dl className="m-0 space-y-2 text-[12px]">
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-slate-500">Atlas frames</dt>
                <dd className="tabular-nums text-slate-200">{Object.keys(frames).length}</dd>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-slate-500">Table cells</dt>
                <dd className="tabular-nums text-slate-200">
                  {POSE_NAMES.length * DIRECTIONS.length}
                </dd>
              </div>
              {tally.unmapped > 0 && (
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-rose-400">Unmapped</dt>
                  <dd className="tabular-nums text-rose-400">{tally.unmapped}</dd>
                </div>
              )}
              {tally.missing > 0 && (
                <div className="flex items-baseline justify-between gap-2">
                  <dt className="text-amber-400">Frame not in atlas</dt>
                  <dd className="tabular-nums text-amber-400">{tally.missing}</dd>
                </div>
              )}
              <div className="flex items-baseline justify-between gap-2">
                <dt className="text-slate-500" title="Drawn but not referenced by the manifest">
                  Unreferenced
                </dt>
                <dd className="tabular-nums text-slate-200">{tally.unused}</dd>
              </div>
            </dl>
          )}

          {art && sheet && (
            <PortraitCheck
              portrait={art.portrait}
              idle={poseRow(cast, art.key, "idle", sheet.frames).SE}
              sheet={sheet}
              image={art.image}
            />
          )}
        </div>

        <div className="shrink-0 border-t border-slate-800/60 p-4 text-[11px] text-slate-600">
          <p>Check against CHARACTER_ASSET_SPEC.md §8:</p>
          {/* A list, not `<br>`: giving `<br>` children is a render-time throw no type check
              catches. Preflight is off (tailwind.config.cjs), so every list states its own
              padding or keeps the browser's 40px indent. */}
          <ul className="mt-1.5 space-y-1 leading-relaxed list-disc pl-4 marker:text-slate-700">
            <li>Turn on "Stack" to see whether the midpoint between the feet drifts</li>
            <li>Turn on "Canvas" to see whether limbs in strike and fall frames get clipped</li>
            <li>Switch to "In game" to see whether figures cover the buildings they stand by</li>
          </ul>
        </div>
      </aside>

      {/* ---- the table ----------------------------------------------------- */}
      <main className="flex-1 min-w-0 h-full flex flex-col">
        <div className="shrink-0 flex items-center gap-1 gap-y-1.5 flex-wrap px-3 py-2 border-b border-slate-800/70 bg-slate-950/60">
          <button onClick={() => setPlaying((p) => !p)} className={chip(playing)}>
            {playing ? "⏸ Pause" : "▶ Play"}
          </button>
          <button
            onClick={() => {
              setPlaying(false);
              setTick((t) => t + 1);
            }}
            className={chip(false)}
          >
            ⏭ Step
          </button>
          <button onClick={() => setStack((s) => !s)} className={`${chip(stack)} ml-2`}>
            Stack
          </button>
          <button onClick={() => setOutline((o) => !o)} className={chip(outline)}>
            Canvas
          </button>
          <span className="ml-auto flex items-center gap-1">
            {zooms.map(([label, value]) => (
              <button key={label} onClick={() => setZoom(value)} className={chip(zoom === value)}>
                {label}
              </button>
            ))}
          </span>
        </div>

        {/* No padding at the TOP: it belongs to the scrollable content, so rows would
            slide through it above the stuck header. The header carries it instead. */}
        <div className="flex-1 min-h-0 overflow-auto px-4 pb-4">
          {sheets === null && <div className="pt-4 text-slate-500 text-sm">Loading atlas…</div>}
          {sheets !== null && !sheet && (
            <div className="pt-4 text-rose-300/80 text-sm">
              Could not load this body's atlas ({art?.atlas ?? "none"}). Only the atlas knows which region each frame name maps to.
            </div>
          )}
          {sheet && art && (
            <table className="border-collapse">
              {/* Sticky in both directions. The whole table is a grid you read one cell
                  against another, and by the eighth row a heading that scrolled away
                  leaves you guessing whether the third column is SW or NW. */}
              <thead>
                <tr>
                  <th className="sticky top-0 left-0 z-20 bg-slate-950 w-24 pt-4" />
                  {DIRECTIONS.map((dir) => (
                    <th
                      key={dir}
                      className="sticky top-0 z-10 bg-slate-950 text-[11px] font-medium tracking-widest text-slate-400 px-3 pt-4 pb-2 border-b border-slate-800"
                    >
                      {dir}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(({ pose, row }) => (
                  <tr key={pose} className="border-b border-slate-800/50 hover:bg-slate-900/40 transition">
                    <th className="sticky left-0 z-10 bg-slate-950 align-top text-left pr-4 py-3 w-24">
                      <div className="text-[13px] text-slate-200 font-medium">{pose}</div>
                      {Object.values(row).every((c) => c.kind === "unmapped") && (
                        <div className="text-[10px] text-rose-400 mt-0.5">Not in manifest</div>
                      )}
                    </th>
                    {DIRECTIONS.map((dir) => (
                      <td key={dir} className="align-bottom px-3 py-3">
                        <FrameCell
                          cell={row[dir]}
                          sheet={sheet}
                          image={art.image}
                          zoom={zoom}
                          tick={tick}
                          stack={stack}
                          outline={outline}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </>
  );
}

/** One (pose, heading) box: the frame on screen now, and what it is called. */
function FrameCell({
  cell,
  sheet,
  image,
  zoom,
  tick,
  stack,
  outline,
}: {
  cell: Cell;
  sheet: Sheet;
  image: string;
  zoom: number;
  tick: number;
  stack: boolean;
  outline: boolean;
}) {
  if (cell.kind === "unmapped") {
    return (
      <div className="text-[10px] text-rose-400 border border-rose-900/60 bg-rose-950/20 rounded px-2 py-1">
        Unmapped
      </div>
    );
  }
  if (cell.kind === "missing") {
    return (
      <div className="text-[10px] font-mono text-amber-300 border border-amber-900/60 bg-amber-950/20 rounded px-2 py-1 max-w-[12rem] break-all">
        Not in atlas: {cell.names.join(" ")}
      </div>
    );
  }

  const at = tick % cell.frames.length;
  const shown = cell.frames[at];
  const styleOf = (rect: FrameRect) => frameStyle(rect, sheet, zoom, cell.flip, image);

  // At small zooms the frame name is wider than the picture and sets the column width, so a
  // name changing length between frames (talkNEClosed → talkNEOpen) would reflow the table
  // twice a cycle, which reads as the art flickering. Don't drop the widest-name reservation
  // below.
  const widest = cell.frames.reduce((a, f) => (f.name.length > a.length ? f.name : a), "");
  const label = (name: string) => (
    <span className="flex items-center gap-1 whitespace-nowrap">
      <span className="text-amber-300/80">{name}</span>
      {cell.frames.length > 1 && <span>×{cell.frames.length}</span>}
      {cell.flip && <span title="This asset set draws one image for this pose; the engine mirrors it for west-facing headings">mirrored</span>}
    </span>
  );

  return (
    <div>
      <div
        className={`relative ${outline ? "outline outline-1 outline-slate-800" : ""}`}
        style={styleOf(shown.rect).box}
      >
        {/* Every frame of the pose at once, so a foot that moves between them shows. */}
        {stack &&
          cell.frames.map((f) => (
            <div
              key={f.name}
              className="absolute inset-0 opacity-40"
              style={styleOf(f.rect).clip}
            />
          ))}
        {!stack && <div style={styleOf(shown.rect).clip} />}
      </div>
      {/* Both in one grid cell: the invisible one sizes the column, the live one shows. */}
      <div className="mt-0.5 grid text-[10px] font-mono text-slate-500">
        <div className="col-start-1 row-start-1 invisible" aria-hidden>
          {label(widest)}
        </div>
        <div className="col-start-1 row-start-1">{label(shown.name)}</div>
      </div>
    </div>
  );
}

const PORTRAIT_H = 220;

/**
 * The portrait beside the SE idle frame: the same person, clothes and stance, for checking by eye
 * what template_check cannot measure (the face, the style).
 */
function PortraitCheck({
  portrait,
  idle,
  sheet,
  image,
}: {
  portrait?: string;
  idle: Cell;
  sheet: Sheet;
  image: string;
}) {
  const frame = idle.kind === "frames" ? idle.frames[0] : null;
  // The frame keeps its blank margin, so its body draws shorter than the cropped portrait's.
  const style = frame && frameStyle(frame.rect, sheet, PORTRAIT_H / frame.rect.height, idle.kind === "frames" && idle.flip, image);
  return (
    <section className="mt-5">
      <h2 className="text-[11px] font-medium uppercase tracking-widest text-slate-500 mb-3">Portrait</h2>
      {!portrait ? (
        <p className="text-[11px] leading-relaxed text-amber-300/80">
          This body has no portrait, so the cognition checkup enlarges its idle frame and it looks blurry.
        </p>
      ) : (
        <div className="flex items-end gap-3">
          <figure className="m-0">
            <a href={portrait} target="_blank" rel="noreferrer" title="Open the original in a new tab">
              <img src={portrait} alt="Portrait" style={{ height: PORTRAIT_H }} className="block w-auto" />
            </a>
            <figcaption className="mt-1 text-[10px] text-slate-500">Portrait</figcaption>
          </figure>
          {frame && style && (
            <figure className="m-0">
              <div style={style.box}>
                <div style={style.clip} />
              </div>
              <figcaption className="mt-1 text-[10px] text-slate-500 font-mono">{frame.name}</figcaption>
            </figure>
          )}
        </div>
      )}
    </section>
  );
}
