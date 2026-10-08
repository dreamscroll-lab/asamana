/**
 * SplashScreen — the launch screen: the brand standing inside its own world.
 *
 * Reuses EmergingWorld unchanged (only larger) so the brand image can't drift into two
 * versions; nothing paints over the sphere. Depth comes from three cues on the word's side:
 *
 * 1. The word casts a brand-colored shadow.
 * 2. A thin haze sits in front at the limb — the atmosphere.
 * 3. Pointer parallax: the word travels further than the world, and the other way.
 *
 * The gradient is the app's wordmark ramp; violet/indigo are remapped onto the brand
 * purple in tailwind.config.cjs.
 */
import { useEffect, useRef, useState } from "react";

import { THEME } from "../lib/theme";
import EmergingWorld, { HALO } from "./EmergingWorld";

const MAX_SPHERE = 480; // past this the word runs wider than a comfortable measure
const MIN_SPHERE = 190;
const WORD_SHIFT = 14; // px of parallax travel at the edge of the pane…
const WORLD_SHIFT = -6; // …against the world drifting the other way, and less

export default function SplashScreen() {
  const stage = useRef<HTMLDivElement>(null);
  const still = useRef(false);
  const [sphere, setSphere] = useState(0); // the world's own diameter, in CSS px
  const [tilt, setTilt] = useState({ x: 0, y: 0 }); // pointer, -1..1 from center

  // Fit to the pane, not the window: the sidebar beside it collapses.
  useEffect(() => {
    still.current = !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const el = stage.current;
    if (!el) return;
    const fit = () => {
      const r = el.getBoundingClientRect();
      // 0.645 of a side would fill the pane exactly (the box is HALO times wider); 0.5 and
      // 0.56 leave room for the halo to fade out and the ~one-diameter word to clear.
      setSphere(
        Math.round(Math.max(MIN_SPHERE, Math.min(r.width * 0.5, r.height * 0.56, MAX_SPHERE))),
      );
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const move = (e: React.PointerEvent<HTMLDivElement>) => {
    if (still.current) return;
    const r = e.currentTarget.getBoundingClientRect();
    setTilt({
      x: ((e.clientX - r.left) / r.width) * 2 - 1,
      y: ((e.clientY - r.top) / r.height) * 2 - 1,
    });
  };

  const box = Math.round(sphere * HALO); // what the canvas actually occupies
  const shift = (k: number) =>
    `translate3d(${(tilt.x * k).toFixed(1)}px, ${(tilt.y * k).toFixed(1)}px, 0)`;

  return (
    <div
      ref={stage}
      onPointerMove={move}
      onPointerLeave={() => setTilt({ x: 0, y: 0 })}
      className="relative min-h-screen overflow-hidden grid place-items-center bg-slate-950 select-none"
    >
      {/* Behind the canvas: lifts the ground the halo fades into, not the world. */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ background: `radial-gradient(56% 52% at 50% 47%, ${THEME.accent}24, ${THEME.bg} 74%)` }}
      />

      {sphere > 0 && (
        <div className="relative" style={{ width: box, height: box }}>
          {/* 1 — the world, untouched: furthest back, and it moves the least */}
          <div
            className="absolute inset-0 transition-transform duration-700 ease-out animate-[fadeIn_1.6s_ease-out]"
            style={{ transform: shift(WORLD_SHIFT) }}
          >
            <EmergingWorld size={box} />
          </div>

          {/* 2 — the mark. drop-shadow rides the wrapper: on the gradient element
                  itself, background-clip:text and filter fight over the same box. */}
          <div
            className="absolute inset-0 grid place-items-center transition-transform duration-700 ease-out"
            style={{ transform: shift(WORD_SHIFT) }}
          >
            <div
              className="text-center"
              style={{
                filter: `drop-shadow(0 ${Math.round(sphere * 0.03)}px ${Math.round(sphere * 0.09)}px ${THEME.accent}a6)`,
              }}
            >
              <div
                className="whitespace-nowrap font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-indigo-200 to-cyan-300 animate-[riseIn_1.1s_ease-out_0.3s_both]"
                style={{ fontSize: sphere * 0.22, lineHeight: 1, letterSpacing: "-0.03em" }}
              >
                Asamana
              </div>
              {/* Upper-case and wide so it doesn't read as a second wordmark. The negative
                  margin eats the trailing tracking that would push the line off-center. */}
              <div
                className="whitespace-nowrap uppercase font-semibold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-indigo-200 to-cyan-300 animate-[riseIn_1.1s_ease-out_0.65s_both]"
                style={{
                  fontSize: sphere * 0.055,
                  marginTop: sphere * 0.045,
                  letterSpacing: sphere * 0.024,
                  marginRight: -sphere * 0.024,
                }}
              >
                AI as a human
              </div>
            </div>
          </div>

          {/* 3 — the atmosphere, in front of the letters. `closest-side` is load-bearing:
                  the default farthest-corner lights the box's corners into a bright square.
                  The 0.5/0.8 stops then straddle the limb; `screen` only adds light. */}
          <div
            className="absolute inset-0 pointer-events-none transition-transform duration-700 ease-out"
            style={{
              transform: shift(WORLD_SHIFT), // the air belongs to the world, so it travels with it
              mixBlendMode: "screen",
              background: `radial-gradient(circle closest-side, transparent 50%, ${THEME.accent}2b 80%, transparent 100%)`,
            }}
          />
        </div>
      )}
    </div>
  );
}
