import type { StepEvent } from "../types";

/**
 * The world's clock and, separately, the engine's counter. Never run them together in one
 * sentence ("…暮色渐浓 · 第 15 步"): that gives the machine's tally the standing of an hour.
 *
 *   · `world_time.label` is the world's own (narrative-layer) name for this moment; its form
 *     is the theme's to choose, so it is shown whole — splitting it on commas would nail this
 *     to one theme's calendar (CLAUDE.md Rule 7).
 *   · `step` is a code-layer ordinal for scheduling and replay; it gets a small, cold badge.
 *
 * One shared component so every surface keeps the amber-plate / cold-badge distinction;
 * `size` is the only thing that legitimately differs.
 */
const SIZES = {
  lg: {
    row: "flex items-center gap-2.5",
    plate: "flex items-center gap-2 rounded-lg border border-amber-900/40 bg-amber-950/20 px-3 py-1.5",
    glyph: "text-amber-500/80 text-sm leading-none",
    label: "text-[13px] font-medium tracking-wide text-amber-100/90",
  },
  sm: {
    row: "flex items-center gap-2",
    plate: "flex items-center gap-2 rounded-lg border border-amber-900/40 bg-amber-950/20 px-2.5 py-1",
    glyph: "text-amber-500/80 text-xs leading-none",
    label: "text-[12px] font-medium tracking-wide text-amber-100/90",
  },
} as const;

export default function WorldClock({
  step,
  size = "lg",
  className = "",
}: {
  step: StepEvent;
  size?: keyof typeof SIZES;
  className?: string;
}) {
  const s = SIZES[size];
  return (
    <div className={className ? `${s.row} ${className}` : s.row}>
      <div className={s.plate}>
        <span className={s.glyph}>◷</span>
        <span className={s.label}>{step.world_time.label}</span>
      </div>
      <span className="rounded-md border border-slate-800 bg-slate-950/60 px-1.5 py-1 text-[10px] tabular-nums text-slate-500">
        第 {step.step} 步
      </span>
    </div>
  );
}
