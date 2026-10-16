import type { DialogueTurn } from "../../types";

// Fallback palette for a speaker the world carries no identity color for — per
// CONVERSATION, so the two sides of an exchange at least stay apart from each other. Never
// reached when the speaker is a known character: they wear their own color.
const SPEAKER_COLORS = [
  "text-indigo-300",
  "text-teal-300",
  "text-rose-300",
  "text-amber-300",
  "text-sky-300",
  "text-fuchsia-300",
];

/** A conversation's lines, each speaker named in their identity color (`inkOf`). */
export default function Transcript({
  turns,
  inkOf,
}: {
  turns: DialogueTurn[];
  inkOf: (id: string) => string | undefined;
}) {
  const speakers = [...new Set(turns.map((t) => t.speaker).filter(Boolean))];
  const colorOf = (s: string) => SPEAKER_COLORS[Math.max(0, speakers.indexOf(s)) % SPEAKER_COLORS.length];
  return (
    <div className="mt-2 space-y-1.5 rounded-lg bg-slate-950/50 border border-slate-800/60 px-3 py-2.5">
      {turns.map((t, i) => {
        const idColor = inkOf(t.speaker_id);
        return (
          <div key={i} className="text-sm leading-relaxed">
            {t.speaker && (
              <span
                className={`font-semibold ${idColor ? "" : colorOf(t.speaker)}`}
                style={idColor ? { color: idColor } : undefined}
              >
                {t.speaker}
              </span>
            )}
            {t.speaker && <span className="text-slate-600">：</span>}
            <span className="text-slate-200">{t.line}</span>
          </div>
        );
      })}
    </div>
  );
}
