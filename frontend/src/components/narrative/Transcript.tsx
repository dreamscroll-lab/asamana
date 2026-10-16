import type { DialogueTurn } from "../../types";

import { identityColor, toHex } from "../../lib/color";

/** A conversation's lines, each speaker named in their identity color (`inkOf`). */
export default function Transcript({
  turns,
  inkOf,
}: {
  turns: DialogueTurn[];
  inkOf: (id: string) => string | undefined;
}) {
  return (
    <div className="mt-2 space-y-1.5 rounded-lg bg-slate-950/50 border border-slate-800/60 px-3 py-2.5">
      {turns.map((t, i) => {
        const idColor = toHex(identityColor(inkOf(t.speaker_id), t.speaker_id));
        return (
          <div key={i} className="text-sm leading-relaxed">
            {t.speaker && (
              <span
                className="font-semibold"
                style={{ color: idColor }}
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
