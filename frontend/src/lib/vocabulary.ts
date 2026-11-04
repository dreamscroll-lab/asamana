import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { Vocabulary } from "../types";

// The backend's closed vocabularies are the same for every world, so asked once per page load
// and shared by every surface that offers them. A failed ask is asked again by the next user.
let asked: Promise<Vocabulary> | null = null;

/** Every emotion and need a client may name, with its label. null = not loaded (yet). */
export function useVocabulary(): Vocabulary | null {
  const [vocabulary, setVocabulary] = useState<Vocabulary | null>(null);
  useEffect(() => {
    let live = true;
    asked ??= api.vocabulary().catch((cause: unknown) => {
      asked = null;
      throw cause;
    });
    asked.then((v) => { if (live) setVocabulary(v); }, () => {});
    return () => { live = false; };
  }, []);
  return vocabulary;
}
