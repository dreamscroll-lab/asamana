import type { Intervention } from "../../types";

export type SubmissionState =
  | { kind: "working" }
  | { kind: "accepted"; preview: string }
  | { kind: "refused"; reason: string }
  | { kind: "failed"; reason: string };

export interface Submission {
  id: number;
  text: string;
  occurrence: number | null; // null: no reliable pre-submit history baseline
  after: number | null; // number of recorded directives before this submission
  state: SubmissionState;
}

export type ThreadItem =
  | { kind: "submission"; key: string; submission: Submission }
  | { kind: "landed"; key: string; intervention: Intervention };

/** Identical sentences can be injected repeatedly before their history catches up. */
export function nextOccurrence(text: string, history: Intervention[] | null, submissions: Submission[]): number | null {
  if (history === null) return null;
  let occurrence = history.filter((item) => item.directive_text === text).length;
  for (const submission of submissions) {
    // A lost response leaves it unknown whether the request consumed a history occurrence.
    if (submission.text === text && submission.state.kind === "failed") return null;
    if (submission.text === text && (submission.state.kind === "working" || submission.state.kind === "accepted")) {
      if (submission.occurrence === null) return null;
      occurrence = Math.max(occurrence, submission.occurrence);
    }
  }
  return occurrence + 1;
}

export function buildThread(history: Intervention[], submissions: Submission[]): ThreadItem[] {
  const occurrences = new Map<string, number[]>();
  history.forEach((item, index) => {
    const indexes = occurrences.get(item.directive_text) ?? [];
    indexes.push(index);
    occurrences.set(item.directive_text, indexes);
  });
  const matches = new Map<number, number>();
  const recorded: ThreadItem[] = history.map((intervention, index) => ({
    kind: "landed", key: `landed-${index}`, intervention,
  }));
  for (const submission of submissions) {
    if (submission.occurrence === null || (submission.state.kind !== "accepted" && submission.state.kind !== "working")) continue;
    const index = occurrences.get(submission.text)?.[submission.occurrence - 1];
    if (index === undefined) continue;
    matches.set(submission.id, index);
    const key = `submission-${submission.id}`;
    recorded[index] = submission.state.kind === "working"
      ? { kind: "submission", key, submission }
      : { kind: "landed", key, intervention: history[index] };
  }

  const items: ThreadItem[] = [];
  let cursor = 0;
  const appendHistory = (end: number): void => {
    while (cursor < Math.min(end, recorded.length)) items.push(recorded[cursor++]);
  };
  const chronological = [...submissions].reverse();
  const laterAnchors: number[] = [];
  let later = history.length;
  for (let i = chronological.length - 1; i >= 0; i--) {
    laterAnchors[i] = later;
    const submission = chronological[i];
    later = Math.min(later, submission.after ?? history.length, matches.get(submission.id) ?? history.length);
  }
  // An unknown baseline must not flush a later submission's server record ahead of this message.
  for (const [index, submission] of chronological.entries()) {
    appendHistory(submission.after ?? laterAnchors[index]);
    const match = matches.get(submission.id);
    if (match === undefined) {
      items.push({ kind: "submission", key: `submission-${submission.id}`, submission });
    } else {
      appendHistory(match + 1);
    }
  }
  appendHistory(recorded.length);
  return items;
}
