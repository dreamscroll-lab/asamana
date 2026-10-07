/**
 * Which scenes you have already checked, and how they came out, so a full sweep can be paused.
 *
 * Kept per template: the art is the map's, so a pose correct on one template says nothing about
 * another. Browser-local and disposable: one developer's notes; without storage they just don't
 * survive a reload.
 */

export type Mark = "pass" | "fail";

const KEY = "asamana.lab.marks";

export type Marks = Record<string, Mark>;

export const markKey = (template: string, sceneId: string): string => `${template}/${sceneId}`;

export function loadMarks(): Marks {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Marks) : {};
  } catch {
    return {};
  }
}

export function saveMarks(marks: Marks): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(marks));
  } catch {
    /* no storage — the marks live for this page only */
  }
}
