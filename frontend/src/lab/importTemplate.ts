/**
 * Uploading a map template, and what the backend can say about it.
 *
 * Not routed through `api/client`'s `req`, which assumes a JSON body and flattens failures into
 * `new Error(detail)`: the body here is an archive, and a contract failure is a list of problems to
 * render, not an exception to stringify.
 */

import { API_BASE } from "../api/client";

export type ImportOutcome =
  | { ok: true; files: number; connections: number }
  | { ok: false; kind: "conflict"; message: string }
  | { ok: false; kind: "rejected"; problems: string[] }
  | { ok: false; kind: "error"; message: string };

/**
 * The backend's name rule, mirrored so a doomed upload is refused before 20 MB crosses the wire.
 * The name becomes a path under `worlds/templates/`, where a leading underscore marks a reference
 * sample, not a buildable map.
 */
export const TEMPLATE_NAME_RE = /^[a-z][a-z0-9_]{0,63}$/;

/** A starting point for the name field, from whatever the archive was called. */
export function suggestTemplateName(filename: string): string {
  return filename
    .replace(/\.zip$/i, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^[^a-z]+/, "")
    .replace(/_+$/, "")
    .slice(0, 64);
}

/** Read `detail` out of an error body, in either shape it can arrive in. */
function problemsIn(detail: unknown): string[] | null {
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const listed = (detail as { problems?: unknown }).problems;
    if (Array.isArray(listed)) return listed.map(String);
  }
  // A bare list is FastAPI's own validation error, not a contract verdict.
  return null;
}

function messageIn(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail)) {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return fallback;
}

export async function importTemplate(name: string, file: Blob): Promise<ImportOutcome> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/templates/${name}/archive`, {
      method: "PUT",
      body: file,
    });
  } catch (err) {
    return { ok: false, kind: "error", message: (err as Error).message };
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    /* an empty or non-JSON body; the status still says what happened */
  }
  const detail = (body as { detail?: unknown } | null)?.detail;

  if (response.ok) {
    const done = body as { files?: number; connections?: number } | null;
    return { ok: true, files: Number(done?.files ?? 0), connections: Number(done?.connections ?? 0) };
  }
  // Already taken — by the directory name or by the world_name. Either way the fix is
  // a different name or a deletion, never the same upload again.
  if (response.status === 409) {
    return { ok: false, kind: "conflict", message: messageIn(detail, "Name already taken") };
  }

  const problems = problemsIn(detail);
  if (problems) return { ok: false, kind: "rejected", problems };
  return { ok: false, kind: "error", message: messageIn(detail, response.statusText) };
}

/**
 * Remove an installed map. Worlds already built on it are untouched (each froze its own copy at
 * build); only building new worlds on it goes away.
 */
export async function deleteTemplate(name: string): Promise<{ ok: true } | { ok: false; message: string }> {
  try {
    const response = await fetch(`${API_BASE}/api/templates/${name}`, { method: "DELETE" });
    if (response.ok) return { ok: true };
    const body = await response.json().catch(() => null);
    return { ok: false, message: messageIn((body as { detail?: unknown } | null)?.detail, response.statusText) };
  } catch (err) {
    return { ok: false, message: (err as Error).message };
  }
}
