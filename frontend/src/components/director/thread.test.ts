import { describe, expect, it } from "vitest";

import type { Intervention } from "../../types";
import { buildThread, nextOccurrence, type Submission } from "./thread";

const landed = (text: string, step = 1): Intervention => ({
  step, time_label: "暮色", directive_text: text, narrative: "A recorded result",
});
const submitted = (id: number, state: Submission["state"] = { kind: "working" }, occurrence: number | null = 1, after: number | null = 0): Submission => ({
  id, text: "Bring rain", occurrence, after, state,
});

describe("director thread", () => {
  it("immediately shows the submitted sentence before any response or history arrives", () => {
    const pending = submitted(1);
    expect(buildThread([], [pending])).toEqual([{ kind: "submission", key: "submission-1", submission: pending }]);
  });

  it("keeps the accepted result visible until its history record arrives", () => {
    const accepted = submitted(1, { kind: "accepted", preview: "Rain begins" });
    expect(buildThread([], [accepted])[0]).toMatchObject({ submission: { state: { preview: "Rain begins" } } });
    const history = [landed(accepted.text)];
    const items = buildThread(history, [accepted]);
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ kind: "landed", intervention: history[0] });
  });

  it("does not confuse a new submission with an identical older instruction", () => {
    const history = [landed("Bring rain")];
    expect(nextOccurrence("Bring rain", history, [])).toBe(2);
    const accepted = submitted(1, { kind: "accepted", preview: "More rain" }, 2, 1);
    expect(buildThread(history, [accepted]).map((item) => item.kind)).toEqual(["landed", "submission"]);
    expect(buildThread([...history, landed("Bring rain", 2)], [accepted]).map((item) => item.kind)).toEqual(["landed", "landed"]);
  });

  it("keeps a sentence visible when its pre-submit history could not be read", () => {
    expect(nextOccurrence("Bring rain", null, [])).toBeNull();
    const accepted = submitted(1, { kind: "accepted", preview: "New rain" }, null, null);
    const items = buildThread([landed("Bring rain")], [accepted]);
    expect(items.at(-1)).toMatchObject({ submission: accepted });
    expect(nextOccurrence("Bring rain", [landed("Bring rain")], [accepted])).toBeNull();
  });

  it("assigns the pending sentence after older identical history once its baseline loads", () => {
    const pending = submitted(1, { kind: "working" }, null, null);
    const history = [landed(pending.text)];
    expect(buildThread(history, [pending]).at(-1)).toMatchObject({ submission: pending });
    const accepted = { ...pending, after: history.length, occurrence: nextOccurrence(pending.text, history, []), state: { kind: "accepted" as const, preview: "New rain" } };
    expect(accepted.occurrence).toBe(2);
    expect(buildThread(history, [accepted]).at(-1)).toMatchObject({ submission: accepted });
  });

  it("reconciles repeated accepted submissions one at a time as history catches up", () => {
    const first = submitted(1, { kind: "accepted", preview: "Rain" });
    expect(nextOccurrence("Bring rain", [], [first])).toBe(2);
    const second = submitted(2, { kind: "accepted", preview: "More rain" }, 2);
    const items = buildThread([landed("Bring rain")], [second, first]);
    expect(items).toHaveLength(2);
    expect(items[1]).toMatchObject({ submission: { id: 2 } });
    expect(items[0].kind).toBe("landed");
  });

  it("retains refusals and network uncertainty beside their original sentences", () => {
    const refused = submitted(1, { kind: "refused", reason: "A place is required" }, 2, 1);
    const failed = submitted(2, { kind: "failed", reason: "Network error" }, 2, 1);
    const items = buildThread([landed("Bring rain")], [failed, refused]);
    expect(items).toHaveLength(3);
    expect(items[0].kind).toBe("landed");
    expect(items[1]).toMatchObject({ submission: refused });
    expect(items[2]).toMatchObject({ submission: failed });
    expect(nextOccurrence("Bring rain", [], [refused])).toBe(1);
  });

  it("does not hide a retry against history from an earlier transport-uncertain request", () => {
    const failed = submitted(1, { kind: "failed", reason: "Response lost" });
    const occurrence = nextOccurrence("Bring rain", [], [failed]);
    expect(occurrence).toBeNull();
    const retry = submitted(2, { kind: "accepted", preview: "Rain" }, occurrence);
    const items = buildThread([landed("Bring rain")], [retry, failed]);
    expect(items[1]).toMatchObject({ submission: { id: 2 } });
    expect(items).toHaveLength(3);
  });

  it("does not replace a working status with history before its own request has settled", () => {
    const pending = submitted(1);
    const items = buildThread([landed(pending.text)], [pending]);
    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ submission: { state: { kind: "working" } } });
  });

  it("appends recent submissions below older history without mutating inputs", () => {
    const history = [landed("Earlier", 1), landed("Later", 2)];
    const recent = submitted(2, { kind: "working" }, 1, history.length);
    const items = buildThread(history, [recent]);
    expect(items.map((item) => item.kind === "landed" ? item.intervention.directive_text : item.submission.text)).toEqual([
      "Earlier", "Later", "Bring rain",
    ]);
    expect(history.map((item) => item.step)).toEqual([1, 2]);
  });

  it("preserves chronological positions and keys as accepted messages replace local ones", () => {
    const first = { ...submitted(1, { kind: "accepted", preview: "Rain begins" }), text: "Rain" };
    const refused = { ...submitted(2, { kind: "refused", reason: "No place" }), text: "Invalid" };
    const last = { ...submitted(3, { kind: "accepted", preview: "Snow begins" }), text: "Snow" };
    const submissions = [last, refused, first];
    for (const history of [[], [landed("Rain")], [landed("Rain"), landed("Snow", 2)]]) {
      const items = buildThread(history, submissions);
      expect(items.map((item) => item.key)).toEqual(["submission-1", "submission-2", "submission-3"]);
      expect(items.map((item) => item.kind === "landed" ? item.intervention.directive_text : item.submission.text)).toEqual([
        "Rain", "Invalid", "Snow",
      ]);
    }
    expect(submissions.map((item) => item.id)).toEqual([3, 2, 1]);
  });

  it("keeps a refusal after the older server history known when it was sent", () => {
    const refused = submitted(1, { kind: "refused", reason: "No place" }, 1, 1);
    const history = [landed("Earlier", 1), landed("Later", 2)];
    const items = buildThread(history, [refused]);
    expect(items.map((item) => item.kind === "landed" ? item.intervention.directive_text : item.submission.text)).toEqual([
      "Earlier", "Bring rain", "Later",
    ]);
  });

  it("does not flush a later confirmed message ahead of an earlier unknown-baseline message", () => {
    const first = { ...submitted(1, { kind: "accepted", preview: "Rain" }, null, null), text: "Rain" };
    const last = { ...submitted(2, { kind: "accepted", preview: "Snow" }, 1, 1), text: "Snow" };
    const history = [landed("Earlier"), landed("Snow", 2)];
    const items = buildThread(history, [last, first]);
    expect(items.map((item) => item.kind === "landed" ? item.intervention.directive_text : item.submission.text)).toEqual([
      "Earlier", "Rain", "Snow",
    ]);
    expect(items.map((item) => item.key)).toEqual(["landed-0", "submission-1", "submission-2"]);
  });
});
