import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { Scoped } from "../../lib/step";
import type { MessageSummary } from "../../types";
import type { StepCast } from "./cast";
import MessageRow from "./MessageRow";

function letter(anonymous: boolean, sender_id: string | null): Scoped<MessageSummary> {
  return {
    kind: "message", seq: 1, message_id: "letter", sender_id, anonymous,
    sender_name: anonymous ? "匿名" : "甲", receiver_ids: ["reader"], scope: "direct",
    content: "请留步", spoken: "请留步", audience: ["reader"], location_id: "hall",
  };
}

describe("message signatures", () => {
  it.each([
    { anonymous: false, sender_id: "sender" },
    { anonymous: true, sender_id: "sender" },
    { anonymous: true, sender_id: null },
  ])("uses the public signature ($anonymous, $sender_id)", ({ anonymous, sender_id }) => {
    const inkOf = vi.fn((id?: string) => id === "sender" ? "#123456" : "");
    const cast: StepCast = {
      nameOf: () => "乙", inkOf, isMain: () => false, conditionOf: () => "", placeOf: () => "大厅",
    };
    const html = renderToStaticMarkup(<MessageRow m={letter(anonymous, sender_id)} cast={cast} />);
    expect(html).toContain(anonymous ? "匿名" : "甲");
    expect(html).toContain("请留步");
    if (anonymous) {
      expect(inkOf).not.toHaveBeenCalledWith("sender");
      expect(html).not.toContain("#123456");
    } else {
      expect(inkOf).toHaveBeenCalledWith("sender");
      expect(html).toContain("#123456");
    }
  });
});
