import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { StepView } from "../types";
import WorldClock from "./WorldClock";

const step = { step: 15, world_time: { label: "六月初三 辰时" } } as unknown as StepView;

describe("the world clock", () => {
  it("shows the step duration beside the step, never in the world time", () => {
    const markup = renderToStaticMarkup(<WorldClock step={step} stepSeconds={10800} />);
    expect(markup).toContain("第 15 步");
    expect(markup).toContain("⏳ 每步时长 约 3 小时");
    expect(markup).not.toContain("六月初三 辰时 · ");
  });

  it("shows the step alone when the duration is unknown", () => {
    const markup = renderToStaticMarkup(<WorldClock step={step} />);
    expect(markup).toContain("第 15 步");
    expect(markup).not.toContain("每步时长");
  });
});
