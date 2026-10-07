import { describe, expect, it } from "vitest";

import { routeFraction } from "./transitProgress";

describe("routeFraction", () => {
  // A short first leg reached in 1 step, a long second leg taking 2 more.
  const legs = [100, 300];
  const arrivals = [0, 1, 3];

  it("stands exactly on a waypoint on the step the engine puts him there", () => {
    expect(routeFraction(legs, arrivals, 1)).toBe(0.25);
  });

  it("spreads a multi-step leg over that leg's own length", () => {
    expect(routeFraction(legs, arrivals, 2)).toBe(0.625);
  });

  it("is not elapsed/total when legs differ", () => {
    expect(routeFraction(legs, arrivals, 1)).not.toBeCloseTo(1 / 3);
  });

  it("stands on the furthest of several waypoints reached on the same step", () => {
    // Two short legs crossed within step 1, then a leg finished on step 2.
    expect(routeFraction([100, 100, 200], [0, 1, 1, 2], 1)).toBe(0.5);
  });

  it("ends at the destination", () => {
    expect(routeFraction(legs, arrivals, 3)).toBe(1);
    expect(routeFraction(legs, arrivals, 0)).toBe(0);
  });
});
