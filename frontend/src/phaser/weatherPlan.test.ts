/**
 * Contract for the weather decision layer.
 *
 * Each assertion covers a real trap (see the list atop weatherPlan.ts); the decisions are pure,
 * so these run without a browser, WebGL or a human looking.
 *
 * Aesthetics are left to the scene bench's weather group.
 */

import { describe, expect, it } from "vitest";

import {
  FALL_TILES,
  FIRE_RADIUS_TILES,
  MAX_AREA_SCALE,
  MAX_LIVE_PARTICLES,
  MAX_SITES,
  SITED_PHENOMENA,
  type WeatherArea,
  planWeather,
} from "./weatherPlan";

const TILE = 32;
const CTX = { tilePx: TILE, zoom: 1 };

/** A diamond footprint (as it looks after isometric projection), width/height in tiles. */
function area(wTiles: number, hTiles: number, onScreen = true): WeatherArea {
  const w = wTiles * TILE;
  const h = hTiles * TILE;
  return {
    footprint: [
      { x: w / 2, y: 0 }, { x: w, y: h / 2 }, { x: w / 2, y: h }, { x: 0, y: h / 2 },
    ],
    box: { x: 0, y: 0, width: w, height: h },
    onScreen,
  };
}

const SMALL = area(4, 2);      // one ward
const HUGE = area(60, 30);     // the whole map
const PHENOMENA = ["rain", "snow", "wind", "fire", "smoke", "quake", "dark"] as const;

describe("词汇表边界", () => {
  it("每一种已知现象都产出可画的东西", () => {
    // Silently emitting nothing is how the Polygon bug showed up, and it got past both the compiler and runtime.
    for (const p of PHENOMENA) {
      const plan = planWeather(p, "medium", SMALL, CTX);
      expect(plan, p).not.toBeNull();
      const drawsSomething =
        plan!.layers.length > 0 || (plan!.glows?.length ?? 0) > 0 || plan!.shake != null || plan!.veil != null;
      expect(drawsSomething, p).toBe(true);
    }
  });

  it("不认识的现象什么都不画", () => {
    // If the engine adds a phenomenon, an older renderer receives the new word unchanged. Drawing
    // weather nobody reported would be wrong. Returning null rather than an empty plan matters
    // too: the caller then lets the broadcast use its normal presentation, so the message is
    // still shown.
    expect(planWeather("aurora", "high", SMALL, CTX)).toBeNull();
    expect(planWeather("none", "high", SMALL, CTX)).toBeNull();
    expect(planWeather("", "high", SMALL, CTX)).toBeNull();
  });
});

describe("尺度的参照物", () => {
  it("火的大小与它落在哪个地点无关", () => {
    // Sizing by place width (e.g. min(ward width × 0.28, 120)) hits the cap in wide wards and turns
    // the glow into an additive blob hundreds of pixels across, washing out the flames and
    // embers. A fire is big because it burns hard, not because the ward is big.
    const small = planWeather("fire", "medium", SMALL, CTX)!;
    const huge = planWeather("fire", "medium", HUGE, CTX)!;
    expect(small.glows![0].radiusPx).toBeCloseTo(huge.glows![0].radiusPx);
    expect(small.glows![0].radiusPx).toBeCloseTo(FIRE_RADIUS_TILES * TILE);
  });

  it("火的大小随严重程度变", () => {
    const low = planWeather("fire", "low", SMALL, CTX)!.glows![0].radiusPx;
    const high = planWeather("fire", "high", SMALL, CTX)!.glows![0].radiusPx;
    expect(high).toBeGreaterThan(low);
  });

  it("下落行程与这片地方多大**完全无关**", () => {
    // `max(box.height, floor)` fails at both ends: a 200ms flicker in a ward, ~10k live
    // particles over the whole map (see FALL_TILES).
    const traveled = (a: WeatherArea) => {
      const streak = planWeather("rain", "medium", a, CTX)!.layers.find((l) => l.texture === "streak")!;
      return ((streak.lifespanMs as number) / 1000) * (streak.velocity.y as number);
    };
    expect(traveled(SMALL)).toBeCloseTo(FALL_TILES * TILE);
    expect(traveled(HUGE)).toBeCloseTo(FALL_TILES * TILE);
  });

  it("下落类从地点**上方**出生,不是从地点中间", () => {
    const rain = planWeather("rain", "medium", SMALL, CTX)!;
    const zone = rain.layers[0].zone;
    expect(zone.kind).toBe("ward");
    const lowest = Math.max(...(zone as { points: { y: number }[] }).points.map((p) => p.y));
    expect(lowest).toBeLessThan(SMALL.box.y);   // the whole zone is above the place's top edge
  });
});

/** The largest size in a layer, converted back to world pixels.
 *  Texture size is per axis: a streak is 3×28, and scaling both axes by one number would
 *  produce a huge width that doesn't exist. */
function maxWorldPx(layer: { size: Record<string, unknown>; texture: string }): number {
  const perAxis = layer.texture === "streak" ? { x: 3, y: 28, uniform: 28 } : { x: 32, y: 32, uniform: 32 };
  let biggest = 0;
  for (const [axis, v] of Object.entries(layer.size)) {
    const texPx = perAxis[axis as keyof typeof perAxis];
    const nums = typeof v === "number" ? [v] : Object.values((v ?? {}) as Record<string, number>);
    for (const n of nums) biggest = Math.max(biggest, n * texPx);
  }
  return biggest;
}

describe("粒子是**世界量**,不是屏幕贴花", () => {
  it("一滴雨不会比一个 tile 长太多", () => {
    // At whole-map view, 1/zoom compensation makes raindrops more than 3× bigger in the world,
    // longer than a roof. This map is a world the viewer looks into, not a screen filter, so
    // everything scales with the world, rain included. This must hold at every zoom level.
    for (const zoom of [0.2, 0.5, 1, 2, 4]) {
      const plan = planWeather("rain", "high", SMALL, { tilePx: TILE, zoom })!;
      for (const layer of plan.layers) {
        expect(maxWorldPx(layer), `zoom=${zoom}`).toBeLessThan(TILE * 1.5);
      }
    }
  });

  it("每一种现象的粒子都在合理的世界尺度内", () => {
    // Smoke puffs are big anyway. Wind's length expresses speed, not size: a gust two or three
    // tiles long reads as strong wind, not as a huge object. Everything else should be well under
    // one tile.
    const ceiling: Record<string, number> = { smoke: 2.2, fire: 2.2, wind: 3.2 };
    for (const p of PHENOMENA) {
      for (const zoom of [0.2, 1, 4]) {
        const plan = planWeather(p, "high", SMALL, { tilePx: TILE, zoom })!;
        for (const layer of plan.layers) {
          expect(maxWorldPx(layer), `${p}@${zoom}`).toBeLessThan(TILE * (ceiling[p] ?? 1.5));
        }
      }
    }
  });

  it("拉得再远,粒子也不会细到消失", () => {
    // Since particles are world-sized, they shrink toward zero when zoomed far out. A minimum screen size keeps the rain visible.
    const plan = planWeather("rain", "medium", SMALL, { tilePx: TILE, zoom: 0.05 })!;
    for (const layer of plan.layers) {
      expect(maxWorldPx(layer) * 0.05).toBeGreaterThanOrEqual(1);
    }
  });

  it("**位置完全不受缩放影响**", () => {
    // Positions must not depend on zoom: emitter.setScale() would push the rain off the map.
    const a = planWeather("rain", "medium", SMALL, { tilePx: TILE, zoom: 1 })!;
    const b = planWeather("rain", "medium", SMALL, { tilePx: TILE, zoom: 0.25 })!;
    const pts = (p: typeof a) => JSON.stringify((p.layers[0].zone as { points: unknown }).points);
    expect(pts(a)).toEqual(pts(b));

    // Same for the glow, which is part of the scene: position and radius are world quantities.
    // This uses the fire plan because rain has no glow, and on rain the check would pass
    // trivially as `undefined === undefined`.
    const f = (zoom: number) => planWeather("fire", "medium", SMALL, { tilePx: TILE, zoom })!.glows![0];
    expect(f(1).radiusPx).toEqual(f(0.25).radiusPx);
    expect([f(1).x, f(1).y]).toEqual([f(0.25).x, f(0.25).y]);
  });
});

describe("范围:限定一地 vs 全域", () => {
  // There are three kinds of spawn zone, each with its own reason; requiring the footprint
  // everywhere would be too strict, since wind and fire legitimately differ. The real invariant
  // is that every layer is anchored to this place, not that every layer uses the polygon.
  it("弥漫与下落类铺满这块地方的**轮廓**", () => {
    // An isometric diamond's bounding rect has about twice its area; emitting from it would rain
    // onto neighboring wards. Smoke is excluded because it is sited; see "浓烟成束" below.
    for (const p of ["rain", "snow", "quake", "wind"]) {
      const plan = planWeather(p, "medium", SMALL, CTX)!;
      for (const layer of plan.layers) expect(layer.zone.kind, p).toBe("ward");
    }
  });

  it("火是**点源**:从火心一小块发出,不铺满整个地方", () => {
    // A fire in a ward doesn't mean the whole ward is burning.
    const plan = planWeather("fire", "medium", SMALL, CTX)!;
    const cx = SMALL.box.x + SMALL.box.width / 2;
    for (const layer of plan.layers) {
      const z = layer.zone as { kind: string; x: number; width: number };
      expect(z.kind).toBe("rect");
      expect(z.x + z.width / 2).toBeCloseTo(cx);        // with one fire, it sits at the center of the place
      expect(z.width).toBeLessThan(SMALL.box.width);    // and is smaller than the place
    }
  });

  it("**一大片地方起火 = 多处在烧**,不是中间一处烧得更旺", () => {
    // Fire is a point source: a larger area gets more sites, not a higher emission rate. If area
    // were expressed through the rate, a world-wide fire would be a single cluster at the center
    // of the map.
    const one = planWeather("fire", "medium", SMALL, CTX)!;
    const many = planWeather("fire", "medium", HUGE, CTX)!;
    expect(one.glows!.length).toBe(1);
    expect(many.glows!.length).toBeGreaterThan(1);
    expect(many.glows!.length).toBeLessThanOrEqual(MAX_SITES);
    // The sites are actually spread out, not stacked on one point.
    const xs = many.glows!.map((g) => g.x);
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(HUGE.box.width * 0.2);
    // Each fire is still the same size: a fire is big because it burns hard, not because the place is big.
    expect(many.glows![0].radiusPx).toBeCloseTo(one.glows![0].radiusPx);
  });

  it("**浓烟成束**,不是一片弥漫的雾", () => {
    // Scattered randomly over the whole footprint like rain, smoke becomes gray blotches with no
    // source or direction and looks like a dirty screen. Smoke rises from the thing that is burning.
    for (const a of [SMALL, HUGE]) {
      const plan = planWeather("smoke", "medium", a, CTX)!;
      expect(plan.layers.length).toBeGreaterThan(0);
      for (const layer of plan.layers) {
        const z = layer.zone as { kind: string; width: number };
        expect(z.kind).toBe("rect");
        // The base is narrow. A wide spawn band spreads the column into fog immediately.
        expect(z.width).toBeLessThan(a.box.width * 0.3);
        // Vertical speed must dominate horizontal speed; that ratio is what keeps it a column.
        const vy = layer.velocity.y as { min: number; max: number };
        const vx = layer.velocity.x as { min: number; max: number };
        expect(Math.abs((vy.min + vy.max) / 2)).toBeGreaterThan(Math.abs(vx.max) * 3);
      }
    }
  });

  it("一片地方冒烟 = 好几束,数量有上限", () => {
    expect(planWeather("smoke", "medium", SMALL, CTX)!.layers).toHaveLength(1);
    const many = planWeather("smoke", "medium", HUGE, CTX)!.layers;
    expect(many.length).toBeGreaterThan(1);
    expect(many.length).toBeLessThanOrEqual(MAX_SITES);
  });

  it("有源现象的名单与引擎一致", () => {
    // core/interfaces/phenomenon.py is the authority. This mirror only serves the scene bench and
    // the drawing code; a sited broadcast with no location is lowered to none at the Broadcast
    // boundary and never reaches the renderer.
    expect([...SITED_PHENOMENA].sort()).toEqual(["fire", "quake", "smoke"]);
  });

  it("风在整片范围里各处生起,不是从一条缝里灌进来", () => {
    // A narrow inflow band works within one ward, but across the whole map it would only cover one
    // edge: particles live 1.5s and travel under 1000px, so the far side never gets any wind.
    const plan = planWeather("wind", "medium", HUGE, CTX)!;
    for (const layer of plan.layers) {
      expect(layer.zone.kind).toBe("ward");
      const pts = (layer.zone as { points: { x: number }[] }).points;
      const span = Math.max(...pts.map((q) => q.x)) - Math.min(...pts.map((q) => q.x));
      expect(span).toBeCloseTo(HUGE.box.width);        // spans the whole area, not just one edge
    }
  });

  it("天昏照轮廓铺,不铺满全屏", () => {
    // Darkness in one place must not dim the rest of the map, so there is no full-screen veil whatever the scope.
    const plan = planWeather("dark", "high", SMALL, CTX)!;
    expect(plan.veil).not.toBeNull();
    // The shape comes from area.footprint; the plan has no separate "view" branch that could go wrong.
    expect(plan.veil!.alpha).toBeGreaterThan(0);
  });

  it("天昏**只有罩子**,不飘「云」", () => {
    // Soft dots can't draw clouds, which are recognized by their outline; blurry dark patches just
    // look like a dirty screen. The veil alone conveys darkness, and extra blobs only make it worse.
    expect(planWeather("dark", "high", SMALL, CTX)!.layers).toHaveLength(0);
  });

  it("**范围越大,发的越多** —— 否则全域等于没下", () => {
    // With an absolute rate (125 drops/s) regardless of area, a ward gets a downpour while the whole
    // map is diluted to 1%, so world-wide rain shows nothing on screen. Point sources (fire, smoke)
    // are excluded: they scale by number of sites, see the two tests above.
    for (const p of ["rain", "snow", "wind", "quake"]) {
      const small = planWeather(p, "medium", SMALL, CTX)!.layers[0].perSecond;
      const huge = planWeather(p, "medium", HUGE, CTX)!.layers[0].perSecond;
      expect(huge, p).toBeGreaterThan(small * 3);
      // Capped by the budget: keeping the same density would be correct but needs over 10k particles.
      expect(huge, p).toBeLessThanOrEqual(small * MAX_AREA_SCALE + 1e-6);
    }
  });

  it("一坊大小的范围就是基准,不被放大", () => {
    // The multiplier starts at 1, so local weather keeps its tuned density when area is taken into account.
    const ward = planWeather("rain", "medium", area(3, 2), CTX)!.layers[0].perSecond;
    const tiny = planWeather("rain", "medium", area(1, 1), CTX)!.layers[0].perSecond;
    expect(ward).toBeCloseTo(tiny);
  });

  it("地动:那片地不在镜头里就不晃镜头", () => {
    // The camera stands for the viewer's eyes. A quake in a distant ward must not shake the whole view.
    expect(planWeather("quake", "high", area(4, 2, true), CTX)!.shake).not.toBeNull();
    expect(planWeather("quake", "high", area(4, 2, false), CTX)!.shake).toBeNull();
  });

  it("看不见的地动仍然留下落尘", () => {
    // No camera shake doesn't mean nothing happened: the ground still moved, the viewer just wasn't shaken.
    expect(planWeather("quake", "high", area(4, 2, false), CTX)!.layers.length).toBeGreaterThan(0);
  });
});

describe("严重程度是四维,不是一个旋钮", () => {
  const rate = (p: string, sev: string) =>
    planWeather(p, sev, SMALL, CTX)!.layers[0].perSecond;

  it("所有现象都随严重程度变强", () => {
    for (const p of PHENOMENA) {
      // Darkness has no particles; higher severity shows up in the veil.
      if (p === "dark") {
        const veil = (sev: string) => planWeather(p, sev, SMALL, CTX)!.veil!.alpha;
        expect(veil("high"), p).toBeGreaterThan(veil("low"));
        continue;
      }
      expect(rate(p, "high"), p).toBeGreaterThan(rate(p, "low"));
    }
  });

  it("雨下得大会**更急**", () => {
    const speed = (sev: string) =>
      planWeather("rain", sev, SMALL, CTX)!.layers[0].velocity.y as number;
    expect(speed("high")).toBeGreaterThan(speed("low"));
  });

  it("雪与雨**同速**下落 —— 区别在形状,不在快慢", () => {
    // Snow and rain share one speed constant: realistic snow would look like a frozen frame.
    const speed = (p: string, sev: string) =>
      planWeather(p, sev, SMALL, CTX)!.layers[p === "rain" ? 1 : 0].velocity.y as number;
    for (const sev of ["low", "medium", "high"]) {
      expect(speed("snow", sev), sev).toBeCloseTo(speed("rain", sev));
    }
  });

  it("雪下得大**不会更细长** —— 否则它就成了白色的雨", () => {
    // The part of vigor snow ignores is flake shape: heavier snow means bigger, denser flakes, not thinner, longer ones.
    const shape = (sev: string) => planWeather("snow", sev, SMALL, CTX)!.layers[0].size;
    expect(shape("high").x).toBeUndefined();     // uniform only: flakes are round
    expect(shape("high").uniform).toBeDefined();
  });

  it("雨小而密,雪大而疏", () => {
    // Rain and snow fall at the same speed, so shape and density must clearly differ or snow looks like white rain.
    const rain = planWeather("rain", "medium", SMALL, CTX)!;
    const snow = planWeather("snow", "medium", SMALL, CTX)!;
    const dropWidth = (rain.layers[1].size.x as number) * 3;
    const flake = ((snow.layers[0].size.uniform as { max: number }).max) * 32;
    expect(dropWidth).toBeLessThan(flake);
    const sum = (p: typeof rain) => p.layers.reduce((n, l) => n + l.perSecond, 0);
    expect(sum(rain)).toBeGreaterThan(sum(snow) * 1.5);
  });

  it("雨滴的**长宽比不随缩放改变** —— 拉远了也不会变胖", () => {
    // A per-axis minimum would fatten a drop as the camera pulls back (see boost).
    const ratio = (zoom: number) => {
      const L = planWeather("rain", "medium", SMALL, { tilePx: TILE, zoom })!.layers[1];
      return ((L.size.y as number) * 28) / ((L.size.x as number) * 3);
    };
    for (const zoom of [0.05, 0.2, 1, 4]) expect(ratio(zoom), `${zoom}`).toBeCloseTo(ratio(1));
  });

  it("雪下得大会更大片", () => {
    const size = (sev: string) =>
      (planWeather("snow", sev, SMALL, CTX)!.layers[0].size.uniform as { max: number }).max;
    expect(size("high")).toBeGreaterThan(size("low"));
  });

  it("烟浓不等于烟升得快", () => {
    const vy = (sev: string) =>
      JSON.stringify(planWeather("smoke", sev, SMALL, CTX)!.layers[0].velocity.y);
    expect(vy("high")).toEqual(vy("low"));
  });

  it("地动的强弱是幅度,不是次数", () => {
    const shake = (sev: string) => planWeather("quake", sev, SMALL, CTX)!.shake!.intensity;
    expect(shake("high")).toBeGreaterThan(shake("low"));
  });

  it("不认识的档位退到中档,不是崩掉", () => {
    const weird = planWeather("rain", "catastrophic", SMALL, CTX)!;
    const medium = planWeather("rain", "medium", SMALL, CTX)!;
    expect(weird.layers[0].perSecond).toEqual(medium.layers[0].perSecond);
  });
});

describe("一场雨的形状由雨丝讲,不由水花讲", () => {
  // When rain looks like hail, the drops aren't the problem. Near streaks are only 2.9px wide on a
  // real map; 10px round splashes at hundreds per second would be all you actually see.
  const rainLayers = (sev: string, ctx = CTX) => {
    const plan = planWeather("rain", sev, SMALL, ctx)!;
    return { far: plan.layers[0], near: plan.layers[1], splash: plan.layers[2] };
  };

  it("水花比雨丝短", () => {
    for (const sev of ["low", "medium", "high"]) {
      const { near, splash } = rainLayers(sev);
      const dropLen = (near.size.y as number) * 28;
      expect(maxWorldPx(splash), sev).toBeLessThan(dropLen);
    }
  });

  it("水花远比雨丝少", () => {
    // Not every drop needs a visible splash; splashes only show that the rain hits the ground.
    const { far, near, splash } = rainLayers("medium");
    expect(splash.perSecond * 8).toBeLessThan(far.perSecond + near.perSecond);
  });

  it("拉远了水花也不会反超雨丝", () => {
    // The visibility minimum is based on each particle's longest dimension, so small round splashes
    // are boosted much more than thin streaks and grow first as the camera pulls back. The two
    // have to be checked together.
    //
    // At extreme zoom-out they tie: the minimum makes every particle 1.5 screen pixels, so neither
    // is bigger (and there are far fewer splashes). The requirement is "never larger", not
    // "always smaller".
    for (const zoom of [0.05, 0.2, 1, 4]) {
      const { near, splash } = rainLayers("medium", { tilePx: TILE, zoom });
      expect(maxWorldPx(splash), `${zoom}`).toBeLessThanOrEqual((near.size.y as number) * 28);
    }
  });
});

describe("粒子预算", () => {
  it("再大的范围、再高的档位,同时活着的粒子都有上限", () => {
    // Live particles = rate × lifespan. Each is reasonable on its own (area sets the rate, travel
    // sets the lifespan), but the product can approach 10k, and only the whole plan sees it.
    for (const tilePx of [32, 64]) {
      for (const p of PHENOMENA) {
        for (const sev of ["low", "medium", "high"]) {
          const plan = planWeather(p, sev, HUGE, { tilePx, zoom: 1 })!;
          const live = plan.layers.reduce((n, l) => {
            const life = typeof l.lifespanMs === "number"
              ? l.lifespanMs
              : (l.lifespanMs.min + l.lifespanMs.max) / 2;
            return n + (l.perSecond * life) / 1000;
          }, 0);
          expect(live, `${p}/${sev}/${tilePx}`).toBeLessThanOrEqual(MAX_LIVE_PARTICLES + 1e-6);
        }
      }
    }
  });

  it("超预算时按同一个比例收,不是砍掉某一层", () => {
    // Weather should thin out, never lose a layer: rain without its far layer looks flat.
    const big = planWeather("rain", "high", HUGE, { tilePx: 64, zoom: 1 })!;
    const small = planWeather("rain", "high", SMALL, { tilePx: 64, zoom: 1 })!;
    expect(big.layers).toHaveLength(small.layers.length);
    const ratio = (p: typeof big) => p.layers[1].perSecond / p.layers[0].perSecond;
    expect(ratio(big)).toBeCloseTo(ratio(small));
  });
});

describe("透明度不会溢出", () => {
  it("最高档的 alpha 仍 ≤ 1", () => {
    // presence is 1.15 at high, so multiplying directly would push alpha above 1.
    for (const p of PHENOMENA) {
      for (const layer of planWeather(p, "high", SMALL, CTX)!.layers) {
        const values = typeof layer.alpha === "number"
          ? [layer.alpha]
          : Object.values(layer.alpha as Record<string, number>);
        for (const v of values) expect(v, `${p}`).toBeLessThanOrEqual(1);
      }
    }
  });
});
