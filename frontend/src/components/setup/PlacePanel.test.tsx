import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import PlacePanel, { droppedNames, namesByPlace, placeRows, withPlaceNames } from "./PlacePanel";
import type { Places } from "../../lib/places";

const places: Places = {
  donggong: { location_id: "donggong", name: "东宫", description: "皇太子居所", sub_places: [] },
  gate: { location_id: "gate", name: "玄武门", description: "", sub_places: [] },
  market: { location_id: "market", name: "西市", description: "胡商云集", sub_places: [] },
};

function markup(editable: boolean, subPlaces = [
  { location_id: "donggong", name: "显德殿" }, { location_id: "gate", name: "城楼" },
  { location_id: "donggong", name: "崇教殿" },
]) {
  return renderToStaticMarkup(
    <PlacePanel subPlaces={subPlaces} places={places} editable={editable}
      onSave={vi.fn()} onEditingChange={vi.fn()} />,
  );
}

describe("place panel", () => {
  it("groups the sub-places under their places", () => {
    expect(namesByPlace([
      { location_id: "a", name: "甲" }, { location_id: "b", name: "乙" }, { location_id: "a", name: "丙" },
    ])).toEqual({ a: ["甲", "丙"], b: ["乙"] });
  });

  it("lists every place, then a place the table lacks as 某地, never its id", () => {
    const rows = placeRows(places, [{ location_id: "nowhere", name: "荒亭" }]);
    expect(rows.map((r) => r.name)).toEqual(["东宫", "玄武门", "西市", "某地"]);
    expect(rows[3].location_id).toBe("nowhere");
  });

  it("shows each place's name, description and sub-places", () => {
    const html = markup(false);
    expect(html).toContain("东宫");
    expect(html).toContain("皇太子居所");
    expect(html).toContain("子地点：显德殿、崇教殿");
    expect(html).toContain("子地点：城楼");
    expect(html).not.toContain("编辑");
  });

  it("collapses to the places with sub-places and offers the rest", () => {
    const html = markup(false);
    expect(html).not.toContain("西市");
    expect(html).toContain("展开全部 3 个地点");
  });

  it("saves one place's edit with every other place's sub-places kept", () => {
    const subPlaces = [{ location_id: "donggong", name: "显德殿" }, { location_id: "gate", name: "城楼" }];
    expect(withPlaceNames(placeRows(places, subPlaces), subPlaces, "market", [" 胡商邸店 "])).toEqual([
      { location_id: "donggong", name: "显德殿" }, { location_id: "gate", name: "城楼" },
      { location_id: "market", name: "胡商邸店" },
    ]);
    expect(withPlaceNames(placeRows(places, subPlaces), subPlaces, "donggong", [])).toEqual([
      { location_id: "gate", name: "城楼" },
    ]);
  });

  it("offers an edit per place only while the setup is revisable", () => {
    expect(markup(true).match(/编辑/g)).toHaveLength(2);
    expect(markup(true)).toContain("编辑");
    expect(markup(true, [])).toContain("暂无子地点");
    expect(markup(true, [])).toContain("展开全部 3 个地点");
  });

  it("names the sub-places an edit drops, a rename included, but not ones it only adds", () => {
    expect(droppedNames(["显德殿", "崇教殿"], ["显德殿 ", "丽正殿"])).toEqual(["崇教殿"]);
    expect(droppedNames(["显德殿"], ["显德殿", "丽正殿"])).toEqual([]);
  });
});
