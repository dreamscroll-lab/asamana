/** Importing a map: what each kind of backend answer turns into. */

import { afterEach, describe, expect, it, vi } from "vitest";

import { importTemplate, suggestTemplateName, TEMPLATE_NAME_RE } from "./importTemplate";

function answer(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: status >= 200 && status < 300,
      status,
      statusText: "no",
      json: async () => body,
    })),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("导入的结果", () => {
  it("装好了就报落地的文件数,和按地图推导写入的连接数", async () => {
    answer(200, { template: "city", files: 33, connections: 61 });
    expect(await importTemplate("city", new Blob())).toEqual({
      ok: true,
      files: 33,
      connections: 61,
    });
  });

  it("名字被占用:原样带出后端的说法,因为两种占用(目录名 / world_name)要分开说", async () => {
    answer(409, { detail: "world_name '长安' is already used by the map 'changan_iso'" });
    expect(await importTemplate("city", new Blob())).toEqual({
      ok: false,
      kind: "conflict",
      message: "world_name '长安' is already used by the map 'changan_iso'",
    });
  });

  it("契约不过:逐条拿到,顺序不动", async () => {
    answer(422, { detail: { message: "nope", problems: ["缺 world_name", "路网断了"] } });
    expect(await importTemplate("city", new Blob())).toEqual({
      ok: false,
      kind: "rejected",
      problems: ["缺 world_name", "路网断了"],
    });
  });

  it("FastAPI 自己的 422 是一串对象,不能当成契约判词", async () => {
    answer(422, { detail: [{ loc: ["query", "overwrite"], msg: "not a boolean" }] });
    const outcome = await importTemplate("city", new Blob());
    expect(outcome.ok).toBe(false);
    expect(outcome).not.toHaveProperty("problems");
  });

  it("其余状态给出 detail 原文", async () => {
    answer(400, { detail: "archive exceeds 64 MB" });
    expect(await importTemplate("city", new Blob())).toEqual({
      ok: false,
      kind: "error",
      message: "archive exceeds 64 MB",
    });
  });

  it("网络就没通也是一种结果,不往外抛", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new Error("offline"))));
    expect(await importTemplate("city", new Blob())).toEqual({
      ok: false,
      kind: "error",
      message: "offline",
    });
  });

  it("请求里没有任何覆盖开关 —— 同名一律拒绝", async () => {
    answer(200, { files: 1, connections: 0 });
    await importTemplate("city", new Blob());
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/templates/city/archive");
  });
});

describe("从文件名猜目录名", () => {
  it.each([
    ["Chang'an Iso v2.zip", "chang_an_iso_v2"],
    ["长安.zip", ""],
    ["__metro__.ZIP", "metro"],
    ["9lives.zip", "lives"],
  ])("%s → %s", (filename, expected) => {
    expect(suggestTemplateName(filename)).toBe(expected);
  });

  it("猜出来的非空名字一定是后端收的名字", () => {
    for (const filename of ["Chang'an Iso v2.zip", "METRO map.zip", "a--b.zip"]) {
      expect(TEMPLATE_NAME_RE.test(suggestTemplateName(filename))).toBe(true);
    }
  });
});

describe("名字的词汇", () => {
  it.each(["Evil", "_example", "9lives", "a.b", "with-hyphen", "", "a/b", "../evil"])(
    "%s 进不来",
    (bad) => expect(TEMPLATE_NAME_RE.test(bad)).toBe(false),
  );

  it.each(["metro", "changan_iso", "changan_iso_v2"])("%s 进得来", (good) =>
    expect(TEMPLATE_NAME_RE.test(good)).toBe(true),
  );
});
