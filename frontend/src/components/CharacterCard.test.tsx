import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentProfile } from "../types";

const { agentProfile, useCast, usePlaces } = vi.hoisted(() => ({
  agentProfile: vi.fn(),
  useCast: vi.fn(() => { throw new Error("Draft cards must not load the global cast"); }),
  usePlaces: vi.fn(() => { throw new Error("Draft cards must not load global places"); }),
}));
vi.mock("../api/client", () => ({ api: { agentProfile } }));
vi.mock("../lib/cast", () => ({ useCast, loadCast: vi.fn() }));
vi.mock("../lib/places", async (importOriginal) => ({
  ...await importOriginal<typeof import("../lib/places")>(),
  usePlaces,
}));
import CharacterCard from "./CharacterCard";

const profile: AgentProfile = {
  agent_id: "draft-agent", name: "草稿角色", role: "工匠", age: 31, gender: "女",
  background: "在城中修缮器物", appearance: "总带着工具袋", color: "#5a8b6c",
  core_traits: ["谨慎"], core_values: ["诚实"], self_image: "可靠的手艺人",
  life_goal: "开一家自己的店", secret: "藏着一张旧地图", is_main_character: true,
};

function renderCard(current: AgentProfile, viewer: string | null = null) {
  return renderToStaticMarkup(<CharacterCard
    cast={{ [current.agent_id]: current }}
    places={{}}
    profile={current}
    roster={[{ id: current.agent_id, is_main_character: true }]}
    activeId={current.agent_id}
    onActiveChange={() => {}}
    viewer={viewer}
  />);
}

beforeEach(() => vi.clearAllMocks());

describe("draft character card", () => {
  it("renders the supplied setup identity and private profile without global reads", () => {
    const markup = renderCard(profile);
    for (const text of [profile.name, profile.background, profile.self_image, profile.life_goal, profile.secret]) {
      expect(markup).toContain(text);
    }
    expect(useCast).not.toHaveBeenCalled();
    expect(usePlaces).not.toHaveBeenCalled();
    expect(agentProfile).not.toHaveBeenCalled();
  });

  it("renders revised setup fields from new props without retaining the previous profile", () => {
    expect(renderCard(profile)).toContain(profile.secret);
    const revised = { ...profile, name: "改名后的角色", secret: "新秘密", life_goal: "前往海边" };
    const markup = renderCard(revised);
    expect(markup).toContain(revised.name);
    expect(markup).toContain(revised.secret);
    expect(markup).toContain(revised.life_goal);
    expect(markup).not.toContain(profile.name);
    expect(markup).not.toContain(profile.secret);
    expect(markup).not.toContain(profile.life_goal);
    expect(agentProfile).not.toHaveBeenCalled();
  });

  it("keeps another character's dossier private even when a profile is supplied", () => {
    const markup = renderCard(profile, "another-agent");
    expect(markup).toContain(profile.name);
    expect(markup).not.toContain(profile.secret);
    expect(markup).not.toContain(profile.self_image);
    expect(markup).not.toContain(profile.life_goal);
  });
});
