// A character's inner state read off their body, like a medical chart: the standing figure the map
// draws, with each cognition field pinned to the part of the body it lives in. A summoned overlay
// over the map, as the relationship graph is.
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { api, API_BASE } from "../api/client";
import { reportedDead, unfinishedGoals } from "../lib/agentState";
import { avatarStyle, emotionColor } from "../lib/avatar";
import { identityColor, parseHex, toHex } from "../lib/color";
import { dyePixels, hsvOf } from "../lib/dye";
import { type FrameRect, loadAtlasFrames } from "../lib/atlasFrames";
import { loadCast } from "../phaser/mapSource";
import type { CharacterSet, Demographics } from "../phaser/skins";
import type { AgentProfile, AgentStateSummary, WorldGraph } from "../types";
import type { CardAgent } from "./CharacterCard";

// Facing the viewer and slightly right, so the callouts on both sides read as one chart. A
// portrait stands this way too (template_check holds its outline to the SE idle frame).
const FACING = "SE" as const;

// --- the figure -------------------------------------------------------------------------------

const casts = new Map<string, Promise<CharacterSet>>();

/** The world's cast manifest, fetched once per world and shared by every open of the overlay. */
function castOf(worldId: string): Promise<CharacterSet> {
  let cast = casts.get(worldId);
  if (!cast) {
    cast = loadCast(`${API_BASE}/api/worlds/${worldId}`);
    cast.catch(() => casts.delete(worldId)); // a failed fetch must not stick for the session
    casts.set(worldId, cast);
  }
  return cast;
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(url));
    img.src = url;
  });
}

interface Figure {
  url: string;
  aspect: number; // width / height of the figure's opaque bounds
  // The body's height as a fraction of its idle frame. Every frame of a cast is the same size and
  // stands on its bottom edge, so this is the body's height on one scale shared by the whole cast.
  stature: number;
}

// Alpha at or below this is cut-out residue, not body (worlds/template_check.py PORTRAIT_ALPHA_FLOOR).
const ALPHA_FLOOR = 8;

/** One image region, dyed in the identity colour and cropped to the opaque body. */
function cutBody(img: HTMLImageElement, rect: FrameRect, flip: boolean, color: string): HTMLCanvasElement {
  const frame = document.createElement("canvas");
  frame.width = rect.width;
  frame.height = rect.height;
  const ctx = frame.getContext("2d", { willReadFrequently: true })!;
  if (flip) {
    ctx.translate(rect.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(img, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
  const pixels = ctx.getImageData(0, 0, rect.width, rect.height);
  const dye = parseHex(color);
  if (dye !== null) dyePixels(pixels.data, hsvOf(dye));

  let [x0, y0, x1, y1] = [rect.width, rect.height, -1, -1];
  for (let y = 0; y < rect.height; y++) {
    for (let x = 0; x < rect.width; x++) {
      if (pixels.data[(y * rect.width + x) * 4 + 3] <= ALPHA_FLOOR) continue;
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
  }
  if (x1 < 0) throw new Error("empty figure");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.putImageData(pixels, 0, 0);

  const out = document.createElement("canvas");
  out.width = x1 - x0 + 1;
  out.height = y1 - y0 + 1;
  out.getContext("2d")!.drawImage(frame, x0, y0, out.width, out.height, 0, 0, out.width, out.height);
  return out;
}

/**
 * This person as the map draws them standing — the portrait when the art ships one, else the idle
 * frame — dyed in their identity colour and cropped to the body, so the chart's anchors are
 * fractions of the body rather than of an image's empty margin. Its size always comes from the
 * idle frame, which a cropped portrait no longer knows.
 */
async function drawFigure(worldId: string, who: Demographics, color: string): Promise<Figure> {
  const cast = await castOf(worldId);
  const body = cast.bodyFor(who);
  const sheet = cast.atlases.find((a) => a.key === body);
  const art = cast.poseArt(body, "idle", FACING);
  if (!sheet || !art) throw new Error(`no idle art for ${body}`);
  const [frames, atlas, portrait] = await Promise.all([
    loadAtlasFrames(sheet.atlas),
    loadImage(sheet.image),
    sheet.portrait ? loadImage(sheet.portrait) : Promise.resolve(null),
  ]);
  const rect = frames[art.frames[0][1]];
  if (!rect) throw new Error(`frame ${art.frames[0][1]} missing`);

  const idle = cutBody(atlas, rect, art.flip, color);
  const out = portrait
    ? cutBody(portrait, { x: 0, y: 0, width: portrait.naturalWidth, height: portrait.naturalHeight }, art.flip, color)
    : idle;
  return { url: out.toDataURL(), aspect: out.width / out.height, stature: idle.height / rect.height };
}

// --- the chart --------------------------------------------------------------------------------

type OrganKey = "mind" | "aim" | "mood" | "heart" | "need" | "hands" | "feet";

interface Organ {
  key: OrganKey;
  title: string;
  icon: string;
  accent: string; // text colour class
  side: "left" | "right";
  // Where the leader line lands, as a fraction of the figure's opaque bounds (SE-facing idle art).
  anchor: [number, number];
}

// Order within a side is top to bottom: callout slots are dealt in this order.
const ORGANS: Organ[] = [
  { key: "mind", title: "人设", icon: "🧠", accent: "text-violet-300", side: "left", anchor: [0.42, 0.04] },
  { key: "mood", title: "情绪", icon: "🎭", accent: "text-amber-300", side: "left", anchor: [0.52, 0.1] },
  { key: "need", title: "需求", icon: "🔥", accent: "text-orange-300", side: "left", anchor: [0.45, 0.42] },
  { key: "feet", title: "位置", icon: "📍", accent: "text-sky-300", side: "left", anchor: [0.38, 0.95] },
  { key: "aim", title: "目标", icon: "🎯", accent: "text-cyan-300", side: "right", anchor: [0.62, 0.08] },
  { key: "heart", title: "关系", icon: "🤝", accent: "text-rose-300", side: "right", anchor: [0.6, 0.27] },
  { key: "hands", title: "行动", icon: "✋", accent: "text-emerald-300", side: "right", anchor: [0.82, 0.52] },
];

// Half a callout's height, for fitting its detail above or below it.
const CARD_HALF_H = 16;

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

function Meter({ value, min = 0, max = 1, tone }: { value: number; min?: number; max?: number; tone: string }) {
  const pct = ((Math.min(max, Math.max(min, value)) - min) / (max - min)) * 100;
  return (
    <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
      <div className={`h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="text-[10px] font-bold tracking-wider text-slate-500 mb-1">{label}</div>
      <div className="text-xs text-slate-300 leading-relaxed">{children}</div>
    </div>
  );
}

function Chips({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-slate-600">—</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((t, i) => (
        <span key={i} className="px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-700/60 text-slate-200">
          {t}
        </span>
      ))}
    </div>
  );
}

function Numbered({ items }: { items: string[] }) {
  if (!items.length) return <span className="text-slate-600">—</span>;
  return (
    <ol className="space-y-0.5">
      {items.map((g, i) => (
        <li key={i} className="flex gap-1.5">
          <span className="shrink-0 font-mono tabular-nums text-slate-500">{i + 1}.</span>
          <span>{g}</span>
        </li>
      ))}
    </ol>
  );
}

const pct = (v: number) => Math.round(v * 100);

export default function CharacterBody({
  worldId,
  agent,
  live,
  graph,
}: {
  worldId: string;
  agent: CardAgent;
  live?: AgentStateSummary;
  graph: WorldGraph | null;
}) {
  const [profile, setProfile] = useState<AgentProfile | null>(null);
  const [figure, setFigure] = useState<Figure | null>(null);
  const [figureFailed, setFigureFailed] = useState(false);
  const [hovered, setHovered] = useState<OrganKey | null>(null);
  const [stageRef, stage] = useSize<HTMLDivElement>();

  useEffect(() => {
    let alive = true;
    setProfile(null);
    setFigure(null);
    setFigureFailed(false);
    api
      .agentProfile(worldId, agent.id)
      .then((p) => alive && setProfile(p))
      // No profile, no body to choose: say so rather than leave the figure loading.
      .catch(() => alive && setFigureFailed(true));
    return () => {
      alive = false;
    };
  }, [worldId, agent.id]);

  // Wait for the profile: the body is chosen by gender and age, and drawing a guess first would
  // swap one person for another on screen.
  useEffect(() => {
    if (!profile) return;
    let alive = true;
    setFigure(null);
    setFigureFailed(false);
    drawFigure(worldId, { gender: profile.gender, age: profile.age }, agent.color || profile.color)
      .then((f) => alive && setFigure(f))
      .catch(() => alive && setFigureFailed(true));
    return () => {
      alive = false;
    };
  }, [worldId, profile, agent.color]);

  const dead = live ? reportedDead(live) : false;
  const { planned, owed } = useMemo(() => unfinishedGoals(live), [live]);

  // Ties held by this person, strongest first: how much they lean on someone either way.
  const ties = useMemo(() => {
    if (!graph) return null;
    const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
    return graph.edges
      .filter((e) => e.from_id === agent.id && nodes.has(e.to_id))
      .map((e) => ({ edge: e, to: nodes.get(e.to_id)! }))
      .sort((a, b) => Math.abs(b.edge.affection) + b.edge.trust - (Math.abs(a.edge.affection) + a.edge.trust));
  }, [graph, agent.id]);

  const details: Record<OrganKey, ReactNode> = {
    mind: (
      <>
        <Section label="性格特点"><Chips items={profile?.core_traits ?? []} /></Section>
        <Section label="价值观"><Chips items={profile?.core_values ?? []} /></Section>
        <Section label="自我认知">{profile?.self_image || "—"}</Section>
        {/* Most people have none, so no "—" placeholder. */}
        {profile?.secret && <Section label="🤫 秘密"><span className="text-rose-300">{profile.secret}</span></Section>}
      </>
    ),
    aim: (
      <>
        <Section label="人生目标">{profile?.life_goal || "—"}</Section>
        <Section label="长期目标"><Numbered items={live?.long_term_goals ?? []} /></Section>
      </>
    ),
    mood: live ? (
      <>
        <Section label="此刻情绪">
          <span className={`text-sm font-semibold ${emotionColor(live.emotion_valence)}`}>{live.emotion_label}</span>
        </Section>
        {live.emotion_intensity != null && (
          <Section label={`强度 ${pct(live.emotion_intensity)}`}>
            <Meter value={live.emotion_intensity} tone="bg-amber-400/80" />
          </Section>
        )}
        {live.emotion_valence != null && (
          <Section label={`正负值 ${pct(live.emotion_valence)}`}>
            <Meter
              value={live.emotion_valence}
              min={-1}
              tone={live.emotion_valence < 0 ? "bg-rose-400/80" : "bg-teal-400/80"}
            />
          </Section>
        )}
      </>
    ) : (
      <span className="text-slate-600">—</span>
    ),
    heart:
      ties == null ? (
        <span className="text-slate-500 animate-pulse">读取关系…</span>
      ) : ties.length === 0 ? (
        <span className="text-slate-600">尚无关系</span>
      ) : (
        <div className="space-y-2.5">
          {ties.map(({ edge, to }) => (
            <div key={to.id} className="bg-slate-950/50 border border-slate-800/70 rounded-lg px-2.5 py-2 space-y-1.5">
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-4 h-4 rounded shrink-0`}
                  style={avatarStyle(to.color, to.id)}
                />
                <span className="text-slate-100 font-medium truncate">{to.name}</span>
                <span className="text-slate-500 truncate">{edge.labels.join(" · ")}</span>
              </div>
              <div className="grid grid-cols-[3rem_1fr_2rem] items-center gap-x-2 gap-y-1 text-[10px] text-slate-500">
                <span>信任</span>
                <Meter value={edge.trust} tone="bg-emerald-400/80" />
                <span className="font-mono text-right">{pct(edge.trust)}</span>
                <span>好感</span>
                <Meter value={edge.affection} min={-1} tone={edge.affection < 0 ? "bg-rose-400/80" : "bg-pink-400/80"} />
                <span className="font-mono text-right">{pct(edge.affection)}</span>
              </div>
              {edge.history_summary && <p className="text-[11px] text-slate-400">{edge.history_summary}</p>}
            </div>
          ))}
        </div>
      ),
    need: (
      <Section label="当前需求">
        {live?.dominant_need ? (
          <span className="text-sm font-semibold text-orange-300">{live.dominant_need_label}</span>
        ) : (
          "—"
        )}
      </Section>
    ),
    hands: (
      <>
        <Section label="状态">{live ? live.activity_label || "—" : "—"}</Section>
        <Section label="短期目标"><Numbered items={planned} /></Section>
        {owed.length > 0 && <Section label="待办事项"><Numbered items={owed} /></Section>}
      </>
    ),
    feet: (
      <>
        <Section label="地点">{live?.location || "—"}</Section>
        {live?.condition && <Section label="处境"><span className="text-amber-300">{live.condition}</span></Section>}
      </>
    ),
  };

  // Geometry, in stage pixels: the figure stands centred under its nameplate, the callouts stack
  // in two columns beside it, and each leader line runs out from its callout and elbows onto the body.
  // The callouts keep their width and the figure gives way.
  //
  // The scale is set by the idle FRAME, not the body: the frame fills the stage height and the
  // body stands on its bottom edge at its own stature, so a child draws shorter than an adult as
  // on the map. Don't scale the body to fill the stage: that draws every age the same height.
  const gutter = 36;
  const plateH = 64;
  const plateGap = 20; // between the nameplate and the top of the head
  const floor = 16;
  const cardW = 84; // a title only; the content opens on hover
  const aspect = figure?.aspect ?? 0.38;
  const stature = figure?.stature ?? 0.75;
  const frameH = Math.min(
    stage.h - floor,
    (stage.h - floor - plateH - plateGap) / stature, // the nameplate still fits over the head (short stages only)
    (stage.w - 2 * (cardW + gutter + 8)) / aspect / stature,
  );
  const figH = Math.max(0, frameH * stature);
  const figW = figH * aspect;
  const figX = (stage.w - figW) / 2;
  const figY = stage.h - floor - figH;
  const layout = (["left", "right"] as const).flatMap((side) => {
    const organs = ORGANS.filter((o) => o.side === side);
    return organs.map((o, i) => {
      const y = stage.h * (0.12 + (0.76 * (i + 0.5)) / organs.length);
      const edgeX = side === "left" ? figX - gutter : figX + figW + gutter;
      const ax = figX + o.anchor[0] * figW;
      const ay = figY + o.anchor[1] * figH;
      const elbowX = side === "left" ? Math.min(edgeX + 20, ax - 6) : Math.max(edgeX - 20, ax + 6);
      // The detail opens beside its callout, over the figure, so the column stays free to hover
      // the next one. It grows down from the callout's top or, where the stage has more room the
      // other way, up from its bottom, and stops at the stage edge (the overlay clips).
      const down = stage.h - (y - CARD_HALF_H) - 8;
      const up = y + CARD_HALF_H - 8;
      return { o, y, edgeX, ax, ay, elbowX, up: up > down, room: Math.max(up, down) };
    });
  });
  const accent = toHex(identityColor(agent.color, agent.id));
  const role = agent.role || profile?.role;

  return (
    <div ref={stageRef} className="relative h-full min-h-0" onMouseLeave={() => setHovered(null)}>
      {stage.w > 0 && (
        <>
          {/* Floor glow in the identity colour: the map's ground ring, at chart scale. */}
          <div
            className="absolute rounded-[50%] blur-md opacity-40"
            style={{
              left: figX - figW * 0.2,
              top: figY + figH * 0.94,
              width: figW * 1.4,
              height: figH * 0.08,
              background: accent,
            }}
          />

          {/* Nameplate over the head, as the map hangs one over every figure. */}
          <div
            className="absolute -translate-x-1/2 flex flex-col items-center gap-1 text-center"
            style={{ left: figX + figW / 2, top: figY - plateH - plateGap, width: Math.max(220, figW * 1.6) }}
          >
            <div className="text-base font-extrabold text-slate-100 truncate max-w-full">
              {agent.name}
              {agent.is_main_character && <span className="text-indigo-400"> ★</span>}
            </div>
            <div className="text-[11px] text-slate-400 truncate max-w-full">
              {[role, profile?.gender, profile?.age != null ? `${profile.age} 岁` : ""].filter(Boolean).join(" · ")}
            </div>
            {live?.vitality != null && (
              <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                <span>生命值</span>
                <div className="w-24">
                  <Meter value={live.vitality} tone={live.vitality < 0.3 ? "bg-rose-500" : "bg-emerald-500/80"} />
                </div>
                <span className="font-mono">{pct(live.vitality)}</span>
              </div>
            )}
          </div>

          {figure ? (
            <img
              src={figure.url}
              alt={agent.name}
              draggable={false}
              className={`absolute select-none ${dead ? "grayscale opacity-50" : ""}`}
              style={{ left: figX, top: figY, width: figW, height: figH }}
            />
          ) : (
            <div
              className="absolute grid place-items-center text-xs text-slate-500"
              style={{ left: figX, top: figY, width: figW, height: figH }}
            >
              {figureFailed ? "形象图不可用" : <span className="animate-pulse">描绘中…</span>}
            </div>
          )}
          {dead && (
            <div
              className="absolute px-3 py-1 border-2 border-rose-500/80 text-rose-400 text-lg font-black tracking-[0.3em] rounded"
              style={{ left: figX + figW / 2, top: figY + figH * 0.35, transform: "translateX(-50%) rotate(-12deg)" }}
            >
              OVER
            </div>
          )}

          <svg className="absolute inset-0 pointer-events-none" width={stage.w} height={stage.h}>
            {layout.map(({ o, y, edgeX, ax, ay, elbowX }) => {
              const on = o.key === hovered;
              return (
                <g key={o.key} className={on ? "text-slate-100" : "text-slate-500"}>
                  <polyline
                    points={`${edgeX},${y} ${elbowX},${y} ${ax},${ay}`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={on ? 1.5 : 1}
                    strokeDasharray={on ? undefined : "3,3"}
                    opacity={on ? 0.9 : 0.6}
                  />
                  <circle cx={ax} cy={ay} r={on ? 4 : 3} fill={on ? accent : "currentColor"} />
                  {on && (
                    <circle cx={ax} cy={ay} r={4} fill="none" stroke={accent} strokeWidth={1.5}>
                      <animate attributeName="r" from="4" to="14" dur="1.4s" repeatCount="indefinite" />
                      <animate attributeName="opacity" from="0.9" to="0" dur="1.4s" repeatCount="indefinite" />
                    </circle>
                  )}
                </g>
              );
            })}
          </svg>

          {layout.map(({ o, y, edgeX, up, room }) => {
            const on = o.key === hovered;
            return (
              // Callout and its detail share one hover region, so the pointer can move into a long
              // detail (and scroll it) without it closing.
              <div
                key={o.key}
                className={`absolute -translate-y-1/2 ${on ? "z-10" : ""}`}
                style={{ top: y, width: cardW, left: o.side === "left" ? edgeX - cardW : edgeX }}
                onMouseEnter={() => setHovered(o.key)}
                onMouseLeave={() => setHovered((h) => (h === o.key ? null : h))}
              >
                <div
                  className="rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-2 backdrop-blur-sm cursor-default"
                  style={{ borderColor: on ? accent : undefined }}
                >
                  <span className={`text-xs font-bold ${o.accent}`}>
                    {o.icon} {o.title}
                  </span>
                </div>
                {on && (
                  // The padding bridges the gap to the callout, so crossing it keeps the hover.
                  <div
                    className={`absolute ${up ? "bottom-0" : "top-0"} ${
                      o.side === "left" ? "left-full pl-2" : "right-full pr-2"
                    }`}
                  >
                    <div
                      style={{ maxHeight: room, borderColor: accent }}
                      className="w-72 overflow-y-auto rounded-xl border bg-slate-900/60 backdrop-blur-md p-3 space-y-3 shadow-[0_10px_30px_rgb(0,0,0,0.5)]"
                    >
                      {details[o.key]}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
