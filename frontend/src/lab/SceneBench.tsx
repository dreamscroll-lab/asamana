import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import MapStage from "../components/MapStage";
import NarrativeStep from "../components/narrative/NarrativeStep";
import Checklist from "./Checklist";
import FrameReadout, { type Figure } from "./FrameReadout";
import { loadMarks, markKey, saveMarks, type Mark, type Marks } from "./marks";
import { resolvePlaces } from "./places";
import { buildScenes, castDemographics } from "./scenes";
import { renderStepInto } from "../phaser/driveStep";
import type { MapSource } from "../phaser/mapSource";
import type { TiledWorldScene } from "../phaser/TiledWorldScene";

/** A hairline between control groups (play, beat, camera, speed), or the row reads as equal buttons. */
const Rule = () => <span className="w-px h-4 bg-slate-800 mx-1.5 shrink-0" />;

/** Breath between two of a scene's steps, so the beats stay separable by eye. */
const STEP_GAP_MS = 420;
const SPEEDS = [0.1, 0.25, 0.5, 1, 2];

/**
 * The scenes bench: every deed and every heading, on demand, so rare acts' art can be
 * checked without building a world. Drives the same renderer off hand-written step payloads
 * (see lab/scenes/).
 *
 * Only the fixtures and the bench live here. How a step is drawn belongs to the shared path
 * (`MapStage`, `loadMapSource`, `renderStepInto`, `setCast`); a second answer here would certify a
 * build nobody ships. Choosing and fetching the template belongs to the shell (`views/LabView`).
 */
export default function SceneBench({
  template,
  source,
  pickId,
  onPick,
}: {
  template: string;
  source: MapSource;
  pickId: string | null;
  onPick: (sceneId: string) => void;
}) {
  const [scene, setScene] = useState<TiledWorldScene | null>(null);
  const [take, setTake] = useState(0); // bumped to replay the beat on screen
  const [beat, setBeat] = useState(0); // which step of the scene is on screen
  const [ready, setReady] = useState(false); // this beat's timeline has finished
  const [playing, setPlaying] = useState(true);
  const [crash, setCrash] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1);
  const [follow, setFollow] = useState(true);
  const [loop, setLoop] = useState(false);
  const [logOpen, setLogOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [marks, setMarks] = useState<Marks>(loadMarks);
  const loopRef = useRef(loop);
  loopRef.current = loop;

  useEffect(() => {
    if (scene) scene.setSpeed(speed);
  }, [scene, speed]);

  // Functions of the loaded map only; a template whose places can't fill the roles (see places.ts)
  // yields null, and the stage says so.
  const places = useMemo(() => resolvePlaces(source), [source]);
  const scenes = useMemo(() => (places ? buildScenes(places) : []), [places]);
  // Keep the same scene selected across a template switch when it still exists.
  const pick = scenes.find((s) => s.id === pickId) ?? scenes[0];

  // Owned here, not in the list, because ↑/↓ walk it too and must skip hidden scenes. The `watch`
  // text is searched too: you tend to remember what a scene asked you to look at, not its title.
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return scenes;
    return scenes.filter((s) => `${s.title} ${s.group} ${s.watch}`.toLowerCase().includes(needle));
  }, [scenes, query]);

  // Put the scene in the address so a reload after a render edit comes back to it.
  useEffect(() => {
    if (pick && pick.id !== pickId) onPick(pick.id);
  }, [pick, pickId, onPick]);

  // The roster's gender and age, as the observation view feeds from step-0 profiles. Declared
  // before the playback effect so a token is built with the right body, not re-skinned a frame later.
  useEffect(() => {
    scene?.setCast(castDemographics());
  }, [scene]);

  // Every figure this scene puts on the map, read off its steps. Focus is always all of them,
  // NPCs included: a figure outside the focus is dimmed to 0.3 alpha and its action effects are
  // gated out, and here everyone on the map is under examination.
  const figures: Figure[] = useMemo(() => {
    const seen = new Map<string, Figure>();
    for (const st of pick?.steps ?? []) {
      for (const s of Object.values(st.agent_states)) {
        if (!seen.has(s.agent_id)) {
          seen.set(s.agent_id, { id: s.agent_id, name: s.agent_name, color: s.color || "#94a3b8" });
        }
      }
      for (const n of st.npcs ?? []) {
        if (!seen.has(n.npc_id)) {
          seen.set(n.npc_id, { id: n.npc_id, name: n.name || "无名", color: n.color || "#64748b" });
        }
      }
    }
    return [...seen.values()];
  }, [pick]);

  useEffect(() => {
    const ids = figures.map((f) => f.id);
    scene?.setFocus({ agents: follow ? ids : [], place: null, follow });
  }, [scene, follow, figures]);

  // A beat is finished when its timeline is, as in the observation view's driver, not after a guessed interval.
  useEffect(() => {
    const st = pick?.steps[beat];
    if (!scene || !pick || !st) return;
    let cancelled = false;
    setReady(false);
    setCrash(null);
    renderStepInto(scene, st).then((err) => {
      if (cancelled) return;
      // A live world swallows this; the bench stops and says which beat broke.
      if (err) {
        setCrash(err.message);
        setPlaying(false);
      }
      setReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, [scene, pick, beat, take]);

  // Advance only once the beat has finished and while playing, so pausing holds the frame.
  useEffect(() => {
    if (!playing || !ready || !pick) return;
    const timer = setTimeout(() => {
      if (beat + 1 < pick.steps.length) setBeat(beat + 1);
      else if (loopRef.current) {
        setBeat(0);
        setTake((n) => n + 1);
      } else setPlaying(false);
    }, STEP_GAP_MS);
    return () => clearTimeout(timer);
  }, [playing, ready, beat, pick]);


  // Keyed on the scene, not done in the picker, because the address bar changes it too; a beat index
  // left over from a longer scene would point past this one's end and render nothing.
  useEffect(() => {
    setBeat(0);
    setTake((n) => n + 1);
    setPlaying(true);
  }, [pick?.id]);
  const replay = useCallback(() => {
    setBeat(0);
    setTake((n) => n + 1);
    setPlaying(true);
  }, []);
  // Stepping by hand is inspecting, so it stops the clock.
  const stepBeat = useCallback(
    (delta: number) => {
      setPlaying(false);
      setBeat((b) => Math.min(Math.max(b + delta, 0), (pick?.steps.length ?? 1) - 1));
    },
    [pick],
  );
  const hop = useCallback(
    (delta: number) => {
      const i = visible.findIndex((s) => s.id === pick?.id);
      const next = visible[Math.min(Math.max(i + delta, 0), visible.length - 1)];
      if (next && next.id !== pick?.id) onPick(next.id);
    },
    [visible, pick, onPick],
  );

  const mark = useCallback(
    (verdict: Mark | null) => {
      if (!pick) return;
      setMarks((prev) => {
        const next = { ...prev };
        const key = markKey(template, pick.id);
        if (verdict === null || next[key] === verdict) delete next[key];
        else next[key] = verdict;
        saveMarks(next);
        return next;
      });
    },
    [pick, template],
  );

  // Stands aside while a field has the caret, or every space typed would pause playback.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing =
        el?.tagName === "INPUT" || el?.tagName === "TEXTAREA" || el?.tagName === "SELECT";
      if (typing) {
        if (e.key === "Escape") el?.blur();
        return;
      }
      const hit: Record<string, () => void> = {
        " ": () => setPlaying((p) => !p),
        ArrowLeft: () => stepBeat(-1),
        ArrowRight: () => stepBeat(1),
        ArrowUp: () => hop(-1),
        ArrowDown: () => hop(1),
        r: replay,
        R: replay,
        "1": () => mark("pass"),
        "2": () => mark("fail"),
        "/": () => document.querySelector<HTMLInputElement>("[data-lab-search]")?.focus(),
      };
      const act = hit[e.key];
      if (!act) return;
      e.preventDefault();
      act();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [stepBeat, hop, replay, mark]);

  const verdict = pick ? marks[markKey(template, pick.id)] : undefined;
  const done = scenes.filter((s) => marks[markKey(template, s.id)]).length;

  // nowrap: a wrapped label makes the whole transport strip taller.
  const chip = (on: boolean) =>
    `text-[11px] whitespace-nowrap px-2 py-0.5 rounded-md border transition ${
      on
        ? "bg-indigo-600 border-indigo-500 text-white"
        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
    }`;

  return (
    <>
      {/* ---- the checklist ------------------------------------------------ */}
      <aside className="w-72 shrink-0 h-full flex flex-col border-r border-slate-800/70 bg-slate-950">
        <div className="shrink-0 p-3 pb-2 border-b border-slate-800/60">
          <div className="flex items-baseline gap-2 mb-2">
            <h2 className="text-xs font-medium text-slate-400">Scenes</h2>
            <span className="ml-auto text-[10px] tabular-nums text-slate-500">
              Checked {done}/{scenes.length}
            </span>
          </div>
          {/* Pinned above the list so it doesn't scroll away with its results. */}
          <div className="relative">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search scenes  /"
              className="w-full text-[12px] bg-slate-900 border border-slate-800 rounded-md pl-2 pr-6 py-1 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/70"
              data-lab-search
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-300 text-[11px]"
                title="Clear"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto p-3 pt-2">
          <Checklist
            scenes={visible}
            template={template}
            marks={marks}
            pickId={pick?.id ?? null}
            onPick={onPick}
            query={query}
          />
        </div>

        <div className="shrink-0 border-t border-slate-800/60 p-3 space-y-2">
          {/* Which places this map cast into the roles (see lab/places): a heading that reads
              wrong is often the casting, not the art. */}
          {places && (
            <details className="text-[10px] text-slate-600 leading-snug">
              <summary className="cursor-pointer hover:text-slate-400 transition">Places cast from this map</summary>
              <p className="mt-1">
                Hub <span className="text-slate-500">{places.pivot.name}</span>; NE{" "}
                <span className="text-slate-500">{places.ne.name}</span>, SE{" "}
                <span className="text-slate-500">{places.se.name}</span>; long trip to{" "}
                <span className="text-slate-500">{places.far.name}</span>; open{" "}
                <span className="text-slate-500">{places.open.name}</span>; tight{" "}
                <span className="text-slate-500">{places.tight.name}</span>, connected to{" "}
                <span className="text-slate-500">{places.next.name}</span>.
              </p>
            </details>
          )}
          <p className="text-[10px] text-slate-600 leading-relaxed">
            Space play/pause · ←→ beat · ↑↓ scene · R replay · 1 pass / 2 issue · / search
          </p>
        </div>
      </aside>

      {/* ---- the stage ----------------------------------------------------- */}
      <main className="flex-1 min-w-0 h-full flex flex-col">
        <div className="shrink-0 px-4 py-3 border-b border-slate-800/70 bg-slate-950/60">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-[15px] font-medium text-slate-50">{pick?.title ?? "—"}</span>
            {pick && (
              <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-800 text-slate-500">
                {pick.group}
              </span>
            )}
            <span className="ml-auto flex items-center gap-1.5">
              <button
                onClick={() => mark("pass")}
                className={chip(verdict === "pass")}
                title="Mark as passed (1)"
              >
                ✓ Pass
              </button>
              <button
                onClick={() => mark("fail")}
                className={`text-[11px] whitespace-nowrap px-2 py-0.5 rounded-md border transition ${
                  verdict === "fail"
                    ? "bg-rose-600 border-rose-500 text-white"
                    : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                }`}
                title="Mark as having an issue (2)"
              >
                ✗ Issue
              </button>
              <button
                onClick={() => setLogOpen((o) => !o)}
                className={`${chip(logOpen)} ml-1`}
                title={logOpen ? "Hide feed" : "Show feed"}
              >
    Feed
              </button>
            </span>
          </div>
          {/* What this scene asks you to judge: the tool's most important text, so never the
              smallest type. Marked by an edge rule, not a glyph, so the first word is read first. */}
          {pick && (
            <p className="mt-2.5 max-h-28 overflow-y-auto border-l-2 border-amber-500/50 pl-3 text-[13px] leading-relaxed text-amber-100/85 whitespace-pre-line">
              {pick.watch}
            </p>
          )}
        </div>

        <div className="relative flex-1 min-h-0">
          {/* Say so rather than stage every scene somewhere it doesn't mean. */}
          {!pick && (
            <div className="h-full grid place-items-center text-rose-300/80 text-sm px-8 text-center">
              The places on map "{template}" can't supply the relations the scene bench needs (a hub with
              neighbors due NE and due SE, and a pair of places connected on the graph). The map needs at
              least three places, some of whose centers line up along a single grid axis.
            </div>
          )}
          {pick && (
            // Remounted per scene, since tokens live until their agent dies and the last scene's
            // cast would linger. Not per replay: the first step re-places the cast, and a reboot
            // would reset the camera to the whole city.
            <MapStage key={`${template}:${pick.id}`} source={source} onScene={setScene} />
          )}
          <FrameReadout scene={scene} figures={figures} />
          {crash && (
            <div className="absolute bottom-2 left-2 right-2 rounded-md border border-rose-800/70 bg-rose-950/90 px-3 py-2 text-[11px] text-rose-200 font-mono">
              Render threw on beat {beat + 1}: {crash}
            </div>
          )}
        </div>

        {/* ---- the transport ---------------------------------------------- */}
        <div className="shrink-0 flex items-center flex-wrap gap-1 gap-y-1.5 px-3 py-2 border-t border-slate-800/70 bg-slate-950/50">
          <button onClick={() => stepBeat(-1)} disabled={beat <= 0} className={`${chip(false)} disabled:opacity-30`}>
            ⏮
          </button>
          <button onClick={() => setPlaying((p) => !p)} className={chip(playing)}>
            {playing ? "⏸ Pause" : "▶ Play"}
          </button>
          <button
            onClick={() => stepBeat(1)}
            disabled={!pick || beat >= pick.steps.length - 1}
            className={`${chip(false)} disabled:opacity-30`}
          >
            ⏭
          </button>
          <button onClick={replay} className={chip(false)} title="Replay from the first beat (R)">
            Replay
          </button>
          <button onClick={() => setLoop((v) => !v)} className={chip(loop)}>
            Loop
          </button>
          <Rule />

          {/* One dot per beat to jump to; scenes are short enough not to need a scrubber. */}
          {pick && (
            <div className="flex items-center gap-1 ml-2 shrink-0">
              <span className="text-[10px] whitespace-nowrap tabular-nums text-slate-500 mr-1">
                Beat {beat + 1}/{pick.steps.length}
              </span>
              {pick.steps.map((_, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setPlaying(false);
                    setBeat(i);
                    if (i === beat) setTake((n) => n + 1);
                  }}
                  title={`Beat ${i + 1}`}
                  className={`w-2 h-2 rounded-full transition ${
                    i === beat ? "bg-indigo-400 scale-125" : "bg-slate-700 hover:bg-slate-500"
                  }`}
                />
              ))}
            </div>
          )}

          <span className="ml-auto flex items-center gap-1">
            <button onClick={() => setFollow((v) => !v)} className={chip(follow)}>
              Follow
            </button>
            <button
              onClick={() => {
                setFollow(false);
                scene?.resetView();
              }}
              className={chip(false)}
            >
              Fit map
            </button>
            <Rule />
            <span className="text-[10px] whitespace-nowrap text-slate-500 mr-1">Speed</span>
            {SPEEDS.map((m) => (
              <button key={m} onClick={() => setSpeed(m)} className={chip(speed === m)}>
                {m}×
              </button>
            ))}
          </span>
        </div>
      </main>

      {/* ---- the same beat, as the feed draws it ---------------------------- */}
      {/* The map and the feed read the same record and must agree, so show both. Uses the world
          view's own component: a bespoke preview could drift and become the thing under test. */}
      {pick && logOpen && (
        <aside className="w-[380px] shrink-0 h-full overflow-y-auto border-l border-slate-800/70 bg-slate-950/60 p-3">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
            Feed · beat {beat + 1}
          </div>
          <NarrativeStep step={pick.steps[beat] ?? pick.steps[0]} />
        </aside>
      )}
    </>
  );
}
