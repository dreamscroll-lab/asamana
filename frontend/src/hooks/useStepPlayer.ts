/**
 * useStepPlayer — the completion-gated playback state machine.
 *
 * Owns:
 *   - live buffer (liveOrdered, liveCursor) + the narrative feed (feed)
 *   - playback step list + index + fetch cache (steps, pbIndex, pbStep, stepCache)
 *   - the render-gating flag (rendering) that keeps live/playback advancing one step
 *     at a time, waiting for the map to signal completion before the next step loads
 *   - relationship graph fetch + cache (graph, graphCache)
 *
 * The caller (WorldView) drives the WS connection and calls onLiveStep() for each
 * incoming step event, and calls doneRendering() when the map finishes a step render.
 * WorldView keeps mode, panel toggles, and all UI controls (run/pause/stop etc.);
 * it passes graphActive so the graph fetch runs only while the graph overlay is open.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { StepEvent, WorldGraph } from "../types";

const FEED_CAP = 60;

export interface StepPlayerControls {
  /** The step to display in the current mode (null = nothing yet). */
  displayStep: StepEvent | null;
  /** Newest-first narrative feed: the story up to the current step, in BOTH modes. */
  feed: StepEvent[];
  /** Ordered list of available step numbers for playback. */
  steps: number[];
  /** Current playback cursor (index into steps[]). */
  pbIndex: number;
  playing: boolean;
  /** Latest relationship graph for the displayed step. */
  graph: WorldGraph | null;
  /** Call from the WS onFrame handler for each incoming step/snapshot event. */
  onLiveStep: (ev: StepEvent) => void;
  /** Call when the map finishes rendering a step (clears the render gate). */
  doneRendering: () => void;
  goTo: (index: number) => void;
  seek: (delta: number) => void;
  togglePlay: () => void;
}

export function useStepPlayer(
  id: string,
  mode: "live" | "playback",
  graphActive: boolean,
): StepPlayerControls {
  // --- live buffer ---
  const [liveOrdered, setLiveOrdered] = useState<StepEvent[]>([]);
  const [liveCursor, setLiveCursor] = useState(-1);
  const [liveFeed, setLiveFeed] = useState<StepEvent[]>([]);

  // --- playback ---
  const [steps, setSteps] = useState<number[]>([]);
  const [pbIndex, setPbIndex] = useState(0);
  const [pbStep, setPbStep] = useState<StepEvent | null>(null);
  const [playing, setPlaying] = useState(false);
  const [rendering, setRendering] = useState(false);
  const stepCache = useRef(new Map<number, StepEvent>());
  // Every step this playback session has shown, newest-first. Don't cap the store: the cap
  // belongs on the view, after the cursor filter. Capped here, scrubbing back past the kept
  // window would filter out every survivor and leave the feed empty.
  const [pbSeen, setPbSeen] = useState<StepEvent[]>([]);

  // --- graph ---
  const [graph, setGraph] = useState<WorldGraph | null>(null);
  const graphCache = useRef(new Map<number, WorldGraph>());

  // --- derived ---
  const displayStep = mode === "live" ? liveOrdered[liveCursor] ?? null : pbStep;

  // The narrative feed says the same thing in both modes: THE STORY SO FAR, newest-first.
  // Live grows as steps arrive; playback grows as the scrubber advances and recedes when
  // you scrub back, so the story never runs ahead of the map beside it.
  //
  // Playback does NOT backfill unwatched steps: a step is a ~40KB snapshot, so a 60-step
  // window would be megabytes of map payload the feed never reads, and the feed would mean
  // "all history" here but "what I've seen" in live. A jump-seek leaves gaps; those steps
  // were skipped, not hidden.
  const pbCurrent = steps[pbIndex];
  const feed = useMemo(() => {
    if (mode === "live") return liveFeed;
    if (pbCurrent == null) return [];
    return pbSeen.filter((e) => e.step <= pbCurrent).slice(0, FEED_CAP);
  }, [mode, liveFeed, pbSeen, pbCurrent]);

  // Record a played step into the feed store (deduped, kept newest-first).
  const rememberPbStep = (ev: StepEvent) =>
    setPbSeen((prev) =>
      prev.some((e) => e.step === ev.step) ? prev : [ev, ...prev].sort((a, b) => b.step - a.step),
    );

  const onLiveStep = (ev: StepEvent) => {
    setLiveOrdered((prev) => (prev.some((e) => e.step === ev.step) ? prev : [...prev, ev]));
    setLiveFeed((prev) => [ev, ...prev.filter((e) => e.step !== ev.step)].slice(0, FEED_CAP));
  };

  const doneRendering = () => setRendering(false);

  // Load playback step list when switching to playback mode.
  useEffect(() => {
    if (mode !== "playback") {
      setPlaying(false);
      return;
    }
    let live = true;
    (async () => {
      try {
        const s = await api.listSteps(id);
        if (!live) return;
        setSteps(s);
        setPbIndex(Math.max(s.length - 1, 0));
      } catch (_e) {
        // error surfaces to WorldView via the existing error state
      }
    })();
    return () => { live = false; };
  }, [mode, id]);

  // Fetch the selected step (cached) whenever the cursor or step list changes.
  useEffect(() => {
    if (mode !== "playback" || !steps.length) return;
    const step = steps[pbIndex];
    const cached = stepCache.current.get(step);
    if (cached) { setPbStep(cached); rememberPbStep(cached); return; }
    let live = true;
    api
      .getStep(id, step)
      .then((ev) => {
        stepCache.current.set(step, ev);
        if (live) { setPbStep(ev); rememberPbStep(ev); }
      })
      .catch(() => {
        // Check `live` here too, not just in .then: a failed fetch for a superseded step must not
        // reopen the gate the current step just closed. That would advance playback before the
        // current step finishes drawing, leaving the map on the old step and the scrubber on the new.
        if (live) setRendering(false); // don't wedge the gate on a fetch failure
      });
    return () => { live = false; };
  }, [pbIndex, steps, mode, id]);

  // Reset the render gate when switching modes so neither side is wedged.
  useEffect(() => {
    setRendering(false);
  }, [mode]);

  // Live advancement: completion-gated — step to the next only once the current
  // step's render timeline has fully finished (rendering=false).
  useEffect(() => {
    if (mode !== "live" || rendering) return;
    if (liveCursor < liveOrdered.length - 1) {
      setRendering(true);
      setLiveCursor((c) => c + 1);
    }
  }, [mode, rendering, liveCursor, liveOrdered.length]);

  // Playback: advance only while playing and the gate is open.
  useEffect(() => {
    if (mode !== "playback" || !playing || rendering) return;
    if (pbIndex >= steps.length - 1) { setPlaying(false); return; }
    setRendering(true);
    setPbIndex((i) => i + 1);
  }, [mode, playing, rendering, pbIndex, steps.length]);

  // Relationship graph for the displayed step (cached), fetched only when the
  // graph overlay is summoned.
  const displayStepNo = displayStep?.step;
  useEffect(() => {
    if (!graphActive || displayStepNo == null) return;
    const cached = graphCache.current.get(displayStepNo);
    if (cached) { setGraph(cached); return; }
    let live = true;
    api
      .graph(id, displayStepNo)
      .then((g) => {
        graphCache.current.set(displayStepNo, g);
        if (live) setGraph(g);
      })
      .catch(() => {});
    return () => { live = false; };
  }, [graphActive, id, displayStepNo]);

  // --- playback controls ---
  // Closing the gate promises a render will reopen it. A seek that lands on the current
  // index (e.g. ⏮ at index 0, after clamping) re-renders nothing, so doneRendering never
  // fires and playback wedges for good. A seek that moves nowhere is a no-op, gate included.
  const goTo = (index: number) => {
    const clamped = Math.min(Math.max(index, 0), Math.max(steps.length - 1, 0));
    if (clamped === pbIndex) return;
    setRendering(true);
    setPbIndex(clamped);
  };
  const seek = (delta: number) => goTo(pbIndex + delta);
  const togglePlay = () => {
    if (!playing && pbIndex >= steps.length - 1) goTo(0);
    setPlaying((p) => !p);
  };

  return {
    displayStep,
    feed,
    steps,
    pbIndex,
    playing,
    graph,
    onLiveStep,
    doneRendering,
    goTo,
    seek,
    togglePlay,
  };
}
