import { useEffect, useRef } from "react";
import Phaser from "phaser";

import { TiledWorldScene } from "../phaser/TiledWorldScene";
import type { MapSource } from "../phaser/mapSource";
import type { MapSelect } from "../types";
import { THEME } from "../lib/theme";

// The canvas renders at a fixed pixel budget in the panel's own shape, then FITs
// (CSS-downscales) into it. Sizing in CSS px renders at 1× on Hi-DPI (soft text); a fixed
// 4:3 buffer would crop the map to a 4:3 window as soon as you zoom in.
const RENDER_PIXELS = 2560 * 1920;

function renderSize(w: number, h: number): { width: number; height: number } {
  if (!w || !h) return { width: 2560, height: 1920 };
  const k = Math.sqrt(RENDER_PIXELS / (w * h));
  return { width: Math.round(w * k), height: Math.round(h * k) };
}

/** Hand Phaser the panel's box: the display size, and a render buffer of the same shape. */
function fitTo(game: Phaser.Game | null, w: number, h: number): void {
  if (!game?.isBooted) return;
  const { width, height } = renderSize(w, h);
  game.scale.setParentSize(w, h);
  game.scale.setGameSize(width, height);
}

interface Props {
  source: MapSource;
  /** Handed the live scene once it is ready, and null when it goes away. */
  onScene: (scene: TiledWorldScene | null) => void;
  onSelect?: (t: MapSelect) => void;
  /** The user grabbed the camera by hand (drag / zoom) — drop any auto-follow. */
  onGrabCamera?: () => void;
  /**
   * A value whose change means the stage's box changed in a way the ResizeObserver might
   * miss (layout mode flip, chrome collapse). Belt-and-braces on top of it.
   */
  refit?: unknown;
  /** Overlays drawn on top of the canvas (focus toolbar, readouts). */
  children?: React.ReactNode;
}

/**
 * The one place a `TiledWorldScene` is mounted, sized and torn down — shared by the
 * observation view and the scene bench. Don't fork it: the bench exists to certify what the
 * real renderer draws, so a drifted host certifies something nobody ships.
 *
 * Only boot, canvas size, refit and teardown live here; map data (`loadMapSource`), what
 * drives the steps, and the surrounding chrome stay with the consumers.
 */
export default function MapStage({
  source,
  onScene,
  onSelect,
  onGrabCamera,
  refit,
  children,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLDivElement>(null);
  const gameRef = useRef<Phaser.Game | null>(null);
  // The scene is built ONCE but must call the latest handlers, which change identity on
  // every parent render — so it calls through refs rather than being rebuilt.
  const onSceneRef = useRef(onScene);
  onSceneRef.current = onScene;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;
  const onGrabRef = useRef(onGrabCamera);
  onGrabRef.current = onGrabCamera;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    // StrictMode-safe: the ready callback is gated on `disposed`, and any canvas an async
    // boot leaves behind after teardown is swept, so the double-mount cannot strand one.
    let disposed = false;
    const scene = new TiledWorldScene(
      source,
      () => {
        if (!disposed) onSceneRef.current(scene);
      },
      (t) => onSelectRef.current?.(t),
      () => onGrabRef.current?.(),
    );
    const box = host.getBoundingClientRect();
    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent: host,
      backgroundColor: THEME.bg,
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        ...renderSize(box.width, box.height),
      },
      scene,
    });
    gameRef.current = game;
    return () => {
      disposed = true;
      gameRef.current = null;
      onSceneRef.current(null);
      game.destroy(true);
      host.replaceChildren();
    };
  }, [source]);

  // Phaser's FIT only listens to window resize, and `scale.refresh()` alone reads a stale
  // parent box — so hand Phaser the size from the observer entry. Observe the normal-flow
  // box, not the absolute canvas host: a RO there fires unreliably when the containing
  // block resizes. Deferred a frame to read settled layout.
  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    let raf = 0;
    const ro = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box?.width || !box.height) return;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => fitTo(gameRef.current, box.width, box.height));
    });
    ro.observe(area);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [source]);

  useEffect(() => {
    const area = areaRef.current;
    if (!area) return;
    const box = area.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const raf = requestAnimationFrame(() => fitTo(gameRef.current, box.width, box.height));
    return () => cancelAnimationFrame(raf);
  }, [refit]);

  return (
    <div ref={areaRef} className="relative w-full h-full overflow-hidden">
      <div ref={hostRef} className="absolute inset-0" />
      {children}
    </div>
  );
}
