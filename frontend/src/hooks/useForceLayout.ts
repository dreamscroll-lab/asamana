/**
 * useForceLayout — force-directed physics simulation for a relationship graph.
 *
 * Takes nodes and edges; runs mutual repulsion + edge springs + gentle centering
 * each animation frame until kinetic energy settles. Returns positions, a zoom-to-
 * fit viewBox, the SVG ref (used internally for coordinate mapping), and pointer
 * event handlers for drag interaction.
 *
 * The component owns its own `highlighted` and `selectedEdge` state; the hook
 * fires `onTap(id)` when a node pointer-up is a click (not a drag), letting the
 * component handle the toggle and any external callback.
 */

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { GraphEdge, GraphNode } from "../types";

// Simulation geometry (viewBox px).
const W = 640;
const H = 520;

const REPULSION = 34000;
const REST = 240;
const SPRING = 0.028;
const BOUND = 90;
const CENTER = 0.0032;
const DAMPING = 0.82;
const SETTLE = 0.05;

interface Body {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface ForceLayoutResult {
  positions: Record<string, { x: number; y: number }>;
  viewBox: string;
  svgRef: React.RefObject<SVGSVGElement>;
  onNodeDown: (e: ReactPointerEvent, id: string) => void;
  onNodeMove: (e: ReactPointerEvent, id: string) => void;
  onNodeUp:   (e: ReactPointerEvent, id: string) => void;
  /** Scatter bodies to initial circle positions and re-run the sim. The component
   *  is responsible for resetting its own highlighted/selectedEdge state after. */
  resetBodies: () => void;
}

export function useForceLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  onTap: (id: string) => void,
): ForceLayoutResult {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [viewBox, setViewBox] = useState(`0 0 ${W} ${H}`);

  const svgRef    = useRef<SVGSVGElement>(null);
  const bodies    = useRef<Record<string, Body>>({});
  const edgesRef  = useRef(edges);
  edgesRef.current = edges;
  const pinned    = useRef<string | null>(null);
  const dragMoved = useRef(false);
  const rafRef    = useRef(0);
  const runningRef = useRef(false);
  const nodesRef  = useRef(nodes);
  nodesRef.current = nodes;
  const onTapRef  = useRef(onTap);
  onTapRef.current = onTap;

  const tick = useCallback(() => {
    const b = bodies.current;
    const ids = Object.keys(b);
    const cx = W / 2;
    const cy = H / 2;
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const A = b[ids[i]];
        const B = b[ids[j]];
        let dx = A.x - B.x;
        let dy = A.y - B.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) d2 = 1;
        const d = Math.sqrt(d2);
        const f = REPULSION / d2;
        const fx = (f * dx) / d;
        const fy = (f * dy) / d;
        A.vx += fx;
        A.vy += fy;
        B.vx -= fx;
        B.vy -= fy;
      }
    }
    for (const e of edgesRef.current) {
      const A = b[e.from_id];
      const B = b[e.to_id];
      if (!A || !B) continue;
      const dx = B.x - A.x;
      const dy = B.y - A.y;
      const d = Math.hypot(dx, dy) || 1;
      const f = (d - REST) * SPRING;
      const fx = (f * dx) / d;
      const fy = (f * dy) / d;
      A.vx += fx;
      A.vy += fy;
      B.vx -= fx;
      B.vy -= fy;
    }
    let energy = 0;
    for (const id of ids) {
      const P = b[id];
      if (id === pinned.current) {
        P.vx = 0;
        P.vy = 0;
        continue;
      }
      P.vx += (cx - P.x) * CENTER;
      P.vy += (cy - P.y) * CENTER;
      P.vx *= DAMPING;
      P.vy *= DAMPING;
      P.x = Math.max(-BOUND, Math.min(W + BOUND, P.x + P.vx));
      P.y = Math.max(-BOUND, Math.min(H + BOUND, P.y + P.vy));
      energy += P.vx * P.vx + P.vy * P.vy;
    }
    setPositions(Object.fromEntries(ids.map((id) => [id, { x: b[id].x, y: b[id].y }])));
    if (!pinned.current && ids.length) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const id of ids) {
        const P = b[id];
        if (P.x < minX) minX = P.x;
        if (P.x > maxX) maxX = P.x;
        if (P.y < minY) minY = P.y;
        if (P.y > maxY) maxY = P.y;
      }
      const padX = 58, padTop = 30, padBot = 44;
      const vw = Math.max(maxX - minX + padX * 2, 240);
      const vh = Math.max(maxY - minY + padTop + padBot, 240);
      setViewBox(`${(minX - padX).toFixed(0)} ${(minY - padTop).toFixed(0)} ${vw.toFixed(0)} ${vh.toFixed(0)}`);
    }
    if (energy > SETTLE || pinned.current) {
      rafRef.current = requestAnimationFrame(tick);
    } else {
      runningRef.current = false;
    }
  }, []);

  const wake = useCallback(() => {
    if (!runningRef.current) {
      runningRef.current = true;
      rafRef.current = requestAnimationFrame(tick);
    }
  }, [tick]);

  const nodeKey = nodes.map((n) => n.id).join(",");
  useEffect(() => {
    const b = bodies.current;
    const cx = W / 2;
    const cy = H / 2;
    const r = Math.min(W, H) * 0.3;
    nodesRef.current.forEach((n, i) => {
      if (!b[n.id]) {
        const a = (i * 2 * Math.PI) / Math.max(nodesRef.current.length, 1) - Math.PI / 2;
        b[n.id] = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), vx: 0, vy: 0 };
      }
    });
    for (const id of Object.keys(b)) if (!nodesRef.current.some((n) => n.id === id)) delete b[id];
    wake();
  }, [nodeKey, wake]);

  useEffect(() => { wake(); }, [edges, wake]);

  // Clear the running flag with the frame: StrictMode remounts after this cleanup, and a flag left
  // set would make every later wake() a no-op on a loop that no longer runs.
  useEffect(() => () => {
    cancelAnimationFrame(rafRef.current);
    runningRef.current = false;
  }, []);

  const toSvg = (e: { clientX: number; clientY: number }) => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return { x: 0, y: 0 };
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(m.inverse());
    return { x: p.x, y: p.y };
  };

  const onNodeDown = (e: ReactPointerEvent, id: string) => {
    e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    pinned.current = id;
    dragMoved.current = false;
    wake();
  };
  const onNodeMove = (e: ReactPointerEvent, id: string) => {
    if (pinned.current !== id) return;
    const q = toSvg(e);
    const P = bodies.current[id];
    if (!P) return;
    if (Math.hypot(q.x - P.x, q.y - P.y) > 2) dragMoved.current = true;
    P.x = Math.max(-BOUND, Math.min(W + BOUND, q.x));
    P.y = Math.max(-BOUND, Math.min(H + BOUND, q.y));
    P.vx = 0;
    P.vy = 0;
    wake();
  };
  const onNodeUp = (e: ReactPointerEvent, id: string) => {
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
    const moved = dragMoved.current;
    pinned.current = null;
    wake();
    if (!moved) onTapRef.current(id);
  };

  const resetBodies = useCallback(() => {
    bodies.current = {};
    const cx = W / 2;
    const cy = H / 2;
    const r = Math.min(W, H) * 0.3;
    nodesRef.current.forEach((n, i) => {
      const a = (i * 2 * Math.PI) / Math.max(nodesRef.current.length, 1) - Math.PI / 2;
      bodies.current[n.id] = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), vx: 0, vy: 0 };
    });
    wake();
  }, [wake]);

  return { positions, viewBox, svgRef, onNodeDown, onNodeMove, onNodeUp, resetBodies };
}
