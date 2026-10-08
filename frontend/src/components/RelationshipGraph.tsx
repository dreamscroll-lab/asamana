// Interactive relationship graph over GET /api/worlds/{id}/graph. Nodes are laid out by a
// small force-directed simulation and are draggable (a grabbed node pins). Edges are
// directed (A→B and B→A bow apart); color encodes affection, dashes low trust. The
// backend sends no geometry; all layout is computed here.
//
// Backend scales: trust ∈ [0,1], affection ∈ [-1,1]. Displayed as 0–100 / −100..100.
import { useMemo, useState } from "react";
import type { GraphEdge, WorldGraph } from "../types";
import { THEME } from "../lib/theme";
import { useForceLayout } from "../hooks/useForceLayout";


function affectionColor(a: number): { stroke: string; marker: string } {
  if (a > 0.4) return { stroke: THEME.good, marker: "arrow-positive" }; // teal — 挚爱 (close)
  if (a < -0.3) return { stroke: THEME.bad, marker: "arrow-negative" }; // rose — 嫌隙 (rift)
  return { stroke: THEME.neutral, marker: "arrow-neutral" }; // slate — 中性 (neutral)
}

const pct = (x: number) => Math.round(x * 100);

export default function RelationshipGraph({
  graph,
  activeId,
  onNodeClick,
}: {
  graph: WorldGraph | null;
  activeId?: string | null;
  onNodeClick?: (id: string) => void;
}) {
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [highlighted, setHighlighted] = useState<string | null>(null);

  const nodes = useMemo(() => graph?.nodes ?? [], [graph]);
  const edges = useMemo(() => {
    const ids = new Set(nodes.map((n) => n.id));
    return (graph?.edges ?? []).filter(
      (e) => ids.has(e.from_id) && ids.has(e.to_id) && e.from_id !== e.to_id,
    );
  }, [graph, nodes]);

  const nameOf = useMemo(() => {
    const m = new Map(nodes.map((n) => [n.id, n.name]));
    return (id: string) => m.get(id) ?? id;
  }, [nodes]);

  // Focus for a clicked node: itself + targets of its outgoing edges. Incoming edges are
  // others' views of it, not this character's relationships.
  const focus = useMemo(() => {
    if (!highlighted) return null;
    const s = new Set<string>([highlighted]);
    for (const e of edges) if (e.from_id === highlighted) s.add(e.to_id);
    return s;
  }, [highlighted, edges]);

  const onTap = (id: string) => {
    setHighlighted((cur) => (cur === id ? null : id));
    onNodeClick?.(id);
  };
  const { positions, viewBox, svgRef, onNodeDown, onNodeMove, onNodeUp, resetBodies } =
    useForceLayout(nodes, edges, onTap);

  function relayout() {
    resetBodies();
    setHighlighted(null);
  }

  if (!nodes.length) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-8 text-center text-sm text-slate-500 h-full grid place-items-center">
        暂无关系数据。
      </div>
    );
  }

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-2xl overflow-hidden flex flex-col h-full">
      {/* Header (in flow, above the canvas): clicked-edge detail / hint + relayout */}
      <div className="flex items-start gap-2 px-3 py-2.5 border-b border-slate-800/60 bg-slate-950/40 shrink-0">
        <div className="flex-1 min-w-0">
          {selectedEdge ? (
            // Every line shares one left edge; justify-between would fling the stats to the
            // far right of a wide panel.
            <div className="w-full space-y-1.5">
              <div className="flex items-center flex-wrap gap-x-2 gap-y-1.5">
                <span className="text-xs font-bold text-slate-100 bg-indigo-950 px-2 py-0.5 rounded">
                  {nameOf(selectedEdge.from_id)}
                </span>
                <span className="text-slate-500 text-xs">→</span>
                <span className="text-xs font-bold text-slate-100 bg-purple-950 px-2 py-0.5 rounded">
                  {nameOf(selectedEdge.to_id)}
                </span>
                {selectedEdge.labels.length > 0 && (
                  <span className="text-xs font-semibold text-indigo-300">
                    “{selectedEdge.labels.join(" · ")}”
                  </span>
                )}
                {/* trust ∈ [0,1] → 0–100; affection ∈ [−1,1] → −100..100 (can be < 0). */}
                <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/40 border border-emerald-900/50">
                  <span className="text-emerald-500/80 font-medium">信任</span>
                  <span className="text-emerald-300 font-mono font-bold">{pct(selectedEdge.trust)}</span>
                </span>
                <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-rose-950/40 border border-rose-900/50">
                  <span className="text-rose-500/80 font-medium">好感</span>
                  <span className="text-rose-300 font-mono font-bold">{pct(selectedEdge.affection)}</span>
                </span>
              </div>
              {/* from_id's own account of the relation (directional: B→A has its own). */}
              {selectedEdge.history_summary && (
                <p className="text-[11px] leading-relaxed text-slate-300">
                  {selectedEdge.history_summary}
                </p>
              )}
            </div>
        ) : (
          <div className="text-xs text-slate-500 flex items-center space-x-1.5">
            <svg className="w-4 h-4 text-slate-500 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>拖拽节点自由布局 · 点击连线看档案 · 点击人物聚焦</span>
          </div>
          )}
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            relayout();
          }}
          title="重新布局"
          className="shrink-0 text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-950/90 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-600 transition"
        >
          ↺ 布局
        </button>
      </div>

      <div className="flex-1 w-full min-h-[360px]" onClick={() => setSelectedEdge(null)}>
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={viewBox}
          className="select-none touch-none"
          style={{ minHeight: 360 }}
        >
          <defs>
            {[
              ["arrow-neutral", THEME.neutral],
              ["arrow-positive", THEME.good],
              ["arrow-negative", THEME.bad],
            ].map(([id, fill]) => (
              <marker
                key={id}
                id={id}
                viewBox="0 0 10 10"
                refX="28"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" fill={fill} />
              </marker>
            ))}
          </defs>

          {/* Directed relationship arcs */}
          {edges.map((rel, i) => {
            const a = positions[rel.from_id];
            const b = positions[rel.to_id];
            if (!a || !b) return null;

            const shown = highlighted ? rel.from_id === highlighted : true; // outgoing only
            const activeTouch = activeId ? rel.from_id === activeId || rel.to_id === activeId : false;
            const isSel = selectedEdge?.from_id === rel.from_id && selectedEdge?.to_id === rel.to_id;
            const { stroke, marker } = affectionColor(rel.affection);

            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dr = Math.hypot(dx, dy) || 1;
            const path = `M${a.x},${a.y} A${dr * 1.3},${dr * 1.3} 0 0,1 ${b.x},${b.y}`;
            // Label rides the bowed side of the arc; A→B and B→A bow opposite ways
            // (direction flips the perpendicular) so a bidirectional pair separates.
            const off = 20;
            const lx = (a.x + b.x) / 2 + (dy / dr) * off;
            const ly = (a.y + b.y) / 2 - (dx / dr) * off;
            // Every label, not just the first: a relation's evolution is the labels it accrues.
            // Capped so a many-label edge can't overrun the graph.
            const joinedLabels = rel.labels.join(" · ");
            const label =
              joinedLabels.length > 16 ? joinedLabels.slice(0, 16) + "…" : joinedLabels;
            const labelW = label ? Math.max(24, label.length * 6.6 + 8) : 0;

            return (
              <g
                key={i}
                className="cursor-pointer transition-opacity duration-300"
                style={{ opacity: shown ? 1 : 0.07 }}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedEdge(rel);
                }}
              >
                <path d={path} fill="none" stroke="transparent" strokeWidth={12} />
                <path
                  d={path}
                  fill="none"
                  stroke={isSel ? THEME.accent : stroke}
                  strokeWidth={isSel ? 3 : activeTouch ? 2 : 1.2}
                  markerEnd={`url(#${marker})`}
                  strokeDasharray={rel.trust < 0.3 ? "3,3" : "none"}
                />
                {shown && label && (
                  <g transform={`translate(${lx}, ${ly})`}>
                    <rect
                      x={-labelW / 2}
                      y="-6.5"
                      width={labelW}
                      height="13"
                      rx="3"
                      fill={THEME.bg}
                      stroke={isSel ? THEME.accent : THEME.neutral800}
                      strokeWidth="0.7"
                    />
                    <text
                      textAnchor="middle"
                      dominantBaseline="central"
                      fill={isSel ? THEME.brand300 : THEME.neutral400}
                      fontSize="6.5"
                      className="font-medium pointer-events-none"
                    >
                      {label}
                    </text>
                  </g>
                )}
              </g>
            );
          })}

          {/* Character nodes — draggable */}
          {nodes.map((n) => {
            const p = positions[n.id];
            if (!p) return null;
            const dim = focus ? !focus.has(n.id) : false;
            const isActive = activeId === n.id;
            const touched = selectedEdge && (selectedEdge.from_id === n.id || selectedEdge.to_id === n.id);
            return (
              <g
                key={n.id}
                transform={`translate(${p.x}, ${p.y})`}
                className="cursor-grab active:cursor-grabbing"
                style={{ opacity: dim ? 0.3 : 1 }}
                onPointerDown={(e) => onNodeDown(e, n.id)}
                onPointerMove={(e) => onNodeMove(e, n.id)}
                onPointerUp={(e) => onNodeUp(e, n.id)}
              >
                {(isActive || touched) && (
                  <circle r="28" fill="none" stroke={isActive ? THEME.accent : THEME.accentSoft} strokeWidth="1" className="animate-ping opacity-25" />
                )}
                {/* Resting ring = the character's fixed identity color (same hue as
                    the map token + card); active/touched keep the accent highlight. */}
                <circle
                  r="20"
                  fill={THEME.bg}
                  stroke={isActive ? THEME.accent : touched ? THEME.brand400 : n.color || THEME.neutral700}
                  strokeWidth={isActive ? 3.5 : n.color ? 3 : 2}
                />
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={isActive ? THEME.brand400 : THEME.neutral200}
                  fontSize="14"
                  fontWeight="bold"
                  className="pointer-events-none"
                >
                  {n.name[0]}
                </text>
                <g transform="translate(0, 32)" className="pointer-events-none">
                  <rect
                    x="-46"
                    y="-9"
                    width="92"
                    height="18"
                    rx="5"
                    fill={THEME.panel}
                    stroke={isActive ? THEME.accentDeep : THEME.neutral800}
                    strokeWidth="1"
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    fill={n.color || (isActive ? THEME.brand300 : THEME.neutral300)}
                    fontSize="10"
                    fontWeight="600"
                  >
                    {n.name}
                    {n.is_main_character ? " ★" : ""}
                  </text>
                </g>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Legend */}
      <div className="p-3 border-t border-slate-800 bg-slate-950/40 grid grid-cols-3 gap-2 text-center text-[10px] text-slate-400">
        <div className="flex items-center justify-center space-x-1">
          <span className="w-2.5 h-0.5 bg-rose-500 rounded" />
          <span>红线 · 嫌隙 (低好感)</span>
        </div>
        <div className="flex items-center justify-center space-x-1">
          <span className="w-2.5 h-0.5 bg-slate-500 rounded" />
          <span>灰线 · 中性 · 虚线=低信任</span>
        </div>
        <div className="flex items-center justify-center space-x-1">
          <span className="w-2.5 h-0.5 bg-teal-500 rounded" />
          <span>绿线 · 挚爱 (高好感)</span>
        </div>
      </div>
    </div>
  );
}
