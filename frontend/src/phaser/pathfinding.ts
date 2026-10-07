/** Pure grid pathfinding (no Phaser). WorldMap.adoptGround adopts the grid and delegates here. */

/** Largest connected component of walkable cells, as cell keys (y * cols + x). */
export function computeMainComponent(
  blocked: boolean[][],
  cols: number,
  rows: number,
): Set<number> {
  const seen = new Set<number>();
  let best = new Set<number>();
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const k0 = y * cols + x;
      if (blocked[y][x] || seen.has(k0)) continue;
      const comp = new Set<number>([k0]);
      const q: [number, number][] = [[x, y]];
      seen.add(k0);
      while (q.length) {
        const [cx, cy] = q.pop()!;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
          const nk = ny * cols + nx;
          if (blocked[ny][nx] || seen.has(nk)) continue;
          seen.add(nk);
          comp.add(nk);
          q.push([nx, ny]);
        }
      }
      if (comp.size > best.size) best = comp;
    }
  }
  return best;
}

/**
 * Snap a fractional grid position to the nearest tile that is on the road
 * network (main walkable component). Falls back to any walkable tile within
 * a 15-cell Manhattan ring, then to the original position.
 */
export function snap(
  gx: number,
  gy: number,
  blocked: boolean[][],
  cols: number,
  rows: number,
  mainComponent: Set<number>,
): [number, number] {
  const inB = (x: number, y: number) => x >= 0 && y >= 0 && x < cols && y < rows;
  const onRoad = (x: number, y: number) =>
    inB(x, y) && !blocked[y][x] && mainComponent.has(y * cols + x);
  if (onRoad(gx, gy)) return [gx, gy];
  let anyWalkable: [number, number] | null = null;
  for (let r = 1; r < 16; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const x = gx + dx;
        const y = gy + dy;
        if (onRoad(x, y)) return [x, y];
        if (!anyWalkable && inB(x, y) && !blocked[y][x]) anyWalkable = [x, y];
      }
    }
  }
  return anyWalkable ?? [gx, gy];
}

/**
 * Fold a 4-connected route's unit stairs into as few long axis-aligned runs as possible.
 * A* returns diagonals as staircases, and each stair flips the figure's facing: the problem
 * is too many TURNS, not too many waypoints.
 *
 * Every segment must stay parallel to a grid axis — the cast has only four headings. Don't
 * string-pull along straight lines: a diagonal segment projects to a screen angle `dirOf`
 * can't read, and the figure slides sideways while facing a diagonal. So the fold is by
 * ELBOW: from the last corner, take the furthest point reachable by one run or two runs at
 * a right angle. An L covers |dx| + |dy| cells, so route length is unchanged.
 *
 * `passable` must be the SAME test the search ran under, including cells it only AVOIDED
 * (`toll`, ground the art hides): folding across them for free would undo the detour. Where
 * no elbow is clear the stairs survive, so this never invents a route the search rejected.
 */
export function foldStairs(
  path: [number, number][],
  passable: (x: number, y: number) => boolean,
): [number, number][] {
  if (path.length <= 2) return path.slice();
  /** Every cell of an axis-aligned run, inclusive. False for a non-axial pair. */
  const runClear = (a: [number, number], b: [number, number]): boolean => {
    if (a[0] !== b[0] && a[1] !== b[1]) return false;
    if (a[0] === b[0]) {
      const [lo, hi] = a[1] < b[1] ? [a[1], b[1]] : [b[1], a[1]];
      for (let y = lo; y <= hi; y++) if (!passable(a[0], y)) return false;
      return true;
    }
    const [lo, hi] = a[0] < b[0] ? [a[0], b[0]] : [b[0], a[0]];
    for (let x = lo; x <= hi; x++) if (!passable(x, a[1])) return false;
    return true;
  };

  const out: [number, number][] = [path[0]];
  let anchor = 0;
  while (anchor < path.length - 1) {
    const a = path[anchor];
    // Furthest reachable in one run or one elbow. Falls back to the next stair, so a
    // blocked neighbourhood still advances and this cannot spin.
    let far = anchor + 1;
    let corner: [number, number] | null = null;
    for (let j = path.length - 1; j > anchor + 1; j--) {
      const b = path[j];
      if (runClear(a, b)) {
        far = j;
        corner = null;
        break;
      }
      const viaX: [number, number] = [b[0], a[1]];
      const viaY: [number, number] = [a[0], b[1]];
      if (runClear(a, viaX) && runClear(viaX, b)) {
        far = j;
        corner = viaX;
        break;
      }
      if (runClear(a, viaY) && runClear(viaY, b)) {
        far = j;
        corner = viaY;
        break;
      }
    }
    if (corner) out.push(corner);
    out.push(path[far]);
    anchor = far;
  }
  return out;
}

/**
 * A* on a 4-connected grid: start..goal inclusive, or null if unreachable.
 *
 * `toll[y][x]` is extra entry cost for ground the art hides (see isoDepth). Don't block
 * those cells instead: that shatters the road network into pieces. With a toll a route
 * detours when cheaper and crosses hidden ground only when it must.
 *
 * Tolls must be >= 0 so the Manhattan heuristic stays admissible.
 */
export function astar(
  start: [number, number],
  goal: [number, number],
  blocked: boolean[][],
  cols: number,
  rows: number,
  toll?: number[][],
): [number, number][] | null {
  const [sx, sy] = start;
  const [gx, gy] = goal;
  const inB = (x: number, y: number) => x >= 0 && y >= 0 && x < cols && y < rows;
  if (!inB(gx, gy) || blocked[gy][gx]) return null;
  const key = (x: number, y: number) => y * cols + x;
  const h = (x: number, y: number) => Math.abs(x - gx) + Math.abs(y - gy);
  const open: { f: number; g: number; x: number; y: number }[] = [{ f: h(sx, sy), g: 0, x: sx, y: sy }];
  const came = new Map<number, number>();
  const gcost = new Map<number, number>([[key(sx, sy), 0]]);
  const closed = new Set<number>();
  while (open.length) {
    let bi = 0;
    for (let i = 1; i < open.length; i++) if (open[i].f < open[bi].f) bi = i;
    const cur = open.splice(bi, 1)[0];
    const ck = key(cur.x, cur.y);
    if (cur.x === gx && cur.y === gy) {
      const path: [number, number][] = [[gx, gy]];
      let k: number | undefined = ck;
      while ((k = came.get(k)) !== undefined) path.push([k % cols, Math.floor(k / cols)]);
      return path.reverse();
    }
    if (closed.has(ck)) continue;
    closed.add(ck);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const nx = cur.x + dx;
      const ny = cur.y + dy;
      if (!inB(nx, ny) || blocked[ny][nx]) continue;
      const ng = cur.g + 1 + (toll?.[ny]?.[nx] ?? 0);
      const nk = key(nx, ny);
      if (ng < (gcost.get(nk) ?? Infinity)) {
        gcost.set(nk, ng);
        came.set(nk, ck);
        open.push({ f: ng + h(nx, ny), g: ng, x: nx, y: ny });
      }
    }
  }
  return null;
}
