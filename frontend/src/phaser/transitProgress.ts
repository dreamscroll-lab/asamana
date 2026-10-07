/**
 * How far along its drawn route a mover is, as a fraction of the route's length.
 *
 * The engine places him by STEPS — he stands on `path[i]` at elapsed == arrivals[i] — and legs
 * differ in cost, so a fraction taken as elapsed/total drifts off the waypoints: the map would
 * draw him short of (or past) the room whose occupants the engine says can see him. Within a
 * leg the step fraction is spread over that leg's own pixel length. Several waypoints can share
 * a step; he stands on the furthest of them.
 */
export function routeFraction(legLengths: number[], arrivals: number[], elapsed: number): number {
  const total = legLengths.reduce((s, d) => s + d, 0);
  if (!total) return 1;
  let walked = 0;
  for (let i = 0; i < legLengths.length; i++) {
    const start = arrivals[i];
    const end = arrivals[i + 1];
    if (elapsed < end || i === legLengths.length - 1) {
      const f = end > start ? Math.max(0, Math.min(1, (elapsed - start) / (end - start))) : 1;
      return (walked + f * legLengths[i]) / total;
    }
    walked += legLengths[i];
  }
  return 1;
}
