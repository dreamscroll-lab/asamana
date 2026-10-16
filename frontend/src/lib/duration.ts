/**
 * A world duration in the units a reader thinks in: a full day reads as "约 1 天", not
 * "24 小时". Mirrors the backend's core.duration.describe_duration: the wire carries seconds,
 * and this is the one place that turns them into words.
 */
export function durationLabel(seconds: number): string {
  const total = Math.floor(seconds / 60);
  const parts: [number, string][] = [[Math.floor(total / 1440), "天"], [Math.floor((total % 1440) / 60), "小时"], [total % 60, "分钟"]];
  return `约 ${parts.filter(([n]) => n).map(([n, unit]) => `${n} ${unit}`).join(" ") || "0 分钟"}`;
}
