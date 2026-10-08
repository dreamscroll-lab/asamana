import type { StepCast } from "./cast";

/**
 * A person's name in their identity color, the same one they wear everywhere else. Falls back
 * to plain slate, not another hue: a color that isn't theirs would read as an identity.
 */
export default function Who({ cast, id, name }: { cast: StepCast; id?: string; name: string }) {
  const color = cast.inkOf(id);
  return (
    <strong className={color ? "" : "text-slate-200"} style={color ? { color } : undefined}>
      {name}
    </strong>
  );
}
