import type { StepCast } from "./cast";

/** A person's name in their identity color; unnamed references use plain slate. */
export default function Who({ cast, id, name }: { cast: StepCast; id?: string; name: string }) {
  const color = cast.inkOf(id);
  return (
    <strong className={color ? "" : "text-slate-200"} style={color ? { color } : undefined}>
      {name}
    </strong>
  );
}
