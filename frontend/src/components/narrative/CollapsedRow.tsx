// A beat that isn't about the focused characters, folded down to one line. Collapsed rather
// than dimmed like the map: a dimmed card still costs its full height in a scrolling list.
// The beat stays on the timeline, in order, one click from being read.
export default function CollapsedRow({
  icon,
  text,
  onExpand,
}: {
  icon: string;
  text: string;
  onExpand: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onExpand}
      title="展开"
      className="group w-full flex items-center gap-1.5 px-1 py-0.5 rounded-md text-left opacity-50 hover:opacity-100 hover:bg-slate-800/30 transition"
    >
      <span className="shrink-0 text-[11px] text-slate-600">{icon}</span>
      <span className="truncate text-[12px] text-slate-500">{text}</span>
      <span className="ml-auto shrink-0 text-[10px] text-slate-700 group-hover:text-slate-500">＋</span>
    </button>
  );
}
