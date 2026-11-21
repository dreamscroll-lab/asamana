// The control-bar buttons, shared by the views and panels that sit beside each other on the
// world screen, so a run control and a turn's answer look like one family.
export const BTN =
  "px-3 py-1.5 rounded-lg text-xs bg-slate-950 border border-slate-800 text-slate-300 hover:border-slate-700 hover:text-white transition disabled:opacity-40 disabled:cursor-not-allowed";
export const BTN_PRIMARY =
  "px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 border border-indigo-500 text-white hover:bg-indigo-500 transition disabled:opacity-40 disabled:cursor-not-allowed";
// The chosen one of a row of BTN options. A whole class string rather than colors stacked on BTN:
// Tailwind settles two utilities for one property by stylesheet order, not class order, so the
// added colors would lose to BTN's own.
export const BTN_SELECTED =
  "px-3 py-1.5 rounded-lg text-xs bg-indigo-950 border border-indigo-400 text-indigo-100 transition disabled:opacity-40 disabled:cursor-not-allowed";
