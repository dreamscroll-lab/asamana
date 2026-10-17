export const CONSOLE_DOCK = "absolute top-2 right-2 z-30";
export const CONSOLE_COLLAPSED = `${CONSOLE_DOCK} w-9 h-9 grid place-items-center rounded-full text-sm bg-slate-950/75 border border-slate-800/70 backdrop-blur-md shadow-lg shadow-black/40 hover:border-indigo-700/70 transition`;
export const CONSOLE_PANEL = `${CONSOLE_DOCK} w-[min(26rem,calc(100%-1rem))] h-[min(42rem,calc(100%-1rem))] flex-col rounded-2xl border border-slate-800/70 bg-slate-950/80 backdrop-blur-md shadow-xl shadow-black/40`;
export const CONSOLE_HEADER = "shrink-0 space-y-1.5 px-3 pt-2.5 pb-2 border-b border-slate-800/50";
export const CONSOLE_CLOSE = "pt-1 text-[11px] text-slate-500 hover:text-slate-200 transition";
export const CONSOLE_THREAD = "min-h-0 flex-1 overflow-y-auto px-3 py-2 space-y-2";
export const CONSOLE_FOOTER = "shrink-0 border-t border-slate-800/50 px-3 py-2 space-y-1.5";
export const CONSOLE_INPUT = "min-h-14 w-full resize-none bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-2 text-[13px] text-slate-200 focus:outline-none focus:border-indigo-500/70 disabled:opacity-50";

export const CONSOLE_RAISED = "shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_4px_10px_-4px_rgba(0,0,0,0.6)]";
export const CONSOLE_WORLD_MESSAGE = `max-w-[92%] break-words rounded-xl rounded-tl-sm border border-slate-800/70 bg-slate-900/70 px-3 py-2 text-[13px] leading-relaxed text-slate-200 ${CONSOLE_RAISED}`;
export const CONSOLE_USER_MESSAGE = `max-w-[85%] break-words rounded-xl rounded-tr-sm bg-indigo-600/35 px-3 py-2 text-[13px] leading-relaxed text-slate-100 ${CONSOLE_RAISED}`;
