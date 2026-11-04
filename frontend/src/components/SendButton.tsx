import type { KeyboardEvent } from "react";

/** Enter sends, Shift+Enter breaks the line; never while an IME is composing. */
export function isSendKey(e: KeyboardEvent<HTMLTextAreaElement>): boolean {
  return e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing;
}

export const SEND_HINT = "↵ 发送 · ⇧↵ 换行";

/** A paper plane, replaced by a spinner while the words are on their way. */
export default function SendButton({
  disabled,
  busy,
  onClick,
  title = "发送（↵）",
}: {
  disabled: boolean;
  busy: boolean;
  onClick: () => void;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      title={title}
      aria-label="发送"
      className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-indigo-600 text-white shadow-[0_4px_12px_-4px_rgba(79,70,229,0.7)] transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {busy ? (
        <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
      ) : (
        <svg className="h-4 w-4 translate-x-px" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 2 11 13" />
          <path d="M22 2 15 22l-4-9-9-4 20-7Z" />
        </svg>
      )}
    </button>
  );
}
