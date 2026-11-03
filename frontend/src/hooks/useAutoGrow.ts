/**
 * useAutoGrow — a textarea that grows with what is typed, up to `maxPx`, then scrolls inside.
 *
 * Returns a callback ref, so a textarea that mounts later (a conditional field, a reopened
 * panel) is fitted on mount without waiting for the value to change. The resting height stays
 * the element's own: `rows` and any CSS `min-height` set the floor.
 */

import { useCallback, useLayoutEffect, useRef } from "react";

export function useAutoGrow(value: string, maxPx: number): (el: HTMLTextAreaElement | null) => void {
  const box = useRef<HTMLTextAreaElement | null>(null);

  const fit = useCallback((el: HTMLTextAreaElement) => {
    // Reset to "auto" before measuring: scrollHeight reports the taller of (content, current
    // box), so measuring at the current height would only ever grow. "auto" keeps `rows` as
    // the floor. scrollHeight excludes borders, which a border-box height includes.
    const border = el.offsetHeight - el.clientHeight;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight + border, maxPx)}px`;
  }, [maxPx]);

  useLayoutEffect(() => {
    if (box.current) fit(box.current);
  }, [value, fit]);

  return useCallback((el: HTMLTextAreaElement | null) => {
    box.current = el;
    if (el) fit(el);
  }, [fit]);
}
