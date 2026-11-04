import type { TextareaHTMLAttributes } from "react";
import { useAutoGrow } from "../hooks/useAutoGrow";

/** `useAutoGrow` for a textarea rendered in a list, where a hook can't be called per item. */
export default function AutoGrowTextarea({ value, maxPx, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement> & {
  value: string; maxPx: number;
}) {
  const grow = useAutoGrow(value, maxPx);
  return <textarea ref={grow} value={value} {...props} />;
}
