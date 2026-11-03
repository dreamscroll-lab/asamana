import { useState } from "react";
import type { ActorInterruptAnswer, ActorTurn } from "../../types";
import { BTN } from "../buttonStyles";
import { CONSOLE_INPUT, CONSOLE_INPUT_MAX_PX, CONSOLE_WORLD_MESSAGE } from "../consoleStyles";
import { useAutoGrow } from "../../hooks/useAutoGrow";

export default function InterruptQuestion({ turn, open, busy, maxLength, onAnswer }: {
  turn: ActorTurn; open: boolean; busy: boolean; maxLength: number;
  onAnswer: (answer: ActorInterruptAnswer) => void;
}) {
  const [thought, setThought] = useState("");
  const box = useAutoGrow(thought, CONSOLE_INPUT_MAX_PX);
  const answer = (interrupt: boolean) => {
    const text = thought.trim();
    onAnswer({ interrupt, ...(text ? { thought: text } : {}) });
  };
  return <div className={`${CONSOLE_WORLD_MESSAGE} ${open ? "border-amber-800/60" : "opacity-70"} space-y-2`}>
    <p>{turn.reason && `${turn.reason}。`}{turn.doing && <>你正在：{turn.doing}。</>}是否中断当前行动？</p>
    {open && <>
      <label className="block text-xs text-slate-400">想法（可选）
        <textarea ref={box} className={`${CONSOLE_INPUT} mt-1 w-full`} rows={2} maxLength={maxLength}
          disabled={busy} value={thought} onChange={(event) => setThought(event.target.value)}
          placeholder="用你自己的口吻说明为什么中断或继续" />
      </label>
      <div className="flex gap-2">
        <button className={BTN} disabled={busy} onClick={() => answer(false)}>▶ 继续</button>
        <button className={BTN} disabled={busy} onClick={() => answer(true)}>✋ 中断</button>
      </div>
    </>}
  </div>;
}
