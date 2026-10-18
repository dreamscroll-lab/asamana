import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useCast } from "../../lib/cast";
import { api } from "../../api/client";
import type { ActorOperations, ActorOperationParameters, ActorTurn, MapOperationTarget } from "../../types";
import { BTN, BTN_PRIMARY } from "../buttonStyles";
import { targetName, reasonText } from "./actionSummary";
import { missingParameters, OPERATION_LABEL } from "./shortcuts";
import { useActorSubmission } from "./submission";
import { isSendKey } from "../SendButton";

export default function MapOperations({ worldId, turn, selection, onClose }: {
  worldId: string; turn: ActorTurn; selection: MapOperationTarget; onClose: () => void;
}) {
  const [capabilities, setCapabilities] = useState<ActorOperations | null>(null);
  const [operation, setOperation] = useState("");
  const [parameters, setParameters] = useState<ActorOperationParameters>({});
  const [description, setDescription] = useState("");
  const [error, setError] = useState("");
  const { busy, run } = useActorSubmission();
  const b = turn.briefing!;
  const { cast } = useCast(worldId);
  const self = { agent_id: turn.agent_id, name: cast[turn.agent_id]?.name ?? "自己" };
  const active = useRef(true);
  useEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  useEffect(() => {
    let current = true;
    api.actorOperations(worldId, turn.turn_id, selection.target).then((result) => {
      if (current && result.turn_id === turn.turn_id) setCapabilities(result);
    }).catch((e) => { if (current) setError(String(e)); });
    return () => { current = false; };
  }, [worldId, turn.turn_id, selection.target]);
  useEffect(() => {
    const escape = (e: globalThis.KeyboardEvent) => { if (e.key === "Escape") { e.stopImmediatePropagation(); e.preventDefault(); onClose(); } };
    window.addEventListener("keydown", escape, true);
    return () => window.removeEventListener("keydown", escape, true);
  }, [onClose]);
  const capability = capabilities?.operations.find((o) => o.operation === operation);
  const isRequired = (key: string) => capability?.required_parameters.includes(key) ?? false;
  const fieldLabel = (key: string, label: string) => isRequired(key) ? label : `${label}（可选）`;
  const confirmOnEnter = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!isSendKey(e)) return;
    e.preventDefault();
    e.currentTarget.form?.requestSubmit();
  };
  const submit = () => run(async () => {
    if (!capability?.available || missingParameters(capability, parameters, description)) return "";
    let result;
    try { result = await api.actorIntent(worldId, {
      turn_id: turn.turn_id, operation: capability.operation, target: selection.target,
      parameters, ...(description.trim() ? { description: description.trim() } : {}),
    }); } catch (error) {
      if (active.current) setError(String(error));
      return "";
    }
    if (!active.current) return "";
    if (result.status === "rejected" && result.reason === "stale_intent") { setError(reasonText(result.reason)); return ""; }
    onClose();
    return "";
  });
  const top = `clamp(8px, ${selection.y}px, calc(100% - 360px))`;
  return <div className="absolute inset-0 z-[35]"
    onPointerDown={(event) => { if (event.button === 0) onClose(); }}
    onContextMenu={(event) => event.preventDefault()}>
    <section role="dialog" aria-label="快捷操作" className="absolute w-72 max-w-[calc(100%-1rem)] overflow-auto rounded-xl border border-slate-600 bg-slate-950/95 shadow-xl p-3 text-sm space-y-3"
      style={{ left: `clamp(8px, ${selection.x}px, calc(100% - 296px))`, top, maxHeight: `calc(100% - ${top} - 8px)` }}
      onPointerDown={(e) => e.stopPropagation()}>
      <header className="flex items-center justify-between gap-2"><strong>{targetName(selection.target, b, self)}</strong><button type="button" className={BTN} aria-label="关闭" title="关闭" onClick={onClose}><span aria-hidden="true">×</span></button></header>
      {!capabilities && !error && <p className="text-slate-400">正在查看可用操作…</p>}
      {capabilities?.operations.length === 0 && <p className="text-slate-400">此刻没有可用操作。</p>}
      <div className="flex flex-wrap gap-2">{capabilities?.operations.map((o) => <button key={o.operation} className={`${BTN} ${operation === o.operation ? "border-indigo-400 text-indigo-200" : ""}`}
        disabled={busy || !o.available} title={o.available ? "" : reasonText(o.reason)}
        onClick={() => { setOperation(o.operation); setParameters({}); setDescription(""); setError(""); }}>{OPERATION_LABEL[o.operation] ?? o.operation}</button>)}</div>
      {capability && <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
        {operation === "broadcast" && <label className="block">{fieldLabel("message", "广播内容")}<textarea required={isRequired("message")} maxLength={500} className="block w-full bg-slate-900 rounded p-2" rows={2} value={parameters.message ?? ""} onChange={(e) => setParameters({ message: e.target.value })} onKeyDown={confirmOnEnter} /></label>}
        {operation !== "broadcast" && <label className="block">{fieldLabel("description", "描述")}<textarea maxLength={500} required={isRequired("description")} rows={2} className="block w-full bg-slate-900 rounded p-2" placeholder="描述希望如何行动" value={description} onChange={(e) => setDescription(e.target.value)} onKeyDown={confirmOnEnter} /></label>}
        <div className="flex justify-end"><button className={BTN_PRIMARY} disabled={busy || !capability.available || missingParameters(capability, parameters, description)}>{busy ? "处理中…" : "确认"}</button></div>
      </form>}
      {error && <p role="alert" className="text-rose-300">{error}</p>}
    </section>
  </div>;
}
