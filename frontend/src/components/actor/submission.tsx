import { createContext, useContext, useRef, useState, type ReactNode } from "react";

const Context = createContext<{
  busy: boolean;
  message: string;
  run: (fn: () => Promise<string>) => Promise<void>;
} | null>(null);

export function ActorSubmission({ scope, children }: { scope: string; children: ReactNode }) {
  const current = useRef(scope);
  current.current = scope;
  const locked = useRef(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState({ scope, text: "" });
  const setMessage = (text: string) => setNotice({ scope, text });
  const message = notice.scope === scope ? notice.text : "";
  const run = async (fn: () => Promise<string>) => {
    if (locked.current) return;
    locked.current = true;
    setBusy(true);
    setMessage("");
    const started = scope;
    try {
      const result = await fn();
      if (current.current === started) setMessage(result);
    } catch (error) {
      if (current.current === started) setMessage(String(error));
    } finally {
      locked.current = false;
      setBusy(false);
    }
  };
  return <Context.Provider value={{ busy, message, run }}>{children}</Context.Provider>;
}

export function useActorSubmission() {
  const value = useContext(Context);
  if (!value) throw new Error("Actor submission provider is missing");
  return value;
}
