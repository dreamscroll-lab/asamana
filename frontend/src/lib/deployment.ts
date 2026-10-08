import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { Deployment } from "../types";

// Fixed for the backend's lifetime, so asked once per page load and shared. A failed ask
// reads as a deployment offering nothing and is asked again until it succeeds: nginx serves the
// page before the backend is up, so the first ask after `asamana.sh start` usually fails.
let asked: Promise<Deployment> | null = null;
const UNREACHABLE: Deployment = { dev_tools: false, model_keys: false };
const RETRY_MS = 3000;

/** What the backend deployment offers. null = not known yet. */
export function useDeployment(): Deployment | null {
  const [deployment, setDeployment] = useState<Deployment | null>(null);
  useEffect(() => {
    let live = true;
    let retry: ReturnType<typeof setTimeout> | undefined;
    const ask = () => {
      asked ??= api.getDeployment().catch(() => {
        asked = null;
        return UNREACHABLE;
      });
      asked.then((d) => {
        if (!live) return;
        setDeployment(d);
        if (d === UNREACHABLE) retry = setTimeout(ask, RETRY_MS);
      });
    };
    ask();
    return () => {
      live = false;
      clearTimeout(retry);
    };
  }, []);
  return deployment;
}

/** Whether the backend mounted its developer routes. null = not known yet. */
export function useDevTools(): boolean | null {
  const d = useDeployment();
  return d === null ? null : d.dev_tools;
}

/** Shown on a control disabled because the deployment has no model keys. */
export const NO_MODEL_KEYS_HINT = "Required API keys are not configured";

/**
 * Whether model-calling controls are usable. Unknown counts as unusable, so a button never
 * enables for a moment and then grays out.
 */
export function useModelKeys(): boolean {
  return useDeployment()?.model_keys ?? false;
}
