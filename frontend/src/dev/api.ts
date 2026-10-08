// REST client for the backend's developer routes. Separate from api/client.ts for the
// same reason the types are: these routes only exist when the backend runs with
// dev_tools_enabled, and nothing on the product path may depend on them.

import { req } from "../api/client";
import type {
  AuditJudge,
  AuditReport,
  CallsPage,
  DevJob,
  DirectorPrompt,
  DirectorVerdict,
  PromptMessage,
  RecallResult,
  ReplayResult,
  StageCatalog,
  StageReport,
  StageScenario,
  StageScenarioFile,
  TraceCall,
  TraceDimensions,
} from "./types";

const post = <T,>(path: string, body: unknown) =>
  req<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) });

export type Grouping = "step" | "stage" | "agent";

export const devApi = {
  // --- Trace ---------------------------------------------------------------
  dimensions: (world: string) => req<TraceDimensions>(`/api/worlds/${world}/trace/dimensions`),
  calls: (
    world: string,
    opts: {
      groupBy: Grouping;
      step?: string;
      stage?: string;
      agentId?: string;
      callId?: string;
      q?: string;
    },
  ) => {
    const q = new URLSearchParams({ group_by: opts.groupBy });
    if (opts.step) q.set("step", opts.step);
    if (opts.stage) q.set("stage", opts.stage);
    if (opts.agentId) q.set("agent_id", opts.agentId);
    if (opts.callId) q.set("call_id", opts.callId);
    if (opts.q) q.set("q", opts.q);
    return req<CallsPage>(`/api/worlds/${world}/trace/calls?${q}`);
  },
  buildCalls: (world: string, opts: { callId?: string; q?: string } = {}) => {
    const q = new URLSearchParams();
    if (opts.callId) q.set("call_id", opts.callId);
    if (opts.q) q.set("q", opts.q);
    return req<CallsPage>(`/api/worlds/${world}/trace/build?${q}`);
  },
  call: (world: string, callId: string) =>
    req<TraceCall>(`/api/worlds/${world}/trace/calls/${encodeURIComponent(callId)}`),

  // --- Jobs (audit + stage suites share one status route) ------------------
  job: (jobId: string) => req<DevJob>(`/api/dev/jobs/${jobId}`),

  // --- Audit ---------------------------------------------------------------
  audit: (world: string) => req<AuditReport>(`/api/worlds/${world}/audit`),
  auditJudge: () => req<AuditJudge>("/api/dev/audit/judge"),
  runAudit: (
    world: string,
    body: { scopes?: string[]; agents?: string[]; steps?: string; judge_model?: string },
  ) => post<DevJob>(`/api/worlds/${world}/audit/run`, body),
  clearAudit: (world: string) =>
    req<{ world_id: string; cleared: boolean }>(`/api/worlds/${world}/audit`, {
      method: "DELETE",
    }),

  // --- Stage suites --------------------------------------------------------
  stages: () => req<StageCatalog>("/api/dev/stages"),
  stageScenarios: (stage: string) =>
    req<StageScenarioFile>(`/api/dev/stages/${stage}/scenarios`),
  stageReport: (stage: string, world: string) =>
    req<StageReport>(`/api/dev/stages/${stage}/runs/${world}`),
  // Scenario library CRUD: writes tuning/scenarios/<stage>.json itself (git-tracked corpus).
  createScenario: (stage: string, entry: StageScenario) =>
    post<{ stage: string; name: string }>(`/api/dev/stages/${stage}/scenarios`, entry),
  /** `at` = the current on-disk name; a different `entry.name` means a rename. */
  saveScenario: (stage: string, at: string, entry: StageScenario) =>
    req<{ stage: string; name: string }>(
      `/api/dev/stages/${stage}/scenarios/${encodeURIComponent(at)}`,
      { method: "PUT", body: JSON.stringify(entry) },
    ),
  deleteScenario: (stage: string, name: string) =>
    req<{ stage: string; deleted: string }>(
      `/api/dev/stages/${stage}/scenarios/${encodeURIComponent(name)}`,
      { method: "DELETE" },
    ),
  // The judge model isn't sent from the page: switching judges is calibrating the judge itself,
  // which goes through the CLI's --judge-model.
  // Empty `scenario_names` = run the full suite; otherwise run them in this order.
  runStage: (stage: string, body: { world_id: string; scenario_names?: string[] }) =>
    post<DevJob>(`/api/dev/stages/${stage}/run`, body),

  // --- Workbench -----------------------------------------------------------
  replay: (body: {
    messages: PromptMessage[];
    scene?: string | null;
    model?: string | null;
    params?: Record<string, unknown> | null;
    temperature: number;
    max_tokens: number;
    json_mode: boolean;
  }) => post<ReplayResult>("/api/llm/replay", body),
  directorPrompt: (world: string, text: string) =>
    post<DirectorPrompt>(`/api/worlds/${world}/director/prompt`, { text }),
  directorInterpret: (world: string, response: string) =>
    post<DirectorVerdict>(`/api/worlds/${world}/director/interpret`, { response }),
  recallAgents: (world: string) =>
    req<{ agents: { id: string; name: string }[] }>(`/api/worlds/${world}/memory/agents`),
  recall: (
    world: string,
    body: {
      agent_id: string;
      query: string;
      related_agent_id?: string | null;
      stream?: string | null;
      top_k: number;
      floor: number;
    },
  ) => post<RecallResult>(`/api/worlds/${world}/memory/recall`, body),
};
