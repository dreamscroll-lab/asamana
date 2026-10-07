// Wire types for the backend's developer routes (interaction/api/dev/). They live
// here rather than in ../types.ts because those routes exist only when the backend
// runs with dev_tools_enabled — the split mirrors the backend's own gating.

import type { WorldTimeView } from "../types";

export interface CallAggregate {
  calls: number;
  failures: number;
  failure_rate: number;
  parse_attempts: number;
  parse_failures: number;
  parse_failure_rate: number;
  adoption_verdicts: number;
  discarded: number;
  discard_rate: number;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  wall_ms?: number | null;
}

export interface PromptMessage {
  role: string;
  content: string;
}

/** One LLM call as recorded at the router boundary (core.interfaces.trace.LLMCallTrace). */
export interface TraceCall {
  call_id: string;
  world_id: string;
  stage: string;
  scene: string;
  model: string;
  step: number | null;
  agent_id: string | null;
  agent_name: string | null;
  prompt_messages: PromptMessage[];
  response_content: string;
  thinking: string;
  thinking_tokens: number;
  temperature: number;
  max_tokens: number;
  json_mode: boolean;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  ok: boolean;
  error: string;
  parse_ok: boolean | null;
  adopted: boolean | null;
  reject_reason: string;
  extra: Record<string, unknown>;
}

export interface TraceDimensions {
  steps: (CallAggregate & { step: number; world_time: WorldTimeView | null; wall_ms: number | null })[];
  stages: string[];
  agents: { id: string; name: string }[];
  has_build: boolean;
  totals: CallAggregate;
}

export interface CallGroup extends CallAggregate {
  key: string;
  label: string;
  items: TraceCall[];
}

export interface CallsPage {
  group_by: string;
  total: CallAggregate;
  groups: CallGroup[];
}

/** A backgrounded `python -m tuning …` subprocess. */
export interface DevJob {
  job_id: string;
  kind: string;
  world_id: string;
  cmd: string;
  status: "running" | "completed" | "failed";
  returncode: number | null;
  log_tail: string[];
}

// --- Audit -----------------------------------------------------------------

export interface AuditMetric {
  id: string;
  name: string;
  weight: number;
  category: "basic" | "elevated" | string;
  inspect: string;
}

export interface AuditScopeMeta {
  label: string;
  unit: "none" | "stage" | "step" | string;
  metrics: AuditMetric[];
}

export interface AuditEntry {
  total: number | null;
  rationale: string;
  /** Metric keys the judge omitted entirely. The prompt requires null instead, so these are defects. */
  dropped?: string[];
  // unit=none → {metricId: score}; otherwise {unitLabel: {metricId: score}}
  scores: Record<string, number | null> | Record<string, Record<string, number | null>>;
  name?: string;
  agent?: string;
  step?: number;
  det?: { detail: string }[];
}

export interface AuditDimensionCard {
  name: string;
  score: number | null;
  weight: number;
  /** weight × coverage — what this dimension actually pulls in the world score. */
  effective_weight: number;
  category: string;
  inspect: string;
  /** Share of this dimension's evidence weight that was actually audited (0-1). */
  coverage: number;
  /** Which scopes supplied the evidence, and how much each one counted after normalisation. */
  sources: Record<
    string,
    { score: number; samples: number; weight: number; share: number; stale: boolean }
  >;
  /** Ran under the current metric set and still scored nothing here — the judge abstained. */
  abstained: string[];
  /** Ran under an older metric set, so this dimension was never put to it. Re-run the scope. */
  stale: string[];
}

export interface AuditSummary {
  world_score: number | null;
  /** Weighted evidence coverage (0-1). Two audits are only comparable at equal coverage. */
  coverage: number | null;
  /** sass fidelity — engineering quality, deliberately not part of world_score. */
  fidelity_score: number | null;
  judge_model: string;
  generated_at: string;
  scope_totals: Record<string, number | null>;
  scope_provenance: Record<
    string,
    { judge_model: string; digest: string; generated_at: string }
  >;
  /** Scopes whose stored scores were produced under a different metric set than the current one. */
  stale_scopes: string[];
  /** True when the scopes feeding world_score were judged by different models. */
  mixed_provenance: boolean;
  scopes_meta: Record<string, AuditScopeMeta>;
  dimensions: Record<string, AuditDimensionCard>;
}

/** Which endpoint this deployment judges with. The provider is a deployment fact
 *  (its declaration carries the base URL and the key), so it is shown, not chosen. */
export interface AuditJudge {
  provider: string;
  model: string;
}

export interface AuditReport {
  summary: AuditSummary | null;
  initialization: AuditEntry | null;
  single_agent_single_step: Record<string, AuditEntry> | null;
  single_agent_multi_step: Record<string, AuditEntry> | null;
  multi_agent_single_step: Record<string, AuditEntry> | null;
  multi_agent_multi_step: AuditEntry | null;
  calls: Record<string, Record<string, { prompt: PromptMessage[]; response: string }>> | null;
}

// --- Stage suites ----------------------------------------------------------

export interface StageSpec {
  key: string;
  label: string;
  criteria: string[];
  judged: boolean;
  scenarios: string[];
}

export interface StageCatalog {
  stages: StageSpec[];
  /** stage key → the worlds that already have a report on disk. */
  runs: Record<string, string[]>;
}

export interface StageScenarioRow {
  name: string;
  criteria_focus?: string[];
  scores: Record<string, number>;
  overall?: string;
  deterministic_passed: boolean;
  issues: string[];
}

/** One scenario entry. `scenario` keys vary by stage, so typing stops at unknown: the page only
 *  shows and edits its text, it doesn't interpret it (phase_harness does). */
export interface StageScenario {
  name: string;
  description?: string;
  criteria_focus?: string[];
  expect?: string;
  scenario?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface StageScenarioFile {
  stage: string;
  path: string;
  /** The scenario file as-is: `scenarios` plus non-scenario keys like `_comment` / `knob_sets`. */
  file: { scenarios: StageScenario[] } & Record<string, unknown>;
}

export interface StageReport {
  stage: string;
  world_id: string;
  summary: {
    world_id: string;
    generated_at: string;
    scenario_count: number;
    criteria_avg: Record<string, number>;
    all_deterministic_passed?: boolean;
    scenarios?: StageScenarioRow[];
  } | null;
  /** scenario name → artifact file stem → parsed JSON (llm_calls / prompt / checks / judge / …).
   *  `llm_calls` is every LLM call of the run, shaped as TraceCall. */
  scenarios: Record<string, Record<string, unknown>>;
}

// --- Workbench instruments -------------------------------------------------

export interface ReplayResult {
  content: string;
  input_tokens: number;
  output_tokens: number;
  model: string;
  provider_spec: string;
  latency_ms: number;
  params: Record<string, unknown>;
}

export interface DirectorPrompt {
  messages: PromptMessage[];
  scene: string;
  temperature: number;
  max_tokens: number;
  json_mode: boolean;
  menus: { cast?: Record<string, string>; location?: Record<string, string>; entity?: Record<string, string> };
  step: number;
  world_time: WorldTimeView;
}

export interface DirectorVerdict {
  accepted: boolean;
  reason: string;
  preview: string;
  plan: {
    channels?: string[];
    broadcast?: { location: string; severity: string; phenomenon: string; content: string };
    message?: { recipients?: string[]; urgency: string; content: string };
    mutations?: { kind: string; target: string; detail: string; observation: string }[];
  } | null;
}

export interface RecallCandidate {
  id: string;
  stream: string;
  kind: string;
  content: string;
  created_step: number;
  importance_raw: number;
  decay: number;
  related_agents: { id: string; name: string }[];
  dense: number;
  sparse: number;
  fused: number;
  relevance_n: number;
  recency_n: number;
  importance_n: number;
  score: number;
  dropped: string | null;
  selected: boolean;
}

export interface RecallResult {
  query: string;
  agent_id: string;
  agent_name: string;
  stream: string;
  related_agent_id: string | null;
  current_step: number;
  floor: number;
  top_k: number;
  candidates: RecallCandidate[];
  stats: {
    candidates: number;
    selected: number;
    dropped_by_floor: number;
    dropped_by_mmr: number;
    dropped_by_top_k: number;
    dense_min: number | null;
    dense_max: number | null;
  };
}
