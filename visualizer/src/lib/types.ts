// Types matching the workspace-substrate JSONL log schema.
// These mirror the Python dataclasses in `rlm/core/types.py`.

export interface ModelUsageSummary {
  total_calls: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost?: number | null;
}

export interface UsageSummary {
  model_usage_summaries: Record<string, ModelUsageSummary>;
  total_cost?: number | null;
}

// Trajectory carried by an RLMChatCompletion.metadata when the completion
// represents a full RLM run (e.g. an `rlm_query` child). Mirrors the shape
// returned by `RLMLogger.get_trajectory()` in Python.
export interface ChildTrajectoryMetadata {
  run_metadata?: RLMConfigMetadata;
  iterations?: WorkspaceIteration[];
}

export interface RLMChatCompletion {
  root_model: string;
  prompt: string | Record<string, unknown> | unknown[];
  response: string;
  usage_summary: UsageSummary;
  execution_time: number;
  metadata?: ChildTrajectoryMetadata | null;
  reasoning_content?: string | null;
}

export type ProvenanceRole = 'user' | 'assistant' | 'system' | 'child';

export interface WorkspaceAction {
  tool: string;
  args: Record<string, unknown>;
  body: string | null;
  raw: string;
}

export interface WorkspaceObservation {
  tool: string;
  stdout: string;
  stderr: string;
  data: Record<string, unknown> | null;
  artifacts: string[];
  execution_time: number | null;
  rlm_calls: RLMChatCompletion[];
  final_answer: string | null;
  final_artifacts: string[];
  error: string | null;
}

export interface WorkspaceSnapshot {
  turn: number;
  commit_sha: string;
  changed_files: string[];
  workspace_root: string;
}

export interface ParseAttempt {
  response: string;
  error: string;
}

export interface WorkspaceIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string }>;
  response: string;
  reasoning: string | null;
  parse_attempts: ParseAttempt[];
  actions: WorkspaceAction[];
  observations: WorkspaceObservation[];
  snapshot: WorkspaceSnapshot | null;
  final_answer: string | null;
  iteration_time: number | null;
  // Set when the turn aborted before any actions were dispatched (e.g.
  // parse-retry exhaustion). When non-null, `actions` and `observations`
  // are empty and `parse_attempts` carries the failed responses.
  error?: string | null;
}

// Run-level metadata persisted in the first JSONL line.
export interface RLMConfigMetadata {
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
}

export interface LogMetadata {
  totalIterations: number;
  totalActions: number;
  totalSubLMCalls: number;
  totalParseRetries: number;
  contextQuestion: string;
  finalAnswer: string | null;
  finalArtifacts: string[];
  totalExecutionTime: number;
  hasErrors: boolean;
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: WorkspaceIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
}

// Pair an action with its corresponding observation (same index).
export interface ActionObservationPair {
  index: number;
  action: WorkspaceAction;
  observation: WorkspaceObservation | null;
}

export function pairActionsWithObservations(
  iteration: WorkspaceIteration,
): ActionObservationPair[] {
  return iteration.actions.map((action, idx) => ({
    index: idx,
    action,
    observation: iteration.observations[idx] ?? null,
  }));
}
