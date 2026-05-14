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
  // Workspace-relative paths the model attached to its `final` action.
  // Empty when the model returned the answer inline (the recommended path).
  final_artifacts?: string[];
  // Host-absolute path to the workspace dir when `cleanup_mode == "keep"`,
  // null otherwise. Combine with `final_artifacts` entries to locate files.
  workspace_root?: string | null;
}

export type ProvenanceRole = 'user' | 'assistant' | 'system' | 'child';

export interface WorkspaceAction {
  tool: string;
  args: Record<string, unknown>;
  body: string | null;
  raw: string;
  call_id?: string | null;
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

export function observationHasError(obs: WorkspaceObservation | null | undefined): boolean {
  if (!obs) return false;
  if (obs.error != null && obs.error.length > 0) return true;

  const exitCode = obs.data?.exit_code;
  if (typeof exitCode === 'number') return exitCode !== 0;
  if (typeof exitCode === 'string' && exitCode.trim().length > 0) {
    const parsed = Number(exitCode);
    return Number.isNaN(parsed) || parsed !== 0;
  }

  return false;
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

// Substrate-level compaction event. Emitted on the turn the cumulative
// prompt crosses `CompactionConfig.threshold_tokens`; the model-authored
// `summary` replaces the pre-compress trajectory in the visible prompt.
// The full iteration history remains in the JSONL log and in workspace
// git snapshots; only the model's view is reset.
export interface CompactionEvent {
  type: 'compaction';
  timestamp: string;
  turn: number;
  tokens_before: number;
  threshold_tokens: number;
  dropped_iterations: number;
  retained_tail_iterations: number;
  summary: string;
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
  // Per-turn token usage summed across all parse-retry attempts. ``null`` for
  // backends that don't surface usage on this code path. ``completion_tokens``
  // far larger than ``reasoning + response`` length is the smoking gun for a
  // backend parser dropping tokens between the wire and the surfaced fields.
  lm_usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  // Post-chat-template prompt the model literally saw on this turn — the
  // system+tools envelope vLLM injects (tool descriptions, ``<tool_call>``
  // wrapping instructions, special tokens) plus the full message history.
  // Best-effort, populated only for self-hosted vLLM.
  rendered_prompt?: string | null;
}

// Run-level metadata persisted in the first JSONL line.
export interface RLMConfigMetadata {
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  action_format?: string | null;
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
