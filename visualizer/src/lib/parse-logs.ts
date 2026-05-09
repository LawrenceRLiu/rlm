import {
  WorkspaceIteration,
  WorkspaceObservation,
  RLMLogFile,
  LogMetadata,
  RLMConfigMetadata,
} from './types';

function getDefaultConfig(): RLMConfigMetadata {
  return {
    root_model: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
  };
}

export interface ParsedJSONL {
  iterations: WorkspaceIteration[];
  config: RLMConfigMetadata;
}

export function parseJSONL(content: string): ParsedJSONL {
  const lines = content
    .trim()
    .split('\n')
    .filter((line) => line.trim());

  const iterations: WorkspaceIteration[] = [];
  let config: RLMConfigMetadata = getDefaultConfig();

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);

      if (parsed.type === 'metadata') {
        config = {
          root_model: parsed.root_model ?? null,
          max_depth: parsed.max_depth ?? null,
          max_iterations: parsed.max_iterations ?? null,
          backend: parsed.backend ?? null,
          backend_kwargs: parsed.backend_kwargs ?? null,
          environment_type: parsed.environment_type ?? null,
          environment_kwargs: parsed.environment_kwargs ?? null,
          other_backends: parsed.other_backends ?? null,
        };
        continue;
      }

      // Iteration entry — coerce missing fields to safe defaults so
      // partially-written logs don't blow up the UI.
      const iter: WorkspaceIteration = {
        type: parsed.type,
        iteration: parsed.iteration ?? 0,
        timestamp: parsed.timestamp ?? '',
        prompt: parsed.prompt ?? [],
        response: parsed.response ?? '',
        reasoning: parsed.reasoning ?? null,
        parse_attempts: parsed.parse_attempts ?? [],
        actions: parsed.actions ?? [],
        observations: parsed.observations ?? [],
        snapshot: parsed.snapshot ?? null,
        final_answer: parsed.final_answer ?? null,
        iteration_time: parsed.iteration_time ?? null,
      };
      iterations.push(iter);
    } catch (e) {
      console.error('Failed to parse line:', line, e);
    }
  }

  return { iterations, config };
}

export function extractContextQuestion(iterations: WorkspaceIteration[]): string {
  if (iterations.length === 0) return 'No context found';

  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt ?? [];

  // The user message in the workspace substrate is the root task text,
  // typically a short prompt seeded into _rlm_query_0.txt.
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      const trimmed = msg.content.trim();
      if (trimmed.length === 0) continue;
      if (trimmed.length <= 240) return trimmed;
      return trimmed.slice(0, 200) + '...';
    }
  }

  return 'Workspace substrate run';
}

// Try to surface a single-line preview of the root task. Used by the
// dashboard list. Falls back to the iteration count when nothing usable
// is in the prompt.
export function extractContextPreview(iterations: WorkspaceIteration[]): string | null {
  const q = extractContextQuestion(iterations);
  if (!q || q === 'No context found' || q === 'Workspace substrate run') {
    return null;
  }
  const oneLine = q.replace(/\s+/g, ' ').trim();
  return oneLine.length > 120 ? oneLine.slice(0, 120) + '...' : oneLine;
}

function observationHasError(obs: WorkspaceObservation): boolean {
  if (obs.error) return true;
  if (obs.stderr && obs.stderr.length > 0) return true;
  return false;
}

export function computeMetadata(iterations: WorkspaceIteration[]): LogMetadata {
  let totalActions = 0;
  let totalSubLMCalls = 0;
  let totalParseRetries = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;
  let finalArtifacts: string[] = [];

  for (const iter of iterations) {
    totalActions += iter.actions.length;
    totalParseRetries += iter.parse_attempts.length;

    if (iter.iteration_time != null) {
      totalExecutionTime += iter.iteration_time;
    }

    for (const obs of iter.observations) {
      if (observationHasError(obs)) hasErrors = true;
      if (obs.rlm_calls) totalSubLMCalls += obs.rlm_calls.length;
    }

    if (iter.final_answer) {
      finalAnswer = iter.final_answer;
      // Pull final_artifacts from whichever observation produced the answer.
      for (const obs of iter.observations) {
        if (obs.final_answer && obs.final_artifacts && obs.final_artifacts.length > 0) {
          finalArtifacts = obs.final_artifacts;
          break;
        }
      }
    }
  }

  return {
    totalIterations: iterations.length,
    totalActions,
    totalSubLMCalls,
    totalParseRetries,
    contextQuestion: extractContextQuestion(iterations),
    finalAnswer,
    finalArtifacts,
    totalExecutionTime,
    hasErrors,
  };
}

export function parseLogFile(fileName: string, content: string): RLMLogFile {
  const { iterations, config } = parseJSONL(content);
  const metadata = computeMetadata(iterations);

  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
  };
}
