'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { ActionObservationPair } from '@/lib/types';
import { CodeWithLineNumbers } from './CodeWithLineNumbers';
import { useDrillDepth, useDrillStack } from './DrillStack';

interface ActionCardProps {
  pair: ActionObservationPair;
}

// Tools that primarily carry executable code in their body. We render the
// body with python-style syntax highlighting for these; everything else
// gets plain text rendering.
const CODE_TOOLS = new Set(['python', 'shell']);

function toolStyle(tool: string): { border: string; bg: string; accent: string; label: string } {
  switch (tool) {
    case 'shell':
    case 'python':
      return {
        border: 'border-emerald-500/30 dark:border-emerald-400/30',
        bg: 'bg-emerald-500/5 dark:bg-emerald-400/5',
        accent: 'text-emerald-600 dark:text-emerald-400',
        label: tool,
      };
    case 'write_file':
    case 'append_file':
    case 'edit_file':
      return {
        border: 'border-amber-500/30 dark:border-amber-400/30',
        bg: 'bg-amber-500/5 dark:bg-amber-400/5',
        accent: 'text-amber-600 dark:text-amber-400',
        label: tool,
      };
    case 'read_file':
    case 'list_directory':
      return {
        border: 'border-sky-500/30 dark:border-sky-400/30',
        bg: 'bg-sky-500/5 dark:bg-sky-400/5',
        accent: 'text-sky-600 dark:text-sky-400',
        label: tool,
      };
    case 'llm_query':
    case 'rlm_query':
      return {
        border: 'border-fuchsia-500/30 dark:border-fuchsia-400/30',
        bg: 'bg-fuchsia-500/5 dark:bg-fuchsia-400/5',
        accent: 'text-fuchsia-600 dark:text-fuchsia-400',
        label: tool,
      };
    case 'final':
      return {
        border: 'border-emerald-500/40 dark:border-emerald-400/40',
        bg: 'bg-emerald-500/10 dark:bg-emerald-400/10',
        accent: 'text-emerald-600 dark:text-emerald-400',
        label: tool,
      };
    default:
      return {
        border: 'border-border',
        bg: 'bg-muted/30',
        accent: 'text-muted-foreground',
        label: tool,
      };
  }
}

function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return '';
  return entries.map(([k, v]) => `${k}="${String(v)}"`).join(' ');
}

export function ActionCard({ pair }: ActionCardProps) {
  const [isOpen, setIsOpen] = useState(true);
  const { action, observation, index } = pair;
  const drillStack = useDrillStack();
  // The depth this card is rendered at: 0 for the root LogViewer, N for
  // an action inside a frame at depth N. When the user clicks a drill
  // button, `push(frame, drillDepth)` truncates the stack to the
  // current depth before adding the new leaf — i.e. opening a sibling
  // sub-call at the same depth automatically collapses any deeper view
  // that was open from the previous sibling. One path at a time.
  const drillDepth = useDrillDepth();
  const style = toolStyle(action.tool);

  const hasError = !!observation?.error || !!observation?.stderr;
  const hasStdout = !!observation?.stdout && observation.stdout.length > 0;
  const subCallCount = observation?.rlm_calls?.length ?? 0;
  const artifactCount = observation?.artifacts?.length ?? 0;
  const execTime = observation?.execution_time?.toFixed(2);
  const argString = formatArgs(action.args);

  const language = CODE_TOOLS.has(action.tool) ? 'python' : 'text';

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card
        className={cn(
          'border overflow-hidden transition-all',
          hasError ? 'border-red-500/40 bg-red-500/5 dark:border-red-400/40 dark:bg-red-400/5' : `${style.border} ${style.bg}`,
        )}
      >
        <CollapsibleTrigger asChild>
          <CardHeader className="py-2 px-4 cursor-pointer hover:bg-muted/30 transition-colors">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <span className={cn('font-mono text-sm', style.accent)}>#{index + 1}</span>
                <CardTitle className="text-sm font-medium">
                  <span className={style.accent}>&lt;{style.label}&gt;</span>
                  {argString && (
                    <span className="text-muted-foreground ml-2 text-xs font-mono">
                      {argString}
                    </span>
                  )}
                </CardTitle>
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {execTime && (
                  <Badge variant="outline" className="font-mono text-xs">
                    {execTime}s
                  </Badge>
                )}
                {hasError && (
                  <Badge variant="destructive" className="text-xs">
                    Error
                  </Badge>
                )}
                {!hasError && hasStdout && (
                  <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-xs">
                    Output
                  </Badge>
                )}
                {subCallCount > 0 && (
                  <Badge className="bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30 text-xs">
                    {subCallCount} sub-LM
                  </Badge>
                )}
                {artifactCount > 0 && (
                  <Badge variant="outline" className="text-xs">
                    {artifactCount} artifact{artifactCount !== 1 ? 's' : ''}
                  </Badge>
                )}
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <span className="text-xs">{isOpen ? '▼' : '▶'}</span>
                </Button>
              </div>
            </div>
          </CardHeader>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="p-0">
            {/* Action body */}
            {action.body !== null && action.body.length > 0 && (
              <div className="bg-muted border-t border-border">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Action body
                  </span>
                </div>
                <div className="code-block p-4 overflow-x-auto">
                  <CodeWithLineNumbers code={action.body} language={language} />
                </div>
              </div>
            )}

            {/* Observation: stdout */}
            {hasStdout && (
              <div className="border-t border-border bg-emerald-500/5 dark:bg-emerald-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-medium">
                    stdout
                  </span>
                </div>
                <pre className="code-block p-4 overflow-x-auto whitespace-pre-wrap">
                  <code className="text-emerald-700 dark:text-emerald-300 text-xs">
                    {observation!.stdout}
                  </code>
                </pre>
              </div>
            )}

            {/* Observation: stderr */}
            {observation?.stderr && observation.stderr.length > 0 && (
              <div className="border-t border-border bg-red-500/5 dark:bg-red-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-red-600 dark:text-red-400 font-medium">
                    stderr
                  </span>
                </div>
                <pre className="code-block p-4 overflow-x-auto whitespace-pre-wrap">
                  <code className="text-red-700 dark:text-red-300 text-xs">
                    {observation.stderr}
                  </code>
                </pre>
              </div>
            )}

            {/* Observation: error string */}
            {observation?.error && (
              <div className="border-t border-border bg-red-500/5 dark:bg-red-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-red-600 dark:text-red-400 font-medium">
                    error
                  </span>
                </div>
                <pre className="code-block p-4 overflow-x-auto whitespace-pre-wrap">
                  <code className="text-red-700 dark:text-red-300 text-xs">
                    {observation.error}
                  </code>
                </pre>
              </div>
            )}

            {/* Observation: artifacts */}
            {observation?.artifacts && observation.artifacts.length > 0 && (
              <div className="border-t border-border bg-muted/40">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    Artifacts ({observation.artifacts.length})
                  </span>
                </div>
                <div className="p-3 flex flex-col gap-1">
                  {observation.artifacts.map((path, i) => (
                    <code
                      key={i}
                      className="text-xs bg-background rounded px-2 py-1 border border-border font-mono text-foreground/80 break-all"
                    >
                      {path}
                    </code>
                  ))}
                </div>
              </div>
            )}

            {/* Observation: data (extra structured fields) */}
            {observation?.data && Object.keys(observation.data).length > 0 && (
              <div className="border-t border-border bg-muted/40">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                    data
                  </span>
                </div>
                <pre className="code-block p-4 overflow-x-auto whitespace-pre-wrap">
                  <code className="text-foreground/80 text-xs font-mono">
                    {JSON.stringify(observation.data, null, 2)}
                  </code>
                </pre>
              </div>
            )}

            {/* Observation: rlm_calls (sub-LM calls) */}
            {observation?.rlm_calls && observation.rlm_calls.length > 0 && (
              <div className="border-t border-border bg-fuchsia-500/5 dark:bg-fuchsia-400/5">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-fuchsia-600 dark:text-fuchsia-400 font-medium">
                    Sub-LM Calls ({observation.rlm_calls.length})
                  </span>
                </div>
                <div className="p-4 space-y-3">
                  {observation.rlm_calls.map((call, i) => {
                    const inputTokens = call.usage_summary?.model_usage_summaries
                      ? Object.values(call.usage_summary.model_usage_summaries).reduce(
                          (acc, m) => acc + (m.total_input_tokens ?? 0),
                          0,
                        )
                      : 0;
                    const outputTokens = call.usage_summary?.model_usage_summaries
                      ? Object.values(call.usage_summary.model_usage_summaries).reduce(
                          (acc, m) => acc + (m.total_output_tokens ?? 0),
                          0,
                        )
                      : 0;
                    // Child trajectories from `rlm_query` carry the per-turn
                    // record on `metadata.iterations`. `llm_query` calls
                    // (single LLM round-trip) leave it null/empty.
                    const childIterations = call.metadata?.iterations ?? [];
                    const hasChildTrajectory = childIterations.length > 0;
                    // Build a stable label for the breadcrumb. Prefer the
                    // child_id stored on the parent observation (set by
                    // RecursionHandler) and fall back to the model name.
                    const childIdRaw = observation.data?.child_id;
                    const childId =
                      typeof childIdRaw === 'string' ? childIdRaw : `sub-${i + 1}`;
                    const drillLabel = `${childId} (${call.root_model})`;

                    return (
                      <div
                        key={i}
                        className="border border-fuchsia-500/30 dark:border-fuchsia-400/30 rounded-lg p-3 bg-background"
                      >
                        <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                          <div className="flex items-center gap-2">
                            <Badge className="bg-fuchsia-500 text-white dark:bg-fuchsia-400 dark:text-fuchsia-950 text-xs">
                              sub-call #{i + 1}
                            </Badge>
                            {hasChildTrajectory && (
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-6 px-2 text-[10px] border-fuchsia-500/40 text-fuchsia-600 dark:text-fuchsia-400 hover:bg-fuchsia-500/10"
                                onClick={() =>
                                  drillStack.push(
                                    {
                                      label: drillLabel,
                                      iterations: childIterations,
                                      config: call.metadata?.run_metadata ?? null,
                                    },
                                    drillDepth,
                                  )
                                }
                                title="Drill into the child's per-turn trajectory"
                              >
                                Open trajectory ({childIterations.length} iter
                                {childIterations.length !== 1 ? 's' : ''}) ↗
                              </Button>
                            )}
                          </div>
                          <div className="flex gap-2 text-xs text-muted-foreground font-mono">
                            <span>{inputTokens} in</span>
                            <span>•</span>
                            <span>{outputTokens} out</span>
                            <span>•</span>
                            <span>{call.execution_time.toFixed(2)}s</span>
                          </div>
                        </div>
                        <div className="text-xs text-muted-foreground mb-1">Prompt:</div>
                        <div className="text-xs bg-muted rounded p-2 mb-2 max-h-32 overflow-y-auto border border-border whitespace-pre-wrap font-mono">
                          {typeof call.prompt === 'string'
                            ? call.prompt.slice(0, 1000) + (call.prompt.length > 1000 ? '...' : '')
                            : JSON.stringify(call.prompt, null, 2).slice(0, 1000)}
                        </div>
                        <div className="text-xs text-muted-foreground mb-1">Response:</div>
                        <div className="text-xs bg-muted rounded p-2 max-h-40 overflow-y-auto border border-border whitespace-pre-wrap font-mono">
                          {call.response.slice(0, 1500) + (call.response.length > 1500 ? '...' : '')}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Final-answer marker */}
            {observation?.final_answer && (
              <div className="border-t border-border bg-emerald-500/10 dark:bg-emerald-400/10">
                <div className="px-3 py-1.5 border-b border-border/50 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-wider text-emerald-600 dark:text-emerald-400 font-medium">
                    final answer
                  </span>
                </div>
                <div className="p-4 text-sm text-emerald-700 dark:text-emerald-300 whitespace-pre-wrap">
                  {observation.final_answer}
                </div>
                {observation.final_artifacts && observation.final_artifacts.length > 0 && (
                  <div className="px-4 pb-4 flex flex-col gap-1">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                      final artifacts
                    </span>
                    {observation.final_artifacts.map((path, i) => (
                      <code
                        key={i}
                        className="text-xs bg-background rounded px-2 py-1 border border-border font-mono text-foreground/80 break-all"
                      >
                        {path}
                      </code>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
