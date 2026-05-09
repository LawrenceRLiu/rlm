'use client';

import { useRef, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { WorkspaceIteration } from '@/lib/types';

interface IterationTimelineProps {
  iterations: WorkspaceIteration[];
  selectedIteration: number;
  onSelectIteration: (index: number) => void;
}

function getIterationStats(iteration: WorkspaceIteration) {
  let totalSubCalls = 0;
  let hasError = false;

  for (const obs of iteration.observations) {
    if (obs.error || (obs.stderr && obs.stderr.length > 0)) hasError = true;
    if (obs.rlm_calls) totalSubCalls += obs.rlm_calls.length;
  }

  const iterTime = iteration.iteration_time ?? 0;
  const promptText = iteration.prompt.map((m) => m.content).join('');
  const estimatedInputTokens = Math.round(promptText.length / 4);
  const estimatedOutputTokens = Math.round(iteration.response.length / 4);
  const changedFiles = iteration.snapshot?.changed_files?.length ?? 0;

  return {
    actions: iteration.actions.length,
    subCalls: totalSubCalls,
    parseRetries: iteration.parse_attempts.length,
    execTime: iterTime,
    hasError,
    hasFinal: iteration.final_answer !== null,
    inputTokens: estimatedInputTokens,
    outputTokens: estimatedOutputTokens,
    changedFiles,
  };
}

export function IterationTimeline({
  iterations,
  selectedIteration,
  onSelectIteration,
}: IterationTimelineProps) {
  const selectedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'center',
      });
    }
  }, [selectedIteration]);

  return (
    <div className="border-b border-border bg-muted/30 flex-shrink-0">
      <div className="px-4 pt-3 pb-2 flex items-center gap-2">
        <div className="w-5 h-5 rounded bg-primary/10 flex items-center justify-center">
          <svg
            className="w-3 h-3 text-primary"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
        </div>
        <span className="text-xs font-semibold text-foreground">
          Workspace Substrate Trajectory
        </span>
        <span className="text-[10px] text-muted-foreground">({iterations.length} total)</span>
        <div className="flex-1" />
        <span className="text-[10px] text-muted-foreground">← scroll →</span>
      </div>

      <ScrollArea className="w-full">
        <div className="flex gap-2 px-3 pb-3">
          {iterations.map((iteration, idx) => {
            const stats = getIterationStats(iteration);
            const isSelected = idx === selectedIteration;
            const responseSnippet = iteration.response.slice(0, 60).replace(/\n/g, ' ');

            return (
              <div
                key={idx}
                ref={isSelected ? selectedRef : null}
                onClick={() => onSelectIteration(idx)}
                className={cn(
                  'flex-shrink-0 w-72 cursor-pointer transition-all duration-150 rounded-lg border',
                  isSelected
                    ? 'border-primary bg-primary/10 shadow-md shadow-primary/15'
                    : stats.hasFinal
                      ? 'border-emerald-500/40 bg-emerald-500/5 hover:border-emerald-500/60 dark:border-emerald-400/40 dark:bg-emerald-400/5'
                      : stats.hasError
                        ? 'border-red-500/40 bg-red-500/5 hover:border-red-500/60 dark:border-red-400/40 dark:bg-red-400/5'
                        : 'border-border hover:border-primary/40 hover:bg-muted/50',
                )}
              >
                <div className="p-2.5 flex items-start gap-3">
                  <div
                    className={cn(
                      'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0',
                      isSelected
                        ? 'bg-primary text-primary-foreground'
                        : stats.hasFinal
                          ? 'bg-emerald-500 text-white dark:bg-emerald-400'
                          : stats.hasError
                            ? 'bg-red-500 text-white dark:bg-red-400'
                            : 'bg-muted text-muted-foreground',
                    )}
                  >
                    {idx + 1}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 mb-1.5 flex-wrap">
                      {stats.hasFinal && (
                        <Badge className="bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30 text-[9px] px-1 py-0 h-4">
                          FINAL
                        </Badge>
                      )}
                      {stats.hasError && (
                        <Badge variant="destructive" className="text-[9px] px-1 py-0 h-4">
                          ERR
                        </Badge>
                      )}
                      {stats.parseRetries > 0 && (
                        <Badge className="bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 text-[9px] px-1 py-0 h-4">
                          {stats.parseRetries} retry
                        </Badge>
                      )}
                      {stats.actions > 0 && (
                        <span className="text-[10px] text-emerald-600 dark:text-emerald-400">
                          {stats.actions} action{stats.actions !== 1 ? 's' : ''}
                        </span>
                      )}
                      {stats.subCalls > 0 && (
                        <span className="text-[10px] text-fuchsia-600 dark:text-fuchsia-400">
                          {stats.subCalls} sub
                        </span>
                      )}
                      {stats.changedFiles > 0 && (
                        <span className="text-[10px] text-sky-600 dark:text-sky-400">
                          Δ{stats.changedFiles}
                        </span>
                      )}
                      <span className="text-[10px] text-muted-foreground ml-auto">
                        {stats.execTime.toFixed(2)}s
                      </span>
                    </div>

                    <p className="text-[10px] text-muted-foreground truncate leading-relaxed">
                      {responseSnippet}
                      {iteration.response.length > 60 ? '...' : ''}
                    </p>

                    <div className="flex items-center gap-2 mt-1 text-[9px] font-mono text-muted-foreground/70">
                      <span>
                        <span className="text-sky-600 dark:text-sky-400">
                          {(stats.inputTokens / 1000).toFixed(1)}k
                        </span>
                        <span className="mx-0.5">→</span>
                        <span className="text-emerald-600 dark:text-emerald-400">
                          {(stats.outputTokens / 1000).toFixed(1)}k
                        </span>
                      </span>
                      {stats.hasFinal && iteration.final_answer && (
                        <>
                          <span className="text-border">│</span>
                          <span className="text-amber-600 dark:text-amber-400 truncate max-w-[100px]">
                            = {iteration.final_answer}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
