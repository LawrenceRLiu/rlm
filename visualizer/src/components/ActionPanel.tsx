'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ActionCard } from './ActionCard';
import { SnapshotPanel } from './SnapshotPanel';
import { ParseRetriesPanel } from './ParseRetriesPanel';
import { ReasoningCollapsible } from './ReasoningCollapsible';
import { WorkspaceIteration, pairActionsWithObservations } from '@/lib/types';

interface ActionPanelProps {
  iteration: WorkspaceIteration | null;
}

export function ActionPanel({ iteration }: ActionPanelProps) {
  if (!iteration) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-muted/30 border border-border flex items-center justify-center">
            <span className="text-3xl opacity-50">◇</span>
          </div>
          <p className="text-muted-foreground text-sm">
            Select an iteration to view actions and observations
          </p>
        </div>
      </div>
    );
  }

  const pairs = pairActionsWithObservations(iteration);
  const totalSubCalls = iteration.observations.reduce(
    (acc, o) => acc + (o.rlm_calls?.length ?? 0),
    0,
  );
  const retryCount = iteration.parse_attempts.length;
  const hasReasoning = !!iteration.reasoning && iteration.reasoning.length > 0;
  const changedFiles = iteration.snapshot?.changed_files ?? [];

  return (
    <div className="h-full flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-border bg-muted/30">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
              <span className="text-emerald-500 text-sm">⟨⟩</span>
            </div>
            <div>
              <h2 className="font-semibold text-sm">Actions & Observations</h2>
              <p className="text-[11px] text-muted-foreground">
                Iteration {iteration.iteration} •{' '}
                {iteration.timestamp ? new Date(iteration.timestamp).toLocaleString() : '—'}
              </p>
            </div>
          </div>
        </div>

        {/* Quick stats */}
        <div className="flex gap-2 flex-wrap">
          <Badge variant="outline" className="text-xs">
            {iteration.actions.length} action{iteration.actions.length !== 1 ? 's' : ''}
          </Badge>
          {totalSubCalls > 0 && (
            <Badge className="bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-400 border-fuchsia-500/30 text-xs">
              {totalSubCalls} sub-LM call{totalSubCalls !== 1 ? 's' : ''}
            </Badge>
          )}
          {retryCount > 0 && (
            <Badge className="bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 text-xs">
              {retryCount} parse retr{retryCount !== 1 ? 'ies' : 'y'}
            </Badge>
          )}
          {hasReasoning && (
            <Badge className="bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30 text-xs">
              has reasoning
            </Badge>
          )}
          {iteration.snapshot && (
            <Badge variant="outline" className="text-xs font-mono">
              {iteration.snapshot.commit_sha.slice(0, 7)}
            </Badge>
          )}
          {iteration.final_answer && (
            <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-xs">
              Final
            </Badge>
          )}
        </div>
      </div>

      <Tabs defaultValue="actions" className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-shrink-0 px-4 pt-3">
          <TabsList className="w-full grid grid-cols-3">
            <TabsTrigger value="actions" className="text-xs">
              Actions ({iteration.actions.length})
            </TabsTrigger>
            <TabsTrigger value="snapshot" className="text-xs">
              Snapshot ({changedFiles.length})
            </TabsTrigger>
            <TabsTrigger value="meta" className="text-xs">
              Meta ({retryCount + (hasReasoning ? 1 : 0)})
            </TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-hidden">
          {/* Actions tab */}
          <TabsContent
            value="actions"
            className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col"
          >
            <ScrollArea className="flex-1 h-full">
              <div className="p-4 space-y-4">
                {pairs.length > 0 ? (
                  pairs.map((p) => <ActionCard key={p.index} pair={p} />)
                ) : (
                  <Card className="border-dashed">
                    <CardContent className="p-8 text-center">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-muted/30 border border-border flex items-center justify-center">
                        <span className="text-xl opacity-50">⟨⟩</span>
                      </div>
                      <p className="text-muted-foreground text-sm">
                        No actions in this iteration
                      </p>
                      <p className="text-muted-foreground text-xs mt-1">
                        The model didn&apos;t emit any &lt;action&gt; blocks
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Snapshot tab */}
          <TabsContent
            value="snapshot"
            className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col"
          >
            <ScrollArea className="flex-1 h-full">
              <div className="p-4">
                <SnapshotPanel snapshot={iteration.snapshot} />
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Meta tab: parse retries + reasoning */}
          <TabsContent
            value="meta"
            className="h-full m-0 data-[state=active]:flex data-[state=active]:flex-col"
          >
            <ScrollArea className="flex-1 h-full">
              <div className="p-4 space-y-4">
                <ParseRetriesPanel attempts={iteration.parse_attempts} />
                <ReasoningCollapsible reasoning={iteration.reasoning} />
              </div>
            </ScrollArea>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
