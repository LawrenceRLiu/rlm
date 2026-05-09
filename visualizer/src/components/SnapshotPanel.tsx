'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { WorkspaceSnapshot } from '@/lib/types';

interface SnapshotPanelProps {
  snapshot: WorkspaceSnapshot | null;
}

export function SnapshotPanel({ snapshot }: SnapshotPanelProps) {
  if (!snapshot) {
    return (
      <Card className="border-dashed">
        <CardContent className="p-8 text-center">
          <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-muted/30 border border-border flex items-center justify-center">
            <span className="text-xl opacity-50">◉</span>
          </div>
          <p className="text-muted-foreground text-sm">No snapshot for this iteration</p>
          <p className="text-muted-foreground text-xs mt-1">
            The runtime did not commit a workspace snapshot
          </p>
        </CardContent>
      </Card>
    );
  }

  const changed = snapshot.changed_files ?? [];

  return (
    <div className="space-y-3">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-xs">
                turn {snapshot.turn}
              </Badge>
              <Badge variant="outline" className="font-mono text-xs">
                {snapshot.commit_sha.slice(0, 12)}
              </Badge>
            </div>
            <span className="text-[10px] text-muted-foreground font-mono">
              {changed.length} file{changed.length !== 1 ? 's' : ''} changed
            </span>
          </div>
          <p className="text-[11px] font-mono text-muted-foreground break-all">
            {snapshot.workspace_root}
          </p>
        </CardContent>
      </Card>

      {changed.length > 0 ? (
        <Card>
          <CardContent className="p-4">
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-2">
              Changed files
            </p>
            <div className="flex flex-col gap-1">
              {changed.map((path, i) => (
                <code
                  key={i}
                  className="text-xs bg-muted rounded px-2 py-1 border border-border font-mono text-foreground/80 break-all"
                >
                  {path}
                </code>
              ))}
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="border-dashed">
          <CardContent className="p-4 text-center text-xs text-muted-foreground">
            No file changes recorded for this turn (empty commit).
          </CardContent>
        </Card>
      )}
    </div>
  );
}
