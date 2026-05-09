'use client';

import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import { ParseAttempt } from '@/lib/types';

interface ParseRetriesPanelProps {
  attempts: ParseAttempt[];
}

export function ParseRetriesPanel({ attempts }: ParseRetriesPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!attempts || attempts.length === 0) {
    return null;
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card className="border-amber-500/40 bg-amber-500/5 dark:border-amber-400/40 dark:bg-amber-400/5">
        <CollapsibleTrigger asChild>
          <CardContent className="p-3 cursor-pointer hover:bg-muted/30 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Badge className="bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30 text-xs">
                  parse retries
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {attempts.length} failed parse{attempts.length !== 1 ? 's' : ''} before success
                </span>
              </div>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                <span className="text-xs">{isOpen ? '▼' : '▶'}</span>
              </Button>
            </div>
          </CardContent>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="px-3 pb-3 pt-0 space-y-3">
            {attempts.map((attempt, idx) => (
              <div
                key={idx}
                className="border border-amber-500/30 dark:border-amber-400/30 rounded-lg overflow-hidden"
              >
                <div className="px-3 py-1.5 bg-muted/50 border-b border-border/50 flex items-center gap-2">
                  <Badge variant="outline" className="text-[10px] font-mono">
                    attempt {idx + 1}
                  </Badge>
                  <span className="text-xs text-red-600 dark:text-red-400 truncate">
                    {attempt.error}
                  </span>
                </div>
                <pre className="p-3 text-xs whitespace-pre-wrap font-mono bg-background/60 max-h-48 overflow-y-auto">
                  {attempt.response.slice(0, 2000) +
                    (attempt.response.length > 2000 ? '\n\n... [truncated]' : '')}
                </pre>
              </div>
            ))}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
