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

interface ReasoningCollapsibleProps {
  reasoning: string | null;
}

export function ReasoningCollapsible({ reasoning }: ReasoningCollapsibleProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!reasoning || reasoning.length === 0) {
    return null;
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card className="border-violet-500/40 bg-violet-500/5 dark:border-violet-400/40 dark:bg-violet-400/5">
        <CollapsibleTrigger asChild>
          <CardContent className="p-3 cursor-pointer hover:bg-muted/30 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Badge className="bg-violet-500/15 text-violet-600 dark:text-violet-400 border-violet-500/30 text-xs">
                  reasoning channel
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {reasoning.length.toLocaleString()} chars
                </span>
              </div>
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                <span className="text-xs">{isOpen ? '▼' : '▶'}</span>
              </Button>
            </div>
          </CardContent>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <CardContent className="px-3 pb-3 pt-0">
            <pre className="bg-background/70 rounded-lg border border-violet-500/30 dark:border-violet-400/30 p-3 text-xs whitespace-pre-wrap font-mono max-h-96 overflow-y-auto text-foreground/90">
              {reasoning}
            </pre>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
