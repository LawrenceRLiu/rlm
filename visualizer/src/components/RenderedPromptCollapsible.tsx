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

interface RenderedPromptCollapsibleProps {
  renderedPrompt: string | null | undefined;
}

// Shows the post-chat-template prompt the model literally saw on this turn —
// vLLM's tool-description injection, ``<tool_call>`` wrapping instructions,
// special tokens, plus the full message history. Best-effort, vLLM-only.
export function RenderedPromptCollapsible({ renderedPrompt }: RenderedPromptCollapsibleProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!renderedPrompt || renderedPrompt.length === 0) {
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
                  rendered prompt (post-chat-template)
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {renderedPrompt.length.toLocaleString()} chars
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
            <pre className="bg-background/70 rounded-lg border border-amber-500/30 dark:border-amber-400/30 p-3 text-xs whitespace-pre-wrap font-mono max-h-96 overflow-y-auto text-foreground/90">
              {renderedPrompt}
            </pre>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
