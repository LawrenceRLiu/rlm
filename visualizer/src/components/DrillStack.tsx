'use client';

import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { IterationTimeline } from './IterationTimeline';
import { TrajectoryPanel } from './TrajectoryPanel';
import { ActionPanel } from './ActionPanel';
import { RLMConfigMetadata, WorkspaceIteration } from '@/lib/types';

// One frame on the drill-down stack. Each frame is the trajectory of a
// child opened from a parent's Sub-LM Call. The frame stack is rendered
// as a vertical column of cards: the root sits behind, the depth-1 frame
// is the topmost card, depth-2 sits beneath it, etc. Only one path
// through the call tree is visible at a time — clicking a sub-call at a
// shallower depth pops everything beneath it before pushing.
export interface DrillFrame {
  label: string; // breadcrumb segment, e.g. "child_3_1 (qwen-3.6)"
  iterations: WorkspaceIteration[];
  config?: RLMConfigMetadata | null;
}

interface DrillStackContextValue {
  frames: DrillFrame[];
  // `fromDepth` is the depth of the view the click came from (0 = root,
  // 1 = first frame, ...). Pushing from depth N truncates the stack to
  // length N, then appends the new frame as depth N+1. Result: there is
  // only ever one path visible from the root to the leaf.
  push: (frame: DrillFrame, fromDepth: number) => void;
  pop: () => void;
  popTo: (depth: number) => void;
}

const DrillStackContext = createContext<DrillStackContextValue | null>(null);

// Each rendered trajectory view exposes its own depth via this context so
// the ActionCards inside know what `fromDepth` to send when the user
// clicks "Open trajectory". Default 0 = root.
const DrillDepthContext = createContext<number>(0);

export function useDrillStack(): DrillStackContextValue {
  const ctx = useContext(DrillStackContext);
  if (!ctx) {
    return {
      frames: [],
      push: () => undefined,
      pop: () => undefined,
      popTo: () => undefined,
    };
  }
  return ctx;
}

export function useDrillDepth(): number {
  return useContext(DrillDepthContext);
}

interface DrillStackProviderProps {
  children: ReactNode;
  // Label for the root in the breadcrumb (typically the JSONL filename).
  rootLabel: string;
}

export function DrillStackProvider({ children, rootLabel }: DrillStackProviderProps) {
  const [frames, setFrames] = useState<DrillFrame[]>([]);

  const push = useCallback((frame: DrillFrame, fromDepth: number) => {
    setFrames((prev) => [...prev.slice(0, fromDepth), frame]);
  }, []);

  const pop = useCallback(() => {
    setFrames((prev) => prev.slice(0, -1));
  }, []);

  const popTo = useCallback((depth: number) => {
    setFrames((prev) => prev.slice(0, depth));
  }, []);

  const value = useMemo(
    () => ({ frames, push, pop, popTo }),
    [frames, push, pop, popTo],
  );

  return (
    <DrillStackContext.Provider value={value}>
      {/* Root content lives at depth 0 so its ActionCards know to push
          with fromDepth=0 (which always replaces any open chain). */}
      <DrillDepthContext.Provider value={0}>{children}</DrillDepthContext.Provider>
      {frames.length > 0 && (
        <NestedTrajectoryStack
          rootLabel={rootLabel}
          frames={frames}
          popTo={popTo}
          pop={pop}
        />
      )}
    </DrillStackContext.Provider>
  );
}

interface NestedTrajectoryStackProps {
  rootLabel: string;
  frames: DrillFrame[];
  popTo: (depth: number) => void;
  pop: () => void;
}

function NestedTrajectoryStack({
  rootLabel,
  frames,
  popTo,
  pop,
}: NestedTrajectoryStackProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // When a new frame is pushed (deeper level opened), scroll the stack
  // so the newly added card at the bottom comes into view.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
    });
  }, [frames.length]);

  // Esc pops one frame. Use capture so this handler runs before any
  // ancestor (LogViewer) Esc handler that would close the whole log.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        pop();
      }
    };
    window.addEventListener('keydown', handler, { capture: true });
    return () => window.removeEventListener('keydown', handler, { capture: true });
  }, [pop]);

  const breadcrumb = [
    { label: rootLabel, depth: 0 },
    ...frames.map((f, i) => ({ label: f.label, depth: i + 1 })),
  ];

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col">
      {/* Sticky header: breadcrumb across the whole call tree. */}
      <header className="flex-shrink-0 border-b border-border bg-card/80">
        <div className="px-6 py-3 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium flex-shrink-0">
              Sub-trajectory
            </span>
            <div className="flex items-center gap-1 min-w-0 overflow-x-auto text-sm">
              {breadcrumb.map((seg, i) => {
                const isLast = i === breadcrumb.length - 1;
                return (
                  <div key={i} className="flex items-center gap-1 flex-shrink-0">
                    {i > 0 && <span className="text-muted-foreground">›</span>}
                    {isLast ? (
                      <span className="font-mono text-foreground font-medium">{seg.label}</span>
                    ) : (
                      <button
                        onClick={() => popTo(seg.depth)}
                        className="font-mono text-muted-foreground hover:text-foreground transition-colors underline-offset-2 hover:underline"
                        title={
                          seg.depth === 0
                            ? 'Close all overlays and return to root'
                            : `Pop back to depth ${seg.depth}`
                        }
                      >
                        {seg.label}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-[10px] text-muted-foreground font-mono">
              {frames.length} level{frames.length !== 1 ? 's' : ''} open
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={pop}
              className="text-muted-foreground hover:text-foreground"
              title="Pop topmost (Esc)"
            >
              ✕ Pop
            </Button>
          </div>
        </div>
      </header>

      {/* Vertical stack of trajectory cards: depth 1 at top, deepest at
          bottom. Each card is independently navigable; clicking a
          sub-call inside a non-leaf card pops everything below it. */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto bg-muted/30">
        <div className="flex flex-col gap-4 p-4">
          {frames.map((frame, i) => (
            <DrillFrameCard
              key={`${i}:${frame.label}`}
              frame={frame}
              depth={i + 1}
              isTopmost={i === frames.length - 1}
              onCloseTopmost={pop}
            />
          ))}
        </div>
      </div>

      <div className="border-t border-border bg-muted/30 px-6 py-1.5 flex-shrink-0">
        <div className="flex items-center justify-center gap-6 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">Esc</kbd>
            Pop one level
          </span>
          <span>Click a breadcrumb segment to jump back any number of levels.</span>
        </div>
      </div>
    </div>
  );
}

interface DrillFrameCardProps {
  frame: DrillFrame;
  depth: number;
  isTopmost: boolean;
  onCloseTopmost: () => void;
}

// Each card is a self-contained, navigable trajectory view (timeline +
// prompt/response panel + actions/observations panel) with its own
// `selectedIteration` state. The card's contents are wrapped in a
// `DrillDepthContext.Provider` so any "Open trajectory" button rendered
// inside knows it's clicking from this depth, not the root's depth.
function DrillFrameCard({ frame, depth, isTopmost, onCloseTopmost }: DrillFrameCardProps) {
  const [selectedIteration, setSelectedIteration] = useState(0);

  // When the same frame slot gets replaced (e.g. a new sub-call pushed
  // at this depth), React re-mounts (we key on `${i}:${label}`), so
  // `selectedIteration` resets to 0 naturally.

  return (
    <DrillDepthContext.Provider value={depth}>
      <div
        className={`rounded-lg border bg-card overflow-hidden shadow-sm ${
          isTopmost
            ? 'border-fuchsia-500/40 dark:border-fuchsia-400/40 shadow-fuchsia-500/10'
            : 'border-border'
        }`}
      >
        {/* Per-frame header: depth chip, label, model. Topmost gets a close button. */}
        <div className="px-4 py-2 border-b border-border bg-muted/40 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 min-w-0">
            <Badge
              className={
                isTopmost
                  ? 'bg-fuchsia-500 text-white dark:bg-fuchsia-400 dark:text-fuchsia-950 text-[10px]'
                  : 'bg-muted text-muted-foreground text-[10px]'
              }
            >
              depth {depth}
            </Badge>
            <span className="font-mono text-sm text-foreground truncate">{frame.label}</span>
            <span className="text-[11px] text-muted-foreground font-mono">
              • {frame.iterations.length} iter{frame.iterations.length !== 1 ? 's' : ''}
              {frame.config?.root_model ? ` • ${frame.config.root_model}` : ''}
            </span>
          </div>
          {isTopmost && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onCloseTopmost}
              className="text-muted-foreground hover:text-foreground h-7 px-2 text-xs"
              title="Pop this level (Esc)"
            >
              ✕
            </Button>
          )}
        </div>

        {/* Iteration timeline strip — navigable per frame. */}
        <IterationTimeline
          iterations={frame.iterations}
          selectedIteration={selectedIteration}
          onSelectIteration={setSelectedIteration}
        />

        {/* Split: prompt/response | actions/observations. Min-height keeps
            the card readable when several are stacked. */}
        <div style={{ height: '70vh' }}>
          <ResizablePanelGroup orientation="horizontal">
            <ResizablePanel defaultSize={45} minSize={20} maxSize={80}>
              <div className="h-full border-r border-border">
                <TrajectoryPanel
                  iterations={frame.iterations}
                  selectedIteration={selectedIteration}
                  onSelectIteration={setSelectedIteration}
                />
              </div>
            </ResizablePanel>
            <ResizableHandle
              withHandle
              className="bg-border hover:bg-primary/30 transition-colors"
            />
            <ResizablePanel defaultSize={55} minSize={20} maxSize={80}>
              <div className="h-full bg-background">
                <ActionPanel iteration={frame.iterations[selectedIteration] ?? null} />
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        </div>
      </div>
    </DrillDepthContext.Provider>
  );
}
