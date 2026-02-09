import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphSpec } from '@/types/graph';
import clsx from 'clsx';

interface CycleAnnotation {
  source: string;
  target: string;
}

interface TreescopeResponse {
  html: string;
  has_cycles: boolean;
  cycle_count: number;
  cycles: CycleAnnotation[];
  execution_order: string[] | null;
}

interface InspectionStatus {
  treescope_available: boolean;
  treescope_configured: boolean;
  treescope_version: string | null;
}

async function fetchInspectionStatus(): Promise<InspectionStatus> {
  const response = await fetch('/api/inspection/status');
  if (!response.ok) {
    throw new Error('Failed to fetch inspection status');
  }
  return response.json();
}

async function fetchTreescope(
  graphId: string,
  options: { maxDepth: number; projectCycles: boolean }
): Promise<TreescopeResponse> {
  const response = await fetch(`/api/inspection/treescope/${graphId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      max_depth: options.maxDepth,
      project_cycles: options.projectCycles,
      roundtrip_mode: false,
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to render Treescope');
  }
  return response.json();
}

async function fetchTreescopeInline(
  graph: GraphSpec,
  options: { maxDepth: number; projectCycles: boolean }
): Promise<TreescopeResponse> {
  const response = await fetch('/api/inspection/treescope/inline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph,
      max_depth: options.maxDepth,
      project_cycles: options.projectCycles,
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to render Treescope');
  }
  return response.json();
}

export function TreescopePanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [maxDepth, setMaxDepth] = useState(10);
  const [projectCycles, setProjectCycles] = useState(true);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const graphId = useGraphStore((state) => state.graphId);
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  const currentGraphLabel = useGraphStore((state) => state.currentGraphLabel);

  // Create a stable key for the current graph content so we debounce
  // and re-query when the graph actually changes (nodes/wires).
  const graphKey = useMemo(
    () => JSON.stringify({ nodes: Object.keys(graph.nodes), wires: graph.wires.length }),
    [graph]
  );

  const [debouncedGraphKey, setDebouncedGraphKey] = useState(graphKey);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedGraphKey(graphKey);
    }, 300);
    return () => clearTimeout(timer);
  }, [graphKey]);

  // Whether we have any nodes to visualize
  const hasNodes = Object.keys(graph.nodes).length > 0;

  // Check if treescope is available
  const statusQuery = useQuery({
    queryKey: ['inspection-status'],
    queryFn: fetchInspectionStatus,
    staleTime: 60 * 1000,
    retry: 1,
  });

  // Fetch treescope visualization when graph changes.
  // Use the saved-graph endpoint when we have a graphId and are at the
  // top level; otherwise use the inline endpoint which accepts the
  // current (possibly subgraph-level) graph spec directly.
  const treescopeQuery = useQuery({
    queryKey: ['treescope', graphId ?? 'inline', debouncedGraphKey, graphStack.length, maxDepth, projectCycles],
    queryFn: () => {
      if (graphId && graphStack.length === 0) {
        // Saved graph at top level -- use the persisted endpoint
        return fetchTreescope(graphId, { maxDepth, projectCycles });
      }
      // Either unsaved or inside a subgraph -- use inline rendering
      return fetchTreescopeInline(graph, { maxDepth, projectCycles });
    },
    enabled: isExpanded && hasNodes && statusQuery.data?.treescope_available === true,
    staleTime: 30 * 1000,
    retry: 1,
  });

  // Update iframe content when HTML changes
  useEffect(() => {
    if (iframeRef.current && treescopeQuery.data?.html) {
      const doc = iframeRef.current.contentDocument;
      if (doc) {
        doc.open();
        doc.write(treescopeQuery.data.html);
        doc.close();
      }
    }
  }, [treescopeQuery.data?.html]);

  const cycleCount = treescopeQuery.data?.cycle_count ?? 0;
  const hasCycles = treescopeQuery.data?.has_cycles ?? false;

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  // If treescope is not available, show a minimal panel
  if (statusQuery.isLoading) {
    return (
      <div className="border-t border-slate-100 p-4">
        <div className="text-xs text-slate-400">Loading inspection status...</div>
      </div>
    );
  }

  if (!statusQuery.data?.treescope_available) {
    return (
      <div className="border-t border-slate-100 p-4">
        <div className="text-xs text-slate-400">
          Treescope visualization is not available.
          <br />
          Install with: <code className="bg-slate-100 px-1 rounded">pip install treescope</code>
        </div>
      </div>
    );
  }

  return (
    <div className="border-t border-slate-100">
      {/* Header */}
      <button
        onClick={toggleExpanded}
        className={clsx(
          'flex w-full items-center justify-between px-4 py-3',
          'text-left text-xs font-semibold uppercase tracking-[0.3em] text-slate-400',
          'hover:bg-slate-50 transition-colors'
        )}
      >
        <span className="flex items-center gap-2">
          Treescope
          {hasCycles && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-700 normal-case tracking-normal">
              {cycleCount} cycle{cycleCount !== 1 ? 's' : ''}
            </span>
          )}
        </span>
        <svg
          className={clsx(
            'h-4 w-4 transition-transform',
            isExpanded ? 'rotate-180' : ''
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-3">
          {/* Subgraph scope indicator */}
          {graphStack.length > 0 && (
            <div className="text-[10px] text-slate-400">
              Showing: {currentGraphLabel}
            </div>
          )}

          {/* Options */}
          <div className="flex flex-wrap gap-3 text-xs">
            <label className="flex items-center gap-2 text-slate-500">
              <span>Depth:</span>
              <input
                type="number"
                min={1}
                max={50}
                value={maxDepth}
                onChange={(e) => setMaxDepth(Number(e.target.value))}
                className="w-16 rounded border border-slate-200 px-2 py-1 text-slate-700"
              />
            </label>
            <label className="flex items-center gap-2 text-slate-500">
              <input
                type="checkbox"
                checked={projectCycles}
                onChange={(e) => setProjectCycles(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-slate-300 text-brand-500"
              />
              <span>Show cycles</span>
            </label>
          </div>

          {/* Status messages */}
          {!hasNodes && (
            <div className="rounded bg-slate-50 p-3 text-xs text-slate-500">
              Add nodes to see Treescope visualization.
            </div>
          )}

          {hasNodes && treescopeQuery.isLoading && (
            <div className="rounded bg-slate-50 p-3 text-xs text-slate-500">
              Rendering visualization...
            </div>
          )}

          {hasNodes && treescopeQuery.error && (
            <div className="rounded bg-red-50 p-3 text-xs text-red-600">
              {treescopeQuery.error instanceof Error
                ? treescopeQuery.error.message
                : 'Failed to render visualization'}
            </div>
          )}

          {/* Cycle annotations */}
          {hasCycles && treescopeQuery.data?.cycles && (
            <div className="rounded bg-amber-50 p-3 space-y-1">
              <div className="text-xs font-medium text-amber-800">Feedback Loops</div>
              <ul className="text-xs text-amber-700 space-y-0.5">
                {treescopeQuery.data.cycles.slice(0, 5).map((cycle, idx) => (
                  <li key={idx}>
                    {cycle.source} → {cycle.target}
                  </li>
                ))}
                {treescopeQuery.data.cycles.length > 5 && (
                  <li className="text-amber-600">
                    ... and {treescopeQuery.data.cycles.length - 5} more
                  </li>
                )}
              </ul>
            </div>
          )}

          {/* Treescope iframe */}
          {hasNodes && treescopeQuery.data?.html && (
            <div className="rounded border border-slate-200 overflow-hidden">
              <iframe
                ref={iframeRef}
                title="Treescope Visualization"
                sandbox="allow-scripts"
                className="w-full h-64 bg-white"
                style={{ minHeight: '256px', maxHeight: '400px' }}
              />
            </div>
          )}

          {/* Execution order */}
          {treescopeQuery.data?.execution_order && (
            <details className="text-xs">
              <summary className="cursor-pointer text-slate-500 hover:text-slate-700">
                Execution Order
              </summary>
              <div className="mt-2 rounded bg-slate-50 p-2 font-mono text-slate-600">
                {treescopeQuery.data.execution_order.join(' → ')}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
