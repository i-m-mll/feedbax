import { useCallback, useMemo, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { validateGraph } from '@/features/graph/validation';
import clsx from 'clsx';

export function ValidationPanel() {
  const [isExpanded, setIsExpanded] = useState(true);

  // state.graph is always the currently active layer (root or nested subgraph),
  // because the store updates it whenever the user enters/exits a subgraph.
  const graph = useGraphStore((state) => state.graph);
  const currentGraphLabel = useGraphStore((state) => state.currentGraphLabel);
  const isInSubgraph = useGraphStore((state) => state.graphStack.length > 0);
  const validation = useMemo(() => validateGraph(graph), [graph]);

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const hasIssues = !validation.valid || validation.warnings.length > 0 || validation.cycles.length > 0;
  const errorCount = validation.errors.length;
  const warningCount = validation.warnings.length + validation.cycles.length;

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
          Validation
          {errorCount > 0 && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-700 normal-case tracking-normal">
              {errorCount} error{errorCount !== 1 ? 's' : ''}
            </span>
          )}
          {errorCount === 0 && warningCount > 0 && (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500 normal-case tracking-normal">
              {warningCount} warning{warningCount !== 1 ? 's' : ''}
            </span>
          )}
          {!hasIssues && (
            <span className="rounded-full bg-mint-100 px-2 py-0.5 text-[10px] font-medium text-mint-600 normal-case tracking-normal">
              valid
            </span>
          )}
        </span>
        <svg
          className={clsx('h-4 w-4 transition-transform', isExpanded ? 'rotate-180' : '')}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-2">
          {/* Scope indicator */}
          {isInSubgraph && (
            <div className="text-[10px] text-slate-400">
              Showing: {currentGraphLabel}
            </div>
          )}

          {validation.valid && validation.warnings.length === 0 && validation.cycles.length === 0 ? (
            <div className="text-xs text-mint-500">Graph is valid.</div>
          ) : (
            <div className="space-y-1.5">
              {validation.errors.map((error, index) => (
                <div key={`error-${index}`} className="text-xs text-amber-600">
                  {error.message}
                </div>
              ))}
              {validation.warnings.map((warning, index) => (
                <div key={`warning-${index}`} className="text-xs text-slate-500">
                  {warning.message}
                </div>
              ))}
              {validation.cycles.length > 0 && (
                <div className="text-xs text-purple-500">
                  Cycles: {validation.cycles.map((cycle) => cycle.join(' → ')).join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
