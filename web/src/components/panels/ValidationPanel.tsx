import { useMemo } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { validateGraph } from '@/features/graph/validation';

export function ValidationPanel() {
  const graph = useGraphStore((state) => state.graph);
  const validation = useMemo(() => validateGraph(graph), [graph]);

  return (
    <div className="p-6 space-y-4 text-sm text-slate-600">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Validation</div>
        <div className="text-base font-semibold text-slate-800">Graph Health</div>
      </div>
      <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-2 max-w-3xl">
        {validation.valid ? (
          <div className="text-sm text-mint-500">Graph is valid.</div>
        ) : (
          <div className="space-y-2">
            {validation.errors.map((error, index) => (
              <div key={`error-${index}`} className="text-sm text-amber-600">
                {error.message}
              </div>
            ))}
          </div>
        )}
        {validation.warnings.length > 0 && (
          <div className="mt-2 space-y-1 text-xs text-slate-500">
            {validation.warnings.map((warning, index) => (
              <div key={`warning-${index}`}>{warning.message}</div>
            ))}
          </div>
        )}
        {validation.cycles.length > 0 && (
          <div className="mt-2 text-xs text-purple-500">
            Cycles detected: {validation.cycles.map((cycle) => cycle.join(' → ')).join(', ')}
          </div>
        )}
      </div>
    </div>
  );
}
