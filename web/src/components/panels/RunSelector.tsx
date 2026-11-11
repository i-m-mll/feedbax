import { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { AlertCircle, ChevronDown } from 'lucide-react';
import { useRunStore } from '@/stores/runStore';

interface EvalRunSelectorProps {
  /** Currently selected eval run ID for the page. */
  selectedEvalRunId: string | null;
  /** Callback when the user selects a different eval run. */
  onSelectEvalRun: (id: string | null) => void;
}

/**
 * Per-page evaluation run selector.
 * Rendered in the analysis page settings panel (right sidebar).
 * Uses a form-style dropdown rather than a header pill.
 */
export function EvalRunSelector({ selectedEvalRunId, onSelectEvalRun }: EvalRunSelectorProps) {
  const { evalRuns, selectedTrainingRunId, evalError } = useRunStore();

  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const selectedEval = evalRuns.find((r) => r.id === selectedEvalRunId);

  if (!selectedTrainingRunId) {
    return (
      <div className="text-xs text-slate-400">
        Select a training run first
      </div>
    );
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={clsx(
          'w-full flex items-center justify-between gap-2 text-xs px-3 py-2 rounded-lg border transition-colors text-left',
          selectedEval
            ? 'border-slate-200 text-slate-700 hover:border-slate-300 bg-white'
            : 'border-dashed border-slate-200 text-slate-400 hover:text-slate-600 hover:border-slate-300'
        )}
        title={selectedEval ? `Eval: ${selectedEval.name}` : 'Select evaluation run'}
      >
        <span className="truncate">
          {selectedEval?.name ?? (evalError ? 'Evaluations unavailable' : 'None selected')}
        </span>
        {evalError && <AlertCircle className="w-3 h-3 text-red-500 shrink-0" />}
        <ChevronDown className="w-3 h-3 text-slate-400 shrink-0" />
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 min-w-[200px] bg-white rounded-lg border border-slate-200 shadow-lg py-1">
          {evalError && (
            <div className="mx-2 mb-1 rounded-md border border-red-100 bg-red-50 px-2 py-1.5 text-xs text-red-700">
              {evalError}
            </div>
          )}
          {/* "None" option to clear selection */}
          <button
            onClick={() => { onSelectEvalRun(null); setOpen(false); }}
            className={clsx(
              'w-full text-left px-3 py-1.5 hover:bg-slate-50 transition-colors text-xs',
              selectedEvalRunId === null && 'bg-brand-50'
            )}
          >
            <div className="text-slate-400 italic">None</div>
          </button>
          {evalRuns.length === 0 ? (
            <div className="px-3 py-2 text-xs text-slate-400">No evaluations yet</div>
          ) : (
            evalRuns.map((run) => (
              <button
                key={run.id}
                onClick={() => { onSelectEvalRun(run.id); setOpen(false); }}
                className={clsx(
                  'w-full text-left px-3 py-1.5 hover:bg-slate-50 transition-colors',
                  run.id === selectedEvalRunId && 'bg-brand-50'
                )}
              >
                <div className="text-xs font-medium text-slate-700">{run.name}</div>
                <div className="text-[10px] text-slate-400 flex items-center gap-2">
                  <span>{formatTimestamp(run.createdAt)}</span>
                  {run.description && (
                    <>
                      <span className="text-slate-300">|</span>
                      <span className="truncate max-w-[120px]">{run.description}</span>
                    </>
                  )}
                </div>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
