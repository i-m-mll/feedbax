import { useCallback, useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import { ChevronDown, Plus } from 'lucide-react';
import { useRunStore } from '@/stores/runStore';
import { createTrainingRun } from '@/api/runAPI';

/** Format an ISO timestamp for display. */
function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' +
    d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

/** Format hyperparams as a short summary string. */
function formatHyperparams(hp: Record<string, string | number>): string {
  return Object.entries(hp)
    .map(([k, v]) => `${k}=${v}`)
    .join(', ');
}

// ---------------------------------------------------------------------------
// TrainingRunSelector — global, lives in the bottom shelf header
// ---------------------------------------------------------------------------

interface TrainingRunSelectorProps {
  /** Current active tab ID — controls whether "New Run" is shown. */
  activeTab: string;
}

/**
 * Global training run selector pill.
 * Rendered in the bottom shelf header (left side, before tab pills).
 */
export function TrainingRunSelector({ activeTab }: TrainingRunSelectorProps) {
  const {
    trainingRuns,
    selectedTrainingRunId,
    loading,
    loadTrainingRuns,
    selectTrainingRun,
    addTrainingRun,
  } = useRunStore();

  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Load runs on mount
  useEffect(() => {
    loadTrainingRuns();
  }, [loadTrainingRuns]);

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

  const selectedTraining = trainingRuns.find((r) => r.id === selectedTrainingRunId);

  const handleCreateRun = useCallback(async () => {
    const name = prompt('Training run name:');
    if (!name) return;
    const run = await createTrainingRun(name);
    addTrainingRun(run);
    await selectTrainingRun(run.id);
    setOpen(false);
  }, [addTrainingRun, selectTrainingRun]);

  if (loading && trainingRuns.length === 0) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-400">
        <span>Loading runs...</span>
      </div>
    );
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={clsx(
          'flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full border transition-colors',
          selectedTraining
            ? 'border-slate-200 text-slate-700 hover:border-slate-300 bg-white'
            : 'border-dashed border-slate-300 text-slate-400 hover:text-slate-600'
        )}
        title={selectedTraining ? `Training: ${selectedTraining.name}` : 'Select training run'}
      >
        <span className="text-[10px] uppercase tracking-wider text-slate-400 mr-0.5">Run</span>
        <span className="max-w-[100px] truncate">
          {selectedTraining?.name ?? 'None'}
        </span>
        <ChevronDown className="w-3 h-3 text-slate-400" />
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-1 z-50 min-w-[220px] bg-white rounded-lg border border-slate-200 shadow-lg py-1">
          {trainingRuns.map((run) => (
            <button
              key={run.id}
              onClick={() => { selectTrainingRun(run.id); setOpen(false); }}
              className={clsx(
                'w-full text-left px-3 py-1.5 hover:bg-slate-50 transition-colors',
                run.id === selectedTrainingRunId && 'bg-brand-50'
              )}
            >
              <div className="text-xs font-medium text-slate-700">{run.name}</div>
              <div className="text-[10px] text-slate-400 flex items-center gap-2">
                <span>{formatTimestamp(run.createdAt)}</span>
                {Object.keys(run.hyperparams).length > 0 && (
                  <>
                    <span className="text-slate-300">|</span>
                    <span className="truncate max-w-[140px]">
                      {formatHyperparams(run.hyperparams)}
                    </span>
                  </>
                )}
              </div>
            </button>
          ))}
          {activeTab === 'training' && (
            <>
              {trainingRuns.length > 0 && (
                <div className="border-t border-slate-100 my-0.5" />
              )}
              <button
                onClick={handleCreateRun}
                className="w-full text-left px-3 py-1.5 hover:bg-slate-50 transition-colors flex items-center gap-1.5 text-xs text-brand-600"
              >
                <Plus className="w-3 h-3" />
                <span>New training run</span>
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// EvalRunSelector — per-page, lives in analysis page settings sidebar
// ---------------------------------------------------------------------------

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
  const { evalRuns, selectedTrainingRunId } = useRunStore();

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
          {selectedEval?.name ?? 'None selected'}
        </span>
        <ChevronDown className="w-3 h-3 text-slate-400 shrink-0" />
      </button>

      {open && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 min-w-[200px] bg-white rounded-lg border border-slate-200 shadow-lg py-1">
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

// ---------------------------------------------------------------------------
// Legacy re-export for backward compatibility
// ---------------------------------------------------------------------------

/** @deprecated Use TrainingRunSelector and EvalRunSelector separately. */
export const RunSelector = TrainingRunSelector;
