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

interface RunSelectorProps {
  /** Current active tab ID — controls whether "New Run" is shown. */
  activeTab: string;
}

export function RunSelector({ activeTab }: RunSelectorProps) {
  const {
    trainingRuns,
    evalRuns,
    selectedTrainingRunId,
    selectedEvalRunId,
    loading,
    loadTrainingRuns,
    selectTrainingRun,
    selectEvalRun,
    addTrainingRun,
  } = useRunStore();

  const [trainingOpen, setTrainingOpen] = useState(false);
  const [evalOpen, setEvalOpen] = useState(false);
  const trainingRef = useRef<HTMLDivElement>(null);
  const evalRef = useRef<HTMLDivElement>(null);

  // Load runs on mount
  useEffect(() => {
    loadTrainingRuns();
  }, [loadTrainingRuns]);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (trainingRef.current && !trainingRef.current.contains(e.target as Node)) {
        setTrainingOpen(false);
      }
      if (evalRef.current && !evalRef.current.contains(e.target as Node)) {
        setEvalOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const selectedTraining = trainingRuns.find((r) => r.id === selectedTrainingRunId);
  const selectedEval = evalRuns.find((r) => r.id === selectedEvalRunId);

  const handleCreateRun = useCallback(async () => {
    const name = prompt('Training run name:');
    if (!name) return;
    const run = await createTrainingRun(name);
    addTrainingRun(run);
    await selectTrainingRun(run.id);
    setTrainingOpen(false);
  }, [addTrainingRun, selectTrainingRun]);

  if (loading && trainingRuns.length === 0) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-400">
        <span>Loading runs...</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {/* Training run selector */}
      <div ref={trainingRef} className="relative">
        <button
          onClick={() => { setTrainingOpen(!trainingOpen); setEvalOpen(false); }}
          className={clsx(
            'flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full border transition-colors',
            selectedTraining
              ? 'border-slate-200 text-slate-700 hover:border-slate-300 bg-white'
              : 'border-dashed border-slate-300 text-slate-400 hover:text-slate-600'
          )}
          title={selectedTraining ? `Training: ${selectedTraining.name}` : 'Select training run'}
        >
          <span className="text-[10px] uppercase tracking-wider text-slate-400 mr-0.5">Train</span>
          <span className="max-w-[100px] truncate">
            {selectedTraining?.name ?? 'None'}
          </span>
          <ChevronDown className="w-3 h-3 text-slate-400" />
        </button>

        {trainingOpen && (
          <div className="absolute top-full left-0 mt-1 z-50 min-w-[220px] bg-white rounded-lg border border-slate-200 shadow-lg py-1">
            {trainingRuns.map((run) => (
              <button
                key={run.id}
                onClick={() => { selectTrainingRun(run.id); setTrainingOpen(false); }}
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

      {/* Eval run selector — only shown when a training run is selected */}
      {selectedTrainingRunId && (
        <div ref={evalRef} className="relative">
          <button
            onClick={() => { setEvalOpen(!evalOpen); setTrainingOpen(false); }}
            className={clsx(
              'flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full border transition-colors',
              selectedEval
                ? 'border-slate-200 text-slate-700 hover:border-slate-300 bg-white'
                : 'border-dashed border-slate-300 text-slate-400 hover:text-slate-600'
            )}
            title={selectedEval ? `Eval: ${selectedEval.name}` : 'Select evaluation run'}
          >
            <span className="text-[10px] uppercase tracking-wider text-slate-400 mr-0.5">Eval</span>
            <span className="max-w-[100px] truncate">
              {selectedEval?.name ?? 'None'}
            </span>
            <ChevronDown className="w-3 h-3 text-slate-400" />
          </button>

          {evalOpen && (
            <div className="absolute top-full left-0 mt-1 z-50 min-w-[200px] bg-white rounded-lg border border-slate-200 shadow-lg py-1">
              {evalRuns.length === 0 ? (
                <div className="px-3 py-2 text-xs text-slate-400">No evaluations yet</div>
              ) : (
                evalRuns.map((run) => (
                  <button
                    key={run.id}
                    onClick={() => { selectEvalRun(run.id); setEvalOpen(false); }}
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
      )}
    </div>
  );
}
