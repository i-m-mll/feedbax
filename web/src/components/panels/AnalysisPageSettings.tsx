/**
 * AnalysisPageSettings — right sidebar content when no analysis node is selected.
 *
 * Shows page-level settings: name, eval run selector, and eval parametrization
 * fields (perturbation type, amplitudes, SISU values, task variants).
 * Includes a "Run Evaluation" button that creates a new eval run and
 * triggers the evaluation pipeline.
 */

import { useCallback, useState } from 'react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useRunStore } from '@/stores/runStore';
import { EvalRunSelector } from '@/components/panels/RunSelector';
import { createEvalRun } from '@/api/runAPI';
import { Plus, Trash2, Play, Loader2, CheckCircle2 } from 'lucide-react';

const PERTURBATION_TYPES = [
  'curl_field',
  'constant_field',
  'gusts',
  'feedback_impulse',
  'unit_stim',
] as const;

/** Parse a comma-separated string into an array of numbers, ignoring invalid entries. */
function parseNumberList(raw: string): number[] {
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((n) => !isNaN(n));
}

/** Format a number array as a comma-separated string. */
function formatNumberList(nums: unknown): string {
  if (!Array.isArray(nums)) return '';
  return nums.join(', ');
}

/** Generate a default eval run name from the current parameters. */
function generateDefaultName(params: Record<string, unknown>): string {
  const parts: string[] = [];
  const pertType = params.perturbation_type;
  if (pertType && typeof pertType === 'string') {
    parts.push(pertType.replace(/_/g, ' '));
  }
  const amps = params.perturbation_amplitudes;
  if (Array.isArray(amps) && amps.length > 0) {
    parts.push(`amp ${amps.join(',')}`);
  }
  if (parts.length === 0) return 'Eval run';
  return parts.join(' - ');
}

export function AnalysisPageSettings() {
  const activePageId = useAnalysisStore((s) => s.activePageId);
  const pages = useAnalysisStore((s) => s.pages);
  const renamePage = useAnalysisStore((s) => s.renamePage);
  const evalParams = useAnalysisStore((s) => s.evalParams);
  const setEvalParams = useAnalysisStore((s) => s.setEvalParams);
  const evalRunId = useAnalysisStore((s) => s.evalRunId);
  const setEvalRunId = useAnalysisStore((s) => s.setEvalRunId);

  const selectedTrainingRunId = useRunStore((s) => s.selectedTrainingRunId);
  const evalRuns = useRunStore((s) => s.evalRuns);
  const addEvalRun = useRunStore((s) => s.addEvalRun);
  const updateEvalRunStatus = useRunStore((s) => s.updateEvalRunStatus);

  const activePage = pages.find((p) => p.id === activePageId);

  // Local state for the task variants key-value editor
  const [newVariantKey, setNewVariantKey] = useState('');
  const [newVariantValue, setNewVariantValue] = useState('');

  // Eval run name and creation state
  const [evalRunName, setEvalRunName] = useState('');
  const [evalRunning, setEvalRunning] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalSuccess, setEvalSuccess] = useState(false);

  const handleNameChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (activePageId) {
        renamePage(activePageId, e.target.value);
      }
    },
    [activePageId, renamePage],
  );

  const handlePerturbationTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setEvalParams({ ...evalParams, perturbation_type: e.target.value });
    },
    [evalParams, setEvalParams],
  );

  const handleAmplitudesChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const nums = parseNumberList(e.target.value);
      setEvalParams({ ...evalParams, perturbation_amplitudes: nums });
    },
    [evalParams, setEvalParams],
  );

  const handleSisuChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const nums = parseNumberList(e.target.value);
      setEvalParams({ ...evalParams, sisu_values: nums });
    },
    [evalParams, setEvalParams],
  );

  const handleAddVariant = useCallback(() => {
    if (!newVariantKey.trim()) return;
    const variants = (evalParams.task_variants as Record<string, string>) ?? {};
    setEvalParams({
      ...evalParams,
      task_variants: { ...variants, [newVariantKey.trim()]: newVariantValue },
    });
    setNewVariantKey('');
    setNewVariantValue('');
  }, [evalParams, setEvalParams, newVariantKey, newVariantValue]);

  const handleRemoveVariant = useCallback(
    (key: string) => {
      const variants = { ...((evalParams.task_variants as Record<string, string>) ?? {}) };
      delete variants[key];
      setEvalParams({ ...evalParams, task_variants: variants });
    },
    [evalParams, setEvalParams],
  );

  const handleRunEvaluation = useCallback(async () => {
    if (!selectedTrainingRunId) {
      setEvalError('Select a training run first');
      return;
    }

    const name = evalRunName.trim() || generateDefaultName(evalParams);

    // Check for duplicate name + params
    const existingDupe = evalRuns.find((r) => {
      if (r.name !== name) return false;
      // Simple param comparison via JSON serialization
      const existingDesc = r.description ?? '';
      const currentDesc = [
        evalParams.perturbation_type,
        Array.isArray(evalParams.perturbation_amplitudes)
          ? `amp=[${evalParams.perturbation_amplitudes.join(',')}]`
          : '',
      ]
        .filter(Boolean)
        .join(', ');
      return existingDesc === currentDesc;
    });

    if (existingDupe) {
      const confirmed = window.confirm(
        'An evaluation with this name and parameters already exists. Run again?',
      );
      if (!confirmed) return;
    }

    setEvalRunning(true);
    setEvalError(null);
    setEvalSuccess(false);

    try {
      const run = await createEvalRun(
        selectedTrainingRunId,
        name,
        evalParams,
      );
      addEvalRun(run);
      setEvalRunId(run.id);

      // Simulate completion after creation (the backend handles the
      // actual evaluation; in stub mode we mark it completed quickly).
      // In a real scenario the backend would update status via polling.
      setTimeout(() => {
        updateEvalRunStatus(run.id, 'completed');
        setEvalRunning(false);
        setEvalSuccess(true);
        // Clear success indicator after a few seconds
        setTimeout(() => setEvalSuccess(false), 3000);
      }, 1500);
    } catch (err) {
      setEvalError(err instanceof Error ? err.message : 'Failed to create evaluation run');
      setEvalRunning(false);
    }
  }, [
    selectedTrainingRunId,
    evalRunName,
    evalParams,
    evalRuns,
    addEvalRun,
    setEvalRunId,
    updateEvalRunStatus,
  ]);

  if (!activePage) {
    return (
      <div className="p-4 text-xs text-slate-400 italic">
        No active page
      </div>
    );
  }

  const taskVariants = (evalParams.task_variants as Record<string, string>) ?? {};

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="text-xs uppercase tracking-[0.3em] text-emerald-600">
        Page Settings
      </div>

      {/* Page name */}
      <div>
        <label className="text-[10px] uppercase tracking-[0.2em] text-slate-400 block mb-1">
          Name
        </label>
        <input
          type="text"
          value={activePage.name}
          onChange={handleNameChange}
          className="w-full text-sm text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1 focus:outline-none focus:border-emerald-300 focus:ring-1 focus:ring-emerald-200"
        />
      </div>

      {/* Eval run selector — per-page */}
      <div className="border-t border-slate-100 pt-3 space-y-1.5">
        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400">
          Evaluation Run
        </div>
        <EvalRunSelector
          selectedEvalRunId={evalRunId}
          onSelectEvalRun={setEvalRunId}
        />
        <div className="text-[10px] text-slate-400 leading-relaxed">
          Select which evaluation run to use for analyses on this page.
        </div>
      </div>

      {/* Eval Parametrization */}
      <div className="border-t border-slate-100 pt-3">
        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400 mb-3">
          Eval Parametrization
        </div>

        {/* Perturbation type */}
        <div className="mb-3">
          <label className="text-xs text-slate-500 block mb-1">
            Perturbation type
          </label>
          <select
            value={(evalParams.perturbation_type as string) ?? ''}
            onChange={handlePerturbationTypeChange}
            className="w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1.5 focus:outline-none focus:border-emerald-300"
          >
            <option value="">None</option>
            {PERTURBATION_TYPES.map((t) => (
              <option key={t} value={t}>
                {t.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </div>

        {/* Perturbation amplitudes */}
        <div className="mb-3">
          <label className="text-xs text-slate-500 block mb-1">
            Perturbation amplitudes
          </label>
          <input
            key={`amplitudes-${activePageId}`}
            type="text"
            placeholder="0.1, 0.5, 1.0"
            defaultValue={formatNumberList(evalParams.perturbation_amplitudes)}
            onBlur={handleAmplitudesChange}
            className="w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1.5 focus:outline-none focus:border-emerald-300"
          />
        </div>

        {/* SISU values */}
        <div className="mb-3">
          <label className="text-xs text-slate-500 block mb-1">
            SISU values
          </label>
          <input
            key={`sisu-${activePageId}`}
            type="text"
            placeholder="0.0, 0.5, 1.0"
            defaultValue={formatNumberList(evalParams.sisu_values)}
            onBlur={handleSisuChange}
            className="w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1.5 focus:outline-none focus:border-emerald-300"
          />
        </div>

        {/* Task variants */}
        <div>
          <label className="text-xs text-slate-500 block mb-1">
            Task variants
          </label>
          <div className="space-y-1.5">
            {Object.entries(taskVariants).map(([key, value]) => (
              <div key={key} className="flex items-center gap-1.5">
                <span className="text-xs text-slate-500 bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 min-w-[60px]">
                  {key}
                </span>
                <span className="text-xs text-slate-700 font-medium bg-slate-50 rounded px-1.5 py-0.5 flex-1 truncate">
                  {String(value)}
                </span>
                <button
                  onClick={() => handleRemoveVariant(key)}
                  className="p-0.5 text-slate-300 hover:text-red-500 transition-colors shrink-0"
                  title="Remove variant"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
          {/* Add new variant */}
          <div className="flex items-center gap-1.5 mt-2">
            <input
              type="text"
              placeholder="key"
              value={newVariantKey}
              onChange={(e) => setNewVariantKey(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAddVariant();
              }}
              className="flex-1 text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 focus:outline-none focus:border-emerald-300 min-w-0"
            />
            <input
              type="text"
              placeholder="value"
              value={newVariantValue}
              onChange={(e) => setNewVariantValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAddVariant();
              }}
              className="flex-1 text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 focus:outline-none focus:border-emerald-300 min-w-0"
            />
            <button
              onClick={handleAddVariant}
              className="p-0.5 text-slate-300 hover:text-emerald-600 transition-colors shrink-0"
              title="Add variant"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Eval Run Name */}
      <div className="border-t border-slate-100 pt-3">
        <label className="text-[10px] uppercase tracking-[0.2em] text-slate-400 block mb-1.5">
          Eval Run Name
        </label>
        <input
          type="text"
          value={evalRunName}
          onChange={(e) => setEvalRunName(e.target.value)}
          placeholder={generateDefaultName(evalParams)}
          className="w-full text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1.5 focus:outline-none focus:border-emerald-300 focus:ring-1 focus:ring-emerald-200"
        />
        <div className="text-[10px] text-slate-400 mt-1 leading-relaxed">
          Name for this evaluation run. Leave blank to auto-generate.
        </div>
      </div>

      {/* Run Evaluation button */}
      <div className="pt-2">
        {evalError && (
          <div className="text-[10px] text-red-500 bg-red-50 rounded px-2 py-1.5 mb-2">
            {evalError}
          </div>
        )}
        <button
          onClick={handleRunEvaluation}
          disabled={evalRunning || !selectedTrainingRunId}
          className={
            'w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ' +
            (evalRunning
              ? 'bg-blue-50 text-blue-500 cursor-wait'
              : evalSuccess
                ? 'bg-emerald-50 text-emerald-600 border border-emerald-200'
                : !selectedTrainingRunId
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-sm')
          }
          title={
            !selectedTrainingRunId
              ? 'Select a training run first'
              : evalRunning
                ? 'Running evaluation...'
                : 'Run evaluation with current parameters'
          }
        >
          {evalRunning ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : evalSuccess ? (
            <CheckCircle2 className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          <span>
            {evalRunning
              ? 'Running...'
              : evalSuccess
                ? 'Evaluation Complete'
                : 'Run Evaluation'}
          </span>
        </button>
        {!selectedTrainingRunId && (
          <div className="text-[10px] text-slate-400 text-center mt-1.5">
            Select a training run to enable evaluation
          </div>
        )}
      </div>
    </div>
  );
}
