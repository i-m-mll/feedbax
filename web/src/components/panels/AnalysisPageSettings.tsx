/**
 * AnalysisPageSettings — right sidebar content when no analysis node is selected.
 *
 * Shows page-level settings: name, eval run selector, and eval parametrization
 * fields (perturbation type, amplitudes, SISU values, task variants).
 */

import { useCallback, useState } from 'react';
import { useAnalysisStore } from '@/stores/analysisStore';
import { EvalRunSelector } from '@/components/panels/RunSelector';
import { Plus, Trash2 } from 'lucide-react';

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

export function AnalysisPageSettings() {
  const activePageId = useAnalysisStore((s) => s.activePageId);
  const pages = useAnalysisStore((s) => s.pages);
  const renamePage = useAnalysisStore((s) => s.renamePage);
  const evalParams = useAnalysisStore((s) => s.evalParams);
  const setEvalParams = useAnalysisStore((s) => s.setEvalParams);
  const evalRunId = useAnalysisStore((s) => s.evalRunId);
  const setEvalRunId = useAnalysisStore((s) => s.setEvalRunId);

  const activePage = pages.find((p) => p.id === activePageId);

  // Local state for the task variants key-value editor
  const [newVariantKey, setNewVariantKey] = useState('');
  const [newVariantValue, setNewVariantValue] = useState('');

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
    </div>
  );
}
