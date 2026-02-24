import { useTrainingStore } from '@/stores/trainingStore';
import type { LossTermSpec, NormFunction, TimeAggregationSpec } from '@/types/training';
import { NORM_LABELS } from '@/types/training';
import { ProbeSelector } from './ProbeSelector';
import { TimeAggregationEditor } from './TimeAggregationEditor';
import clsx from 'clsx';
import { useCallback, useMemo } from 'react';
import { Trash2, X } from 'lucide-react';

interface LossTermDetailProps {
  path: string[];
  onClose: () => void;
}

export function LossTermDetail({ path, onClose }: LossTermDetailProps) {
  const trainingSpec = useTrainingStore((state) => state.trainingSpec);
  const updateLossTerm = useTrainingStore((state) => state.updateLossTerm);
  const removeLossTerm = useTrainingStore((state) => state.removeLossTerm);
  const lossValidationErrors = useTrainingStore((state) => state.lossValidationErrors);

  // Get the term at the current path
  const term = useMemo(() => {
    let current: LossTermSpec | undefined = trainingSpec.loss;
    for (const key of path) {
      current = current?.children?.[key];
      if (!current) return null;
    }
    return current;
  }, [trainingSpec.loss, path]);

  // Get validation errors for this term
  const termErrors = useMemo(() => {
    const pathStr = path.join('/');
    return lossValidationErrors.filter(
      (e) => e.path.join('/') === pathStr || e.path.join('/').startsWith(pathStr + '/')
    );
  }, [lossValidationErrors, path]);

  const handleLabelChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      updateLossTerm(path, { label: event.target.value });
    },
    [path, updateLossTerm]
  );

  const handleTypeChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      updateLossTerm(path, { type: event.target.value });
    },
    [path, updateLossTerm]
  );

  const handleWeightChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const weight = parseFloat(event.target.value);
      if (!Number.isNaN(weight)) {
        updateLossTerm(path, { weight });
      }
    },
    [path, updateLossTerm]
  );

  const handleSelectorChange = useCallback(
    (selector: string) => {
      updateLossTerm(path, { selector: selector || undefined });
    },
    [path, updateLossTerm]
  );

  const handleNormChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const norm = event.target.value as NormFunction | '';
      updateLossTerm(path, { norm: norm || undefined });
    },
    [path, updateLossTerm]
  );

  const handleTimeAggChange = useCallback(
    (timeAgg: TimeAggregationSpec) => {
      updateLossTerm(path, { time_agg: timeAgg });
    },
    [path, updateLossTerm]
  );

  const handleDelete = useCallback(() => {
    if (path.length === 0) {
      // Cannot delete root
      return;
    }
    if (confirm('Are you sure you want to remove this loss term?')) {
      removeLossTerm(path);
      onClose();
    }
  }, [path, removeLossTerm, onClose]);

  if (!term) {
    return (
      <div className="p-4 text-sm text-slate-500">
        Loss term not found at path: {path.join(' > ')}
      </div>
    );
  }

  const isRoot = path.length === 0;
  const hasChildren = term.children && Object.keys(term.children).length > 0;

  const inputClass = clsx(
    'rounded-md border border-slate-200 px-2 py-1.5 text-sm',
    'focus:border-brand-300 focus:ring-1 focus:ring-brand-200'
  );

  const selectClass = clsx(
    'rounded-md border border-slate-200 bg-white px-2 py-1.5 text-sm',
    'focus:border-brand-300 focus:ring-1 focus:ring-brand-200'
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-[0.2em] text-slate-400">
          {isRoot ? 'Root Loss' : 'Loss Term'}
        </div>
        <div className="flex items-center gap-1">
          {!isRoot && (
            <button
              type="button"
              onClick={handleDelete}
              className="p-1 rounded text-slate-400 hover:text-red-500 hover:bg-red-50"
              title="Remove this term"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Validation errors */}
      {termErrors.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-2 space-y-1">
          {termErrors.map((error, i) => (
            <div key={i} className="text-xs text-amber-700">
              {error.field}: {error.message}
            </div>
          ))}
        </div>
      )}

      {/* Basic fields */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs text-slate-500 mb-1">Label</label>
          <input
            type="text"
            value={term.label}
            onChange={handleLabelChange}
            className={clsx(inputClass, 'w-full')}
          />
        </div>

        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs text-slate-500 mb-1">Type</label>
            <select value={term.type} onChange={handleTypeChange} className={clsx(selectClass, 'w-full')}>
              <option value="Composite">Composite</option>
              <option value="TargetStateLoss">Target State</option>
              <option value="EffortLoss">Effort</option>
              <option value="RegularizationLoss">Regularization</option>
            </select>
          </div>
          <div className="w-24">
            <label className="block text-xs text-slate-500 mb-1">Weight</label>
            <input
              type="number"
              min={0}
              step={0.01}
              value={term.weight}
              onChange={handleWeightChange}
              className={clsx(inputClass, 'w-full')}
            />
          </div>
        </div>
      </div>

      {/* Probe selector (only for non-composite terms) */}
      {!hasChildren && (
        <div>
          <label className="block text-xs text-slate-500 mb-1">Probe</label>
          <ProbeSelector value={term.selector} onChange={handleSelectorChange} />
        </div>
      )}

      {/* Norm function (only for non-composite terms) */}
      {!hasChildren && (
        <div>
          <label className="block text-xs text-slate-500 mb-1">Norm Function</label>
          <select
            value={term.norm ?? ''}
            onChange={handleNormChange}
            className={clsx(selectClass, 'w-full')}
          >
            <option value="">Default (squared L2)</option>
            {Object.entries(NORM_LABELS).map(([norm, label]) => (
              <option key={norm} value={norm}>
                {label}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Time aggregation (only for non-composite terms) */}
      {!hasChildren && (
        <div>
          <label className="block text-xs text-slate-500 mb-2">Time Aggregation</label>
          <TimeAggregationEditor value={term.time_agg} onChange={handleTimeAggChange} />
        </div>
      )}

      {/* Children summary */}
      {hasChildren && (
        <div>
          <label className="block text-xs text-slate-500 mb-1">Child Terms</label>
          <div className="text-sm text-slate-600">
            {Object.keys(term.children!).length} child term(s)
          </div>
        </div>
      )}
    </div>
  );
}
