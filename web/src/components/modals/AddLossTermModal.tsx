import { useTrainingStore } from '@/stores/trainingStore';
import type { LossTermSpec, NormFunction } from '@/types/training';
import { NORM_LABELS } from '@/types/training';
import { ProbeSelector } from '@/components/panels/ProbeSelector';
import clsx from 'clsx';
import { useCallback, useState } from 'react';
import { X } from 'lucide-react';

interface AddLossTermModalProps {
  parentPath: string[];
  onClose: () => void;
}

export function AddLossTermModal({ parentPath, onClose }: AddLossTermModalProps) {
  const addLossTerm = useTrainingStore((state) => state.addLossTerm);

  const [key, setKey] = useState('');
  const [label, setLabel] = useState('');
  const [type, setType] = useState<string>('TargetStateLoss');
  const [weight, setWeight] = useState(1.0);
  const [selector, setSelector] = useState('');
  const [norm, setNorm] = useState<NormFunction | ''>('squared_l2');

  const [error, setError] = useState<string | null>(null);

  const handleSubmit = useCallback(
    (event: React.FormEvent) => {
      event.preventDefault();
      setError(null);

      // Validate key
      const keyTrimmed = key.trim();
      if (!keyTrimmed) {
        setError('Key is required');
        return;
      }
      if (!/^[a-z][a-z0-9_]*$/i.test(keyTrimmed)) {
        setError('Key must be a valid identifier (letters, numbers, underscores)');
        return;
      }

      // Validate label
      const labelTrimmed = label.trim();
      if (!labelTrimmed) {
        setError('Label is required');
        return;
      }

      // Build the new term
      const newTerm: LossTermSpec = {
        type,
        label: labelTrimmed,
        weight,
      };

      if (selector) {
        newTerm.selector = selector;
      }

      if (norm) {
        newTerm.norm = norm;
      }

      // Add default time aggregation
      newTerm.time_agg = {
        mode: 'all',
      };

      addLossTerm(parentPath, keyTrimmed, newTerm);
      onClose();
    },
    [key, label, type, weight, selector, norm, parentPath, addLossTerm, onClose]
  );

  const inputClass = clsx(
    'rounded-md border border-slate-200 px-3 py-2 text-sm w-full',
    'focus:border-brand-300 focus:ring-1 focus:ring-brand-200'
  );

  const selectClass = clsx(
    'rounded-md border border-slate-200 bg-white px-3 py-2 text-sm w-full',
    'focus:border-brand-300 focus:ring-1 focus:ring-brand-200'
  );

  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <h2 className="text-lg font-semibold text-slate-800">Add Loss Term</h2>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm text-slate-600 mb-1">
              Key <span className="text-slate-400">(identifier)</span>
            </label>
            <input
              type="text"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="e.g., position_error"
              className={inputClass}
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm text-slate-600 mb-1">Label</label>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g., Position Error"
              className={inputClass}
            />
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-sm text-slate-600 mb-1">Type</label>
              <select
                value={type}
                onChange={(e) => setType(e.target.value)}
                className={selectClass}
              >
                <option value="TargetStateLoss">Target State</option>
                <option value="EffortLoss">Effort</option>
                <option value="RegularizationLoss">Regularization</option>
                <option value="Composite">Composite</option>
              </select>
            </div>
            <div className="w-28">
              <label className="block text-sm text-slate-600 mb-1">Weight</label>
              <input
                type="number"
                min={0}
                step={0.01}
                value={weight}
                onChange={(e) => setWeight(parseFloat(e.target.value) || 0)}
                className={inputClass}
              />
            </div>
          </div>

          {type !== 'Composite' && (
            <>
              <div>
                <label className="block text-sm text-slate-600 mb-1">Probe</label>
                <ProbeSelector value={selector} onChange={setSelector} />
              </div>

              <div>
                <label className="block text-sm text-slate-600 mb-1">Norm Function</label>
                <select
                  value={norm}
                  onChange={(e) => setNorm(e.target.value as NormFunction | '')}
                  className={selectClass}
                >
                  <option value="">Default</option>
                  {Object.entries(NORM_LABELS).map(([n, l]) => (
                    <option key={n} value={n}>
                      {l}
                    </option>
                  ))}
                </select>
              </div>
            </>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-600 hover:bg-slate-100 rounded-lg"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm text-white bg-brand-500 hover:bg-brand-600 rounded-lg font-medium"
            >
              Add Term
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
