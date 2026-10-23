import type { TimeAggregationSpec, TimeAggregationMode, DiscountType } from '@/types/training';
import { TIME_AGG_LABELS, DISCOUNT_LABELS } from '@/types/training';
import clsx from 'clsx';
import { useCallback } from 'react';

interface TimeAggregationEditorProps {
  value: TimeAggregationSpec | undefined;
  onChange: (value: TimeAggregationSpec) => void;
  disabled?: boolean;
  className?: string;
}

const DEFAULT_TIME_AGG: TimeAggregationSpec = {
  mode: 'all',
};

export function TimeAggregationEditor({
  value,
  onChange,
  disabled,
  className,
}: TimeAggregationEditorProps) {
  const timeAgg = value ?? DEFAULT_TIME_AGG;

  const handleModeChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const mode = event.target.value as TimeAggregationMode;
      const newValue: TimeAggregationSpec = { ...timeAgg, mode };
      // Clear mode-specific fields when changing mode
      if (mode !== 'range') {
        delete newValue.start;
        delete newValue.end;
      }
      if (mode !== 'segment') {
        delete newValue.segment_name;
      }
      if (mode !== 'custom') {
        delete newValue.time_idxs;
      }
      onChange(newValue);
    },
    [timeAgg, onChange]
  );

  const handleStartChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const start = parseInt(event.target.value, 10);
      onChange({ ...timeAgg, start: Number.isNaN(start) ? undefined : start });
    },
    [timeAgg, onChange]
  );

  const handleEndChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const end = parseInt(event.target.value, 10);
      onChange({ ...timeAgg, end: Number.isNaN(end) ? undefined : end });
    },
    [timeAgg, onChange]
  );

  const handleSegmentChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onChange({ ...timeAgg, segment_name: event.target.value || undefined });
    },
    [timeAgg, onChange]
  );

  const handleCustomIndicesChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const indices = event.target.value
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !Number.isNaN(n));
      onChange({ ...timeAgg, time_idxs: indices.length > 0 ? indices : undefined });
    },
    [timeAgg, onChange]
  );

  const handleDiscountChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const discount = event.target.value as DiscountType;
      const newValue: TimeAggregationSpec = { ...timeAgg, discount };
      if (discount === 'none') {
        delete newValue.discount;
        delete newValue.discount_exp;
      }
      if (discount !== 'power') {
        delete newValue.discount_exp;
      }
      onChange(newValue);
    },
    [timeAgg, onChange]
  );

  const handleDiscountExpChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const exp = parseFloat(event.target.value);
      onChange({ ...timeAgg, discount_exp: Number.isNaN(exp) ? undefined : exp });
    },
    [timeAgg, onChange]
  );

  const inputClass = clsx(
    'rounded-md border border-slate-200 px-2 py-1 text-xs',
    'focus:border-brand-300 focus:ring-1 focus:ring-brand-200',
    disabled && 'opacity-50 cursor-not-allowed'
  );

  const selectClass = clsx(
    'rounded-md border border-slate-200 bg-white px-2 py-1 text-xs',
    'focus:border-brand-300 focus:ring-1 focus:ring-brand-200',
    disabled && 'opacity-50 cursor-not-allowed'
  );

  return (
    <div className={clsx('space-y-2', className)}>
      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-500 w-16">Mode</label>
        <select
          value={timeAgg.mode}
          onChange={handleModeChange}
          disabled={disabled}
          className={selectClass}
        >
          {Object.entries(TIME_AGG_LABELS).map(([mode, label]) => (
            <option key={mode} value={mode}>
              {label}
            </option>
          ))}
        </select>
      </div>

      {timeAgg.mode === 'range' && (
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 w-16">Range</label>
          <input
            type="number"
            min={0}
            placeholder="Start"
            value={timeAgg.start ?? ''}
            onChange={handleStartChange}
            disabled={disabled}
            className={clsx(inputClass, 'w-16')}
          />
          <span className="text-xs text-slate-400">to</span>
          <input
            type="number"
            min={0}
            placeholder="End"
            value={timeAgg.end ?? ''}
            onChange={handleEndChange}
            disabled={disabled}
            className={clsx(inputClass, 'w-16')}
          />
        </div>
      )}

      {timeAgg.mode === 'segment' && (
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 w-16">Segment</label>
          <input
            type="text"
            placeholder="e.g., movement"
            value={timeAgg.segment_name ?? ''}
            onChange={handleSegmentChange}
            disabled={disabled}
            className={clsx(inputClass, 'flex-1')}
          />
        </div>
      )}

      {timeAgg.mode === 'custom' && (
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 w-16">Indices</label>
          <input
            type="text"
            placeholder="e.g., 0, 10, 50, 100"
            value={timeAgg.time_idxs?.join(', ') ?? ''}
            onChange={handleCustomIndicesChange}
            disabled={disabled}
            className={clsx(inputClass, 'flex-1')}
          />
        </div>
      )}

      <div className="flex items-center gap-2">
        <label className="text-xs text-slate-500 w-16">Discount</label>
        <select
          value={timeAgg.discount ?? 'none'}
          onChange={handleDiscountChange}
          disabled={disabled}
          className={selectClass}
        >
          {Object.entries(DISCOUNT_LABELS).map(([type, label]) => (
            <option key={type} value={type}>
              {label}
            </option>
          ))}
        </select>
        {timeAgg.discount === 'power' && (
          <>
            <label className="text-xs text-slate-500">exp</label>
            <input
              type="number"
              min={0}
              step={0.5}
              value={timeAgg.discount_exp ?? ''}
              onChange={handleDiscountExpChange}
              disabled={disabled}
              className={clsx(inputClass, 'w-14')}
            />
          </>
        )}
      </div>
    </div>
  );
}
