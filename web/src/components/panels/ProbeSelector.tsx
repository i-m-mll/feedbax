import { useTrainingStore } from '@/stores/trainingStore';
import type { ProbeInfo } from '@/types/training';
import clsx from 'clsx';
import { useCallback, useMemo } from 'react';

interface ProbeSelectorProps {
  value: string | undefined;
  onChange: (selector: string) => void;
  disabled?: boolean;
  className?: string;
}

export function ProbeSelector({ value, onChange, disabled, className }: ProbeSelectorProps) {
  const availableProbes = useTrainingStore((state) => state.availableProbes);
  const setHighlightedProbeSelector = useTrainingStore(
    (state) => state.setHighlightedProbeSelector
  );

  // Group probes by node
  const groupedProbes = useMemo(() => {
    const groups: Record<string, ProbeInfo[]> = {};
    for (const probe of availableProbes) {
      if (!groups[probe.node]) {
        groups[probe.node] = [];
      }
      groups[probe.node].push(probe);
    }
    return groups;
  }, [availableProbes]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      onChange(event.target.value);
    },
    [onChange]
  );

  const handleMouseEnter = useCallback(() => {
    if (value) {
      setHighlightedProbeSelector(value);
    }
  }, [value, setHighlightedProbeSelector]);

  const handleMouseLeave = useCallback(() => {
    setHighlightedProbeSelector(null);
  }, [setHighlightedProbeSelector]);

  const selectedProbe = useMemo(
    () => availableProbes.find((p) => p.selector === value),
    [availableProbes, value]
  );

  return (
    <div
      className={clsx('relative', className)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <select
        value={value ?? ''}
        onChange={handleChange}
        disabled={disabled}
        className={clsx(
          'w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm',
          'focus:border-brand-300 focus:ring-1 focus:ring-brand-200',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <option value="">Select a probe...</option>
        {Object.entries(groupedProbes).map(([node, probes]) => (
          <optgroup key={node} label={node}>
            {probes.map((probe) => (
              <option key={probe.selector} value={probe.selector}>
                {probe.label} ({probe.timing})
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      {selectedProbe?.description && (
        <div className="mt-1 text-xs text-slate-400">{selectedProbe.description}</div>
      )}
    </div>
  );
}
