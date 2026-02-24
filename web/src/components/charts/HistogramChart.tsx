import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { HistogramResponse } from '@/types/statistics';

const GENERIC_PALETTE = [
  '#2f7cf6', '#2fbf7f', '#f2b92d', '#9b59b6',
  '#e74c3c', '#1abc9c', '#f39c12', '#3498db',
  '#e67e22', '#27ae60', '#8e44ad', '#c0392b',
];

export function HistogramChart({ data }: { data: HistogramResponse }) {
  // Build row-based data: each row = one bin range, columns = per-group counts.
  const { rows, groupKeys, groupLabels } = useMemo(() => {
    if (!data.groups.length) return { rows: [], groupKeys: [], groupLabels: {} as Record<string, string> };

    const keys = data.groups.map((g) => g.group_key);
    const labels: Record<string, string> = {};
    for (const g of data.groups) labels[g.group_key] = g.group_label;

    // Use the first group's bins as the canonical bin edges
    const bins = data.groups[0].bins;

    const rows = bins.map((bin, bi) => {
      const mid = (bin.lo + bin.hi) / 2;
      const label = `${bin.lo.toFixed(2)}\u2013${bin.hi.toFixed(2)}`;
      const row: Record<string, number | string> = { bin: label, mid };
      for (const g of data.groups) {
        row[g.group_key] = g.bins[bi]?.count ?? 0;
      }
      return row;
    });

    return { rows, groupKeys: keys, groupLabels: labels };
  }, [data]);

  if (!rows.length) {
    return <div className="text-sm text-slate-400 p-4">No histogram data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <XAxis
          dataKey="bin"
          tick={{ fontSize: 9, fill: '#94a3b8' }}
          tickLine={false}
          axisLine={{ stroke: '#e2e8f0' }}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#94a3b8' }}
          tickLine={false}
          axisLine={{ stroke: '#e2e8f0' }}
          label={{ value: 'Count', angle: -90, position: 'insideLeft', offset: 0, fontSize: 10, fill: '#94a3b8' }}
        />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #e2e8f0' }}
        />
        {groupKeys.length > 1 && (
          <Legend wrapperStyle={{ fontSize: 10 }} />
        )}
        {groupKeys.map((key, idx) => (
          <Bar
            key={key}
            dataKey={key}
            name={groupLabels[key] ?? key}
            fill={GENERIC_PALETTE[idx % GENERIC_PALETTE.length]}
            fillOpacity={0.7}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
