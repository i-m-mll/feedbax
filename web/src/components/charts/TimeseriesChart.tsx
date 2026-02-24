import { useMemo } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { TimeseriesResponse } from '@/types/statistics';

const TASK_COLORS: Record<number, string> = {
  0: '#2f7cf6',
  1: '#2fbf7f',
  2: '#f2b92d',
  3: '#9b59b6',
};

const GENERIC_PALETTE = [
  '#2f7cf6', '#2fbf7f', '#f2b92d', '#9b59b6',
  '#e74c3c', '#1abc9c', '#f39c12', '#3498db',
  '#e67e22', '#27ae60', '#8e44ad', '#c0392b',
];

function colorForIndex(idx: number, total: number): string {
  if (total <= 4) return Object.values(TASK_COLORS)[idx] ?? GENERIC_PALETTE[idx % GENERIC_PALETTE.length];
  return GENERIC_PALETTE[idx % GENERIC_PALETTE.length];
}

export function TimeseriesChart({ data }: { data: TimeseriesResponse }) {
  // Transform series data into row-based format for Recharts.
  // Each row: { t: number, <group>_p50, <group>_p25, ..., <group>_p95 }
  const { rows, seriesKeys } = useMemo(() => {
    if (!data.series.length) return { rows: [], seriesKeys: [] };

    const keys = data.series.map((s) => s.group_key);
    const timesteps = data.series[0].timesteps;

    const rows = timesteps.map((t, ti) => {
      const row: Record<string, number> = { t };
      for (const s of data.series) {
        row[`${s.group_key}_p50`] = s.p50[ti];
        row[`${s.group_key}_p25`] = s.p25[ti];
        row[`${s.group_key}_p75`] = s.p75[ti];
        row[`${s.group_key}_p05`] = s.p05[ti];
        row[`${s.group_key}_p95`] = s.p95[ti];
      }
      return row;
    });

    return { rows, seriesKeys: keys };
  }, [data]);

  if (!rows.length) {
    return <div className="text-sm text-slate-400 p-4">No timeseries data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <XAxis
          dataKey="t"
          tick={{ fontSize: 10, fill: '#94a3b8' }}
          tickLine={false}
          axisLine={{ stroke: '#e2e8f0' }}
          label={{ value: 'Time step', position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#94a3b8' }}
        />
        <YAxis
          tick={{ fontSize: 10, fill: '#94a3b8' }}
          tickLine={false}
          axisLine={{ stroke: '#e2e8f0' }}
        />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #e2e8f0' }}
        />
        {seriesKeys.length > 1 && (
          <Legend
            wrapperStyle={{ fontSize: 10 }}
            formatter={(value: string) => {
              // Show only the group label for the p50 line
              const s = data.series.find((ser) => `${ser.group_key}_p50` === value);
              return s?.group_label ?? value;
            }}
          />
        )}

        {data.series.map((s, idx) => {
          const color = colorForIndex(idx, data.series.length);
          return (
            <g key={s.group_key}>
              {/* p05-p95 band (lightest) */}
              <Area
                dataKey={`${s.group_key}_p95`}
                stroke="none"
                fill={color}
                fillOpacity={0.08}
                isAnimationActive={false}
                legendType="none"
              />
              <Area
                dataKey={`${s.group_key}_p05`}
                stroke="none"
                fill="#fff"
                fillOpacity={1}
                isAnimationActive={false}
                legendType="none"
              />
              {/* p25-p75 band (darker) */}
              <Area
                dataKey={`${s.group_key}_p75`}
                stroke="none"
                fill={color}
                fillOpacity={0.18}
                isAnimationActive={false}
                legendType="none"
              />
              <Area
                dataKey={`${s.group_key}_p25`}
                stroke="none"
                fill="#fff"
                fillOpacity={1}
                isAnimationActive={false}
                legendType="none"
              />
            </g>
          );
        })}

        {/* Median lines (on top) */}
        {data.series.map((s, idx) => {
          const color = colorForIndex(idx, data.series.length);
          return (
            <Line
              key={`${s.group_key}_line`}
              dataKey={`${s.group_key}_p50`}
              stroke={color}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
              name={`${s.group_key}_p50`}
            />
          );
        })}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
