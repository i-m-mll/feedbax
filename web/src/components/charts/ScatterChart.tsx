import { useMemo } from 'react';
import {
  ScatterChart as RechartsScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ZAxis,
} from 'recharts';
import type { ScatterResponse } from '@/types/statistics';
import { METRIC_LABELS } from '@/types/statistics';

const TASK_COLORS: Record<number, string> = {
  0: '#2f7cf6',
  1: '#2fbf7f',
  2: '#f2b92d',
  3: '#9b59b6',
};

const TASK_TYPE_NAMES: Record<number, string> = {
  0: 'Reach',
  1: 'Hold',
  2: 'Track',
  3: 'Swing',
};

const GENERIC_PALETTE = [
  '#2f7cf6', '#2fbf7f', '#f2b92d', '#9b59b6',
  '#e74c3c', '#1abc9c', '#f39c12', '#3498db',
];

export function ScatterPlotChart({ data }: { data: ScatterResponse }) {
  // Group points by task_type
  const groups = useMemo(() => {
    const map = new Map<number, { x: number; y: number }[]>();
    for (const p of data.points) {
      if (!map.has(p.task_type)) map.set(p.task_type, []);
      map.get(p.task_type)!.push({ x: p.x, y: p.y });
    }
    return Array.from(map.entries())
      .sort(([a], [b]) => a - b)
      .map(([taskType, points]) => ({ taskType, points }));
  }, [data]);

  const xLabel = METRIC_LABELS[data.x_metric] ?? data.x_metric;
  const yLabel = METRIC_LABELS[data.y_metric] ?? data.y_metric;

  if (!data.points.length) {
    return <div className="text-sm text-slate-400 p-4">No scatter data</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RechartsScatterChart margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          tick={{ fontSize: 10, fill: '#94a3b8' }}
          tickLine={false}
          axisLine={{ stroke: '#e2e8f0' }}
          label={{ value: xLabel, position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#94a3b8' }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tick={{ fontSize: 10, fill: '#94a3b8' }}
          tickLine={false}
          axisLine={{ stroke: '#e2e8f0' }}
          label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 0, fontSize: 10, fill: '#94a3b8' }}
        />
        <ZAxis range={[20, 20]} />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #e2e8f0' }}
          cursor={{ strokeDasharray: '3 3' }}
        />
        {groups.length > 1 && (
          <Legend wrapperStyle={{ fontSize: 10 }} />
        )}
        {groups.map(({ taskType, points }) => (
          <Scatter
            key={taskType}
            name={TASK_TYPE_NAMES[taskType] ?? `Task ${taskType}`}
            data={points}
            fill={TASK_COLORS[taskType] ?? GENERIC_PALETTE[taskType % GENERIC_PALETTE.length]}
            fillOpacity={0.6}
            isAnimationActive={false}
          />
        ))}
      </RechartsScatterChart>
    </ResponsiveContainer>
  );
}
