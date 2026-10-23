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
import {
  TASK_TYPE_NAMES,
  chartAxisLine,
  chartAxisTick,
  chartColorForTaskType,
  chartLegendStyle,
  chartTooltipContentStyle,
} from '@/components/ui/chartTheme';

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
          tick={chartAxisTick}
          tickLine={false}
          axisLine={chartAxisLine}
          label={{ value: xLabel, position: 'insideBottomRight', offset: -4, fontSize: 10, fill: '#94a3b8' }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tick={chartAxisTick}
          tickLine={false}
          axisLine={chartAxisLine}
          label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 0, fontSize: 10, fill: '#94a3b8' }}
        />
        <ZAxis range={[20, 20]} />
        <Tooltip
          contentStyle={chartTooltipContentStyle}
          cursor={{ strokeDasharray: '3 3' }}
        />
        {groups.length > 1 && (
          <Legend wrapperStyle={chartLegendStyle} />
        )}
        {groups.map(({ taskType, points }) => (
          <Scatter
            key={taskType}
            name={TASK_TYPE_NAMES[taskType] ?? `Task ${taskType}`}
            data={points}
            fill={chartColorForTaskType(taskType)}
            fillOpacity={0.6}
            isAnimationActive={false}
          />
        ))}
      </RechartsScatterChart>
    </ResponsiveContainer>
  );
}
