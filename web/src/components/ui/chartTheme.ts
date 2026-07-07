export const TASK_COLORS: Record<number, string> = {
  0: '#2f7cf6',
  1: '#2fbf7f',
  2: '#f2b92d',
  3: '#9b59b6',
};

export const TASK_TYPE_NAMES: Record<number, string> = {
  0: 'Reach',
  1: 'Hold',
  2: 'Track',
  3: 'Swing',
};

export const CHART_PALETTE = [
  '#2f7cf6',
  '#2fbf7f',
  '#f2b92d',
  '#9b59b6',
  '#e74c3c',
  '#1abc9c',
  '#f39c12',
  '#3498db',
  '#e67e22',
  '#27ae60',
  '#8e44ad',
  '#c0392b',
];

export const chartAxisTick = { fontSize: 10, fill: '#94a3b8' };
export const chartAxisLine = { stroke: '#e2e8f0' };
export const chartTooltipContentStyle = {
  fontSize: 11,
  borderRadius: 8,
  border: '1px solid #e2e8f0',
};
export const chartLegendStyle = { fontSize: 10 };

export function chartColorForIndex(index: number, total = CHART_PALETTE.length): string {
  if (total <= 4) {
    return Object.values(TASK_COLORS)[index] ?? CHART_PALETTE[index % CHART_PALETTE.length];
  }
  return CHART_PALETTE[index % CHART_PALETTE.length];
}

export function chartColorForTaskType(taskType: number): string {
  return TASK_COLORS[taskType] ?? CHART_PALETTE[taskType % CHART_PALETTE.length];
}

