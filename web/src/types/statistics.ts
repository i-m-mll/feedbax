export interface MetricSummary {
  mean: number;
  std: number;
  median: number;
  q25: number;
  q75: number;
  min: number;
  max: number;
  count: number;
}

export interface GroupStatistics {
  group_key: string;
  group_label: string;
  metrics: Record<string, MetricSummary>;
}

export interface StatisticsResponse {
  dataset: string;
  group_by: string;
  groups: GroupStatistics[];
}

export interface TimeseriesPercentiles {
  group_key: string;
  group_label: string;
  timesteps: number[];
  p50: number[];
  p25: number[];
  p75: number[];
  p05: number[];
  p95: number[];
}

export interface TimeseriesResponse {
  dataset: string;
  metric: string;
  group_by: string;
  series: TimeseriesPercentiles[];
}

export interface HistogramBin {
  lo: number;
  hi: number;
  count: number;
}

export interface HistogramGroup {
  group_key: string;
  group_label: string;
  bins: HistogramBin[];
}

export interface HistogramResponse {
  dataset: string;
  metric: string;
  group_by: string;
  groups: HistogramGroup[];
}

export interface ScatterPoint {
  x: number;
  y: number;
  body_idx: number;
  task_type: number;
}

export interface ScatterResponse {
  dataset: string;
  x_metric: string;
  y_metric: string;
  points: ScatterPoint[];
}

export interface DiagnosticCheck {
  name: string;
  status: 'pass' | 'warn' | 'fail';
  reason: string;
  evidence: Record<string, unknown>;
  hint?: string;
}

export interface DiagnosticsResponse {
  dataset: string;
  checks: DiagnosticCheck[];
}

export const METRIC_LABELS: Record<string, string> = {
  final_distance: 'Final Distance',
  effort: 'Muscle Effort',
  convergence_time: 'Convergence Time',
  joint_range_of_motion: 'Joint ROM',
  peak_activation: 'Peak Activation',
  movement_amplitude: 'Movement Amplitude',
  success_rate: 'Success Rate',
};

export const GROUP_BY_OPTIONS = [
  { value: 'none', label: 'All' },
  { value: 'task_type', label: 'Task' },
  { value: 'body_idx', label: 'Body' },
  { value: 'body_x_task', label: 'Body\u00d7Task' },
] as const;
