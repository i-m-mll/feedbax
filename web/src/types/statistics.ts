export type {
  DiagnosticCheck,
  DiagnosticsResponse,
  GroupStatistics,
  HistogramBin,
  HistogramGroup,
  HistogramResponse,
  MetricSummary,
  ScatterPoint,
  ScatterResponse,
  StatisticsResponse,
  TimeseriesPercentiles,
  TimeseriesResponse,
} from '@/generated/studioContracts';

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
