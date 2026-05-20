import type { ScenarioMetricSpec } from '@/features/scenario/integration';
import {
  formatMetric,
  trainingRunMetricValue,
  type TrainingRunSummary,
} from '@/utils/pipelineCollections';

export interface MetricColumnSpec {
  id: string;
  label: string;
  units: string | null;
  source: ScenarioMetricSpec['source'];
  summary: string | null;
  metadata: Record<string, unknown>;
}

export function runMetricColumns(
  metrics: ScenarioMetricSpec[],
  rows: TrainingRunSummary[]
): MetricColumnSpec[] {
  const byId = new Map<string, MetricColumnSpec>();
  for (const metric of metrics) {
    if (!rows.some((row) => trainingRunMetricValue(row, metric.id) !== null)) continue;
    const existing = byId.get(metric.id);
    if (existing && sourceRank(existing.source) <= sourceRank(metric.source)) continue;
    byId.set(metric.id, {
      id: metric.id,
      label: compactMetricLabel(metric.label),
      units: metric.units,
      source: metric.source,
      summary: metric.summary,
      metadata: {
        ...metric.metadata,
        value_schema: metric.valueSchema ?? metric.metadata.value_schema ?? null,
      },
    });
  }

  return Array.from(byId.values())
    .sort((a, b) => {
      const priority = metricPriority(a.id) - metricPriority(b.id);
      return priority || sourceRank(a.source) - sourceRank(b.source) || a.label.localeCompare(b.label);
    })
    .slice(0, 4);
}

export function formatMetricWithUnits(value: number | null, units: string | null): string {
  const formatted = formatMetric(value, 3);
  if (formatted === 'Not recorded' || !units) return formatted;
  return `${formatted} ${units}`;
}

function compactMetricLabel(label: string): string {
  return label
    .replace(/^Final validation loss$/, 'Loss')
    .replace(/^Within-cell velocity RMSE$/, 'Velocity RMSE');
}

function sourceRank(source: ScenarioMetricSpec['source']): number {
  return ['objective', 'analysis', 'task_default', 'manifest'].indexOf(source);
}

function metricPriority(metricId: string): number {
  const order = [
    'final_validation_loss',
    'within_cell_velocity_rmse_m_per_s',
    'peak_velocity_m_per_s',
    'hold_drift_mm',
  ];
  const index = order.indexOf(metricId);
  return index === -1 ? order.length : index;
}
