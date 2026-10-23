import type { TrainingRunCompareResponse } from '@/types/runs';
import type { TrainingRunSummary } from '@/utils/pipelineCollections';
import type { MetricColumnSpec } from '@/utils/runMetricColumns';
import type { TrainAxisColumn } from '@/utils/trainMatrix';

export type CompareFieldKind = 'param' | 'metric';

export interface CompareFieldSpec {
  id: string;
  label: string;
  kind: CompareFieldKind;
}

export interface CompareFieldDiff {
  id: string;
  label: string;
  kind: CompareFieldKind;
  values: Record<string, unknown>;
  identical: boolean;
}

export interface TrainingRunComparison {
  runIds: string[];
  paramFields: CompareFieldDiff[];
  metricFields: CompareFieldDiff[];
}

const BASE_PARAM_FIELDS: CompareFieldSpec[] = [
  { id: 'ramp_shape', label: 'Ramp shape', kind: 'param' },
  { id: 'ramp_duration_steps', label: 'Ramp steps', kind: 'param' },
  { id: 'nn_output_pre_go', label: 'Pre-go output', kind: 'param' },
  { id: 'n_replicates', label: 'Replicates', kind: 'param' },
  { id: 'batch_size', label: 'Batch size', kind: 'param' },
  { id: 'n_warmup_batches', label: 'Warmup batches', kind: 'param' },
  { id: 'learning_rate', label: 'Learning rate', kind: 'param' },
];

export function trainingCompareFields(
  axisColumns: TrainAxisColumn[],
  metricColumns: MetricColumnSpec[],
): { params: CompareFieldSpec[]; metrics: CompareFieldSpec[] } {
  const params = dedupeFields([
    ...axisColumns.map((axis) => ({ id: axis.id, label: axis.label, kind: 'param' as const })),
    ...BASE_PARAM_FIELDS,
  ]);
  const metrics = dedupeFields(
    metricColumns.map((column) => ({
      id: column.id,
      label: column.units ? `${column.label} (${column.units})` : column.label,
      kind: 'metric' as const,
    })),
  );
  return { params, metrics };
}

export function buildTrainingRunComparison(
  rows: TrainingRunSummary[],
  fields: { params: CompareFieldSpec[]; metrics: CompareFieldSpec[] },
  fetched?: TrainingRunCompareResponse | null,
): TrainingRunComparison {
  const fetchedById = new Map((fetched?.rows ?? []).map((row) => [row.id, row]));
  const runIds = rows.map((row) => row.id);
  return {
    runIds,
    paramFields: fields.params.map((field) =>
      diffField(rows, field, (row) => {
        const fetchedValue = fetchedById.get(row.id)?.params[field.id];
        return fetchedValue !== undefined ? fetchedValue : localParamValue(row, field.id);
      }),
    ),
    metricFields: fields.metrics.map((field) =>
      diffField(rows, field, (row) => {
        const fetchedValue = fetchedById.get(row.id)?.metrics[field.id];
        return fetchedValue !== undefined ? fetchedValue : row.metrics[field.id] ?? null;
      }),
    ),
  };
}

export function visibleCompareFields(
  fields: CompareFieldDiff[],
  showIdentical: boolean,
): CompareFieldDiff[] {
  return showIdentical ? fields : fields.filter((field) => !field.identical);
}

function diffField(
  rows: TrainingRunSummary[],
  field: CompareFieldSpec,
  valueForRow: (row: TrainingRunSummary) => unknown,
): CompareFieldDiff {
  const values = Object.fromEntries(rows.map((row) => [row.id, valueForRow(row)]));
  const unique = new Set(Object.values(values).map(stableValueKey));
  return {
    ...field,
    values,
    identical: unique.size <= 1,
  };
}

function localParamValue(row: TrainingRunSummary, id: string): unknown {
  if (row.axisCoordinates[id] !== undefined) return row.axisCoordinates[id];
  if (id === 'ramp_shape') return row.rampShape;
  if (id === 'ramp_duration_steps') return row.rampDurationSteps;
  if (id === 'nn_output_pre_go') return row.nnOutputPreGo;
  if (id === 'n_replicates') return row.replicateCount;
  if (id === 'batch_size') return row.batchSize;
  if (id === 'n_warmup_batches') return row.warmupBatches;
  return null;
}

function dedupeFields(fields: CompareFieldSpec[]): CompareFieldSpec[] {
  const byId = new Map<string, CompareFieldSpec>();
  for (const field of fields) {
    if (!byId.has(field.id)) byId.set(field.id, field);
  }
  return Array.from(byId.values());
}

function stableValueKey(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'null';
  if (typeof value === 'number' && !Number.isFinite(value)) return 'null';
  return JSON.stringify(value);
}
