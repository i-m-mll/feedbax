import type { StudioManifestRef, StudioStageSpec } from '@/types/workspace';

export interface TrainingRunSummary {
  id: string;
  label: string;
  status: string;
  variant: string | null;
  rampShape: string | null;
  rampDurationSteps: number | null;
  nnOutputPreGo: number | null;
  finalValidationLoss: number | null;
  velocityRmse: number | null;
  peakVelocityMean: number | null;
  peakVelocitySd: number | null;
  holdDriftMeanMm: number | null;
  holdDriftSdMm: number | null;
  metrics: Record<string, number | null>;
  replicateCount: number | null;
  batchSize: number | null;
  warmupBatches: number | null;
  checkpointAvailable: boolean;
  sourceIssue: string | null;
  provenanceId: string;
  uri: string | null;
  jobId: string | null;
  axisCoordinates: Record<string, unknown>;
  runSetId: string | null;
  planned: boolean;
  supersededBy: string | null;
}

export interface EvaluationRunSummary {
  id: string;
  label: string;
  status: string;
  selectedTrainingRunId: string | null;
  trainingRunIds: string[];
  targets: string | null;
  sisu: number | null;
  perturbation: string | null;
  sourceIssue: string | null;
  provenanceId: string;
  uri: string | null;
}

export function selectedIds(stage: StudioStageSpec | null | undefined, key: string): string[] {
  const value = stage?.selection_spec[key];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

export function trainingRunSummaries(stage: StudioStageSpec | null | undefined): TrainingRunSummary[] {
  const refs = uniqueRefs(
    stage?.output_collections.flatMap((collection) => collection.item_refs) ?? []
  );
  return refs
    .filter((ref) => ref.role === 'training_run' || ref.kind === 'TrainingRun')
    .map(trainingRunSummary)
    .sort((a, b) => {
      if (a.finalValidationLoss === null && b.finalValidationLoss === null) {
        return a.label.localeCompare(b.label);
      }
      if (a.finalValidationLoss === null) return 1;
      if (b.finalValidationLoss === null) return -1;
      return a.finalValidationLoss - b.finalValidationLoss;
    });
}

export function trainingInputSummaries(
  stage: StudioStageSpec | null | undefined
): TrainingRunSummary[] {
  const refs = uniqueRefs(
    stage?.input_collections.flatMap((collection) => collection.item_refs) ?? []
  );
  return refs
    .filter((ref) => ref.role === 'training_run' || ref.kind === 'TrainingRun')
    .map(trainingRunSummary)
    .sort((a, b) => {
      if (a.finalValidationLoss === null && b.finalValidationLoss === null) {
        return a.label.localeCompare(b.label);
      }
      if (a.finalValidationLoss === null) return 1;
      if (b.finalValidationLoss === null) return -1;
      return a.finalValidationLoss - b.finalValidationLoss;
    });
}

export function evaluationRunSummaries(
  stage: StudioStageSpec | null | undefined
): EvaluationRunSummary[] {
  const refs = uniqueRefs(
    stage?.output_collections.flatMap((collection) => collection.item_refs) ?? []
  );
  return refs
    .filter((ref) => ref.role === 'evaluation_run' || ref.kind === 'EvaluationRun')
    .map(evaluationRunSummary);
}

export function bestTrainingRun(rows: TrainingRunSummary[]): TrainingRunSummary | null {
  return rows.find((row) => row.finalValidationLoss !== null) ?? rows[0] ?? null;
}

export function formatMetric(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return 'Not recorded';
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(digits);
  return value.toPrecision(digits);
}

export function trainingRunMetricValue(
  row: TrainingRunSummary,
  metricId: string
): number | null {
  return row.metrics[metricId] ?? null;
}

export function runParameterSummary(row: TrainingRunSummary): string {
  const parts = [
    row.rampShape ? `${capitalize(row.rampShape)} ramp` : null,
    row.rampDurationSteps !== null ? `${row.rampDurationSteps} steps` : null,
    row.nnOutputPreGo !== null ? `pre-go ${row.nnOutputPreGo}` : null,
    row.replicateCount !== null ? `${row.replicateCount} reps` : null,
  ];
  return parts.filter(Boolean).join(' - ') || 'Parameters not recorded';
}

export function evaluationProtocolLabel(row: EvaluationRunSummary): string {
  const parts = [
    row.targets,
    row.sisu !== null ? `SISU ${row.sisu}` : null,
    row.perturbation ? perturbationLabel(row.perturbation) : null,
  ];
  return parts.filter(Boolean).join(' - ') || 'Protocol not recorded';
}

function uniqueRefs(refs: StudioManifestRef[]): StudioManifestRef[] {
  const byId = new Map<string, StudioManifestRef>();
  for (const ref of refs) byId.set(ref.id, ref);
  return Array.from(byId.values());
}

function trainingRunSummary(ref: StudioManifestRef): TrainingRunSummary {
  const hyperparams = objectValue(ref.metadata.hyperparams);
  const typedCheckpointAvailable = booleanValue(ref.metadata.checkpoint_available);
  const axisCoordinates = axisCoordinatesFromRef(ref);
  return {
    id: ref.id,
    label: stringValue(ref.metadata.name) ?? ref.id,
    status: stringValue(ref.metadata.status) ?? 'unknown',
    variant: stringValue(metadataValue(ref, 'run_variant')),
    rampShape: stringValue(metadataValue(ref, 'ramp_shape')),
    rampDurationSteps: numberValue(metadataValue(ref, 'ramp_duration_steps')),
    nnOutputPreGo: numberValue(metadataValue(ref, 'nn_output_pre_go')),
    finalValidationLoss: numberValue(metadataValue(ref, 'final_validation_loss')),
    velocityRmse: numberValue(metadataValue(ref, 'within_cell_velocity_rmse_m_per_s')),
    peakVelocityMean: nestedNumber(metadataValue(ref, 'peak_velocity_m_per_s'), 'mean'),
    peakVelocitySd: nestedNumber(metadataValue(ref, 'peak_velocity_m_per_s'), 'sd'),
    holdDriftMeanMm: nestedNumber(metadataValue(ref, 'hold_drift_mm'), 'mean'),
    holdDriftSdMm: nestedNumber(metadataValue(ref, 'hold_drift_mm'), 'sd'),
    metrics: trainingRunMetrics(ref),
    replicateCount: numberValue(hyperparams?.n_replicates ?? ref.metadata.n_replicates),
    batchSize: numberValue(hyperparams?.batch_size ?? ref.metadata.batch_size),
    warmupBatches: numberValue(
      hyperparams?.n_warmup_batches ?? ref.metadata.n_warmup_batches
    ),
    checkpointAvailable:
      typedCheckpointAvailable ??
      (Boolean(ref.uri) || stringValue(ref.metadata.checkpoint_uri) !== null),
    sourceIssue: stringValue(ref.metadata.source_issue),
    provenanceId: stringValue(ref.metadata.provenance_id) ?? ref.id,
    uri: ref.uri ?? null,
    jobId: stringValue(ref.metadata.job_id),
    axisCoordinates,
    runSetId: stringValue(ref.metadata.run_set_id),
    planned: booleanValue(ref.metadata.planned) ?? false,
    supersededBy: stringValue(ref.metadata.superseded_by),
  };
}

function trainingRunMetrics(ref: StudioManifestRef): Record<string, number | null> {
  const metrics: Record<string, number | null> = {};
  const typedMetrics = objectValue(ref.metadata.metrics);
  if (typedMetrics) {
    for (const [key, value] of Object.entries(typedMetrics)) {
      const direct = numberValue(value);
      const mean = nestedNumber(value, 'mean');
      if (direct !== null) metrics[key] = direct;
      else if (mean !== null) metrics[key] = mean;
    }
  }
  for (const [key, value] of Object.entries(ref.metadata)) {
    const direct = numberValue(value);
    const mean = nestedNumber(value, 'mean');
    if (direct !== null) metrics[key] = direct;
    else if (mean !== null) metrics[key] = mean;
  }
  return metrics;
}

function metadataValue(ref: StudioManifestRef, key: string): unknown {
  const direct = ref.metadata[key];
  if (direct !== undefined) return direct;
  const metrics = objectValue(ref.metadata.metrics);
  if (metrics && metrics[key] !== undefined) return metrics[key];
  const hyperparams = objectValue(ref.metadata.hyperparams);
  return hyperparams?.[key];
}

function axisCoordinatesFromRef(ref: StudioManifestRef): Record<string, unknown> {
  const studio = objectValue(ref.metadata.studio);
  const direct = objectValue(ref.metadata.axis_coordinates);
  const nested = objectValue(studio?.axis_coordinates);
  const coordinates = nested ?? direct ?? {};
  const hyperparams = objectValue(ref.metadata.hyperparams);
  const axisHyperparams = Object.fromEntries(
    Object.entries(hyperparams ?? ref.metadata)
      .filter(([key]) => key.startsWith('axis_'))
      .map(([key, value]) => [key.slice('axis_'.length), value])
  );
  return {
    ...axisHyperparams,
    ...coordinates,
  };
}

function evaluationRunSummary(ref: StudioManifestRef): EvaluationRunSummary {
  const protocol = objectValue(ref.metadata.eval_protocol);
  const trainingRunIds = arrayOfStrings(ref.metadata.training_run_ids);
  return {
    id: ref.id,
    label: stringValue(ref.metadata.name) ?? ref.id,
    status: stringValue(ref.metadata.status) ?? 'unknown',
    selectedTrainingRunId: stringValue(ref.metadata.selected_training_run_id),
    trainingRunIds,
    targets: stringValue(protocol?.targets),
    sisu: numberValue(protocol?.sisu),
    perturbation: stringValue(protocol?.perturbation),
    sourceIssue: stringValue(ref.metadata.source_issue),
    provenanceId: ref.id,
    uri: ref.uri ?? null,
  };
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function numberValue(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function booleanValue(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function nestedNumber(value: unknown, key: string): number | null {
  return numberValue(objectValue(value)?.[key]);
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : [];
}

function capitalize(value: string): string {
  return value.length === 0 ? value : `${value[0].toUpperCase()}${value.slice(1)}`;
}

function perturbationLabel(value: string): string {
  return value === 'none' ? 'no perturbation' : `${value} perturbation`;
}
