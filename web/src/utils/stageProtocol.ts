import type { TrainingSpec } from '@/types/training';
import type { StudioScenarioSpec, StudioStageSpec } from '@/types/workspace';

export type ExecutionTargetChoice = 'local' | 'gcp' | 'runpod' | 'manual';

export interface TrainingProtocolSnapshot {
  learningRate: number;
  batchCount: number;
  batchSize: number;
  checkpointInterval: number | null;
  computeTarget: ExecutionTargetChoice;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function realizationRecord(stage: StudioStageSpec | null | undefined): Record<string, unknown> {
  const realization = stage?.metadata?.backend_realization;
  return isRecord(realization) ? realization : {};
}

function computeTarget(value: unknown): ExecutionTargetChoice {
  if (value === 'managed' || value === 'gcp') return 'gcp';
  if (value === 'runpod' || value === 'manual') return value;
  return 'local';
}

export function executionTargetLabel(target: ExecutionTargetChoice): string {
  return {
    local: 'Local worker',
    gcp: 'GCP',
    runpod: 'RunPod',
    manual: 'Manual export',
  }[target];
}

export function executionTargetIsBillable(target: ExecutionTargetChoice): boolean {
  return target === 'gcp' || target === 'runpod';
}

export function executionBackendForTarget(
  target: ExecutionTargetChoice
): 'local' | 'runpod' | null {
  if (target === 'local') return 'local';
  if (target === 'runpod') return 'runpod';
  return null;
}

function learningRate(trainingSpec: TrainingSpec | null | undefined): number {
  const value = trainingSpec?.optimizer.params.learning_rate;
  return typeof value === 'number' && Number.isFinite(value) ? value : 0.001;
}

export function trainingProtocolSnapshot(
  stage: StudioStageSpec | null | undefined,
  scenario: StudioScenarioSpec | null | undefined
): TrainingProtocolSnapshot {
  const realization = realizationRecord(stage);
  const trainingSpec = scenario?.training_spec;
  return {
    learningRate: learningRate(trainingSpec),
    batchCount: trainingSpec?.n_batches ?? 0,
    batchSize: trainingSpec?.batch_size ?? 0,
    checkpointInterval: trainingSpec?.checkpoint_interval ?? null,
    computeTarget: computeTarget(realization.execution_target),
  };
}

export function stageExecutionTarget(
  stage: StudioStageSpec | null | undefined
): ExecutionTargetChoice {
  return computeTarget(realizationRecord(stage).execution_target);
}

export function stageMetadataWithExecutionTarget(
  stage: StudioStageSpec,
  executionTarget: ExecutionTargetChoice
): Record<string, unknown> {
  return {
    ...stage.metadata,
    backend_realization: {
      ...realizationRecord(stage),
      execution_target: executionTarget,
    },
  };
}

export function trainingSpecWithProtocolPatch(
  trainingSpec: TrainingSpec,
  patch: Partial<TrainingProtocolSnapshot>
): TrainingSpec {
  return {
    ...trainingSpec,
    optimizer: {
      ...trainingSpec.optimizer,
      params: {
        ...trainingSpec.optimizer.params,
        ...(patch.learningRate !== undefined ? { learning_rate: patch.learningRate } : {}),
      },
    },
    n_batches: patch.batchCount ?? trainingSpec.n_batches,
    batch_size: patch.batchSize ?? trainingSpec.batch_size,
    checkpoint_interval:
      patch.checkpointInterval === undefined
        ? trainingSpec.checkpoint_interval
        : patch.checkpointInterval ?? undefined,
  };
}
