import type {
  ProbeResponse,
  ValidateLossResponse,
  ValidationErrorResponse,
} from '@/generated/studioContracts';

export const LOSS_TERM_SPEC_SCHEMA_ID = 'feedbax.spec.training.loss_term';
export const LOSS_TERM_SPEC_SCHEMA_VERSION = 'feedbax.spec.training.loss_term.v2';

export type {
  EarlyStoppingSpec,
  DomainDiagnostic,
  LossTermSpec,
  OptimizerSpec,
  ProbeResponse,
  TaskSpec,
  TimeAggregationSpec,
  TrainingConfig,
  TrainingErrorEvent,
  TrainingLogEvent,
  TrainingProgressEvent,
  TrainingSpec,
  TrainingStartPayload,
  TrainingStartResponse,
  TrainingStatusPayload,
  TrainingStatusResponse,
  TrainingTrajectoryEvent,
  TrainingTrajectoryPayload,
  TrainingWebSocketEvent,
  ValidateLossResponse,
  ValidationErrorResponse,
} from '@/generated/studioContracts';

export type TrainingProgress = Omit<
  import('@/generated/studioContracts').TrainingProgressEvent,
  'type' | 'job_id' | 'seq' | 'emitted_at_ms'
> &
  Partial<
    Pick<
      import('@/generated/studioContracts').TrainingProgressEvent,
      'type' | 'job_id' | 'seq' | 'emitted_at_ms'
    >
  >;

export interface TrainingLogLine {
  batch: number;
  level: 'info' | 'warning' | 'error';
  message: string;
  timestamp: number;  // Date.now() when received
}

export type ProbeInfo = ProbeResponse;
export type LossValidationError = ValidationErrorResponse;
export type LossValidationResult = ValidateLossResponse;

export type NormFunction = 'squared_l2' | 'l2' | 'l1' | 'huber';

export type TimeAggregationMode = 'all' | 'mean' | 'sum' | 'final' | 'range' | 'segment' | 'custom';

export type DiscountType = 'none' | 'power' | 'linear';

export const NORM_LABELS: Record<NormFunction, string> = {
  squared_l2: 'Squared L2',
  l2: 'L2',
  l1: 'L1',
  huber: 'Huber',
};

export const TIME_AGG_LABELS: Record<TimeAggregationMode, string> = {
  all: 'All steps',
  mean: 'Mean',
  sum: 'Sum',
  final: 'Final step',
  range: 'Time range',
  segment: 'Segment',
  custom: 'Custom indices',
};

export const DISCOUNT_LABELS: Record<DiscountType, string> = {
  none: 'None',
  power: 'Power decay',
  linear: 'Linear decay',
};

/**
 * Full payload sent to POST /api/training.
 * graph_spec and training_config are optional so that old callers without
 * the new fields remain compatible.
 */
export interface TrainingStartRequestPayload {
  graph_id: string;
  training_spec: import('@/generated/studioContracts').TrainingSpec;
  task_spec: import('@/generated/studioContracts').TaskSpec;
  task_binding_spec?: import('@/types/workspace').StudioTaskBindingSpec;
  graph_spec?: import('@/types/graph').GraphSpec;
  training_config?: import('@/generated/studioContracts').TrainingConfig;
}
