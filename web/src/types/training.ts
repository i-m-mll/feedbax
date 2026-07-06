import type { ParamValue, RetentionPolicySpec } from '@/types/graph';

export const LOSS_TERM_SPEC_SCHEMA_ID = 'feedbax.spec.training.loss_term';
export const LOSS_TERM_SPEC_SCHEMA_VERSION = 'feedbax.spec.training.loss_term.v2';

export interface OptimizerSpec {
  type: string;
  params: Record<string, ParamValue>;
}

export interface TimeAggregationSpec {
  mode: 'all' | 'mean' | 'sum' | 'final' | 'range' | 'segment' | 'custom';
  start?: number;
  end?: number;
  segment_name?: string;
  time_idxs?: number[];
  discount?: 'none' | 'power' | 'linear';
  discount_exp?: number;
}

export interface LossTermSpec {
  schema_id?: typeof LOSS_TERM_SPEC_SCHEMA_ID;
  schema_version?: string;
  type: string;
  label: string;
  weight: number;
  selector?: string;
  target_selector?: string | null;
  target_value?: unknown;
  retention?: RetentionPolicySpec | null;
  norm?: 'squared_l2' | 'l2' | 'l1' | 'huber';
  matrix?: unknown;
  matrix_kind?: 'dense' | 'diagonal';
  time_agg?: TimeAggregationSpec;
  children?: Record<string, LossTermSpec>;
}

export interface TrainingSpec {
  optimizer: OptimizerSpec;
  loss: LossTermSpec;
  n_batches: number;
  batch_size: number;
  n_epochs?: number;
  checkpoint_interval?: number;
  early_stopping?: {
    metric: string;
    patience: number;
    min_delta: number;
  };
}

export interface TaskSpec {
  type: string;
  params: Record<string, ParamValue>;
  timeline?: Record<string, ParamValue>;
}

export interface TrainingProgress {
  batch: number;
  total_batches: number;
  loss: number;
  loss_terms: Record<string, number>;
  grad_norm: number;
  step_time_ms: number;
  metrics: Record<string, number>;  // keep for backwards compat
  status: string;
}

export interface TrainingLogLine {
  batch: number;
  level: 'info' | 'warning' | 'error';
  message: string;
  timestamp: number;  // Date.now() when received
}

// --- Probe and Loss Types ---

export interface ProbeInfo {
  id: string;
  label: string;
  node: string;
  timing: 'input' | 'output';
  selector: string;
  description?: string;
}

export interface LossValidationError {
  path: string[];
  field: string;
  message: string;
}

export interface LossValidationResult {
  valid: boolean;
  errors: LossValidationError[];
}

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

// --- Training request payload types (Phase 6) ---

/** Runtime controls sent alongside the canonical GraphSpec. */
export interface TrainingConfig {
  n_batches: number;
  batch_size: number;
  learning_rate: number;
  grad_clip: number;
  n_reach_steps: number;
}

/**
 * Full payload sent to POST /api/training.
 * graph_spec and training_config are optional so that old callers without
 * the new fields remain compatible.
 */
export interface TrainingStartPayload {
  graph_id: string;
  training_spec: TrainingSpec;
  task_spec: TaskSpec;
  task_binding_spec?: import('@/types/workspace').StudioTaskBindingSpec;
  graph_spec?: import('@/types/graph').GraphSpec;
  training_config?: TrainingConfig;
}
