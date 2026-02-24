import type { ParamValue } from '@/types/graph';

export interface OptimizerSpec {
  type: string;
  params: Record<string, ParamValue>;
}

export interface TimeAggregationSpec {
  mode: 'all' | 'final' | 'range' | 'segment' | 'custom';
  start?: number;
  end?: number;
  segment_name?: string;
  time_idxs?: number[];
  discount?: 'none' | 'power' | 'linear';
  discount_exp?: number;
}

export interface LossTermSpec {
  type: string;
  label: string;
  weight: number;
  selector?: string;
  norm?: 'squared_l2' | 'l2' | 'l1' | 'huber';
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
  metrics: Record<string, number>;
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

export type TimeAggregationMode = 'all' | 'final' | 'range' | 'segment' | 'custom';

export type DiscountType = 'none' | 'power' | 'linear';

export const NORM_LABELS: Record<NormFunction, string> = {
  squared_l2: 'Squared L2',
  l2: 'L2',
  l1: 'L1',
  huber: 'Huber',
};

export const TIME_AGG_LABELS: Record<TimeAggregationMode, string> = {
  all: 'All steps',
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
