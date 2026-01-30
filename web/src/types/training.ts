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
