import type { ParamValue } from '@/types/graph';

export interface OptimizerSpec {
  type: string;
  params: Record<string, ParamValue>;
}

export interface LossTermSpec {
  type: string;
  weight: number;
  params: Record<string, ParamValue>;
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
