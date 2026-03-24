/** Metadata for a single training run. */
export interface TrainingRun {
  id: string;
  name: string;
  createdAt: string; // ISO 8601
  status: 'running' | 'completed' | 'failed' | 'stopped';
  /** Key hyperparameters for at-a-glance differentiation. */
  hyperparams: Record<string, string | number>;
}

/** Metadata for a single evaluation run within a training run. */
export interface EvalRun {
  id: string;
  trainingRunId: string;
  name: string;
  createdAt: string; // ISO 8601
  status: 'running' | 'completed' | 'failed';
  /** Brief description of what this evaluation tested. */
  description?: string;
}

/** Parameters for creating a new evaluation run. */
export interface CreateEvalRunParams {
  trainingRunId: string;
  name: string;
  evalParams: Record<string, unknown>;
}
