export type {
  CreateEvalRunRequest,
  EvalRunInfo,
  TrainingRunInfo,
} from '@/generated/studioContracts';

/** Metadata for a single training run. */
export interface TrainingRun {
  id: string;
  name: string;
  createdAt: string; // ISO 8601
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'stale' | 'stopped';
  /** Key hyperparameters for at-a-glance differentiation. */
  hyperparams: Record<string, string | number>;
  metrics?: Record<string, unknown>;
  uri?: string;
  stageId?: string;
  scenarioId?: string;
  planned?: boolean;
  checkpointAvailable?: boolean;
  sourceIssue?: string;
  provenanceId?: string;
  supersededBy?: string;
}

/** Metadata for a single evaluation run within a training run. */
export interface EvalRun {
  id: string;
  trainingRunId: string;
  name: string;
  createdAt: string; // ISO 8601
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'stale';
  /** Brief description of what this evaluation tested. */
  description?: string;
  trainingRunIds?: string[];
  uri?: string;
}

/** Parameters for creating a new evaluation run. */
export interface CreateEvalRunParams {
  trainingRunId: string;
  name: string;
  evalParams: Record<string, unknown>;
}
