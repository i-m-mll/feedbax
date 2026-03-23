import type { TrainingRun, EvalRun } from '@/types/runs';

/**
 * Stub training runs for UI development.
 * Replace with real API calls when the backend endpoints exist.
 */
const STUB_TRAINING_RUNS: TrainingRun[] = [
  {
    id: 'tr-001',
    name: 'Baseline reach',
    createdAt: '2026-03-20T14:30:00Z',
    status: 'completed',
    hyperparams: { lr: 0.001, hidden: 100, batches: 1000 },
  },
  {
    id: 'tr-002',
    name: 'High-dim hidden',
    createdAt: '2026-03-21T09:15:00Z',
    status: 'completed',
    hyperparams: { lr: 0.001, hidden: 256, batches: 2000 },
  },
];

const STUB_EVAL_RUNS: Record<string, EvalRun[]> = {
  'tr-001': [
    {
      id: 'ev-001a',
      trainingRunId: 'tr-001',
      name: 'Default eval',
      createdAt: '2026-03-20T15:00:00Z',
      status: 'completed',
      description: 'Standard 8-target evaluation',
    },
    {
      id: 'ev-001b',
      trainingRunId: 'tr-001',
      name: 'Perturbed',
      createdAt: '2026-03-20T16:00:00Z',
      status: 'completed',
      description: 'Force-field perturbation test',
    },
  ],
  'tr-002': [
    {
      id: 'ev-002a',
      trainingRunId: 'tr-002',
      name: 'Default eval',
      createdAt: '2026-03-21T10:00:00Z',
      status: 'completed',
      description: 'Standard 8-target evaluation',
    },
    {
      id: 'ev-002b',
      trainingRunId: 'tr-002',
      name: '16 targets',
      createdAt: '2026-03-21T11:30:00Z',
      status: 'completed',
      description: 'Extended target set',
    },
  ],
};

/** Fetch all training runs for the current project. */
export async function fetchTrainingRuns(): Promise<TrainingRun[]> {
  // TODO: Replace with real API call: request<TrainingRun[]>('/api/runs/training')
  return Promise.resolve(STUB_TRAINING_RUNS);
}

/** Fetch evaluation runs for a specific training run. */
export async function fetchEvalRuns(trainingRunId: string): Promise<EvalRun[]> {
  // TODO: Replace with real API call: request<EvalRun[]>(`/api/runs/training/${trainingRunId}/evals`)
  return Promise.resolve(STUB_EVAL_RUNS[trainingRunId] ?? []);
}

/** Create a new training run (stub). */
export async function createTrainingRun(name: string): Promise<TrainingRun> {
  // TODO: Replace with real API call
  const run: TrainingRun = {
    id: `tr-${Date.now()}`,
    name,
    createdAt: new Date().toISOString(),
    status: 'running',
    hyperparams: {},
  };
  return Promise.resolve(run);
}
