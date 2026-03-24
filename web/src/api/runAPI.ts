/**
 * API client for training and evaluation run discovery.
 *
 * Calls the backend endpoints and falls back to stub data when the
 * backend is unavailable (offline / demo mode).
 */

import type { TrainingRun, EvalRun } from '@/types/runs';

// ---------------------------------------------------------------------------
// Wire format -- backend uses snake_case, frontend uses camelCase
// ---------------------------------------------------------------------------

interface TrainingRunWire {
  id: string;
  name: string;
  created_at: string;
  status: string;
  hyperparams: Record<string, string | number>;
}

interface EvalRunWire {
  id: string;
  training_run_id: string;
  name: string;
  created_at: string;
  status: string;
  description?: string | null;
}

function trainingRunFromWire(wire: TrainingRunWire): TrainingRun {
  return {
    id: wire.id,
    name: wire.name,
    createdAt: wire.created_at,
    status: wire.status as TrainingRun['status'],
    hyperparams: wire.hyperparams,
  };
}

function evalRunFromWire(wire: EvalRunWire): EvalRun {
  return {
    id: wire.id,
    trainingRunId: wire.training_run_id,
    name: wire.name,
    createdAt: wire.created_at,
    status: wire.status as EvalRun['status'],
    description: wire.description ?? undefined,
  };
}

// ---------------------------------------------------------------------------
// Stub data -- fallback when the backend is not available
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

async function tryFetch<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) return null;
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

/** Fetch all training runs. Falls back to stub data on error. */
export async function fetchTrainingRuns(): Promise<TrainingRun[]> {
  const wire = await tryFetch<TrainingRunWire[]>('/api/runs/training');
  if (wire !== null) return wire.map(trainingRunFromWire);
  return STUB_TRAINING_RUNS;
}

/** Fetch evaluation runs for a training run. Falls back to stub data on error. */
export async function fetchEvalRuns(trainingRunId: string): Promise<EvalRun[]> {
  const wire = await tryFetch<EvalRunWire[]>(
    `/api/runs/training/${encodeURIComponent(trainingRunId)}/evals`,
  );
  if (wire !== null) return wire.map(evalRunFromWire);
  return STUB_EVAL_RUNS[trainingRunId] ?? [];
}

/** Create a new training run (stub -- no backend endpoint yet). */
export async function createTrainingRun(name: string): Promise<TrainingRun> {
  // TODO: Replace with real API call when backend supports it
  const run: TrainingRun = {
    id: `tr-${Date.now()}`,
    name,
    createdAt: new Date().toISOString(),
    status: 'running',
    hyperparams: {},
  };
  return Promise.resolve(run);
}

/** Create a new evaluation run.
 *
 * Attempts to POST to the backend; falls back to a client-side stub if
 * the endpoint is not available.
 */
export async function createEvalRun(
  trainingRunId: string,
  name: string,
  evalParams: Record<string, unknown>,
): Promise<EvalRun> {
  try {
    const response = await fetch('/api/runs/evaluation', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        training_run_id: trainingRunId,
        name,
        eval_params: evalParams,
      }),
    });
    if (response.ok) {
      const wire = (await response.json()) as EvalRunWire;
      return evalRunFromWire(wire);
    }
  } catch {
    // Fall through to stub
  }

  // Stub fallback — generate a client-side ID
  return {
    id: `ev-${Date.now()}`,
    trainingRunId,
    name,
    createdAt: new Date().toISOString(),
    status: 'running',
    description: summarizeEvalParams(evalParams),
  };
}

/** Build a short human-readable summary from eval params. */
function summarizeEvalParams(params: Record<string, unknown>): string {
  const parts: string[] = [];
  if (params.perturbation_type) parts.push(String(params.perturbation_type));
  if (Array.isArray(params.perturbation_amplitudes) && params.perturbation_amplitudes.length > 0) {
    parts.push(`amp=[${params.perturbation_amplitudes.join(',')}]`);
  }
  return parts.join(', ') || 'Custom evaluation';
}
