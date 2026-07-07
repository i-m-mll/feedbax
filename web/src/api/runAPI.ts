/**
 * API client for training and evaluation run discovery.
 *
 * Calls the backend endpoints and reports backend failures to callers.
 */

import type { TrainingRun, EvalRun } from '@/types/runs';
import { requestJson } from '@/api/request';

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
// API functions
// ---------------------------------------------------------------------------

/** Fetch all training runs. */
export async function fetchTrainingRuns(): Promise<TrainingRun[]> {
  const wire = await requestJson('/api/runs/training') as TrainingRunWire[];
  return wire.map(trainingRunFromWire);
}

/** Fetch evaluation runs for a training run. */
export async function fetchEvalRuns(trainingRunId: string): Promise<EvalRun[]> {
  const wire = await requestJson(
    `/api/runs/training/${encodeURIComponent(trainingRunId)}/evals`,
  ) as EvalRunWire[];
  return wire.map(evalRunFromWire);
}

/** Create a new training run. */
export async function createTrainingRun(name: string): Promise<TrainingRun> {
  const wire = await requestJson('/api/runs/training', {
    method: 'POST',
    body: JSON.stringify({ name }),
  }) as TrainingRunWire;
  return trainingRunFromWire(wire);
}

/** Create a new evaluation run. */
export async function createEvalRun(
  trainingRunId: string,
  name: string,
  evalParams: Record<string, unknown>,
): Promise<EvalRun> {
  const wire = await requestJson('/api/runs/evaluation', {
    method: 'POST',
    body: JSON.stringify({
      training_run_id: trainingRunId,
      name,
      eval_params: evalParams,
    }),
  }) as EvalRunWire;
  return evalRunFromWire(wire);
}

/** Build a short human-readable summary from eval params. */
export function summarizeEvalParams(params: Record<string, unknown>): string {
  const parts: string[] = [];
  if (params.perturbation_type) parts.push(String(params.perturbation_type));
  if (Array.isArray(params.perturbation_amplitudes) && params.perturbation_amplitudes.length > 0) {
    parts.push(`amp=[${params.perturbation_amplitudes.join(',')}]`);
  }
  return parts.join(', ') || 'Custom evaluation';
}
