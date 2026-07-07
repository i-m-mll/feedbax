/**
 * API client for training and evaluation run discovery.
 *
 * Calls the backend endpoints and reports backend failures to callers.
 */

import type { EvalRunInfo, TrainingRun, TrainingRunInfo, EvalRun } from '@/types/runs';
import { parseContract } from '@/generated/studioContracts';
import { asApiRequestError, requestJson } from '@/api/request';

// ---------------------------------------------------------------------------
// Wire format -- backend uses snake_case, frontend uses camelCase
// ---------------------------------------------------------------------------

function displayHyperparams(hyperparams: Record<string, unknown>): Record<string, string | number> {
  return Object.fromEntries(
    Object.entries(hyperparams).filter(
      (entry): entry is [string, string | number] =>
        typeof entry[1] === 'string' || typeof entry[1] === 'number',
    ),
  );
}

function trainingRunFromWire(wire: TrainingRunInfo): TrainingRun {
  return {
    id: wire.id,
    name: wire.name,
    createdAt: wire.created_at,
    status: wire.status as TrainingRun['status'],
    hyperparams: displayHyperparams(wire.hyperparams),
  };
}

function evalRunFromWire(wire: EvalRunInfo): EvalRun {
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
  const path = '/api/runs/training';
  const wire = await requestJson(path) as unknown[];
  try {
    const runs = wire.map((item) => parseContract('TrainingRunInfo', item));
    return runs.map(trainingRunFromWire);
  } catch (error) {
    throw asApiRequestError(error, path, 'Training run response did not match the Studio contract.');
  }
}

/** Fetch evaluation runs for a training run. */
export async function fetchEvalRuns(trainingRunId: string): Promise<EvalRun[]> {
  const path = `/api/runs/training/${encodeURIComponent(trainingRunId)}/evals`;
  const wire = await requestJson(path) as unknown[];
  try {
    const runs = wire.map((item) => parseContract('EvalRunInfo', item));
    return runs.map(evalRunFromWire);
  } catch (error) {
    throw asApiRequestError(error, path, 'Evaluation run response did not match the Studio contract.');
  }
}

/** Create a new training run. */
export async function createTrainingRun(name: string): Promise<TrainingRun> {
  const path = '/api/runs/training';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
  try {
    return trainingRunFromWire(parseContract('TrainingRunInfo', result));
  } catch (error) {
    throw asApiRequestError(error, path, 'Created training run did not match the Studio contract.');
  }
}

/** Create a new evaluation run. */
export async function createEvalRun(
  trainingRunId: string,
  name: string,
  evalParams: Record<string, unknown>,
): Promise<EvalRun> {
  const path = '/api/runs/evaluation';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({
      training_run_id: trainingRunId,
      name,
      eval_params: evalParams,
    }),
  });
  try {
    return evalRunFromWire(parseContract('EvalRunInfo', result));
  } catch (error) {
    throw asApiRequestError(error, path, 'Created evaluation run did not match the Studio contract.');
  }
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
