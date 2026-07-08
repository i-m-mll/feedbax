/**
 * API client for training and evaluation run discovery.
 *
 * Calls the backend endpoints and reports backend failures to callers.
 */

import type {
  EvalRunInfo,
  EvalRun,
  ManifestImportResponse,
  TrainingRun,
  TrainingRunCompareResponse,
  TrainingRunInfo,
} from '@/types/runs';
import type {
  SelectionPreview,
  SelectionRefreshDiff,
  SelectionSpec,
} from '@/generated/studioContracts';
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
    metrics: wire.metrics ?? {},
    uri: wire.uri ?? undefined,
    stageId: wire.stage_id ?? undefined,
    scenarioId: wire.scenario_id ?? undefined,
    planned: wire.planned ?? false,
    checkpointAvailable: wire.checkpoint_available ?? false,
    sourceIssue: wire.source_issue ?? undefined,
    provenanceId: wire.provenance_id ?? wire.id,
    supersededBy: wire.superseded_by ?? undefined,
    legacyCheckpoint: wire.legacy_checkpoint ?? undefined,
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
    trainingRunIds: wire.training_run_ids ?? [wire.training_run_id],
    uri: wire.uri ?? undefined,
  };
}

function compareResponseFromWire(wire: unknown): TrainingRunCompareResponse {
  if (!wire || typeof wire !== 'object' || Array.isArray(wire)) {
    throw new Error('Compare response was not an object.');
  }
  const rows = (wire as { rows?: unknown }).rows;
  if (!Array.isArray(rows)) throw new Error('Compare response rows were not an array.');
  return {
    rows: rows.map((row) => {
      if (!row || typeof row !== 'object' || Array.isArray(row)) {
        throw new Error('Compare row was not an object.');
      }
      const payload = row as Record<string, unknown>;
      if (typeof payload.id !== 'string') throw new Error('Compare row id was not a string.');
      return {
        id: payload.id,
        params:
          payload.params && typeof payload.params === 'object' && !Array.isArray(payload.params)
            ? payload.params as Record<string, unknown>
            : {},
        metrics:
          payload.metrics && typeof payload.metrics === 'object' && !Array.isArray(payload.metrics)
            ? payload.metrics as Record<string, unknown>
            : {},
      };
    }),
  };
}

function importResponseFromWire(wire: unknown): ManifestImportResponse {
  if (!wire || typeof wire !== 'object' || Array.isArray(wire)) {
    throw new Error('Import response was not an object.');
  }
  const payload = wire as Record<string, unknown>;
  const training = Array.isArray(payload.training_runs) ? payload.training_runs : [];
  const evals = Array.isArray(payload.eval_runs) ? payload.eval_runs : [];
  const stringList = (key: string) =>
    Array.isArray(payload[key])
      ? payload[key].filter((item): item is string => typeof item === 'string')
      : [];
  return {
    root: typeof payload.root === 'string' ? payload.root : '',
    sourcePath: typeof payload.source_path === 'string' ? payload.source_path : '',
    importedManifestIds: stringList('imported_manifest_ids'),
    skippedManifestIds: stringList('skipped_manifest_ids'),
    manifestCount: typeof payload.manifest_count === 'number' ? payload.manifest_count : 0,
    artifactCount: typeof payload.artifact_count === 'number' ? payload.artifact_count : 0,
    includedArtifactCount:
      typeof payload.included_artifact_count === 'number' ? payload.included_artifact_count : 0,
    externalArtifactCount:
      typeof payload.external_artifact_count === 'number' ? payload.external_artifact_count : 0,
    indexPath: typeof payload.index_path === 'string' ? payload.index_path : null,
    trainingRuns: training.map((item) => trainingRunFromWire(parseContract('TrainingRunInfo', item))),
    evalRuns: evals.map((item) => evalRunFromWire(parseContract('EvalRunInfo', item))),
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

/** Fetch the durable training manifest payload for snapshot actions. */
export async function fetchTrainingRunManifest(
  trainingRunId: string
): Promise<Record<string, unknown>> {
  const path = `/api/runs/training/${encodeURIComponent(trainingRunId)}/manifest`;
  const result = await requestJson(path);
  if (!result || typeof result !== 'object' || Array.isArray(result)) {
    throw asApiRequestError(
      new Error('Training manifest response was not an object.'),
      path,
      'Training manifest response did not match the expected shape.'
    );
  }
  return result as Record<string, unknown>;
}

/** Fetch the durable evaluation manifest payload for snapshot actions. */
export async function fetchEvalRunManifest(
  evalRunId: string
): Promise<Record<string, unknown>> {
  const path = `/api/runs/evaluation/${encodeURIComponent(evalRunId)}/manifest`;
  const result = await requestJson(path);
  if (!result || typeof result !== 'object' || Array.isArray(result)) {
    throw asApiRequestError(
      new Error('Evaluation manifest response was not an object.'),
      path,
      'Evaluation manifest response did not match the expected shape.'
    );
  }
  return result as Record<string, unknown>;
}

/** Fetch only selected parameter and metric fields for table compare mode. */
export async function compareTrainingRuns(payload: {
  runIds: string[];
  paramFields: string[];
  metricFields: string[];
}): Promise<TrainingRunCompareResponse> {
  const path = '/api/runs/training/compare';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({
      run_ids: payload.runIds,
      param_fields: payload.paramFields,
      metric_fields: payload.metricFields,
    }),
  });
  try {
    return compareResponseFromWire(result);
  } catch (error) {
    throw asApiRequestError(error, path, 'Compare response did not match the expected shape.');
  }
}

export async function importManifestPacket(pathname: string): Promise<ManifestImportResponse> {
  const path = '/api/runs/import/packet';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({ path: pathname }),
  });
  try {
    return importResponseFromWire(result);
  } catch (error) {
    throw asApiRequestError(error, path, 'Manifest packet import response did not match the expected shape.');
  }
}

export async function importRunsDir(pathname: string): Promise<ManifestImportResponse> {
  const path = '/api/runs/import/runs-dir';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({ path: pathname }),
  });
  try {
    return importResponseFromWire(result);
  } catch (error) {
    throw asApiRequestError(error, path, 'Runs directory import response did not match the expected shape.');
  }
}

/** Preview the current manifest-index matches for a SelectionSpec. */
export async function previewSelectionSpec(
  selectionSpec: SelectionSpec,
  limit = 50,
): Promise<SelectionPreview> {
  const path = '/api/runs/selection/preview';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({ selection_spec: selectionSpec, limit }),
  });
  try {
    return parseContract('SelectionPreview', result);
  } catch (error) {
    throw asApiRequestError(error, path, 'Selection preview response did not match the Studio contract.');
  }
}

/** Compare a frozen SelectionSpec with current manifest-index matches. */
export async function refreshSelectionSpec(
  selectionSpec: SelectionSpec,
  payload: { failedParentIds?: string[]; staleParentIds?: string[] } = {},
): Promise<SelectionRefreshDiff> {
  const path = '/api/runs/selection/refresh';
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify({
      selection_spec: selectionSpec,
      failed_parent_ids: payload.failedParentIds ?? [],
      stale_parent_ids: payload.staleParentIds ?? [],
    }),
  });
  try {
    return parseContract('SelectionRefreshDiff', result);
  } catch (error) {
    throw asApiRequestError(error, path, 'Selection refresh response did not match the Studio contract.');
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

/** Cancel a pending or running training run. */
export async function cancelTrainingRun(trainingRunId: string): Promise<TrainingRun> {
  const path = `/api/runs/training/${encodeURIComponent(trainingRunId)}/cancel`;
  const result = await requestJson(path, { method: 'POST' });
  try {
    return trainingRunFromWire(parseContract('TrainingRunInfo', result));
  } catch (error) {
    throw asApiRequestError(error, path, 'Cancelled training run did not match the Studio contract.');
  }
}

/** Delete a pending training run. */
export async function deleteTrainingRun(trainingRunId: string): Promise<TrainingRun> {
  const path = `/api/runs/training/${encodeURIComponent(trainingRunId)}`;
  const result = await requestJson(path, { method: 'DELETE' });
  try {
    return trainingRunFromWire(parseContract('TrainingRunInfo', result));
  } catch (error) {
    throw asApiRequestError(error, path, 'Deleted training run did not match the Studio contract.');
  }
}

/** Mark a completed training run as superseded without deleting it. */
export async function supersedeTrainingRun(
  trainingRunId: string,
  payload: { superseded_by?: string | null; reason?: string | null },
): Promise<TrainingRun> {
  const path = `/api/runs/training/${encodeURIComponent(trainingRunId)}/supersede`;
  const result = await requestJson(path, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  try {
    return trainingRunFromWire(parseContract('TrainingRunInfo', result));
  } catch (error) {
    throw asApiRequestError(error, path, 'Superseded training run did not match the Studio contract.');
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
