import { useQuery } from '@tanstack/react-query';
import { asApiRequestError, requestJson } from '@/api/request';
import type { CatalogEntry } from '@/components/ui/CatalogPicker';
import {
  parseContract,
  type StudioAuthoringCatalog,
  type StudioCheckpointRuleRequest,
  type StudioRemovedRunsRequest,
  type StudioSubsetToRowsRequest,
  type StudioTrialAuthoring,
  type StudioTrialAuthoringRequest,
  type StudioTrainAuthoring,
  type StudioTrainAuthoringRequest,
  type StudioTrainStartRequest,
  type TrainFieldRequest,
  type TrainFieldResponse,
} from '@/generated/studioContracts';

export async function prepareCheckpointRuleEdit(projectId: string, body: StudioCheckpointRuleRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/checkpoint-rule`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('StudioCheckpointRuleRequest', body)), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioCheckpointRuleEdit', response);
    if (edit.revision !== body.revision) throw new Error('The checkpoint edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path, 'The checkpoint edit did not match its contract');
  }
}

export async function prepareTrainStart(
  projectId: string, stepId: string, body: StudioTrainStartRequest, signal: AbortSignal,
) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/train/${encodeURIComponent(stepId)}/start`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('StudioTrainStartRequest', body)), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioTrainStartEdit', response);
    if (edit.revision !== body.revision) throw new Error('The start edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path, 'The start edit did not match its contract');
  }
}

export async function prepareSubsetToRows(projectId: string, body: StudioSubsetToRowsRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/subset-to-rows`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('StudioSubsetToRowsRequest', body)), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioSubsetToRowsEdit', response);
    if (edit.revision !== body.revision) throw new Error('The subset edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path, 'The subset edit did not match its contract');
  }
}

/** The planned runs the files would remove or replace, keyed as the window map keys its rows. */
export async function reviewRemovedRuns(projectId: string, body: StudioRemovedRunsRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/removed-runs`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('StudioRemovedRunsRequest', body)), cache: 'no-store', signal,
    });
    const removed = parseContract('StudioRemovedRuns', response);
    if (removed.revision !== body.revision) throw new Error('The removal review revision changed');
    return removed.runs;
  } catch (error) {
    throw asApiRequestError(error, path, 'The removal review did not match its contract');
  }
}

export async function readStudioTrainAuthoring(
  projectId: string, stepId: string, body: StudioTrainAuthoringRequest, signal: AbortSignal,
): Promise<StudioTrainAuthoring> {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/train/${encodeURIComponent(stepId)}`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('StudioTrainAuthoringRequest', body)), cache: 'no-store', signal,
    });
    const projection = parseContract('StudioTrainAuthoring', response);
    if (projection.project_id !== projectId || projection.step_id !== stepId ||
        projection.revision !== body.revision) {
      throw new Error('Authoring response does not match the requested step revision');
    }
    return projection;
  } catch (error) {
    throw asApiRequestError(error, path, 'The Train authoring response did not match its contract');
  }
}

export async function projectTrainFields(
  projectId: string, stepId: string, body: TrainFieldRequest, signal: AbortSignal,
): Promise<TrainFieldResponse> {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/train/${encodeURIComponent(stepId)}/fields`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('TrainFieldRequest', body)), cache: 'no-store', signal,
    });
    return parseContract('TrainFieldResponse', response);
  } catch (error) {
    throw asApiRequestError(error, path, 'The Train field response did not match its contract');
  }
}

export async function readStudioTrialAuthoring(
  projectId: string, body: StudioTrialAuthoringRequest, signal: AbortSignal,
): Promise<StudioTrialAuthoring> {
  const path = '/api/projects/' + encodeURIComponent(projectId) + '/authoring/trials';
  try {
    const response = await requestJson(path, {
      method: 'POST', body: JSON.stringify(parseContract('StudioTrialAuthoringRequest', body)), cache: 'no-store', signal,
    });
    const projection = parseContract('StudioTrialAuthoring', response);
    if (projection.revision !== body.revision || projection.run.step !== body.run.step ||
        projection.run.row !== body.run.row || projection.run.variant !== body.run.variant) {
      throw new Error('Trial authoring response does not match the selected revision and row');
    }
    return projection;
  } catch (error) {
    throw asApiRequestError(error, path, 'The trial authoring response did not match its contract');
  }
}

export async function readAuthoringCatalog(signal: AbortSignal): Promise<StudioAuthoringCatalog> {
  const path = '/api/projects/authoring/catalog';
  try {
    return parseContract('StudioAuthoringCatalog', await requestJson(path, { signal }));
  } catch (error) {
    throw asApiRequestError(error, path, 'The authoring catalog did not match its contract');
  }
}

/** A registry's declarations as catalog rows: label, and the package that declares each. */
export function catalogEntries(catalog: StudioAuthoringCatalog | undefined, family: string): CatalogEntry[] {
  const registry = catalog?.registries[family];
  return (registry?.entries ?? []).flatMap((entry) => 'key' in entry && 'label' in entry && typeof entry.label === 'string'
    ? [{ key: entry.key, label: entry.label, package: registry!.origins[entry.key]?.package }] : []);
}

/** The registries of declarations, read once per session. */
export function useAuthoringCatalog() {
  return useQuery({
    queryKey: ['authoring-catalog'],
    queryFn: ({ signal }) => readAuthoringCatalog(signal),
    staleTime: Infinity,
  });
}
