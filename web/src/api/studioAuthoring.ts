import { useQuery } from '@tanstack/react-query';
import { asApiRequestError, isHttpConflict, requestJson, sentBody } from '@/api/request';
import type { CatalogEntry } from '@/components/ui/CatalogPicker';
import {
  parseContract,
  type ExpressionPartSite,
  type StudioAuthoringCatalog,
  type StudioCheckpointRuleRequest,
  type StudioRemovedRunsRequest,
  type StudioRunRuleRequest,
  type StudioSetEditRequest,
  type StudioSetsOfferedRequest,
  type StudioSubsetToRowsRequest,
  type StudioTrialAuthoring,
  type StudioTrialAuthoringRequest,
  type StudioTrialExpressionMenuRequest,
  type StudioTrainAuthoring,
  type StudioTrainAuthoringRequest,
  type StudioTrainStartRequest,
  type TaskTrialSampleRequest,
  type TaskTrialSampleResponse,
  type TrainFieldRequest,
  type TrainFieldResponse,
} from '@/generated/studioContracts';

export async function prepareCheckpointRuleEdit(projectId: string, body: StudioCheckpointRuleRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/checkpoint-rule`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioCheckpointRuleRequest', body, path), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioCheckpointRuleEdit', response);
    if (edit.revision !== body.revision) throw new Error('The checkpoint edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export async function prepareTrainStart(
  projectId: string, stepId: string, body: StudioTrainStartRequest, signal: AbortSignal,
) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/train/${encodeURIComponent(stepId)}/start`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioTrainStartRequest', body, path), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioTrainStartEdit', response);
    if (edit.revision !== body.revision) throw new Error('The start edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export async function prepareSubsetToRows(projectId: string, body: StudioSubsetToRowsRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/subset-to-rows`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioSubsetToRowsRequest', body, path), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioSubsetToRowsEdit', response);
    if (edit.revision !== body.revision) throw new Error('The subset edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** The files that store a set edit (Duplicate Run, Save Set, Inline Set, a set applied, a value removed). */
export async function prepareSetEdit(projectId: string, body: StudioSetEditRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/sets`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioSetEditRequest', body, path), cache: 'no-store', signal,
    });
    const edit = parseContract('StudioSetEdit', response);
    if (edit.revision !== body.revision) throw new Error('The set edit revision changed');
    return edit;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** The Set documents whose every entry applies to the step's Model. */
export async function readSetsOffered(projectId: string, body: StudioSetsOfferedRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/sets-offered`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioSetsOfferedRequest', body, path), cache: 'no-store', signal,
    });
    return parseContract('StudioSetsOffered', response).sets;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** The planned runs the files would remove or replace, keyed as the window map keys its rows. */
export async function reviewRemovedRuns(projectId: string, body: StudioRemovedRunsRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/removed-runs`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioRemovedRunsRequest', body, path), cache: 'no-store', signal,
    });
    const removed = parseContract('StudioRemovedRuns', response);
    if (removed.revision !== body.revision) throw new Error('The removal review revision changed');
    return removed.runs;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export async function readStudioTrainAuthoring(
  projectId: string, stepId: string, body: StudioTrainAuthoringRequest, signal: AbortSignal,
): Promise<StudioTrainAuthoring> {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/train/${encodeURIComponent(stepId)}`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioTrainAuthoringRequest', body, path), cache: 'no-store', signal,
    });
    const projection = parseContract('StudioTrainAuthoring', response);
    if (projection.project_id !== projectId || projection.step_id !== stepId ||
        projection.revision !== body.revision) {
      throw new Error('Authoring response does not match the requested step revision');
    }
    return projection;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export async function projectTrainFields(
  projectId: string, stepId: string, body: TrainFieldRequest, signal: AbortSignal,
): Promise<TrainFieldResponse> {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/train/${encodeURIComponent(stepId)}/fields`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('TrainFieldRequest', body, path), cache: 'no-store', signal,
    });
    return parseContract('TrainFieldResponse', response);
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export async function readStudioTrialAuthoring(
  projectId: string, body: StudioTrialAuthoringRequest, signal: AbortSignal,
): Promise<StudioTrialAuthoring> {
  const path = '/api/projects/' + encodeURIComponent(projectId) + '/authoring/trials';
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioTrialAuthoringRequest', body, path), cache: 'no-store', signal,
    });
    const projection = parseContract('StudioTrialAuthoring', response);
    if (projection.revision !== body.revision || projection.run.step !== body.run.step ||
        projection.run.row !== body.run.row || projection.run.variant !== body.run.variant) {
      throw new Error('Trial authoring response does not match the selected revision and row');
    }
    return projection;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** The functions and names that fit at one part of an expression of a trial authoring view. */
export async function readTrialExpressionMenu(
  projectId: string, body: StudioTrialExpressionMenuRequest, signal: AbortSignal,
): Promise<ExpressionPartSite> {
  const path = '/api/projects/' + encodeURIComponent(projectId) + '/authoring/trials/expression-menu';
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioTrialExpressionMenuRequest', body, path), cache: 'no-store', signal,
    });
    const site = parseContract('ExpressionPartSite', response);
    if (JSON.stringify(site.path) !== JSON.stringify(body.part)) throw new Error('The expression menu is not of the requested part');
    return site;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** The trial authoring view a menu was asked of is no longer held by the service. */
export const isExpiredExpressionContext = (error: unknown) => isHttpConflict(error) &&
  typeof error.detail === 'object' && error.detail !== null && 'code' in error.detail &&
  error.detail.code === 'trial_expression_context_unavailable';

export async function readAuthoringCatalog(signal: AbortSignal): Promise<StudioAuthoringCatalog> {
  const path = '/api/projects/authoring/catalog';
  try {
    return parseContract('StudioAuthoringCatalog', await requestJson(path, { signal }));
  } catch (error) {
    throw asApiRequestError(error, path);
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

/** What a run rule being written chooses among the runs of the step it reads, or why it is refused. */
export async function checkRunRule(projectId: string, body: StudioRunRuleRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(projectId)}/authoring/run-rule`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioRunRuleRequest', body, path), cache: 'no-store', signal,
    });
    return parseContract('StudioRunRuleCheck', response);
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** Trials drawn from a task without a model, through the function that draws training trials. */
export async function sampleTaskTrials(body: TaskTrialSampleRequest, signal: AbortSignal): Promise<TaskTrialSampleResponse> {
  const path = '/api/execution/task-trials/sample';
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('TaskTrialSampleRequest', body, path), cache: 'no-store', signal,
    });
    return parseContract('TaskTrialSampleResponse', response);
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}
