import { semanticWorkspace } from '@/api/client';
import { asApiRequestError, requestJson } from '@/api/request';
import {
  parseContract,
  type ContractName,
  type StudioJobEvent,
  type StudioJobSubmissionRequest,
  type StudioRunPreviewRequest,
  type StudioSavedTrainRunRequest,
} from '@/generated/studioContracts';
import type { GraphSpec } from '@/types/graph';
import type { StudioWorkspaceSpec } from '@/types/workspace';

export type AuthoredStudioRunInput = Omit<StudioRunPreviewRequest, 'workspace' | 'graph'> & {
  workspace: StudioWorkspaceSpec;
  graph: GraphSpec;
};
type AuthoredSubmission = AuthoredStudioRunInput &
  Pick<StudioJobSubmissionRequest, 'token' | 'policy'>;

async function readContract<K extends ContractName>(
  path: string,
  contract: K,
  options?: RequestInit,
) {
  try {
    return parseContract(contract, await requestJson(path, { cache: 'no-store', ...options }));
  } catch (error) {
    throw asApiRequestError(error, path, 'The service response did not match its contract');
  }
}

export function previewStudioRuns(input: AuthoredStudioRunInput, signal?: AbortSignal) {
  const body = parseContract('StudioRunPreviewRequest', {
    ...input,
    workspace: semanticWorkspace(input.workspace),
  });
  return readContract('/api/submissions/preview', 'StudioRunPreview', {
    method: 'POST',
    body: JSON.stringify(body),
    signal,
  });
}

export function submitStudioJobs(input: AuthoredSubmission) {
  const body = parseContract('StudioJobSubmissionRequest', {
    ...input,
    workspace: semanticWorkspace(input.workspace),
  });
  return readContract('/api/submissions', 'StudioJobSubmission', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

function savedTrainRunsPath(project: string) {
  return `/api/projects/${encodeURIComponent(project)}/train-runs`;
}

export function fetchSavedTrainRuns(project: string, revision: string, signal?: AbortSignal) {
  const query = new URLSearchParams({ revision, window_map: 'true' });
  return readContract(`${savedTrainRunsPath(project)}?${query}`, 'StudioSavedTrainRuns', { signal });
}

export function runSavedTrain(project: string, input: StudioSavedTrainRunRequest) {
  const body = parseContract('StudioSavedTrainRunRequest', input);
  return readContract(savedTrainRunsPath(project), 'StudioJobSubmission', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function fetchStudioJobs(signal?: AbortSignal) {
  return readContract('/api/jobs', 'StudioJobSnapshot', { signal });
}

export async function fetchStudioJobHistory(
  projectId: string,
  after: number,
  through: number,
  signal?: AbortSignal,
): Promise<StudioJobEvent[]> {
  const events: StudioJobEvent[] = [];
  while (after < through) {
    const params = new URLSearchParams({
      project: projectId,
      after: String(after),
      through: String(through),
    });
    const path = `/api/events/history?${params}`;
    const page = await readContract(path, 'StudioJobEventHistory', { signal });
    if (page.cursor <= after || page.cursor > through) {
      throw asApiRequestError(null, path, 'The retained event range changed');
    }
    events.push(...page.events);
    after = page.cursor;
  }
  return events;
}

export async function stopStudioJobs(jobIds: string[], mode: 'checkpoint' | 'force') {
  const path = '/api/jobs/stop';
  const body = parseContract('StudioJobStopRequest', { job_ids: jobIds, mode });
  try {
    const result = await requestJson(path, { method: 'POST', body: JSON.stringify(body) });
    if (!Array.isArray(result)) throw new Error('Stop results must be a list');
    return result.map((item) => parseContract('StudioJobStopItem', item));
  } catch (error) {
    throw asApiRequestError(error, path, 'The stop response did not match its contract');
  }
}

export function resumeStudioJob(jobId: string, token: string) {
  const body = parseContract('StudioJobResumeRequest', { token });
  return readContract(`/api/jobs/${encodeURIComponent(jobId)}/resume`, 'StudioActivityJob', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function acknowledgeStudioJob(jobId: string) {
  return readContract(`/api/jobs/${encodeURIComponent(jobId)}/seen`, 'StudioActivityJob', {
    method: 'POST',
  });
}
