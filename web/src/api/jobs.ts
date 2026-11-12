import { asApiRequestError, requestJson } from '@/api/request';
import {
  parseContract,
  type ContractName,
  type StudioJobEvent,
  type StudioSavedEvaluationReplayRequest,
  type StudioSavedTrainRunRequest,
} from '@/generated/studioContracts';

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

export function fetchSavedEvaluationReplay(project: string, input: StudioSavedEvaluationReplayRequest, signal?: AbortSignal) {
  const body = parseContract('StudioSavedEvaluationReplayRequest', input);
  return readContract(`/api/projects/${encodeURIComponent(project)}/evaluate-replay`, 'WorkspaceReplayTrial', {
    method: 'POST',
    body: JSON.stringify(body),
    signal,
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
