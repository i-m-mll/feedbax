import { asApiRequestError, requestJson, requestResponse, sentBody } from '@/api/request';
import {
  parseContract,
  type ContractName,
  type ContractTypeMap,
  type StudioJobEvent,
  type StudioMachineApproval,
  type StudioSavedEvaluationRead,
  type StudioSavedEvaluationReplayRequest,
  type StudioSavedMachineQuoteRequest,
  type StudioSavedTrainRunRequest,
  type StudioVenueStream,
  type TargetOption,
} from '@/generated/studioContracts';

async function readContract<K extends ContractName, S extends ContractName>(
  path: string,
  contract: K,
  options?: RequestInit,
  sent?: { contract: S; body: ContractTypeMap[S] },
) {
  try {
    const body = sent && sentBody(sent.contract, sent.body, path);
    return parseContract(contract, await requestJson(path, { cache: 'no-store', ...options, ...sent && { method: 'POST', body } }));
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

function savedTrainRunsPath(project: string) {
  return `/api/projects/${encodeURIComponent(project)}/train-runs`;
}

export function fetchSavedTrainRuns(project: string, revision: string, signal?: AbortSignal, stoppedCheckpointJob?: string) {
  const query = new URLSearchParams({ revision, window_map: 'true' });
  if (stoppedCheckpointJob) query.set('stopped_checkpoint_job', stoppedCheckpointJob);
  return readContract(`${savedTrainRunsPath(project)}?${query}`, 'StudioSavedTrainRuns', { signal });
}

export function runSavedTrain(project: string, input: StudioSavedTrainRunRequest) {
  return readContract(savedTrainRunsPath(project), 'StudioJobSubmission', undefined, { contract: 'StudioSavedTrainRunRequest', body: input });
}

export function fetchExecutionTargets(kinds: readonly TargetOption['run_kinds'][number][], signal?: AbortSignal) {
  const query = new URLSearchParams(kinds.map((kind) => ['kind', kind]));
  return readContract(`/api/orchestration/targets?${query}`, 'ExecutionTargets', { signal });
}

export function quoteSavedRun(project: string, input: StudioSavedMachineQuoteRequest) {
  return readContract(`${savedTrainRunsPath(project)}/quote`, 'StudioMachineQuote', undefined, { contract: 'StudioSavedMachineQuoteRequest', body: input });
}

export function approveSavedRun(project: string, input: StudioMachineApproval) {
  return readContract(`${savedTrainRunsPath(project)}/approve`, 'StudioMachineApprovalReceipt', undefined, { contract: 'StudioMachineApproval', body: input });
}

export function fetchSavedEvaluationReplay(project: string, input: StudioSavedEvaluationReplayRequest, signal?: AbortSignal) {
  return readContract(`/api/projects/${encodeURIComponent(project)}/evaluate-replay`, 'StudioTrialReplayResponse', { signal }, { contract: 'StudioSavedEvaluationReplayRequest', body: input });
}

export function fetchSavedEvaluationObjective(project: string, input: StudioSavedEvaluationRead, signal?: AbortSignal) {
  return readContract(`/api/projects/${encodeURIComponent(project)}/evaluate-objective`, 'StudioEvaluationObjectiveSeries', { signal },
    { contract: 'StudioSavedEvaluationRead', body: input });
}

export function fetchStudioJobs(signal?: AbortSignal) {
  return readContract('/api/jobs', 'StudioJobSnapshot', { signal });
}

export async function warmStudioWorkers() {
  await requestResponse('/api/jobs/warm', { method: 'POST' });
}

export async function closeStudioMachine(id: string) {
  await requestResponse(`/api/sessions/${encodeURIComponent(id)}/close`, { method: 'POST' });
}

export async function fetchStudioJobHistory(
  projectId: string,
  after: number,
  through: number,
  signal?: AbortSignal,
  venue?: StudioVenueStream,
): Promise<StudioJobEvent[]> {
  const events: StudioJobEvent[] = [];
  while (after < through) {
    const params = new URLSearchParams({
      project: projectId,
      after: String(after),
      through: String(through),
    });
    if (venue) {
      params.set('venue', venue.session_id);
      params.set('stream', venue.stream_id);
    }
    const path = `/api/events/history?${params}`;
    const page = await readContract(path, 'StudioJobEventHistory', { signal });
    if (page.cursor <= after || page.cursor > through) {
      throw asApiRequestError(null, path);
    }
    events.push(...page.events);
    after = page.cursor;
  }
  return events;
}

export async function stopStudioJobs(jobIds: string[], mode: 'checkpoint' | 'force') {
  const path = '/api/jobs/stop';
  try {
    const body = sentBody('StudioJobStopRequest', { job_ids: jobIds, mode }, path);
    const result = await requestJson(path, { method: 'POST', body });
    if (!Array.isArray(result)) throw new Error('Stop results must be a list');
    return result.map((item) => parseContract('StudioJobStopItem', item));
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export function resumeStudioJob(jobId: string, token: string) {
  return readContract(`/api/jobs/${encodeURIComponent(jobId)}/resume`, 'StudioActivityJob', undefined, { contract: 'StudioJobResumeRequest', body: { token } });
}

export function acknowledgeStudioJob(jobId: string) {
  return readContract(`/api/jobs/${encodeURIComponent(jobId)}/seen`, 'StudioActivityJob', {
    method: 'POST',
  });
}
