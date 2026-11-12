import { asApiRequestError, requestJson } from '@/api/request';
import {
  parseContract,
  type StudioTrainAuthoring,
  type StudioTrainAuthoringRequest,
} from '@/generated/studioContracts';

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
