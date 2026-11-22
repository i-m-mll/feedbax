import { asApiRequestError, requestJson, sentBody } from '@/api/request';
import { parseContract, type StudioAnalyzeRequest } from '@/generated/studioContracts';

export async function readStudioAnalyze(projectId: string, body: StudioAnalyzeRequest, signal?: AbortSignal) {
  const path = '/api/projects/' + encodeURIComponent(projectId) + '/authoring/analyze';
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioAnalyzeRequest', body, path), cache: 'no-store', signal,
    });
    const reviewed = parseContract('StudioAnalyzeAuthoring', response);
    if (reviewed.revision !== body.revision) throw new Error('Analyze revision changed');
    return reviewed;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}
