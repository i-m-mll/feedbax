import { asApiRequestError, requestJson, sentBody } from '@/api/request';
import { parseContract, type StudioSceneGeometryRequest } from '@/generated/studioContracts';

export async function fetchProjectSceneGeometry(id: string, source: StudioSceneGeometryRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(id)}/scene/geometry`;
  try {
    const result = parseContract('StudioSceneGeometryResponse', await requestJson(path, {
      method: 'POST', body: sentBody('StudioSceneGeometryRequest', source, path), cache: 'no-store', signal,
    }));
    if (result.project_id !== id || result.revision !== source.revision || result.document_id !== source.document_id) {
      throw new Error('Scene response refers to a different authored model.');
    }
    return result;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}
