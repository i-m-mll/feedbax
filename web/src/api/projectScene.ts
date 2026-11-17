import { asApiRequestError, requestJson } from '@/api/request';
import { parseContract, type StudioPlanarSceneRequest } from '@/generated/studioContracts';

export async function fetchProjectPlanarScene(id: string, source: StudioPlanarSceneRequest, signal: AbortSignal) {
  const path = `/api/projects/${encodeURIComponent(id)}/scene/planar`;
  try {
    const body = parseContract('StudioPlanarSceneRequest', source);
    const result = parseContract('StudioPlanarSceneResponse', await requestJson(path, {
      method: 'POST', body: JSON.stringify(body), cache: 'no-store', signal,
    }));
    if (result.project_id !== id || result.revision !== source.revision || result.document_id !== source.document_id ||
        result.node_path.length !== source.node_path.length || result.node_path.some((node, index) => node !== source.node_path[index])) {
      throw new Error('Scene response refers to a different authored model.');
    }
    return result;
  } catch (error) {
    throw asApiRequestError(error, path, 'The Scene response did not match its contract');
  }
}
