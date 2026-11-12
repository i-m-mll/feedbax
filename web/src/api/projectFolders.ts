import { asApiRequestError, requestJson } from '@/api/request';
import {
  parseContract,
  type ContractTypeMap,
  type StudioPresentationWrite,
  type StudioProjectPatch,
  type StudioProjectSnapshot,
} from '@/generated/studioContracts';

function projectPath(id: string) {
  return `/api/projects/${encodeURIComponent(id)}`;
}

async function readSnapshot(id: string, path: string, options: RequestInit): Promise<StudioProjectSnapshot> {
  try {
    const response = await requestJson(path, { ...options, cache: 'no-store' });
    const snapshot = parseContract('StudioProjectSnapshot', response);
    if (snapshot.project_id !== id) throw new Error('Response project ID does not match the folder');
    // Keep authored JSON as received. Zod's reconstructed records can discard own keys such as "__proto__".
    return response as StudioProjectSnapshot;
  } catch (error) {
    throw asApiRequestError(error, path, 'The project response did not match its contract');
  }
}

function mutationBody<K extends 'StudioProjectPatch' | 'StudioPresentationWrite'>(
  contract: K, body: ContractTypeMap[K], path: string,
) {
  try {
    parseContract(contract, body);
    return JSON.stringify(body);
  } catch (error) {
    throw asApiRequestError(error, path, 'The project write did not match its contract');
  }
}

export function fetchProjectFolder(id: string, signal?: AbortSignal) {
  return readSnapshot(id, projectPath(id), { signal });
}

export async function patchProjectFolder(id: string, body: StudioProjectPatch, signal?: AbortSignal) {
  const path = projectPath(id);
  return readSnapshot(id, path, { method: 'PATCH', body: mutationBody('StudioProjectPatch', body, path), signal });
}

export async function writeProjectPresentation(id: string, body: StudioPresentationWrite, signal?: AbortSignal) {
  const path = `${projectPath(id)}/presentation`;
  return readSnapshot(id, path, { method: 'PUT', body: mutationBody('StudioPresentationWrite', body, path), signal });
}
