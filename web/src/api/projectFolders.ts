import { asApiRequestError, requestJson } from '@/api/request';
import {
  parseContract,
  type ContractTypeMap,
  type StudioPresentationWrite,
  type StudioProjectCreateFile,
  type StudioProjectPatch,
  type StudioProjectRename,
} from '@/generated/studioContracts';

const projectsPath = '/api/projects';

function projectPath(id: string) {
  return `${projectsPath}/${encodeURIComponent(id)}`;
}

async function readFolder<K extends 'StudioProjectSnapshot' | 'StudioProjectDocument'>(
  contract: K, id: string | undefined, path: string, options: RequestInit,
): Promise<ContractTypeMap[K]> {
  try {
    const response = await requestJson(path, { ...options, cache: 'no-store' });
    const actualId = contract === 'StudioProjectDocument'
      ? parseContract('StudioProjectDocument', response).snapshot.project_id
      : parseContract('StudioProjectSnapshot', response).project_id;
    if (id !== undefined && actualId !== id) throw new Error('Response project ID does not match the folder');
    // Keep authored JSON as received. Zod's reconstructed records can discard own keys such as "__proto__".
    return response as ContractTypeMap[K];
  } catch (error) {
    throw asApiRequestError(error, path, 'The project response did not match its contract');
  }
}

function mutationBody<K extends 'StudioProjectPatch' | 'StudioPresentationWrite' | 'StudioProjectRename' | 'StudioProjectCreateFile'>(
  contract: K, body: ContractTypeMap[K], path: string,
) {
  try {
    parseContract(contract, body);
    return JSON.stringify(body);
  } catch (error) {
    throw asApiRequestError(error, path, 'The project write did not match its contract');
  }
}

export function createProjectFolder(signal?: AbortSignal) {
  return readFolder('StudioProjectDocument', undefined, projectsPath, { method: 'POST', signal });
}

export function openProjectFolder(id: string, signal?: AbortSignal) {
  return readFolder('StudioProjectDocument', id, `${projectPath(id)}/open`, { signal });
}

export function renameProjectFolder(id: string, body: StudioProjectRename, signal?: AbortSignal) {
  const path = `${projectPath(id)}/rename`;
  return readFolder('StudioProjectDocument', id, path, {
    method: 'POST', body: mutationBody('StudioProjectRename', body, path), signal,
  });
}

export function createProjectFile(id: string, body: StudioProjectCreateFile, signal?: AbortSignal) {
  const path = `${projectPath(id)}/files`;
  return readFolder('StudioProjectDocument', id, path, {
    method: 'POST', body: mutationBody('StudioProjectCreateFile', body, path), signal,
  });
}

export async function patchProjectFolder(id: string, body: StudioProjectPatch, signal?: AbortSignal) {
  const path = projectPath(id);
  return readFolder('StudioProjectSnapshot', id, path, { method: 'PATCH', body: mutationBody('StudioProjectPatch', body, path), signal });
}

export async function writeProjectPresentation(id: string, body: StudioPresentationWrite, signal?: AbortSignal) {
  const path = `${projectPath(id)}/presentation`;
  return readFolder('StudioProjectSnapshot', id, path, { method: 'PUT', body: mutationBody('StudioPresentationWrite', body, path), signal });
}
