import { asApiRequestError, requestJson, sentBody } from '@/api/request';
import {
  parseContract,
  type ContractTypeMap,
  type EvaluateModelRequest,
  type ModelConnectionReviewRequest,
  type ModelEditRequest,
  type ModelFieldRequest,
  type ModelProjectionRequest,
  type ModelUndoRequest,
  type StudioMemberCommand,
  type StudioPresentationWrite,
  type StudioProjectCreateFile,
  type StudioProjectCreateFiles,
  type StudioProjectPatch,
  type StudioProjectPatchBatch,
  type StudioProjectRename,
  type StudioProjectReplay,
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
    throw asApiRequestError(error, path);
  }
}

export async function listProjectFolders(signal?: AbortSignal) {
  try {
    return parseContract('StudioProjectList', await requestJson(projectsPath, { signal, cache: 'no-store' })).projects;
  } catch (error) {
    throw asApiRequestError(error, projectsPath);
  }
}

export function createProjectFolder(signal?: AbortSignal) {
  return readFolder('StudioProjectDocument', undefined, projectsPath, { method: 'POST', signal });
}

export function openProjectFolder(id: string, signal?: AbortSignal) {
  return readFolder('StudioProjectDocument', undefined, `${projectPath(id)}/open`, { signal });
}

export function readProjectFolder(id: string, signal?: AbortSignal) {
  return readFolder('StudioProjectDocument', id, `${projectPath(id)}/open`, { signal });
}

export function renameProjectFolder(id: string, body: StudioProjectRename, signal?: AbortSignal) {
  const path = `${projectPath(id)}/rename`;
  return readFolder('StudioProjectDocument', id, path, {
    method: 'POST', body: sentBody('StudioProjectRename', body, path), signal,
  });
}

export function createProjectFile(id: string, body: StudioProjectCreateFile | StudioProjectCreateFiles, signal?: AbortSignal) {
  const path = `${projectPath(id)}/files`;
  const data = 'files' in body ? sentBody('StudioProjectCreateFiles', body, path)
    : sentBody('StudioProjectCreateFile', body, path);
  return readFolder('StudioProjectDocument', id, path, {
    method: 'POST', body: data, signal,
  });
}

export async function commandProjectMember(id: string, body: StudioMemberCommand, signal?: AbortSignal) {
  const path = `${projectPath(id)}/members/command`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: sentBody('StudioMemberCommand', body, path), signal, cache: 'no-store',
    });
    const result = parseContract('StudioMemberCommandResult', response);
    if (result.snapshot.project_id !== id) throw new Error('Response project ID does not match the folder');
    // Keep authored JSON as received, as readFolder does.
    return response as typeof result;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export async function patchProjectFolder(id: string, body: StudioProjectPatch | StudioProjectPatchBatch, signal?: AbortSignal) {
  const path = projectPath(id);
  const payload = 'files' in body ? sentBody('StudioProjectPatchBatch', body, path, 'PATCH') : sentBody('StudioProjectPatch', body, path, 'PATCH');
  return readFolder('StudioProjectSnapshot', id, path, { method: 'PATCH', body: payload, signal });
}

export function replayProjectHistory(id: string, body: StudioProjectReplay, signal?: AbortSignal) {
  const path = `${projectPath(id)}/history/replay`;
  return readFolder('StudioProjectSnapshot', id, path, {
    method: 'POST', body: sentBody('StudioProjectReplay', body, path), signal,
  });
}

export async function writeProjectPresentation(id: string, body: StudioPresentationWrite, signal?: AbortSignal) {
  const path = `${projectPath(id)}/presentation`;
  return readFolder('StudioProjectSnapshot', id, path, { method: 'PUT', body: sentBody('StudioPresentationWrite', body, path, 'PUT'), signal });
}

type ModelRequest = 'ModelEditRequest' | 'ModelUndoRequest' | 'ModelConnectionReviewRequest' | 'ModelProjectionRequest' | 'ModelFieldRequest' | 'EvaluateModelRequest';
type ModelResponse = 'ModelEditResponse' | 'ModelConnectionReview' | 'ModelProjection' | 'ModelFieldResponse' | 'EvaluateModelState';

async function modelRequest<K extends ModelRequest, R extends ModelResponse>(
  id: string, action: string, request: K, response: R, body: ContractTypeMap[K], signal?: AbortSignal,
): Promise<ContractTypeMap[R]> {
  const path = `${projectPath(id)}/model/${action}`;
  try {
    const raw = await requestJson(path, {
      method: 'POST', body: sentBody(request, body, path), signal, cache: 'no-store',
    });
    parseContract(response, raw);
    return raw as ContractTypeMap[R];
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export function editModel(id: string, body: ModelEditRequest, signal?: AbortSignal) {
  return modelRequest(id, 'edit', 'ModelEditRequest', 'ModelEditResponse', body, signal);
}

export function undoModel(id: string, body: ModelUndoRequest, signal?: AbortSignal) {
  return modelRequest(id, 'undo', 'ModelUndoRequest', 'ModelEditResponse', body, signal);
}

export function reviewModelConnections(id: string, body: ModelConnectionReviewRequest, signal?: AbortSignal) {
  return modelRequest(id, 'connections', 'ModelConnectionReviewRequest', 'ModelConnectionReview', body, signal);
}

export function readModelProjection(id: string, body: ModelProjectionRequest, signal?: AbortSignal) {
  return modelRequest(id, 'projection', 'ModelProjectionRequest', 'ModelProjection', body, signal);
}

export function projectModelFields(id: string, body: ModelFieldRequest, signal?: AbortSignal) {
  return modelRequest(id, 'fields', 'ModelFieldRequest', 'ModelFieldResponse', body, signal);
}

export function readEvaluateModel(id: string, body: EvaluateModelRequest, signal?: AbortSignal) {
  return modelRequest(id, 'evaluate-model', 'EvaluateModelRequest', 'EvaluateModelState', body, signal);
}

