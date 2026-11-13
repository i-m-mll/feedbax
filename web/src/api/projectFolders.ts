import { asApiRequestError, requestJson } from '@/api/request';
import {
  parseContract,
  type ContractTypeMap,
  type EvaluateAttachmentReadRequest,
  type EvaluateAttachmentRequest,
  type ModelConnectionReviewRequest,
  type ModelEditRequest,
  type ModelUndoRequest,
  type StudioDocumentAuthoringRequest,
  type StudioPresentationWrite,
  type StudioProjectCreateFile,
  type StudioProjectPatch,
  type StudioProjectPatchBatch,
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

function mutationBody<K extends 'StudioProjectPatch' | 'StudioProjectPatchBatch' | 'StudioDocumentAuthoringRequest' | 'StudioPresentationWrite' | 'StudioProjectRename' | 'StudioProjectCreateFile' | ModelRequest>(
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

export async function patchProjectFolder(id: string, body: StudioProjectPatch | StudioProjectPatchBatch, signal?: AbortSignal) {
  const path = projectPath(id);
  const payload = 'files' in body ? mutationBody('StudioProjectPatchBatch', body, path) : mutationBody('StudioProjectPatch', body, path);
  return readFolder('StudioProjectSnapshot', id, path, { method: 'PATCH', body: payload, signal });
}

export async function fetchDocumentAuthoring(id: string, body: StudioDocumentAuthoringRequest, signal?: AbortSignal) {
  const path = `${projectPath(id)}/authoring/document`;
  try {
    const response = await requestJson(path, {
      method: 'POST', body: mutationBody('StudioDocumentAuthoringRequest', body, path), signal, cache: 'no-store',
    });
    const definition = parseContract('StudioDocumentAuthoring', response);
    if (definition.revision !== body.revision) throw new Error('Document definition has the wrong revision');
    return definition;
  } catch (error) {
    throw asApiRequestError(error, path, 'The document definition did not match its contract');
  }
}

export async function writeProjectPresentation(id: string, body: StudioPresentationWrite, signal?: AbortSignal) {
  const path = `${projectPath(id)}/presentation`;
  return readFolder('StudioProjectSnapshot', id, path, { method: 'PUT', body: mutationBody('StudioPresentationWrite', body, path), signal });
}

type ModelRequest = 'ModelEditRequest' | 'ModelUndoRequest' | 'ModelConnectionReviewRequest' | 'EvaluateAttachmentReadRequest' | 'EvaluateAttachmentRequest';
type ModelResponse = 'ModelEditResponse' | 'ModelConnectionReview' | 'EvaluateAttachmentState' | 'EvaluateAttachmentEdit';

async function modelRequest<K extends ModelRequest, R extends ModelResponse>(
  id: string, action: string, request: K, response: R, body: ContractTypeMap[K], signal?: AbortSignal,
): Promise<ContractTypeMap[R]> {
  const path = `${projectPath(id)}/model/${action}`;
  try {
    const raw = await requestJson(path, {
      method: 'POST', body: mutationBody(request, body, path), signal, cache: 'no-store',
    });
    parseContract(response, raw);
    return raw as ContractTypeMap[R];
  } catch (error) {
    throw asApiRequestError(error, path, 'The Model response did not match its contract');
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

export function readEvaluateAttachments(id: string, body: EvaluateAttachmentReadRequest, signal?: AbortSignal) {
  return modelRequest(id, 'evaluate-attachments', 'EvaluateAttachmentReadRequest', 'EvaluateAttachmentState', body, signal);
}

export function editEvaluateAttachment(id: string, body: EvaluateAttachmentRequest, signal?: AbortSignal) {
  return modelRequest(id, 'evaluate-attachment', 'EvaluateAttachmentRequest', 'EvaluateAttachmentEdit', body, signal);
}
