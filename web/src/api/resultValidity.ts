import { asApiRequestError, requestJson, sentBody } from '@/api/request';
import { parseContract, type ValidityAcceptance, type ValidityAction } from '@/generated/studioContracts';

async function post(project: string, action: 'check' | 'accept', body: ValidityAction | ValidityAcceptance) {
  const path = `/api/projects/${encodeURIComponent(project)}/result-validity/${action}`;
  try {
    const sent = action === 'check' ? sentBody('ValidityAction', body as ValidityAction, path)
      : sentBody('ValidityAcceptance', body as ValidityAcceptance, path);
    return parseContract('ReceiptValidity', await requestJson(path, { method: 'POST', body: sent }));
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

export function checkResult(project: string, input: ValidityAction) {
  return post(project, 'check', input);
}

export function acceptResult(project: string, input: ValidityAcceptance) {
  return post(project, 'accept', input);
}
