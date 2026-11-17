import { asApiRequestError, requestJson } from '@/api/request';
import { parseContract, type ValidityAcceptance, type ValidityAction } from '@/generated/studioContracts';

async function post(project: string, action: 'check' | 'accept', body: ValidityAction | ValidityAcceptance) {
  const path = `/api/projects/${encodeURIComponent(project)}/result-validity/${action}`;
  try {
    return parseContract('ReceiptValidity', await requestJson(path, { method: 'POST', body: JSON.stringify(body) }));
  } catch (error) {
    throw asApiRequestError(error, path, 'The validity response did not match its contract');
  }
}

export function checkResult(project: string, input: ValidityAction) {
  return post(project, 'check', parseContract('ValidityAction', input));
}

export function acceptResult(project: string, input: ValidityAcceptance) {
  return post(project, 'accept', parseContract('ValidityAcceptance', input));
}
