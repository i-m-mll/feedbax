import { parseContract } from '@/generated/studioContracts';
import { asApiRequestError, requestJson } from './request';

export async function fetchComponents() {
  const path = '/api/components';
  try {
    const response = parseContract('ComponentListResponse', await requestJson(path));
    return response.data.components;
  } catch (error) {
    throw asApiRequestError(error, path, 'The component catalog did not match its contract');
  }
}
