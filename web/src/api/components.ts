import { parseContract } from '@/generated/studioContracts';
import { asApiRequestError, requestJson } from './request';

export async function fetchComponents() {
  const path = '/api/components';
  try {
    const response = parseContract('ComponentListResponse', await requestJson(path));
    return response.data.components;
  } catch (error) {
    throw asApiRequestError(error, path);
  }
}

/** The registered component types, read once per five minutes by every view that names their ports. */
export const componentsQuery = { queryKey: ['components'], queryFn: fetchComponents, staleTime: 300_000 } as const;
