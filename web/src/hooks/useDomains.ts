import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchDomains } from '@/api/client';
import {
  buildDomainContexts,
  resolveDomainContext,
  type DomainContext,
} from '@/features/domains/context';

export function useDomains() {
  const query = useQuery({
    queryKey: ['domains'],
    queryFn: fetchDomains,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });

  const domainContexts = useMemo(
    () => buildDomainContexts(query.data ?? []),
    [query.data]
  );
  const defaultDomainId = query.data?.[0]?.id ?? null;

  return {
    domains: query.data ?? [],
    domainContexts,
    defaultDomainId,
    domainContextFor: (contextId: string): DomainContext | null =>
      resolveDomainContext(contextId, domainContexts, defaultDomainId),
    isLoading: query.isLoading,
    error: query.error,
  };
}
