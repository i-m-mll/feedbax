import { useQuery } from '@tanstack/react-query';
import { fetchGraph, fetchGraphs } from '@/api/client';

export function useGraphsList(enabled = true) {
  return useQuery({
    queryKey: ['graphs'],
    queryFn: ({ signal }) => fetchGraphs(signal),
    enabled,
    networkMode: 'always',
    retry: false,
  });
}

export function useGraph(graphId: string | null) {
  return useQuery({
    queryKey: ['graph', graphId],
    queryFn: () => {
      if (!graphId) {
        throw new Error('Missing graph id');
      }
      return fetchGraph(graphId);
    },
    enabled: Boolean(graphId),
  });
}
