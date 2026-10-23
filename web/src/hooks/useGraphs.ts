import { useQuery } from '@tanstack/react-query';
import { fetchGraph, fetchGraphs } from '@/api/client';

export function useGraphsList() {
  return useQuery({
    queryKey: ['graphs'],
    queryFn: fetchGraphs,
    staleTime: 30 * 1000,
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
