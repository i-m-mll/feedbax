import { useQuery } from '@tanstack/react-query';
import { checkStudioConnection } from '@/api/client';

export function useStudioConnection() {
  return useQuery({
    queryKey: ['studio-connection'],
    queryFn: ({ signal }) => checkStudioConnection(signal),
    networkMode: 'always',
    refetchInterval: 3000,
    retry: false,
  });
}
