import { useQuery } from '@tanstack/react-query';
import { fetchComponents } from '@/api/client';
import { componentLibrary } from '@/data/components';

export function useComponents() {
  const query = useQuery({
    queryKey: ['components'],
    queryFn: fetchComponents,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });

  return {
    components: query.data ?? componentLibrary,
    isLoading: query.isLoading,
    error: query.error,
  };
}
