import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchComponents } from '@/api/client';
import { componentLibrary } from '@/data/components';
import { useGraphStore } from '@/stores/graphStore';

export function useComponents() {
  const query = useQuery({
    queryKey: ['components'],
    queryFn: fetchComponents,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
  const setCompositeTypes = useGraphStore((s) => s.setCompositeTypes);

  useEffect(() => {
    if (query.data) {
      const composites = new Set(
        query.data.filter((c) => c.is_composite).map((c) => c.name)
      );
      if (composites.size > 0) {
        setCompositeTypes(composites);
      }
    }
  }, [query.data, setCompositeTypes]);

  return {
    components: query.data ?? componentLibrary,
    isLoading: query.isLoading,
    error: query.error,
  };
}
