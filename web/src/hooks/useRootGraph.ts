import { useMemo } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import type { GraphSpec } from '@/types/graph';

export function useRootGraph(): GraphSpec {
  const graph = useGraphStore((state) => state.graph);
  const graphStack = useGraphStore((state) => state.graphStack);
  return useMemo(
    () => useGraphStore.getState().capturePersistedGraph().graph,
    [graph, graphStack]
  );
}
