import { useQuery } from '@tanstack/react-query';
import { fetchStudioSchemaRegistry } from '@/api/client';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import { useGraphStore } from '@/stores/graphStore';

export function useStudioSchemaRegistry(
  workspace: StudioWorkspaceSpec | null,
  scenarioId: string | null | undefined
) {
  const graph = useGraphStore((state) => state.graph);
  return useQuery({
    queryKey: [
      'studio-schema-registry',
      workspace?.id ?? null,
      scenarioId ?? null,
      workspace,
      graph,
    ],
    queryFn: () => {
      if (!workspace) {
        throw new Error('Missing Studio workspace');
      }
      return fetchStudioSchemaRegistry({
        workspace,
        graph,
        scenario_id: scenarioId ?? null,
      });
    },
    enabled: Boolean(workspace),
    staleTime: 30 * 1000,
    retry: 1,
  });
}
