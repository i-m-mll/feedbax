import { useQuery } from '@tanstack/react-query';
import { fetchStudioSchemaRegistry } from '@/api/client';
import type { StudioWorkspaceSpec } from '@/types/workspace';
import { useRootGraph } from '@/hooks/useRootGraph';

export function useStudioSchemaRegistry(
  workspace: StudioWorkspaceSpec | null,
  scenarioId: string | null | undefined
) {
  const rootGraph = useRootGraph();
  return useQuery({
    queryKey: [
      'studio-schema-registry',
      workspace?.id ?? null,
      scenarioId ?? null,
      workspace,
      rootGraph,
    ],
    queryFn: () => {
      if (!workspace) {
        throw new Error('Missing Studio workspace');
      }
      return fetchStudioSchemaRegistry({
        workspace,
        graph: rootGraph,
        scenario_id: scenarioId ?? null,
      });
    },
    enabled: Boolean(workspace),
    staleTime: 30 * 1000,
    retry: 1,
  });
}
