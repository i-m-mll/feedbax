import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { createGraph, fetchGraph, fetchGraphs, updateGraph } from '@/api/client';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useTrainingStore } from '@/stores/trainingStore';
import { buildWorkspaceSnapshot, useWorkspaceStore } from '@/stores/workspaceStore';

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

export function useSaveGraph() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      graphId,
      graph,
      uiState,
    }: {
      graphId: string | null;
      graph: GraphSpec;
      uiState: GraphUIState | null;
    }) => {
      const workspace = buildWorkspaceSnapshot({
        workspace: useWorkspaceStore.getState().workspace,
        graph,
        uiState,
        trainingSpec: useTrainingStore.getState().trainingSpec,
        taskSpec: useTrainingStore.getState().taskSpec,
        analysisSnapshot: useAnalysisStore.getState().captureSnapshot(),
      });
      useWorkspaceStore.getState().setWorkspace(workspace);
      if (graphId) {
        return updateGraph(graphId, graph, uiState, undefined, undefined, workspace);
      }
      return createGraph(graph, uiState, workspace);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graphs'] });
    },
  });
}
