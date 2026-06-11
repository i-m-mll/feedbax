import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { createGraph, fetchGraph, fetchGraphs, updateGraph } from '@/api/client';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import { useAnalysisStore } from '@/stores/analysisStore';
import { useGraphStore } from '@/stores/graphStore';
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
    }: {
      graphId: string | null;
      graph: GraphSpec;
      uiState: GraphUIState | null;
    }) => {
      const graphStore = useGraphStore.getState();
      const persistedGraph = graphStore.capturePersistedGraph();
      const workspace = buildWorkspaceSnapshot({
        workspace: useWorkspaceStore.getState().workspace,
        graph: persistedGraph.graph,
        uiState: persistedGraph.uiState,
        trainingSpec: useTrainingStore.getState().trainingSpec,
        taskSpec: useTrainingStore.getState().taskSpec,
        analysisSnapshot: useAnalysisStore.getState().captureSnapshot(),
        graphStackPath: persistedGraph.graphStackPath,
      });
      useWorkspaceStore.getState().setWorkspace(workspace);
      if (graphId) {
        return updateGraph(
          graphId,
          persistedGraph.graph,
          persistedGraph.uiState,
          undefined,
          undefined,
          workspace
        );
      }
      return createGraph(persistedGraph.graph, persistedGraph.uiState, workspace);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['graphs'] });
    },
  });
}
