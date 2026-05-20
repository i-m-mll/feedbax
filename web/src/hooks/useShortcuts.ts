import { useEffect, useCallback } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useSaveGraph } from '@/hooks/useGraphs';

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || target.isContentEditable;
}

export function useAppShortcuts() {
  const { undo, redo, deleteSelected, graph, uiState, graphId, markSaved, nodes } =
    useGraphStore();
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateTaskBindingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTaskBindingSpec
  );
  const saveMutation = useSaveGraph();

  const saveGraph = useCallback(async () => {
    const response = await saveMutation.mutateAsync({ graphId, graph, uiState });
    if ('id' in response) {
      markSaved(response.id);
    } else if (graphId) {
      markSaved(graphId);
    }
  }, [graphId, graph, uiState, markSaved, saveMutation]);

  const deleteSelection = useCallback(() => {
    const selectedNodeIds = nodes
      .filter((node) => node.selected && !node.id.startsWith('tap:'))
      .map((node) => node.id);
    const topPane = getTopPaneState(workspace);
    const taskBindingSpec = getTrainingScenario(workspace)?.task_binding_spec;
    const impactedBindings = (taskBindingSpec?.bindings ?? []).filter((binding) =>
      selectedNodeIds.includes(binding.target_node_id)
    );

    if (
      topPane.active_projection === 'model' &&
      impactedBindings.length > 0 &&
      !window.confirm(
        `Delete selected node? It has ${impactedBindings.length} task binding${
          impactedBindings.length === 1 ? '' : 's'
        } wired into it.`
      )
    ) {
      return;
    }

    if (taskBindingSpec && impactedBindings.length > 0) {
      const impactedIds = new Set(impactedBindings.map((binding) => binding.id));
      updateTaskBindingSpec({
        ...taskBindingSpec,
        bindings: taskBindingSpec.bindings.filter((binding) => !impactedIds.has(binding.id)),
      });
    }

    deleteSelected();
  }, [deleteSelected, nodes, updateTaskBindingSpec, workspace]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;
      const isMod = event.metaKey || event.ctrlKey;

      if (isMod && event.key.toLowerCase() === 's') {
        event.preventDefault();
        saveGraph();
        return;
      }

      if (isMod && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
        return;
      }

      if (isMod && event.key.toLowerCase() === 'y') {
        event.preventDefault();
        redo();
        return;
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        deleteSelection();
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [deleteSelection, redo, undo, saveGraph]);
}
