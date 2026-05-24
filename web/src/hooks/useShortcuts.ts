import { useEffect, useCallback } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useSaveGraph } from '@/hooks/useGraphs';
import {
  ensureTaskBindingSpec,
  scopedTaskBindingSpec,
  taskBindingInGraphPath,
} from '@/features/scenario/taskBindings';

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || target.isContentEditable;
}

export function useAppShortcuts() {
  const { undo, redo, deleteSelected, graph, uiState, graphId, markSaved, markDirty, nodes, graphStack } =
    useGraphStore();
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateTaskBindingSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioTaskBindingSpec
  );
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
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
    const trainingScenario = getTrainingScenario(workspace);
    const rootGraph = graphStack.length > 0 ? graphStack[0].graph : graph;
    const taskBindingSpec = trainingScenario
      ? ensureTaskBindingSpec(trainingScenario.task_binding_spec, rootGraph, trainingScenario.task_spec)
      : null;
    const currentGraphPath = graphStack
      .map((layer) => layer.childNodeId)
      .filter((id): id is string => Boolean(id));
    const scopedBindings = taskBindingSpec
      ? scopedTaskBindingSpec(taskBindingSpec, currentGraphPath).bindings
      : [];
    const selectedTaskBindingId = topPane.selected_entity_id?.startsWith('task_binding:')
      ? topPane.selected_entity_id.slice('task_binding:'.length)
      : null;
    if (taskBindingSpec && selectedTaskBindingId) {
      updateTaskBindingSpec({
        ...taskBindingSpec,
        bindings: taskBindingSpec.bindings.filter(
          (binding) => binding.id !== selectedTaskBindingId
        ),
      });
      selectTopPaneEntity(null);
      markDirty();
      return;
    }
    const impactedBindings = scopedBindings.filter((binding) =>
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
        bindings: taskBindingSpec.bindings.filter(
          (binding) =>
            !impactedIds.has(binding.id) || !taskBindingInGraphPath(binding, currentGraphPath)
        ),
      });
    }

    deleteSelected();
  }, [
    deleteSelected,
    graph,
    graphStack,
    markDirty,
    nodes,
    selectTopPaneEntity,
    updateTaskBindingSpec,
    workspace,
  ]);

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
