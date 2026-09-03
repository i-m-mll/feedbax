import { useEffect, useCallback } from 'react';
import { toast } from 'sonner';
import { useShallow } from 'zustand/react/shallow';
import { useGraphStore } from '@/stores/graphStore';
import { actionErrorMessage } from '@/stores/storeActions';
import {
  getTopPaneState,
  getTrainingScenario,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { saveActiveStudioDocument } from '@/services/studioPersistence';
import {
  ensureTaskBindingSpec,
  scopedTaskBindingSpec,
  taskBindingInGraphPath,
} from '@/features/scenario/taskBindings';

export const FIT_VIEW_SHORTCUT_EVENT = 'feedbax:shortcut-fit-view';

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || target.isContentEditable;
}

function hasOwnedInteractionPath(event: KeyboardEvent) {
  return event.composedPath().some(
    (target) =>
      target instanceof Element && target.closest('[data-studio-interaction-owner]') !== null,
  );
}

export function useAppShortcuts() {
  const {
    undo,
    redo,
    deleteSelected,
    duplicateSelected,
    clearSelection,
    selectAll,
    graph,
    markDirty,
    nodes,
    graphStack,
  } = useGraphStore(
    useShallow((state) => ({
      undo: state.undo,
      redo: state.redo,
      deleteSelected: state.deleteSelected,
      duplicateSelected: state.duplicateSelected,
      clearSelection: state.clearSelection,
      selectAll: state.selectAll,
      graph: state.graph,
      markDirty: state.markDirty,
      nodes: state.nodes,
      graphStack: state.graphStack,
    }))
  );
  const { workspace, updateTaskBindingSpec, selectTopPaneEntity } = useWorkspaceStore(
    useShallow((state) => ({
      workspace: state.workspace,
      updateTaskBindingSpec: state.updateActiveScenarioTaskBindingSpec,
      selectTopPaneEntity: state.selectTopPaneEntity,
    }))
  );
  const saveGraph = useCallback(async () => {
    try {
      const outcome = await saveActiveStudioDocument('shortcut');
      if (!outcome.ok) return;
      toast.success('Project saved.', { id: 'project-save-success' });
    } catch (error) {
      toast.error(actionErrorMessage(error, 'Failed to save project.'), {
        id: 'project-save-error',
      });
    }
  }, []);

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
    if (selectedNodeIds.length > 0) {
      toast.success(
        selectedNodeIds.length === 1
          ? 'Node deleted - Cmd+Z to undo.'
          : 'Nodes deleted - Cmd+Z to undo.',
        { id: 'node-delete-success' },
      );
    }
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

  const duplicateSelection = useCallback(() => {
    try {
      duplicateSelected();
    } catch (error) {
      toast.error(actionErrorMessage(error, 'Failed to duplicate selection.'), {
        id: 'node-duplicate-error',
      });
    }
  }, [duplicateSelected]);

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

      if (isMod && event.key.toLowerCase() === 'd') {
        event.preventDefault();
        duplicateSelection();
        return;
      }

      if (isMod && event.key.toLowerCase() === 'a') {
        event.preventDefault();
        selectTopPaneEntity(null);
        selectAll();
        return;
      }

      if (isMod && event.key === '0') {
        event.preventDefault();
        window.dispatchEvent(new CustomEvent(FIT_VIEW_SHORTCUT_EVENT));
        return;
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        clearSelection();
        selectTopPaneEntity(null);
        return;
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault();
        if (hasOwnedInteractionPath(event)) return;
        deleteSelection();
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [
    clearSelection,
    deleteSelection,
    duplicateSelection,
    redo,
    saveGraph,
    selectAll,
    selectTopPaneEntity,
    undo,
  ]);
}
