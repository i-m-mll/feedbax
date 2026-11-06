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
import { prepareRootTaskBindingRemoval } from '@/services/studioTaskBindings';
import { ensureTaskBindingSpec } from '@/features/scenario/taskBindings';

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
    duplicateSelected,
    clearSelection,
    selectAll,
  } = useGraphStore(
    useShallow((state) => ({
      undo: state.undo,
      redo: state.redo,
      duplicateSelected: state.duplicateSelected,
      clearSelection: state.clearSelection,
      selectAll: state.selectAll,
    }))
  );
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
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
    const { nodes, deleteSelected, markDirty, capturePersistedGraph } = useGraphStore.getState();
    const { workspace, updateActiveScenarioTaskBindingSpec } = useWorkspaceStore.getState();
    const selectedNodeIds = nodes
      .filter((node) => node.selected && !node.id.startsWith('tap:'))
      .map((node) => node.id);
    const topPane = getTopPaneState(workspace);
    const selectedTaskBindingId = topPane.selected_entity_id?.startsWith('task_binding:')
      ? topPane.selected_entity_id.slice('task_binding:'.length)
      : null;
    const trainingScenario = selectedTaskBindingId ? getTrainingScenario(workspace) : null;
    if (trainingScenario && selectedTaskBindingId) {
      const taskBindingSpec = ensureTaskBindingSpec(
        trainingScenario.task_binding_spec, capturePersistedGraph().graph, trainingScenario.task_spec
      );
      updateActiveScenarioTaskBindingSpec({
        ...taskBindingSpec,
        bindings: taskBindingSpec.bindings.filter(
          (binding) => binding.id !== selectedTaskBindingId
        ),
      });
      selectTopPaneEntity(null);
      markDirty();
      return;
    }
    const bindingRemoval = prepareRootTaskBindingRemoval(selectedNodeIds);

    if (
      topPane.active_projection === 'model' &&
      bindingRemoval &&
      bindingRemoval.bindingCount > 0 &&
      !window.confirm(
        `Delete selected node? It has ${bindingRemoval.bindingCount} task binding${
          bindingRemoval.bindingCount === 1 ? '' : 's'
        } wired into it.`
      )
    ) {
      return;
    }

    deleteSelected(bindingRemoval?.edit);
    if (selectedNodeIds.length > 0) {
      toast.success(
        selectedNodeIds.length === 1
          ? 'Node deleted - Cmd+Z to undo.'
          : 'Nodes deleted - Cmd+Z to undo.',
        { id: 'node-delete-success' },
      );
    }
  }, [selectTopPaneEntity]);

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
