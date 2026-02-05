import { useEffect, useCallback } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { useSaveGraph } from '@/hooks/useGraphs';

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || target.isContentEditable;
}

export function useAppShortcuts() {
  const { undo, redo, deleteSelected, graph, uiState, graphId, markSaved } = useGraphStore();
  const saveMutation = useSaveGraph();

  const saveGraph = useCallback(async () => {
    const response = await saveMutation.mutateAsync({ graphId, graph, uiState });
    if ('id' in response) {
      markSaved(response.id);
    } else if (graphId) {
      markSaved(graphId);
    }
  }, [graphId, graph, uiState, markSaved, saveMutation]);

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
        deleteSelected();
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [deleteSelected, redo, undo, saveGraph]);
}
