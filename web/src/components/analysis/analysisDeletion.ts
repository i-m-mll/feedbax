import { useAnalysisStore } from '@/stores/analysisStore';

const DELETE_CONFIRMATION =
  'Delete this analysis node and its connected wires? This cannot currently be undone.';

export function deleteAnalysisNodeWithConfirmation(
  nodeId: string,
  confirmDeletion: (message: string) => boolean = window.confirm,
): boolean {
  if (!confirmDeletion(DELETE_CONFIRMATION)) return false;
  useAnalysisStore.getState().removeNode(nodeId);
  return true;
}
