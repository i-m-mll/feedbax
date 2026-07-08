import { useGraphStore } from '@/stores/graphStore';
import type { ParamValue } from '@/types/graph';

declare global {
  interface Window {
    feedbaxE2E?: {
      graphId: () => string | null;
      currentContext: () => string;
      nodeParam: (nodeId: string, paramName: string) => ParamValue | undefined;
      updateNodeParam: (nodeId: string, paramName: string, value: ParamValue) => void;
      enterSubgraph: (nodeId: string) => void;
      markDirty: () => void;
    };
  }
}

export function installE2EHarness() {
  window.feedbaxE2E = {
    graphId: () => useGraphStore.getState().graphId,
    currentContext: () => useGraphStore.getState().currentContext,
    nodeParam: (nodeId, paramName) =>
      useGraphStore.getState().graph.nodes[nodeId]?.params[paramName],
    updateNodeParam: (nodeId, paramName, value) =>
      useGraphStore.getState().updateNodeParams(nodeId, paramName, value),
    enterSubgraph: (nodeId) => useGraphStore.getState().enterSubgraph(nodeId),
    markDirty: () => useGraphStore.getState().markDirty(),
  };
}
