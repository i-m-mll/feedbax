import { useGraphStore } from '@/stores/graphStore';
import { getTrainingScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import {
  canAuthorTaskBindings,
  ensureTaskBindingSpec,
  ROOT_TASK_BINDING_MESSAGE,
  taskBindingId,
  targetInputOccupied,
} from '@/features/scenario/taskBindings';
import { isNextMuxInputPort } from '@/features/graph/dynamicPorts';
import type { GraphSpec } from '@/types/graph';
import type { StudioTaskBinding, StudioTaskBindingSpec } from '@/types/workspace';

type TaskBindingOutcome =
  | { ok: false; message: string }
  | {
      ok: true;
      binding: StudioTaskBinding;
      spec: StudioTaskBindingSpec;
      graph: GraphSpec;
      created: boolean;
    };

export function upsertRootTaskBinding(
  dataId: string,
  targetNodeId: string,
  targetPort: string
): TaskBindingOutcome {
  const graphState = useGraphStore.getState();
  if (!canAuthorTaskBindings(graphState.graph, graphState.graphStack.length)) {
    return { ok: false, message: ROOT_TASK_BINDING_MESSAGE };
  }
  const graph = graphState.graph;
  const workspaceState = useWorkspaceStore.getState();
  const scenario = getTrainingScenario(workspaceState.workspace);
  if (!scenario) return { ok: false, message: 'A training scenario is required for task bindings.' };
  const spec = ensureTaskBindingSpec(scenario.task_binding_spec, graph, scenario.task_spec);
  const taskData = spec.exposed_data.find((data) => data.id === dataId);
  if (!taskData?.bindable) return { ok: false, message: 'This task datum cannot bind to a port.' };
  const id = taskBindingId(dataId, targetNodeId, targetPort);
  const existing = spec.bindings.find((binding) => binding.id === id);
  if (existing) return { ok: true, binding: existing, spec, graph, created: false };
  const target = graph.nodes[targetNodeId];
  if (!target || (!target.input_ports.includes(targetPort)
    && !isNextMuxInputPort(graph, targetNodeId, targetPort, spec))) {
    return { ok: false, message: 'Task data must target an existing root input port.' };
  }
  if (targetInputOccupied(graph, spec, targetNodeId, targetPort)) {
    return { ok: false, message: `${targetNodeId}.${targetPort} is already occupied.` };
  }
  const binding: StudioTaskBinding = {
    id,
    source_data_id: dataId,
    target_node_id: targetNodeId,
    target_port: targetPort,
    role: taskData.role ?? 'model_input',
    metadata: {},
  };
  const nextSpec = { ...spec, bindings: [...spec.bindings, binding] };
  workspaceState.updateActiveScenarioTaskBindingSpec(nextSpec);
  graphState.markDirty();
  return { ok: true, binding, spec: nextSpec, graph, created: true };
}
