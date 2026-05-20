import type { GraphSpec } from '@/types/graph';
import type {
  StudioTaskBindingSpec,
  StudioTaskDataSpec,
} from '@/types/workspace';

export const TASK_BINDING_SCHEMA_VERSION = 'feedbax.studio.task_bindings.v2';

export const TASK_COMPONENT_TYPES = new Set([
  'ReachingTask',
  'SimpleReaches',
  'DelayedReaches',
  'Stabilization',
]);

export function defaultTaskData(): StudioTaskDataSpec[] {
  return [
    {
      id: 'inputs',
      label: 'Inputs',
      kind: 'signal',
      path: 'inputs',
      bindable: true,
      metadata: {},
    },
    {
      id: 'targets',
      label: 'Targets',
      kind: 'target',
      path: 'targets',
      bindable: false,
      metadata: {},
    },
    {
      id: 'inits',
      label: 'Initial state',
      kind: 'initial_state',
      path: 'inits',
      bindable: false,
      metadata: {},
    },
    {
      id: 'intervene',
      label: 'Intervention',
      kind: 'intervention',
      path: 'intervene',
      bindable: false,
      metadata: {},
    },
  ];
}

export function taskBindingId(dataId: string, nodeId: string, port: string): string {
  return `task:${dataId}->${nodeId}:${port}`;
}

export function taskDataEntityId(
  scenarioId: string | null | undefined,
  dataId: string
): string {
  return `task_data:${scenarioId ?? 'active'}:${dataId}`;
}

export function taskBindingEntityId(bindingId: string): string {
  return `task_binding:${bindingId}`;
}

export function isTaskComponentType(type: string | null | undefined): boolean {
  return Boolean(type && TASK_COMPONENT_TYPES.has(type));
}

function compatibleNetworkInput(graph: GraphSpec): { nodeId: string; port: string } | null {
  const network = graph.nodes.network;
  if (network?.input_ports.includes('input')) {
    return { nodeId: 'network', port: 'input' };
  }
  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    if (node.input_ports.includes('input')) {
      return { nodeId, port: 'input' };
    }
  }
  return null;
}

export function createDefaultTaskBindingSpec(graph: GraphSpec): StudioTaskBindingSpec {
  const data = defaultTaskData();
  const bindingTarget = compatibleNetworkInput(graph);
  const bindings =
    bindingTarget === null
      ? []
      : [
          {
            id: taskBindingId('inputs', bindingTarget.nodeId, bindingTarget.port),
            source_data_id: 'inputs',
            target_node_id: bindingTarget.nodeId,
            target_port: bindingTarget.port,
            role: 'model_input',
            metadata: {},
          },
        ];
  return {
    schema_version: TASK_BINDING_SCHEMA_VERSION,
    exposed_data: data,
    bindings,
    metadata: {},
  };
}

export function ensureTaskBindingSpec(
  spec: StudioTaskBindingSpec | null | undefined,
  graph: GraphSpec
): StudioTaskBindingSpec {
  if (!spec) return createDefaultTaskBindingSpec(graph);
  const defaults = defaultTaskData();
  const byId = new Map(spec.exposed_data.map((data) => [data.id, data]));
  return {
    schema_version: spec.schema_version ?? TASK_BINDING_SCHEMA_VERSION,
    exposed_data: defaults.map((data) => ({ ...data, ...(byId.get(data.id) ?? {}) })),
    bindings: spec.bindings ?? [],
    metadata: spec.metadata ?? {},
  };
}

export function targetInputOccupied(
  graph: GraphSpec,
  taskBindingSpec: StudioTaskBindingSpec | null | undefined,
  targetNodeId: string,
  targetPort: string,
  ignoredBindingId?: string
): boolean {
  return (
    graph.wires.some(
      (wire) => wire.target_node === targetNodeId && wire.target_port === targetPort
    ) ||
    (taskBindingSpec?.bindings ?? []).some(
      (binding) =>
        binding.id !== ignoredBindingId &&
        binding.target_node_id === targetNodeId &&
        binding.target_port === targetPort
    )
  );
}
