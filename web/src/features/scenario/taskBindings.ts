import type { GraphSpec } from '@/types/graph';
import type {
  StudioTaskBindingSpec,
  StudioTaskDataSpec,
  StudioValueSpec,
  ValueSchema,
} from '@/types/workspace';
import { VALUE_SCHEMA_VERSION } from './taskTimeline';

export const TASK_BINDING_SCHEMA_VERSION = 'feedbax.studio.task_bindings.v2';

export const TASK_COMPONENT_TYPES = new Set([
  'ReachingTask',
  'SimpleReaches',
  'DelayedReaches',
  'Stabilization',
]);

function taskDataValueSchema(
  id: string,
  label: string,
  kind: string,
  path: string,
  value: Pick<ValueSchema, 'dtype' | 'shape' | 'units' | 'frame'>,
  metadata: Record<string, unknown> = {}
): ValueSchema {
  return {
    id: `value:task_data:${id}`,
    label,
    kind: 'task_data',
    dtype: value.dtype ?? null,
    shape: value.shape ?? null,
    units: value.units ?? null,
    frame: value.frame ?? null,
    origin: 'declared',
    metadata: { ...metadata, task_data_path: path, task_data_kind: kind },
  };
}

function taskDataValueSpec(
  schema: ValueSchema,
  metadata: Record<string, unknown> = {}
): StudioValueSpec {
  return {
    schema_version: VALUE_SCHEMA_VERSION,
    mode: 'reference',
    dtype: schema.dtype ?? null,
    shape: schema.shape ?? null,
    units: schema.units ?? null,
    frame: schema.frame ?? null,
    metadata: {
      ...metadata,
      value_schema: schema,
      value_schema_id: schema.id,
    },
  };
}

export function defaultTaskData(): StudioTaskDataSpec[] {
  const inputsSchema = taskDataValueSchema(
    'inputs',
    'Inputs',
    'signal',
    'inputs',
    { dtype: 'float32', shape: ['time', 'channels'], units: null, frame: 'task_time' },
    { temporal_support: 'trajectory' }
  );
  const targetsSchema = taskDataValueSchema(
    'targets',
    'Targets',
    'target',
    'targets',
    { dtype: 'float32', shape: ['time', 'target_dims'], units: null, frame: 'task_time' },
    {
      temporal_support: 'materialized_trajectory',
      storage: 'compact_task_params',
      compact_representation: 'delayed_reach_task_params_v1',
      materializes_to: { dtype: 'float32', shape: ['time', 'target_dims'] },
    }
  );
  const initsSchema = taskDataValueSchema(
    'inits',
    'Initial state',
    'initial_state',
    'inits',
    { dtype: 'float32', shape: ['state'], units: null, frame: null },
    { temporal_support: 'initial' }
  );
  const interveneSchema = taskDataValueSchema(
    'intervene',
    'Intervention',
    'intervention',
    'intervene',
    { dtype: 'float32', shape: ['time', 'channels'], units: null, frame: 'task_time' },
    { temporal_support: 'trajectory' }
  );
  return [
    {
      id: 'inputs',
      label: 'Inputs',
      kind: 'signal',
      path: 'inputs',
      bindable: true,
      expected_shape: inputsSchema.shape,
      dtype: inputsSchema.dtype,
      units: inputsSchema.units,
      frame: inputsSchema.frame,
      value_spec: taskDataValueSpec(inputsSchema),
      metadata: { value_schema_id: inputsSchema.id },
    },
    {
      id: 'targets',
      label: 'Targets',
      kind: 'target',
      path: 'targets',
      bindable: false,
      expected_shape: targetsSchema.shape,
      dtype: targetsSchema.dtype,
      units: targetsSchema.units,
      frame: targetsSchema.frame,
      value_spec: taskDataValueSpec(targetsSchema),
      metadata: { value_schema_id: targetsSchema.id },
    },
    {
      id: 'inits',
      label: 'Initial state',
      kind: 'initial_state',
      path: 'inits',
      bindable: false,
      expected_shape: initsSchema.shape,
      dtype: initsSchema.dtype,
      units: initsSchema.units,
      frame: initsSchema.frame,
      value_spec: taskDataValueSpec(initsSchema),
      metadata: { value_schema_id: initsSchema.id },
    },
    {
      id: 'intervene',
      label: 'Intervention',
      kind: 'intervention',
      path: 'intervene',
      bindable: false,
      expected_shape: interveneSchema.shape,
      dtype: interveneSchema.dtype,
      units: interveneSchema.units,
      frame: interveneSchema.frame,
      value_spec: taskDataValueSpec(interveneSchema),
      metadata: { value_schema_id: interveneSchema.id },
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
