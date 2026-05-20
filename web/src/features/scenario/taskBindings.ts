import type { GraphSpec } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type {
  StudioTaskBindingSpec,
  StudioTaskDataSpec,
  StudioValueSpec,
  ValueSchema,
} from '@/types/workspace';
import { VALUE_SCHEMA_VERSION } from './taskTimeline';

export const TASK_BINDING_SCHEMA_VERSION = 'feedbax.studio.task_bindings.v2';
export const GRAPH_BINDABLE_TASK_DATA_ROLES = new Set(['model_input', 'graph_input']);

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
  role: string,
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
    metadata: { ...metadata, task_data_path: path, task_data_kind: kind, task_data_role: role },
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

function taskDataSpec({
  id,
  label,
  kind,
  role,
  path,
  bindable,
  value,
  metadata = {},
}: {
  id: string;
  label: string;
  kind: string;
  role: string;
  path: string;
  bindable: boolean;
  value: Pick<ValueSchema, 'dtype' | 'shape' | 'units' | 'frame'>;
  metadata?: Record<string, unknown>;
}): StudioTaskDataSpec {
  const schema = taskDataValueSchema(id, label, kind, role, path, value, metadata);
  return {
    id,
    label,
    kind,
    role,
    path,
    bindable,
    expected_shape: schema.shape,
    dtype: schema.dtype,
    units: schema.units,
    frame: schema.frame,
    value_spec: taskDataValueSpec(schema),
    metadata: { ...metadata, value_schema_id: schema.id, task_data_role: role },
  };
}

function genericTaskData(): StudioTaskDataSpec[] {
  const inputsData = taskDataSpec({
    id: 'inputs',
    label: 'Inputs',
    kind: 'signal',
    role: 'model_input',
    path: 'inputs',
    bindable: true,
    value: { dtype: 'float32', shape: ['time', 'channels'], units: null, frame: 'task_time' },
    metadata: { temporal_support: 'trajectory' },
  });
  const targetsData = taskDataSpec({
    id: 'targets',
    label: 'Targets',
    kind: 'target',
    role: 'target',
    path: 'targets',
    bindable: false,
    value: { dtype: 'float32', shape: ['time', 'target_dims'], units: null, frame: 'task_time' },
    metadata: {
      temporal_support: 'materialized_trajectory',
      storage: 'compact_task_params',
      compact_representation: 'delayed_reach_task_params_v1',
      materializes_to: { dtype: 'float32', shape: ['time', 'target_dims'] },
    },
  });
  const initsData = taskDataSpec({
    id: 'inits',
    label: 'Initial state',
    kind: 'initial_state',
    role: 'initial_state',
    path: 'inits',
    bindable: false,
    value: { dtype: 'float32', shape: ['state'], units: null, frame: null },
    metadata: { temporal_support: 'initial' },
  });
  const interveneData = taskDataSpec({
    id: 'intervene',
    label: 'Intervention',
    kind: 'intervention',
    role: 'intervention',
    path: 'intervene',
    bindable: false,
    value: { dtype: 'float32', shape: ['time', 'channels'], units: null, frame: 'task_time' },
    metadata: { temporal_support: 'trajectory' },
  });
  return [inputsData, targetsData, initsData, interveneData];
}

function delayedReachTaskData(): StudioTaskDataSpec[] {
  return [
    taskDataSpec({
      id: 'target_position',
      label: 'Target position',
      kind: 'signal',
      role: 'model_input',
      path: 'inputs.effector_target.pos',
      bindable: true,
      value: { dtype: 'float32', shape: ['time', 2], units: null, frame: 'cartesian_effector' },
      metadata: {
        source: 'DelayedReachTaskInputs',
        task_input_field: 'effector_target.pos',
        temporal_support: 'trajectory',
        task_data_surface: 'graph_input',
      },
    }),
    taskDataSpec({
      id: 'hold',
      label: 'Hold/go cue',
      kind: 'signal',
      role: 'model_input',
      path: 'inputs.hold',
      bindable: true,
      value: { dtype: 'float32', shape: ['time', 1], units: null, frame: 'task_time' },
      metadata: {
        source: 'DelayedReachTaskInputs',
        task_input_field: 'hold',
        cue_polarity: 'hold_is_1_go_is_0',
        go_cue_expression: '1 - inputs.hold',
        temporal_support: 'epoch_masked_signal',
        task_data_surface: 'graph_input',
      },
    }),
    taskDataSpec({
      id: 'target_on',
      label: 'Target shown',
      kind: 'signal',
      role: 'model_input',
      path: 'inputs.target_on',
      bindable: true,
      value: { dtype: 'float32', shape: ['time', 1], units: null, frame: 'task_time' },
      metadata: {
        source: 'DelayedReachTaskInputs',
        task_input_field: 'target_on',
        temporal_support: 'epoch_masked_signal',
        task_data_surface: 'graph_input',
      },
    }),
    taskDataSpec({
      id: 'movement_target',
      label: 'Movement target',
      kind: 'target',
      role: 'target',
      path: 'targets.effector',
      bindable: false,
      value: { dtype: 'float32', shape: ['time', 2], units: null, frame: 'cartesian_effector' },
      metadata: {
        source: 'TaskTrialSpec.targets',
        temporal_support: 'materialized_trajectory',
        task_data_surface: 'protocol',
      },
    }),
    ...genericTaskData().filter((data) => data.id === 'inits' || data.id === 'intervene'),
  ];
}

export function defaultTaskData(task?: TaskSpec | null): StudioTaskDataSpec[] {
  if (task?.type === 'DelayedReaches') {
    return delayedReachTaskData();
  }
  return genericTaskData();
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

function shouldPreserveExtraTaskData(
  data: StudioTaskDataSpec,
  task?: TaskSpec | null
): boolean {
  if (task?.type === 'DelayedReaches' && data.id === 'inputs' && data.path === 'inputs') {
    return false;
  }
  return true;
}

export function createDefaultTaskBindingSpec(
  graph: GraphSpec,
  task?: TaskSpec | null
): StudioTaskBindingSpec {
  const data = defaultTaskData(task);
  const bindingTarget = compatibleNetworkInput(graph);
  const defaultBindableDataId = data.some((item) => item.id === 'inputs') ? 'inputs' : null;
  const bindings =
    bindingTarget === null || defaultBindableDataId === null
      ? []
      : [
          {
            id: taskBindingId(defaultBindableDataId, bindingTarget.nodeId, bindingTarget.port),
            source_data_id: defaultBindableDataId,
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
  graph: GraphSpec,
  task?: TaskSpec | null
): StudioTaskBindingSpec {
  if (!spec) return createDefaultTaskBindingSpec(graph, task);
  const defaults = defaultTaskData(task);
  const defaultIds = new Set(defaults.map((data) => data.id));
  const byId = new Map(spec.exposed_data.map((data) => [data.id, data]));
  const exposedData = [
    ...defaults.map((data) => ({ ...data, ...(byId.get(data.id) ?? {}) })),
    ...spec.exposed_data.filter(
      (data) => !defaultIds.has(data.id) && shouldPreserveExtraTaskData(data, task)
    ),
  ];
  const exposedDataIds = new Set(exposedData.map((data) => data.id));
  return {
    schema_version: spec.schema_version ?? TASK_BINDING_SCHEMA_VERSION,
    exposed_data: exposedData,
    bindings: (spec.bindings ?? []).filter((binding) => exposedDataIds.has(binding.source_data_id)),
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
