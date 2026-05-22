import type { GraphSpec } from '@/types/graph';
import type { TaskSpec } from '@/types/training';
import type {
  StudioTaskBinding,
  StudioTaskBindingSpec,
  StudioTaskDataSpec,
  StudioValueSpec,
  ValueSchema,
} from '@/types/workspace';
import { delayedReachTaskDataValueSpec, VALUE_SCHEMA_VERSION } from './taskTimeline';

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
  const delayedTask: TaskSpec = { type: 'DelayedReaches', params: {} };
  const data = [
    taskDataSpec({
      id: 'target_position',
      label: 'Target position',
      kind: 'signal',
      role: 'model_input',
      path: 'inputs.effector_target',
      bindable: true,
      value: { dtype: 'float32', shape: ['time', 4], units: null, frame: 'cartesian_effector' },
      metadata: {
        source: 'DelayedReachTaskInputs',
        task_input_field: 'effector_target',
        component_fields: ['pos', 'vel'],
        component_shapes: { pos: [2], vel: [2] },
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
  return data.map((item) => ({
    ...item,
    value_spec: delayedReachTaskDataValueSpec(item.id, delayedTask) ?? item.value_spec,
  }));
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

export function taskBindingTargetKey(
  targetNodeId: string,
  targetPort: string
): string {
  return `${targetNodeId}.${targetPort}`;
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

function delayedReachMuxBindings(
  graph: GraphSpec,
  data: StudioTaskDataSpec[]
): StudioTaskBinding[] {
  const dataIds = new Set(data.map((item) => item.id));
  const portMap: Array<[string, string]> = [
    ['target_position', 'in_0'],
    ['hold', 'in_1'],
    ['target_on', 'in_2'],
  ];
  const muxEntry = delayedReachMuxEntry(graph, portMap.map(([, port]) => port));
  if (!muxEntry) return [];
  const [muxNodeId, mux] = muxEntry;
  if (
    !portMap.every(
      ([dataId, port]) => dataIds.has(dataId) && mux.input_ports.includes(port)
    )
  ) {
    return [];
  }
  return portMap.map(([dataId, port]) => ({
    id: taskBindingId(dataId, muxNodeId, port),
    source_data_id: dataId,
    target_node_id: muxNodeId,
    target_port: port,
    role: 'model_input',
    metadata: {},
  }));
}

function delayedReachMuxEntry(
  graph: GraphSpec,
  requiredPorts: string[]
): [string, GraphSpec['nodes'][string]] | null {
  const candidates = Object.entries(graph.nodes).filter(
    ([, node]) =>
      node.type === 'Mux' &&
      node.output_ports.includes('output') &&
      requiredPorts.every((port) => node.input_ports.includes(port))
  );
  if (candidates.length === 0) return null;
  const networkInput = compatibleNetworkInput(graph);
  if (networkInput) {
    const connected = candidates.find(([nodeId]) =>
      graph.wires.some(
        (wire) =>
          wire.source_node === nodeId &&
          wire.source_port === 'output' &&
          wire.target_node === networkInput.nodeId &&
          wire.target_port === networkInput.port
      )
    );
    if (connected) return connected;
  }
  return candidates.find(([nodeId]) => nodeId === 'task_mux') ?? candidates[0];
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
  const delayedMuxBindings =
    task?.type === 'DelayedReaches' ? delayedReachMuxBindings(graph, data) : [];
  if (delayedMuxBindings.length > 0) {
    return {
      schema_version: TASK_BINDING_SCHEMA_VERSION,
      exposed_data: data,
      bindings: delayedMuxBindings,
      metadata: {},
    };
  }
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
    ...defaults.map((data) => normalizeDefaultTaskData(data, byId.get(data.id))),
    ...spec.exposed_data.filter(
      (data) => !defaultIds.has(data.id) && shouldPreserveExtraTaskData(data, task)
    ),
  ];
  const exposedDataIds = new Set(exposedData.map((data) => data.id));
  const defaultMuxBindings =
    task?.type === 'DelayedReaches' ? delayedReachMuxBindings(graph, exposedData) : [];
  const bindings = (spec.bindings ?? []).filter((binding) =>
    exposedDataIds.has(binding.source_data_id)
  );
  return {
    schema_version: spec.schema_version ?? TASK_BINDING_SCHEMA_VERSION,
    exposed_data: exposedData,
    bindings: bindings.length > 0 ? bindings : defaultMuxBindings,
    metadata: spec.metadata ?? {},
  };
}

function normalizeDefaultTaskData(
  canonical: StudioTaskDataSpec,
  existing?: StudioTaskDataSpec
): StudioTaskDataSpec {
  if (!existing) return canonical;
  return {
    ...existing,
    label: existing.label || canonical.label,
    kind: canonical.kind,
    role: canonical.role,
    path: canonical.path,
    bindable: canonical.bindable,
    expected_shape: canonical.expected_shape,
    dtype: canonical.dtype,
    units: canonical.units,
    frame: canonical.frame,
    value_spec: normalizeTaskDataValueSpec(canonical.value_spec, existing.value_spec),
    metadata: {
      ...existing.metadata,
      ...canonical.metadata,
    },
  };
}

function normalizeTaskDataValueSpec(
  canonical?: StudioValueSpec,
  existing?: StudioValueSpec
): StudioValueSpec | undefined {
  if (!canonical || !existing) return canonical ?? existing;
  return {
    ...existing,
    dtype: canonical.dtype,
    shape: canonical.shape,
    units: canonical.units,
    frame: canonical.frame,
    metadata: {
      ...existing.metadata,
      value_schema: canonical.metadata.value_schema,
      value_schema_id: canonical.metadata.value_schema_id,
    },
  };
}

export function retargetTaskBindingsForNodeRename(
  spec: StudioTaskBindingSpec,
  previousNodeId: string,
  nextNodeId: string
): StudioTaskBindingSpec {
  if (previousNodeId === nextNodeId) return spec;
  let changed = false;
  const bindings = spec.bindings.map((binding) => {
    if (binding.target_node_id !== previousNodeId) return binding;
    changed = true;
    return {
      ...binding,
      id: taskBindingId(binding.source_data_id, nextNodeId, binding.target_port),
      target_node_id: nextNodeId,
    };
  });
  return changed ? { ...spec, bindings } : spec;
}

export function retargetTaskBindingsForNodePortRename(
  spec: StudioTaskBindingSpec,
  nodeId: string,
  previousPort: string,
  nextPort: string
): StudioTaskBindingSpec {
  if (previousPort === nextPort) return spec;
  let changed = false;
  const bindings = spec.bindings.map((binding) => {
    if (binding.target_node_id !== nodeId || binding.target_port !== previousPort) {
      return binding;
    }
    changed = true;
    return {
      ...binding,
      id: taskBindingId(binding.source_data_id, nodeId, nextPort),
      target_port: nextPort,
    };
  });
  return changed ? { ...spec, bindings } : spec;
}

export function removeTaskBindingsForTargetNodes(
  spec: StudioTaskBindingSpec,
  targetNodeIds: Iterable<string>
): StudioTaskBindingSpec {
  const targetNodes = new Set(targetNodeIds);
  if (targetNodes.size === 0) return spec;
  const bindings = spec.bindings.filter(
    (binding) => !targetNodes.has(binding.target_node_id)
  );
  return bindings.length === spec.bindings.length ? spec : { ...spec, bindings };
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
        taskBindingTargetKey(binding.target_node_id, binding.target_port) ===
          taskBindingTargetKey(targetNodeId, targetPort)
    )
  );
}
