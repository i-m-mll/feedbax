import type { ComponentDefinition } from '@/types/components';
import {
  isCausalGraphSpec,
  type GraphSpec,
  type GraphSubgraphSpec,
  type RetainedObservableSpec,
} from '@/types/graph';
import type {
  PortSchema,
  SelectorTargetSchema,
  StudioSchemaOrigin,
  StudioSchemaRegistry,
  StudioValidationIssue as SchemaValidationIssue,
  TaskDataSchema,
  ValueSchema,
} from '@/types/workspace';
import type { StudioTaskBindingSpec } from '@/types/workspace';
import { validateInterventionSchema } from './interventions';
import {
  MUX_COMPONENT_TYPE,
  muxInputIndex,
  normalizeDynamicPorts,
} from '@/features/graph/dynamicPorts';
import { taskBindingId } from '@/features/scenario/taskBindings';

const GRAPH_BINDABLE_TASK_DATA_ROLES = new Set(['model_input', 'graph_input']);
const PROTOCOL_TASK_DATA_KINDS = new Set([
  'target',
  'initial_state',
  'intervention',
  'eval_control',
  'trial_control',
  'validation',
  'protocol_value',
]);
const PROTOCOL_TASK_DATA_PATH_PREFIXES = [
  'targets',
  'inits',
  'intervene',
  'validation_trials',
  'eval',
  'trial_controls',
  'task.validation_trials',
];

type ComponentPortType = {
  dtype?: string | null;
  shape?: number[] | null;
  rank?: number | null;
};

interface PortSchemaIndex {
  byNodePort: Map<string, PortSchema>;
  byDirectedNodePort: Map<string, PortSchema>;
}

function portLookupKey(nodeId: string, port: string): string {
  return `${nodeId}.${port}`;
}

function directedPortLookupKey(
  nodeId: string,
  port: string,
  direction: 'input' | 'output'
): string {
  return `${portLookupKey(nodeId, port)}:${direction}`;
}

function buildPortSchemaIndex(ports: PortSchema[]): PortSchemaIndex {
  const byNodePort = new Map<string, PortSchema>();
  const byDirectedNodePort = new Map<string, PortSchema>();
  for (const port of ports) {
    if (!port.node_id) continue;
    const nodePortKey = portLookupKey(port.node_id, port.port);
    if (!byNodePort.has(nodePortKey)) {
      byNodePort.set(nodePortKey, port);
    }
    byDirectedNodePort.set(
      directedPortLookupKey(port.node_id, port.port, port.direction),
      port
    );
  }
  return { byNodePort, byDirectedNodePort };
}

export function projectStudioSchema(
  graph: GraphSpec,
  components: ComponentDefinition[],
  taskBindingSpec?: StudioTaskBindingSpec | null
): StudioSchemaRegistry {
  const componentMap = new Map(components.map((component) => [component.name, component]));
  const schemaGraph = normalizeDynamicPorts(graph, taskBindingSpec);
  const taskData = enumerateTaskData(taskBindingSpec);
  const ports = enumerateGraphPorts(schemaGraph, componentMap, taskBindingSpec, taskData);
  const portIndex = buildPortSchemaIndex(ports);
  markBoundPorts(ports, taskBindingSpec, portIndex);
  const selectorTargets = [
    ...portSelectorTargets(ports),
    ...graphStructuralSelectorTargets(schemaGraph, portIndex),
    ...taskDataSelectorTargets(taskData),
    ...knownStateHintTargets(),
  ];

  return {
    kind: 'studio_schema_registry',
    schema_version: 'feedbax.schema.v1',
    generated_at: new Date().toISOString(),
    ports,
    task_data: taskData,
    selector_targets: selectorTargets,
    issues: [
      ...validateMissingSubgraphs(schemaGraph, componentMap),
      ...validateGraphConnections(schemaGraph, portIndex),
      ...validateTaskBindings(schemaGraph, portIndex, taskData, taskBindingSpec),
      ...validateInterventionSchema(graph.taps, { selector_targets: selectorTargets }),
    ],
    metadata: { projected_by: 'feedbax.web.projectStudioSchema' },
  };
}

function validateMissingSubgraphs(
  graph: GraphSpec,
  componentMap: Map<string, ComponentDefinition>
): SchemaValidationIssue[] {
  const issues: SchemaValidationIssue[] = [];
  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const component = componentMap.get(node.type);
    const requiresSubgraph =
      node.type === 'Network' ||
      node.type === 'Subgraph' ||
      Boolean(component?.is_composite);
    if (!requiresSubgraph || graph.subgraphs?.[nodeId]) continue;
    issues.push({
      type: 'missing_subgraph',
      message:
        node.type === 'Network'
          ? `Network node '${nodeId}' has no subgraph. Open it in Studio to generate the internal architecture, then save again.`
          : `${node.type} node '${nodeId}' requires a subgraph, but none was provided`,
      severity: 'error',
      location: { path: `nodes.${nodeId}`, node: nodeId },
    });
  }
  return issues;
}

export function validateConnectionAgainstSchema(
  registry: Pick<StudioSchemaRegistry, 'ports'>,
  sourceNode: string,
  sourcePort: string,
  targetNode: string,
  targetPort: string
): SchemaValidationIssue[] {
  return validateConnectionAgainstSchemaWithIndex(
    buildPortSchemaIndex(registry.ports),
    sourceNode,
    sourcePort,
    targetNode,
    targetPort
  );
}

function validateConnectionAgainstSchemaWithIndex(
  portIndex: PortSchemaIndex,
  sourceNode: string,
  sourcePort: string,
  targetNode: string,
  targetPort: string
): SchemaValidationIssue[] {
  const source = findPort(portIndex, sourceNode, sourcePort, 'output');
  const target = findPort(portIndex, targetNode, targetPort, 'input');
  const sourceAnyDirection = findPortAnyDirection(portIndex, sourceNode, sourcePort);
  const targetAnyDirection = findPortAnyDirection(portIndex, targetNode, targetPort);
  const issues: SchemaValidationIssue[] = [];

  if (!source) {
    issues.push({
      type: sourceAnyDirection ? 'wrong_source_port_direction' : 'unknown_source_port',
      message: sourceAnyDirection
        ? `${sourceNode}.${sourcePort} is not an output port`
        : `${sourceNode}.${sourcePort} does not exist`,
      severity: 'error',
      location: { node: sourceNode, port: sourcePort },
    });
  }
  if (!target) {
    issues.push({
      type: targetAnyDirection ? 'wrong_target_port_direction' : 'unknown_target_port',
      message: targetAnyDirection
        ? `${targetNode}.${targetPort} is not an input port`
        : `${targetNode}.${targetPort} does not exist`,
      severity: 'error',
      location: { node: targetNode, port: targetPort },
    });
  }
  if (source && target) {
    issues.push(
      ...valueSchemaCompatibilityIssues(
        source.value_schema,
        target.value_schema,
        source.label,
        target.label,
        'graph_wire'
      )
    );
  }
  return issues;
}

export function validateTaskDataBindingAgainstSchema(
  registry: Pick<StudioSchemaRegistry, 'ports' | 'task_data'>,
  sourceDataId: string,
  targetNode: string,
  targetPort: string
): SchemaValidationIssue[] {
  const source = registry.task_data.find(
    (item) => item.id === `task_data:${sourceDataId}` || item.id === sourceDataId
  );
  const portIndex = buildPortSchemaIndex(registry.ports);
  const target = findPort(portIndex, targetNode, targetPort, 'input');
  const targetAnyDirection = findPortAnyDirection(portIndex, targetNode, targetPort);
  const issues: SchemaValidationIssue[] = [];

  if (!source) {
    issues.push({
      type: 'unknown_task_data',
      message: `Task Data ${sourceDataId} is not declared`,
      severity: 'error',
      location: { source_data_id: sourceDataId },
    });
  } else if (!source.bindable) {
    issues.push({
      type: 'task_data_not_bindable',
      message: `Task Data ${sourceDataId} has protocol role ${source.role} and is not bindable`,
      severity: 'error',
      location: { source_data_id: sourceDataId },
    });
  }
  if (!target) {
    issues.push({
      type: targetAnyDirection ? 'wrong_target_port_direction' : 'unknown_target_port',
      message: targetAnyDirection
        ? `${targetNode}.${targetPort} is not an input port`
        : `${targetNode}.${targetPort} does not exist`,
      severity: 'error',
      location: { node: targetNode, port: targetPort },
    });
  }
  if (source && target) {
    issues.push(
      ...valueSchemaCompatibilityIssues(
        source.value_schema,
        target.value_schema,
        source.label,
        target.label,
        'task_binding'
      )
    );
  }
  return issues;
}

export function hasBlockingSchemaIssue(issues: SchemaValidationIssue[]): boolean {
  return issues.some((issue) => issue.severity === 'error');
}

function enumerateGraphPorts(
  graph: GraphSpec,
  componentMap: Map<string, ComponentDefinition>,
  taskBindingSpec?: StudioTaskBindingSpec | null,
  taskData: TaskDataSchema[] = []
): PortSchema[] {
  const ports: PortSchema[] = [];
  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const component = componentMap.get(node.type);
    const subgraph = graph.subgraphs?.[nodeId] ?? null;
    const inputPorts = node.input_ports.length > 0 ? node.input_ports : component?.input_ports ?? [];
    const outputPorts =
      node.output_ports.length > 0 ? node.output_ports : component?.output_ports ?? [];
    for (const port of inputPorts) {
      ports.push(
        componentPortSchema(
          nodeId,
          node.type,
          node.params,
          port,
          'input',
          componentInputPortType(node.type, port, component),
          subgraph,
          componentMap
        )
      );
    }
    for (const port of outputPorts) {
      ports.push(
        componentPortSchema(
          nodeId,
          node.type,
          node.params,
          port,
          'output',
          component?.port_types?.outputs?.[port],
          subgraph,
          componentMap
        )
      );
    }
  }
  const componentPortIndex = buildPortSchemaIndex(ports);
  for (const port of graph.input_ports) {
    ports.push(graphPortSchema(port, 'input', graph.input_bindings[port], componentPortIndex));
  }
  for (const port of graph.output_ports) {
    ports.push(graphPortSchema(port, 'output', graph.output_bindings[port], componentPortIndex));
  }
  applyMuxOutputShapes(ports, graph, taskBindingSpec, taskData);
  return ports;
}

function componentInputPortType(
  componentType: string,
  port: string,
  component?: ComponentDefinition
): ComponentPortType | undefined {
  const explicit = component?.port_types?.inputs?.[port];
  if (explicit) return explicit;
  if (componentType === MUX_COMPONENT_TYPE && muxInputIndex(port) !== null) {
    return component?.port_types?.inputs?.in_0 ?? component?.port_types?.inputs?.in_1;
  }
  return undefined;
}

function componentPortSchema(
  nodeId: string,
  componentType: string,
  nodeParams: Record<string, unknown>,
  port: string,
  direction: 'input' | 'output',
  portType?: ComponentPortType,
  subgraph?: GraphSubgraphSpec | null,
  componentMap?: Map<string, ComponentDefinition>
): PortSchema {
  const portId = `port:${nodeId}.${port}:${direction}`;
  let origin: StudioSchemaOrigin = portType ? 'declared' : 'unknown';
  let shape: unknown[] | null = portType?.shape ?? null;
  let rank: number | null = portType?.rank ?? null;
  const metadata: Record<string, unknown> = { component_type: componentType };
  const inferred = componentPortDimension(componentType, nodeParams, port, direction);
  if (inferred) {
    shape = inferred.shape;
    rank = inferred.shape.length;
    origin = 'inferred_static';
    Object.assign(metadata, inferred.metadata);
  }
  const boundary = subgraphBoundaryValueSchema(subgraph, port, direction, componentMap);
  if (!shape && boundary?.shape) {
    shape = boundary.shape;
    rank = boundary.rank ?? boundary.shape.length;
    origin = 'inferred_static';
    metadata.dimension_source = 'subgraph_boundary';
    metadata.boundary_port = port;
  }
  const dtype = portType?.dtype ?? boundary?.dtype ?? null;
  return {
    id: portId,
    label: `${nodeId}.${port}`,
    node_id: nodeId,
    component_type: componentType,
    port,
    direction,
    value_schema: {
      id: `value:${portId}`,
      label: `${nodeId}.${port}`,
      kind: 'graph_port',
      dtype,
      shape,
      rank,
      origin,
      metadata,
    },
    origin,
    metadata: {},
  };
}

function componentPortDimension(
  componentType: string,
  nodeParams: Record<string, unknown>,
  port: string,
  direction: 'input' | 'output'
): { shape: number[]; metadata: Record<string, unknown> } | null {
  let dimensionParam: string | null = null;
  let dimension: number | null = null;
  if (componentType === 'Network') {
    if (direction === 'output' && port === 'output') {
      dimensionParam = 'out_size';
    } else if (direction === 'output' && port === 'hidden') {
      dimensionParam = 'hidden_size';
    }
  } else if (componentType === 'Linear' || componentType === 'MLP') {
    if (direction === 'input' && port === 'input') dimensionParam = 'input_size';
    if (direction === 'output' && port === 'output') dimensionParam = 'output_size';
  } else if (componentType === 'GRU' || componentType === 'LSTM') {
    if (direction === 'input' && port === 'input') dimensionParam = 'input_size';
    if (port === 'output' || port === 'hidden' || port === 'cell') dimensionParam = 'hidden_size';
  } else if (['TwoLinkArm', 'PointMass', 'Mechanics'].includes(componentType) && port === 'force') {
    dimension = 2;
  } else if (componentType === 'Arm6MuscleRigidTendon') {
    if (port === 'excitation') dimension = 6;
    if (port === 'angles' || port === 'angular_velocities' || port === 'torques') dimension = 2;
    if (port === 'forces' || port === 'activations') dimension = 6;
  } else if (componentType === 'PointMass8MuscleRelu') {
    if (port === 'excitation' || port === 'forces' || port === 'activations') {
      const nPairs = positiveIntParam(nodeParams, 'n_pairs');
      dimension = nPairs === null ? null : nPairs * 2;
    }
    if (port === 'force_2d') dimension = 2;
  } else if (componentType === 'FeedbackChannels' && direction === 'output' && port === 'output') {
    dimension = feedbackChannelsWidth(nodeParams);
  }
  if (dimensionParam) dimension = positiveIntParam(nodeParams, dimensionParam);
  if (dimension === null) return null;
  return {
    shape: [dimension],
    metadata: {
      dimension_source: dimensionParam ? 'component_param' : 'component_contract',
      dimension_param: dimensionParam,
    },
  };
}

function feedbackChannelsWidth(params: Record<string, unknown>): number | null {
  const channels = params.channels;
  if (!Array.isArray(channels)) return null;
  const width = channels.reduce((sum, channel) => {
    if (channel === 'position' || channel === 'velocity') return sum + 2;
    if (channel === 'force') return sum + 2;
    return sum;
  }, 0);
  return width > 0 ? width : null;
}

function positiveIntParam(params: Record<string, unknown>, key: string): number | null {
  const value = params[key];
  return Number.isInteger(value) && typeof value === 'number' && value > 0 ? value : null;
}

function subgraphBoundaryValueSchema(
  subgraph: GraphSubgraphSpec | null | undefined,
  port: string,
  direction: 'input' | 'output',
  componentMap?: Map<string, ComponentDefinition>
): ValueSchema | null {
  if (!isCausalGraphSpec(subgraph) || !componentMap) return null;
  const binding = direction === 'input'
    ? subgraph.input_bindings[port]
    : subgraph.output_bindings[port];
  if (!binding) return null;
  const [nodeId, boundPort] = binding;
  const subgraphPorts = enumerateGraphPorts(subgraph, componentMap);
  const bound = findPort(buildPortSchemaIndex(subgraphPorts), nodeId, boundPort, direction);
  return bound?.value_schema ?? null;
}

function graphPortSchema(
  port: string,
  direction: 'input' | 'output',
  binding?: [string, string],
  portIndex?: PortSchemaIndex
): PortSchema {
  const portId = `port:graph.${port}:${direction}`;
  const metadata = binding ? { binding: { node_id: binding[0], port: binding[1] } } : {};
  const bound =
    binding && portIndex
      ? findPort(portIndex, binding[0], binding[1], direction)
      : null;
  return {
    id: portId,
    label: `graph.${port}`,
    port,
    direction,
    value_schema: {
      id: `value:${portId}`,
      label: `graph.${port}`,
      kind: 'graph_port',
      dtype: bound?.value_schema.dtype ?? null,
      shape: bound?.value_schema.shape ?? null,
      rank: bound?.value_schema.rank ?? null,
      origin: 'inferred_static',
      metadata: {
        ...metadata,
        ...(bound ? { dimension_source: 'graph_port_binding' } : {}),
      },
    },
    origin: 'inferred_static',
    metadata,
  };
}

function enumerateTaskData(taskBindingSpec?: StudioTaskBindingSpec | null): TaskDataSchema[] {
  return (taskBindingSpec?.exposed_data ?? []).map((data) => {
    const value = data.value_spec;
    const role = taskDataRole(data);
    const surface = isBindableTaskData(data) ? 'graph_input' : 'protocol';
    const shape = data.expected_shape ?? value?.shape ?? null;
    const sampleShape = taskDataSampleShape(shape);
    const hasTimeAxis = Boolean(shape && shape[0] === 'time' && sampleShape);
    const metadata = {
      ...data.metadata,
      task_data_role: role,
      task_data_surface: surface,
      ...(sampleShape
        ? {
            sample_shape: sampleShape,
            sample_rank: sampleShape.length,
            ...(hasTimeAxis ? { time_axis: 0, view: 'trajectory' } : {}),
          }
        : {}),
    };
    const valueSchema: ValueSchema = {
      id: `value:task_data:${data.id}`,
      label: data.label,
      kind: 'task_data',
      dtype: data.dtype ?? value?.dtype ?? null,
      shape,
      rank: shape ? shape.length : null,
      units: data.units ?? value?.units ?? null,
      frame: data.frame ?? value?.frame ?? null,
      origin: 'declared',
      metadata: {
        ...metadata,
        task_data_path: data.path,
        value_spec: value ?? null,
      },
    };
    return {
      id: `task_data:${data.id}`,
      label: data.label,
      kind: data.kind,
      role,
      path: data.path,
      bindable: isBindableTaskData(data),
      value_schema: valueSchema,
      origin: 'declared',
      metadata,
    };
  });
}

function taskDataSampleShape(shape: unknown[] | null | undefined): unknown[] | null {
  if (!shape) return null;
  if (shape[0] === 'time') return shape.slice(1);
  return [...shape];
}

function applyMuxOutputShapes(
  ports: PortSchema[],
  graph: GraphSpec,
  taskBindingSpec?: StudioTaskBindingSpec | null,
  taskData: TaskDataSchema[] = []
): void {
  const taskDataById = new Map(
    taskData.map((item) => [item.id.replace(/^task_data:/, ''), item.value_schema])
  );
  const outputPorts = new Map(
    ports
      .filter((port) => port.direction === 'output' && port.node_id)
      .map((port) => [`${port.node_id}.${port.port}`, port])
  );
  const inputSources = new Map<string, ValueSchema>();
  for (const wire of graph.wires) {
    const source = outputPorts.get(`${wire.source_node}.${wire.source_port}`);
    if (source) inputSources.set(`${wire.target_node}.${wire.target_port}`, source.value_schema);
  }
  for (const binding of taskBindingSpec?.bindings ?? []) {
    const source = taskDataById.get(binding.source_data_id);
    if (source) inputSources.set(`${binding.target_node_id}.${binding.target_port}`, source);
  }

  for (const port of ports) {
    if (port.component_type !== MUX_COMPONENT_TYPE || port.direction !== 'output' || port.port !== 'output') {
      continue;
    }
    const nodeId = port.node_id;
    const node = nodeId ? graph.nodes[nodeId] : null;
    if (!node || !nodeId) continue;
    const inputShapes = node.input_ports
      .map((inputPort) => compatibilityShape(inputSources.get(`${nodeId}.${inputPort}`)))
      .filter((shape): shape is unknown[] => Array.isArray(shape));
    if (inputShapes.length === 0 || inputShapes.some((shape) => shape.length !== 1)) continue;
    const dimensions = inputShapes.map((shape) => shape[0]);
    if (!dimensions.every((dim): dim is number => Number.isInteger(dim) && typeof dim === 'number' && dim > 0)) {
      continue;
    }
    port.value_schema = {
      ...port.value_schema,
      shape: [dimensions.reduce((sum, dim) => sum + dim, 0)],
      rank: 1,
      origin: 'inferred_static',
      metadata: {
        ...port.value_schema.metadata,
        dimension_source: 'mux_concat_inputs',
        input_shapes: inputShapes,
      },
    };
    port.origin = 'inferred_static';
  }
}

function taskDataRole(data: { kind: string; role?: string | null; bindable: boolean; metadata: Record<string, unknown> }): string {
  const metadataRole = data.metadata.task_data_role;
  if (typeof data.role === 'string' && data.role.length > 0) return data.role;
  if (typeof metadataRole === 'string' && metadataRole.length > 0) return metadataRole;
  if (['signal', 'input', 'model_input', 'graph_input'].includes(data.kind)) {
    return data.bindable ? 'model_input' : 'protocol_value';
  }
  if (data.kind === 'trial_param') return 'trial_control';
  return data.kind;
}

function isBindableTaskData(data: { kind: string; role?: string | null; bindable: boolean; metadata: Record<string, unknown> }): boolean {
  return data.bindable && GRAPH_BINDABLE_TASK_DATA_ROLES.has(taskDataRole(data));
}

function usesProtocolTaskDataPath(path: string): boolean {
  return PROTOCOL_TASK_DATA_PATH_PREFIXES.some(
    (prefix) => path === prefix || path.startsWith(`${prefix}.`)
  );
}

function markBoundPorts(
  ports: PortSchema[],
  taskBindingSpec: StudioTaskBindingSpec | null | undefined,
  portIndex: PortSchemaIndex
): void {
  for (const binding of taskBindingSpec?.bindings ?? []) {
    const port = findPort(portIndex, binding.target_node_id, binding.target_port, 'input');
    if (port) {
      port.bound_task_data_id = `task_data:${binding.source_data_id}`;
    }
  }
}

function portSelectorTargets(ports: PortSchema[]): SelectorTargetSchema[] {
  return ports
    .filter((port) => port.node_id)
    .map((port) => ({
      id: `selector:port:${port.node_id}.${port.port}`,
      label: port.label,
      kind: 'port',
      selector: `port:${port.node_id}.${port.port}`,
      value_schema: port.value_schema,
      origin: port.origin,
      source: { port_id: port.id, direction: port.direction },
      metadata: {},
    }));
}

function taskDataSelectorTargets(taskData: TaskDataSchema[]): SelectorTargetSchema[] {
  return taskData.map((item) => ({
    id: `selector:task_data:${item.id.replace(/^task_data:/, '')}`,
    label: item.label,
    kind: 'task_data',
    selector: `task_data:${item.path}`,
    value_schema: item.value_schema,
    origin: item.origin,
    source: { task_data_id: item.id },
    metadata: {},
  }));
}

function graphStructuralSelectorTargets(
  graph: GraphSpec,
  portIndex: PortSchemaIndex
): SelectorTargetSchema[] {
  const targets: SelectorTargetSchema[] = [];
  for (const [outputName, [nodeId, portName]] of Object.entries(graph.output_bindings)) {
    const port = findPort(portIndex, nodeId, portName, 'output');
    if (!port) continue;
    const selector = `graph_output:${outputName}`;
    targets.push({
      id: `selector:${selector}`,
      label: `Graph output ${outputName}`,
      kind: 'graph_output',
      selector,
      value_schema: port.value_schema,
      origin: port.origin,
      source: { output_name: outputName, node_id: nodeId, port: portName },
      metadata: {
        retention_target: {
          kind: 'graph_output',
          selector,
          node_id: nodeId,
          port: portName,
        },
        default_retention: { mode: 'trajectory' },
      },
    });
  }
  graph.wires.forEach((wire, index) => {
    const port = findPort(portIndex, wire.source_node, wire.source_port, 'output');
    if (!port) return;
    const selector = `edge:${wire.source_node}.${wire.source_port}->${wire.target_node}.${wire.target_port}`;
    const kind = wire.temporality === 'recurrent' ? 'recurrent_carry' : 'edge';
    targets.push({
      id: `selector:${selector}`,
      label: `${wire.source_node}.${wire.source_port} -> ${wire.target_node}.${wire.target_port}`,
      kind,
      selector,
      value_schema: port.value_schema,
      origin: port.origin,
      source: {
        wire_index: index,
        source_node: wire.source_node,
        source_port: wire.source_port,
        target_node: wire.target_node,
        target_port: wire.target_port,
        temporality: wire.temporality ?? 'instant',
      },
      metadata: {
        retention_target: {
          kind,
          selector,
          edge_id: selector,
          node_id: wire.source_node,
          port: wire.source_port,
          timing: 'step',
        },
        default_retention: {
          mode: wire.temporality === 'recurrent' ? 'window' : 'trajectory',
          window_size: wire.temporality === 'recurrent' ? 1 : null,
        },
      },
    });
  });
  for (const observable of graph.retained_observables ?? []) {
    targets.push(retainedObservableSelectorTarget(observable));
  }
  return targets;
}

function retainedObservableSelectorTarget(
  observable: RetainedObservableSpec
): SelectorTargetSchema {
  const selector = observable.target?.selector ?? observable.selector ?? observable.id;
  const compact = selector.includes(':') ? selector : `probe:${selector}`;
  const targetKind = observable.target?.kind;
  const kind =
    targetKind === 'port' ||
    targetKind === 'edge' ||
    targetKind === 'graph_output' ||
    targetKind === 'recurrent_carry' ||
    targetKind === 'state_path' ||
    targetKind === 'task_data'
      ? targetKind
      : 'retained_observable';
  const valueSchema = observable.value_schema as ValueSchema | null | undefined;
  return {
    id: `selector:${compact}`,
    label: observable.label ?? compact,
    kind,
    selector: compact,
    value_schema: valueSchema ?? {
      id: `value:${compact}`,
      label: observable.label ?? compact,
      kind: 'retained_observable',
      origin: 'declared',
      metadata: { retention: observable.retention },
    },
    origin: 'declared',
    source: {
      retained_observable_id: observable.id,
      target: observable.target ?? null,
    },
    metadata: {
      retention_target: observable.target ?? { kind, selector: compact },
      retention: observable.retention,
    },
  };
}

function knownStateHintTargets(): SelectorTargetSchema[] {
  return [
    ['path:states.mechanics.effector.pos', 'Effector position'],
    ['path:states.mechanics.effector.vel', 'Effector velocity'],
    ['path:states.net.hidden', 'Network hidden state'],
    ['path:states.net.output', 'Network output'],
    ['path:states.efferent.output', 'Motor command'],
    ['path:states.feedback.noise', 'Feedback noise'],
    ['path:task.validation_trials.targets', 'Validation targets'],
  ].map(([selector, label]) => ({
    id: `selector:${selector}`,
    label,
    kind: 'state_hint',
    selector,
    value_schema: {
      id: `value:${selector}`,
      label,
      kind: 'state_hint',
      dtype: 'vector',
      origin: 'curated_fallback',
      metadata: {},
    },
    origin: 'curated_fallback',
    source: { curated: true },
    metadata: {},
  }));
}

function validateGraphConnections(
  graph: GraphSpec,
  portIndex: PortSchemaIndex
): SchemaValidationIssue[] {
  const issues: SchemaValidationIssue[] = [];
  const occupied = new Map<string, string>();
  for (const [graphPort, binding] of Object.entries(graph.input_bindings)) {
    occupied.set(`${binding[0]}.${binding[1]}`, `input_bindings.${graphPort}`);
  }
  graph.wires.forEach((wire, index) => {
    const path = `wires.${index}`;
    const targetKey = `${wire.target_node}.${wire.target_port}`;
    const previous = occupied.get(targetKey);
    if (previous) {
      issues.push({
        type: 'graph_input_occupied',
        message: `Input ${targetKey} is already occupied`,
        severity: 'error',
        location: { path, occupied_by: previous },
      });
    }
    occupied.set(targetKey, path);
    issues.push(
      ...validateConnectionAgainstSchemaWithIndex(
        portIndex,
        wire.source_node,
        wire.source_port,
        wire.target_node,
        wire.target_port
      )
    );
    if (wire.temporality === 'recurrent' && !wire.recurrent_initializer) {
      issues.push({
        type: 'recurrent_initializer_missing',
        message: `${wire.source_node}.${wire.source_port} -> ${wire.target_node}.${wire.target_port} needs an initial value`,
        severity: 'error',
        location: { path: `${path}.recurrent_initializer` },
      });
    }
  });
  issues.push(...instantCycleIssues(graph));
  return issues;
}

function instantCycleIssues(graph: GraphSpec): SchemaValidationIssue[] {
  const adjacency = new Map<string, Array<{ target: string; wireIndex: number }>>();
  for (const nodeId of Object.keys(graph.nodes)) {
    adjacency.set(nodeId, []);
  }
  graph.wires.forEach((wire, wireIndex) => {
    if (wire.temporality === 'recurrent') return;
    if (!graph.nodes[wire.source_node] || !graph.nodes[wire.target_node]) return;
    adjacency.get(wire.source_node)?.push({ target: wire.target_node, wireIndex });
  });

  const visited = new Set<string>();
  const active = new Set<string>();
  const pathEdges: number[] = [];

  const visit = (nodeId: string): number[] | null => {
    visited.add(nodeId);
    active.add(nodeId);
    for (const { target, wireIndex } of adjacency.get(nodeId) ?? []) {
      if (!visited.has(target)) {
        pathEdges.push(wireIndex);
        const found = visit(target);
        if (found) return found;
        pathEdges.pop();
      } else if (active.has(target)) {
        return [...pathEdges, wireIndex];
      }
    }
    active.delete(nodeId);
    return null;
  };

  for (const nodeId of Object.keys(graph.nodes)) {
    if (visited.has(nodeId)) continue;
    const cycle = visit(nodeId);
    if (cycle) {
      return [
        {
          type: 'instant_cycle',
          message: 'Instant wires contain a same-step cycle; mark one cycle edge recurrent',
          severity: 'error',
          location: {
            path: `wires.${cycle[0]}`,
            cycle_wires: cycle.join(','),
          },
        },
      ];
    }
  }
  return [];
}

function validateTaskBindings(
  graph: GraphSpec,
  portIndex: PortSchemaIndex,
  taskData: TaskDataSchema[],
  taskBindingSpec?: StudioTaskBindingSpec | null
): SchemaValidationIssue[] {
  const issues: SchemaValidationIssue[] = [];
  const dataById = new Map(taskData.map((data) => [data.id.replace(/^task_data:/, ''), data]));
  const graphOccupied = new Set(graph.wires.map((wire) => `${wire.target_node}.${wire.target_port}`));
  const bindingTargets = new Set<string>();
  const bindingIds = new Set<string>();

  taskBindingSpec?.bindings.forEach((binding, index) => {
    const path = `task_binding_spec.bindings.${index}`;
    const data = dataById.get(binding.source_data_id);
    const target = findPort(portIndex, binding.target_node_id, binding.target_port, 'input');
    const targetKey = `${binding.target_node_id}.${binding.target_port}`;
    const expectedId = taskBindingId(
      binding.source_data_id,
      binding.target_node_id,
      binding.target_port,
      binding.target_graph_path
    );
    if (bindingIds.has(binding.id)) {
      issues.push({
        type: 'duplicate_task_binding',
        message: `Task binding ${binding.id} is declared more than once`,
        severity: 'error',
        location: { path },
      });
    }
    bindingIds.add(binding.id);
    if (binding.id !== expectedId) {
      issues.push({
        type: 'task_binding_id_mismatch',
        message: `Task binding id ${binding.id} does not match ${expectedId}`,
        severity: 'error',
        location: { path },
      });
    }
    if (!data) {
      issues.push({
        type: 'unknown_task_data',
        message: `Task binding source ${binding.source_data_id} is not declared`,
        severity: 'error',
        location: { path },
      });
    } else if (!data.bindable) {
      issues.push({
        type: 'task_data_not_bindable',
        message: `Task Data ${binding.source_data_id} has protocol role ${data.role} and is not bindable`,
        severity: 'error',
        location: { path },
      });
    }
    if (!graph.nodes[binding.target_node_id]) {
      issues.push({
        type: 'unknown_task_binding_target_node',
        message: `Task binding target node ${binding.target_node_id} does not exist`,
        severity: 'error',
        location: { path },
      });
    } else if (!target) {
      issues.push({
        type: 'unknown_task_binding_target_port',
        message: `Task binding target port ${targetKey} does not exist`,
        severity: 'error',
        location: { path },
      });
    } else if (data) {
      issues.push(
        ...valueSchemaCompatibilityIssues(
          data.value_schema,
          target.value_schema,
          data.label,
          target.label,
          'task_binding'
        )
      );
    }
    if (graphOccupied.has(targetKey) || bindingTargets.has(targetKey)) {
      issues.push({
        type: 'task_binding_target_occupied',
        message: `Task binding target ${targetKey} is already occupied`,
        severity: 'error',
        location: { path },
      });
    }
    bindingTargets.add(targetKey);
  });

  taskBindingSpec?.exposed_data.forEach((data, index) => {
    const path = `task_binding_spec.exposed_data.${index}`;
    const role = taskDataRole(data);
    if (data.bindable && !GRAPH_BINDABLE_TASK_DATA_ROLES.has(role)) {
      issues.push({
        type: 'task_data_bindable_role_mismatch',
        message: `Task Data ${data.id} has protocol role ${role}`,
        severity: 'error',
        location: { path },
      });
    }
    if (!data.bindable && GRAPH_BINDABLE_TASK_DATA_ROLES.has(role)) {
      issues.push({
        type: 'task_data_graph_role_not_bindable',
        message: `Task Data ${data.id} has graph-facing role ${role} but is not bindable`,
        severity: 'error',
        location: { path },
      });
    }
    if (data.bindable && (PROTOCOL_TASK_DATA_KINDS.has(data.kind) || usesProtocolTaskDataPath(data.path))) {
      issues.push({
        type: 'task_data_protocol_path_bindable',
        message: `Task Data ${data.id} points at protocol-owned task data`,
        severity: 'error',
        location: { path },
      });
    }
  });
  return issues;
}

function findPort(
  portIndex: PortSchemaIndex,
  nodeId: string,
  port: string,
  direction: 'input' | 'output'
): PortSchema | undefined {
  return portIndex.byDirectedNodePort.get(directedPortLookupKey(nodeId, port, direction));
}

function findPortAnyDirection(
  portIndex: PortSchemaIndex,
  nodeId: string,
  port: string
): PortSchema | undefined {
  return portIndex.byNodePort.get(portLookupKey(nodeId, port));
}

function valueSchemaCompatibilityIssues(
  source: ValueSchema,
  target: ValueSchema,
  sourceLabel: string,
  targetLabel: string,
  issuePrefix: string
): SchemaValidationIssue[] {
  const issues: SchemaValidationIssue[] = [];
  let knownConstraint = false;

  if (source.dtype || target.dtype) knownConstraint = true;
  if (
    source.dtype &&
    target.dtype &&
    source.dtype !== 'any' &&
    target.dtype !== 'any' &&
    !dtypesCompatible(source, target)
  ) {
    issues.push({
      type: `${issuePrefix}_dtype_mismatch`,
      message: `${sourceLabel} has dtype ${source.dtype}, but ${targetLabel} expects ${target.dtype}`,
      severity: 'error',
    });
  }

  const sourceShape = compatibilityShape(source);
  const targetShape = compatibilityShape(target);
  const sourceRank = sourceShape ? sourceShape.length : source.rank;
  const targetRank = targetShape ? targetShape.length : target.rank;

  if (sourceRank !== undefined || targetRank !== undefined) knownConstraint = true;
  if (
    sourceRank !== undefined &&
    sourceRank !== null &&
    targetRank !== undefined &&
    targetRank !== null &&
    sourceRank !== targetRank
  ) {
    issues.push({
      type: `${issuePrefix}_rank_mismatch`,
      message: `${sourceLabel} has rank ${sourceRank}, but ${targetLabel} expects rank ${targetRank}`,
      severity: 'error',
    });
  }

  if (sourceShape || targetShape) knownConstraint = true;
  if (sourceShape && targetShape && !shapesCompatible(sourceShape, targetShape)) {
    issues.push({
      type: `${issuePrefix}_shape_mismatch`,
      message: `${sourceLabel} has shape ${JSON.stringify(sourceShape)}, but ${targetLabel} expects ${JSON.stringify(targetShape)}`,
      severity: 'error',
    });
  }

  if (!knownConstraint || source.origin === 'unknown' || target.origin === 'unknown') {
    issues.push({
      type: `${issuePrefix}_unknown_schema`,
      message: `Compatibility for ${sourceLabel} -> ${targetLabel} cannot be fully checked from static schema data`,
      severity: 'warning',
    });
  }
  return issues;
}

function dtypesCompatible(source: ValueSchema, target: ValueSchema): boolean {
  const sourceDtype = normalizeDtype(source.dtype);
  const targetDtype = normalizeDtype(target.dtype);
  if (!sourceDtype || !targetDtype) return true;
  if (sourceDtype === targetDtype) return true;
  if (sourceDtype === 'any' || targetDtype === 'any') return true;
  if (isConcreteNumericDtype(sourceDtype) && semanticNumericCompatible(source, targetDtype)) {
    return true;
  }
  if (isConcreteNumericDtype(targetDtype) && semanticNumericCompatible(target, sourceDtype)) {
    return true;
  }
  if (sourceDtype === 'state' && targetDtype === 'vector') return true;
  if (sourceDtype === 'vector' && targetDtype === 'state') return true;
  return false;
}

function normalizeDtype(dtype: string | null | undefined): string | null {
  return dtype ? dtype.trim().toLowerCase() : null;
}

function isConcreteNumericDtype(dtype: string): boolean {
  return /^(u?int|float|complex)\d*$/.test(dtype) || dtype === 'number';
}

function semanticNumericCompatible(schema: ValueSchema, semanticDtype: string): boolean {
  if (semanticDtype === 'scalar') return schemaIsScalar(schema);
  if (semanticDtype === 'vector' || semanticDtype === 'matrix' || semanticDtype === 'tensor') {
    return !schemaIsScalar(schema);
  }
  return semanticDtype === 'state';
}

function schemaIsScalar(schema: ValueSchema): boolean {
  if (schema.rank === 0) return true;
  const shape = compatibilityShape(schema);
  if (Array.isArray(shape) && shape.length === 0) return true;
  return false;
}

function shapesCompatible(sourceShape: unknown[], targetShape: unknown[]): boolean {
  if (sourceShape.length !== targetShape.length) return false;
  return sourceShape.every((sourceDim, index) => {
    const targetDim = targetShape[index];
    return isWildcardDim(sourceDim) || isWildcardDim(targetDim) || sourceDim === targetDim;
  });
}

function isWildcardDim(dim: unknown): boolean {
  return dim === null || dim === undefined || dim === 'any' || dim === '*' || dim === -1;
}

function compatibilityShape(schema: ValueSchema | undefined): unknown[] | null {
  if (!schema) return null;
  const sampleShape = schema.metadata.sample_shape;
  if (Array.isArray(sampleShape)) return sampleShape;
  return schema.shape ?? null;
}
