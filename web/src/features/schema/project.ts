import type { ComponentDefinition } from '@/types/components';
import type { GraphSpec } from '@/types/graph';
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

export function projectStudioSchema(
  graph: GraphSpec,
  components: ComponentDefinition[],
  taskBindingSpec?: StudioTaskBindingSpec | null
): StudioSchemaRegistry {
  const componentMap = new Map(components.map((component) => [component.name, component]));
  const schemaGraph = normalizeDynamicPorts(graph, taskBindingSpec);
  const ports = enumerateGraphPorts(schemaGraph, componentMap);
  const taskData = enumerateTaskData(taskBindingSpec);
  markBoundPorts(ports, taskBindingSpec);
  const selectorTargets = [
    ...portSelectorTargets(ports),
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
      ...validateGraphConnections(schemaGraph, ports),
      ...validateTaskBindings(schemaGraph, ports, taskData, taskBindingSpec),
      ...validateInterventionSchema(graph.taps, { selector_targets: selectorTargets }),
    ],
    metadata: { projected_by: 'feedbax.web.projectStudioSchema' },
  };
}

export function validateConnectionAgainstSchema(
  registry: Pick<StudioSchemaRegistry, 'ports'>,
  sourceNode: string,
  sourcePort: string,
  targetNode: string,
  targetPort: string
): SchemaValidationIssue[] {
  const source = findPort(registry.ports, sourceNode, sourcePort, 'output');
  const target = findPort(registry.ports, targetNode, targetPort, 'input');
  const sourceAnyDirection = findPortAnyDirection(registry.ports, sourceNode, sourcePort);
  const targetAnyDirection = findPortAnyDirection(registry.ports, targetNode, targetPort);
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

export function hasBlockingSchemaIssue(issues: SchemaValidationIssue[]): boolean {
  return issues.some((issue) => issue.severity === 'error');
}

function enumerateGraphPorts(
  graph: GraphSpec,
  componentMap: Map<string, ComponentDefinition>
): PortSchema[] {
  const ports: PortSchema[] = [];
  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const component = componentMap.get(node.type);
    const inputPorts = node.input_ports.length > 0 ? node.input_ports : component?.input_ports ?? [];
    const outputPorts =
      node.output_ports.length > 0 ? node.output_ports : component?.output_ports ?? [];
    for (const port of inputPorts) {
      ports.push(
        componentPortSchema(nodeId, node.type, port, 'input', componentInputPortType(node.type, port, component))
      );
    }
    for (const port of outputPorts) {
      ports.push(
        componentPortSchema(
          nodeId,
          node.type,
          port,
          'output',
          component?.port_types?.outputs?.[port]
        )
      );
    }
  }
  for (const port of graph.input_ports) {
    ports.push(graphPortSchema(port, 'input', graph.input_bindings[port]));
  }
  for (const port of graph.output_ports) {
    ports.push(graphPortSchema(port, 'output'));
  }
  return ports;
}

function componentInputPortType(
  componentType: string,
  port: string,
  component?: ComponentDefinition
): { dtype: string; shape?: number[] | null; rank?: number } | undefined {
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
  port: string,
  direction: 'input' | 'output',
  portType?: { dtype: string; shape?: number[] | null; rank?: number }
): PortSchema {
  const portId = `port:${nodeId}.${port}:${direction}`;
  const origin: StudioSchemaOrigin = portType ? 'declared' : 'unknown';
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
      dtype: portType?.dtype ?? null,
      shape: portType?.shape ?? null,
      rank: portType?.rank ?? null,
      origin,
      metadata: { component_type: componentType },
    },
    origin,
    metadata: {},
  };
}

function graphPortSchema(
  port: string,
  direction: 'input' | 'output',
  binding?: [string, string]
): PortSchema {
  const portId = `port:graph.${port}:${direction}`;
  const metadata = binding ? { binding: { node_id: binding[0], port: binding[1] } } : {};
  return {
    id: portId,
    label: `graph.${port}`,
    port,
    direction,
    value_schema: {
      id: `value:${portId}`,
      label: `graph.${port}`,
      kind: 'graph_port',
      origin: 'inferred_static',
      metadata,
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
    const metadata = {
      ...data.metadata,
      task_data_role: role,
      task_data_surface: surface,
    };
    const valueSchema: ValueSchema = {
      id: `value:task_data:${data.id}`,
      label: data.label,
      kind: 'task_data',
      dtype: data.dtype ?? value?.dtype ?? null,
      shape: data.expected_shape ?? value?.shape ?? null,
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

function markBoundPorts(ports: PortSchema[], taskBindingSpec?: StudioTaskBindingSpec | null): void {
  for (const binding of taskBindingSpec?.bindings ?? []) {
    const port = findPort(ports, binding.target_node_id, binding.target_port, 'input');
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

function validateGraphConnections(graph: GraphSpec, ports: PortSchema[]): SchemaValidationIssue[] {
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
      ...validateConnectionAgainstSchema(
        { ports },
        wire.source_node,
        wire.source_port,
        wire.target_node,
        wire.target_port
      )
    );
  });
  return issues;
}

function validateTaskBindings(
  graph: GraphSpec,
  ports: PortSchema[],
  taskData: TaskDataSchema[],
  taskBindingSpec?: StudioTaskBindingSpec | null
): SchemaValidationIssue[] {
  const issues: SchemaValidationIssue[] = [];
  const dataById = new Map(taskData.map((data) => [data.id.replace(/^task_data:/, ''), data]));
  const graphOccupied = new Set(graph.wires.map((wire) => `${wire.target_node}.${wire.target_port}`));
  const bindingTargets = new Set<string>();

  taskBindingSpec?.bindings.forEach((binding, index) => {
    const path = `task_binding_spec.bindings.${index}`;
    const data = dataById.get(binding.source_data_id);
    const target = findPort(ports, binding.target_node_id, binding.target_port, 'input');
    const targetKey = `${binding.target_node_id}.${binding.target_port}`;
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
  ports: PortSchema[],
  nodeId: string,
  port: string,
  direction: 'input' | 'output'
): PortSchema | undefined {
  return ports.find(
    (item) => item.node_id === nodeId && item.port === port && item.direction === direction
  );
}

function findPortAnyDirection(
  ports: PortSchema[],
  nodeId: string,
  port: string
): PortSchema | undefined {
  return ports.find((item) => item.node_id === nodeId && item.port === port);
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

  if (source.rank !== undefined || target.rank !== undefined) knownConstraint = true;
  if (
    source.rank !== undefined &&
    source.rank !== null &&
    target.rank !== undefined &&
    target.rank !== null &&
    source.rank !== target.rank
  ) {
    issues.push({
      type: `${issuePrefix}_rank_mismatch`,
      message: `${sourceLabel} has rank ${source.rank}, but ${targetLabel} expects rank ${target.rank}`,
      severity: 'error',
    });
  }

  if (source.shape || target.shape) knownConstraint = true;
  if (source.shape && target.shape && !shapesCompatible(source.shape, target.shape)) {
    issues.push({
      type: `${issuePrefix}_shape_mismatch`,
      message: `${sourceLabel} has shape ${JSON.stringify(source.shape)}, but ${targetLabel} expects ${JSON.stringify(target.shape)}`,
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
  if (Array.isArray(schema.shape) && schema.shape.length === 0) return true;
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
