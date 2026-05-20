import type { ComponentDefinition } from '@/types/components';
import type { GraphSpec } from '@/types/graph';
import type {
  PortSchema,
  StudioSchemaOrigin,
  StudioSchemaRegistry,
  StudioValidationIssue as SchemaValidationIssue,
  TaskDataSchema,
  ValueSchema,
} from '@/types/workspace';
import type { StudioTaskBindingSpec } from '@/types/workspace';

export function projectStudioSchema(
  graph: GraphSpec,
  components: ComponentDefinition[],
  taskBindingSpec?: StudioTaskBindingSpec | null
): StudioSchemaRegistry {
  const componentMap = new Map(components.map((component) => [component.name, component]));
  const ports = enumerateGraphPorts(graph, componentMap);
  const taskData = enumerateTaskData(taskBindingSpec);
  markBoundPorts(ports, taskBindingSpec);

  return {
    kind: 'studio_schema_registry',
    schema_version: 'feedbax.schema.v1',
    generated_at: new Date().toISOString(),
    ports,
    task_data: taskData,
    selector_targets: [],
    issues: [
      ...validateGraphConnections(graph, ports),
      ...validateTaskBindings(graph, ports, taskData, taskBindingSpec),
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
        componentPortSchema(nodeId, node.type, port, 'input', component?.port_types?.inputs?.[port])
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
        ...data.metadata,
        task_data_path: data.path,
        value_spec: value ?? null,
      },
    };
    return {
      id: `task_data:${data.id}`,
      label: data.label,
      kind: data.kind,
      path: data.path,
      bindable: data.bindable,
      value_schema: valueSchema,
      origin: 'declared',
      metadata: data.metadata,
    };
  });
}

function markBoundPorts(ports: PortSchema[], taskBindingSpec?: StudioTaskBindingSpec | null): void {
  for (const binding of taskBindingSpec?.bindings ?? []) {
    const port = findPort(ports, binding.target_node_id, binding.target_port, 'input');
    if (port) {
      port.bound_task_data_id = `task_data:${binding.source_data_id}`;
    }
  }
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
        message: `Task Data ${binding.source_data_id} is not bindable`,
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
    source.dtype !== target.dtype
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
