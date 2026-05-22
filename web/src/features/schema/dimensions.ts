import type { GraphSpec } from '@/types/graph';
import type { PortSchema, StudioSchemaRegistry, ValueSchema } from '@/types/workspace';

export interface DimensionConstraint {
  node_id: string;
  component_type: string;
  param: string;
  port: string;
  direction: 'input' | 'output';
  boundary_port?: string | null;
  inferred_value: number | null;
  current_value: number | null;
  status: 'inferred' | 'conflict' | 'unknown';
  source_ids: string[];
  message: string;
}

interface DimensionTarget {
  port: string;
  direction: 'input' | 'output';
  param: string;
  source_ports?: string[];
  boundary_port?: string | null;
}

const DIMENSION_TARGETS: Record<string, DimensionTarget[]> = {
  Network: [
    {
      port: 'input+feedback',
      direction: 'input',
      param: 'input_size',
      source_ports: ['input', 'feedback'],
      boundary_port: null,
    },
    { port: 'output', direction: 'output', param: 'out_size' },
  ],
  Linear: [
    { port: 'input', direction: 'input', param: 'input_size' },
    { port: 'output', direction: 'output', param: 'output_size' },
  ],
  MLP: [
    { port: 'input', direction: 'input', param: 'input_size' },
    { port: 'output', direction: 'output', param: 'output_size' },
  ],
  GRU: [
    { port: 'input', direction: 'input', param: 'input_size' },
    { port: 'hidden', direction: 'input', param: 'hidden_size' },
    { port: 'output', direction: 'output', param: 'hidden_size' },
    { port: 'hidden', direction: 'output', param: 'hidden_size' },
  ],
  LSTM: [
    { port: 'input', direction: 'input', param: 'input_size' },
    { port: 'hidden', direction: 'input', param: 'hidden_size' },
    { port: 'output', direction: 'output', param: 'hidden_size' },
    { port: 'hidden', direction: 'output', param: 'hidden_size' },
  ],
};

export function deriveDimensionConstraints(
  graph: GraphSpec,
  registry: Pick<StudioSchemaRegistry, 'ports' | 'task_data'>
): DimensionConstraint[] {
  const constraints: DimensionConstraint[] = [];
  const ports = new Map(
    registry.ports
      .filter((port) => port.node_id)
      .map((port) => [portKey(port.node_id!, port.port, port.direction), port])
  );
  const taskData = new Map(registry.task_data.map((data) => [data.id, data.value_schema]));

  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    for (const target of DIMENSION_TARGETS[node.type] ?? []) {
      const port = ports.get(portKey(nodeId, target.port, target.direction));
      const sourcePorts = target.source_ports ?? [target.port];
      if (!port && !target.source_ports) continue;
      const sources =
        target.direction === 'input'
          ? sourcePorts.flatMap((sourcePort) =>
              inboundSchemas(graph, registry.ports, taskData, nodeId, sourcePort)
            )
          : outboundConsumerSchemas(graph, registry.ports, nodeId, target.port);
      const inferred = target.source_ports
        ? summedVectorWidth(sources.map((source) => source.schema))
        : uniqueVectorWidth(sources.map((source) => source.schema));
      if (inferred === null && sources.length === 0) {
        continue;
      }
      const current = positiveIntParam(node.params, target.param);
      const status =
        inferred === null ? 'unknown' : current !== null && current !== inferred ? 'conflict' : 'inferred';
      constraints.push({
        node_id: nodeId,
        component_type: node.type,
        param: target.param,
        port: target.port,
        direction: target.direction,
        boundary_port: target.boundary_port === undefined ? target.port : target.boundary_port,
        inferred_value: inferred,
        current_value: current,
        status,
        source_ids: sources.map((source) => source.id),
        message: dimensionConstraintMessage(nodeId, target.param, current, inferred, status),
      });
    }
  }
  return constraints;
}

export function applyBoundaryOverrides(
  registry: StudioSchemaRegistry,
  boundaryOverrides: Map<string, ValueSchema>
): StudioSchemaRegistry {
  if (boundaryOverrides.size === 0) return registry;
  return {
    ...registry,
    ports: registry.ports.map((port) => {
      if (port.node_id) return port;
      const override = boundaryOverrides.get(boundaryOverrideKey(port.direction, port.port));
      if (!override) return port;
      return {
        ...port,
        value_schema: {
          ...port.value_schema,
          dtype: override.dtype ?? port.value_schema.dtype,
          shape: override.shape ?? port.value_schema.shape,
          rank: override.rank ?? (override.shape ? override.shape.length : port.value_schema.rank),
          metadata: {
            ...port.value_schema.metadata,
            inherited_boundary_schema: true,
          },
        },
      };
    }),
  };
}

export function deriveSubgraphBoundaryOverrides(
  parentGraph: GraphSpec,
  childNodeId: string,
  parentRegistry: Pick<StudioSchemaRegistry, 'ports' | 'task_data'>
): Map<string, ValueSchema> {
  const overrides = new Map<string, ValueSchema>();
  if (!parentGraph.nodes[childNodeId]) return overrides;

  const parentConstraints = deriveDimensionConstraints(parentGraph, parentRegistry);
  const constrainedShapes = new Map<string, number>();
  for (const constraint of parentConstraints) {
    if (constraint.node_id !== childNodeId || typeof constraint.inferred_value !== 'number') continue;
    if (constraint.boundary_port === null) continue;
    constrainedShapes.set(
      boundaryOverrideKey(constraint.direction, constraint.boundary_port ?? constraint.port),
      constraint.inferred_value
    );
  }

  const taskData = new Map(parentRegistry.task_data.map((data) => [data.id, data.value_schema]));
  for (const port of parentRegistry.ports) {
    if (port.node_id !== childNodeId) continue;
    const key = boundaryOverrideKey(port.direction, port.port);
    const inferred =
      constrainedShapes.get(key) ??
      (port.direction === 'input'
        ? uniqueVectorWidth(
            inboundSchemas(parentGraph, parentRegistry.ports, taskData, childNodeId, port.port)
              .map((source) => source.schema)
          )
        : null);
    overrides.set(key, {
      ...port.value_schema,
      shape: typeof inferred === 'number' ? [inferred] : port.value_schema.shape,
      rank: typeof inferred === 'number' ? 1 : port.value_schema.rank,
    });
  }
  return overrides;
}

export function boundaryOverrideKey(direction: 'input' | 'output', port: string): string {
  return `${direction}:${port}`;
}

function inboundSchemas(
  graph: GraphSpec,
  ports: PortSchema[],
  taskData: Map<string, ValueSchema>,
  nodeId: string,
  port: string
): Array<{ id: string; schema: ValueSchema }> {
  const sources: Array<{ id: string; schema: ValueSchema }> = [];
  const sourceByPort = new Map(
    ports
      .filter((item) => item.node_id && item.direction === 'output')
      .map((item) => [portKey(item.node_id!, item.port, item.direction), item])
  );
  const target = ports.find(
    (item) => item.node_id === nodeId && item.port === port && item.direction === 'input'
  );
  const boundTaskDataId = target?.bound_task_data_id;
  if (boundTaskDataId) {
    const schema = taskData.get(boundTaskDataId);
    if (schema) sources.push({ id: boundTaskDataId, schema });
  }
  for (const graphPort of graphBoundaryPorts(ports, 'input', nodeId, port)) {
    sources.push({ id: graphPort.id, schema: graphPort.value_schema });
  }
  for (const wire of graph.wires) {
    if (wire.target_node !== nodeId || wire.target_port !== port) continue;
    const source = sourceByPort.get(portKey(wire.source_node, wire.source_port, 'output'));
    if (source) sources.push({ id: source.id, schema: source.value_schema });
  }
  return sources;
}

function outboundConsumerSchemas(
  graph: GraphSpec,
  ports: PortSchema[],
  nodeId: string,
  port: string
): Array<{ id: string; schema: ValueSchema }> {
  const targets = new Map(
    ports
      .filter((item) => item.node_id && item.direction === 'input')
      .map((item) => [portKey(item.node_id!, item.port, item.direction), item])
  );
  const consumers: Array<{ id: string; schema: ValueSchema }> = [];
  for (const graphPort of graphBoundaryPorts(ports, 'output', nodeId, port)) {
    consumers.push({ id: graphPort.id, schema: graphPort.value_schema });
  }
  for (const wire of graph.wires) {
    if (wire.source_node !== nodeId || wire.source_port !== port) continue;
    const target = targets.get(portKey(wire.target_node, wire.target_port, 'input'));
    if (target) consumers.push({ id: target.id, schema: target.value_schema });
  }
  return consumers;
}

function graphBoundaryPorts(
  ports: PortSchema[],
  direction: 'input' | 'output',
  nodeId: string,
  port: string
): PortSchema[] {
  return ports.filter((item) => {
    if (item.node_id || item.direction !== direction) return false;
    const binding = item.metadata.binding;
    if (!binding || typeof binding !== 'object') return false;
    const typed = binding as { node_id?: unknown; port?: unknown };
    return typed.node_id === nodeId && typed.port === port;
  });
}

function uniqueVectorWidth(schemas: ValueSchema[]): number | null {
  const widths = schemas
    .map((schema) => compatibilityShape(schema))
    .filter((shape): shape is unknown[] => Array.isArray(shape) && shape.length === 1)
    .map((shape) => shape[0])
    .filter((value): value is number => Number.isInteger(value) && typeof value === 'number' && value > 0);
  const unique = [...new Set(widths)];
  return unique.length === 1 ? unique[0] : null;
}

function summedVectorWidth(schemas: ValueSchema[]): number | null {
  if (schemas.length === 0) return null;
  const widths = schemas.map((schema) => {
    const shape = compatibilityShape(schema);
    if (!Array.isArray(shape) || shape.length !== 1) return null;
    const value = shape[0];
    return Number.isInteger(value) && typeof value === 'number' && value > 0 ? value : null;
  });
  if (widths.some((width) => width === null)) return null;
  return (widths as number[]).reduce((sum, width) => sum + width, 0);
}

function compatibilityShape(schema: ValueSchema): unknown[] | null {
  const sampleShape = schema.metadata.sample_shape;
  if (Array.isArray(sampleShape)) return sampleShape;
  return schema.shape ?? null;
}

function positiveIntParam(params: Record<string, unknown>, key: string): number | null {
  const value = params[key];
  return Number.isInteger(value) && typeof value === 'number' && value > 0 ? value : null;
}

function portKey(nodeId: string, port: string, direction: 'input' | 'output'): string {
  return `${nodeId}.${port}:${direction}`;
}

function dimensionConstraintMessage(
  nodeId: string,
  param: string,
  current: number | null,
  inferred: number | null,
  status: DimensionConstraint['status']
): string {
  if (status === 'conflict') {
    return `${nodeId}.${param} is ${current}, but connected schemas imply ${inferred}`;
  }
  if (status === 'inferred') {
    return `${nodeId}.${param} is constrained to ${inferred} by connected schemas`;
  }
  return `${nodeId}.${param} cannot be inferred from the connected schemas`;
}
