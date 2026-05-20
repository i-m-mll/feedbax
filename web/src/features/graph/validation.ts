import type { GraphSpec, WireSpec } from '@/types/graph';
import type { StudioSchemaRegistry } from '@/types/workspace';

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  cycles: string[][];
}

export interface ValidationError {
  type:
    | 'missing_input'
    | 'invalid_wire'
    | 'duplicate_wire'
    | 'unbound_port'
    | 'graph_input_occupied'
    | 'wrong_source_port_direction'
    | 'wrong_target_port_direction'
    | 'unknown_source_node'
    | 'unknown_source_port'
    | 'unknown_target_node'
    | 'unknown_target_port'
    | 'graph_wire_dtype_mismatch'
    | 'graph_wire_rank_mismatch'
    | 'graph_wire_shape_mismatch'
    | 'task_binding_target_occupied'
    | 'unknown_task_data'
    | 'task_data_not_bindable'
    | 'unknown_task_binding_target_node'
    | 'unknown_task_binding_target_port'
    | 'task_binding_dtype_mismatch'
    | 'task_binding_rank_mismatch'
    | 'task_binding_shape_mismatch'
    | string;
  message: string;
  location?: { node?: string; port?: string; wire?: WireSpec } & Record<string, unknown>;
}

export interface ValidationWarning {
  type:
    | 'unconnected_output'
    | 'potential_type_mismatch'
    | 'graph_wire_unknown_schema'
    | 'task_binding_unknown_schema'
    | string;
  message: string;
  location?: { node?: string; port?: string } & Record<string, unknown>;
}

export function validateGraph(
  graph: GraphSpec,
  schemaRegistry?: Pick<StudioSchemaRegistry, 'issues' | 'ports'> | null
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  const taskBoundInputs = new Set(
    (schemaRegistry?.ports ?? [])
      .filter((port) => port.direction === 'input' && port.node_id && port.bound_task_data_id)
      .map((port) => `${port.node_id}.${port.port}`)
  );

  for (const [nodeName, node] of Object.entries(graph.nodes)) {
    for (const inputPort of node.input_ports) {
      const hasWire = graph.wires.some(
        (w) => w.target_node === nodeName && w.target_port === inputPort
      );
      const hasBinding = Object.values(graph.input_bindings).some(
        ([n, p]) => n === nodeName && p === inputPort
      );
      const hasTaskDataBinding = taskBoundInputs.has(`${nodeName}.${inputPort}`);

      if (!hasWire && !hasBinding && !hasTaskDataBinding) {
        errors.push({
          type: 'missing_input',
          message: `Input port '${nodeName}.${inputPort}' is not connected`,
          location: { node: nodeName, port: inputPort },
        });
      }
    }

    for (const outputPort of node.output_ports) {
      const hasWire = graph.wires.some(
        (w) => w.source_node === nodeName && w.source_port === outputPort
      );
      const hasBinding = Object.values(graph.output_bindings).some(
        ([n, p]) => n === nodeName && p === outputPort
      );

      if (!hasWire && !hasBinding) {
        warnings.push({
          type: 'unconnected_output',
          message: `Output port '${nodeName}.${outputPort}' is not connected`,
          location: { node: nodeName, port: outputPort },
        });
      }
    }
  }

  for (const issue of schemaRegistry?.issues ?? []) {
    const validationIssue = {
      type: issue.type,
      message: issue.message,
      location: issue.location ?? undefined,
    };
    if (issue.severity === 'error') {
      errors.push(validationIssue);
    } else {
      warnings.push(validationIssue);
    }
  }

  const cycles = detectCycles(graph);

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    cycles,
  };
}

function detectCycles(graph: GraphSpec): string[][] {
  const adjacency: Record<string, Set<string>> = {};
  for (const nodeName of Object.keys(graph.nodes)) {
    adjacency[nodeName] = new Set();
  }
  for (const wire of graph.wires) {
    adjacency[wire.source_node].add(wire.target_node);
  }

  const cycles: string[][] = [];
  const visited = new Set<string>();
  const recursionStack = new Set<string>();
  const path: string[] = [];

  function dfs(node: string): void {
    visited.add(node);
    recursionStack.add(node);
    path.push(node);

    for (const neighbor of adjacency[node] ?? []) {
      if (!visited.has(neighbor)) {
        dfs(neighbor);
      } else if (recursionStack.has(neighbor)) {
        const cycleStart = path.indexOf(neighbor);
        cycles.push(path.slice(cycleStart));
      }
    }

    path.pop();
    recursionStack.delete(node);
  }

  for (const node of Object.keys(graph.nodes)) {
    if (!visited.has(node)) {
      dfs(node);
    }
  }

  return cycles;
}
