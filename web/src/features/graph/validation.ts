import type { GraphSpec, WireSpec } from '@/types/graph';

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  cycles: string[][];
}

export interface ValidationError {
  type: 'missing_input' | 'invalid_wire' | 'duplicate_wire' | 'unbound_port';
  message: string;
  location?: { node?: string; port?: string; wire?: WireSpec };
}

export interface ValidationWarning {
  type: 'unconnected_output' | 'potential_type_mismatch';
  message: string;
  location?: { node?: string; port?: string };
}

export function validateGraph(graph: GraphSpec): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  for (const [nodeName, node] of Object.entries(graph.nodes)) {
    for (const inputPort of node.input_ports) {
      const hasWire = graph.wires.some(
        (w) => w.target_node === nodeName && w.target_port === inputPort
      );
      const hasBinding = Object.values(graph.input_bindings).some(
        ([n, p]) => n === nodeName && p === inputPort
      );

      if (!hasWire && !hasBinding) {
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
