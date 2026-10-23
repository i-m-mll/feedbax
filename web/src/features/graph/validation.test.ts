import { describe, expect, it } from 'vitest';
import { validateGraph } from './validation';
import type { GraphSpec } from '@/types/graph';
import type { ValidationResult } from './validation';

function twoNodeCycle(recurrent = false): GraphSpec {
  return {
    nodes: {
      a: {
        type: 'Gain',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
      b: {
        type: 'Gain',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [
      {
        source_node: 'a',
        source_port: 'output',
        target_node: 'b',
        target_port: 'input',
      },
      {
        source_node: 'b',
        source_port: 'output',
        target_node: 'a',
        target_port: 'input',
        temporality: recurrent ? 'recurrent' : 'instant',
        recurrent_initializer: recurrent ? { kind: 'zeros', scope: 'trial' } : null,
      },
    ],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
  };
}

describe('validateGraph recurrence', () => {
  it('reports same-step cycles only across instant wires', () => {
    expect(validateGraph(twoNodeCycle()).cycles).toEqual([['a', 'b']]);
    expect(validateGraph(twoNodeCycle(true)).cycles).toEqual([]);
  });

  it('validates port connectivity without repeated wire scans on large graphs', () => {
    const nodeCount = 140;
    const graph: GraphSpec = {
      nodes: Object.fromEntries(
        Array.from({ length: nodeCount }, (_, index) => [
          `node_${index}`,
          {
            type: 'Gain',
            params: {},
            input_ports: ['input'],
            output_ports: ['output'],
          },
        ])
      ),
      wires: Array.from({ length: nodeCount - 1 }, (_, index) => ({
        source_node: `node_${index}`,
        source_port: 'output',
        target_node: `node_${index + 1}`,
        target_port: 'input',
      })),
      input_ports: ['input'],
      output_ports: ['output'],
      input_bindings: { input: ['node_0', 'input'] },
      output_bindings: { output: [`node_${nodeCount - 1}`, 'output'] },
    };
    const originalSome = Array.prototype.some;
    let someCalls = 0;
    let result: ValidationResult | null = null;

    Array.prototype.some = function countedSome<T>(
      this: T[],
      predicate: (value: T, index: number, array: T[]) => unknown,
      thisArg?: unknown
    ): boolean {
      someCalls += 1;
      return originalSome.call(this, predicate, thisArg);
    };
    try {
      result = validateGraph(graph);
    } finally {
      Array.prototype.some = originalSome;
    }

    expect(result?.valid).toBe(true);
    expect(someCalls).toBe(0);
  });
});
