import { describe, expect, it } from 'vitest';
import { validateGraph } from './validation';
import type { GraphSpec } from '@/types/graph';

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
});
