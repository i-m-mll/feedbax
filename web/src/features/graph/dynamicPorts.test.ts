import { describe, expect, it } from 'vitest';
import {
  nextMuxInputPort,
  normalizeDynamicPorts,
  visibleMuxInputPorts,
} from '@/features/graph/dynamicPorts';
import type { GraphSpec } from '@/types/graph';
import type { StudioTaskBindingSpec } from '@/types/workspace';

function graph(inputPorts = ['in_0', 'in_1']): GraphSpec {
  return {
    nodes: {
      mux: {
        type: 'Mux',
        params: { n_inputs: inputPorts.length },
        input_ports: inputPorts,
        output_ports: ['output'],
      },
      source: {
        type: 'Gain',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
  };
}

function taskBindings(...ports: string[]): StudioTaskBindingSpec {
  return {
    schema_version: 'feedbax.studio.task_bindings.v2',
    exposed_data: ports.map((port) => ({
      id: port,
      label: port,
      kind: 'signal',
      role: 'model_input',
      path: `inputs.${port}`,
      bindable: true,
      metadata: {},
    })),
    bindings: ports.map((port) => ({
      id: `task:${port}->mux:${port}`,
      source_data_id: port,
      target_node_id: 'mux',
      target_port: port,
      role: 'model_input',
      metadata: {},
    })),
    metadata: {},
  };
}

describe('dynamic graph ports', () => {
  it('grows a mux to the highest wired input and shrinks trailing unused ports', () => {
    const expanded = normalizeDynamicPorts({
      ...graph(),
      wires: [
        {
          source_node: 'source',
          source_port: 'output',
          target_node: 'mux',
          target_port: 'in_2',
        },
      ],
    });

    expect(expanded.nodes.mux.input_ports).toEqual(['in_0', 'in_1', 'in_2']);
    expect(expanded.nodes.mux.params.n_inputs).toBe(3);

    const shrunk = normalizeDynamicPorts({
      ...expanded,
      wires: [],
    });

    expect(shrunk.nodes.mux.input_ports).toEqual(['in_0', 'in_1']);
    expect(shrunk.nodes.mux.params.n_inputs).toBe(2);
  });

  it('keeps task-bound mux inputs materialized for rendering', () => {
    const bindings = taskBindings('in_0', 'in_1', 'in_2');
    const normalized = normalizeDynamicPorts(graph(), bindings);
    const visible = visibleMuxInputPorts(graph(), 'mux', bindings);

    expect(normalized.nodes.mux.input_ports).toEqual(['in_0', 'in_1', 'in_2']);
    expect(nextMuxInputPort(graph(), 'mux', bindings)).toBe('in_3');
    expect(visible?.ports).toEqual(['in_0', 'in_1', 'in_2', 'in_3']);
    expect(visible?.nextPort).toBe('in_3');
  });

  it('shows one extra mux input only when all current inputs are occupied', () => {
    const base = graph();

    expect(nextMuxInputPort(base, 'mux')).toBeNull();
    expect(
      nextMuxInputPort(
        {
          ...base,
          wires: [
            {
              source_node: 'source',
              source_port: 'output',
              target_node: 'mux',
              target_port: 'in_0',
            },
          ],
        },
        'mux',
        taskBindings('in_1')
      )
    ).toBe('in_2');
  });
});
