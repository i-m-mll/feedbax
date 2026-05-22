import { beforeEach, describe, expect, it } from 'vitest';
import { useGraphStore } from './graphStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {
    network: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
  },
};

function graphWithNetworkSubgraph(): GraphSpec {
  return {
    nodes: {
      source: {
        type: 'Constant',
        params: {},
        input_ports: [],
        output_ports: ['output'],
      },
      network: {
        type: 'Network',
        params: {},
        input_ports: ['input', 'feedback'],
        output_ports: ['output'],
      },
      sink: {
        type: 'Gain',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [
      {
        source_node: 'source',
        source_port: 'output',
        target_node: 'network',
        target_port: 'feedback',
      },
      {
        source_node: 'network',
        source_port: 'output',
        target_node: 'sink',
        target_port: 'input',
      },
    ],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
    subgraphs: {
      network: {
        nodes: {
          input_mux: {
            type: 'Mux',
            params: { n_inputs: 2 },
            input_ports: ['in_0', 'in_1'],
            output_ports: ['output'],
          },
          cell: {
            type: 'GRU',
            params: { input_size: 6, hidden_size: 100 },
            input_ports: ['input', 'hidden'],
            output_ports: ['output', 'hidden'],
          },
        },
        wires: [
          {
            source_node: 'input_mux',
            source_port: 'output',
            target_node: 'cell',
            target_port: 'input',
          },
        ],
        input_ports: ['input', 'feedback'],
        output_ports: ['output'],
        input_bindings: {
          input: ['input_mux', 'in_0'],
          feedback: ['input_mux', 'in_1'],
        },
        output_bindings: {
          output: ['cell', 'output'],
        },
      },
    },
  };
}

describe('graphStore boundary aliases', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(graphWithNetworkSubgraph(), uiState);
  });

  it('renames a subgraph boundary input without renaming the internal bound port', () => {
    useGraphStore
      .getState()
      .renameSubgraphBoundaryPort('network', 'input', 'feedback', 'proprioception');

    const graph = useGraphStore.getState().graph;
    const network = graph.nodes.network;
    const subgraph = graph.subgraphs?.network;

    expect(network.input_ports).toEqual(['input', 'proprioception']);
    expect(subgraph?.input_ports).toEqual(['input', 'proprioception']);
    expect(subgraph?.input_bindings.proprioception).toEqual(['input_mux', 'in_1']);
    expect(subgraph?.input_bindings.feedback).toBeUndefined();
    expect(graph.wires[0]).toMatchObject({
      target_node: 'network',
      target_port: 'proprioception',
    });
  });
});

describe('graphStore recurrent connections', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(
      {
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
        ],
        input_ports: [],
        output_ports: [],
        input_bindings: {},
        output_bindings: {},
      },
      {
        viewport: { x: 0, y: 0, zoom: 1 },
        node_states: {
          a: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
          b: { position: { x: 240, y: 0 }, collapsed: false, selected: false },
        },
      }
    );
  });

  it('marks the newly dropped cycle-closing edge as recurrent', () => {
    useGraphStore.getState().onConnect({
      source: 'b',
      sourceHandle: 'output',
      target: 'a',
      targetHandle: 'input',
    });

    const wire = useGraphStore
      .getState()
      .graph.wires.find((item) => item.source_node === 'b' && item.target_node === 'a');
    const edge = useGraphStore
      .getState()
      .edges.find((item) => item.type === 'routed' && item.source === 'b' && item.target === 'a');

    expect(wire?.temporality).toBe('recurrent');
    expect(wire?.recurrent_initializer?.kind).toBe('zeros');
    expect(edge?.data?.temporality).toBe('recurrent');
  });
});
