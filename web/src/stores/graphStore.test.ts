import { beforeEach, describe, expect, it } from 'vitest';
import { useGraphStore } from './graphStore';
import type { GraphSpec, GraphUIState } from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';

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

describe('graphStore retained observables', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(graphWithNetworkSubgraph(), uiState);
  });

  it('adds, updates, and removes explicit retained observables on the graph spec', () => {
    useGraphStore.getState().addRetainedObservable({
      id: 'obs:network_output',
      label: 'Network output',
      selector: 'port:network.output',
      target: {
        kind: 'port',
        selector: 'port:network.output',
        node_id: 'network',
        port: 'output',
        timing: 'output',
      },
      retention: { mode: 'trajectory' },
      metadata: {},
    });

    expect(useGraphStore.getState().graph.retained_observables).toEqual([
      expect.objectContaining({
        id: 'obs:network_output',
        retention: { mode: 'trajectory' },
      }),
    ]);

    useGraphStore.getState().updateRetainedObservable('obs:network_output', {
      label: 'Network output window',
      retention: { mode: 'window', window_size: 16 },
    });

    expect(useGraphStore.getState().graph.retained_observables?.[0]).toMatchObject({
      label: 'Network output window',
      retention: { mode: 'window', window_size: 16 },
    });

    useGraphStore.getState().removeRetainedObservable('obs:network_output');

    expect(useGraphStore.getState().graph.retained_observables).toEqual([]);
  });
});

describe('graphStore template insertion', () => {
  const baseGraph: GraphSpec = {
    nodes: {
      source: {
        type: 'Constant',
        params: {},
        input_ports: [],
        output_ports: ['output'],
      },
    },
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
  };
  const baseUi: GraphUIState = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {
      source: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
    },
  };
  const templateComponent: ComponentDefinition = {
    name: 'Network Template',
    category: 'Neural Networks',
    description: 'Template',
    param_schema: [],
    input_ports: ['input', 'feedback'],
    output_ports: ['output'],
    icon: 'network',
    default_params: {},
    template_id: 'feedbax.templates.network',
    template_kind: 'executable',
    template_graph: {
      nodes: {
        input_mux: {
          type: 'Mux',
          params: { n_inputs: 2 },
          input_ports: ['in_0', 'in_1'],
          output_ports: ['output'],
        },
        cell: {
          type: 'GRU',
          params: { input_size: 2, hidden_size: 3 },
          input_ports: ['input', 'hidden'],
          output_ports: ['output', 'hidden'],
        },
        readout: {
          type: 'Linear',
          params: { input_size: 3, output_size: 1 },
          input_ports: ['input'],
          output_ports: ['output'],
        },
      },
      wires: [
        {
          source_node: 'input_mux',
          source_port: 'output',
          target_node: 'cell',
          target_port: 'input',
        },
        {
          source_node: 'cell',
          source_port: 'output',
          target_node: 'readout',
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
        output: ['readout', 'output'],
      },
      subgraphs: {
        cell: {
          nodes: {
            inner: {
              type: 'Linear',
              params: { input_size: 3, output_size: 3 },
              input_ports: ['input'],
              output_ports: ['output'],
            },
          },
          wires: [],
          input_ports: ['input'],
          output_ports: ['output'],
          input_bindings: { input: ['inner', 'input'] },
          output_bindings: { output: ['inner', 'output'] },
        },
      },
    },
    template_ui_state: {
      viewport: { x: 0, y: 0, zoom: 1 },
      node_states: {
        input_mux: { position: { x: 100, y: 50 }, collapsed: false, selected: false },
        cell: { position: { x: 340, y: 50 }, collapsed: false, selected: false },
        readout: { position: { x: 580, y: 50 }, collapsed: false, selected: false },
      },
      subgraph_states: {
        cell: {
          viewport: { x: 0, y: 0, zoom: 1 },
          node_states: {
            inner: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
          },
        },
      },
    },
  };

  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(baseGraph, baseUi);
  });

  it('imports template graphs into the active graph level without replacing existing nodes', () => {
    useGraphStore.getState().addNodeFromComponent(templateComponent, { x: 200, y: 160 });

    const state = useGraphStore.getState();
    expect(Object.keys(state.graph.nodes)).toEqual(['source', 'input_mux', 'cell', 'readout']);
    expect(state.graph.nodes['Network Template']).toBeUndefined();
    expect(state.graph.nodes.cell.params._subgraph).toBeUndefined();
    expect(state.graph.wires).toEqual([
      {
        source_node: 'input_mux',
        source_port: 'output',
        target_node: 'cell',
        target_port: 'input',
      },
      {
        source_node: 'cell',
        source_port: 'output',
        target_node: 'readout',
        target_port: 'input',
      },
    ]);
    expect(state.graph.subgraphs?.cell?.nodes.inner.type).toBe('Linear');
    expect(state.uiState.subgraph_states?.cell?.node_states.inner).toBeDefined();
    expect(state.uiState.node_states.input_mux.position).toEqual({ x: 200, y: 160 });
    expect(state.nodes.find((node) => node.id === 'cell')?.type).toBe('component');
    expect(state.edges.some((edge) => edge.source === 'input_mux' && edge.target === 'cell')).toBe(true);
  });

  it('keeps later normal component insertion synchronous after template import', () => {
    const gain: ComponentDefinition = {
      name: 'Gain',
      category: 'Math',
      description: 'Gain',
      param_schema: [],
      input_ports: ['input'],
      output_ports: ['output'],
      icon: 'math',
      default_params: { gain: 1 },
    };

    useGraphStore.getState().addNodeFromComponent(templateComponent, { x: 200, y: 160 });
    useGraphStore.getState().addNodeFromComponent(gain, { x: 900, y: 160 });

    const state = useGraphStore.getState();
    expect(state.graph.nodes.gain).toMatchObject({
      type: 'Gain',
      input_ports: ['input'],
      output_ports: ['output'],
    });
    expect(state.uiState.node_states.gain.position).toEqual({ x: 900, y: 160 });
    expect(state.nodes.some((node) => node.id === 'gain')).toBe(true);
  });
});
