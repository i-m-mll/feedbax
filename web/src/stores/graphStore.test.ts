import { beforeEach, describe, expect, it } from 'vitest';
import { useGraphStore } from './graphStore';
import {
  isAcausalGraphSpec,
  isCausalGraphSpec,
  type AcausalGraphSpec,
  type GraphSpec,
  type GraphUIState,
} from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';

const CAUSAL_DOMAIN_ID = 'feedbax.domain.causal';
const ACAUSAL_DOMAIN_ID = 'feedbax.domain.acausal';

const uiState: GraphUIState = {
  viewport: { x: 0, y: 0, zoom: 1 },
  node_states: {
    network: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
  },
};

function compositeComponent(
  name: string,
  interiorDomain = CAUSAL_DOMAIN_ID
): ComponentDefinition {
  return {
    name,
    category: 'Structure',
    description: `${name} composite`,
    param_schema: [],
    input_ports: ['input'],
    output_ports: ['output'],
    icon: 'Layers',
    default_params: {},
    domain: CAUSAL_DOMAIN_ID,
    interior_domain: interiorDomain,
    is_composite: true,
  };
}

function acausalComponent(name: string): ComponentDefinition {
  return {
    name,
    category: 'Mechanics',
    description: `${name} acausal component`,
    param_schema: [],
    input_ports: ['flange'],
    output_ports: [],
    icon: 'Circle',
    default_params: {},
    domain: ACAUSAL_DOMAIN_ID,
    port_types: {
      inputs: {
        flange: { kind: 'conserving', physical_domain: 'translational' },
        flange_2: { kind: 'conserving', physical_domain: 'translational' },
      },
    },
  };
}

function installCompositeRegistry() {
  useGraphStore.getState().setComponentRegistry([
    compositeComponent('Network'),
    compositeComponent('Subgraph'),
  ]);
}

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

function graphWithTwoNodes(): { graph: GraphSpec; uiState: GraphUIState } {
  return {
    graph: {
      nodes: {
        a: {
          type: 'Gain',
          params: { gain: 1 },
          input_ports: ['input'],
          output_ports: ['output'],
        },
        b: {
          type: 'Gain',
          params: { gain: 1 },
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
    uiState: {
      viewport: { x: 0, y: 0, zoom: 1 },
      node_states: {
        a: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
        b: { position: { x: 240, y: 0 }, collapsed: false, selected: false },
      },
    },
  };
}

function graphWithThreeLevelSubgraph(): { graph: GraphSpec; uiState: GraphUIState } {
  const innerGraph: GraphSpec = {
    nodes: {
      core: {
        type: 'Linear',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [],
    input_ports: ['input'],
    output_ports: ['output'],
    input_bindings: { input: ['core', 'input'] },
    output_bindings: { output: ['core', 'output'] },
  };
  const blockGraph: GraphSpec = {
    nodes: {
      inner: {
        type: 'Subgraph',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [],
    input_ports: ['input'],
    output_ports: ['output'],
    input_bindings: { input: ['inner', 'input'] },
    output_bindings: { output: ['inner', 'output'] },
    subgraphs: {
      inner: innerGraph,
    },
  };
  const networkGraph: GraphSpec = {
    nodes: {
      block: {
        type: 'Subgraph',
        params: {},
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [],
    input_ports: ['input'],
    output_ports: ['output'],
    input_bindings: { input: ['block', 'input'] },
    output_bindings: { output: ['block', 'output'] },
    subgraphs: {
      block: blockGraph,
    },
  };
  const graph: GraphSpec = {
    nodes: {
      network: {
        type: 'Network',
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
    subgraphs: {
      network: networkGraph,
    },
  };
  const uiState: GraphUIState = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {
      network: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
    },
    subgraph_states: {
      network: {
        viewport: { x: 10, y: 20, zoom: 0.9 },
        node_states: {
          block: { position: { x: 100, y: 50 }, collapsed: false, selected: false },
        },
        subgraph_states: {
          block: {
            viewport: { x: 30, y: 40, zoom: 0.8 },
            node_states: {
              inner: { position: { x: 200, y: 80 }, collapsed: false, selected: false },
            },
            subgraph_states: {
              inner: {
                viewport: { x: 50, y: 60, zoom: 0.7 },
                node_states: {
                  core: { position: { x: 300, y: 100 }, collapsed: false, selected: false },
                },
              },
            },
          },
        },
      },
    },
  };
  return { graph, uiState };
}

describe('graphStore boundary aliases', () => {
  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(graphWithNetworkSubgraph(), uiState);
    installCompositeRegistry();
  });

  it('renames a subgraph boundary input without renaming the internal bound port', () => {
    useGraphStore
      .getState()
      .renameSubgraphBoundaryPort('network', 'input', 'feedback', 'proprioception');

    const graph = useGraphStore.getState().graph;
    const network = graph.nodes.network;
    const subgraph = graph.subgraphs?.network;

    expect(network.input_ports).toEqual(['input', 'proprioception']);
    expect(isCausalGraphSpec(subgraph)).toBe(true);
    if (!isCausalGraphSpec(subgraph)) throw new Error('expected causal network subgraph');
    expect(subgraph?.input_ports).toEqual(['input', 'proprioception']);
    expect(subgraph?.input_bindings.proprioception).toEqual(['input_mux', 'in_1']);
    expect(subgraph?.input_bindings.feedback).toBeUndefined();
    expect(graph.wires[0]).toMatchObject({
      target_node: 'network',
      target_port: 'proprioception',
    });
  });

  it('deletes same-named child nodes without deleting parent graph nodes', () => {
    const graph = graphWithNetworkSubgraph();
    graph.nodes.input_mux = {
      type: 'Mux',
      params: { n_inputs: 2 },
      input_ports: ['in_0', 'in_1'],
      output_ports: ['output'],
    };
    useGraphStore.getState().hydrateGraph(graph, {
      ...uiState,
      node_states: {
        ...uiState.node_states,
        input_mux: { position: { x: 100, y: 0 }, collapsed: false, selected: false },
      },
    });

    useGraphStore.getState().enterSubgraph('network');
    useGraphStore.getState().setSelectedNode('input_mux');
    useGraphStore.getState().deleteSelected();

    expect(useGraphStore.getState().graph.nodes.input_mux).toBeUndefined();
    useGraphStore.getState().exitToBreadcrumb(0);
    const parentGraph = useGraphStore.getState().graph;

    expect(parentGraph.nodes.input_mux).toBeDefined();
    expect(parentGraph.subgraphs?.network.nodes.input_mux).toBeUndefined();
  });

  it('folds active subgraph edits into the root graph snapshot for persistence', () => {
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

    useGraphStore.getState().enterSubgraph('network');
    useGraphStore.getState().addNodeFromComponent(gain, { x: 480, y: 120 });

    const activeGraph = useGraphStore.getState().graph;
    expect(activeGraph.nodes.gain).toBeDefined();

    const persisted = useGraphStore.getState().capturePersistedGraph();

    expect(persisted.graph.nodes.network).toBeDefined();
    expect(persisted.graph.subgraphs?.network.nodes.gain).toMatchObject({
      type: 'Gain',
      params: { gain: 1 },
    });
    expect(persisted.uiState.subgraph_states?.network?.node_states.gain.position).toEqual({
      x: 480,
      y: 120,
    });
  });

  it('folds three active subgraph layers into the root graph snapshot', () => {
    const { graph, uiState } = graphWithThreeLevelSubgraph();
    const gain: ComponentDefinition = {
      name: 'Gain',
      category: 'Math',
      description: 'Gain',
      param_schema: [],
      input_ports: ['input'],
      output_ports: ['output'],
      icon: 'math',
      default_params: { gain: 3 },
    };

    useGraphStore.getState().hydrateGraph(graph, uiState);
    useGraphStore.getState().enterSubgraph('network');
    useGraphStore.getState().enterSubgraph('block');
    useGraphStore.getState().enterSubgraph('inner');
    useGraphStore.getState().addNodeFromComponent(gain, { x: 520, y: 160 });

    const persisted = useGraphStore.getState().capturePersistedGraph();

    expect(persisted.graphStackPath).toEqual(['network', 'block', 'inner']);
    expect(
      persisted.graph.subgraphs?.network.subgraphs?.block.subgraphs?.inner.nodes.gain
    ).toMatchObject({
      type: 'Gain',
      params: { gain: 3 },
    });
    expect(
      persisted.uiState.subgraph_states?.network.subgraph_states?.block.subgraph_states?.inner
        .node_states.gain.position
    ).toEqual({ x: 520, y: 160 });
  });

  it('restores the active subgraph path after a persisted graph reload', () => {
    const { graph, uiState } = graphWithThreeLevelSubgraph();
    const gain: ComponentDefinition = {
      name: 'Gain',
      category: 'Math',
      description: 'Gain',
      param_schema: [],
      input_ports: ['input'],
      output_ports: ['output'],
      icon: 'math',
      default_params: { gain: 5 },
    };

    useGraphStore.getState().hydrateGraph(graph, uiState, 'graph-1');
    useGraphStore.getState().enterSubgraph('network');
    useGraphStore.getState().enterSubgraph('block');
    useGraphStore.getState().enterSubgraph('inner');
    useGraphStore.getState().addNodeFromComponent(gain, { x: 640, y: 180 });

    const persisted = useGraphStore.getState().capturePersistedGraph();
    useGraphStore
      .getState()
      .hydrateGraph(persisted.graph, persisted.uiState, 'graph-1', persisted.graphStackPath);

    const state = useGraphStore.getState();
    expect(state.graphStack.map((layer) => layer.childNodeId)).toEqual([
      'network',
      'block',
      'inner',
    ]);
    expect(state.currentGraphLabel).toBe('inner');
    expect(state.currentContext).toBe(CAUSAL_DOMAIN_ID);
    expect(state.graph.nodes.gain).toMatchObject({
      type: 'Gain',
      params: { gain: 5 },
    });
    expect(state.uiState.node_states.gain.position).toEqual({ x: 640, y: 180 });
  });

  it('restores an acausal graph stack context after registry metadata loads', () => {
    const graph: GraphSpec = {
      nodes: {
        system: {
          type: 'AcausalSystem',
          params: {},
          input_ports: ['input'],
          output_ports: ['state'],
        },
      },
      wires: [],
      input_ports: [],
      output_ports: [],
      input_bindings: {},
      output_bindings: {},
      subgraphs: {
        system: {
          nodes: {},
          wires: [],
          input_ports: [],
          output_ports: [],
          input_bindings: {},
          output_bindings: {},
        },
      },
    };
    const acausalUiState: GraphUIState = {
      viewport: { x: 0, y: 0, zoom: 1 },
      node_states: {
        system: { position: { x: 120, y: 80 }, collapsed: false, selected: false },
      },
      subgraph_states: {
        system: { viewport: { x: 10, y: 20, zoom: 0.8 }, node_states: {} },
      },
    };

    useGraphStore.getState().hydrateGraph(graph, acausalUiState, 'graph-1', ['system']);
    expect(useGraphStore.getState().currentContext).toBe('top-level');

    useGraphStore.getState().setComponentRegistry([
      compositeComponent('AcausalSystem', ACAUSAL_DOMAIN_ID),
    ]);

    const state = useGraphStore.getState();
    expect(state.currentGraphLabel).toBe('system');
    expect(state.currentContext).toBe(ACAUSAL_DOMAIN_ID);
    expect(state.graphStack[0].contextType).toBe(ACAUSAL_DOMAIN_ID);
  });

  it('edits and persists acausal conserving connections with multi-edge ports', () => {
    const interior: AcausalGraphSpec = {
      schema_id: 'feedbax.spec.acausal_graph',
      schema_version: 'feedbax.spec.acausal_graph.v1',
      physical_domain: 'translational',
      solver: { solver_type: 'implicit_euler', dt: 0.01 },
      nodes: {
        mass: { type: 'Mass', params: {}, input_ports: ['flange'], output_ports: [] },
        spring: {
          type: 'Spring',
          params: {},
          input_ports: ['flange', 'flange_2'],
          output_ports: [],
        },
        ground: { type: 'Ground', params: {}, input_ports: ['flange'], output_ports: [] },
      },
      connections: [],
    };
    const graph: GraphSpec = {
      nodes: {
        system: {
          type: 'AcausalSystem',
          params: {},
          input_ports: ['input'],
          output_ports: ['state'],
        },
      },
      wires: [],
      input_ports: [],
      output_ports: [],
      input_bindings: {},
      output_bindings: {},
      subgraphs: { system: interior },
    };
    const ui: GraphUIState = {
      viewport: { x: 0, y: 0, zoom: 1 },
      node_states: {
        system: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
      },
      subgraph_states: {
        system: {
          viewport: { x: 0, y: 0, zoom: 1 },
          node_states: {
            mass: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
            spring: { position: { x: 200, y: 0 }, collapsed: false, selected: false },
            ground: { position: { x: 0, y: 140 }, collapsed: false, selected: false },
          },
        },
      },
    };

    useGraphStore.getState().setComponentRegistry([
      compositeComponent('AcausalSystem', ACAUSAL_DOMAIN_ID),
      acausalComponent('Mass'),
      acausalComponent('Spring'),
      acausalComponent('Ground'),
    ]);
    useGraphStore.getState().hydrateGraph(graph, ui, 'graph-1');
    useGraphStore.getState().enterSubgraph('system');
    useGraphStore.getState().onConnect({
      source: 'mass',
      sourceHandle: 'flange',
      target: 'spring',
      targetHandle: 'flange',
    });
    useGraphStore.getState().onConnect({
      source: 'ground',
      sourceHandle: 'flange',
      target: 'spring',
      targetHandle: 'flange',
    });

    const state = useGraphStore.getState();
    expect(isAcausalGraphSpec(state.graph)).toBe(true);
    expect(state.edges).toHaveLength(2);
    expect(state.edges.every((edge) => edge.type === 'conserving')).toBe(true);

    const persisted = state.capturePersistedGraph();
    const savedInterior = persisted.graph.subgraphs?.system;
    expect(isAcausalGraphSpec(savedInterior)).toBe(true);
    if (!isAcausalGraphSpec(savedInterior)) throw new Error('expected acausal interior');
    expect(savedInterior.connections).toHaveLength(2);
    expect(persisted.uiState.subgraph_states?.system?.node_states.spring.position).toEqual({
      x: 200,
      y: 0,
    });
  });

  it('fails loudly when a parent subgraph entry vanishes before persistence', () => {
    const { graph, uiState } = graphWithThreeLevelSubgraph();
    useGraphStore.getState().hydrateGraph(graph, uiState);
    useGraphStore.getState().enterSubgraph('network');
    useGraphStore.getState().enterSubgraph('block');

    useGraphStore.setState((state) => ({
      graphStack: state.graphStack.map((layer, index) =>
        index === 0
          ? {
              ...layer,
              graph: {
                ...layer.graph,
                nodes: {},
              },
            }
          : layer
      ),
    }));

    expect(() => useGraphStore.getState().capturePersistedGraph()).toThrow(
      'parent graph no longer contains subgraph node "network"'
    );
  });

  it('preserves active subgraph undo history across enter and exit', () => {
    const gain: ComponentDefinition = {
      name: 'Gain',
      category: 'Math',
      description: 'Gain',
      param_schema: [],
      input_ports: ['input'],
      output_ports: ['output'],
      icon: 'math',
      default_params: { gain: 7 },
    };

    useGraphStore.getState().enterSubgraph('network');
    useGraphStore.getState().addNodeFromComponent(gain, { x: 520, y: 160 });
    expect(useGraphStore.getState().graph.nodes.gain).toBeDefined();
    expect(useGraphStore.getState().past).toHaveLength(1);

    useGraphStore.getState().exitToBreadcrumb(0);
    expect(useGraphStore.getState().graph.nodes.network).toBeDefined();

    useGraphStore.getState().enterSubgraph('network');
    expect(useGraphStore.getState().past).toHaveLength(1);
    useGraphStore.getState().undo();

    expect(useGraphStore.getState().graph.nodes.gain).toBeUndefined();
  });

  it('duplicates a composite node with its internal graph and UI state', () => {
    useGraphStore.getState().setSelectedNode('network');
    useGraphStore.getState().duplicateSelected();

    const state = useGraphStore.getState();
    expect(state.past).toHaveLength(1);
    expect(state.graph.nodes.network2).toMatchObject({
      type: 'Network',
      input_ports: ['input', 'feedback'],
      output_ports: ['output'],
    });
    expect(state.graph.subgraphs?.network2).toEqual(state.graph.subgraphs?.network);
    expect(state.graph.subgraphs?.network2).not.toBe(state.graph.subgraphs?.network);
    expect(state.uiState.subgraph_states?.network2).toEqual(
      state.uiState.subgraph_states?.network
    );
    expect(state.uiState.subgraph_states?.network2).not.toBe(
      state.uiState.subgraph_states?.network
    );
    expect(state.uiState.node_states.network2.position).toEqual({ x: 40, y: 40 });
    expect(state.uiState.node_states.network2.selected).toBe(true);

    useGraphStore.getState().undo();
    expect(useGraphStore.getState().graph.nodes.network2).toBeUndefined();
  });

  it('raises when duplicating a composite node with no source subgraph', () => {
    const graph = graphWithNetworkSubgraph();
    delete graph.subgraphs;
    useGraphStore.getState().hydrateGraph(graph, uiState);
    installCompositeRegistry();
    useGraphStore.getState().setSelectedNode('network');

    expect(() => useGraphStore.getState().duplicateSelected()).toThrow(
      'Cannot duplicate composite node "network": source subgraph is missing.'
    );
  });
});

describe('graphStore React Flow identity preservation', () => {
  beforeEach(() => {
    const { graph, uiState } = graphWithTwoNodes();
    useGraphStore.getState().hydrateGraph(graph, uiState);
  });

  it('keeps untouched node and edge references stable for a single-node param edit', () => {
    const before = useGraphStore.getState();
    const previousA = before.nodes.find((node) => node.id === 'a');
    const previousB = before.nodes.find((node) => node.id === 'b');
    const previousEdge = before.edges.find((edge) => edge.source === 'a' && edge.target === 'b');

    useGraphStore.getState().updateNodeParams('a', 'gain', 2);

    const after = useGraphStore.getState();
    const nextA = after.nodes.find((node) => node.id === 'a');
    const nextB = after.nodes.find((node) => node.id === 'b');
    const nextEdge = after.edges.find((edge) => edge.source === 'a' && edge.target === 'b');

    expect(nextA).not.toBe(previousA);
    expect(nextB).toBe(previousB);
    expect(nextEdge).toBe(previousEdge);
    expect(after.graph.nodes.a.params.gain).toBe(2);
  });

  it('keeps unrelated graph entity references stable for selection-only changes', () => {
    const before = useGraphStore.getState();
    const previousA = before.nodes.find((node) => node.id === 'a');
    const previousB = before.nodes.find((node) => node.id === 'b');
    const previousEdges = before.edges;

    useGraphStore.getState().setSelectedNode('a');

    const nodeSelected = useGraphStore.getState();
    const selectedA = nodeSelected.nodes.find((node) => node.id === 'a');
    const selectedB = nodeSelected.nodes.find((node) => node.id === 'b');
    expect(selectedA).not.toBe(previousA);
    expect(selectedB).toBe(previousB);
    expect(nodeSelected.edges).toBe(previousEdges);

    const previousNodes = nodeSelected.nodes;
    const previousEdge = nodeSelected.edges[0];
    useGraphStore.getState().setSelectedEdge(previousEdge.id);

    const edgeSelected = useGraphStore.getState();
    expect(edgeSelected.nodes).toBe(previousNodes);
    expect(edgeSelected.edges[0]).not.toBe(previousEdge);
  });

  it('records undo snapshots for collapse and reverse UI mutations', () => {
    useGraphStore.getState().toggleNodeCollapse('a');
    expect(useGraphStore.getState().uiState.node_states.a.collapsed).toBe(true);
    expect(useGraphStore.getState().past).toHaveLength(1);
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().uiState.node_states.a.collapsed).toBe(false);

    useGraphStore.getState().toggleNodeReversed('a');
    expect(useGraphStore.getState().uiState.node_states.a.reversed).toBe(true);
    expect(useGraphStore.getState().past).toHaveLength(1);
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().uiState.node_states.a.reversed).toBe(false);

    useGraphStore.getState().setAllNodesCollapsed(true);
    expect(useGraphStore.getState().uiState.node_states.a.collapsed).toBe(true);
    expect(useGraphStore.getState().uiState.node_states.b.collapsed).toBe(true);
    expect(useGraphStore.getState().past).toHaveLength(1);
    useGraphStore.getState().undo();
    expect(useGraphStore.getState().uiState.node_states.a.collapsed).toBe(false);
    expect(useGraphStore.getState().uiState.node_states.b.collapsed).toBe(false);
  });
});

describe('graphStore subgraph entry templates', () => {
  const unpopulatedMuscleGraph: GraphSpec = {
    nodes: {
      muscle: {
        type: 'Arm6MuscleRigidTendon',
        params: {},
        input_ports: ['excitation', 'angles', 'angular_velocities'],
        output_ports: ['torques'],
      },
    },
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
  };
  const unpopulatedUi: GraphUIState = {
    viewport: { x: 0, y: 0, zoom: 1 },
    node_states: {
      muscle: { position: { x: 0, y: 0 }, collapsed: false, selected: false },
    },
  };

  beforeEach(() => {
    useGraphStore.getState().hydrateGraph(unpopulatedMuscleGraph, unpopulatedUi);
    useGraphStore.setState({
      _componentRegistry: new Map([
        [
          'Arm6MuscleRigidTendon',
          {
            name: 'Arm6MuscleRigidTendon',
            category: 'Mechanics',
            description: 'Composite metadata while templates load',
            param_schema: [],
            input_ports: ['input'],
            output_ports: ['output'],
            icon: 'activity',
            default_params: {},
            domain: CAUSAL_DOMAIN_ID,
            interior_domain: CAUSAL_DOMAIN_ID,
            is_composite: true,
          },
        ],
      ]),
      _isRegistryLoaded: false,
      lastSubgraphError: null,
    });
  });

  it('reports registry-loading failure without creating a layer or synthesized subgraph', () => {
    useGraphStore.getState().enterSubgraph('muscle');

    const state = useGraphStore.getState();
    const renderedNode = state.nodes.find((node) => node.id === 'muscle');
    expect(state.graphStack).toEqual([]);
    expect(state.graph.subgraphs?.muscle).toBeUndefined();
    expect(state.lastSubgraphError).toContain('component templates are still loading');
    expect(renderedNode?.type).toBe('component');
    expect(renderedNode?.data.subgraph).toBeUndefined();
    expect(state.capturePersistedGraph().graph.subgraphs?.muscle).toBeUndefined();
  });

  it('reports missing backend templates instead of fabricating frontend subgraphs', () => {
    useGraphStore.getState().setComponentRegistry([
      {
        name: 'Arm6MuscleRigidTendon',
        category: 'Mechanics',
        description: 'Composite without a loaded template',
        param_schema: [],
        input_ports: ['excitation', 'angles', 'angular_velocities'],
        output_ports: ['torques'],
        icon: 'activity',
        default_params: {},
        domain: CAUSAL_DOMAIN_ID,
        interior_domain: CAUSAL_DOMAIN_ID,
        is_composite: true,
      },
    ]);

    useGraphStore.getState().enterSubgraph('muscle');

    const state = useGraphStore.getState();
    expect(state.graphStack).toEqual([]);
    expect(state.graph.subgraphs?.muscle).toBeUndefined();
    expect(state.lastSubgraphError).toContain('no backend template_graph');
    expect(state.lastSubgraphError).toContain('cannot synthesize');
    expect(state.capturePersistedGraph().graph.subgraphs?.muscle).toBeUndefined();
  });

  it('uses registry template_graph as the only fresh subgraph source', () => {
    const templateGraph: GraphSpec = {
      nodes: {
        inner: {
          type: 'Gain',
          params: { gain: 1 },
          input_ports: ['input'],
          output_ports: ['output'],
        },
      },
      wires: [],
      input_ports: ['input'],
      output_ports: ['output'],
      input_bindings: { input: ['inner', 'input'] },
      output_bindings: { output: ['inner', 'output'] },
    };

    useGraphStore.getState().setComponentRegistry([
      {
        name: 'Arm6MuscleRigidTendon',
        category: 'Mechanics',
        description: 'Composite with a backend template',
        param_schema: [],
        input_ports: ['input'],
        output_ports: ['output'],
        icon: 'activity',
        default_params: {},
        domain: CAUSAL_DOMAIN_ID,
        interior_domain: CAUSAL_DOMAIN_ID,
        is_composite: true,
        template_graph: templateGraph,
        template_ui_state: {
          viewport: { x: 0, y: 0, zoom: 1 },
          node_states: {
            inner: { position: { x: 10, y: 20 }, collapsed: false, selected: false },
          },
        },
      },
    ]);

    useGraphStore.getState().enterSubgraph('muscle');

    const state = useGraphStore.getState();
    const persisted = state.capturePersistedGraph();
    expect(state.graph.nodes.inner.type).toBe('Gain');
    expect(state.graph.nodes.inner.params).toEqual({ gain: 1 });
    expect(state.lastSubgraphError).toBeNull();
    expect(persisted.graph.subgraphs?.muscle?.nodes.inner.type).toBe('Gain');
    expect(persisted.graph.subgraphs?.muscle?.nodes.activation_dynamics).toBeUndefined();
    expect(persisted.graph.nodes.muscle.params._subgraph).toBeUndefined();
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
    name: 'Recurrent Controller',
    category: 'Neural Networks',
    description: 'Template',
    param_schema: [],
    input_ports: ['input', 'feedback'],
    output_ports: ['output'],
    icon: 'network',
    default_params: {},
    template_id: 'feedbax.templates.recurrent_controller',
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
      retained_observables: [
        {
          id: 'obs:cell_output',
          label: 'Cell output',
          selector: 'port:cell.output',
          target: {
            kind: 'port',
            selector: 'port:cell.output',
            node_id: 'cell',
            port: 'output',
            timing: 'output',
          },
          retention: { mode: 'trajectory' },
          metadata: {},
        },
        {
          id: 'obs:cell_to_readout',
          label: 'Cell to readout',
          selector: 'edge:cell.output->readout.input',
          target: {
            kind: 'edge',
            selector: 'edge:cell.output->readout.input',
            edge_id: 'cell:output->readout:input',
            timing: 'step',
          },
          retention: { mode: 'trajectory' },
          metadata: {},
        },
        {
          id: 'obs:cell_hidden_carry',
          label: 'Cell hidden carry',
          selector: 'recurrent_carry:cell.hidden->cell.hidden',
          target: {
            kind: 'recurrent_carry',
            selector: 'recurrent_carry:cell.hidden->cell.hidden',
            edge_id: 'cell:hidden->cell:hidden',
            timing: 'step',
          },
          retention: { mode: 'trajectory' },
          metadata: {},
        },
        {
          id: 'obs:graph_output',
          label: 'Graph output',
          selector: 'graph_output:output',
          target: {
            kind: 'graph_output',
            selector: 'graph_output:output',
            node_id: 'readout',
            port: 'output',
            path: 'output',
            timing: 'step',
          },
          retention: { mode: 'trajectory' },
          metadata: {},
        },
        {
          id: 'obs:cell_state',
          label: 'Cell state',
          selector: 'path:states.cell.hidden',
          target: {
            kind: 'state_path',
            selector: 'path:states.cell.hidden',
            node_id: 'cell',
            path: 'states.cell.hidden',
            timing: 'step',
          },
          retention: { mode: 'trajectory' },
          metadata: {},
        },
      ],
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
    expect(state.graph.nodes['Recurrent Controller']).toBeUndefined();
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
    const cellNode = state.nodes.find((node) => node.id === 'cell');
    expect(cellNode?.type).toBe('subgraph');
    expect(cellNode?.data.subgraph).toBeDefined();
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

  it('remaps retained observable selectors when imported template nodes collide', () => {
    useGraphStore.getState().addNodeFromComponent(templateComponent, { x: 200, y: 160 });
    useGraphStore.getState().addNodeFromComponent(templateComponent, { x: 600, y: 160 });

    const observables = useGraphStore.getState().graph.retained_observables ?? [];
    const secondImport = observables.filter((observable) =>
      observable.id.startsWith('feedbax_templates_recurrent_controller:observable:')
    );
    expect(secondImport).toHaveLength(5);

    expect(secondImport.find((observable) => observable.label === 'Cell output')).toMatchObject({
      selector: 'port:feedbax_templates_recurrent_controller_cell.output',
      target: {
        selector: 'port:feedbax_templates_recurrent_controller_cell.output',
        node_id: 'feedbax_templates_recurrent_controller_cell',
      },
    });
    expect(secondImport.find((observable) => observable.label === 'Cell to readout')).toMatchObject({
      selector: 'edge:feedbax_templates_recurrent_controller_cell.output->feedbax_templates_recurrent_controller_readout.input',
      target: {
        selector: 'edge:feedbax_templates_recurrent_controller_cell.output->feedbax_templates_recurrent_controller_readout.input',
        edge_id: 'feedbax_templates_recurrent_controller_cell:output->feedbax_templates_recurrent_controller_readout:input',
      },
    });
    expect(secondImport.find((observable) => observable.label === 'Cell hidden carry')).toMatchObject({
      selector: 'recurrent_carry:feedbax_templates_recurrent_controller_cell.hidden->feedbax_templates_recurrent_controller_cell.hidden',
      target: {
        edge_id: 'feedbax_templates_recurrent_controller_cell:hidden->feedbax_templates_recurrent_controller_cell:hidden',
      },
    });
    expect(secondImport.find((observable) => observable.label === 'Graph output')).toMatchObject({
      selector: 'graph_output:output',
      target: {
        node_id: 'feedbax_templates_recurrent_controller_readout',
        port: 'output',
      },
    });
    expect(secondImport.find((observable) => observable.label === 'Cell state')).toMatchObject({
      selector: 'path:states.feedbax_templates_recurrent_controller_cell.hidden',
      target: {
        node_id: 'feedbax_templates_recurrent_controller_cell',
        path: 'states.feedbax_templates_recurrent_controller_cell.hidden',
      },
    });
  });
});
