import { create } from 'zustand';
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from '@xyflow/react';
import type {
  GraphSpec,
  GraphUIState,
  GraphNodeData,
  GraphEdgeData,
  ComponentSpec,
  EdgeUIState,
  EdgeRouting,
} from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';

const DEFAULT_VIEWPORT = { x: 0, y: 0, zoom: 1 };
const DEFAULT_POSITION = { x: 200, y: 200 };
const MAX_HISTORY = 50;
const DEFAULT_EDGE_STYLE: EdgeRouting['style'] = 'bezier';
const COMPOSITE_TYPES = new Set(['Network', 'Subgraph']);

interface GraphLayer {
  graph: GraphSpec;
  uiState: GraphUIState;
  graphId: string | null;
  label: string;
  childNodeId?: string;
}

function wireId(wire: {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
}) {
  return `${wire.source_node}:${wire.source_port}->${wire.target_node}:${wire.target_port}`;
}

function buildEdgeStates(
  graph: GraphSpec,
  uiState: GraphUIState,
  defaultStyle: EdgeRouting['style']
): Record<string, EdgeUIState> {
  const existing = uiState.edge_states ?? {};
  const next: Record<string, EdgeUIState> = {};
  for (const wire of graph.wires) {
    const id = wireId(wire);
    next[id] = existing[id] ?? {
      routing: { style: defaultStyle, points: [] },
    };
  }
  return next;
}

function applyEdgeStates(
  edges: Edge<GraphEdgeData>[],
  edgeStates: Record<string, EdgeUIState>,
  defaultStyle: EdgeRouting['style']
) {
  return edges.map((edge) => {
    const routing =
      edgeStates[edge.id]?.routing ?? { style: defaultStyle, points: [] };
    return {
      ...edge,
      type: 'routed',
      data: {
        ...edge.data,
        routing,
      },
    };
  });
}

function createInitialGraph(): { graph: GraphSpec; uiState: GraphUIState } {
  const graph: GraphSpec = {
    nodes: {
      task: {
        type: 'SimpleReaches',
        params: {
          n_steps: 200,
          workspace: [
            [-1.0, -1.0],
            [1.0, 1.0],
          ],
          eval_n_directions: 7,
          eval_reach_length: 0.5,
          eval_grid_n: 1,
        },
        input_ports: [],
        output_ports: ['inputs', 'targets', 'inits', 'intervene'],
      },
      network: {
        type: 'Network',
        params: {
          hidden_size: 100,
          input_size: 6,
          out_size: 2,
          hidden_type: 'GRUCell',
          out_nonlinearity: 'tanh',
        },
        input_ports: ['input', 'feedback'],
        output_ports: ['output', 'hidden'],
      },
      mechanics: {
        type: 'Mechanics',
        params: {
          plant_type: 'TwoLinkArm',
          dt: 0.01,
        },
        input_ports: ['force'],
        output_ports: ['effector', 'state'],
      },
      feedback: {
        type: 'Channel',
        params: {
          delay: 5,
          noise_std: 0.01,
        },
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [
      {
        source_node: 'task',
        source_port: 'inputs',
        target_node: 'network',
        target_port: 'input',
      },
      {
        source_node: 'feedback',
        source_port: 'output',
        target_node: 'network',
        target_port: 'feedback',
      },
      {
        source_node: 'network',
        source_port: 'output',
        target_node: 'mechanics',
        target_port: 'force',
      },
      {
        source_node: 'mechanics',
        source_port: 'effector',
        target_node: 'feedback',
        target_port: 'input',
      },
    ],
    input_ports: [],
    output_ports: ['effector'],
    input_bindings: {},
    output_bindings: {
      effector: ['mechanics', 'effector'],
    },
    subgraphs: {},
    metadata: {
      name: 'Reaching Task Model',
      description: 'Two-link arm reaching to targets',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      version: '1.0.0',
    },
  };

  const baseUiState: GraphUIState = {
    viewport: DEFAULT_VIEWPORT,
    node_states: {
      task: { position: { x: 120, y: 200 }, collapsed: false, selected: false },
      network: { position: { x: 380, y: 200 }, collapsed: false, selected: false },
      mechanics: { position: { x: 660, y: 200 }, collapsed: false, selected: false },
      feedback: { position: { x: 520, y: 400 }, collapsed: false, selected: false },
    },
  };

  const uiState: GraphUIState = {
    ...baseUiState,
    edge_states: buildEdgeStates(graph, baseUiState, DEFAULT_EDGE_STYLE),
  };

  return { graph, uiState };
}

function migrateGraphSpec(graph: GraphSpec): GraphSpec {
  const renamePort = (
    nodeId: string,
    port: string,
    spec: ComponentSpec
  ) => {
    if (spec.type === 'Network' && port === 'target') {
      return 'input';
    }
    return port;
  };

  const nodes = Object.fromEntries(
    Object.entries(graph.nodes).map(([id, spec]) => {
      let nextType = spec.type;
      if (nextType === 'SimpleStagedNetwork') nextType = 'Network';
      if (nextType === 'FeedbackChannel') nextType = 'Channel';
      const nextParams = { ...spec.params };
      if (nextType === 'Network') {
        if ('output_size' in nextParams && !('out_size' in nextParams)) {
          nextParams.out_size = nextParams.output_size;
        }
      }
      const nextSpec: ComponentSpec = {
        ...spec,
        type: nextType,
        params: nextParams,
      };
      if (nextType === 'Network' && spec.input_ports.includes('target')) {
        nextSpec.input_ports = spec.input_ports.map((port) =>
          port === 'target' ? 'input' : port
        );
      }
      return [id, nextSpec];
    })
  );
  const wires = graph.wires.map((wire) => {
    const sourceSpec = nodes[wire.source_node];
    const targetSpec = nodes[wire.target_node];
    return {
      ...wire,
      source_port: sourceSpec ? renamePort(wire.source_node, wire.source_port, sourceSpec) : wire.source_port,
      target_port: targetSpec ? renamePort(wire.target_node, wire.target_port, targetSpec) : wire.target_port,
    };
  });
  const input_bindings = Object.fromEntries(
    Object.entries(graph.input_bindings).map(([name, binding]) => {
      const [nodeId, port] = binding;
      const spec = nodes[nodeId];
      const nextPort = spec ? renamePort(nodeId, port, spec) : port;
      return [name === 'target' ? 'input' : name, [nodeId, nextPort] as [string, string]];
    })
  );
  const input_ports = graph.input_ports.map((port) => (port === 'target' ? 'input' : port));
  const subgraphs = graph.subgraphs
    ? Object.fromEntries(
        Object.entries(graph.subgraphs).map(([id, subgraph]) => [id, migrateGraphSpec(subgraph)])
      )
    : undefined;
  return {
    ...graph,
    nodes,
    wires,
    input_ports,
    input_bindings,
    subgraphs,
  };
}

function createNetworkSubgraph(label: string): { graph: GraphSpec; uiState: GraphUIState } {
  const graph: GraphSpec = {
    nodes: {
      merge: {
        type: 'Sum',
        params: {},
        input_ports: ['a', 'b'],
        output_ports: ['output'],
      },
      encoder: {
        type: 'MLP',
        params: {
          input_size: 6,
          output_size: 64,
          hidden_sizes: [64],
          activation: 'relu',
        },
        input_ports: ['input'],
        output_ports: ['output'],
      },
      core: {
        type: 'GRU',
        params: {
          input_size: 64,
          hidden_size: 64,
          output_size: 64,
          num_layers: 1,
        },
        input_ports: ['input'],
        output_ports: ['output', 'hidden'],
      },
      decoder: {
        type: 'MLP',
        params: {
          input_size: 64,
          output_size: 2,
          hidden_sizes: [64],
          activation: 'tanh',
        },
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [
      {
        source_node: 'merge',
        source_port: 'output',
        target_node: 'encoder',
        target_port: 'input',
      },
      {
        source_node: 'encoder',
        source_port: 'output',
        target_node: 'core',
        target_port: 'input',
      },
      {
        source_node: 'core',
        source_port: 'output',
        target_node: 'decoder',
        target_port: 'input',
      },
    ],
    input_ports: ['input', 'feedback'],
    output_ports: ['output', 'hidden'],
    input_bindings: {
      input: ['merge', 'a'],
      feedback: ['merge', 'b'],
    },
    output_bindings: {
      output: ['decoder', 'output'],
      hidden: ['core', 'hidden'],
    },
    subgraphs: {},
    metadata: {
      name: `${label} Graph`,
      description: 'Internal staged network graph.',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      version: '1.0.0',
    },
  };

  const baseUiState: GraphUIState = {
    viewport: DEFAULT_VIEWPORT,
    node_states: {
      merge: { position: { x: 140, y: 220 }, collapsed: false, selected: false },
      encoder: { position: { x: 360, y: 220 }, collapsed: false, selected: false },
      core: { position: { x: 620, y: 220 }, collapsed: false, selected: false },
      decoder: { position: { x: 880, y: 220 }, collapsed: false, selected: false },
    },
  };

  const uiState: GraphUIState = {
    ...baseUiState,
    edge_states: buildEdgeStates(graph, baseUiState, DEFAULT_EDGE_STYLE),
  };

  return { graph, uiState };
}

function createEmptySubgraph(label: string): { graph: GraphSpec; uiState: GraphUIState } {
  const now = new Date().toISOString();
  const graph: GraphSpec = {
    nodes: {},
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
    subgraphs: {},
    metadata: {
      name: `${label} Graph`,
      description: 'Empty subgraph workspace.',
      created_at: now,
      updated_at: now,
      version: '1.0.0',
    },
  };
  const baseUiState: GraphUIState = {
    viewport: DEFAULT_VIEWPORT,
    node_states: {},
  };
  const uiState: GraphUIState = {
    ...baseUiState,
    edge_states: buildEdgeStates(graph, baseUiState, DEFAULT_EDGE_STYLE),
  };
  return { graph, uiState };
}

function deriveSubgraphPorts(graph: GraphSpec): GraphSpec {
  const wiredInputs = new Set(
    graph.wires.map((wire) => `${wire.target_node}:${wire.target_port}`)
  );
  const wiredOutputs = new Set(
    graph.wires.map((wire) => `${wire.source_node}:${wire.source_port}`)
  );

  const inputBindings: Record<string, [string, string]> = {};
  const outputBindings: Record<string, [string, string]> = {};

  const addUnique = (
    name: string,
    used: Set<string>,
    nodeId: string,
    port: string
  ) => {
    let candidate = name;
    if (used.has(candidate)) {
      candidate = `${nodeId}.${name}`;
    }
    let idx = 2;
    while (used.has(candidate)) {
      candidate = `${nodeId}.${name}.${idx}`;
      idx += 1;
    }
    used.add(candidate);
    return candidate;
  };

  const usedInputs = new Set<string>();
  for (const [name, binding] of Object.entries(graph.input_bindings ?? {})) {
    const key = `${binding[0]}:${binding[1]}`;
    if (wiredInputs.has(key)) continue;
    inputBindings[name] = binding;
    usedInputs.add(name);
  }

  for (const [nodeId, nodeSpec] of Object.entries(graph.nodes)) {
    for (const port of nodeSpec.input_ports) {
      const key = `${nodeId}:${port}`;
      if (wiredInputs.has(key)) continue;
      if (Object.values(inputBindings).some(([n, p]) => n === nodeId && p === port)) {
        continue;
      }
      const name = addUnique(port, usedInputs, nodeId, port);
      inputBindings[name] = [nodeId, port];
    }
  }

  const usedOutputs = new Set<string>();
  for (const [name, binding] of Object.entries(graph.output_bindings ?? {})) {
    const key = `${binding[0]}:${binding[1]}`;
    if (wiredOutputs.has(key)) continue;
    outputBindings[name] = binding;
    usedOutputs.add(name);
  }

  for (const [nodeId, nodeSpec] of Object.entries(graph.nodes)) {
    for (const port of nodeSpec.output_ports) {
      const key = `${nodeId}:${port}`;
      if (wiredOutputs.has(key)) continue;
      if (Object.values(outputBindings).some(([n, p]) => n === nodeId && p === port)) {
        continue;
      }
      const name = addUnique(port, usedOutputs, nodeId, port);
      outputBindings[name] = [nodeId, port];
    }
  }

  return {
    ...graph,
    input_ports: Object.keys(inputBindings),
    output_ports: Object.keys(outputBindings),
    input_bindings: inputBindings,
    output_bindings: outputBindings,
  };
}

function arraysEqual<T>(left: T[], right: T[]) {
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i += 1) {
    if (left[i] !== right[i]) return false;
  }
  return true;
}

function isWrapperGraph(
  parent: GraphSpec,
  child: GraphSpec,
  childNodeId?: string
) {
  if (!childNodeId) return false;
  if (parent.wires.length > 0) return false;
  const nodeIds = Object.keys(parent.nodes);
  if (nodeIds.length !== 1 || nodeIds[0] !== childNodeId) return false;
  const node = parent.nodes[childNodeId];
  if (!node || node.type !== 'Subgraph') return false;
  if (!parent.subgraphs || parent.subgraphs[childNodeId] !== child) return false;
  if (!arraysEqual(parent.input_ports, node.input_ports)) return false;
  if (!arraysEqual(parent.output_ports, node.output_ports)) return false;
  if (!arraysEqual(parent.input_ports, child.input_ports)) return false;
  if (!arraysEqual(parent.output_ports, child.output_ports)) return false;
  for (const port of parent.input_ports) {
    const binding = parent.input_bindings[port];
    if (!binding || binding[0] !== childNodeId || binding[1] !== port) return false;
  }
  for (const port of parent.output_ports) {
    const binding = parent.output_bindings[port];
    if (!binding || binding[0] !== childNodeId || binding[1] !== port) return false;
  }
  return true;
}

function normalizeUiState(
  graph: GraphSpec,
  uiState: GraphUIState | null | undefined,
  defaultEdgeStyle: EdgeRouting['style']
): GraphUIState {
  const base: GraphUIState = uiState ?? { viewport: DEFAULT_VIEWPORT, node_states: {} };
  const node_states = { ...base.node_states };

  let offset = 0;
  for (const nodeId of Object.keys(graph.nodes)) {
    if (!node_states[nodeId]) {
      node_states[nodeId] = {
        position: { x: DEFAULT_POSITION.x + offset, y: DEFAULT_POSITION.y + offset },
        collapsed: false,
        selected: false,
      };
      offset += 60;
    }
  }

  const edge_states = buildEdgeStates(graph, base, defaultEdgeStyle);

  const subgraph_states: Record<string, GraphUIState> = {};
  if (graph.subgraphs) {
    for (const [nodeId, subgraph] of Object.entries(graph.subgraphs)) {
      const childState = base.subgraph_states?.[nodeId];
      subgraph_states[nodeId] = normalizeUiState(subgraph, childState, defaultEdgeStyle);
    }
  }

  return {
    viewport: base.viewport ?? DEFAULT_VIEWPORT,
    node_states,
    edge_states,
    subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
  };
}

function buildNodes(graph: GraphSpec, uiState: GraphUIState): Node<GraphNodeData>[] {
  return Object.entries(graph.nodes).map(([id, spec]) => {
    const ui = uiState.node_states[id] ?? {
      position: DEFAULT_POSITION,
      collapsed: false,
      selected: false,
    };
    const size = ui.size;
    return {
      id,
      type: 'component',
      position: ui.position,
      style: size ? { width: size.width, height: size.height } : undefined,
      data: {
        label: id,
        spec,
        collapsed: ui.collapsed,
        size,
      },
      selected: ui.selected,
    };
  });
}

function buildEdges(
  graph: GraphSpec,
  uiState: GraphUIState,
  defaultStyle: EdgeRouting['style']
): Edge<GraphEdgeData>[] {
  const edgeStates = buildEdgeStates(graph, uiState, defaultStyle);
  return graph.wires.map((wire) => {
    const id = wireId(wire);
    return {
      id,
      source: wire.source_node,
      target: wire.target_node,
      sourceHandle: wire.source_port,
      targetHandle: wire.target_port,
      type: 'routed',
      data: {
        routing: edgeStates[id]?.routing ?? { style: defaultStyle, points: [] },
      },
    };
  });
}

function edgesToWires(edges: Edge<GraphEdgeData>[]): GraphSpec['wires'] {
  return edges
    .filter((edge) => edge.source && edge.target && edge.sourceHandle && edge.targetHandle)
    .map((edge) => ({
      source_node: edge.source,
      source_port: edge.sourceHandle as string,
      target_node: edge.target,
      target_port: edge.targetHandle as string,
    }));
}

function updateNodeSpec(nodes: Node<GraphNodeData>[], nodeId: string, spec: ComponentSpec) {
  return nodes.map((node) =>
    node.id === nodeId
      ? {
          ...node,
          data: {
            ...node.data,
            spec,
          },
        }
      : node
  );
}

function createNodeName(graph: GraphSpec, base: string) {
  const sanitized = base.charAt(0).toLowerCase() + base.slice(1);
  if (!(sanitized in graph.nodes)) {
    return sanitized;
  }
  let index = 2;
  while (`${sanitized}${index}` in graph.nodes) {
    index += 1;
  }
  return `${sanitized}${index}`;
}

function cloneSnapshot(graph: GraphSpec, uiState: GraphUIState) {
  if (typeof structuredClone === 'function') {
    return structuredClone({ graph, uiState });
  }
  return JSON.parse(JSON.stringify({ graph, uiState })) as { graph: GraphSpec; uiState: GraphUIState };
}

interface GraphStoreState {
  graphId: string | null;
  graph: GraphSpec;
  uiState: GraphUIState;
  nodes: Node<GraphNodeData>[];
  edges: Edge<GraphEdgeData>[];
  edgeStyle: 'bezier' | 'elbow';
  graphStack: GraphLayer[];
  currentGraphLabel: string;
  isDirty: boolean;
  lastSavedAt: string | null;
  past: { graph: GraphSpec; uiState: GraphUIState }[];
  future: { graph: GraphSpec; uiState: GraphUIState }[];
  hydrateGraph: (graph: GraphSpec, uiState?: GraphUIState | null, graphId?: string | null) => void;
  markSaved: (graphId: string) => void;
  resetGraph: () => void;
  undo: () => void;
  redo: () => void;
  deleteSelected: () => void;
  setEdgeStyle: (style: 'bezier' | 'elbow') => void;
  addEdgePoint: (edgeId: string, point: { x: number; y: number }) => void;
  updateEdgePoint: (edgeId: string, index: number, point: { x: number; y: number }) => void;
  removeEdgePoint: (edgeId: string, index: number) => void;
  toggleEdgeStyleForEdge: (edgeId: string) => void;
  enterSubgraph: (nodeId: string) => void;
  wrapInParentGraph: () => void;
  exitToBreadcrumb: (index: number) => void;
  renameNode: (nodeId: string, nextId: string) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection, styleOverride?: 'bezier' | 'elbow') => void;
  addNodeFromComponent: (component: ComponentDefinition, position: { x: number; y: number }) => void;
  updateNodeParams: (nodeId: string, paramName: string, value: ComponentSpec['params'][string]) => void;
  setSelectedNode: (nodeId: string | null) => void;
  toggleNodeCollapse: (nodeId: string) => void;
}

const initial = createInitialGraph();

export const useGraphStore = create<GraphStoreState>((set, get) => ({
  graphId: null,
  graph: initial.graph,
  uiState: initial.uiState,
  nodes: buildNodes(initial.graph, initial.uiState),
  edges: buildEdges(initial.graph, initial.uiState, DEFAULT_EDGE_STYLE),
  edgeStyle: DEFAULT_EDGE_STYLE,
  graphStack: [],
  currentGraphLabel: initial.graph.metadata?.name ?? 'Model',
  isDirty: false,
  lastSavedAt: null,
  past: [],
  future: [],
  hydrateGraph: (graph, uiState, graphId) => {
    const edgeStyle = get().edgeStyle;
    const migrated = migrateGraphSpec(graph);
    const normalized = normalizeUiState(migrated, uiState, edgeStyle);
    set({
      graphId: graphId ?? null,
      graph: migrated,
      uiState: normalized,
      nodes: buildNodes(migrated, normalized),
      edges: buildEdges(migrated, normalized, edgeStyle),
      graphStack: [],
      currentGraphLabel: migrated.metadata?.name ?? 'Model',
      isDirty: false,
      past: [],
      future: [],
    });
  },
  markSaved: (graphId) => {
    set({
      graphId,
      isDirty: false,
      lastSavedAt: new Date().toISOString(),
    });
  },
  resetGraph: () => {
    const fresh = createInitialGraph();
    set({
      graphId: null,
      graph: fresh.graph,
      uiState: fresh.uiState,
      nodes: buildNodes(fresh.graph, fresh.uiState),
      edges: buildEdges(fresh.graph, fresh.uiState, DEFAULT_EDGE_STYLE),
      graphStack: [],
      currentGraphLabel: fresh.graph.metadata?.name ?? 'Model',
      isDirty: false,
      lastSavedAt: null,
      past: [],
      future: [],
    });
  },
  undo: () => {
    set((state) => {
      if (state.past.length === 0) return state;
      const previous = state.past[state.past.length - 1];
      const past = state.past.slice(0, -1);
      const future = [cloneSnapshot(state.graph, state.uiState), ...state.future];
      const normalized = normalizeUiState(previous.graph, previous.uiState, state.edgeStyle);
      return {
        ...state,
        graph: previous.graph,
        uiState: normalized,
        nodes: buildNodes(previous.graph, normalized),
        edges: buildEdges(previous.graph, normalized, state.edgeStyle),
        past,
        future,
        isDirty: true,
      };
    });
  },
  redo: () => {
    set((state) => {
      if (state.future.length === 0) return state;
      const next = state.future[0];
      const future = state.future.slice(1);
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const normalized = normalizeUiState(next.graph, next.uiState, state.edgeStyle);
      return {
        ...state,
        graph: next.graph,
        uiState: normalized,
        nodes: buildNodes(next.graph, normalized),
        edges: buildEdges(next.graph, normalized, state.edgeStyle),
        past,
        future,
        isDirty: true,
      };
    });
  },
  setEdgeStyle: (style) => {
    set((state) => {
      const edge_states = buildEdgeStates(state.graph, state.uiState, style);
      return {
        edgeStyle: style,
        uiState: {
          ...state.uiState,
          edge_states,
        },
        edges: applyEdgeStates(state.edges, edge_states, style),
      };
    });
  },
  addEdgePoint: (edgeId, point) => {
    set((state) => {
      const edge_states = buildEdgeStates(state.graph, state.uiState, state.edgeStyle);
      const existing = edge_states[edgeId];
      if (!existing) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const routing: EdgeRouting = {
        ...existing.routing,
        style: 'elbow',
        points: [...existing.routing.points, point],
      };
      const nextEdgeStates = {
        ...edge_states,
        [edgeId]: { routing },
      };
      return {
        uiState: {
          ...state.uiState,
          edge_states: nextEdgeStates,
        },
        edges: applyEdgeStates(state.edges, nextEdgeStates, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  updateEdgePoint: (edgeId, index, point) => {
    set((state) => {
      const edge_states = buildEdgeStates(state.graph, state.uiState, state.edgeStyle);
      const existing = edge_states[edgeId];
      if (!existing) return state;
      if (index < 0 || index >= existing.routing.points.length) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const points = existing.routing.points.map((pt, idx) => (idx === index ? point : pt));
      const nextEdgeStates = {
        ...edge_states,
        [edgeId]: {
          routing: {
            ...existing.routing,
            points,
          },
        },
      };
      return {
        uiState: {
          ...state.uiState,
          edge_states: nextEdgeStates,
        },
        edges: applyEdgeStates(state.edges, nextEdgeStates, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  removeEdgePoint: (edgeId, index) => {
    set((state) => {
      const edge_states = buildEdgeStates(state.graph, state.uiState, state.edgeStyle);
      const existing = edge_states[edgeId];
      if (!existing) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const points = existing.routing.points.filter((_, idx) => idx !== index);
      const nextEdgeStates = {
        ...edge_states,
        [edgeId]: {
          routing: {
            ...existing.routing,
            points,
          },
        },
      };
      return {
        uiState: {
          ...state.uiState,
          edge_states: nextEdgeStates,
        },
        edges: applyEdgeStates(state.edges, nextEdgeStates, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  toggleEdgeStyleForEdge: (edgeId) => {
    set((state) => {
      const edge_states = buildEdgeStates(state.graph, state.uiState, state.edgeStyle);
      const existing: EdgeUIState = edge_states[edgeId] ?? {
        routing: { style: state.edgeStyle, points: [] },
      };
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextStyle: EdgeRouting['style'] =
        existing.routing.style === 'bezier' ? 'elbow' : 'bezier';
      const nextEdgeStates: Record<string, EdgeUIState> = {
        ...edge_states,
        [edgeId]: {
          routing: {
            ...existing.routing,
            style: nextStyle,
          },
        },
      };
      return {
        uiState: {
          ...state.uiState,
          edge_states: nextEdgeStates,
        },
        edges: applyEdgeStates(state.edges, nextEdgeStates, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  enterSubgraph: (nodeId) => {
    set((state) => {
      const nodeSpec = state.graph.nodes[nodeId];
      const hasSubgraph = Boolean(state.graph.subgraphs?.[nodeId]);
      if (!nodeSpec || (!COMPOSITE_TYPES.has(nodeSpec.type) && !hasSubgraph)) {
        return state;
      }
      const parentLabel = state.currentGraphLabel || state.graph.metadata?.name || 'Model';
      const cachedGraph = state.graph.subgraphs?.[nodeId];
      const cachedUi = state.uiState.subgraph_states?.[nodeId];
      const nextLayerBase = cachedGraph
        ? { graph: cachedGraph, uiState: cachedUi ?? { viewport: DEFAULT_VIEWPORT, node_states: {} } }
        : nodeSpec.type === 'Network'
          ? createNetworkSubgraph(nodeId)
          : createEmptySubgraph(nodeId);
      const derivedNext = deriveSubgraphPorts(nextLayerBase.graph);
      const normalized = normalizeUiState(derivedNext, nextLayerBase.uiState, state.edgeStyle);
      const parentGraph: GraphSpec = {
        ...state.graph,
        subgraphs: {
          ...(state.graph.subgraphs ?? {}),
          [nodeId]: derivedNext,
        },
      };
      const parentUi: GraphUIState = {
        ...state.uiState,
        subgraph_states: {
          ...(state.uiState.subgraph_states ?? {}),
          [nodeId]: normalized,
        },
      };
      return {
        graphStack: [
          ...state.graphStack,
          {
            graph: parentGraph,
            uiState: parentUi,
            graphId: state.graphId,
            label: parentLabel,
            childNodeId: nodeId,
          },
        ],
        graph: derivedNext,
        uiState: normalized,
        nodes: buildNodes(derivedNext, normalized),
        edges: buildEdges(derivedNext, normalized, state.edgeStyle),
        currentGraphLabel: nodeId,
        past: [],
        future: [],
      };
    });
  },
  wrapInParentGraph: () => {
    set((state) => {
      const lastLayer = state.graphStack[state.graphStack.length - 1];
      if (lastLayer && isWrapperGraph(lastLayer.graph, state.graph, lastLayer.childNodeId)) {
        return state;
      }
      const derivedCurrent = deriveSubgraphPorts(state.graph);
      const normalizedCurrent = normalizeUiState(derivedCurrent, state.uiState, state.edgeStyle);
      const childNodeId = createNodeName({ nodes: {} } as GraphSpec, 'model');
      const now = new Date().toISOString();
      const parentGraph: GraphSpec = {
        nodes: {
          [childNodeId]: {
            type: 'Subgraph',
            params: {},
            input_ports: derivedCurrent.input_ports,
            output_ports: derivedCurrent.output_ports,
          },
        },
        wires: [],
        input_ports: derivedCurrent.input_ports,
        output_ports: derivedCurrent.output_ports,
        input_bindings: Object.fromEntries(
          derivedCurrent.input_ports.map((port) => [port, [childNodeId, port]])
        ) as Record<string, [string, string]>,
        output_bindings: Object.fromEntries(
          derivedCurrent.output_ports.map((port) => [port, [childNodeId, port]])
        ) as Record<string, [string, string]>,
        subgraphs: {
          [childNodeId]: derivedCurrent,
        },
        metadata: {
          name: state.graph.metadata?.name ?? 'Workspace',
          description: state.graph.metadata?.description ?? '',
          created_at: state.graph.metadata?.created_at ?? now,
          updated_at: now,
          version: state.graph.metadata?.version ?? '1.0.0',
        },
      };

      const parentUi: GraphUIState = {
        viewport: DEFAULT_VIEWPORT,
        node_states: {
          [childNodeId]: {
            position: { x: 320, y: 220 },
            collapsed: false,
            selected: false,
          },
        },
        subgraph_states: {
          [childNodeId]: normalizedCurrent,
        },
      };
      const normalizedParent = normalizeUiState(parentGraph, parentUi, state.edgeStyle);
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const updatedStack = [...state.graphStack];
      if (updatedStack.length > 0) {
        const parentLayer = updatedStack[updatedStack.length - 1];
        const parentChildId = parentLayer.childNodeId;
        if (parentChildId && parentLayer.graph.nodes[parentChildId]) {
          const nextGraph: GraphSpec = {
            ...parentLayer.graph,
            nodes: {
              ...parentLayer.graph.nodes,
              [parentChildId]: {
                ...parentLayer.graph.nodes[parentChildId],
                input_ports: parentGraph.input_ports,
                output_ports: parentGraph.output_ports,
              },
            },
            subgraphs: {
              ...(parentLayer.graph.subgraphs ?? {}),
              [parentChildId]: parentGraph,
            },
          };
          const nextUi: GraphUIState = {
            ...parentLayer.uiState,
            subgraph_states: {
              ...(parentLayer.uiState.subgraph_states ?? {}),
              [parentChildId]: normalizedParent,
            },
          };
          updatedStack[updatedStack.length - 1] = {
            ...parentLayer,
            graph: nextGraph,
            uiState: nextUi,
          };
        }
      }
      updatedStack.push({
        graph: parentGraph,
        uiState: normalizedParent,
        graphId: state.graphId,
        label: parentGraph.metadata?.name ?? 'Workspace',
        childNodeId,
      });

      return {
        graph: derivedCurrent,
        uiState: normalizedCurrent,
        nodes: buildNodes(derivedCurrent, normalizedCurrent),
        edges: buildEdges(derivedCurrent, normalizedCurrent, state.edgeStyle),
        graphStack: updatedStack,
        currentGraphLabel: state.currentGraphLabel,
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  exitToBreadcrumb: (index) => {
    set((state) => {
      if (state.graphStack.length === 0) return state;
      if (index >= state.graphStack.length) return state;
      const derivedCurrent = deriveSubgraphPorts(state.graph);
      let childGraph = derivedCurrent;
      let childUi = normalizeUiState(derivedCurrent, state.uiState, state.edgeStyle);

      const stack = [...state.graphStack];
      for (let i = stack.length - 1; i >= index; i -= 1) {
        const layer = stack[i];
        const childId = layer.childNodeId;
        if (!childId || !layer.graph.nodes[childId]) continue;
        const nextGraph: GraphSpec = {
          ...layer.graph,
          nodes: {
            ...layer.graph.nodes,
            [childId]: {
              ...layer.graph.nodes[childId],
              input_ports: childGraph.input_ports,
              output_ports: childGraph.output_ports,
            },
          },
          subgraphs: {
            ...(layer.graph.subgraphs ?? {}),
            [childId]: childGraph,
          },
        };
        const nextUi: GraphUIState = {
          ...layer.uiState,
          subgraph_states: {
            ...(layer.uiState.subgraph_states ?? {}),
            [childId]: childUi,
          },
        };
        stack[i] = {
          ...layer,
          graph: nextGraph,
          uiState: nextUi,
        };
        childGraph = nextGraph;
        childUi = nextUi;
      }

      const nextStack = stack.slice(0, index);
      const nextLayer = stack[index];
      const normalized = normalizeUiState(nextLayer.graph, nextLayer.uiState, state.edgeStyle);
      return {
        graphStack: nextStack,
        graph: nextLayer.graph,
        uiState: normalized,
        nodes: buildNodes(nextLayer.graph, normalized),
        edges: buildEdges(nextLayer.graph, normalized, state.edgeStyle),
        graphId: nextLayer.graphId,
        currentGraphLabel: nextLayer.label,
        past: [],
        future: [],
      };
    });
  },
  deleteSelected: () => {
    set((state) => {
      const selectedNodeIds = state.nodes.filter((node) => node.selected).map((node) => node.id);
      const selectedEdgeIds = state.edges.filter((edge) => edge.selected).map((edge) => edge.id);
      if (selectedNodeIds.length === 0 && selectedEdgeIds.length === 0) {
        return state;
      }
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nodes = state.nodes.filter((node) => !selectedNodeIds.includes(node.id));
      const edges = state.edges.filter(
        (edge) =>
          !selectedEdgeIds.includes(edge.id) &&
          !selectedNodeIds.includes(edge.source) &&
          !selectedNodeIds.includes(edge.target)
      );
      const graphNodes = { ...state.graph.nodes };
      const subgraphs = { ...(state.graph.subgraphs ?? {}) };
      for (const nodeId of selectedNodeIds) {
        delete graphNodes[nodeId];
        delete subgraphs[nodeId];
      }
      const input_bindings = { ...state.graph.input_bindings };
      const output_bindings = { ...state.graph.output_bindings };
      for (const [key, binding] of Object.entries(input_bindings)) {
        if (selectedNodeIds.includes(binding[0])) {
          delete input_bindings[key];
        }
      }
      for (const [key, binding] of Object.entries(output_bindings)) {
        if (selectedNodeIds.includes(binding[0])) {
          delete output_bindings[key];
        }
      }
      const subgraph_states = { ...(state.uiState.subgraph_states ?? {}) };
      for (const nodeId of selectedNodeIds) {
        delete subgraph_states[nodeId];
      }
      const uiState = {
        ...state.uiState,
        node_states: Object.fromEntries(
          Object.entries(state.uiState.node_states).filter(([nodeId]) => !selectedNodeIds.includes(nodeId))
        ),
        subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
      };
      let nextGraph: GraphSpec = {
        ...state.graph,
        nodes: graphNodes,
        wires: edgesToWires(edges),
        input_bindings,
        output_bindings,
        subgraphs: Object.keys(subgraphs).length ? subgraphs : undefined,
      };
      if (state.graphStack.length > 0) {
        nextGraph = deriveSubgraphPorts(nextGraph);
      }
      const edge_states = buildEdgeStates(nextGraph, uiState, state.edgeStyle);
      return {
        graph: nextGraph,
        uiState: {
          ...uiState,
          edge_states,
        },
        nodes,
        edges: applyEdgeStates(edges, edge_states, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  renameNode: (nodeId, nextId) => {
    set((state) => {
      const trimmed = nextId.trim();
      if (!trimmed || trimmed === nodeId || state.graph.nodes[trimmed]) {
        return state;
      }
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);

      const graphNodes = { ...state.graph.nodes };
      const nodeSpec = graphNodes[nodeId];
      if (!nodeSpec) return state;
      delete graphNodes[nodeId];
      graphNodes[trimmed] = nodeSpec;
      const subgraphs = { ...(state.graph.subgraphs ?? {}) };
      if (subgraphs[nodeId]) {
        subgraphs[trimmed] = subgraphs[nodeId];
        delete subgraphs[nodeId];
      }

      const wires = state.graph.wires.map((wire) => ({
        ...wire,
        source_node: wire.source_node === nodeId ? trimmed : wire.source_node,
        target_node: wire.target_node === nodeId ? trimmed : wire.target_node,
      }));

      const input_bindings = Object.fromEntries(
        Object.entries(state.graph.input_bindings).map(([key, value]) => [
          key,
          [value[0] === nodeId ? trimmed : value[0], value[1]],
        ])
      ) as Record<string, [string, string]>;
      const output_bindings = Object.fromEntries(
        Object.entries(state.graph.output_bindings).map(([key, value]) => [
          key,
          [value[0] === nodeId ? trimmed : value[0], value[1]],
        ])
      ) as Record<string, [string, string]>;

      const node_states = { ...state.uiState.node_states };
      const nodeState = node_states[nodeId];
      if (nodeState) {
        delete node_states[nodeId];
        node_states[trimmed] = { ...nodeState, selected: true };
      }
      const subgraph_states = { ...(state.uiState.subgraph_states ?? {}) };
      if (subgraph_states[nodeId]) {
        subgraph_states[trimmed] = subgraph_states[nodeId];
        delete subgraph_states[nodeId];
      }

      const graph: GraphSpec = {
        ...state.graph,
        nodes: graphNodes,
        wires,
        input_bindings,
        output_bindings,
        subgraphs: Object.keys(subgraphs).length ? subgraphs : undefined,
      };

      const previousEdgeStates = state.uiState.edge_states ?? {};
      const edge_states: Record<string, EdgeUIState> = {};
      for (const wire of wires) {
        const newId = wireId(wire);
        const oldId = wireId({
          source_node: wire.source_node === trimmed ? nodeId : wire.source_node,
          source_port: wire.source_port,
          target_node: wire.target_node === trimmed ? nodeId : wire.target_node,
          target_port: wire.target_port,
        });
        edge_states[newId] =
          previousEdgeStates[oldId] ??
          previousEdgeStates[newId] ?? { routing: { style: state.edgeStyle, points: [] } };
      }

      const uiState: GraphUIState = {
        ...state.uiState,
        node_states,
        edge_states,
        subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
      };

      return {
        graph,
        uiState,
        nodes: buildNodes(graph, uiState),
        edges: buildEdges(graph, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  onNodesChange: (changes) => {
    set((state) => {
      const removedIds = changes
        .filter((change) => change.type === 'remove')
        .map((change) => change.id);
      const shouldRecord = changes.some(
        (change) =>
          change.type === 'remove' ||
          change.type === 'add' ||
          change.type === 'dimensions' ||
          (change.type === 'position' && (change as { dragging?: boolean }).dragging === false)
      );
      const past = shouldRecord
        ? [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY)
        : state.past;
      let graph = state.graph;
      let uiState = state.uiState;
      let edges = state.edges;

      if (removedIds.length > 0) {
        const graphNodes = { ...graph.nodes };
        const subgraphs = { ...(graph.subgraphs ?? {}) };
        for (const nodeId of removedIds) {
          delete graphNodes[nodeId];
          delete subgraphs[nodeId];
        }
        const input_bindings = { ...graph.input_bindings };
        const output_bindings = { ...graph.output_bindings };
        for (const [key, binding] of Object.entries(input_bindings)) {
          if (removedIds.includes(binding[0])) {
            delete input_bindings[key];
          }
        }
        for (const [key, binding] of Object.entries(output_bindings)) {
          if (removedIds.includes(binding[0])) {
            delete output_bindings[key];
          }
        }
        edges = edges.filter(
          (edge) => !removedIds.includes(edge.source) && !removedIds.includes(edge.target)
        );
        const node_states = { ...uiState.node_states };
        for (const nodeId of removedIds) {
          delete node_states[nodeId];
        }
        const subgraph_states = { ...(uiState.subgraph_states ?? {}) };
        for (const nodeId of removedIds) {
          delete subgraph_states[nodeId];
        }
        uiState = {
          ...uiState,
          node_states,
          subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
        };
        graph = {
          ...graph,
          nodes: graphNodes,
          wires: edgesToWires(edges),
          input_bindings,
          output_bindings,
          subgraphs: Object.keys(subgraphs).length ? subgraphs : undefined,
        };
        if (state.graphStack.length > 0) {
          graph = deriveSubgraphPorts(graph);
        }
      }
      const sizeUpdates = new Map<string, { width: number; height: number }>();
      for (const change of changes) {
        if (change.type === 'dimensions') {
          const dims = (change as { dimensions?: { width: number; height: number } }).dimensions;
          if (dims) {
            sizeUpdates.set(change.id, dims);
          }
        }
      }
      const nextNodes = applyNodeChanges<Node<GraphNodeData>>(
        changes as NodeChange<Node<GraphNodeData>>[],
        state.nodes
      );
      const node_states = { ...uiState.node_states };
      for (const node of nextNodes) {
        const existing = node_states[node.id] ?? {
          position: { x: node.position.x, y: node.position.y },
          collapsed: false,
          selected: false,
        };
        const size =
          sizeUpdates.get(node.id) ??
          (node.width && node.height ? { width: node.width, height: node.height } : existing.size);
        node_states[node.id] = {
          ...existing,
          position: { x: node.position.x, y: node.position.y },
          selected: !!node.selected,
          size,
        };
      }
      const updatedNodes = nextNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          size: node_states[node.id]?.size ?? node.data.size,
        },
      }));
      const dirty = changes.some((change) => change.type !== 'select');
      const edge_states = buildEdgeStates(graph, uiState, state.edgeStyle);
      return {
        graph,
        nodes: updatedNodes,
        edges: applyEdgeStates(edges, edge_states, state.edgeStyle),
        uiState: {
          ...uiState,
          node_states,
          edge_states,
        },
        past,
        future: shouldRecord ? [] : state.future,
        isDirty: state.isDirty || dirty,
      };
    });
  },
  onEdgesChange: (changes) => {
    set((state) => {
      const shouldRecord = changes.some((change) => change.type !== 'select');
      const past = shouldRecord
        ? [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY)
        : state.past;
      const nextEdges = applyEdgeChanges<Edge<GraphEdgeData>>(
        changes as EdgeChange<Edge<GraphEdgeData>>[],
        state.edges
      );
      let graph: GraphSpec = {
        ...state.graph,
        wires: edgesToWires(nextEdges),
      };
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }
      const edge_states = buildEdgeStates(graph, state.uiState, state.edgeStyle);
      const dirty = changes.length > 0;
      return {
        edges: applyEdgeStates(nextEdges, edge_states, state.edgeStyle),
        graph,
        uiState: {
          ...state.uiState,
          edge_states,
        },
        past,
        future: shouldRecord ? [] : state.future,
        isDirty: state.isDirty || dirty,
      };
    });
  },
  onConnect: (connection, styleOverride) => {
    if (!connection.source || !connection.target) return;
    if (!connection.sourceHandle || !connection.targetHandle) return;
    const alreadyUsed = get().edges.some(
      (edge) =>
        edge.target === connection.target &&
        edge.targetHandle === connection.targetHandle
    );
    if (alreadyUsed) return;
    const edgeStyle = styleOverride ?? get().edgeStyle;
    const edge: Edge<GraphEdgeData> = {
      ...connection,
      id: wireId({
        source_node: connection.source,
        source_port: connection.sourceHandle,
        target_node: connection.target,
        target_port: connection.targetHandle,
      }),
      type: 'routed',
      data: {
        routing: {
          style: edgeStyle,
          points: [],
        },
      },
    };
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextEdges = addEdge(edge, state.edges);
      let graph: GraphSpec = {
        ...state.graph,
        wires: edgesToWires(nextEdges),
      };
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }
      const edge_states = buildEdgeStates(graph, state.uiState, state.edgeStyle);
      edge_states[edge.id] = {
        routing: {
          style: edgeStyle,
          points: [],
        },
      };
      return {
        edges: applyEdgeStates(nextEdges, edge_states, state.edgeStyle),
        graph,
        uiState: {
          ...state.uiState,
          edge_states,
        },
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  addNodeFromComponent: (component, position) => {
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const name = createNodeName(state.graph, component.name);
      const spec: ComponentSpec = {
        type: component.name,
        params: { ...component.default_params },
        input_ports: component.input_ports,
        output_ports: component.output_ports,
      };
      let graph: GraphSpec = {
        ...state.graph,
        nodes: {
          ...state.graph.nodes,
          [name]: spec,
        },
      };
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }
      const uiState: GraphUIState = {
        ...state.uiState,
        node_states: {
          ...state.uiState.node_states,
          [name]: {
            position,
            collapsed: false,
            selected: true,
          },
        },
      };
      const nodes = buildNodes(graph, uiState).map((node) => ({
        ...node,
        selected: node.id === name,
      }));
      return {
        graph,
        uiState,
        nodes,
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  updateNodeParams: (nodeId, paramName, value) => {
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nodeSpec = state.graph.nodes[nodeId];
      if (!nodeSpec) return state;
      const updatedSpec = {
        ...nodeSpec,
        params: {
          ...nodeSpec.params,
          [paramName]: value,
        },
      };
      return {
        graph: {
          ...state.graph,
          nodes: {
            ...state.graph.nodes,
            [nodeId]: updatedSpec,
          },
        },
        nodes: updateNodeSpec(state.nodes, nodeId, updatedSpec),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  setSelectedNode: (nodeId) => {
    set((state) => {
      const nodes = state.nodes.map((node) => ({
        ...node,
        selected: node.id === nodeId,
      }));
      const node_states = { ...state.uiState.node_states };
      for (const node of nodes) {
        const existing = node_states[node.id] ?? {
          position: node.position,
          collapsed: false,
          selected: false,
        };
        node_states[node.id] = {
          ...existing,
          selected: node.id === nodeId,
        };
      }
      return {
        nodes,
        uiState: {
          ...state.uiState,
          node_states,
        },
      };
    });
  },
  toggleNodeCollapse: (nodeId) => {
    set((state) => {
      const nodeState = state.uiState.node_states[nodeId];
      if (!nodeState) return state;
      const uiState: GraphUIState = {
        ...state.uiState,
        node_states: {
          ...state.uiState.node_states,
          [nodeId]: {
            ...nodeState,
            collapsed: !nodeState.collapsed,
          },
        },
      };
      const nodes = state.nodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                collapsed: !node.data.collapsed,
              },
            }
          : node
      );
      return {
        uiState,
        nodes,
        isDirty: true,
      };
    });
  },
}));
