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
const COMPOSITE_TYPES = new Set(['SimpleStagedNetwork']);

interface GraphLayer {
  graph: GraphSpec;
  uiState: GraphUIState;
  graphId: string | null;
  label: string;
  key: string;
}

interface SubgraphEntry {
  graph: GraphSpec;
  uiState: GraphUIState;
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
      network: {
        type: 'SimpleStagedNetwork',
        params: {
          hidden_size: 100,
          input_size: 6,
          output_size: 2,
          hidden_type: 'GRUCell',
          out_nonlinearity: 'tanh',
        },
        input_ports: ['target', 'feedback'],
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
        type: 'FeedbackChannel',
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
    input_ports: ['target'],
    output_ports: ['effector'],
    input_bindings: {
      target: ['network', 'target'],
    },
    output_bindings: {
      effector: ['mechanics', 'effector'],
    },
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
      network: { position: { x: 300, y: 200 }, collapsed: false, selected: false },
      mechanics: { position: { x: 600, y: 200 }, collapsed: false, selected: false },
      feedback: { position: { x: 450, y: 400 }, collapsed: false, selected: false },
    },
  };

  const uiState: GraphUIState = {
    ...baseUiState,
    edge_states: buildEdgeStates(graph, baseUiState, DEFAULT_EDGE_STYLE),
  };

  return { graph, uiState };
}

function createNetworkSubgraph(label: string): { graph: GraphSpec; uiState: GraphUIState } {
  const graph: GraphSpec = {
    nodes: {
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
    input_ports: ['input'],
    output_ports: ['output'],
    input_bindings: {
      input: ['encoder', 'input'],
    },
    output_bindings: {
      output: ['decoder', 'output'],
    },
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
      encoder: { position: { x: 200, y: 220 }, collapsed: false, selected: false },
      core: { position: { x: 480, y: 220 }, collapsed: false, selected: false },
      decoder: { position: { x: 760, y: 220 }, collapsed: false, selected: false },
    },
  };

  const uiState: GraphUIState = {
    ...baseUiState,
    edge_states: buildEdgeStates(graph, baseUiState, DEFAULT_EDGE_STYLE),
  };

  return { graph, uiState };
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

  return {
    viewport: base.viewport ?? DEFAULT_VIEWPORT,
    node_states,
    edge_states,
  };
}

function buildNodes(graph: GraphSpec, uiState: GraphUIState): Node<GraphNodeData>[] {
  return Object.entries(graph.nodes).map(([id, spec]) => {
    const ui = uiState.node_states[id] ?? {
      position: DEFAULT_POSITION,
      collapsed: false,
      selected: false,
    };
    return {
      id,
      type: 'component',
      position: ui.position,
      data: {
        label: id,
        spec,
        collapsed: ui.collapsed,
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
  currentGraphKey: string;
  subgraphs: Record<string, SubgraphEntry>;
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
  currentGraphKey: '',
  subgraphs: {},
  isDirty: false,
  lastSavedAt: null,
  past: [],
  future: [],
  hydrateGraph: (graph, uiState, graphId) => {
    const edgeStyle = get().edgeStyle;
    const normalized = normalizeUiState(graph, uiState, edgeStyle);
    set({
      graphId: graphId ?? null,
      graph,
      uiState: normalized,
      nodes: buildNodes(graph, normalized),
      edges: buildEdges(graph, normalized, edgeStyle),
      graphStack: [],
      currentGraphLabel: graph.metadata?.name ?? 'Model',
      currentGraphKey: '',
      subgraphs: {},
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
      currentGraphKey: '',
      subgraphs: {},
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
      const existing = edge_states[edgeId] ?? {
        routing: { style: state.edgeStyle, points: [] },
      };
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextStyle = existing.routing.style === 'bezier' ? 'elbow' : 'bezier';
      const nextEdgeStates = {
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
      if (!nodeSpec || !COMPOSITE_TYPES.has(nodeSpec.type)) {
        return state;
      }
      const parentLabel = state.currentGraphLabel || state.graph.metadata?.name || 'Model';
      const parentKey = state.currentGraphKey;
      const nextKey = parentKey ? `${parentKey}/${nodeId}` : nodeId;
      const cached = state.subgraphs[nextKey];
      const nextLayer = cached ?? createNetworkSubgraph(nodeId);
      const normalized = normalizeUiState(nextLayer.graph, nextLayer.uiState, state.edgeStyle);
      return {
        graphStack: [
          ...state.graphStack,
          {
            graph: state.graph,
            uiState: state.uiState,
            graphId: state.graphId,
            label: parentLabel,
            key: parentKey,
          },
        ],
        graph: nextLayer.graph,
        uiState: normalized,
        nodes: buildNodes(nextLayer.graph, normalized),
        edges: buildEdges(nextLayer.graph, normalized, state.edgeStyle),
        currentGraphLabel: nodeId,
        currentGraphKey: nextKey,
        past: [],
        future: [],
      };
    });
  },
  exitToBreadcrumb: (index) => {
    set((state) => {
      if (state.graphStack.length === 0) return state;
      if (index >= state.graphStack.length) return state;
      const currentKey = state.currentGraphKey;
      const subgraphs = {
        ...state.subgraphs,
        ...(currentKey
          ? {
              [currentKey]: {
                graph: state.graph,
                uiState: state.uiState,
              },
            }
          : {}),
      };
      const nextStack = state.graphStack.slice(0, index);
      const nextLayer = state.graphStack[index];
      const normalized = normalizeUiState(nextLayer.graph, nextLayer.uiState, state.edgeStyle);
      return {
        graphStack: nextStack,
        graph: nextLayer.graph,
        uiState: normalized,
        nodes: buildNodes(nextLayer.graph, normalized),
        edges: buildEdges(nextLayer.graph, normalized, state.edgeStyle),
        graphId: nextLayer.graphId,
        currentGraphLabel: nextLayer.label,
        currentGraphKey: nextLayer.key,
        subgraphs,
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
      for (const nodeId of selectedNodeIds) {
        delete graphNodes[nodeId];
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
      const uiState = {
        ...state.uiState,
        node_states: Object.fromEntries(
          Object.entries(state.uiState.node_states).filter(([nodeId]) => !selectedNodeIds.includes(nodeId))
        ),
      };
      const nextGraph: GraphSpec = {
        ...state.graph,
        nodes: graphNodes,
        wires: edgesToWires(edges),
        input_bindings,
        output_bindings,
      };
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
      );
      const output_bindings = Object.fromEntries(
        Object.entries(state.graph.output_bindings).map(([key, value]) => [
          key,
          [value[0] === nodeId ? trimmed : value[0], value[1]],
        ])
      );

      const node_states = { ...state.uiState.node_states };
      const nodeState = node_states[nodeId];
      if (nodeState) {
        delete node_states[nodeId];
        node_states[trimmed] = { ...nodeState, selected: true };
      }

      const graph: GraphSpec = {
        ...state.graph,
        nodes: graphNodes,
        wires,
        input_bindings,
        output_bindings,
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
      const shouldRecord = changes.some(
        (change) =>
          change.type === 'remove' ||
          change.type === 'add' ||
          (change.type === 'position' && (change as { dragging?: boolean }).dragging === false)
      );
      const past = shouldRecord
        ? [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY)
        : state.past;
      const nextNodes = applyNodeChanges(changes, state.nodes);
      const node_states = { ...state.uiState.node_states };
      for (const node of nextNodes) {
        const existing = node_states[node.id] ?? {
          position: { x: node.position.x, y: node.position.y },
          collapsed: false,
          selected: false,
        };
        node_states[node.id] = {
          ...existing,
          position: { x: node.position.x, y: node.position.y },
          selected: !!node.selected,
        };
      }
      const dirty = changes.some((change) => change.type !== 'select');
      return {
        nodes: nextNodes,
        uiState: {
          ...state.uiState,
          node_states,
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
      const nextEdges = applyEdgeChanges(changes, state.edges);
      const graph: GraphSpec = {
        ...state.graph,
        wires: edgesToWires(nextEdges),
      };
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
      const graph: GraphSpec = {
        ...state.graph,
        wires: edgesToWires(nextEdges),
      };
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
      const graph: GraphSpec = {
        ...state.graph,
        nodes: {
          ...state.graph.nodes,
          [name]: spec,
        },
      };
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
