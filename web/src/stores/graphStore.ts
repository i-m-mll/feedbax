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
import type { GraphSpec, GraphUIState, GraphNodeData, ComponentSpec } from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';

const DEFAULT_VIEWPORT = { x: 0, y: 0, zoom: 1 };
const DEFAULT_POSITION = { x: 200, y: 200 };
const MAX_HISTORY = 50;

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

  const uiState: GraphUIState = {
    viewport: DEFAULT_VIEWPORT,
    node_states: {
      network: { position: { x: 300, y: 200 }, collapsed: false, selected: false },
      mechanics: { position: { x: 600, y: 200 }, collapsed: false, selected: false },
      feedback: { position: { x: 450, y: 400 }, collapsed: false, selected: false },
    },
  };

  return { graph, uiState };
}

function wireId(wire: { source_node: string; source_port: string; target_node: string; target_port: string }) {
  return `${wire.source_node}:${wire.source_port}->${wire.target_node}:${wire.target_port}`;
}

function normalizeUiState(graph: GraphSpec, uiState?: GraphUIState | null): GraphUIState {
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

  return {
    viewport: base.viewport ?? DEFAULT_VIEWPORT,
    node_states,
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

function buildEdges(graph: GraphSpec, edgeStyle: 'bezier' | 'elbow' = 'bezier'): Edge[] {
  return graph.wires.map((wire) => ({
    id: wireId(wire),
    source: wire.source_node,
    target: wire.target_node,
    sourceHandle: wire.source_port,
    targetHandle: wire.target_port,
    type: edgeStyle === 'bezier' ? 'wire' : 'elbow',
  }));
}

function edgesToWires(edges: Edge[]): GraphSpec['wires'] {
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
  edges: Edge[];
  edgeStyle: 'bezier' | 'elbow';
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
  renameNode: (nodeId: string, nextId: string) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (connection: Connection) => void;
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
  edges: buildEdges(initial.graph, 'bezier'),
  edgeStyle: 'bezier',
  isDirty: false,
  lastSavedAt: null,
  past: [],
  future: [],
  hydrateGraph: (graph, uiState, graphId) => {
    const normalized = normalizeUiState(graph, uiState);
    const edgeStyle = get().edgeStyle;
    set({
      graphId: graphId ?? null,
      graph,
      uiState: normalized,
      nodes: buildNodes(graph, normalized),
      edges: buildEdges(graph, edgeStyle),
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
      edges: buildEdges(fresh.graph, 'bezier'),
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
      const normalized = normalizeUiState(previous.graph, previous.uiState);
      return {
        ...state,
        graph: previous.graph,
        uiState: normalized,
        nodes: buildNodes(previous.graph, normalized),
        edges: buildEdges(previous.graph, state.edgeStyle),
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
      const normalized = normalizeUiState(next.graph, next.uiState);
      return {
        ...state,
        graph: next.graph,
        uiState: normalized,
        nodes: buildNodes(next.graph, normalized),
        edges: buildEdges(next.graph, state.edgeStyle),
        past,
        future,
        isDirty: true,
      };
    });
  },
  setEdgeStyle: (style) => {
    set((state) => ({
      edgeStyle: style,
      edges: state.edges.map((edge) => ({
        ...edge,
        type: style === 'bezier' ? 'wire' : 'elbow',
      })),
    }));
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
      return {
        graph: {
          ...state.graph,
          nodes: graphNodes,
          wires: edgesToWires(edges),
          input_bindings,
          output_bindings,
        },
        uiState,
        nodes,
        edges,
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

      const uiState: GraphUIState = {
        ...state.uiState,
        node_states,
      };

      return {
        graph,
        uiState,
        nodes: buildNodes(graph, uiState),
        edges: buildEdges(graph, state.edgeStyle),
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
      const dirty = changes.length > 0;
      return {
        edges: nextEdges,
        graph: {
          ...state.graph,
          wires: edgesToWires(nextEdges),
        },
        past,
        future: shouldRecord ? [] : state.future,
        isDirty: state.isDirty || dirty,
      };
    });
  },
  onConnect: (connection) => {
    if (!connection.source || !connection.target) return;
    if (!connection.sourceHandle || !connection.targetHandle) return;
    const alreadyUsed = get().edges.some(
      (edge) =>
        edge.target === connection.target &&
        edge.targetHandle === connection.targetHandle
    );
    if (alreadyUsed) return;
    const edgeStyle = get().edgeStyle;
    const edge: Edge = {
      ...connection,
      id: wireId({
        source_node: connection.source,
        source_port: connection.sourceHandle,
        target_node: connection.target,
        target_port: connection.targetHandle,
      }),
      type: edgeStyle === 'bezier' ? 'wire' : 'elbow',
    };
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextEdges = addEdge(edge, state.edges);
      return {
        edges: nextEdges,
        graph: {
          ...state.graph,
          wires: edgesToWires(nextEdges),
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
