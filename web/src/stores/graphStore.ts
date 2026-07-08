import { create } from 'zustand';
import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Position,
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
  WireSpec,
  TapSpec,
  TapUIState,
  TapNodeData,
  SubgraphPreview,
  RetainedObservableSpec,
  AcausalGraphSpec,
  AcausalConnectionSpec,
} from '@/types/graph';
import { isAcausalGraphSpec, isCausalGraphSpec } from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';
import type { StudioTaskBindingSpec } from '@/types/workspace';
import {
  expandMuxForPort,
  normalizeDynamicPorts,
  normalizeMuxSpec,
} from '@/features/graph/dynamicPorts';
import { normalizeGraphForStudioAuthoring } from '@/features/graph/normalization';
import {
  acausalConnectionsFromEdges,
  acausalEdgesFromGraph,
  connectionFromReactFlow,
  portIsConserving,
} from '@/features/domains/acausal';

const DEFAULT_VIEWPORT = { x: 0, y: 0, zoom: 1 };
const DEFAULT_POSITION = { x: 200, y: 200 };
const MAX_HISTORY = 50;
const DEFAULT_EDGE_STYLE: EdgeRouting['style'] = 'bezier';
const DEFAULT_NODE_WIDTH = 220;
const DEFAULT_NODE_HEIGHT = 120;
const HEADER_HEIGHT = 40;
const TAP_WIDTH = 28;
const TAP_HEIGHT = 18;

type EditableGraphSpec = GraphSpec | AcausalGraphSpec;
type GraphHistoryEntry = { graph: EditableGraphSpec; uiState: GraphUIState };
type LayerHistory = { past: GraphHistoryEntry[]; future: GraphHistoryEntry[] };

export interface GraphLayer {
  graph: GraphSpec;
  uiState: GraphUIState;
  graphId: string | null;
  label: string;
  childNodeId?: string;
  contextType?: string;
  persistInterior?: boolean;
}

export interface StateMergeRequest {
  sourceNode: string;
  targetNode: string;
  sourceOutputs: string[];
  targetInputs: string[];
  currentSources: Record<string, WireSpec | null>;
  suggested: Record<string, string | null>;
  hasExistingConnections: boolean;
}

function recurrentZeroInitializer(width?: number, stateSlot = 'value') {
  return {
    kind: 'zeros',
    scope: 'trial',
    source: 'state_initializer',
    state_slot: stateSlot,
    ...(typeof width === 'number' ? { shape: [width] } : {}),
  };
}

function networkRecurrentWires(cellType: 'GRU' | 'LSTM', hiddenSize: number): WireSpec[] {
  const wires: WireSpec[] = [
    {
      source_node: 'cell',
      source_port: 'hidden',
      target_node: 'cell',
      target_port: 'hidden',
      temporality: 'recurrent',
      recurrent_initializer: recurrentZeroInitializer(hiddenSize, 'hidden'),
    },
  ];
  if (cellType === 'LSTM') {
    wires.push({
      source_node: 'cell',
      source_port: 'cell',
      target_node: 'cell',
      target_port: 'cell',
      temporality: 'recurrent',
      recurrent_initializer: recurrentZeroInitializer(hiddenSize, 'cell'),
    });
  }
  return wires;
}

function wireId(wire: {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
}) {
  return `${wire.source_node}:${wire.source_port}->${wire.target_node}:${wire.target_port}`;
}

function closesInstantCycle(
  graph: GraphSpec,
  sourceNode: string,
  targetNode: string
): boolean {
  if (sourceNode === targetNode) return true;
  const adjacency = new Map<string, Set<string>>();
  for (const nodeId of Object.keys(graph.nodes)) {
    adjacency.set(nodeId, new Set());
  }
  for (const wire of graph.wires) {
    if (wire.temporality === 'recurrent') continue;
    if (!graph.nodes[wire.source_node] || !graph.nodes[wire.target_node]) continue;
    const targets = adjacency.get(wire.source_node) ?? new Set<string>();
    targets.add(wire.target_node);
    adjacency.set(wire.source_node, targets);
  }
  const visited = new Set<string>();
  const stack = [targetNode];
  while (stack.length > 0) {
    const nodeId = stack.pop()!;
    if (nodeId === sourceNode) return true;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    for (const next of adjacency.get(nodeId) ?? []) {
      stack.push(next);
    }
  }
  return false;
}

function buildStateMergeRequest(graph: GraphSpec, sourceNode: string, targetNode: string): StateMergeRequest | null {
  const sourceSpec = graph.nodes[sourceNode];
  const targetSpec = graph.nodes[targetNode];
  if (!sourceSpec || !targetSpec) return null;
  const currentSources: Record<string, WireSpec | null> = {};
  for (const input of targetSpec.input_ports) {
    const existing =
      graph.wires.find(
        (wire) => wire.target_node === targetNode && wire.target_port === input
      ) ?? null;
    currentSources[input] = existing;
  }
  const suggested: Record<string, string | null> = {};
  for (const input of targetSpec.input_ports) {
    suggested[input] = sourceSpec.output_ports.includes(input) ? input : null;
  }
  const hasExistingConnections = Object.values(currentSources).some(Boolean);
  return {
    sourceNode,
    targetNode,
    sourceOutputs: [...sourceSpec.output_ports],
    targetInputs: [...targetSpec.input_ports],
    currentSources,
    suggested,
    hasExistingConnections,
  };
}

function applyStateMerge(
  graph: GraphSpec,
  sourceNode: string,
  targetNode: string,
  mapping: Record<string, string>
): GraphSpec {
  const selectedInputs = new Set(Object.keys(mapping));
  const wires = graph.wires.filter(
    (wire) => !(wire.target_node === targetNode && selectedInputs.has(wire.target_port))
  );
  const nextWires = [...wires];
  for (const [targetPort, sourcePort] of Object.entries(mapping)) {
    if (!sourcePort) continue;
    nextWires.push({
      source_node: sourceNode,
      source_port: sourcePort,
      target_node: targetNode,
      target_port: targetPort,
    });
  }
  return {
    ...graph,
    wires: nextWires,
  };
}

function isTapNodeId(nodeId: string) {
  return nodeId.startsWith('tap:');
}

function tapNodeId(tapId: string) {
  return `tap:${tapId}`;
}

function createTapId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `tap-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function buildEdgeStates(
  graph: GraphSpec | AcausalGraphSpec,
  uiState: GraphUIState,
  defaultStyle: EdgeRouting['style']
): Record<string, EdgeUIState> {
  if (isAcausalGraphSpec(graph)) {
    return {};
  }
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
    if (edge.type === 'state-flow' || edge.type === 'conserving') {
      return edge;
    }
    const routing =
      edgeStates[edge.id]?.routing ?? { style: defaultStyle, points: [] };
    if (edge.type === 'routed' && edge.data?.routing === routing) {
      return edge;
    }
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

function samePoint(
  a: { x: number; y: number } | undefined,
  b: { x: number; y: number } | undefined
) {
  return a?.x === b?.x && a?.y === b?.y;
}

function sameSize(
  a: { width: number; height: number } | undefined,
  b: { width: number; height: number } | undefined
) {
  return a?.width === b?.width && a?.height === b?.height;
}

function sameStringArray(a: string[] | undefined, b: string[] | undefined) {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  return a.every((value, index) => value === b[index]);
}

function sameRouting(a: EdgeRouting | undefined, b: EdgeRouting | undefined) {
  if (a === b) return true;
  if (!a || !b || a.style !== b.style || a.points.length !== b.points.length) return false;
  return a.points.every((point, index) => samePoint(point, b.points[index]));
}

function sameStateSlots(
  a: GraphNodeData['state_slots'] | undefined,
  b: GraphNodeData['state_slots'] | undefined
) {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  return a.every((slot, index) => {
    const other = b[index];
    return (
      slot.id === other.id &&
      slot.label === other.label &&
      JSON.stringify(slot.shape ?? null) === JSON.stringify(other.shape ?? null) &&
      JSON.stringify(slot.initializer ?? null) === JSON.stringify(other.initializer ?? null)
    );
  });
}

function sameSubgraphPreview(a: SubgraphPreview | undefined, b: SubgraphPreview | undefined) {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    sameStringArray(a.inputPorts, b.inputPorts) &&
    sameStringArray(a.outputPorts, b.outputPorts) &&
    sameGraphNodes(
      a.nodes as Node<GraphNodeData | TapNodeData>[],
      b.nodes as Node<GraphNodeData | TapNodeData>[]
    ) &&
    sameGraphEdges(a.edges as Edge<GraphEdgeData>[], b.edges as Edge<GraphEdgeData>[])
  );
}

function sameGraphNodeData(
  previous: GraphNodeData | TapNodeData,
  next: GraphNodeData | TapNodeData
) {
  if ('tap' in previous || 'tap' in next) {
    return 'tap' in previous && 'tap' in next && previous.tap === next.tap;
  }
  return (
    previous.label === next.label &&
    previous.spec === next.spec &&
    previous.collapsed === next.collapsed &&
    previous.reversed === next.reversed &&
    sameSize(previous.size, next.size) &&
    sameStringArray(previous.connected_inputs, next.connected_inputs) &&
    sameStringArray(previous.connected_outputs, next.connected_outputs) &&
    previous.state_in === next.state_in &&
    previous.state_out === next.state_out &&
    sameStateSlots(previous.state_slots, next.state_slots) &&
    sameSubgraphPreview(previous.subgraph, next.subgraph)
  );
}

function sameGraphNode(
  previous: Node<GraphNodeData | TapNodeData>,
  next: Node<GraphNodeData | TapNodeData>
) {
  return (
    previous.id === next.id &&
    previous.type === next.type &&
    previous.selected === next.selected &&
    samePoint(previous.position, next.position) &&
    sameSize(
      previous.style as { width: number; height: number } | undefined,
      next.style as { width: number; height: number } | undefined
    ) &&
    sameGraphNodeData(previous.data, next.data)
  );
}

function sameGraphNodes(
  previous: Node<GraphNodeData | TapNodeData>[],
  next: Node<GraphNodeData | TapNodeData>[]
) {
  return (
    previous.length === next.length &&
    previous.every((node, index) => sameGraphNode(node, next[index]))
  );
}

function sameGraphEdgeData(
  previous: GraphEdgeData | undefined,
  next: GraphEdgeData | undefined
) {
  if (previous === next) return true;
  if (!previous || !next) return false;
  return (
    sameRouting(previous.routing, next.routing) &&
    previous.primary === next.primary &&
    previous.strength === next.strength &&
    previous.schema_status === next.schema_status &&
    previous.schema_message === next.schema_message &&
    previous.temporality === next.temporality &&
    previous.recurrent_initializer === next.recurrent_initializer
  );
}

function sameGraphEdge(previous: Edge<GraphEdgeData>, next: Edge<GraphEdgeData>) {
  const previousPositions = previous as Edge<GraphEdgeData> & {
    sourcePosition?: Position;
    targetPosition?: Position;
  };
  const nextPositions = next as Edge<GraphEdgeData> & {
    sourcePosition?: Position;
    targetPosition?: Position;
  };
  return (
    previous.id === next.id &&
    previous.type === next.type &&
    previous.source === next.source &&
    previous.target === next.target &&
    previous.sourceHandle === next.sourceHandle &&
    previous.targetHandle === next.targetHandle &&
    previousPositions.sourcePosition === nextPositions.sourcePosition &&
    previousPositions.targetPosition === nextPositions.targetPosition &&
    previous.selected === next.selected &&
    previous.selectable === next.selectable &&
    previous.deletable === next.deletable &&
    previous.zIndex === next.zIndex &&
    sameGraphEdgeData(previous.data, next.data)
  );
}

function sameGraphEdges(previous: Edge<GraphEdgeData>[], next: Edge<GraphEdgeData>[]) {
  return (
    previous.length === next.length &&
    previous.every((edge, index) => sameGraphEdge(edge, next[index]))
  );
}

function reconcileById<T extends { id: string }>(
  previous: T[],
  next: T[],
  sameEntity: (previous: T, next: T) => boolean
) {
  const previousById = new Map(previous.map((entity) => [entity.id, entity]));
  let changed = previous.length !== next.length;
  const reconciled = next.map((entity) => {
    const existing = previousById.get(entity.id);
    if (existing && sameEntity(existing, entity)) {
      return existing;
    }
    changed = true;
    return entity;
  });
  if (!changed && previous.every((entity, index) => entity === reconciled[index])) {
    return previous;
  }
  return reconciled;
}

function reconcileNodes(
  previous: Node<GraphNodeData | TapNodeData>[],
  graph: GraphSpec | AcausalGraphSpec,
  uiState: GraphUIState
) {
  return reconcileById(previous, buildNodes(graph, uiState), sameGraphNode);
}

function reconcileEdges(
  previous: Edge<GraphEdgeData>[],
  graph: GraphSpec | AcausalGraphSpec,
  uiState: GraphUIState,
  edgeStyle: EdgeRouting['style']
) {
  return reconcileById(previous, buildEdges(graph, uiState, edgeStyle), sameGraphEdge);
}

function setNodeSelection(
  nodes: Node<GraphNodeData | TapNodeData>[],
  selectedId: string | null
) {
  let changed = false;
  const next = nodes.map((node) => {
    const selected = node.id === selectedId;
    if (node.selected === selected) return node;
    changed = true;
    return { ...node, selected };
  });
  return changed ? next : nodes;
}

function setEdgeSelection(edges: Edge<GraphEdgeData>[], selectedId: string | null) {
  let changed = false;
  const next = edges.map((edge) => {
    const selected = edge.id === selectedId;
    if (edge.selected === selected) return edge;
    changed = true;
    return { ...edge, selected };
  });
  return changed ? next : edges;
}

export function createInitialGraph(): { graph: GraphSpec; uiState: GraphUIState } {
  const graph: GraphSpec = {
    nodes: {
      input_mux: {
        type: 'Mux',
        params: { n_inputs: 2 },
        input_ports: ['in_0', 'in_1'],
        output_ports: ['output'],
      },
      cell: {
        type: 'GRU',
        params: {
          input_size: 6,
          hidden_size: 100,
        },
        input_ports: ['input', 'hidden'],
        output_ports: ['output', 'hidden'],
      },
      readout: {
        type: 'Linear',
        params: {
          input_size: 100,
          output_size: 2,
          activation: 'tanh',
          use_bias: true,
        },
        input_ports: ['input'],
        output_ports: ['output'],
      },
      mechanics: {
        type: 'TwoLinkArm',
        params: {
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
        source_node: 'feedback',
        source_port: 'output',
        target_node: 'input_mux',
        target_port: 'in_1',
      },
      {
        source_node: 'input_mux',
        source_port: 'output',
        target_node: 'cell',
        target_port: 'input',
      },
      {
        source_node: 'cell',
        source_port: 'hidden',
        target_node: 'readout',
        target_port: 'input',
      },
      {
        source_node: 'cell',
        source_port: 'hidden',
        target_node: 'cell',
        target_port: 'hidden',
        temporality: 'recurrent',
        recurrent_initializer: {
          kind: 'zeros',
          scope: 'trial',
          shape: [100],
          source: 'state_initializer',
          state_slot: 'hidden',
        },
      },
      {
        source_node: 'readout',
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
    taps: [],
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
      input_mux: { position: { x: 300, y: 200 }, collapsed: false, selected: false },
      cell: { position: { x: 480, y: 200 }, collapsed: false, selected: false },
      readout: { position: { x: 660, y: 200 }, collapsed: false, selected: false },
      mechanics: { position: { x: 840, y: 200 }, collapsed: false, selected: false },
      feedback: { position: { x: 560, y: 400 }, collapsed: false, selected: false },
    },
  };

  const uiState: GraphUIState = {
    ...baseUiState,
    edge_states: buildEdgeStates(graph, baseUiState, DEFAULT_EDGE_STYLE),
  };

  return { graph, uiState };
}

export function createBlankGraph(): GraphSpec {
  const now = new Date().toISOString();
  return {
    nodes: {},
    wires: [],
    input_ports: [],
    output_ports: [],
    input_bindings: {},
    output_bindings: {},
    metadata: { name: '', created_at: now, updated_at: now, version: '1.0.0' },
  };
}

function interiorDomainForNode(
  nodeSpec: ComponentSpec | undefined,
  componentRegistry: Map<string, ComponentDefinition>
): string | null {
  if (!nodeSpec) return null;
  return componentRegistry.get(nodeSpec.type)?.interior_domain ?? null;
}

function refreshLayerContexts(
  graphStack: GraphLayer[],
  componentRegistry: Map<string, ComponentDefinition>
) {
  let currentContext = 'top-level';
  const nextStack = graphStack.map((layer) => {
    const contextType = interiorDomainForNode(
      layer.childNodeId ? layer.graph.nodes[layer.childNodeId] : undefined,
      componentRegistry
    ) ?? layer.contextType;
    if (contextType) currentContext = contextType;
    return { ...layer, contextType };
  });
  return { graphStack: nextStack, currentContext };
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
      if (isInternalStateInput(nodeSpec, port)) continue;
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

function isInternalStateInput(nodeSpec: ComponentSpec, port: string): boolean {
  return (nodeSpec.type === 'GRU' || nodeSpec.type === 'LSTM') && (port === 'hidden' || port === 'cell');
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
  graph: EditableGraphSpec,
  uiState: GraphUIState | null | undefined,
  defaultEdgeStyle: EdgeRouting['style']
): GraphUIState {
  const base: GraphUIState = uiState ?? { viewport: DEFAULT_VIEWPORT, node_states: {} };
  const node_states = { ...base.node_states };

  // Merge loaded state OVER defaults: for each node in the graph, create a
  // default entry first, then spread any loaded state on top so that saved
  // values (e.g. reversed: true) are never overwritten by defaults.
  let offset = 0;
  for (const nodeId of Object.keys(graph.nodes ?? {})) {
    const defaults = {
      position: { x: DEFAULT_POSITION.x + offset, y: DEFAULT_POSITION.y + offset },
      collapsed: false,
      selected: false,
      reversed: false,
    };
    const loaded = node_states[nodeId];
    if (loaded) {
      // Spread defaults under loaded state so any saved field wins,
      // while missing fields (e.g. reversed absent from older saves) get defaults.
      node_states[nodeId] = { ...defaults, ...loaded };
    } else {
      node_states[nodeId] = defaults;
      offset += 60;
    }
  }

  const edge_states = buildEdgeStates(graph, base, defaultEdgeStyle);

  const subgraph_states: Record<string, GraphUIState> = {};
  if (graph.subgraphs) {
    for (const [nodeId, subgraph] of Object.entries(graph.subgraphs)) {
      if (!isCausalGraphSpec(subgraph) && !isAcausalGraphSpec(subgraph)) continue;
      const childState = base.subgraph_states?.[nodeId];
      subgraph_states[nodeId] = normalizeUiState(subgraph, childState, defaultEdgeStyle);
    }
  }
  const tap_states = base.tap_states
    ? Object.fromEntries(
        Object.entries(base.tap_states).filter(([id]) =>
          !isAcausalGraphSpec(graph) && (graph.taps ?? []).some((tap) => tap.id === id)
        )
      )
    : undefined;

  return {
    viewport: base.viewport ?? DEFAULT_VIEWPORT,
    node_states,
    edge_states,
    subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
    tap_states: tap_states && Object.keys(tap_states).length ? tap_states : undefined,
  };
}

function subgraphPreviewFromGraph(
  graph: EditableGraphSpec,
  uiState?: GraphUIState,
): SubgraphPreview {
  const normalizedUiState = normalizeUiState(graph, uiState, DEFAULT_EDGE_STYLE);
  return {
    nodes: buildComponentNodes(graph, normalizedUiState),
    edges: buildEdges(graph, normalizedUiState, DEFAULT_EDGE_STYLE),
    inputPorts: isAcausalGraphSpec(graph) ? [] : graph.input_ports,
    outputPorts: isAcausalGraphSpec(graph) ? [] : graph.output_ports,
  };
}

function buildComponentNodes(graph: EditableGraphSpec, uiState: GraphUIState): Node<GraphNodeData>[] {
  const connectedInputs = new Map<string, Set<string>>();
  const connectedOutputs = new Map<string, Set<string>>();
  const stateIn = new Set<string>();
  const stateOut = new Set<string>();
  for (const wire of isAcausalGraphSpec(graph) ? [] : graph.wires) {
    if (!isTapNodeId(wire.target_node)) {
      const inputs = connectedInputs.get(wire.target_node) ?? new Set<string>();
      inputs.add(wire.target_port);
      connectedInputs.set(wire.target_node, inputs);
      stateIn.add(wire.target_node);
    }
    if (!isTapNodeId(wire.source_node)) {
      const outputs = connectedOutputs.get(wire.source_node) ?? new Set<string>();
      outputs.add(wire.source_port);
      connectedOutputs.set(wire.source_node, outputs);
      stateOut.add(wire.source_node);
    }
  }
  if (isAcausalGraphSpec(graph)) {
    for (const connection of graph.connections ?? []) {
      for (const [nodeId, port] of [connection.a, connection.b]) {
        const inputs = connectedInputs.get(nodeId) ?? new Set<string>();
        inputs.add(port);
        connectedInputs.set(nodeId, inputs);
        const outputs = connectedOutputs.get(nodeId) ?? new Set<string>();
        outputs.add(port);
        connectedOutputs.set(nodeId, outputs);
      }
    }
  }
  return Object.entries(graph.nodes ?? {}).map(([id, spec]) => {
    const ui = uiState.node_states[id] ?? {
      position: DEFAULT_POSITION,
      collapsed: false,
      selected: false,
    };
    const size = ui.size;
    const subgraphGraph = graph.subgraphs?.[id];
    const subgraphUiState = uiState.subgraph_states?.[id];
    const isSubgraphNode = isCausalGraphSpec(subgraphGraph) || isAcausalGraphSpec(subgraphGraph);
    const subgraph = isSubgraphNode
      ? subgraphPreviewFromGraph(subgraphGraph, subgraphUiState)
      : undefined;
    return {
      id,
      type: isSubgraphNode ? 'subgraph' : 'component',
      position: ui.position,
      style: size ? { width: size.width, height: size.height } : undefined,
      data: {
        label: id,
        spec,
        current_domain: isAcausalGraphSpec(graph) ? 'feedbax.domain.acausal' : undefined,
        collapsed: ui.collapsed,
        reversed: ui.reversed ?? false,
        size,
        connected_inputs: Array.from(connectedInputs.get(id) ?? []),
        connected_outputs: Array.from(connectedOutputs.get(id) ?? []),
        state_in: stateIn.has(id),
        state_out: stateOut.has(id),
        state_slots: stateSlotsForNodeSpec(spec),
        subgraph,
      },
      selected: ui.selected,
    };
  });
}

function computeTapPosition(
  graph: GraphSpec,
  uiState: GraphUIState,
  tap: TapSpec
): { x: number; y: number } {
  const afterNode = tap.position.afterNode;
  const sourceState = uiState.node_states[afterNode];
  if (!sourceState) return { x: DEFAULT_POSITION.x, y: DEFAULT_POSITION.y };
  const sourceSize = sourceState.size ?? {
    width: DEFAULT_NODE_WIDTH,
    height: DEFAULT_NODE_HEIGHT,
  };
  const sourcePoint = {
    x: sourceState.position.x + sourceSize.width,
    y: sourceState.position.y + HEADER_HEIGHT / 2,
  };
  const targetNode =
    tap.position.targetNode ??
    graph.wires.find(
      (wire) => wire.source_node === afterNode && graph.nodes[wire.target_node]
    )?.target_node;
  if (!targetNode) {
    return {
      x: sourcePoint.x + 160 - TAP_WIDTH / 2,
      y: sourcePoint.y - TAP_HEIGHT / 2,
    };
  }
  const targetState = uiState.node_states[targetNode];
  if (!targetState) {
    return {
      x: sourcePoint.x + 160 - TAP_WIDTH / 2,
      y: sourcePoint.y - TAP_HEIGHT / 2,
    };
  }
  const targetSize = targetState.size ?? {
    width: DEFAULT_NODE_WIDTH,
    height: DEFAULT_NODE_HEIGHT,
  };
  const targetPoint = {
    x: targetState.position.x,
    y: targetState.position.y + HEADER_HEIGHT / 2,
  };
  return {
    x: (sourcePoint.x + targetPoint.x) / 2 - TAP_WIDTH / 2,
    y: (sourcePoint.y + targetPoint.y) / 2 - TAP_HEIGHT / 2,
  };
}

function buildTapNodes(graph: EditableGraphSpec, uiState: GraphUIState): Node<TapNodeData>[] {
  if (isAcausalGraphSpec(graph)) return [];
  const taps = graph.taps ?? [];
  return taps.map((tap) => {
    const tapState = uiState.tap_states?.[tap.id];
    const position = tapState?.position ?? computeTapPosition(graph, uiState, tap);
    return {
      id: tapNodeId(tap.id),
      type: 'tap',
      position,
      data: {
        tap,
      },
      style: { width: TAP_WIDTH, height: TAP_HEIGHT },
      selected: tapState?.selected ?? false,
    };
  });
}

function buildNodes(graph: EditableGraphSpec, uiState: GraphUIState): Node<GraphNodeData | TapNodeData>[] {
  return [...buildComponentNodes(graph, uiState), ...buildTapNodes(graph, uiState)];
}

function stateSlotsForNodeSpec(spec: ComponentSpec): GraphNodeData['state_slots'] {
  const hiddenSize = typeof spec.params.hidden_size === 'number' ? spec.params.hidden_size : undefined;
  if (spec.type === 'GRU') {
    return [
      {
        id: 'hidden',
        label: 'Hidden state',
        shape: typeof hiddenSize === 'number' ? [hiddenSize] : null,
        initializer: recurrentZeroInitializer(hiddenSize, 'hidden'),
      },
    ];
  }
  if (spec.type === 'LSTM') {
    return [
      {
        id: 'hidden',
        label: 'Hidden state',
        shape: typeof hiddenSize === 'number' ? [hiddenSize] : null,
        initializer: recurrentZeroInitializer(hiddenSize, 'hidden'),
      },
      {
        id: 'cell',
        label: 'Cell state',
        shape: typeof hiddenSize === 'number' ? [hiddenSize] : null,
        initializer: recurrentZeroInitializer(hiddenSize, 'cell'),
      },
    ];
  }
  return [];
}

function buildStateEdges(graph: GraphSpec, uiState: GraphUIState): Edge<GraphEdgeData>[] {
  const countsByTarget = new Map<string, Map<string, number>>();
  for (const wire of graph.wires) {
    if (isTapNodeId(wire.source_node) || isTapNodeId(wire.target_node)) {
      continue;
    }
    const sources = countsByTarget.get(wire.target_node) ?? new Map<string, number>();
    sources.set(wire.source_node, (sources.get(wire.source_node) ?? 0) + 1);
    countsByTarget.set(wire.target_node, sources);
  }

  const reversedNodes = new Set(
    Object.entries(uiState.node_states)
      .filter(([, state]) => state.reversed)
      .map(([nodeId]) => nodeId)
  );

  const edges: Edge<GraphEdgeData>[] = [];
  for (const [target, sources] of countsByTarget.entries()) {
    let maxCount = 0;
    for (const count of sources.values()) {
      if (count > maxCount) maxCount = count;
    }
    for (const [source, count] of sources.entries()) {
      edges.push({
        id: `state:${source}->${target}`,
        source,
        target,
        sourceHandle: '__state_out',
        targetHandle: '__state_in',
        type: 'state-flow',
        selectable: true,
        deletable: false,
        zIndex: 0,
        sourcePosition: reversedNodes.has(source) ? Position.Left : Position.Right,
        targetPosition: reversedNodes.has(target) ? Position.Right : Position.Left,
        data: {
          primary: count === maxCount,
          strength: count,
        },
      } as Edge<GraphEdgeData>);
    }
  }
  return edges;
}

function buildEdges(
  graph: EditableGraphSpec,
  uiState: GraphUIState,
  defaultStyle: EdgeRouting['style']
): Edge<GraphEdgeData>[] {
  if (isAcausalGraphSpec(graph)) {
    return acausalEdgesFromGraph(graph);
  }
  const edgeStates = buildEdgeStates(graph, uiState, defaultStyle);
  const collapsed = new Set(
    Object.entries(uiState.node_states)
      .filter(([, state]) => state.collapsed)
      .map(([nodeId]) => nodeId)
  );
  const reversedNodes = new Set(
    Object.entries(uiState.node_states)
      .filter(([, state]) => state.reversed)
      .map(([nodeId]) => nodeId)
  );
  const isCollapsed = (nodeId: string) => collapsed.has(nodeId);
  const isComponent = (nodeId: string) => !isTapNodeId(nodeId);
  const portEdges = graph.wires
    .filter(
      (wire) =>
        !(isComponent(wire.source_node) && isCollapsed(wire.source_node)) &&
        !(isComponent(wire.target_node) && isCollapsed(wire.target_node))
    )
    .map((wire) => {
      const id = wireId(wire);
      return {
        id,
        source: wire.source_node,
        target: wire.target_node,
        sourceHandle: wire.source_port,
        targetHandle: wire.target_port,
        type: 'routed',
        zIndex: 1,
        sourcePosition: reversedNodes.has(wire.source_node) ? Position.Left : Position.Right,
        targetPosition: reversedNodes.has(wire.target_node) ? Position.Right : Position.Left,
        data: {
          routing: edgeStates[id]?.routing ?? { style: defaultStyle, points: [] },
          temporality: wire.temporality ?? 'instant',
          recurrent_initializer: wire.recurrent_initializer ?? null,
        },
      };
    });
  return [...buildStateEdges(graph, uiState), ...portEdges];
}

function edgesToWires(edges: Edge<GraphEdgeData>[]): GraphSpec['wires'] {
  return edges
    .filter(
      (edge) =>
        edge.type !== 'state-flow' &&
        edge.source &&
        edge.target &&
        edge.sourceHandle &&
        edge.targetHandle
    )
    .map((edge) => ({
      source_node: edge.source,
      source_port: edge.sourceHandle as string,
      target_node: edge.target,
      target_port: edge.targetHandle as string,
      temporality: edge.data?.temporality,
      recurrent_initializer: edge.data?.recurrent_initializer ?? null,
    }));
}

function mergeAcausalConnection(
  connections: AcausalConnectionSpec[],
  connection: AcausalConnectionSpec
): AcausalConnectionSpec[] {
  const nextId = `${connection.a[0]}:${connection.a[1]}|${connection.b[0]}:${connection.b[1]}`;
  const existing = new Set(
    connections.map((item) => `${item.a[0]}:${item.a[1]}|${item.b[0]}:${item.b[1]}`)
  );
  return existing.has(nextId) ? connections : [...connections, connection];
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

function cloneGraphSpec<T>(value: T): T {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function sanitizeNodeId(value: string, fallback = 'node'): string {
  const sanitized = value
    .trim()
    .replace(/[^A-Za-z0-9_]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return sanitized || fallback;
}

function uniqueImportedNodeId(
  used: Set<string>,
  preferred: string,
  fallbackPrefix: string
): string {
  const cleanPreferred = sanitizeNodeId(preferred);
  if (!used.has(cleanPreferred)) {
    used.add(cleanPreferred);
    return cleanPreferred;
  }

  const base = `${sanitizeNodeId(fallbackPrefix, 'template')}_${cleanPreferred}`;
  let candidate = base;
  let index = 2;
  while (used.has(candidate)) {
    candidate = `${base}_${index}`;
    index += 1;
  }
  used.add(candidate);
  return candidate;
}

function uniqueImportedId(used: Set<string>, preferred: string, fallbackPrefix: string): string {
  if (!used.has(preferred)) {
    used.add(preferred);
    return preferred;
  }
  const base = `${fallbackPrefix}:${preferred}`;
  let candidate = base;
  let index = 2;
  while (used.has(candidate)) {
    candidate = `${base}:${index}`;
    index += 1;
  }
  used.add(candidate);
  return candidate;
}

function remapNodeSelector(selector: string, nodeMap: Record<string, string>): string {
  const remapNodePort = (value: string) => {
    const [node, ...rest] = value.split('.');
    if (!node || rest.length === 0) return value;
    return `${nodeMap[node] ?? node}.${rest.join('.')}`;
  };
  const remapEdgeEndpoint = (value: string) => {
    if (value.includes('.')) return remapNodePort(value);
    const [node, ...rest] = value.split(':');
    if (!node || rest.length === 0) return value;
    return `${nodeMap[node] ?? node}:${rest.join(':')}`;
  };
  const remapEdgeId = (value: string) => {
    const [source, target] = value.split('->');
    if (!source || !target) return value;
    return `${remapEdgeEndpoint(source)}->${remapEdgeEndpoint(target)}`;
  };
  const remapStatePath = (value: string) => {
    const [node, ...rest] = value.split('.');
    if (!node || rest.length === 0) return value;
    return `${nodeMap[node] ?? node}.${rest.join('.')}`;
  };

  if (selector.startsWith('port:')) {
    return `port:${remapNodePort(selector.slice('port:'.length))}`;
  }
  if (selector.startsWith('edge:') || selector.startsWith('recurrent_carry:')) {
    const prefix = selector.startsWith('edge:') ? 'edge:' : 'recurrent_carry:';
    return `${prefix}${remapEdgeId(selector.slice(prefix.length))}`;
  }
  if (selector.includes('->')) {
    return remapEdgeId(selector);
  }
  if (selector.startsWith('path:states.')) {
    return `path:states.${remapStatePath(selector.slice('path:states.'.length))}`;
  }
  if (selector.startsWith('state_path:states.')) {
    return `state_path:states.${remapStatePath(selector.slice('state_path:states.'.length))}`;
  }
  if (selector.startsWith('states.')) {
    return `states.${remapStatePath(selector.slice('states.'.length))}`;
  }
  return selector;
}

function remapRetainedObservable(
  observable: RetainedObservableSpec,
  nodeMap: Record<string, string>,
  usedIds: Set<string>,
  fallbackPrefix: string
): RetainedObservableSpec {
  const id = uniqueImportedId(usedIds, observable.id, `${fallbackPrefix}:observable`);
  const selector = observable.selector
    ? remapNodeSelector(observable.selector, nodeMap)
    : observable.selector;
  const target = observable.target
    ? {
        ...observable.target,
        selector: remapNodeSelector(observable.target.selector, nodeMap),
        node_id: observable.target.node_id
          ? nodeMap[observable.target.node_id] ?? observable.target.node_id
          : observable.target.node_id,
        edge_id: observable.target.edge_id
          ? remapNodeSelector(observable.target.edge_id, nodeMap)
          : observable.target.edge_id,
        path: observable.target.path
          ? remapNodeSelector(observable.target.path, nodeMap)
          : observable.target.path,
      }
    : observable.target;
  return {
    ...observable,
    id,
    selector,
    target,
  };
}

function importTemplateGraphIntoGraph(
  graph: GraphSpec,
  uiState: GraphUIState,
  templateGraph: EditableGraphSpec,
  templateUiState: GraphUIState | undefined,
  dropPosition: { x: number; y: number },
  templateName: string
): { graph: GraphSpec; uiState: GraphUIState; importedNodeIds: string[] } {
  const imported = cloneGraphSpec(templateGraph);
  if (isAcausalGraphSpec(imported)) {
    return { graph, uiState, importedNodeIds: [] };
  }
  const usedNodeIds = new Set(Object.keys(graph.nodes ?? {}));
  const fallbackPrefix = sanitizeNodeId(templateName, 'template');
  const nodeMap: Record<string, string> = {};
  for (const nodeId of Object.keys(imported.nodes ?? {})) {
    nodeMap[nodeId] = uniqueImportedNodeId(usedNodeIds, nodeId, fallbackPrefix);
  }

  const templatePositions = Object.entries(imported.nodes ?? {}).map(([nodeId], index) => {
    const position = templateUiState?.node_states?.[nodeId]?.position;
    return {
      nodeId,
      position: position ?? {
        x: DEFAULT_POSITION.x + index * (DEFAULT_NODE_WIDTH + 40),
        y: DEFAULT_POSITION.y,
      },
    };
  });
  const minX = Math.min(...templatePositions.map((item) => item.position.x));
  const minY = Math.min(...templatePositions.map((item) => item.position.y));

  const importedNodes = Object.fromEntries(
    Object.entries(imported.nodes ?? {}).map(([nodeId, spec]) => [
      nodeMap[nodeId],
      cloneGraphSpec(spec),
    ])
  );

  const nodeStates: GraphUIState['node_states'] = Object.fromEntries(
    Object.entries(uiState.node_states).map(([nodeId, state]) => [
      nodeId,
      { ...state, selected: false },
    ])
  );
  for (const { nodeId, position } of templatePositions) {
    const remappedNodeId = nodeMap[nodeId];
    const sourceUi = templateUiState?.node_states?.[nodeId];
    nodeStates[remappedNodeId] = {
      position: {
        x: dropPosition.x + (position.x - minX),
        y: dropPosition.y + (position.y - minY),
      },
      collapsed: sourceUi?.collapsed ?? false,
      selected: true,
      reversed: sourceUi?.reversed ?? false,
      size: sourceUi?.size,
    };
  }

  const subgraphStates: GraphUIState['subgraph_states'] = {
    ...(uiState.subgraph_states ?? {}),
  };
  for (const [nodeId, state] of Object.entries(templateUiState?.subgraph_states ?? {})) {
    const remappedNodeId = nodeMap[nodeId];
    if (remappedNodeId) {
      subgraphStates[remappedNodeId] = cloneGraphSpec(state);
    }
  }

  const nextUiState: GraphUIState = {
    ...uiState,
    node_states: nodeStates,
    subgraph_states:
      Object.keys(subgraphStates).length > 0 ? subgraphStates : uiState.subgraph_states,
  };

  const importedWires = imported.wires.map((wire) => ({
    ...wire,
    source_node: nodeMap[wire.source_node] ?? wire.source_node,
    target_node: nodeMap[wire.target_node] ?? wire.target_node,
  }));

  const importedSubgraphs: NonNullable<GraphSpec['subgraphs']> = {};
  for (const [nodeId, subgraph] of Object.entries(imported.subgraphs ?? {})) {
    const remappedNodeId = nodeMap[nodeId];
    if (remappedNodeId) {
      importedSubgraphs[remappedNodeId] = cloneGraphSpec(subgraph);
    }
  }

  const usedObservableIds = new Set((graph.retained_observables ?? []).map((item) => item.id));
  const retained_observables = [
    ...(graph.retained_observables ?? []),
    ...(imported.retained_observables ?? []).map((observable) =>
      remapRetainedObservable(observable, nodeMap, usedObservableIds, fallbackPrefix)
    ),
  ];

  const nextGraph: GraphSpec = {
    ...graph,
    nodes: {
      ...graph.nodes,
      ...importedNodes,
    },
    wires: [...graph.wires, ...importedWires],
    subgraphs:
      Object.keys(importedSubgraphs).length > 0 || graph.subgraphs
        ? {
            ...(graph.subgraphs ?? {}),
            ...importedSubgraphs,
          }
        : graph.subgraphs,
    retained_observables:
      retained_observables.length > 0 ? retained_observables : graph.retained_observables,
  };

  return {
    graph: nextGraph,
    uiState: nextUiState,
    importedNodeIds: Object.values(nodeMap),
  };
}

function cloneSnapshot(graph: EditableGraphSpec, uiState: GraphUIState) {
  if (typeof structuredClone === 'function') {
    return structuredClone({ graph, uiState });
  }
  return JSON.parse(JSON.stringify({ graph, uiState })) as { graph: GraphSpec; uiState: GraphUIState };
}

function applyNodeParamUpdatesToGraph(
  graph: GraphSpec,
  updates: Array<{ nodeId: string; param: string; value: ComponentSpec['params'][string] }>
): GraphSpec {
  if (updates.length === 0) return graph;
  let changed = false;
  const nodes = { ...graph.nodes };
  for (const update of updates) {
    const node = nodes[update.nodeId];
    if (!node || node.params[update.param] === update.value) continue;
    nodes[update.nodeId] = {
      ...node,
      params: {
        ...node.params,
        [update.param]: update.value,
      },
    };
    changed = true;
  }
  return changed ? { ...graph, nodes } : graph;
}

function renameBoundaryPortInGraph(
  graph: GraphSpec,
  direction: 'input' | 'output',
  previousPort: string,
  nextPort: string
): GraphSpec | null {
  const portsKey = direction === 'input' ? 'input_ports' : 'output_ports';
  const bindingsKey = direction === 'input' ? 'input_bindings' : 'output_bindings';
  const ports = graph[portsKey];
  if (!ports.includes(previousPort) || ports.includes(nextPort)) return null;

  const bindings = { ...graph[bindingsKey] };
  if (bindings[previousPort]) {
    bindings[nextPort] = bindings[previousPort];
    delete bindings[previousPort];
  }
  return {
    ...graph,
    [portsKey]: ports.map((port) => (port === previousPort ? nextPort : port)),
    [bindingsKey]: bindings,
  };
}

function renameNodePortReferences(
  graph: GraphSpec,
  nodeId: string,
  direction: 'input' | 'output',
  previousPort: string,
  nextPort: string
): GraphSpec {
  const wires = graph.wires.map((wire) => {
    if (
      direction === 'input' &&
      wire.target_node === nodeId &&
      wire.target_port === previousPort
    ) {
      return { ...wire, target_port: nextPort };
    }
    if (
      direction === 'output' &&
      wire.source_node === nodeId &&
      wire.source_port === previousPort
    ) {
      return { ...wire, source_port: nextPort };
    }
    return wire;
  });
  const input_bindings =
    direction === 'input'
      ? Object.fromEntries(
          Object.entries(graph.input_bindings).map(([key, value]) => [
            key,
            [value[0], value[0] === nodeId && value[1] === previousPort ? nextPort : value[1]],
          ])
        ) as Record<string, [string, string]>
      : graph.input_bindings;
  const output_bindings =
    direction === 'output'
      ? Object.fromEntries(
          Object.entries(graph.output_bindings).map(([key, value]) => [
            key,
            [value[0], value[0] === nodeId && value[1] === previousPort ? nextPort : value[1]],
          ])
        ) as Record<string, [string, string]>
      : graph.output_bindings;
  return { ...graph, wires, input_bindings, output_bindings };
}

export interface GraphSnapshot {
  graph: GraphSpec;
  uiState: GraphUIState;
  graphId: string | null;
  saveRevision: number | null;
  isDirty: boolean;
  lastSavedAt: string | null;
  graphStack: GraphLayer[];
  currentGraphLabel: string;
  currentContext: string;
  edgeStyle: 'bezier' | 'elbow';
  past: GraphHistoryEntry[];
  future: GraphHistoryEntry[];
  selectedTapId: string | null;
  selectedEdgeId: string | null;
  pendingStateMerge: StateMergeRequest | null;
}

export interface PersistableGraphSnapshot {
  graph: GraphSpec;
  uiState: GraphUIState;
  graphStackPath: string[];
}

interface GraphStoreState {
  graphId: string | null;
  saveRevision: number | null;
  graph: GraphSpec;
  uiState: GraphUIState;
  nodes: Node<GraphNodeData | TapNodeData>[];
  edges: Edge<GraphEdgeData>[];
  edgeStyle: 'bezier' | 'elbow';
  graphStack: GraphLayer[];
  currentGraphLabel: string;
  _compositeTypes: Set<string>;
  _componentRegistry: Map<string, ComponentDefinition>;
  _isRegistryLoaded: boolean;
  currentContext: string;
  isDirty: boolean;
  lastSavedAt: string | null;
  lastSubgraphError: string | null;
  past: GraphHistoryEntry[];
  future: GraphHistoryEntry[];
  graphHistory: Record<string, LayerHistory>;
  selectedTapId: string | null;
  selectedEdgeId: string | null;
  pendingStateMerge: StateMergeRequest | null;
  hydrateGraph: (
    graph: GraphSpec,
    uiState?: GraphUIState | null,
    graphId?: string | null,
    graphStackPath?: string[] | null,
    saveRevision?: number | null
  ) => void;
  capturePersistedGraph: () => PersistableGraphSnapshot;
  captureGraphStackPath: () => string[];
  restoreSnapshot: (snapshot: GraphSnapshot) => void;
  markSaved: (graphId: string, saveRevision?: number | null) => void;
  markDirty: () => void;
  resetGraph: () => void;
  undo: () => void;
  redo: () => void;
  deleteSelected: () => void;
  duplicateSelected: () => void;
  setEdgeStyle: (style: 'bezier' | 'elbow') => void;
  addEdgePoint: (edgeId: string, point: { x: number; y: number }) => void;
  updateEdgePoint: (edgeId: string, index: number, point: { x: number; y: number }) => void;
  removeEdgePoint: (edgeId: string, index: number) => void;
  toggleEdgeStyleForEdge: (edgeId: string) => void;
  enterSubgraph: (nodeId: string) => void;
  wrapInParentGraph: () => void;
  exitToBreadcrumb: (index: number) => void;
  renameNode: (nodeId: string, nextId: string) => void;
  renameSubgraphBoundaryPort: (
    nodeId: string,
    direction: 'input' | 'output',
    previousPort: string,
    nextPort: string
  ) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onConnect: (
    connection: Connection,
    styleOverride?: 'bezier' | 'elbow',
    paramUpdates?: Array<{ nodeId: string; param: string; value: ComponentSpec['params'][string] }>,
    wireOptions?: Pick<WireSpec, 'temporality' | 'recurrent_initializer'>
  ) => void;
  addNodeFromComponent: (component: ComponentDefinition, position: { x: number; y: number }) => void;
  updateNodeParams: (
    nodeId: string,
    paramName: string,
    value: ComponentSpec['params'][string],
    taskBindingSpec?: StudioTaskBindingSpec | null
  ) => void;
  updateNodeParamsBatch: (
    updates: Array<{ nodeId: string; param: string; value: ComponentSpec['params'][string] }>,
    taskBindingSpec?: StudioTaskBindingSpec | null
  ) => void;
  setSelectedNode: (nodeId: string | null) => void;
  setSelectedTap: (tapId: string | null) => void;
  setSelectedEdge: (edgeId: string | null) => void;
  clearSelection: () => void;
  selectAll: () => void;
  toggleNodeCollapse: (nodeId: string) => void;
  toggleNodeReversed: (nodeId: string) => void;
  setAllNodesCollapsed: (collapsed: boolean) => void;
  addTapForEdge: (edgeId: string, type: TapSpec['type']) => void;
  addTap: (afterNode: string, type: TapSpec['type']) => void;
  updateTap: (tapId: string, updates: Partial<TapSpec>) => void;
  removeTap: (tapId: string) => void;
  addRetainedObservable: (observable: RetainedObservableSpec) => void;
  updateRetainedObservable: (
    observableId: string,
    updates: Partial<RetainedObservableSpec>
  ) => void;
  removeRetainedObservable: (observableId: string) => void;
  confirmStateMerge: (mapping: Record<string, string>) => void;
  cancelStateMerge: () => void;
  setCompositeTypes: (types: Set<string>) => void;
  setComponentRegistry: (components: ComponentDefinition[]) => void;
}

export const graphStoreSlices = {
  topology: (state: GraphStoreState) => ({
    graph: state.graph,
    updateNodeParams: state.updateNodeParams,
    updateNodeParamsBatch: state.updateNodeParamsBatch,
    addNodeFromComponent: state.addNodeFromComponent,
    deleteSelected: state.deleteSelected,
    duplicateSelected: state.duplicateSelected,
    renameNode: state.renameNode,
    renameSubgraphBoundaryPort: state.renameSubgraphBoundaryPort,
    onConnect: state.onConnect,
    addTap: state.addTap,
    addTapForEdge: state.addTapForEdge,
    updateTap: state.updateTap,
    removeTap: state.removeTap,
  }),
  flowAdapter: (state: GraphStoreState) => ({
    nodes: state.nodes,
    edges: state.edges,
    edgeStyle: state.edgeStyle,
    uiState: state.uiState,
    onNodesChange: state.onNodesChange,
    onEdgesChange: state.onEdgesChange,
    setEdgeStyle: state.setEdgeStyle,
    addEdgePoint: state.addEdgePoint,
    updateEdgePoint: state.updateEdgePoint,
    removeEdgePoint: state.removeEdgePoint,
    toggleEdgeStyleForEdge: state.toggleEdgeStyleForEdge,
    toggleNodeCollapse: state.toggleNodeCollapse,
    toggleNodeReversed: state.toggleNodeReversed,
    setAllNodesCollapsed: state.setAllNodesCollapsed,
  }),
  subgraphNavigation: (state: GraphStoreState) => ({
    graphStack: state.graphStack,
    currentGraphLabel: state.currentGraphLabel,
    currentContext: state.currentContext,
    lastSubgraphError: state.lastSubgraphError,
    enterSubgraph: state.enterSubgraph,
    wrapInParentGraph: state.wrapInParentGraph,
    exitToBreadcrumb: state.exitToBreadcrumb,
    captureGraphStackPath: state.captureGraphStackPath,
  }),
  selection: (state: GraphStoreState) => ({
    selectedTapId: state.selectedTapId,
    selectedEdgeId: state.selectedEdgeId,
    pendingStateMerge: state.pendingStateMerge,
    setSelectedNode: state.setSelectedNode,
    setSelectedTap: state.setSelectedTap,
    setSelectedEdge: state.setSelectedEdge,
    clearSelection: state.clearSelection,
    selectAll: state.selectAll,
    confirmStateMerge: state.confirmStateMerge,
    cancelStateMerge: state.cancelStateMerge,
  }),
  history: (state: GraphStoreState) => ({
    past: state.past,
    future: state.future,
    undo: state.undo,
    redo: state.redo,
  }),
  registryPersistence: (state: GraphStoreState) => ({
    graphId: state.graphId,
    isDirty: state.isDirty,
    lastSavedAt: state.lastSavedAt,
    hydrateGraph: state.hydrateGraph,
    capturePersistedGraph: state.capturePersistedGraph,
    restoreSnapshot: state.restoreSnapshot,
    markSaved: state.markSaved,
    markDirty: state.markDirty,
    setCompositeTypes: state.setCompositeTypes,
    setComponentRegistry: state.setComponentRegistry,
  }),
} as const;

const initial = createInitialGraph();

function capturePersistedGraphFromState(state: GraphStoreState): PersistableGraphSnapshot {
  if (state.graphStack.length === 0) {
    return {
      graph: state.graph as GraphSpec,
      uiState: state.uiState,
      graphStackPath: [],
    };
  }

  let childGraph: EditableGraphSpec = isAcausalGraphSpec(state.graph)
    ? state.graph
    : deriveSubgraphPorts(state.graph);
  let childUi = normalizeUiState(childGraph, state.uiState, state.edgeStyle);
  for (let i = state.graphStack.length - 1; i >= 0; i -= 1) {
    const layer = state.graphStack[i];
    if (layer.persistInterior === false) {
      childGraph = layer.graph;
      childUi = normalizeUiState(layer.graph, layer.uiState, state.edgeStyle);
      continue;
    }
    const childId = layer.childNodeId;
    if (!childId) {
      throw new Error(
        `Cannot persist nested graph: stack layer ${i} does not identify its parent node.`
      );
    }
    if (!layer.graph.nodes[childId]) {
      throw new Error(
        `Cannot persist nested graph: parent graph no longer contains subgraph node "${childId}".`
      );
    }
    const nextGraph: EditableGraphSpec = isAcausalGraphSpec(layer.graph)
      ? {
          ...layer.graph,
          subgraphs: {
            ...(layer.graph.subgraphs ?? {}),
            [childId]: childGraph as AcausalGraphSpec,
          },
        }
      : {
          ...layer.graph,
          nodes: {
            ...layer.graph.nodes,
            [childId]: {
              ...layer.graph.nodes[childId],
              ...(!isAcausalGraphSpec(childGraph)
                ? {
                    input_ports: childGraph.input_ports,
                    output_ports: childGraph.output_ports,
                  }
                : {}),
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
    childGraph = nextGraph;
    childUi = normalizeUiState(nextGraph, nextUi, state.edgeStyle);
  }

  return {
    graph: childGraph as GraphSpec,
    uiState: childUi,
    graphStackPath: captureGraphStackPathFromState(state),
  };
}

function captureGraphStackPathFromState(state: Pick<GraphStoreState, 'graphStack'>): string[] {
  return state.graphStack
    .filter((layer) => layer.persistInterior !== false)
    .map((layer) => layer.childNodeId)
    .filter((nodeId): nodeId is string => Boolean(nodeId));
}

function graphLayerKey(path: string[]) {
  return JSON.stringify(path);
}

function emptyLayerHistory(): LayerHistory {
  return { past: [], future: [] };
}

function activeGraphLayerPath(state: Pick<GraphStoreState, 'graphStack'>) {
  return captureGraphStackPathFromState(state);
}

function historyForPath(
  graphHistory: Record<string, LayerHistory>,
  path: string[]
): LayerHistory {
  return graphHistory[graphLayerKey(path)] ?? emptyLayerHistory();
}

function graphHistoryWithActiveLayer(
  state: Pick<GraphStoreState, 'graphStack' | 'graphHistory' | 'past' | 'future'>
): Record<string, LayerHistory> {
  if (state.graphStack[state.graphStack.length - 1]?.persistInterior === false) {
    return state.graphHistory;
  }
  return {
    ...state.graphHistory,
    [graphLayerKey(activeGraphLayerPath(state))]: {
      past: state.past,
      future: state.future,
    },
  };
}

function restoreGraphStackPathFromRoot({
  graph,
  uiState,
  graphId,
  graphStackPath,
  rootLabel,
  edgeStyle,
  componentRegistry,
}: {
  graph: GraphSpec;
  uiState: GraphUIState;
  graphId: string | null;
  graphStackPath: string[];
  rootLabel: string;
  edgeStyle: 'bezier' | 'elbow';
  componentRegistry: Map<string, ComponentDefinition>;
}): Pick<
  GraphStoreState,
  'graph' | 'uiState' | 'nodes' | 'edges' | 'graphStack' | 'currentGraphLabel' | 'currentContext'
> {
  if (graphStackPath.length === 0) {
    return {
      graph,
      uiState,
      nodes: buildNodes(graph, uiState),
      edges: buildEdges(graph, uiState, edgeStyle),
      graphStack: [],
      currentGraphLabel: rootLabel,
      currentContext: 'top-level',
    };
  }

  const graphStack: GraphLayer[] = [];
  let parentGraph: EditableGraphSpec = graph;
  let parentUi = uiState;
  let parentLabel = rootLabel;
  let currentContext = 'top-level';

  for (const nodeId of graphStackPath) {
    const nodeSpec = parentGraph.nodes[nodeId];
    if (!nodeSpec) {
      throw new Error(
        `Cannot restore nested graph path: parent graph no longer contains node "${nodeId}".`
      );
    }
    const childGraph = parentGraph.subgraphs?.[nodeId];
    if (!childGraph) {
      throw new Error(
        `Cannot restore nested graph path: node "${nodeId}" has no saved subgraph.`
      );
    }
    const childUi = normalizeUiState(
      childGraph,
      parentUi.subgraph_states?.[nodeId] ?? { viewport: DEFAULT_VIEWPORT, node_states: {} },
      edgeStyle
    );
    const context = interiorDomainForNode(nodeSpec, componentRegistry) ?? undefined;
    graphStack.push({
      graph: parentGraph as GraphSpec,
      uiState: parentUi,
      graphId,
      label: parentLabel,
      childNodeId: nodeId,
      contextType: context,
    });
    parentGraph = childGraph;
    parentUi = childUi;
    parentLabel = nodeId;
    currentContext = context ?? currentContext;
  }

  return {
    graph: parentGraph as GraphSpec,
    uiState: parentUi,
    nodes: buildNodes(parentGraph, parentUi),
    edges: buildEdges(parentGraph, parentUi, edgeStyle),
    graphStack,
    currentGraphLabel: parentLabel,
    currentContext,
  };
}

export function createGraphSnapshotFromPersistedGraph({
  graph,
  uiState,
  graphId,
  saveRevision,
  label,
  graphStackPath,
  edgeStyle = DEFAULT_EDGE_STYLE,
}: {
  graph: GraphSpec;
  uiState: GraphUIState | null;
  graphId: string | null;
  saveRevision?: number | null;
  label: string;
  graphStackPath?: string[] | null;
  edgeStyle?: 'bezier' | 'elbow';
}): GraphSnapshot {
  const migrated = normalizeGraphForStudioAuthoring(graph);
  const normalized = normalizeUiState(migrated, uiState, edgeStyle);
  const rootLabel = label || migrated.metadata?.name || 'Untitled';
  const restored = restoreGraphStackPathFromRoot({
    graph: migrated,
    uiState: normalized,
    graphId,
    graphStackPath: graphStackPath ?? [],
    rootLabel,
    edgeStyle,
    componentRegistry: new Map<string, ComponentDefinition>(),
  });
  return {
    graph: restored.graph as GraphSpec,
    uiState: restored.uiState,
    graphId,
    saveRevision: saveRevision ?? null,
    isDirty: false,
    lastSavedAt: null,
    graphStack: restored.graphStack,
    currentGraphLabel: restored.currentGraphLabel,
    currentContext: restored.currentContext,
    edgeStyle,
    past: [],
    future: [],
    selectedTapId: null,
    selectedEdgeId: null,
    pendingStateMerge: null,
  };
}

export const useGraphStore = create<GraphStoreState>((set, get) => ({
  graphId: null,
  saveRevision: null,
  graph: initial.graph,
  uiState: initial.uiState,
  nodes: buildNodes(initial.graph, initial.uiState),
  edges: buildEdges(initial.graph, initial.uiState, DEFAULT_EDGE_STYLE),
  edgeStyle: DEFAULT_EDGE_STYLE,
  graphStack: [],
  currentGraphLabel: initial.graph.metadata?.name ?? 'Model',
  _compositeTypes: new Set<string>(),
  _componentRegistry: new Map<string, ComponentDefinition>(),
  _isRegistryLoaded: false,
  currentContext: 'top-level',
  isDirty: false,
  lastSavedAt: null,
  lastSubgraphError: null,
  past: [],
  future: [],
  graphHistory: {},
  selectedTapId: null,
  selectedEdgeId: null,
  pendingStateMerge: null,
  hydrateGraph: (graph, uiState, graphId, graphStackPath = [], saveRevision = null) => {
    const edgeStyle = get().edgeStyle;
    const migrated = normalizeGraphForStudioAuthoring(graph);
    const normalized = normalizeUiState(migrated, uiState, edgeStyle);
    const restored = restoreGraphStackPathFromRoot({
      graph: migrated,
      uiState: normalized,
      graphId: graphId ?? null,
      graphStackPath: graphStackPath ?? [],
      rootLabel: migrated.metadata?.name ?? 'Model',
      edgeStyle,
      componentRegistry: get()._componentRegistry,
    });
    set({
      graphId: graphId ?? null,
      saveRevision,
      graph: restored.graph,
      uiState: restored.uiState,
      nodes: restored.nodes,
      edges: restored.edges,
      graphStack: restored.graphStack,
      currentGraphLabel: restored.currentGraphLabel,
      currentContext: restored.currentContext,
      isDirty: false,
      lastSubgraphError: null,
      past: [],
      future: [],
      graphHistory: {},
      selectedTapId: null,
      selectedEdgeId: null,
      pendingStateMerge: null,
    });
  },
  capturePersistedGraph: () => capturePersistedGraphFromState(get()),
  captureGraphStackPath: () => captureGraphStackPathFromState(get()),
  restoreSnapshot: (snapshot) => {
    const { edgeStyle } = snapshot;
    const graph = normalizeGraphForStudioAuthoring(snapshot.graph);
    const normalized = normalizeUiState(graph, snapshot.uiState, edgeStyle);
    set({
      graphId: snapshot.graphId,
      saveRevision: snapshot.saveRevision ?? null,
      graph,
      uiState: normalized,
      nodes: buildNodes(graph, normalized),
      edges: buildEdges(graph, normalized, edgeStyle),
      edgeStyle,
      graphStack: snapshot.graphStack,
      currentGraphLabel: snapshot.currentGraphLabel,
      currentContext: snapshot.currentContext,
      isDirty: snapshot.isDirty,
      lastSavedAt: snapshot.lastSavedAt,
      lastSubgraphError: null,
      past: snapshot.past,
      future: snapshot.future,
      graphHistory: {},
      selectedTapId: snapshot.selectedTapId,
      selectedEdgeId: snapshot.selectedEdgeId,
      pendingStateMerge: snapshot.pendingStateMerge,
    });
  },
  markSaved: (graphId, saveRevision) => {
    set({
      graphId,
      ...(saveRevision !== undefined ? { saveRevision } : {}),
      isDirty: false,
      lastSavedAt: new Date().toISOString(),
    });
  },
  markDirty: () => {
    set({ isDirty: true });
  },
  resetGraph: () => {
    const fresh = createInitialGraph();
    set({
      graphId: null,
      saveRevision: null,
      graph: fresh.graph,
      uiState: fresh.uiState,
      nodes: buildNodes(fresh.graph, fresh.uiState),
      edges: buildEdges(fresh.graph, fresh.uiState, DEFAULT_EDGE_STYLE),
      graphStack: [],
      currentGraphLabel: fresh.graph.metadata?.name ?? 'Model',
      currentContext: 'top-level',
      isDirty: false,
      lastSavedAt: null,
      lastSubgraphError: null,
      past: [],
      future: [],
      graphHistory: {},
      selectedTapId: null,
      selectedEdgeId: null,
      pendingStateMerge: null,
    });
  },
  undo: () => {
    set((state) => {
      if (state.past.length === 0) return state;
      const previous = state.past[state.past.length - 1];
      const graph = isAcausalGraphSpec(previous.graph)
        ? previous.graph
        : normalizeDynamicPorts(previous.graph);
      const past = state.past.slice(0, -1);
      const future = [cloneSnapshot(state.graph, state.uiState), ...state.future];
      const normalized = normalizeUiState(graph, previous.uiState, state.edgeStyle);
      return {
        ...state,
        graph: graph as GraphSpec,
        uiState: normalized,
        nodes: reconcileNodes(state.nodes, graph, normalized),
        edges: reconcileEdges(state.edges, graph, normalized, state.edgeStyle),
        past,
        future,
        isDirty: true,
        selectedTapId: null,
        selectedEdgeId: null,
      };
    });
  },
  redo: () => {
    set((state) => {
      if (state.future.length === 0) return state;
      const next = state.future[0];
      const graph = isAcausalGraphSpec(next.graph)
        ? next.graph
        : normalizeDynamicPorts(next.graph);
      const future = state.future.slice(1);
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const normalized = normalizeUiState(graph, next.uiState, state.edgeStyle);
      return {
        ...state,
        graph: graph as GraphSpec,
        uiState: normalized,
        nodes: reconcileNodes(state.nodes, graph, normalized),
        edges: reconcileEdges(state.edges, graph, normalized, state.edgeStyle),
        past,
        future,
        isDirty: true,
        selectedTapId: null,
        selectedEdgeId: null,
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
      const componentDef = nodeSpec ? get()._componentRegistry.get(nodeSpec.type) : undefined;
      const isComposite = Boolean(componentDef?.is_composite);
      const isReadOnlyInspector =
        Boolean(componentDef?.interior_domain) &&
        componentDef?.template_kind === 'display' &&
        !componentDef?.template_graph &&
        !hasSubgraph;
      if (!nodeSpec || (!isComposite && !hasSubgraph && !isReadOnlyInspector)) {
        return state;
      }
      const context = interiorDomainForNode(nodeSpec, get()._componentRegistry);
      if (!context) {
        return {
          lastSubgraphError:
            `Cannot open "${nodeId}" because component domain metadata is still loading.`,
        };
      }
      const parentLabel = state.currentGraphLabel || state.graph.metadata?.name || 'Model';
      if (isReadOnlyInspector) {
        const baseInspectorGraph = createBlankGraph();
        const inspectorGraph = {
          ...baseInspectorGraph,
          metadata: {
            ...baseInspectorGraph.metadata,
            name: nodeId,
          },
        };
        const inspectorUi = normalizeUiState(
          inspectorGraph,
          { viewport: DEFAULT_VIEWPORT, node_states: {} },
          state.edgeStyle
        );
        const graphHistory = graphHistoryWithActiveLayer(state);
        return {
          graphStack: [
            ...state.graphStack,
            {
              graph: state.graph,
              uiState: state.uiState,
              graphId: state.graphId,
              label: parentLabel,
              childNodeId: nodeId,
              contextType: context,
              persistInterior: false,
            },
          ],
          graph: inspectorGraph,
          uiState: inspectorUi,
          nodes: [],
          edges: [],
          currentGraphLabel: nodeId,
          currentContext: context,
          past: [],
          future: [],
          graphHistory,
        };
      }
      const cachedGraph = state.graph.subgraphs?.[nodeId];
      const cachedUi = state.uiState.subgraph_states?.[nodeId];

      // Bug d5e8b8f: Only derive ports for freshly-created subgraphs.
      // Cached subgraphs already have user-customized bindings/ports.
      let derivedNext: EditableGraphSpec;
      let nextUiState: GraphUIState;
      if (cachedGraph) {
        derivedNext = cachedGraph;
        nextUiState = cachedUi ?? { viewport: DEFAULT_VIEWPORT, node_states: {} };
      } else {
        if (!get()._isRegistryLoaded) {
          return {
            lastSubgraphError:
              `Cannot open "${nodeId}" because component templates are still loading.`,
          };
        }
        if (!componentDef?.template_graph) {
          return {
            lastSubgraphError:
              `${nodeSpec.type} node "${nodeId}" has no backend template_graph; `
              + 'Studio cannot synthesize a subgraph.',
          };
        }
        const templateGraph = cloneGraphSpec(componentDef.template_graph) as EditableGraphSpec;
        derivedNext = isAcausalGraphSpec(templateGraph)
          ? templateGraph
          : deriveSubgraphPorts(templateGraph);
        nextUiState = cloneGraphSpec(
          componentDef.template_ui_state ?? { viewport: DEFAULT_VIEWPORT, node_states: {} }
        );
      }
      const normalized = normalizeUiState(derivedNext, nextUiState, state.edgeStyle);
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
      const graphHistory = graphHistoryWithActiveLayer(state);
      const nextHistory = historyForPath(graphHistory, [
        ...activeGraphLayerPath(state),
        nodeId,
      ]);
      return {
        graphStack: [
          ...state.graphStack,
          {
            graph: parentGraph,
            uiState: parentUi,
            graphId: state.graphId,
            label: parentLabel,
            childNodeId: nodeId,
            contextType: context,
          },
        ],
        graph: derivedNext as GraphSpec,
        uiState: normalized,
        nodes: reconcileNodes(state.nodes, derivedNext, normalized),
        edges: reconcileEdges(state.edges, derivedNext, normalized, state.edgeStyle),
        currentGraphLabel: nodeId,
        currentContext: context,
        lastSubgraphError: null,
        past: nextHistory.past,
        future: nextHistory.future,
        graphHistory,
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
        nodes: reconcileNodes(state.nodes, derivedCurrent, normalizedCurrent),
        edges: reconcileEdges(state.edges, derivedCurrent, normalizedCurrent, state.edgeStyle),
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
      const derivedCurrent: EditableGraphSpec = isAcausalGraphSpec(state.graph)
        ? state.graph
        : deriveSubgraphPorts(state.graph);
      let childGraph: EditableGraphSpec = derivedCurrent;
      let childUi = normalizeUiState(derivedCurrent, state.uiState, state.edgeStyle);

      const stack = [...state.graphStack];
      for (let i = stack.length - 1; i >= index; i -= 1) {
        const layer = stack[i];
        if (layer.persistInterior === false) {
          childGraph = layer.graph;
          childUi = layer.uiState;
          continue;
        }
        const childId = layer.childNodeId;
        if (!childId) {
          throw new Error(
            `Cannot exit nested graph: stack layer ${i} does not identify its parent node.`
          );
        }
        if (!layer.graph.nodes[childId]) {
          throw new Error(
            `Cannot exit nested graph: parent graph no longer contains subgraph node "${childId}".`
          );
        }
        const nextGraph: EditableGraphSpec = isAcausalGraphSpec(layer.graph)
          ? {
              ...layer.graph,
              subgraphs: {
                ...(layer.graph.subgraphs ?? {}),
                [childId]: childGraph as AcausalGraphSpec,
              },
            }
          : {
              ...layer.graph,
              nodes: {
                ...layer.graph.nodes,
                [childId]: {
                  ...layer.graph.nodes[childId],
                  ...(!isAcausalGraphSpec(childGraph)
                    ? {
                        input_ports: childGraph.input_ports,
                        output_ports: childGraph.output_ports,
                      }
                    : {}),
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
          graph: nextGraph as GraphSpec,
          uiState: nextUi,
        };
        childGraph = nextGraph;
        childUi = nextUi;
      }

      const nextStack = stack.slice(0, index);
      const nextLayer = stack[index];
      const normalized = normalizeUiState(nextLayer.graph, nextLayer.uiState, state.edgeStyle);
      const graphHistory = graphHistoryWithActiveLayer(state);
      const nextHistory = historyForPath(
        graphHistory,
        nextStack.map((layer) => layer.childNodeId).filter((id): id is string => Boolean(id))
      );
      return {
        graphStack: nextStack,
        graph: nextLayer.graph,
        uiState: normalized,
        nodes: reconcileNodes(state.nodes, nextLayer.graph, normalized),
        edges: reconcileEdges(state.edges, nextLayer.graph, normalized, state.edgeStyle),
        graphId: nextLayer.graphId,
        currentGraphLabel: nextLayer.label,
        currentContext: nextStack.length > 0
          ? (nextStack[nextStack.length - 1].contextType ?? 'top-level')
          : 'top-level',
        past: nextHistory.past,
        future: nextHistory.future,
        graphHistory,
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
      const selectedTapIds = selectedNodeIds
        .filter(isTapNodeId)
        .map((id) => id.replace(/^tap:/, ''));
      const selectedComponentIds = selectedNodeIds.filter((id) => !isTapNodeId(id));
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      if (isAcausalGraphSpec(state.graph)) {
        const selectedNodeSet = new Set(selectedComponentIds);
        const remainingConnections = (state.graph.connections ?? []).filter((connection) => {
          const connectionId = `acausal:${[`${connection.a[0]}:${connection.a[1]}`, `${connection.b[0]}:${connection.b[1]}`].sort().join('|')}`;
          return (
            !selectedEdgeIds.includes(connectionId) &&
            !selectedNodeSet.has(connection.a[0]) &&
            !selectedNodeSet.has(connection.b[0])
          );
        });
        const graphNodes = { ...(state.graph.nodes ?? {}) };
        const subgraphs = { ...(state.graph.subgraphs ?? {}) };
        for (const nodeId of selectedComponentIds) {
          delete graphNodes[nodeId];
          delete subgraphs[nodeId];
        }
        const node_states = Object.fromEntries(
          Object.entries(state.uiState.node_states).filter(
            ([nodeId]) => !selectedComponentIds.includes(nodeId)
          )
        ) as GraphUIState['node_states'];
        const subgraph_states = { ...(state.uiState.subgraph_states ?? {}) };
        for (const nodeId of selectedComponentIds) {
          delete subgraph_states[nodeId];
        }
        const graph: AcausalGraphSpec = {
          ...state.graph,
          nodes: graphNodes,
          connections: remainingConnections,
          subgraphs: Object.keys(subgraphs).length ? subgraphs : undefined,
        };
        const uiState: GraphUIState = {
          ...state.uiState,
          node_states,
          subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
        };
        return {
          graph: graph as unknown as GraphSpec,
          uiState,
          nodes: buildNodes(graph, uiState),
          edges: buildEdges(graph, uiState, state.edgeStyle),
          past,
          future: [],
          isDirty: true,
          selectedEdgeId: null,
          selectedTapId: null,
        };
      }
      const nodes = state.nodes.filter((node) => !selectedNodeIds.includes(node.id));
      const tapsToRemove = new Set(selectedTapIds);
      for (const tap of state.graph.taps ?? []) {
        if (
          selectedComponentIds.includes(tap.position.afterNode) ||
          (tap.position.targetNode && selectedComponentIds.includes(tap.position.targetNode))
        ) {
          tapsToRemove.add(tap.id);
        }
      }
      const removedTapNodeIds = new Set([...tapsToRemove].map(tapNodeId));
      const edges = state.edges.filter(
        (edge) =>
          !selectedEdgeIds.includes(edge.id) &&
          !selectedNodeIds.includes(edge.source) &&
          !selectedNodeIds.includes(edge.target) &&
          !removedTapNodeIds.has(edge.source) &&
          !removedTapNodeIds.has(edge.target)
      );
      const graphNodes = { ...state.graph.nodes };
      const subgraphs = { ...(state.graph.subgraphs ?? {}) };
      for (const nodeId of selectedComponentIds) {
        delete graphNodes[nodeId];
        delete subgraphs[nodeId];
      }
      const taps = (state.graph.taps ?? []).filter((tap) => !tapsToRemove.has(tap.id));
      const input_bindings = { ...state.graph.input_bindings };
      const output_bindings = { ...state.graph.output_bindings };
      for (const [key, binding] of Object.entries(input_bindings)) {
        if (selectedComponentIds.includes(binding[0])) {
          delete input_bindings[key];
        }
      }
      for (const [key, binding] of Object.entries(output_bindings)) {
        if (selectedComponentIds.includes(binding[0])) {
          delete output_bindings[key];
        }
      }
      const subgraph_states = { ...(state.uiState.subgraph_states ?? {}) };
      for (const nodeId of selectedComponentIds) {
        delete subgraph_states[nodeId];
      }
      const tap_states = { ...(state.uiState.tap_states ?? {}) };
      for (const tapId of Array.from(tapsToRemove) as string[]) {
        delete tap_states[tapId];
      }
      const uiState = {
        ...state.uiState,
        node_states: Object.fromEntries(
          Object.entries(state.uiState.node_states).filter(([nodeId]) => !selectedComponentIds.includes(nodeId))
        ),
        subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
        tap_states: Object.keys(tap_states).length ? tap_states : undefined,
      };
      let nextGraph: GraphSpec = {
        ...state.graph,
        nodes: graphNodes,
        wires: edgesToWires(edges),
        input_bindings,
        output_bindings,
        taps,
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
        selectedTapId:
          state.selectedTapId && tapsToRemove.has(state.selectedTapId)
            ? null
            : state.selectedTapId,
        selectedEdgeId:
          state.selectedEdgeId && selectedEdgeIds.includes(state.selectedEdgeId)
            ? null
            : state.selectedEdgeId,
      };
    });
  },
  duplicateSelected: () => {
    set((state) => {
      const selectedNodeIds = state.nodes
        .filter((node) => node.selected && !isTapNodeId(node.id))
        .map((node) => node.id)
        .filter((nodeId) => state.graph.nodes[nodeId]);
      if (selectedNodeIds.length === 0) return state;

      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nodeMap: Record<string, string> = {};
      let graphForNames: GraphSpec = state.graph;
      for (const nodeId of selectedNodeIds) {
        const nextId = createNodeName(graphForNames, nodeId);
        nodeMap[nodeId] = nextId;
        graphForNames = {
          ...graphForNames,
          nodes: {
            ...graphForNames.nodes,
            [nextId]: state.graph.nodes[nodeId],
          },
        };
      }

      const nodes = Object.fromEntries(
        Object.entries(state.graph.nodes).map(([nodeId, spec]) => [
          nodeId,
          cloneGraphSpec(spec),
        ])
      ) as GraphSpec['nodes'];
      for (const nodeId of selectedNodeIds) {
        nodes[nodeMap[nodeId]] = cloneGraphSpec(state.graph.nodes[nodeId]);
      }

      const selectedNodeSet = new Set(selectedNodeIds);
      const duplicatedWires = state.graph.wires
        .filter(
          (wire) =>
            selectedNodeSet.has(wire.source_node) && selectedNodeSet.has(wire.target_node)
        )
        .map((wire) => ({
          ...cloneGraphSpec(wire),
          source_node: nodeMap[wire.source_node],
          target_node: nodeMap[wire.target_node],
        }));
      let graph: GraphSpec = {
        ...state.graph,
        nodes,
        wires: [...state.graph.wires.map((wire) => cloneGraphSpec(wire)), ...duplicatedWires],
      };

      const subgraphs = { ...(state.graph.subgraphs ?? {}) };
      const subgraph_states = { ...(state.uiState.subgraph_states ?? {}) };
      for (const nodeId of selectedNodeIds) {
        const nodeSpec = state.graph.nodes[nodeId];
        const targetNodeId = nodeMap[nodeId];
        const isComposite = get()._compositeTypes.has(nodeSpec.type);
        const sourceSubgraph = state.graph.subgraphs?.[nodeId];
        const sourceSubgraphUiState = state.uiState.subgraph_states?.[nodeId];
        if (isComposite && !sourceSubgraph) {
          throw new Error(
            `Cannot duplicate composite node "${nodeId}": source subgraph is missing.`
          );
        }
        if (sourceSubgraph && !sourceSubgraphUiState) {
          throw new Error(
            `Cannot duplicate composite node "${nodeId}": source subgraph UI state is missing.`
          );
        }
        if (sourceSubgraph && sourceSubgraphUiState) {
          subgraphs[targetNodeId] = cloneGraphSpec(sourceSubgraph);
          subgraph_states[targetNodeId] = cloneGraphSpec(sourceSubgraphUiState);
        }
      }
      graph = {
        ...graph,
        subgraphs: Object.keys(subgraphs).length ? subgraphs : undefined,
      };
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }

      const node_states = Object.fromEntries(
        Object.entries(state.uiState.node_states).map(([nodeId, nodeState]) => [
          nodeId,
          { ...nodeState, selected: false },
        ])
      ) as GraphUIState['node_states'];
      for (const nodeId of selectedNodeIds) {
        const targetNodeId = nodeMap[nodeId];
        const sourceNodeState = state.uiState.node_states[nodeId];
        if (!sourceNodeState) {
          throw new Error(
            `Cannot duplicate node "${nodeId}": source node UI state is missing.`
          );
        }
        node_states[targetNodeId] = {
          ...cloneGraphSpec(sourceNodeState),
          position: {
            x: sourceNodeState.position.x + 40,
            y: sourceNodeState.position.y + 40,
          },
          selected: true,
        };
      }

      const tap_states = state.uiState.tap_states
        ? Object.fromEntries(
            Object.entries(state.uiState.tap_states).map(([tapId, tapState]) => [
              tapId,
              { ...tapState, selected: false },
            ])
          )
        : undefined;
      const edgeStateSeed: Record<string, EdgeUIState> = { ...(state.uiState.edge_states ?? {}) };
      for (const sourceWire of state.graph.wires) {
        if (
          !selectedNodeSet.has(sourceWire.source_node) ||
          !selectedNodeSet.has(sourceWire.target_node)
        ) {
          continue;
        }
        const oldWireId = wireId(sourceWire);
        const newWireId = wireId({
          ...sourceWire,
          source_node: nodeMap[sourceWire.source_node],
          target_node: nodeMap[sourceWire.target_node],
        });
        const oldEdgeState = state.uiState.edge_states?.[oldWireId];
        if (oldEdgeState) {
          edgeStateSeed[newWireId] = cloneGraphSpec(oldEdgeState);
        }
      }

      const uiState: GraphUIState = {
        ...state.uiState,
        node_states,
        edge_states: edgeStateSeed,
        subgraph_states: Object.keys(subgraph_states).length ? subgraph_states : undefined,
        tap_states: tap_states && Object.keys(tap_states).length ? tap_states : undefined,
      };
      const edge_states = buildEdgeStates(graph, uiState, state.edgeStyle);
      const normalizedUiState: GraphUIState = {
        ...uiState,
        edge_states,
      };

      return {
        graph,
        uiState: normalizedUiState,
        nodes: buildNodes(graph, normalizedUiState),
        edges: buildEdges(graph, normalizedUiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
        selectedTapId: null,
        selectedEdgeId: null,
        pendingStateMerge: null,
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

      const taps = (state.graph.taps ?? []).map((tap) => {
        const position = { ...tap.position };
        if (position.afterNode === nodeId) {
          position.afterNode = trimmed;
        }
        if (position.targetNode === nodeId) {
          position.targetNode = trimmed;
        }
        return { ...tap, position };
      });

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
        taps,
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
        nodes: reconcileNodes(state.nodes, graph, uiState),
        edges: reconcileEdges(state.edges, graph, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  renameSubgraphBoundaryPort: (nodeId, direction, previousPort, nextPort) => {
    set((state) => {
      const trimmed = nextPort.trim();
      if (!trimmed || trimmed === previousPort) return state;
      const nodeSpec = state.graph.nodes[nodeId];
      const subgraph = state.graph.subgraphs?.[nodeId];
      if (!nodeSpec || !isCausalGraphSpec(subgraph)) return state;

      const renamedSubgraph = renameBoundaryPortInGraph(
        subgraph,
        direction,
        previousPort,
        trimmed
      );
      if (!renamedSubgraph) return state;

      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      let graph: GraphSpec = {
        ...state.graph,
        nodes: {
          ...state.graph.nodes,
          [nodeId]: {
            ...nodeSpec,
            input_ports: renamedSubgraph.input_ports,
            output_ports: renamedSubgraph.output_ports,
          },
        },
        subgraphs: {
          ...(state.graph.subgraphs ?? {}),
          [nodeId]: renamedSubgraph,
        },
      };
      graph = renameNodePortReferences(graph, nodeId, direction, previousPort, trimmed);
      const edge_states = buildEdgeStates(graph, state.uiState, state.edgeStyle);
      const uiState = {
        ...state.uiState,
        edge_states,
      };
      return {
        graph,
        uiState,
        nodes: reconcileNodes(state.nodes, graph, uiState),
        edges: reconcileEdges(state.edges, graph, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  onNodesChange: (changes) => {
    set((state) => {
      const removedRawIds = changes
        .filter((change) => change.type === 'remove' && 'id' in change)
        .map((change) => (change as { id: string }).id);
      const removedTapIds = removedRawIds
        .filter((id) => isTapNodeId(id))
        .map((id) => id.replace(/^tap:/, ''));
      const removedIds = removedRawIds.filter((id) => !isTapNodeId(id));
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

      if (removedTapIds.length > 0) {
        const taps = (graph.taps ?? []).filter((tap) => !removedTapIds.includes(tap.id));
        edges = edges.filter(
          (edge) =>
            !removedTapIds.some((tapId) => edge.source === tapNodeId(tapId) || edge.target === tapNodeId(tapId))
        );
        graph = {
          ...graph,
          taps,
          wires: edgesToWires(edges),
        };
        uiState = {
          ...uiState,
          tap_states: Object.fromEntries(
            Object.entries(uiState.tap_states ?? {}).filter(([id]) => !removedTapIds.includes(id))
          ),
        };
      }
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
        if (state.graphStack.length > 0 && isCausalGraphSpec(graph)) {
          graph = deriveSubgraphPorts(graph);
        }
      }
      if (removedIds.length > 0) {
        const danglingTapIds = (graph.taps ?? [])
          .filter(
            (tap) =>
              removedIds.includes(tap.position.afterNode) ||
              (tap.position.targetNode && removedIds.includes(tap.position.targetNode))
          )
          .map((tap) => tap.id);
        if (danglingTapIds.length > 0) {
          const removedTapNodeIds = new Set(danglingTapIds.map(tapNodeId));
          edges = edges.filter(
            (edge) =>
              !removedTapNodeIds.has(edge.source) && !removedTapNodeIds.has(edge.target)
          );
          graph = {
            ...graph,
            taps: (graph.taps ?? []).filter((tap) => !danglingTapIds.includes(tap.id)),
            wires: edgesToWires(edges),
          };
          const tap_states = { ...(uiState.tap_states ?? {}) };
          for (const tapId of danglingTapIds) {
            delete tap_states[tapId];
          }
          uiState = {
            ...uiState,
            tap_states: Object.keys(tap_states).length ? tap_states : undefined,
          };
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
      const nextNodes = applyNodeChanges<Node<GraphNodeData | TapNodeData>>(
        changes as NodeChange<Node<GraphNodeData | TapNodeData>>[],
        state.nodes
      );
      const node_states = { ...uiState.node_states };
      const tap_states: Record<string, TapUIState> = { ...(uiState.tap_states ?? {}) };
      for (const node of nextNodes) {
        if (isTapNodeId(node.id)) {
          const tapId = node.id.replace(/^tap:/, '');
          tap_states[tapId] = {
            position: node.position,
            selected: node.selected,
          };
          continue;
        }
        const existing = node_states[node.id] ?? {
          position: { x: node.position.x, y: node.position.y },
          collapsed: false,
          selected: false,
          size: undefined,
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
      const updatedNodes = nextNodes.map((node) => {
        if (isTapNodeId(node.id)) {
          return node;
        }
        const size =
          node_states[node.id]?.size ?? (node.data as GraphNodeData).size;
        return {
          ...node,
          data: {
            ...(node.data as GraphNodeData),
            size,
          },
        };
      });
      const dirty = changes.some((change) => {
        if (change.type === 'select') return false;
        if (change.type === 'dimensions' && !(change as { resizing?: boolean }).resizing) return false;
        return true;
      });
      const edge_states = buildEdgeStates(graph, uiState, state.edgeStyle);
      const nextSelectedTapId =
        state.selectedTapId && uiState.tap_states?.[state.selectedTapId]
          ? state.selectedTapId
          : null;
      return {
        graph,
        nodes: updatedNodes,
        edges: applyEdgeStates(edges, edge_states, state.edgeStyle),
        uiState: {
          ...uiState,
          node_states,
          edge_states,
          tap_states,
        },
        past,
        future: shouldRecord ? [] : state.future,
        isDirty: state.isDirty || dirty,
        selectedTapId: nextSelectedTapId,
      };
    });
  },
  onEdgesChange: (changes) => {
    set((state) => {
      const shouldRecord = changes.some((change) => {
        if (change.type === 'select') return false;
        return true;
      });
      const past = shouldRecord
        ? [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY)
        : state.past;
      const nextEdges = applyEdgeChanges<Edge<GraphEdgeData>>(
        changes as EdgeChange<Edge<GraphEdgeData>>[],
        state.edges
      );
      if (isAcausalGraphSpec(state.graph)) {
        const graph: AcausalGraphSpec = {
          ...state.graph,
          connections: acausalConnectionsFromEdges(nextEdges),
        };
        const selectedEdgeId = nextEdges.find((edge) => edge.selected)?.id ?? null;
        return {
          graph: graph as unknown as GraphSpec,
          edges: reconcileEdges(state.edges, graph, state.uiState, state.edgeStyle),
          past,
          future: shouldRecord ? [] : state.future,
          isDirty: state.isDirty || changes.length > 0,
          selectedEdgeId,
        };
      }
      let graph: GraphSpec = {
        ...state.graph,
        wires: edgesToWires(nextEdges),
      };
      graph = normalizeDynamicPorts(graph);
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }
      const edge_states = buildEdgeStates(graph, state.uiState, state.edgeStyle);
      const dirty = changes.length > 0;
      const selectedEdgeId = nextEdges.find((edge) => edge.selected)?.id ?? null;
      return {
        graph,
        nodes: reconcileNodes(state.nodes, graph, state.uiState),
        edges: reconcileById(
          state.edges,
          applyEdgeStates(buildEdges(graph, state.uiState, state.edgeStyle), edge_states, state.edgeStyle),
          sameGraphEdge
        ),
        uiState: {
          ...state.uiState,
          edge_states,
        },
        past,
        future: shouldRecord ? [] : state.future,
        isDirty: state.isDirty || dirty,
        selectedEdgeId,
      };
    });
  },
  onConnect: (connection, styleOverride, paramUpdates = [], wireOptions) => {
    if (!connection.source || !connection.target) return;
    if (!connection.sourceHandle || !connection.targetHandle) return;
    if (isAcausalGraphSpec(get().graph)) {
      const acausalConnection = connectionFromReactFlow(connection);
      if (!acausalConnection) return;
      set((state) => {
        if (!isAcausalGraphSpec(state.graph)) return state;
        const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
        const graph: AcausalGraphSpec = {
          ...state.graph,
          connections: mergeAcausalConnection(state.graph.connections ?? [], acausalConnection),
        };
        return {
          graph: graph as unknown as GraphSpec,
          edges: reconcileEdges(state.edges, graph, state.uiState, state.edgeStyle),
          past,
          future: [],
          isDirty: true,
        };
      });
      return;
    }
    const isState =
      connection.sourceHandle.startsWith('__state') ||
      connection.targetHandle.startsWith('__state');
    if (isState) {
      if (connection.sourceHandle !== '__state_out' || connection.targetHandle !== '__state_in') {
        return;
      }
      if (connection.source === connection.target) {
        return;
      }
      set((state) => {
        const request = buildStateMergeRequest(state.graph, connection.source!, connection.target!);
        if (!request) return state;
        const autoMappings = Object.entries(request.suggested).filter(([, value]) => value);
        if (!request.hasExistingConnections && autoMappings.length > 0) {
          const mapping = Object.fromEntries(autoMappings) as Record<string, string>;
          const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
          let graph = applyStateMerge(state.graph, request.sourceNode, request.targetNode, mapping);
          if (state.graphStack.length > 0) {
            graph = deriveSubgraphPorts(graph);
          }
          const edge_states = buildEdgeStates(graph, state.uiState, state.edgeStyle);
          return {
            graph,
            nodes: reconcileNodes(state.nodes, graph, state.uiState),
            edges: reconcileEdges(state.edges, graph, state.uiState, state.edgeStyle),
            uiState: {
              ...state.uiState,
              edge_states,
            },
            past,
            future: [],
            isDirty: true,
            pendingStateMerge: null,
            selectedEdgeId: null,
          };
        }
        return {
          pendingStateMerge: request,
          selectedEdgeId: null,
        };
      });
      return;
    }
    const alreadyUsed = get().edges.some(
      (edge) =>
        edge.target === connection.target &&
        edge.targetHandle === connection.targetHandle
    );
    if (alreadyUsed) return;
    const edgeStyle = styleOverride ?? get().edgeStyle;
    const temporality =
      wireOptions?.temporality ??
      (closesInstantCycle(get().graph, connection.source, connection.target)
        ? 'recurrent'
        : 'instant');
    const recurrent_initializer =
      temporality === 'recurrent'
        ? (wireOptions?.recurrent_initializer ?? recurrentZeroInitializer())
        : null;
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
        temporality,
        recurrent_initializer,
      },
    };
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextEdges = addEdge(edge, state.edges);
      let graph: GraphSpec = {
        ...state.graph,
        wires: edgesToWires(nextEdges),
      };
      graph = expandMuxForPort(graph, connection.target!, connection.targetHandle!);
      graph = normalizeDynamicPorts(graph);
      graph = applyNodeParamUpdatesToGraph(graph, paramUpdates);
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
        graph,
        nodes: reconcileNodes(state.nodes, graph, state.uiState),
        edges: reconcileById(
          state.edges,
          applyEdgeStates(buildEdges(graph, state.uiState, state.edgeStyle), edge_states, state.edgeStyle),
          sameGraphEdge
        ),
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
      if (component.template_graph) {
        const imported = importTemplateGraphIntoGraph(
          state.graph,
          state.uiState,
          component.template_graph,
          component.template_ui_state,
          position,
          component.template_id ?? component.name
        );
        let graph = imported.graph;
        if (state.graphStack.length > 0 && isCausalGraphSpec(graph)) {
          graph = deriveSubgraphPorts(graph);
        }
        const edge_states = buildEdgeStates(graph, imported.uiState, state.edgeStyle);
        const uiState: GraphUIState = {
          ...imported.uiState,
          edge_states,
        };
        const importedIds = new Set(imported.importedNodeIds);
        const nodes = buildNodes(graph, uiState).map((node) => ({
          ...node,
          selected: importedIds.has(node.id),
        }));
        return {
          graph,
          uiState,
          nodes,
          edges: buildEdges(graph, uiState, state.edgeStyle),
          past,
          future: [],
          isDirty: true,
        };
      }

      const name = createNodeName(state.graph, component.name);
      let spec: ComponentSpec = {
        type: component.name,
        params: { ...component.default_params },
        input_ports: component.input_ports,
        output_ports: component.output_ports,
      };
      spec = normalizeMuxSpec(
        spec,
        Number(spec.params.n_inputs) || spec.input_ports.length || 2
      );
      let graph: EditableGraphSpec = {
        ...state.graph,
        nodes: {
          ...(state.graph.nodes ?? {}),
          [name]: spec,
        },
      };
      if (state.graphStack.length > 0 && !isAcausalGraphSpec(graph)) {
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
        subgraph_states: state.uiState.subgraph_states,
      };
      const nodes = buildNodes(graph, uiState).map((node) => ({
        ...node,
        selected: node.id === name,
      }));
      return {
        graph,
        uiState,
        nodes,
        edges: buildEdges(graph, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  updateNodeParams: (nodeId, paramName, value, taskBindingSpec) => {
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
      const nextSpec = normalizeMuxSpec(
        updatedSpec,
        paramName === 'n_inputs' ? Number(value) : updatedSpec.input_ports.length
      );
      let graph: GraphSpec = {
        ...state.graph,
        nodes: {
          ...state.graph.nodes,
          [nodeId]: nextSpec,
        },
      };
      graph = normalizeDynamicPorts(graph, taskBindingSpec);
      return {
        graph,
        nodes: reconcileNodes(state.nodes, graph, state.uiState),
        edges: reconcileEdges(state.edges, graph, state.uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  updateNodeParamsBatch: (updates, taskBindingSpec) => {
    if (updates.length === 0) return;
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      let graph = applyNodeParamUpdatesToGraph(state.graph, updates);
      graph = normalizeDynamicPorts(graph, taskBindingSpec);
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }
      return {
        graph,
        nodes: reconcileNodes(state.nodes, graph, state.uiState),
        edges: reconcileEdges(state.edges, graph, state.uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  setSelectedNode: (nodeId) => {
    set((state) => {
      const nodes = setNodeSelection(state.nodes, nodeId);
      const node_states = { ...state.uiState.node_states };
      for (const node of nodes) {
        if (isTapNodeId(node.id)) continue;
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
      const tap_states = { ...(state.uiState.tap_states ?? {}) };
      for (const node of nodes) {
        if (!isTapNodeId(node.id)) continue;
        const tapId = node.id.replace(/^tap:/, '');
        const existing = tap_states[tapId] ?? { position: node.position };
        tap_states[tapId] = {
          ...existing,
          selected: false,
        };
      }
      return {
        nodes,
        uiState: {
          ...state.uiState,
          node_states,
          tap_states,
        },
        selectedTapId: null,
        selectedEdgeId: null,
        pendingStateMerge: null,
      };
    });
  },
  setSelectedTap: (tapId) => {
    set((state) => {
      const targetId = tapId ? tapNodeId(tapId) : null;
      const nodes = setNodeSelection(state.nodes, targetId);
      const node_states = { ...state.uiState.node_states };
      for (const node of nodes) {
        if (isTapNodeId(node.id)) continue;
        const existing = node_states[node.id] ?? {
          position: node.position,
          collapsed: false,
          selected: false,
        };
        node_states[node.id] = {
          ...existing,
          selected: false,
        };
      }
      const tap_states = { ...(state.uiState.tap_states ?? {}) };
      if (tapId) {
        const node = nodes.find((item) => item.id === targetId);
        const position = node?.position ?? DEFAULT_POSITION;
        const existing = tap_states[tapId] ?? { position };
        tap_states[tapId] = { ...existing, position, selected: true };
      }
      return {
        nodes,
        uiState: {
          ...state.uiState,
          node_states,
          tap_states,
        },
        selectedTapId: tapId,
        selectedEdgeId: null,
        pendingStateMerge: null,
      };
    });
  },
  setSelectedEdge: (edgeId) => {
    set((state) => {
      const edges = setEdgeSelection(state.edges, edgeId);
      return {
        edges,
        selectedEdgeId: edgeId,
        pendingStateMerge: null,
      };
    });
  },
  clearSelection: () => {
    set((state) => {
      const nodes = state.nodes.map((node) =>
        node.selected ? { ...node, selected: false } : node
      );
      const edges = state.edges.map((edge) =>
        edge.selected ? { ...edge, selected: false } : edge
      );
      const node_states = Object.fromEntries(
        Object.entries(state.uiState.node_states).map(([nodeId, nodeState]) => [
          nodeId,
          { ...nodeState, selected: false },
        ])
      ) as GraphUIState['node_states'];
      const tap_states = state.uiState.tap_states
        ? Object.fromEntries(
            Object.entries(state.uiState.tap_states).map(([tapId, tapState]) => [
              tapId,
              { ...tapState, selected: false },
            ])
          )
        : undefined;
      return {
        nodes,
        edges,
        uiState: {
          ...state.uiState,
          node_states,
          tap_states,
        },
        selectedTapId: null,
        selectedEdgeId: null,
        pendingStateMerge: null,
      };
    });
  },
  selectAll: () => {
    set((state) => {
      const nodes = state.nodes.map((node) => ({ ...node, selected: true }));
      const edges = state.edges.map((edge) => ({ ...edge, selected: true }));
      const node_states = Object.fromEntries(
        Object.entries(state.uiState.node_states).map(([nodeId, nodeState]) => [
          nodeId,
          { ...nodeState, selected: true },
        ])
      ) as GraphUIState['node_states'];
      const tap_states = state.uiState.tap_states
        ? Object.fromEntries(
            Object.entries(state.uiState.tap_states).map(([tapId, tapState]) => [
              tapId,
              { ...tapState, selected: true },
            ])
          )
        : undefined;
      return {
        nodes,
        edges,
        uiState: {
          ...state.uiState,
          node_states,
          tap_states,
        },
        selectedTapId: null,
        selectedEdgeId: null,
        pendingStateMerge: null,
      };
    });
  },
  toggleNodeCollapse: (nodeId) => {
    set((state) => {
      const nodeState = state.uiState.node_states[nodeId];
      if (!nodeState) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextCollapsed = !nodeState.collapsed;
      const uiState: GraphUIState = {
        ...state.uiState,
        node_states: {
          ...state.uiState.node_states,
          [nodeId]: {
            ...nodeState,
            collapsed: nextCollapsed,
            size: undefined,
          },
        },
      };
      const nodes = state.nodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              data: {
                ...node.data,
                collapsed: nextCollapsed,
                size: undefined,
              },
            }
          : node
      );
      return {
        uiState,
        nodes,
        edges: buildEdges(state.graph, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  toggleNodeReversed: (nodeId) => {
    set((state) => {
      const nodeState = state.uiState.node_states[nodeId];
      if (!nodeState) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const nextReversed = !nodeState.reversed;
      const uiState: GraphUIState = {
        ...state.uiState,
        node_states: {
          ...state.uiState.node_states,
          [nodeId]: {
            ...nodeState,
            reversed: nextReversed,
          },
        },
      };
      const nodes = state.nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, reversed: nextReversed } }
          : node
      );
      return {
        uiState,
        nodes,
        edges: buildEdges(state.graph, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  setAllNodesCollapsed: (collapsed) => {
    set((state) => {
      const shouldUpdate = Object.values(state.uiState.node_states).some(
        (nodeState) => nodeState.collapsed !== collapsed || nodeState.size !== undefined
      );
      if (!shouldUpdate) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const node_states = { ...state.uiState.node_states };
      for (const nodeId of Object.keys(node_states)) {
        node_states[nodeId] = {
          ...node_states[nodeId],
          collapsed,
          size: undefined,
        };
      }
      const nodes = state.nodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          collapsed,
          size: undefined,
        },
      }));
      return {
        uiState: {
          ...state.uiState,
          node_states,
        },
        nodes,
        edges: buildEdges(state.graph, { ...state.uiState, node_states }, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  addTapForEdge: (edgeId, type) => {
    set((state) => {
      const edge = state.edges.find((item) => item.id === edgeId);
      if (!edge || edge.type !== 'state-flow') return state;
      if (!edge.source || !edge.target) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const taps = [...(state.graph.taps ?? [])];
      const id = createTapId();
      taps.push({
        id,
        type,
        position: { afterNode: edge.source, targetNode: edge.target },
        paths: {},
      });
      const uiState: GraphUIState = {
        ...state.uiState,
        tap_states: {
          ...(state.uiState.tap_states ?? {}),
          [id]: {
            position: computeTapPosition({ ...state.graph, taps }, state.uiState, {
              id,
              type,
              position: { afterNode: edge.source, targetNode: edge.target },
              paths: {},
            }),
            selected: true,
          },
        },
      };
      return {
        graph: {
          ...state.graph,
          taps,
        },
        uiState,
        nodes: buildNodes({ ...state.graph, taps }, uiState).map((node) => ({
          ...node,
          selected: node.id === tapNodeId(id),
        })),
        edges: buildEdges({ ...state.graph, taps }, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
        selectedTapId: id,
        selectedEdgeId: null,
      };
    });
  },
  addTap: (afterNode, type) => {
    set((state) => {
      if (!state.graph.nodes[afterNode]) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const taps = [...(state.graph.taps ?? [])];
      const id = createTapId();
      const targetNode = state.graph.wires.find(
        (wire) => wire.source_node === afterNode && state.graph.nodes[wire.target_node]
      )?.target_node;
      taps.push({
        id,
        type,
        position: { afterNode, targetNode },
        paths: {},
      });
      const uiState: GraphUIState = {
        ...state.uiState,
        tap_states: {
          ...(state.uiState.tap_states ?? {}),
          [id]: {
            position: computeTapPosition(state.graph, state.uiState, {
              id,
              type,
              position: { afterNode },
              paths: {},
            }),
            selected: true,
          },
        },
      };
      const nodes = buildNodes({ ...state.graph, taps }, uiState).map((node) => ({
        ...node,
        selected: node.id === tapNodeId(id),
      }));
      return {
        graph: {
          ...state.graph,
          taps,
        },
        uiState,
        nodes,
        edges: buildEdges({ ...state.graph, taps }, uiState, state.edgeStyle),
        past,
        future: [],
        isDirty: true,
        selectedTapId: id,
      };
    });
  },
  updateTap: (tapId, updates) => {
    set((state) => {
      const currentTap = (state.graph.taps ?? []).find((tap) => tap.id === tapId);
      if (!currentTap) return state;
      const nextTap: TapSpec = { ...currentTap, ...updates };
      const taps = (state.graph.taps ?? []).map((tap) => (tap.id === tapId ? nextTap : tap));
      let wires = state.graph.wires;
      if (updates.paths) {
        const prevKeys = new Set(Object.keys(currentTap.paths ?? {}));
        const nextKeys = new Set(Object.keys(nextTap.paths ?? {}));
        const removed = [...prevKeys].filter((key) => !nextKeys.has(key));
        if (removed.length > 0) {
          wires = wires.filter(
            (wire) =>
              !(
                wire.source_node === tapNodeId(tapId) &&
                removed.includes(wire.source_port)
              )
          );
        }
      }
      let uiState = state.uiState;
      if (updates.position) {
        const positionChanged =
          updates.position.afterNode !== currentTap.position.afterNode ||
          updates.position.targetNode !== currentTap.position.targetNode;
        if (positionChanged) {
          const tap_states = { ...(state.uiState.tap_states ?? {}) };
          const nextPosition = computeTapPosition({ ...state.graph, taps }, state.uiState, nextTap);
          tap_states[tapId] = {
            position: nextPosition,
            selected: tap_states[tapId]?.selected ?? false,
          };
          uiState = {
            ...state.uiState,
            tap_states,
          };
        }
      }
      const graph = { ...state.graph, taps, wires };
      const nodes = buildNodes(graph, uiState);
      return {
        graph,
        uiState,
        nodes,
        edges: buildEdges(graph, uiState, state.edgeStyle),
        isDirty: true,
      };
    });
  },
  removeTap: (tapId) => {
    set((state) => {
      const taps = (state.graph.taps ?? []).filter((tap) => tap.id !== tapId);
      const edges = state.edges.filter(
        (edge) =>
          edge.source !== tapNodeId(tapId) && edge.target !== tapNodeId(tapId)
      );
      const tap_states = { ...(state.uiState.tap_states ?? {}) };
      delete tap_states[tapId];
      const graph: GraphSpec = {
        ...state.graph,
        taps,
        wires: edgesToWires(edges),
      };
      const uiState: GraphUIState = {
        ...state.uiState,
        tap_states: Object.keys(tap_states).length ? tap_states : undefined,
      };
      return {
        graph,
        uiState,
        nodes: buildNodes(graph, uiState),
        edges: buildEdges(graph, uiState, state.edgeStyle),
        isDirty: true,
        selectedTapId: state.selectedTapId === tapId ? null : state.selectedTapId,
      };
    });
  },
  addRetainedObservable: (observable) => {
    set((state) => {
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      const observables = [...(state.graph.retained_observables ?? []), observable];
      return {
        graph: {
          ...state.graph,
          retained_observables: observables,
        },
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  updateRetainedObservable: (observableId, updates) => {
    set((state) => {
      const current = (state.graph.retained_observables ?? []).find(
        (observable) => observable.id === observableId
      );
      if (!current) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      return {
        graph: {
          ...state.graph,
          retained_observables: (state.graph.retained_observables ?? []).map((observable) =>
            observable.id === observableId
              ? {
                  ...observable,
                  ...updates,
                  id: observable.id,
                  metadata: {
                    ...observable.metadata,
                    ...(updates.metadata ?? {}),
                  },
                }
              : observable
          ),
        },
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  removeRetainedObservable: (observableId) => {
    set((state) => {
      const current = state.graph.retained_observables ?? [];
      if (!current.some((observable) => observable.id === observableId)) return state;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      return {
        graph: {
          ...state.graph,
          retained_observables: current.filter((observable) => observable.id !== observableId),
        },
        past,
        future: [],
        isDirty: true,
      };
    });
  },
  confirmStateMerge: (mapping) => {
    set((state) => {
      if (!state.pendingStateMerge) return state;
      const { sourceNode, targetNode } = state.pendingStateMerge;
      const past = [...state.past, cloneSnapshot(state.graph, state.uiState)].slice(-MAX_HISTORY);
      let graph = applyStateMerge(state.graph, sourceNode, targetNode, mapping);
      if (state.graphStack.length > 0) {
        graph = deriveSubgraphPorts(graph);
      }
      const edge_states = buildEdgeStates(graph, state.uiState, state.edgeStyle);
      return {
        graph,
        nodes: buildNodes(graph, state.uiState),
        edges: buildEdges(graph, state.uiState, state.edgeStyle),
        uiState: {
          ...state.uiState,
          edge_states,
        },
        past,
        future: [],
        isDirty: true,
        pendingStateMerge: null,
      };
    });
  },
  cancelStateMerge: () => {
    set({ pendingStateMerge: null });
  },
  setCompositeTypes: (types) => {
    set({ _compositeTypes: types });
  },
  setComponentRegistry: (components) => {
    set((state) => {
      const componentRegistry = new Map<string, ComponentDefinition>(
        components.map((c) => [c.name, c])
      );
      const refreshed = refreshLayerContexts(state.graphStack, componentRegistry);
      return {
        _compositeTypes: new Set(components.filter((c) => c.is_composite).map((c) => c.name)),
        _componentRegistry: componentRegistry,
        _isRegistryLoaded: true,
        graphStack: refreshed.graphStack,
        currentContext: refreshed.currentContext,
      };
    });
  },
}));
