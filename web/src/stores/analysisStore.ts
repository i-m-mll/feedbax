/**
 * Zustand store for the analysis DAG graph state.
 *
 * Manages analysis nodes, wires, transforms, and selection state.
 * Mirrors the pattern of graphStore but for the analysis pipeline.
 */

import { create } from 'zustand';
import type { Node, Edge, OnNodesChange, OnEdgesChange, Connection } from '@xyflow/react';
import { applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
import dagre from '@dagrejs/dagre';
import type {
  AnalysisNodeSpec,
  AnalysisWire,
  AnalysisGraphSpec,
  AnalysisClassDef,
  TransformSpec,
} from '@/types/analysis';

// ---------------------------------------------------------------------------
// React Flow data interfaces for analysis nodes/edges
// ---------------------------------------------------------------------------

export interface AnalysisNodeData extends Record<string, unknown> {
  spec: AnalysisNodeSpec;
  label: string;
}

export interface DataSourceNodeData extends Record<string, unknown> {
  label: string;
  outputs: string[];
}

export interface TransformNodeData extends Record<string, unknown> {
  transform: TransformSpec;
  label: string;
}

export interface AnalysisEdgeData extends Record<string, unknown> {
  implicit: boolean;
  transform?: TransformSpec;
}

// ---------------------------------------------------------------------------
// Layout helpers — dagre-based left-to-right DAG positioning
// ---------------------------------------------------------------------------

/** Default node dimensions for dagre layout. */
const NODE_WIDTH = 200;
const NODE_HEIGHT = 80;
const TRANSFORM_NODE_WIDTH = 160;
const TRANSFORM_NODE_HEIGHT = 50;
const DATA_SOURCE_NODE_WIDTH = 180;
const DATA_SOURCE_NODE_HEIGHT = 120;

/**
 * Use dagre to compute left-to-right DAG layout for analysis nodes.
 * Includes data source, analysis nodes, and any transform nodes.
 */
function layoutNodes(
  specs: Record<string, AnalysisNodeSpec>,
  wires: AnalysisWire[],
  dataSourceId: string,
  dataSourceOutputs: string[],
  transformNodes: Array<{ id: string; transform: TransformSpec }> = [],
): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'LR', nodesep: 60, ranksep: 120, marginx: 40, marginy: 40 });

  // Add data source node
  g.setNode(dataSourceId, { width: DATA_SOURCE_NODE_WIDTH, height: DATA_SOURCE_NODE_HEIGHT });

  // Add analysis nodes
  for (const [id, spec] of Object.entries(specs)) {
    g.setNode(id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }

  // Add transform nodes
  for (const tn of transformNodes) {
    g.setNode(tn.id, { width: TRANSFORM_NODE_WIDTH, height: TRANSFORM_NODE_HEIGHT });
  }

  // Add edges from wires
  for (const wire of wires) {
    g.setEdge(wire.sourceId, wire.targetId);
  }

  dagre.layout(g);

  const nodes: Node[] = [];

  // Data source node
  const dsNode = g.node(dataSourceId);
  nodes.push({
    id: dataSourceId,
    type: 'dataSource',
    position: { x: dsNode.x - DATA_SOURCE_NODE_WIDTH / 2, y: dsNode.y - DATA_SOURCE_NODE_HEIGHT / 2 },
    data: {
      label: 'AnalysisInputData',
      outputs: dataSourceOutputs,
    } satisfies DataSourceNodeData,
  });

  // Analysis nodes
  for (const [id, spec] of Object.entries(specs)) {
    const n = g.node(id);
    nodes.push({
      id,
      type: spec.role === 'dependency' ? 'analysisDep' : 'analysis',
      position: { x: n.x - NODE_WIDTH / 2, y: n.y - NODE_HEIGHT / 2 },
      data: {
        spec,
        label: spec.label,
      } satisfies AnalysisNodeData,
    });
  }

  // Transform nodes
  for (const tn of transformNodes) {
    const n = g.node(tn.id);
    nodes.push({
      id: tn.id,
      type: 'transform',
      position: { x: n.x - TRANSFORM_NODE_WIDTH / 2, y: n.y - TRANSFORM_NODE_HEIGHT / 2 },
      data: {
        transform: tn.transform,
        label: tn.transform.label,
      } satisfies TransformNodeData,
    });
  }

  return nodes;
}

function buildEdges(wires: AnalysisWire[]): Edge[] {
  return wires.map((wire) => ({
    id: wire.id,
    source: wire.sourceId,
    sourceHandle: wire.sourcePort,
    target: wire.targetId,
    targetHandle: wire.targetPort,
    type: wire.implicit ? 'analysisImplicit' : 'analysisExplicit',
    data: {
      implicit: wire.implicit,
      transform: wire.transform,
    } satisfies AnalysisEdgeData,
  }));
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

interface AnalysisStoreState {
  // Graph spec
  graphSpec: AnalysisGraphSpec | null;

  // React Flow state
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;

  // Selection
  selectedNodeId: string | null;
  selectedTransformId: string | null;

  // Available analysis classes (from palette)
  analysisClasses: AnalysisClassDef[];

  // Actions
  setAnalysisClasses: (classes: AnalysisClassDef[]) => void;
  loadGraph: (spec: AnalysisGraphSpec) => void;
  setSelectedNode: (id: string | null) => void;
  setSelectedTransform: (id: string | null) => void;
  addAnalysisNode: (classDef: AnalysisClassDef, position: { x: number; y: number }) => void;
  removeNode: (id: string) => void;
  connectNodes: (connection: Connection) => void;
  updateNodeParams: (id: string, params: Record<string, unknown>) => void;
  addTransformToEdge: (edgeId: string, transformType: string) => void;
  removeTransformFromEdge: (edgeId: string) => void;
}

const DATA_SOURCE_ID = '__data_source__';
const DATA_SOURCE_OUTPUTS = ['states', 'inputs', 'outputs', 'targets', 'metadata'];

let nextNodeId = 1;
function genNodeId(): string {
  return `analysis_${nextNodeId++}`;
}

let nextWireId = 1;
function genWireId(): string {
  return `wire_${nextWireId++}`;
}

export const useAnalysisStore = create<AnalysisStoreState>((set, get) => ({
  graphSpec: null,
  nodes: [],
  edges: [],
  selectedNodeId: null,
  selectedTransformId: null,
  analysisClasses: [],

  onNodesChange: (changes) => {
    set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) }));
  },

  onEdgesChange: (changes) => {
    set((state) => ({ edges: applyEdgeChanges(changes, state.edges) }));
  },

  setAnalysisClasses: (classes) => {
    set({ analysisClasses: classes });
  },

  loadGraph: (spec) => {
    // Collect any existing transform specs from wires for layout
    const transformNodes: Array<{ id: string; transform: TransformSpec }> = [];
    const expandedWires: AnalysisWire[] = [];
    for (const wire of spec.wires) {
      if (wire.transform) {
        const tId = wire.transform.id;
        transformNodes.push({ id: tId, transform: wire.transform });
        // Split wire: source -> transform, transform -> target
        expandedWires.push({
          ...wire,
          id: `${wire.id}__to_transform`,
          targetId: tId,
          targetPort: 'in',
          transform: undefined,
        });
        expandedWires.push({
          id: `${wire.id}__from_transform`,
          sourceId: tId,
          sourcePort: 'out',
          targetId: wire.targetId,
          targetPort: wire.targetPort,
          implicit: wire.implicit,
        });
      } else {
        expandedWires.push(wire);
      }
    }

    const nodes = layoutNodes(spec.nodes, expandedWires, spec.dataSourceId, DATA_SOURCE_OUTPUTS, transformNodes);
    const edges = buildEdges(expandedWires);
    set({ graphSpec: spec, nodes, edges });
  },

  setSelectedNode: (id) => {
    set({ selectedNodeId: id, selectedTransformId: null });
  },

  setSelectedTransform: (id) => {
    set({ selectedTransformId: id, selectedNodeId: null });
  },

  addAnalysisNode: (classDef, position) => {
    const id = genNodeId();
    const spec: AnalysisNodeSpec = {
      id,
      type: classDef.name,
      label: classDef.name,
      category: classDef.category,
      inputPorts: [...classDef.inputPorts],
      outputPorts: [...classDef.outputPorts],
      params: { ...classDef.defaultParams },
      role: classDef.category === 'Preprocessing' ? 'dependency' : 'analysis',
    };

    const newNode: Node = {
      id,
      type: spec.role === 'dependency' ? 'analysisDep' : 'analysis',
      position,
      data: { spec, label: spec.label } satisfies AnalysisNodeData,
    };

    set((state) => ({
      nodes: [...state.nodes, newNode],
      graphSpec: state.graphSpec
        ? {
            ...state.graphSpec,
            nodes: { ...state.graphSpec.nodes, [id]: spec },
          }
        : null,
    }));
  },

  removeNode: (id) => {
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: state.selectedNodeId === id ? null : state.selectedNodeId,
    }));
  },

  connectNodes: (connection) => {
    if (!connection.source || !connection.target) return;
    const wireId = genWireId();
    const wire: AnalysisWire = {
      id: wireId,
      sourceId: connection.source,
      sourcePort: connection.sourceHandle ?? 'out',
      targetId: connection.target,
      targetPort: connection.targetHandle ?? 'in',
      implicit: connection.source === DATA_SOURCE_ID,
    };
    const edge: Edge = {
      id: wireId,
      source: connection.source,
      sourceHandle: connection.sourceHandle,
      target: connection.target,
      targetHandle: connection.targetHandle,
      type: wire.implicit ? 'analysisImplicit' : 'analysisExplicit',
      data: { implicit: wire.implicit } satisfies AnalysisEdgeData,
    };

    set((state) => ({
      edges: [...state.edges, edge],
      graphSpec: state.graphSpec
        ? { ...state.graphSpec, wires: [...state.graphSpec.wires, wire] }
        : null,
    }));
  },

  updateNodeParams: (id, params) => {
    set((state) => ({
      nodes: state.nodes.map((n) => {
        if (n.id !== id) return n;
        const data = n.data as AnalysisNodeData;
        return {
          ...n,
          data: {
            ...data,
            spec: { ...data.spec, params: { ...data.spec.params, ...params } },
          },
        };
      }),
    }));
  },

  addTransformToEdge: (edgeId, transformType) => {
    const state = get();
    const originalEdge = state.edges.find((e) => e.id === edgeId);
    if (!originalEdge) return;

    const transformId = `transform_${edgeId}`;
    const transform: TransformSpec = {
      id: transformId,
      type: transformType,
      label: transformType,
      params: {},
    };

    // Position the transform node midway between source and target nodes
    const sourceNode = state.nodes.find((n) => n.id === originalEdge.source);
    const targetNode = state.nodes.find((n) => n.id === originalEdge.target);
    const midX = sourceNode && targetNode
      ? (sourceNode.position.x + targetNode.position.x) / 2
      : (sourceNode?.position.x ?? 0) + 140;
    const midY = sourceNode && targetNode
      ? (sourceNode.position.y + targetNode.position.y) / 2
      : sourceNode?.position.y ?? 0;

    const transformNode: Node = {
      id: transformId,
      type: 'transform',
      position: { x: midX, y: midY },
      data: {
        transform,
        label: transform.label,
      } satisfies TransformNodeData,
    };

    // Replace original edge with two edges: source->transform, transform->target
    const edgeToTransform: Edge = {
      id: `${edgeId}__to_transform`,
      source: originalEdge.source,
      sourceHandle: originalEdge.sourceHandle,
      target: transformId,
      targetHandle: 'in',
      type: originalEdge.type,
      data: { implicit: (originalEdge.data as AnalysisEdgeData)?.implicit ?? false } satisfies AnalysisEdgeData,
    };

    const edgeFromTransform: Edge = {
      id: `${edgeId}__from_transform`,
      source: transformId,
      sourceHandle: 'out',
      target: originalEdge.target,
      targetHandle: originalEdge.targetHandle,
      type: originalEdge.type,
      data: { implicit: (originalEdge.data as AnalysisEdgeData)?.implicit ?? false } satisfies AnalysisEdgeData,
    };

    // Also update the graphSpec wire to record the transform
    const updatedWires = state.graphSpec?.wires.map((w) => {
      if (w.id !== edgeId) return w;
      return { ...w, transform };
    });

    set({
      nodes: [...state.nodes, transformNode],
      edges: [
        ...state.edges.filter((e) => e.id !== edgeId),
        edgeToTransform,
        edgeFromTransform,
      ],
      graphSpec: state.graphSpec
        ? { ...state.graphSpec, wires: updatedWires ?? state.graphSpec.wires }
        : null,
    });
  },

  removeTransformFromEdge: (edgeId) => {
    const state = get();
    const transformId = `transform_${edgeId}`;

    // Find the two split edges
    const toEdge = state.edges.find((e) => e.id === `${edgeId}__to_transform`);
    const fromEdge = state.edges.find((e) => e.id === `${edgeId}__from_transform`);

    if (!toEdge || !fromEdge) {
      // Fallback: just remove transform metadata from the edge data
      set({
        edges: state.edges.map((e) => {
          if (e.id !== edgeId) return e;
          const data = { ...e.data } as AnalysisEdgeData;
          delete data.transform;
          return { ...e, data };
        }),
      });
      return;
    }

    // Reconstruct the original edge
    const restoredEdge: Edge = {
      id: edgeId,
      source: toEdge.source,
      sourceHandle: toEdge.sourceHandle,
      target: fromEdge.target,
      targetHandle: fromEdge.targetHandle,
      type: toEdge.type,
      data: { implicit: (toEdge.data as AnalysisEdgeData)?.implicit ?? false } satisfies AnalysisEdgeData,
    };

    // Update graphSpec wire to remove transform
    const updatedWires = state.graphSpec?.wires.map((w) => {
      if (w.id !== edgeId) return w;
      const { transform: _, ...rest } = w;
      return rest;
    });

    set({
      nodes: state.nodes.filter((n) => n.id !== transformId),
      edges: [
        ...state.edges.filter((e) =>
          e.id !== `${edgeId}__to_transform` && e.id !== `${edgeId}__from_transform`
        ),
        restoredEdge,
      ],
      graphSpec: state.graphSpec
        ? { ...state.graphSpec, wires: updatedWires ?? state.graphSpec.wires }
        : null,
    });
  },
}));
