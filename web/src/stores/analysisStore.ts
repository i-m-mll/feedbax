/**
 * Zustand store for the analysis DAG graph state.
 *
 * Manages analysis nodes, wires, transforms, and selection state.
 * Mirrors the pattern of graphStore but for the analysis pipeline.
 */

import { create } from 'zustand';
import type { Node, Edge, OnNodesChange, OnEdgesChange, Connection } from '@xyflow/react';
import { applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
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
// Layout helpers — simple left-to-right positioning
// ---------------------------------------------------------------------------

const NODE_X_SPACING = 280;
const NODE_Y_SPACING = 120;
const DATA_SOURCE_X = 40;
const FIRST_ANALYSIS_X = DATA_SOURCE_X + NODE_X_SPACING;

function layoutNodes(
  specs: Record<string, AnalysisNodeSpec>,
  dataSourceId: string,
  dataSourceOutputs: string[]
): Node[] {
  const nodes: Node[] = [];

  // Data source node always on the left
  nodes.push({
    id: dataSourceId,
    type: 'dataSource',
    position: { x: DATA_SOURCE_X, y: 160 },
    data: {
      label: 'AnalysisInputData',
      outputs: dataSourceOutputs,
    } satisfies DataSourceNodeData,
  });

  // Layout analysis nodes in columns by dependency depth
  const nodeIds = Object.keys(specs);
  let col = 0;
  const placed = new Set<string>();
  const remaining = new Set(nodeIds);

  // Simple topological layering
  while (remaining.size > 0) {
    const batch: string[] = [];
    for (const id of remaining) {
      batch.push(id);
    }
    // For now, place all remaining in one column (TODO: proper topo sort with wires)
    let row = 0;
    for (const id of batch) {
      const spec = specs[id];
      nodes.push({
        id,
        type: spec.role === 'dependency' ? 'analysisDep' : 'analysis',
        position: {
          x: FIRST_ANALYSIS_X + col * NODE_X_SPACING,
          y: 40 + row * NODE_Y_SPACING,
        },
        data: {
          spec,
          label: spec.label,
        } satisfies AnalysisNodeData,
      });
      placed.add(id);
      remaining.delete(id);
      row++;
    }
    col++;
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
    const nodes = layoutNodes(spec.nodes, spec.dataSourceId, DATA_SOURCE_OUTPUTS);
    const edges = buildEdges(spec.wires);
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
    const transform: TransformSpec = {
      id: `transform_${edgeId}`,
      type: transformType,
      label: transformType,
      params: {},
    };

    set((state) => ({
      edges: state.edges.map((e) => {
        if (e.id !== edgeId) return e;
        return {
          ...e,
          data: { ...e.data, transform } satisfies AnalysisEdgeData,
        };
      }),
    }));
  },

  removeTransformFromEdge: (edgeId) => {
    set((state) => ({
      edges: state.edges.map((e) => {
        if (e.id !== edgeId) return e;
        const data = { ...e.data } as AnalysisEdgeData;
        delete data.transform;
        return { ...e, data };
      }),
    }));
  },
}));
