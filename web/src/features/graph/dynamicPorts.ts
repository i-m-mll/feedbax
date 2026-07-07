import type { GraphSpec, ComponentSpec } from '@/types/graph';
import type { StudioTaskBindingSpec } from '@/types/workspace';

export const MUX_COMPONENT_TYPE = 'Mux';
export const MIN_MUX_INPUT_PORTS = 2;

export function muxInputPort(index: number): string {
  return `in_${index}`;
}

export function muxInputIndex(port: string | null | undefined): number | null {
  const match = /^in_(\d+)$/.exec(port ?? '');
  if (!match) return null;
  const index = Number(match[1]);
  return Number.isInteger(index) && index >= 0 ? index : null;
}

export function isMuxSpec(spec: ComponentSpec | null | undefined): boolean {
  return spec?.type === MUX_COMPONENT_TYPE;
}

export function muxInputPorts(count: number): string[] {
  const safeCount = Math.max(MIN_MUX_INPUT_PORTS, Math.ceil(count));
  return Array.from({ length: safeCount }, (_, index) => muxInputPort(index));
}

export function muxPortCountFromSpec(spec: ComponentSpec): number {
  const paramCount = Number(spec.params?.n_inputs);
  return Math.max(
    MIN_MUX_INPUT_PORTS,
    Number.isFinite(paramCount) ? Math.ceil(paramCount) : 0,
    spec.input_ports.length
  );
}

interface MuxBindingIndex {
  maxInputIndexByNode: Map<string, number>;
  occupiedInputPortsByNode: Map<string, Set<string>>;
}

function addMuxInputReference(index: MuxBindingIndex, nodeId: string, port: string): void {
  const inputIndex = muxInputIndex(port);
  if (inputIndex === null) return;
  index.maxInputIndexByNode.set(
    nodeId,
    Math.max(index.maxInputIndexByNode.get(nodeId) ?? -1, inputIndex)
  );
  const occupiedPorts = index.occupiedInputPortsByNode.get(nodeId) ?? new Set<string>();
  occupiedPorts.add(port);
  index.occupiedInputPortsByNode.set(nodeId, occupiedPorts);
}

function buildMuxBindingIndex(
  graph: GraphSpec,
  taskBindingSpec?: StudioTaskBindingSpec | null
): MuxBindingIndex {
  const index: MuxBindingIndex = {
    maxInputIndexByNode: new Map(),
    occupiedInputPortsByNode: new Map(),
  };
  for (const wire of graph.wires) {
    addMuxInputReference(index, wire.target_node, wire.target_port);
  }
  for (const binding of taskBindingSpec?.bindings ?? []) {
    addMuxInputReference(index, binding.target_node_id, binding.target_port);
  }
  return index;
}

function maxBoundMuxInputIndex(index: MuxBindingIndex, nodeId: string): number {
  return index.maxInputIndexByNode.get(nodeId) ?? -1;
}

function currentMuxInputPortSet(spec: ComponentSpec): Set<string> {
  return new Set(muxInputPorts(muxPortCountFromSpec(spec)));
}

export function normalizeMuxSpec(
  spec: ComponentSpec,
  inputCount: number
): ComponentSpec {
  if (!isMuxSpec(spec)) return spec;
  const count = Math.max(MIN_MUX_INPUT_PORTS, Math.ceil(inputCount));
  return {
    ...spec,
    params: {
      ...spec.params,
      n_inputs: count,
    },
    input_ports: muxInputPorts(count),
    output_ports: spec.output_ports.length > 0 ? spec.output_ports : ['output'],
  };
}

export function normalizeDynamicPorts(
  graph: GraphSpec,
  taskBindingSpec?: StudioTaskBindingSpec | null
): GraphSpec {
  let changed = false;
  const muxBindingIndex = buildMuxBindingIndex(graph, taskBindingSpec);
  const nodes = Object.fromEntries(
    Object.entries(graph.nodes).map(([nodeId, spec]) => {
      if (!isMuxSpec(spec)) return [nodeId, spec];
      const requiredCount = Math.max(
        MIN_MUX_INPUT_PORTS,
        maxBoundMuxInputIndex(muxBindingIndex, nodeId) + 1
      );
      const normalized = normalizeMuxSpec(spec, requiredCount);
      if (
        normalized.input_ports.length !== spec.input_ports.length ||
        normalized.params.n_inputs !== spec.params.n_inputs
      ) {
        changed = true;
      }
      return [nodeId, normalized];
    })
  );
  return changed ? { ...graph, nodes } : graph;
}

function materializeMuxSpecForBindingsWithIndex(
  graph: GraphSpec,
  nodeId: string,
  muxBindingIndex: MuxBindingIndex
): ComponentSpec | null {
  const spec = graph.nodes[nodeId];
  if (!isMuxSpec(spec)) return null;
  return normalizeMuxSpec(
    spec,
    Math.max(muxPortCountFromSpec(spec), maxBoundMuxInputIndex(muxBindingIndex, nodeId) + 1)
  );
}

export function materializeMuxSpecForBindings(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): ComponentSpec | null {
  return materializeMuxSpecForBindingsWithIndex(
    graph,
    nodeId,
    buildMuxBindingIndex(graph, taskBindingSpec)
  );
}

export function visibleMuxInputPorts(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): { ports: string[]; nextPort: string | null } | null {
  const muxBindingIndex = buildMuxBindingIndex(graph, taskBindingSpec);
  const spec = materializeMuxSpecForBindingsWithIndex(graph, nodeId, muxBindingIndex);
  if (!spec) return null;
  const nextPort = nextMuxInputPortWithIndex(graph, nodeId, muxBindingIndex, spec);
  return {
    ports: nextPort ? [...spec.input_ports, nextPort] : spec.input_ports,
    nextPort,
  };
}

function muxHasSpareInputWithIndex(
  spec: ComponentSpec,
  nodeId: string,
  muxBindingIndex: MuxBindingIndex
): boolean {
  const currentPorts = currentMuxInputPortSet(spec);
  const occupiedPorts = muxBindingIndex.occupiedInputPortsByNode.get(nodeId) ?? new Set<string>();
  for (const port of currentPorts) {
    if (!occupiedPorts.has(port)) return true;
  }
  return false;
}

export function muxHasSpareInput(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): boolean {
  const spec = graph.nodes[nodeId];
  if (!isMuxSpec(spec)) return false;
  return muxHasSpareInputWithIndex(spec, nodeId, buildMuxBindingIndex(graph, taskBindingSpec));
}

function nextMuxInputPortWithIndex(
  graph: GraphSpec,
  nodeId: string,
  muxBindingIndex: MuxBindingIndex,
  materializedSpec?: ComponentSpec | null
): string | null {
  const spec = materializedSpec ?? graph.nodes[nodeId];
  if (!isMuxSpec(spec)) return null;
  if (muxHasSpareInputWithIndex(spec, nodeId, muxBindingIndex)) return null;
  return muxInputPort(muxPortCountFromSpec(spec));
}

export function nextMuxInputPort(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): string | null {
  const muxBindingIndex = buildMuxBindingIndex(graph, taskBindingSpec);
  return nextMuxInputPortWithIndex(
    graph,
    nodeId,
    muxBindingIndex,
    materializeMuxSpecForBindingsWithIndex(graph, nodeId, muxBindingIndex)
  );
}

export function isNextMuxInputPort(
  graph: GraphSpec,
  nodeId: string | null | undefined,
  port: string | null | undefined,
  taskBindingSpec?: StudioTaskBindingSpec | null
): boolean {
  if (!nodeId || !port) return false;
  return nextMuxInputPort(graph, nodeId, taskBindingSpec) === port;
}

export function expandMuxForPort(
  graph: GraphSpec,
  nodeId: string,
  port: string
): GraphSpec {
  const spec = graph.nodes[nodeId];
  const index = muxInputIndex(port);
  if (!isMuxSpec(spec) || index === null) return graph;
  const nextSpec = normalizeMuxSpec(spec, index + 1);
  if (nextSpec === spec || nextSpec.input_ports.length === spec.input_ports.length) return graph;
  return {
    ...graph,
    nodes: {
      ...graph.nodes,
      [nodeId]: nextSpec,
    },
  };
}
