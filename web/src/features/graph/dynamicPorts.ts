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

function maxBoundMuxInputIndex(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): number {
  let maxIndex = -1;
  for (const wire of graph.wires) {
    if (wire.target_node !== nodeId) continue;
    const index = muxInputIndex(wire.target_port);
    if (index !== null) maxIndex = Math.max(maxIndex, index);
  }
  for (const binding of taskBindingSpec?.bindings ?? []) {
    if (binding.target_node_id !== nodeId) continue;
    const index = muxInputIndex(binding.target_port);
    if (index !== null) maxIndex = Math.max(maxIndex, index);
  }
  return maxIndex;
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
  const nodes = Object.fromEntries(
    Object.entries(graph.nodes).map(([nodeId, spec]) => {
      if (!isMuxSpec(spec)) return [nodeId, spec];
      const requiredCount = Math.max(
        MIN_MUX_INPUT_PORTS,
        maxBoundMuxInputIndex(graph, nodeId, taskBindingSpec) + 1
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

export function materializeMuxSpecForBindings(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): ComponentSpec | null {
  const spec = graph.nodes[nodeId];
  if (!isMuxSpec(spec)) return null;
  return normalizeMuxSpec(
    spec,
    Math.max(muxPortCountFromSpec(spec), maxBoundMuxInputIndex(graph, nodeId, taskBindingSpec) + 1)
  );
}

export function visibleMuxInputPorts(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): { ports: string[]; nextPort: string | null } | null {
  const spec = materializeMuxSpecForBindings(graph, nodeId, taskBindingSpec);
  if (!spec) return null;
  const materializedGraph =
    spec === graph.nodes[nodeId]
      ? graph
      : { ...graph, nodes: { ...graph.nodes, [nodeId]: spec } };
  const nextPort = nextMuxInputPort(materializedGraph, nodeId, taskBindingSpec);
  return {
    ports: nextPort ? [...spec.input_ports, nextPort] : spec.input_ports,
    nextPort,
  };
}

export function muxHasSpareInput(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): boolean {
  const spec = graph.nodes[nodeId];
  if (!isMuxSpec(spec)) return false;
  const currentPorts = currentMuxInputPortSet(spec);
  for (const port of currentPorts) {
    const wired = graph.wires.some(
      (wire) => wire.target_node === nodeId && wire.target_port === port
    );
    const taskBound = (taskBindingSpec?.bindings ?? []).some(
      (binding) => binding.target_node_id === nodeId && binding.target_port === port
    );
    if (!wired && !taskBound) return true;
  }
  return false;
}

export function nextMuxInputPort(
  graph: GraphSpec,
  nodeId: string,
  taskBindingSpec?: StudioTaskBindingSpec | null
): string | null {
  const materializedSpec = materializeMuxSpecForBindings(graph, nodeId, taskBindingSpec);
  const spec = materializedSpec ?? graph.nodes[nodeId];
  if (!isMuxSpec(spec)) return null;
  const materializedGraph =
    materializedSpec && materializedSpec !== graph.nodes[nodeId]
      ? { ...graph, nodes: { ...graph.nodes, [nodeId]: materializedSpec } }
      : graph;
  if (muxHasSpareInput(materializedGraph, nodeId, taskBindingSpec)) return null;
  return muxInputPort(muxPortCountFromSpec(spec));
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
