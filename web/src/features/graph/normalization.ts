import {
  isAcausalGraphSpec,
  isCausalGraphSpec,
  type ComponentSpec,
  type GraphSpec,
  type GraphSubgraphSpec,
  type TapSpec,
} from '@/types/graph';
import type { StudioTaskBindingSpec } from '@/types/workspace';
import { taskBindingId } from '@/features/scenario/taskBindings';
import { normalizeDynamicPorts, type ComponentCatalogue } from '@/features/graph/dynamicPorts';

function createTapId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `tap-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
}

function migrateLegacyTaps(graph: GraphSpec): TapSpec[] {
  const taps: TapSpec[] = graph.taps ? [...graph.taps] : [];
  const usedIds = new Set(taps.map((tap) => tap.id));

  const addTap = (tap: TapSpec) => {
    if (usedIds.has(tap.id)) {
      tap = { ...tap, id: createTapId() };
    }
    usedIds.add(tap.id);
    taps.push(tap);
  };

  if (graph.barnacles) {
    for (const [nodeId, barnacles] of Object.entries(graph.barnacles)) {
      for (const barnacle of barnacles) {
        const paths: Record<string, string> = {};
        const usedNames = new Set<string>();
        for (const path of barnacle.read_paths ?? []) {
          const base = path.split('.').slice(-1)[0] || 'value';
          let name = base;
          let idx = 2;
          while (usedNames.has(name)) {
            name = `${base}_${idx}`;
            idx += 1;
          }
          usedNames.add(name);
          paths[name] = path;
        }
        const transform =
          barnacle.kind === 'intervention'
            ? {
                type: barnacle.label || 'intervention',
                params: {
                  read_paths: barnacle.read_paths ?? [],
                  write_paths: barnacle.write_paths ?? [],
                  transform: barnacle.transform ?? '',
                },
              }
            : undefined;
        addTap({
          id: barnacle.id,
          type: barnacle.kind,
          position: { afterNode: nodeId },
          paths,
          transform,
        });
      }
    }
  }

  if (graph.user_ports) {
    for (const [nodeId, ports] of Object.entries(graph.user_ports)) {
      const paths: Record<string, string> = {};
      for (const port of ports.outputs ?? []) {
        paths[port] = port;
      }
      for (const port of ports.inputs ?? []) {
        if (!(port in paths)) {
          paths[port] = port;
        }
      }
      if (Object.keys(paths).length > 0) {
        addTap({
          id: createTapId(),
          type: 'probe',
          position: { afterNode: nodeId },
          paths,
        });
      }
    }
  }

  return taps;
}

export function normalizeGraphAuthoringTypes(graph: GraphSpec): GraphSpec {
  const nodes = graph.nodes;
  const wires = graph.wires;
  const input_bindings = graph.input_bindings;
  const input_ports = graph.input_ports;
  const subgraphs =
    graph.subgraphs
      ? Object.fromEntries(
          Object.entries(graph.subgraphs).map(([id, subgraph]) => [
            id,
            isCausalGraphSpec(subgraph) ? normalizeGraphAuthoringTypes(subgraph) : subgraph,
          ])
        )
      : undefined;
  const taps =
    graph.taps || graph.barnacles || graph.user_ports ? migrateLegacyTaps(graph) : undefined;
  const normalized: GraphSpec = {
    ...graph,
    nodes,
    wires,
    input_ports,
    input_bindings,
  };
  if (taps) normalized.taps = taps;
  if (subgraphs) normalized.subgraphs = subgraphs;
  if (graph.barnacles) normalized.barnacles = undefined;
  if (graph.user_ports) normalized.user_ports = undefined;
  return normalized;
}

export function normalizeGraphForStudioAuthoring<T extends GraphSubgraphSpec>(
  graph: T,
  components?: ComponentCatalogue,
  taskBindingSpec?: StudioTaskBindingSpec | null
): T {
  if (isAcausalGraphSpec(graph)) return graph;
  const normalized = normalizeGraphAuthoringTypes(graph) as T;
  return components ? normalizeDynamicPorts(normalized, components, taskBindingSpec) : normalized;
}

export function normalizeTaskBindingSpecForStudioAuthoring(
  taskBindingSpec: StudioTaskBindingSpec | null | undefined,
  graph: GraphSpec
): StudioTaskBindingSpec | null | undefined {
  if (!taskBindingSpec) return taskBindingSpec;
  let changed = false;
  const bindings = taskBindingSpec.bindings.map((binding) => {
    if (
      binding.target_node_id === 'mux' &&
      !graph.nodes.mux &&
      graph.nodes.input_mux &&
      graph.nodes.input_mux.type === 'feedbax.Mux'
    ) {
      changed = true;
      return {
        ...binding,
        id: taskBindingId(
          binding.source_data_id,
          'input_mux',
          binding.target_port
        ),
        target_node_id: 'input_mux',
      };
    }
    const target = graph.nodes[binding.target_node_id];
    if (target?.type !== 'feedbax.Network' || binding.target_port !== 'target') {
      return binding;
    }
    changed = true;
    return {
      ...binding,
      id: taskBindingId(
        binding.source_data_id,
        binding.target_node_id,
        'input'
      ),
      target_port: 'input',
    };
  });
  return changed ? { ...taskBindingSpec, bindings } : taskBindingSpec;
}
