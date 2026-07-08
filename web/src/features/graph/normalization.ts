import { isCausalGraphSpec, type ComponentSpec, type GraphSpec, type TapSpec } from '@/types/graph';
import type { StudioTaskBindingSpec, StudioWorkspaceSpec } from '@/types/workspace';
import { taskBindingId } from '@/features/scenario/taskBindings';
import { normalizeDynamicPorts } from '@/features/graph/dynamicPorts';

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
      if (nextType === 'FeedbackChannel') nextType = 'Channel';
      if (nextType === 'PenzaiSubgraph') nextType = 'PenzaiAdapter';
      const nextParams = { ...spec.params };
      const nextSpec: ComponentSpec = {
        ...spec,
        type: nextType,
        params: nextParams,
      };
      return [id, nextSpec];
    })
  );
  const wires = graph.wires.map((wire) => {
    const sourceSpec = nodes[wire.source_node];
    const targetSpec = nodes[wire.target_node];
    const normalizedWire = {
      ...wire,
      source_port: sourceSpec ? renamePort(wire.source_node, wire.source_port, sourceSpec) : wire.source_port,
      target_port: targetSpec ? renamePort(wire.target_node, wire.target_port, targetSpec) : wire.target_port,
      temporality: wire.temporality,
      recurrent_initializer: wire.recurrent_initializer,
    };
    return normalizedWire;
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

export function normalizeGraphForStudioAuthoring(
  graph: GraphSpec,
  taskBindingSpec?: StudioTaskBindingSpec | null
): GraphSpec {
  return normalizeDynamicPorts(normalizeGraphAuthoringTypes(graph), taskBindingSpec);
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
      graph.nodes.input_mux.type === 'Mux'
    ) {
      changed = true;
      return {
        ...binding,
        id: taskBindingId(
          binding.source_data_id,
          'input_mux',
          binding.target_port,
          binding.target_graph_path
        ),
        target_node_id: 'input_mux',
      };
    }
    const target = graph.nodes[binding.target_node_id];
    if (target?.type !== 'Network' || binding.target_port !== 'target') {
      return binding;
    }
    changed = true;
    return {
      ...binding,
      id: taskBindingId(
        binding.source_data_id,
        binding.target_node_id,
        'input',
        binding.target_graph_path
      ),
      target_port: 'input',
    };
  });
  return changed ? { ...taskBindingSpec, bindings } : taskBindingSpec;
}

export function normalizeWorkspaceGraphsForStudioAuthoring(
  workspace: StudioWorkspaceSpec | null
): StudioWorkspaceSpec | null {
  if (!workspace) return workspace;
  let changed = false;
  const scenarios = Object.fromEntries(
    Object.entries(workspace.scenarios).map(([scenarioId, scenario]) => {
      if (!scenario.graph) return [scenarioId, scenario];
      const graph = normalizeGraphForStudioAuthoring(
        scenario.graph,
        scenario.task_binding_spec
      );
      const taskBindingSpec = normalizeTaskBindingSpecForStudioAuthoring(
        scenario.task_binding_spec,
        graph
      );
      if (graph === scenario.graph && taskBindingSpec === scenario.task_binding_spec) {
        return [scenarioId, scenario];
      }
      changed = true;
      return [scenarioId, { ...scenario, graph, task_binding_spec: taskBindingSpec }];
    })
  );
  return changed ? { ...workspace, scenarios } : workspace;
}
