import type { ComponentSpec, GraphSpec, TapSpec } from '@/types/graph';
import type { StudioTaskBindingSpec, StudioWorkspaceSpec } from '@/types/workspace';
import { normalizeDynamicPorts } from '@/features/graph/dynamicPorts';

function standardNetworkSubgraph(nodeId: string, params: ComponentSpec['params']): GraphSpec {
  const now = new Date().toISOString();
  const hiddenTypeRaw = typeof params.hidden_type === 'string' ? params.hidden_type : 'GRUCell';
  const cellType = hiddenTypeRaw === 'LSTMCell' || hiddenTypeRaw === 'LSTM' ? 'LSTM' : 'GRU';
  const hiddenSize = typeof params.hidden_size === 'number' ? params.hidden_size : 100;
  const inputSize = typeof params.input_size === 'number' ? params.input_size : 6;
  const outputSize =
    typeof params.out_size === 'number'
      ? params.out_size
      : typeof params.output_size === 'number'
        ? params.output_size
        : 2;
  const readoutActivation =
    typeof params.out_nonlinearity === 'string' ? params.out_nonlinearity : 'tanh';
  const cellInputPorts = cellType === 'LSTM' ? ['input', 'hidden', 'cell'] : ['input', 'hidden'];
  const cellOutputPorts = cellType === 'LSTM' ? ['output', 'hidden', 'cell'] : ['output', 'hidden'];
  const recurrentWires = networkRecurrentWires(cellType, hiddenSize);

  return {
    nodes: {
      input_mux: {
        type: 'Mux',
        params: {
          n_inputs: 2,
        },
        input_ports: ['in_0', 'in_1'],
        output_ports: ['output'],
      },
      cell: {
        type: cellType,
        params: {
          input_size: inputSize,
          hidden_size: hiddenSize,
        },
        input_ports: cellInputPorts,
        output_ports: cellOutputPorts,
      },
      readout: {
        type: 'Linear',
        params: {
          input_size: hiddenSize,
          output_size: outputSize,
          use_bias: true,
          activation: readoutActivation,
        },
        input_ports: ['input'],
        output_ports: ['output'],
      },
    },
    wires: [
      {
        source_node: 'input_mux',
        source_port: 'output',
        target_node: 'cell',
        target_port: 'input',
      },
      {
        source_node: 'cell',
        source_port: 'output',
        target_node: 'readout',
        target_port: 'input',
      },
      ...recurrentWires,
    ],
    input_ports: ['input', 'feedback'],
    output_ports: ['output', 'hidden'],
    input_bindings: {
      input: ['input_mux', 'in_0'],
      feedback: ['input_mux', 'in_1'],
    },
    output_bindings: {
      output: ['readout', 'output'],
      hidden: ['cell', 'hidden'],
    },
    taps: [],
    subgraphs: {},
    metadata: {
      name: `${nodeId} internals`,
      description: 'Auto-generated Network subgraph',
      created_at: now,
      updated_at: now,
      version: '1.0.0',
    },
  };
}

function recurrentZeroInitializer(width: number, stateSlot: string) {
  return {
    kind: 'zeros',
    scope: 'trial',
    shape: [width],
    source: 'state_initializer',
    state_slot: stateSlot,
  };
}

function networkRecurrentWires(cellType: 'GRU' | 'LSTM', hiddenSize: number): GraphSpec['wires'] {
  const wires: GraphSpec['wires'] = [
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

function recurrentFeedbackInitializer() {
  return {
    kind: 'zeros',
    scope: 'trial',
    source: 'state_initializer',
    state_slot: 'feedback',
  };
}

function instantPathExists(wires: GraphSpec['wires'], sourceNode: string, targetNode: string): boolean {
  const adjacency = new Map<string, Set<string>>();
  for (const wire of wires) {
    if (wire.temporality === 'recurrent') continue;
    const targets = adjacency.get(wire.source_node) ?? new Set<string>();
    targets.add(wire.target_node);
    adjacency.set(wire.source_node, targets);
  }
  const visited = new Set<string>();
  const stack = [sourceNode];
  while (stack.length > 0) {
    const nodeId = stack.pop()!;
    if (nodeId === targetNode) return true;
    if (visited.has(nodeId)) continue;
    visited.add(nodeId);
    for (const next of adjacency.get(nodeId) ?? []) {
      stack.push(next);
    }
  }
  return false;
}

function normalizeNetworkInputPorts(inputPorts: string[]): string[] {
  const next = inputPorts
    .map((port) => (port === 'target' ? 'input' : port))
    .filter((port) => port !== 'hidden' && port !== 'cell');
  if (!next.includes('input')) next.unshift('input');
  if (!next.includes('feedback')) next.push('feedback');
  return [...new Set(next)];
}

function normalizeNetworkOutputPorts(outputPorts: string[]): string[] {
  const next = outputPorts.length > 0 ? [...outputPorts] : ['output'];
  if (!next.includes('output')) next.unshift('output');
  if (!next.includes('hidden')) next.push('hidden');
  return [...new Set(next)];
}

function unwrapLegacyNetworkModelLayer(subgraph: GraphSpec): GraphSpec {
  const nodeIds = Object.keys(subgraph.nodes);
  const modelNode = subgraph.nodes.model;
  const inner = subgraph.subgraphs?.model;
  if (nodeIds.length !== 1 || nodeIds[0] !== 'model' || modelNode?.type !== 'Subgraph' || !inner) {
    return subgraph;
  }
  if (subgraph.wires.length > 0) return subgraph;
  const forwardsInputs = Object.entries(subgraph.input_bindings).every(
    ([port, binding]) => binding[0] === 'model' && binding[1] === port
  );
  const forwardsOutputs = Object.entries(subgraph.output_bindings).every(
    ([port, binding]) => binding[0] === 'model' && binding[1] === port
  );
  if (!forwardsInputs || !forwardsOutputs) return subgraph;
  return {
    ...inner,
    input_ports: [...new Set([...subgraph.input_ports, ...inner.input_ports])],
    output_ports: [...new Set([...subgraph.output_ports, ...inner.output_ports])],
    metadata: subgraph.metadata ?? inner.metadata,
  };
}

function normalizeNetworkSubgraph(subgraph: GraphSpec, params: ComponentSpec['params']): GraphSpec {
  subgraph = unwrapLegacyNetworkModelLayer(subgraph);
  const inputBinding = subgraph.input_bindings.input;
  const feedbackBinding = subgraph.input_bindings.feedback;
  const cellId =
    inputBinding && subgraph.nodes[inputBinding[0]]?.type.match(/^(GRU|LSTM)$/)
      ? inputBinding[0]
      : Object.entries(subgraph.nodes).find(([, node]) => node.type === 'GRU' || node.type === 'LSTM')?.[0];
  if (!cellId) return subgraph;
  const cellNode = subgraph.nodes[cellId];
  const cellType = cellNode.type === 'LSTM' ? 'LSTM' : 'GRU';
  const hiddenSize = typeof cellNode.params.hidden_size === 'number' ? cellNode.params.hidden_size : 100;
  const recurrentWires = networkRecurrentWires(cellType, hiddenSize);
  const recurrentTargets = new Set(
    recurrentWires.map((wire) => `${wire.target_node}.${wire.target_port}`)
  );
  const existingRecurrentTargets = new Set(
    subgraph.wires
      .filter((wire) => wire.temporality === 'recurrent')
      .map((wire) => `${wire.target_node}.${wire.target_port}`)
  );
  const missingRecurrentWires = recurrentWires.filter(
    (wire) => !existingRecurrentTargets.has(`${wire.target_node}.${wire.target_port}`)
  );
  const inputMux = subgraph.nodes.input_mux;
  const hasMuxWire = subgraph.wires.some(
    (wire) =>
      wire.source_node === 'input_mux' &&
      wire.source_port === 'output' &&
      wire.target_node === cellId &&
      wire.target_port === 'input'
  );
  const needsMux =
    !inputMux ||
    !hasMuxWire ||
    !feedbackBinding ||
    inputBinding?.[0] !== 'input_mux' ||
    inputBinding?.[1] !== 'in_0' ||
    missingRecurrentWires.length > 0 ||
    subgraph.output_bindings.hidden?.[0] !== cellId ||
    subgraph.output_bindings.hidden?.[1] !== 'hidden';
  if (!needsMux) {
    return {
      ...subgraph,
      input_ports: normalizeNetworkInputPorts(subgraph.input_ports),
      output_ports: normalizeNetworkOutputPorts(subgraph.output_ports),
    };
  }
  const nodes = {
    ...subgraph.nodes,
    input_mux: inputMux ?? {
      type: 'Mux',
      params: { n_inputs: 2 },
      input_ports: ['in_0', 'in_1'],
      output_ports: ['output'],
    },
  };
  const wires = [
    ...subgraph.wires.filter(
      (wire) =>
        !(wire.target_node === cellId && wire.target_port === 'input') &&
        !recurrentTargets.has(`${wire.target_node}.${wire.target_port}`)
    ),
    {
      source_node: 'input_mux',
      source_port: 'output',
      target_node: cellId,
      target_port: 'input',
    },
    ...recurrentWires,
  ];
  return {
    ...subgraph,
    nodes,
    wires,
    input_ports: normalizeNetworkInputPorts(subgraph.input_ports),
    output_ports: normalizeNetworkOutputPorts(subgraph.output_ports),
    input_bindings: {
      ...subgraph.input_bindings,
      input: ['input_mux', 'in_0'],
      feedback: ['input_mux', 'in_1'],
    },
    output_bindings: {
      ...subgraph.output_bindings,
      hidden: [cellId, 'hidden'] as [string, string],
    },
    metadata: {
      ...subgraph.metadata,
      updated_at: new Date().toISOString(),
      description:
        subgraph.metadata?.description ?? 'Auto-generated Network subgraph',
    },
  };
}

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

  const generatedSubgraphs: Record<string, GraphSpec> = {};
  let nodes = Object.fromEntries(
    Object.entries(graph.nodes).map(([id, spec]) => {
      const wasRuntimeNetwork = spec.type === 'SimpleStagedNetwork';
      let nextType = spec.type;
      if (nextType === 'SimpleStagedNetwork') nextType = 'Network';
      if (nextType === 'FeedbackChannel') nextType = 'Channel';
      if (nextType === 'PenzaiSubgraph') nextType = 'PenzaiAdapter';
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
      if (nextType === 'Network') {
        if (graph.subgraphs?.[id] || wasRuntimeNetwork) {
          nextSpec.input_ports = spec.input_ports.map((port) =>
            port === 'target' ? 'input' : port
          );
        } else {
          nextSpec.input_ports = normalizeNetworkInputPorts(spec.input_ports);
          nextSpec.output_ports = normalizeNetworkOutputPorts(spec.output_ports);
        }
      }
      if (wasRuntimeNetwork && !graph.subgraphs?.[id]) {
        generatedSubgraphs[id] = standardNetworkSubgraph(id, nextParams);
      }
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
    if (
      normalizedWire.temporality !== 'recurrent' &&
      targetSpec?.type === 'Network' &&
      normalizedWire.source_port === 'output' &&
      normalizedWire.target_port === 'feedback' &&
      instantPathExists(graph.wires, normalizedWire.target_node, normalizedWire.source_node)
    ) {
      return {
        ...normalizedWire,
        temporality: 'recurrent' as const,
        recurrent_initializer: recurrentFeedbackInitializer(),
      };
    }
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
    graph.subgraphs || Object.keys(generatedSubgraphs).length > 0
      ? {
          ...generatedSubgraphs,
          ...(graph.subgraphs
            ? Object.fromEntries(
                Object.entries(graph.subgraphs).map(([id, subgraph]) => {
                  const normalized = normalizeGraphAuthoringTypes(subgraph);
                  const owner = nodes[id];
                  return [
                    id,
                    owner?.type === 'Network'
                      ? normalizeNetworkSubgraph(normalized, owner.params)
                      : normalized,
                  ];
                })
              )
            : {}),
        }
      : undefined;
  if (subgraphs) {
    nodes = Object.fromEntries(
      Object.entries(nodes).map(([id, node]) => {
        const subgraph = subgraphs[id];
        if (node.type !== 'Network' || !subgraph) return [id, node];
        return [
          id,
          {
            ...node,
            input_ports: subgraph.input_ports,
            output_ports: subgraph.output_ports,
          },
        ];
      })
    );
  }
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
        id: `task:${binding.source_data_id}->input_mux:${binding.target_port}`,
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
      id: `task:${binding.source_data_id}->${binding.target_node_id}:input`,
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
