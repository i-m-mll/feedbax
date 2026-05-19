import type { GraphSpec, TapSpec, WireSpec } from '@/types/graph';
import type {
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioEntity,
  StudioScenarioEntityKind,
  StudioScenarioEntityRegistry,
  StudioScenarioEntityRelation,
  StudioScenarioSpec,
  StudioSelectorRef,
} from '@/types/workspace';

const MECHANICS_TYPE_HINTS = [
  'mechanic',
  'plant',
  'arm',
  'muscle',
  'mass',
  'body',
  'joint',
  'skeleton',
];

const TASK_TYPE_HINTS = ['task', 'reach', 'environment', 'trial'];

export function graphNodeEntityId(nodeId: string): string {
  return `graph_node:${nodeId}`;
}

export function graphPortEntityId(
  nodeId: string,
  direction: 'input' | 'output',
  port: string
): string {
  return `graph_port:${nodeId}:${direction}:${port}`;
}

export function graphEdgeId(wire: WireSpec): string {
  return `${wire.source_node}:${wire.source_port}->${wire.target_node}:${wire.target_port}`;
}

export function graphEdgeEntityId(edgeId: string): string {
  return `graph_edge:${edgeId}`;
}

export function probeEntityId(tapId: string): string {
  return `probe:${tapId}`;
}

export function taskEntityId(scenarioId: string | null | undefined): string {
  return `task_object:${scenarioId ?? 'active'}:task`;
}

export function mechanicsEntityId(
  scenarioId: string | null | undefined,
  nodeId = 'active'
): string {
  return `mechanics_object:${scenarioId ?? 'active'}:${nodeId}`;
}

export function objectiveEntityId(termId: string): string {
  return `objective_term:${termId}`;
}

export function entityKindLabel(kind: StudioScenarioEntityKind): string {
  switch (kind) {
    case 'graph_node':
      return 'Graph Node';
    case 'graph_port':
      return 'Graph Port';
    case 'graph_edge':
      return 'Graph Edge';
    case 'task_object':
      return 'Task';
    case 'mechanics_object':
      return 'Mechanics';
    case 'objective_term':
      return 'Objective';
    case 'probe':
      return 'Probe';
    case 'metric':
      return 'Metric';
    case 'temporal_event':
      return 'Temporal Event';
    case 'temporal_phase':
      return 'Temporal Phase';
    case 'stage_protocol':
      return 'Stage Protocol';
    case 'artifact_overlay':
      return 'Artifact Overlay';
    default:
      return kind;
  }
}

export function selectorToEntityId(selector: StudioSelectorRef | null | undefined): string | null {
  if (!selector) return null;
  const graphPortNodeId =
    typeof selector.metadata.graph_port_node_id === 'string'
      ? selector.metadata.graph_port_node_id
      : null;
  const graphPortName =
    typeof selector.metadata.graph_port_name === 'string'
      ? selector.metadata.graph_port_name
      : null;
  if (graphPortNodeId && graphPortName) {
    const direction = selector.metadata.graph_port_direction === 'input' ? 'input' : 'output';
    return graphPortEntityId(graphPortNodeId, direction, graphPortName);
  }
  if (selector.namespace === 'graph_port' && selector.target_id && selector.path) {
    const direction = selector.metadata.direction === 'input' ? 'input' : 'output';
    return graphPortEntityId(selector.target_id, direction, selector.path);
  }
  if (selector.namespace === 'probe' && selector.target_id) {
    return probeEntityId(selector.target_id);
  }
  if (selector.namespace === 'task_object') {
    return selector.target_id ? `task_object:${selector.target_id}` : null;
  }
  if (selector.namespace === 'mechanics_object' || selector.namespace === 'biomechanics_object') {
    return selector.target_id ? `mechanics_object:${selector.target_id}` : null;
  }
  return null;
}

export function entityIdFromGraphSelection(selection: {
  nodeId?: string | null;
  tapId?: string | null;
  edgeId?: string | null;
}): string | null {
  if (selection.tapId) return probeEntityId(selection.tapId);
  if (selection.nodeId) return graphNodeEntityId(selection.nodeId);
  if (selection.edgeId) return graphEdgeEntityId(selection.edgeId);
  return null;
}

function addEntity(
  registry: StudioScenarioEntityRegistry,
  entity: StudioScenarioEntity,
  root = false
) {
  registry.entities[entity.id] = entity;
  if (root) registry.root_entity_ids.push(entity.id);
}

function relation(
  kind: StudioScenarioEntityRelation['kind'],
  entityId: string,
  label?: string
): StudioScenarioEntityRelation {
  return {
    kind,
    entity_id: entityId,
    label: label ?? null,
    metadata: {},
  };
}

function isObjectiveSpec(value: StudioScenarioSpec['objective_spec']): value is StudioObjectiveSpec {
  return Boolean(value && Array.isArray((value as StudioObjectiveSpec).terms));
}

function isMechanicsNode(nodeId: string, type: string): boolean {
  const normalized = `${nodeId} ${type}`.toLowerCase();
  return MECHANICS_TYPE_HINTS.some((hint) => normalized.includes(hint));
}

function isTaskNode(nodeId: string, type: string): boolean {
  const normalized = `${nodeId} ${type}`.toLowerCase();
  return TASK_TYPE_HINTS.some((hint) => normalized.includes(hint));
}

function addGraphEntities(registry: StudioScenarioEntityRegistry, graph: GraphSpec) {
  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const entityId = graphNodeEntityId(nodeId);
    addEntity(registry, {
      id: entityId,
      kind: 'graph_node',
      label: nodeId,
      summary: node.type,
      scenario_id: registry.scenario_id,
      stage_id: registry.stage_id,
      selector: null,
      relations: [],
      metadata: {
        node_id: nodeId,
        component_type: node.type,
        input_ports: node.input_ports,
        output_ports: node.output_ports,
      },
    }, true);

    for (const port of node.input_ports) {
      addEntity(registry, {
        id: graphPortEntityId(nodeId, 'input', port),
        kind: 'graph_port',
        label: `${nodeId}.${port}`,
        summary: 'Input port',
        scenario_id: registry.scenario_id,
        stage_id: registry.stage_id,
        selector: {
          namespace: 'graph_port',
          compact: `port:${nodeId}.${port}`,
          target_id: nodeId,
          path: port,
          role: 'editable',
          metadata: { direction: 'input' },
        },
        relations: [relation('contains', entityId, 'node')],
        metadata: { node_id: nodeId, port, direction: 'input' },
      });
    }

    for (const port of node.output_ports) {
      addEntity(registry, {
        id: graphPortEntityId(nodeId, 'output', port),
        kind: 'graph_port',
        label: `${nodeId}.${port}`,
        summary: 'Output port',
        scenario_id: registry.scenario_id,
        stage_id: registry.stage_id,
        selector: {
          namespace: 'graph_port',
          compact: `port:${nodeId}.${port}`,
          target_id: nodeId,
          path: port,
          role: 'observed',
          metadata: { direction: 'output' },
        },
        relations: [relation('contains', entityId, 'node')],
        metadata: { node_id: nodeId, port, direction: 'output' },
      });
    }
  }

  for (const wire of graph.wires) {
    const edgeId = graphEdgeId(wire);
    addEntity(registry, {
      id: graphEdgeEntityId(edgeId),
      kind: 'graph_edge',
      label: `${wire.source_node}.${wire.source_port} -> ${wire.target_node}.${wire.target_port}`,
      summary: 'Port wire',
      scenario_id: registry.scenario_id,
      stage_id: registry.stage_id,
      selector: null,
      relations: [
        relation('source', graphPortEntityId(wire.source_node, 'output', wire.source_port)),
        relation('target', graphPortEntityId(wire.target_node, 'input', wire.target_port)),
      ],
      metadata: { edge_id: edgeId, wire },
    }, true);
  }

  for (const tap of graph.taps ?? []) {
    addProbeEntity(registry, tap);
  }
}

function addProbeEntity(registry: StudioScenarioEntityRegistry, tap: TapSpec) {
  addEntity(registry, {
    id: probeEntityId(tap.id),
    kind: 'probe',
    label: tap.type === 'probe' ? 'Probe' : 'Intervention',
    summary: Object.keys(tap.paths ?? {}).join(', ') || tap.position.afterNode,
    scenario_id: registry.scenario_id,
    stage_id: registry.stage_id,
    selector: {
      namespace: 'probe',
      compact: `probe:${tap.id}`,
      target_id: tap.id,
      path: null,
      role: tap.type === 'probe' ? 'observed' : 'editable',
      metadata: {},
    },
    relations: [relation('binds', graphNodeEntityId(tap.position.afterNode), 'after node')],
    metadata: { tap },
  }, true);
}

function addTaskEntity(
  registry: StudioScenarioEntityRegistry,
  scenario: StudioScenarioSpec,
  graph: GraphSpec | null
) {
  if (!scenario.task_spec) return;
  const taskNode = Object.entries(graph?.nodes ?? {}).find(([nodeId, node]) =>
    isTaskNode(nodeId, node.type)
  );
  const relations = taskNode ? [relation('binds', graphNodeEntityId(taskNode[0]), 'graph node')] : [];
  addEntity(registry, {
    id: taskEntityId(scenario.id),
    kind: 'task_object',
    label: scenario.task_spec.type,
    summary: 'Active task/workspace',
    scenario_id: scenario.id,
    stage_id: scenario.stage_id ?? null,
    selector: {
      namespace: 'task_object',
      compact: `task:${scenario.id}`,
      target_id: scenario.id,
      path: 'task_spec',
      role: 'editable',
      metadata: {},
    },
    relations,
    metadata: {
      task_spec: scenario.task_spec,
      graph_node_id: taskNode?.[0] ?? null,
      binding_state: taskNode ? 'bound' : 'unbound',
      inheritance_state: scenario.parent_scenario_id ? 'inherited_or_overridden' : 'owned',
      parent_scenario_id: scenario.parent_scenario_id ?? null,
    },
  }, true);
}

function addMechanicsEntities(
  registry: StudioScenarioEntityRegistry,
  scenario: StudioScenarioSpec,
  graph: GraphSpec | null
) {
  const mechanicsNodes = Object.entries(graph?.nodes ?? {}).filter(([nodeId, node]) =>
    isMechanicsNode(nodeId, node.type)
  );

  if (!scenario.biomechanics_spec && mechanicsNodes.length === 0) return;

  if (mechanicsNodes.length === 0) {
    addEntity(registry, {
      id: mechanicsEntityId(scenario.id),
      kind: 'mechanics_object',
      label: 'Mechanics',
      summary: 'Scenario mechanics',
      scenario_id: scenario.id,
      stage_id: scenario.stage_id ?? null,
      selector: {
        namespace: 'mechanics_object',
        compact: `mechanics:${scenario.id}`,
        target_id: scenario.id,
        path: 'biomechanics_spec',
        role: 'editable',
        metadata: {},
      },
      relations: [],
      metadata: {
        biomechanics_spec: scenario.biomechanics_spec ?? null,
        binding_state: 'unbound',
        inheritance_state: scenario.parent_scenario_id ? 'inherited_or_overridden' : 'owned',
        parent_scenario_id: scenario.parent_scenario_id ?? null,
      },
    }, true);
    return;
  }

  for (const [nodeId, node] of mechanicsNodes) {
    addEntity(registry, {
      id: mechanicsEntityId(scenario.id, nodeId),
      kind: 'mechanics_object',
      label: nodeId,
      summary: node.type,
      scenario_id: scenario.id,
      stage_id: scenario.stage_id ?? null,
      selector: {
        namespace: 'mechanics_object',
        compact: `mechanics:${scenario.id}.${nodeId}`,
        target_id: `${scenario.id}:${nodeId}`,
        path: nodeId,
        role: 'editable',
        metadata: {},
      },
      relations: [relation('binds', graphNodeEntityId(nodeId), 'graph node')],
      metadata: {
        node_id: nodeId,
        component_type: node.type,
        params: node.params,
        biomechanics_spec: scenario.biomechanics_spec ?? null,
        binding_state: 'bound',
        inheritance_state: scenario.parent_scenario_id ? 'inherited_or_overridden' : 'owned',
        parent_scenario_id: scenario.parent_scenario_id ?? null,
      },
    }, true);
  }
}

function addObjectiveEntities(
  registry: StudioScenarioEntityRegistry,
  objectiveSpec: StudioObjectiveSpec | null
) {
  for (const term of objectiveSpec?.terms ?? []) {
    addObjectiveTermEntity(registry, term);
  }
}

function addObjectiveTermEntity(
  registry: StudioScenarioEntityRegistry,
  term: StudioObjectiveTermSpec
) {
  const relations: StudioScenarioEntityRelation[] = [];
  const sourceEntityId = selectorToEntityId(term.source_selector);
  const targetEntityId = selectorToEntityId(term.target_selector);
  if (sourceEntityId) relations.push(relation('source', sourceEntityId));
  if (targetEntityId) relations.push(relation('target', targetEntityId));

  addEntity(registry, {
    id: objectiveEntityId(term.id),
    kind: 'objective_term',
    label: term.label,
    summary: `${term.role} · ${term.type_id}`,
    scenario_id: registry.scenario_id,
    stage_id: registry.stage_id,
    selector: term.source_selector ?? null,
    relations,
    metadata: { term },
  }, true);
}

export function buildScenarioEntityRegistry({
  scenario,
  graph,
}: {
  scenario: StudioScenarioSpec | null | undefined;
  graph?: GraphSpec | null;
}): StudioScenarioEntityRegistry {
  const resolvedGraph = graph ?? scenario?.graph ?? null;
  const registry: StudioScenarioEntityRegistry = {
    scenario_id: scenario?.id ?? null,
    stage_id: scenario?.stage_id ?? null,
    entities: {},
    root_entity_ids: [],
    metadata: {},
  };

  if (resolvedGraph) addGraphEntities(registry, resolvedGraph);
  if (scenario) {
    addTaskEntity(registry, scenario, resolvedGraph);
    addMechanicsEntities(registry, scenario, resolvedGraph);
    addObjectiveEntities(
      registry,
      isObjectiveSpec(scenario.objective_spec) ? scenario.objective_spec : null
    );
  }

  return registry;
}

export function getScenarioEntity(
  registry: StudioScenarioEntityRegistry,
  entityId: string | null | undefined
): StudioScenarioEntity | null {
  if (!entityId) return null;
  return registry.entities[entityId] ?? null;
}
