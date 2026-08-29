import { describe, expect, it } from 'vitest';
import {
  buildScenarioEntityRegistry,
  entityIdFromGraphSelection,
  graphEdgeEntityId,
  graphEdgeId,
  graphNodeEntityId,
  graphPortEntityId,
  mechanicsEntityId,
  objectiveEntityId,
  parseGraphPortEntityId,
  probeEntityId,
  retainedObservableEntityId,
  selectorToEntityId,
  stateFlowEdgeId,
  taskBindingEntityId,
  taskEntityId,
  taskDataEntityId,
} from '@/features/scenario/entities';
import type { GraphSpec } from '@/types/graph';
import type { StudioScenarioSpec } from '@/types/workspace';

const graph: GraphSpec = {
  nodes: {
    network: {
      type: 'Network',
      params: {},
      input_ports: ['input'],
      output_ports: ['output'],
    },
    mechanics: {
      type: 'TwoLinkArm',
      params: {},
      input_ports: ['force'],
      output_ports: ['effector'],
    },
  },
  wires: [
    {
      source_node: 'network',
      source_port: 'output',
      target_node: 'mechanics',
      target_port: 'force',
    },
  ],
  input_ports: [],
  output_ports: ['effector'],
  input_bindings: {},
  output_bindings: {},
  taps: [
    {
      id: 'tap:effector',
      type: 'probe',
      position: { afterNode: 'mechanics' },
      paths: { effector: 'effector' },
    },
  ],
  retained_observables: [
    {
      id: 'obs:effector',
      label: 'Effector trajectory',
      target: {
        kind: 'graph_output',
        selector: 'graph_output:effector',
        node_id: 'mechanics',
        port: 'effector',
        metadata: {},
      },
      retention: {
        mode: 'trajectory',
        metadata: {},
      },
      value_schema: {
        id: 'value:obs:effector',
        label: 'Effector trajectory',
        kind: 'retained_observable',
        shape: ['time', 2],
        origin: 'declared',
        metadata: {},
      },
      metadata: {},
    },
  ],
};

const scenario: StudioScenarioSpec = {
  id: 'scenario:train',
  schema_version: 'feedbax.spec.studio.scenario.v3',
  label: 'Training scenario',
  stage_id: 'stage:train',
  parent_scenario_id: null,
  training_spec: null,
  task_spec: { type: 'ReachingTask', params: { n_targets: 8 } },
  task_binding_spec: {
    schema_version: 'feedbax.studio.task_bindings.v2',
    exposed_data: [
      {
        id: 'inputs',
        label: 'Inputs',
        kind: 'signal',
        path: 'inputs',
        bindable: true,
        metadata: {},
      },
      {
        id: 'targets',
        label: 'Targets',
        kind: 'target',
        path: 'targets',
        bindable: false,
        metadata: {},
      },
    ],
    bindings: [
      {
        id: 'task:inputs->network:input',
        source_data_id: 'inputs',
        target_node_id: 'network',
        target_port: 'input',
        role: 'model_input',
        metadata: {},
      },
    ],
    metadata: {},
  },
  objective_spec: {
    schema_version: 'feedbax.studio.objective.v1',
    terms: [
      {
        id: 'endpoint',
        type_id: 'TargetStateLoss',
        label: 'Endpoint',
        role: 'loss',
        source_selector: {
          namespace: 'graph_port',
          compact: 'port:mechanics.effector',
          target_id: 'mechanics',
          path: 'effector',
          metadata: {},
        },
        target_selector: null,
        operator: 'distance',
        penalty: 'squared_l2',
        temporal_selector: { mode: 'final' },
        weight: 1,
        metadata: {},
      },
    ],
    metadata: {},
  },
  probe_specs: [],
  temporal_spec: null,
  biomechanics_spec: null,
  analysis_spec: null,
  report_spec: null,
  validation: {
    valid: null,
    checked_at: null,
    errors: [],
    warnings: [],
    metadata: {},
  },
  ui_state: {},
  metadata: {},
};

describe('graph port entity ids', () => {
  it('parses graph port entity ids with colon-containing ports', () => {
    expect(parseGraphPortEntityId('graph_port:network:input:feedback:alias')).toEqual({
      nodeId: 'network',
      direction: 'input',
      port: 'feedback:alias',
    });
    expect(parseGraphPortEntityId('graph_node:network')).toBeNull();
  });
});

describe('scenario entity registry', () => {
  it('derives graph, task, mechanics, probe, and objective entities', () => {
    const registry = buildScenarioEntityRegistry({ scenario, graph });
    const wire = graph.wires[0];

    expect(registry.entities[graphNodeEntityId('network')]).toMatchObject({
      kind: 'graph_node',
      label: 'network',
      summary: 'Network',
    });
    expect(registry.entities[graphPortEntityId('mechanics', 'output', 'effector')]).toMatchObject({
      kind: 'graph_port',
      selector: { compact: 'port:mechanics.effector' },
    });
    expect(registry.entities[graphEdgeEntityId(graphEdgeId(wire))]).toMatchObject({
      kind: 'graph_edge',
      metadata: {
        edge_type: 'port_wire',
        temporality: 'instant',
        recurrent_initializer: null,
      },
    });
    expect(registry.entities[graphEdgeEntityId(stateFlowEdgeId('network', 'mechanics'))]).toMatchObject({
      kind: 'graph_edge',
      label: 'network → mechanics',
      summary: 'Full state flow',
      selector: {
        namespace: 'state_path',
        compact: 'path:state:network->mechanics',
      },
      metadata: { edge_type: 'state_flow' },
    });
    expect(registry.entities[probeEntityId('tap:effector')]).toMatchObject({
      kind: 'probe',
      summary: 'effector',
    });
    expect(registry.entities[retainedObservableEntityId('obs:effector')]).toMatchObject({
      kind: 'retained_observable',
      label: 'Effector trajectory',
      summary: 'Captured observable',
      selector: {
        namespace: 'graph_output',
        compact: 'graph_output:effector',
        metadata: {
          retention: { mode: 'trajectory' },
          graph_port_node_id: 'mechanics',
          graph_port_name: 'effector',
        },
      },
      relations: [{ kind: 'target', entity_id: graphPortEntityId('mechanics', 'output', 'effector') }],
    });
    expect(registry.entities[taskEntityId('scenario:train')]).toMatchObject({
      kind: 'task_object',
      label: 'ReachingTask',
      relations: [],
      metadata: { binding_state: 'scenario_boundary', inheritance_state: 'owned' },
    });
    expect(registry.entities[taskDataEntityId('scenario:train', 'inputs')]).toMatchObject({
      kind: 'task_data',
      selector: {
        namespace: 'task_data',
        compact: 'task_data:inputs',
      },
    });
    expect(registry.entities[taskBindingEntityId('task:inputs->network:input')]).toMatchObject({
      kind: 'task_binding',
      relations: [
        {
          kind: 'source',
          entity_id: taskDataEntityId('scenario:train', 'inputs'),
        },
        {
          kind: 'target',
          entity_id: graphPortEntityId('network', 'input', 'input'),
        },
      ],
    });
    expect(registry.entities[mechanicsEntityId('scenario:train', 'mechanics')]).toMatchObject({
      kind: 'mechanics_object',
      relations: [{ kind: 'binds', entity_id: graphNodeEntityId('mechanics') }],
      metadata: { binding_state: 'bound', inheritance_state: 'owned' },
    });
    expect(registry.entities[objectiveEntityId('endpoint')]).toMatchObject({
      kind: 'objective_term',
      label: 'Endpoint',
      relations: [
        {
          kind: 'source',
          entity_id: graphPortEntityId('mechanics', 'output', 'effector'),
        },
      ],
    });
  });

  it('maps graph selections to canonical entity ids', () => {
    expect(entityIdFromGraphSelection({ nodeId: 'network' })).toBe(graphNodeEntityId('network'));
    expect(entityIdFromGraphSelection({ tapId: 'tap:effector' })).toBe(
      probeEntityId('tap:effector')
    );
    expect(entityIdFromGraphSelection({ edgeId: 'state:network->mechanics' })).toBe(
      graphEdgeEntityId('state:network->mechanics')
    );
    expect(entityIdFromGraphSelection({})).toBeNull();
  });

  it('maps graph-port selectors back to the selected port direction', () => {
    const registry = buildScenarioEntityRegistry({ scenario, graph });
    const inputSelector = registry.entities[graphPortEntityId('mechanics', 'input', 'force')]
      .selector;
    const outputSelector = registry.entities[graphPortEntityId('mechanics', 'output', 'effector')]
      .selector;

    expect(inputSelector).toMatchObject({ metadata: { direction: 'input' } });
    expect(outputSelector).toMatchObject({ metadata: { direction: 'output' } });
    expect(selectorToEntityId(inputSelector)).toBe(
      graphPortEntityId('mechanics', 'input', 'force')
    );
    expect(selectorToEntityId(outputSelector)).toBe(
      graphPortEntityId('mechanics', 'output', 'effector')
    );
    expect(selectorToEntityId({
      namespace: 'state_path',
      compact: 'path:states.mechanics.effector.pos',
      target_id: 'mechanics',
      path: 'states.mechanics.effector.pos',
      metadata: {
        graph_port_node_id: 'mechanics',
        graph_port_name: 'effector',
        graph_port_direction: 'output',
        subpath: 'position',
      },
    })).toBe(graphPortEntityId('mechanics', 'output', 'effector'));
    expect(selectorToEntityId({
      namespace: 'state_path',
      compact: 'path:state:network->mechanics',
      target_id: 'mechanics',
      path: 'state',
      metadata: { state_flow_edge_id: 'state:network->mechanics' },
    })).toBe(graphEdgeEntityId('state:network->mechanics'));
    expect(selectorToEntityId({
      namespace: 'graph_edge',
      compact: 'edge:network.output->mechanics.force',
      target_id: 'network:output->mechanics:force',
      path: null,
      metadata: {},
    })).toBe(graphEdgeEntityId(graphEdgeId(graph.wires[0])));
    expect(selectorToEntityId({
      namespace: 'graph_output',
      compact: 'graph_output:effector',
      target_id: 'effector',
      path: 'effector',
      metadata: {
        graph_port_node_id: 'mechanics',
        graph_port_name: 'effector',
      },
    })).toBe(graphPortEntityId('mechanics', 'output', 'effector'));
    expect(selectorToEntityId({
      namespace: 'retained_observable',
      compact: 'probe:obs:effector',
      target_id: 'obs:effector',
      path: null,
      metadata: {},
    })).toBe(retainedObservableEntityId('obs:effector'));
    expect(selectorToEntityId({
      namespace: 'task_data',
      compact: 'task_data:inputs',
      target_id: 'scenario:train',
      path: 'inputs',
      metadata: {},
    })).toBe(taskDataEntityId('scenario:train', 'inputs'));
  });

  it('marks task and mechanics entities from child scenarios as inherited or overridden', () => {
    const registry = buildScenarioEntityRegistry({
      scenario: {
        ...scenario,
        id: 'scenario:eval',
        parent_scenario_id: 'scenario:train',
      },
      graph,
    });

    expect(registry.entities[taskEntityId('scenario:eval')]).toMatchObject({
      metadata: {
        binding_state: 'scenario_boundary',
        inheritance_state: 'inherited_or_overridden',
        parent_scenario_id: 'scenario:train',
      },
    });
    expect(registry.entities[mechanicsEntityId('scenario:eval', 'mechanics')]).toMatchObject({
      metadata: {
        binding_state: 'bound',
        inheritance_state: 'inherited_or_overridden',
        parent_scenario_id: 'scenario:train',
      },
    });
  });
});
