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
  probeEntityId,
  selectorToEntityId,
  stateFlowEdgeId,
  taskBindingEntityId,
  taskEntityId,
  taskOutputEntityId,
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
};

const scenario: StudioScenarioSpec = {
  id: 'scenario:train',
  schema_version: 'feedbax.studio.scenario.v1',
  label: 'Training scenario',
  stage_id: 'stage:train',
  parent_scenario_id: null,
  graph,
  graph_ui_state: null,
  training_spec: null,
  task_spec: { type: 'ReachingTask', params: { n_targets: 8 } },
  task_binding_spec: {
    schema_version: 'feedbax.studio.task_bindings.v1',
    exposed_outputs: [
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
        source_output_id: 'inputs',
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

describe('scenario entity registry', () => {
  it('derives graph, task, mechanics, probe, and objective entities', () => {
    const registry = buildScenarioEntityRegistry({ scenario });
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
    expect(registry.entities[taskEntityId('scenario:train')]).toMatchObject({
      kind: 'task_object',
      label: 'ReachingTask',
      relations: [],
      metadata: { binding_state: 'scenario_boundary', inheritance_state: 'owned' },
    });
    expect(registry.entities[taskOutputEntityId('scenario:train', 'inputs')]).toMatchObject({
      kind: 'task_output',
      selector: {
        namespace: 'task_output',
        compact: 'task_output:inputs',
      },
    });
    expect(registry.entities[taskBindingEntityId('task:inputs->network:input')]).toMatchObject({
      kind: 'task_binding',
      relations: [
        {
          kind: 'source',
          entity_id: taskOutputEntityId('scenario:train', 'inputs'),
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
    const registry = buildScenarioEntityRegistry({ scenario });
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
      namespace: 'task_output',
      compact: 'task_output:inputs',
      target_id: 'scenario:train',
      path: 'inputs',
      metadata: {},
    })).toBe(taskOutputEntityId('scenario:train', 'inputs'));
  });

  it('marks task and mechanics entities from child scenarios as inherited or overridden', () => {
    const registry = buildScenarioEntityRegistry({
      scenario: {
        ...scenario,
        id: 'scenario:eval',
        parent_scenario_id: 'scenario:train',
      },
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
