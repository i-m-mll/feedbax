import { describe, expect, it } from 'vitest';
import type { ComponentDefinition } from '@/types/components';
import type { GraphSpec } from '@/types/graph';
import type { StudioScenarioSpec } from '@/types/workspace';
import {
  buildScenarioEntityRegistry,
  graphNodeEntityId,
  mechanicsEntityId,
  objectiveEntityId,
  taskEntityId,
} from '@/features/scenario/entities';
import {
  buildResolvedScene,
  objectiveProjectionItems,
  relatedProjectionItems,
  workspaceProjectionItems,
} from '@/features/scenario/projections';
import type { StudioScenarioEntityRegistry } from '@/types/workspace';

const registry: StudioScenarioEntityRegistry = {
  scenario_id: 'scenario:train',
  stage_id: 'stage:train',
  root_entity_ids: [],
  metadata: {},
  entities: {
    [graphNodeEntityId('mechanics')]: {
      id: graphNodeEntityId('mechanics'),
      kind: 'graph_node',
      label: 'mechanics',
      summary: 'TwoLinkArm',
      scenario_id: 'scenario:train',
      stage_id: 'stage:train',
      selector: null,
      relations: [],
      metadata: {},
    },
    'mechanics_object:scenario:train:mechanics': {
      id: 'mechanics_object:scenario:train:mechanics',
      kind: 'mechanics_object',
      label: 'mechanics',
      summary: 'TwoLinkArm',
      scenario_id: 'scenario:train',
      stage_id: 'stage:train',
      selector: null,
      relations: [{ kind: 'binds', entity_id: graphNodeEntityId('mechanics'), metadata: {} }],
      metadata: {},
    },
    'task_object:scenario:train:task': {
      id: 'task_object:scenario:train:task',
      kind: 'task_object',
      label: 'ReachingTask',
      summary: 'Active task/workspace',
      scenario_id: 'scenario:train',
      stage_id: 'stage:train',
      selector: null,
      relations: [],
      metadata: {},
    },
    [objectiveEntityId('endpoint')]: {
      id: objectiveEntityId('endpoint'),
      kind: 'objective_term',
      label: 'Endpoint',
      summary: 'loss',
      scenario_id: 'scenario:train',
      stage_id: 'stage:train',
      selector: null,
      relations: [
        {
          kind: 'source',
          entity_id: 'mechanics_object:scenario:train:mechanics',
          metadata: {},
        },
      ],
      metadata: {},
    },
    'retained_observable:obs:effector': {
      id: 'retained_observable:obs:effector',
      kind: 'retained_observable',
      label: 'Effector trajectory',
      summary: 'Captured observable',
      scenario_id: 'scenario:train',
      stage_id: 'stage:train',
      selector: null,
      relations: [{ kind: 'target', entity_id: 'mechanics_object:scenario:train:mechanics', metadata: {} }],
      metadata: {},
    },
  },
};

describe('scenario projection helpers', () => {
  it('builds workspace items from task, mechanics, and spatial objectives', () => {
    expect(workspaceProjectionItems(registry).map((item) => item.entity_id)).toEqual([
      'mechanics_object:scenario:train:mechanics',
      'task_object:scenario:train:task',
      'retained_observable:obs:effector',
      objectiveEntityId('endpoint'),
    ]);
  });

  it('lists objectives and related entities', () => {
    expect(objectiveProjectionItems(registry).map((item) => item.label)).toEqual(['Endpoint']);
    expect(
      relatedProjectionItems(registry, 'mechanics_object:scenario:train:mechanics').map(
        (item) => item.entity_id
      )
    ).toEqual([
      graphNodeEntityId('mechanics'),
      objectiveEntityId('endpoint'),
      'retained_observable:obs:effector',
    ]);
  });
});

const graph: GraphSpec = {
  nodes: {
    mechanics: {
      type: 'TwoLinkArm',
      params: { link_lengths: [0.3, 0.4] },
      input_ports: ['force'],
      output_ports: ['effector', 'state'],
    },
  },
  wires: [],
  input_ports: [],
  output_ports: ['effector'],
  input_bindings: {},
  output_bindings: {},
  taps: [],
  retained_observables: [],
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
  task_spec: {
    type: 'ReachingTask',
    params: {
      workspace: [[-0.5, -0.5], [0.5, 0.5]],
      eval_reach_length: 0.8,
      eval_n_directions: 4,
    },
  },
  task_binding_spec: null,
  objective_spec: null,
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

const components: ComponentDefinition[] = [
  {
    name: 'TwoLinkArm',
    category: 'Mechanics',
    description: 'Arm',
    param_schema: [
      { name: 'link_lengths', type: 'array', default: [0.3, 0.33], required: false },
    ],
    input_ports: ['force'],
    output_ports: ['effector', 'state'],
    representation: {
      frame: 'world.xy',
      units: 'm',
      dim: 2,
      anchors: [
        {
          id: 'shoulder',
          semantic_role: 'joint',
          binding: { kind: 'literal', value: [0, 0], dim: 2 },
          metadata: { joint_index: 0 },
        },
        {
          id: 'elbow',
          semantic_role: 'joint',
          metadata: { joint_index: 1 },
        },
        {
          id: 'effector',
          semantic_role: 'endpoint',
          binding: {
            kind: 'selector',
            selector: {
              namespace: 'mechanics_object',
              compact: 'output:effector',
              target_id: 'effector',
              path: 'pos',
              metadata: {},
            },
            anchor_subpath: 'position',
            dim: 2,
          },
        },
      ],
      elements: [
        {
          id: 'links',
          archetype: 'planar_chain',
          anchors: ['shoulder', 'elbow', 'effector'],
          bindings: {
            joint_angles: {
              kind: 'selector',
              selector: {
                namespace: 'mechanics_object',
                compact: 'output:state',
                target_id: 'state',
                path: 'skeleton.angle',
                metadata: {},
              },
            },
            link_lengths: { kind: 'param_path', path: 'link_lengths' },
          },
          metadata: { chain_kind: 'two_link_arm' },
        },
      ],
      scale_invariant: false,
      metadata: {},
    },
  },
  {
    name: 'SimpleReaches',
    category: 'Tasks',
    description: 'Reach task',
    param_schema: [
      { name: 'workspace', type: 'bounds2d', default: [[-1, -1], [1, 1]], required: true },
      { name: 'eval_reach_length', type: 'float', default: 0.5, required: false },
      { name: 'eval_n_directions', type: 'int', default: 7, required: false },
    ],
    input_ports: [],
    output_ports: ['inputs', 'targets', 'inits', 'intervene'],
    representation: {
      frame: 'world.xy',
      units: 'm',
      dim: 2,
      anchors: [
        {
          id: 'start',
          semantic_role: 'origin',
          binding: { kind: 'trial_spec_path', path: 'inits.effector.pos', dim: 2 },
        },
        {
          id: 'goal',
          semantic_role: 'target',
          binding: { kind: 'trial_spec_path', path: 'targets.effector.pos', dim: 2 },
        },
      ],
      elements: [
        {
          id: 'workspace',
          archetype: 'region',
          bindings: { bounds: { kind: 'param_path', path: 'workspace', dim: 2 } },
          metadata: { region_kind: 'workspace_bounds' },
        },
        {
          id: 'reach-distribution',
          archetype: 'distribution_glyph',
          anchors: ['start', 'goal'],
          bindings: {
            workspace: { kind: 'param_path', path: 'workspace', dim: 2 },
            eval_reach_length: { kind: 'param_path', path: 'eval_reach_length' },
            eval_n_directions: { kind: 'param_path', path: 'eval_n_directions' },
          },
          metadata: { distribution: 'workspace_uniform' },
        },
      ],
      scale_invariant: true,
      metadata: {},
    },
  },
];

describe('resolved scene projection', () => {
  it('builds represented mechanics and task geometry from catalog metadata and live params', () => {
    const sceneRegistry = buildScenarioEntityRegistry({ scenario, graph });
    const scene = buildResolvedScene({ scenario, graph, registry: sceneRegistry, components });
    const mechanicsId = mechanicsEntityId(scenario.id, 'mechanics');
    const taskId = taskEntityId(scenario.id);
    const arm = scene.elements.find((element) => element.entity_id === mechanicsId);
    const region = scene.elements.find(
      (element) => element.entity_id === taskId && element.archetype === 'region'
    );
    const distribution = scene.elements.find(
      (element) => element.entity_id === taskId && element.archetype === 'distribution_glyph'
    );
    const goalAnchor = scene.anchors.find(
      (anchor) => anchor.entity_id === taskId && anchor.local_id === 'goal'
    );

    expect(arm?.geometry).toEqual({
      kind: 'polyline',
      points: [[0, 0], [0.3, 0], [0.7, 0]],
    });
    expect(region?.geometry).toEqual({ kind: 'bounds', min: [-0.5, -0.5], max: [0.5, 0.5] });
    expect(distribution?.geometry.kind).toBe('points');
    expect(goalAnchor?.position).toEqual([0.8, 0]);
    expect(goalAnchor?.selector).toMatchObject({
      namespace: 'task_data',
      compact: 'task_data:targets.effector.pos',
      target_id: scenario.id,
    });
    expect(goalAnchor?.objective_roles).toContain('objective-target');
    expect(
      scene.required_selectors.map((selector) => selector.compact).sort()
    ).toEqual(['output:effector', 'output:state']);
    expect(scene.elements.some((element) => element.archetype === 'objective_link')).toBe(false);
    expect(scene.validation.map((message) => message.type)).toContain('workspace_goal_out_of_reach');
  });

  it('projects objective links from objective terms instead of representation elements', () => {
    const scenarioWithObjective: StudioScenarioSpec = {
      ...scenario,
      objective_spec: {
        schema_version: 'feedbax.studio.objective.v1',
        legacy_loss_spec: null,
        metadata: {},
        terms: [
          {
            id: 'reach_goal',
            type_id: 'TargetStateLoss',
            label: 'Reach goal',
            role: 'loss',
            source_selector: {
              namespace: 'state_path',
              compact: 'path:states.mechanics.effector.pos',
              target_id: 'mechanics',
              path: 'states.mechanics.effector.pos',
              metadata: {
                graph_port_node_id: 'mechanics',
                graph_port_name: 'effector',
                graph_port_direction: 'output',
              },
            },
            target_selector: {
              namespace: 'task_data',
              compact: 'task_data:targets.effector.pos',
              target_id: scenario.id,
              path: 'targets.effector.pos',
              metadata: {},
            },
            operator: 'distance',
            penalty: 'squared_l2',
            temporal_selector: { mode: 'range', start: 10, end: 20 },
            weight: 1,
            units: 'm',
            validation: null,
            metadata: {},
          },
        ],
      },
    };
    const sceneRegistry = buildScenarioEntityRegistry({ scenario: scenarioWithObjective, graph });
    const scene = buildResolvedScene({
      scenario: scenarioWithObjective,
      graph,
      registry: sceneRegistry,
      components,
    });
    const objectiveElement = scene.elements.find(
      (element) => element.entity_id === objectiveEntityId('reach_goal')
    );

    expect(objectiveElement).toMatchObject({
      archetype: 'objective_link',
      geometry: { kind: 'link', points: [[0.7, 0], [0.8, 0]] },
      metadata: {
        timing: { mode: 'range', start: 10, end: 20 },
      },
    });
    expect(scene.entities.find((entity) => entity.id === objectiveEntityId('reach_goal'))).toMatchObject({
      kind: 'objective_term',
      element_ids: [objectiveElement?.id],
    });
  });

  it('creates selectable placeholders for represented entity kinds without catalog metadata', () => {
    const sceneRegistry = buildScenarioEntityRegistry({ scenario, graph });
    const scene = buildResolvedScene({ scenario, graph, registry: sceneRegistry, components: [] });

    expect(scene.entities.map((entity) => entity.id)).toContain(
      mechanicsEntityId(scenario.id, 'mechanics')
    );
    expect(scene.entities.map((entity) => entity.id)).toContain(taskEntityId(scenario.id));
    expect(scene.validation.map((message) => message.type)).toEqual([
      'workspace_unrepresented_entity',
      'workspace_unrepresented_entity',
    ]);
  });
});
