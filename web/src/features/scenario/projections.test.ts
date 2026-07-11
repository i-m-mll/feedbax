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
  schema_version: 'feedbax.spec.studio.scenario.v2',
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
          planar_chain: {
            frame_ids: ['world', 'link0', 'link1'],
            reference_pose: {
              coordinate_space: 'configuration',
              values: [0, 0],
            },
            pose_fallback: 'zero',
          },
          metadata: { chain_kind: 'two_link_arm' },
        },
      ],
      scale_invariant: false,
      reachability: {
        kind: 'radial',
        origin_anchor: 'shoulder',
        radius_binding: { kind: 'param_path', path: 'link_lengths' },
        radius_transform: 'sum_abs',
        label: 'arm reach',
        units: 'm',
      },
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

  it('uses provider reference configuration only when correlated pose data is absent', () => {
    const mechanics = components[0];
    const links = mechanics.representation!.elements![0];
    const representedMechanics: ComponentDefinition = {
      ...mechanics,
      representation: {
        ...mechanics.representation!,
        elements: [
          {
            ...links,
            planar_chain: {
              ...links.planar_chain!,
              reference_pose: {
                coordinate_space: 'configuration',
                values: [Math.PI / 2, 0],
              },
            },
          },
        ],
      },
    };
    const graphWithUndeclaredPoseParams: GraphSpec = {
      ...graph,
      nodes: {
        mechanics: {
          ...graph.nodes.mechanics,
          params: {
            ...graph.nodes.mechanics.params,
            rest_joint_angles: [0, 0],
            initial_angles: [0, 0],
          },
        },
      },
    };
    const representedScenario = { ...scenario, graph: graphWithUndeclaredPoseParams };
    const sceneRegistry = buildScenarioEntityRegistry({
      scenario: representedScenario,
      graph: graphWithUndeclaredPoseParams,
    });
    const authoring = buildResolvedScene({
      scenario: representedScenario,
      graph: graphWithUndeclaredPoseParams,
      registry: sceneRegistry,
      components: [representedMechanics, components[1]],
    });
    const mechanicsId = mechanicsEntityId(scenario.id, 'mechanics');
    const authoringArm = authoring.elements.find(
      (element) => element.entity_id === mechanicsId && element.local_id === 'links'
    );

    expect(authoringArm?.geometry.kind).toBe('polyline');
    if (authoringArm?.geometry.kind !== 'polyline') throw new Error('expected arm geometry');
    expect(authoringArm.geometry.points[1][0]).toBeCloseTo(0);
    expect(authoringArm.geometry.points[1][1]).toBeCloseTo(0.3);
    expect(authoringArm.geometry.points[2][0]).toBeCloseTo(0);
    expect(authoringArm.geometry.points[2][1]).toBeCloseTo(0.7);

    const dataBacked = buildResolvedScene({
      scenario: representedScenario,
      graph: graphWithUndeclaredPoseParams,
      registry: sceneRegistry,
      components: [representedMechanics, components[1]],
      poseValues: { 'mechanics::output:state': [0, 0] },
      requirePoseValues: true,
    });
    const dataBackedArm = dataBacked.elements.find(
      (element) => element.entity_id === mechanicsId && element.local_id === 'links'
    );

    expect(dataBackedArm?.geometry).toEqual({
      kind: 'polyline',
      points: [[0, 0], [0.3, 0], [0.7, 0]],
    });
  });

  it('validates reachability from provider metadata without component name sniffing', () => {
    const genericGraph: GraphSpec = {
      ...graph,
      nodes: {
        mechanics: { ...graph.nodes.mechanics, type: 'CustomRadialPlant' },
      },
    };
    const genericScenario = { ...scenario, graph: genericGraph };
    const sceneRegistry = buildScenarioEntityRegistry({
      scenario: genericScenario,
      graph: genericGraph,
    });
    const scene = buildResolvedScene({
      scenario: genericScenario,
      graph: genericGraph,
      registry: sceneRegistry,
      components: [{ ...components[0], name: 'CustomRadialPlant' }, components[1]],
    });

    expect(scene.validation).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: 'workspace_goal_out_of_reach',
          message: expect.stringContaining('arm reach 0.700 m'),
        }),
      ])
    );
  });

  it('resolves wired body-local muscle paths from host pose and changes under rewiring', () => {
    const muscleComponent: ComponentDefinition = {
      name: 'MusclePaths',
      category: 'Mechanics',
      description: 'Graph-bound muscle paths',
      param_schema: [],
      input_ports: ['angles'],
      output_ports: [],
      representation: {
        elements: [{
          id: 'muscles',
          archetype: 'muscle_path',
          frame_provider: { kind: 'from_input_port', input_port: 'angles' },
          metadata: {},
        }],
        muscle_path_geometry: {
          paths: [{
            id: 'flexor',
            points: [
              { frame: 'world', position: [0, 0] },
              { frame: 'link1', position: [0.1, 0] },
            ],
          }],
        },
      },
    };
    const dynamicGraph: GraphSpec = {
      ...graph,
      nodes: {
        hostA: { ...graph.nodes.mechanics, params: { link_lengths: [0.3, 0.4] } },
        hostB: { ...graph.nodes.mechanics, params: { link_lengths: [0.6, 0.2] } },
        muscles: { type: 'MusclePaths', params: {}, input_ports: ['angles'], output_ports: [] },
      },
      wires: [{ source_node: 'hostA', source_port: 'state', target_node: 'muscles', target_port: 'angles' }],
    };
    const dynamicScenario = { ...scenario, graph: dynamicGraph };
    const sceneRegistry = buildScenarioEntityRegistry({ scenario: dynamicScenario, graph: dynamicGraph });
    const posed = buildResolvedScene({
      scenario: dynamicScenario,
      graph: dynamicGraph,
      registry: sceneRegistry,
      components: [...components, muscleComponent],
      poseValues: { 'hostA::output:state': [Math.PI / 2, 0] },
      requirePoseValues: true,
    });
    const muscle = posed.elements.find((element) => element.local_id === 'muscles:flexor');
    expect(muscle?.geometry.kind).toBe('polyline');
    if (muscle?.geometry.kind !== 'polyline') throw new Error('expected muscle polyline');
    expect(muscle.geometry.points[1][0]).toBeCloseTo(0);
    expect(muscle.geometry.points[1][1]).toBeCloseTo(0.4);

    const rewiredGraph = {
      ...dynamicGraph,
      wires: [{ source_node: 'hostB', source_port: 'state', target_node: 'muscles', target_port: 'angles' }],
    };
    const rewiredScenario = { ...dynamicScenario, graph: rewiredGraph };
    const rewired = buildResolvedScene({
      scenario: rewiredScenario,
      graph: rewiredGraph,
      registry: buildScenarioEntityRegistry({ scenario: rewiredScenario, graph: rewiredGraph }),
      components: [...components, muscleComponent],
      poseValues: { 'hostB::output:state': [0, 0] },
      requirePoseValues: true,
    });
    const rewiredMuscle = rewired.elements.find((element) => element.local_id === 'muscles:flexor');
    expect(rewiredMuscle?.geometry).toEqual({ kind: 'polyline', points: [[0, 0], [0.7, 0]] });
  });

  it('resolves self-contained analytical paths from its declared chain and current pose', () => {
    const analytical: ComponentDefinition = {
      ...components[0],
      name: 'AnalyticalPlant',
      representation: {
        ...components[0].representation!,
        elements: [
          {
            ...components[0].representation!.elements![0],
            bindings: {
              ...components[0].representation!.elements![0].bindings,
              link_lengths: { kind: 'literal', value: [0.3, 0.4], dim: 2 },
            },
          },
          {
            id: 'muscles',
            archetype: 'muscle_path',
            frame_provider: { kind: 'from_representation_element', element_id: 'links' },
          },
        ],
        muscle_path_geometry: {
          paths: [{
            id: 'flexor',
            points: [
              { frame: 'world', position: [0, 0] },
              { frame: 'link1', position: [0.1, 0] },
            ],
          }],
        },
      },
    };
    const analyticalGraph: GraphSpec = {
      ...graph,
      nodes: {
        analytical: { type: 'AnalyticalPlant', params: {}, input_ports: [], output_ports: ['state'] },
      },
      wires: [],
    };
    const analyticalScenario = { ...scenario, graph: analyticalGraph };
    const registry = buildScenarioEntityRegistry({
      scenario: analyticalScenario,
      graph: analyticalGraph,
    });
    const authoring = buildResolvedScene({
      scenario: analyticalScenario,
      graph: analyticalGraph,
      registry,
      components: [analytical, components[1]],
    });
    const authoringPath = authoring.elements.find(
      (element) => element.local_id === 'muscles:flexor'
    );
    expect(authoringPath?.geometry).toEqual({
      kind: 'polyline',
      points: [[0, 0], [0.4, 0]],
    });
    expect(authoring.validation).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'workspace_unresolved_frame_provider' }),
    ]));

    const posed = buildResolvedScene({
      scenario: analyticalScenario,
      graph: analyticalGraph,
      registry,
      components: [analytical, components[1]],
      poseValues: { 'analytical::output:state': [Math.PI / 2, 0] },
      requirePoseValues: true,
    });
    const posedPath = posed.elements.find((element) => element.local_id === 'muscles:flexor');
    expect(posedPath?.geometry.kind).toBe('polyline');
    if (posedPath?.geometry.kind !== 'polyline') throw new Error('expected posed muscle path');
    expect(posedPath.geometry.points[1][0]).toBeCloseTo(0);
    expect(posedPath.geometry.points[1][1]).toBeCloseTo(0.4);
  });

  it('reports missing same-entity chain elements and pose selectors', () => {
    const selfBound: ComponentDefinition = {
      ...components[0],
      name: 'SelfBoundPlant',
      representation: {
        ...components[0].representation!,
        elements: [
          {
            id: 'muscles',
            archetype: 'muscle_path',
            frame_provider: { kind: 'from_representation_element', element_id: 'missing' },
          },
        ],
        muscle_path_geometry: {
          paths: [{ id: 'path', points: [
            { frame: 'world', position: [0, 0] },
            { frame: 'link0', position: [0.1, 0] },
          ] }],
        },
      },
    };
    const selfGraph: GraphSpec = {
      ...graph,
      nodes: { self: { type: 'SelfBoundPlant', params: {}, input_ports: [], output_ports: ['state'] } },
      wires: [],
    };
    const selfScenario = { ...scenario, graph: selfGraph };
    const registry = buildScenarioEntityRegistry({ scenario: selfScenario, graph: selfGraph });
    const missingElement = buildResolvedScene({
      scenario: selfScenario,
      graph: selfGraph,
      registry,
      components: [selfBound, components[1]],
    });
    expect(missingElement.validation).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'workspace_muscle_path_host_element_missing' }),
    ]));

    const missingSelector: ComponentDefinition = {
      ...selfBound,
      representation: {
        ...selfBound.representation!,
        elements: [
          {
            id: 'links',
            archetype: 'planar_chain',
            bindings: { link_lengths: { kind: 'literal', value: [0.3, 0.4] } },
            planar_chain: { frame_ids: ['world', 'link0', 'link1'], pose_fallback: 'zero' },
          },
          {
            id: 'muscles',
            archetype: 'muscle_path',
            frame_provider: { kind: 'from_representation_element', element_id: 'links' },
          },
        ],
      },
    };
    const selectorError = buildResolvedScene({
      scenario: selfScenario,
      graph: selfGraph,
      registry,
      components: [missingSelector, components[1]],
    });
    expect(selectorError.validation).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'workspace_muscle_path_host_binding_invalid' }),
    ]));
  });

  it('reports unresolved provider, host, pose, and frame without synthesizing geometry', () => {
    const baseMuscle: ComponentDefinition = {
      name: 'MusclePaths', category: 'Mechanics', description: 'paths',
      param_schema: [], input_ports: ['angles'], output_ports: [],
      representation: {
        elements: [{ id: 'muscles', archetype: 'muscle_path', metadata: {} }],
        muscle_path_geometry: { paths: [{ id: 'bad', points: [
          { frame: 'world', position: [0, 0] }, { frame: 'link9', position: [1, 0] },
        ] }] },
      },
    };
    const dynamicGraph: GraphSpec = {
      ...graph,
      nodes: {
        host: graph.nodes.mechanics,
        muscles: { type: 'MusclePaths', params: {}, input_ports: ['angles'], output_ports: [] },
      },
      wires: [{ source_node: 'host', source_port: 'state', target_node: 'muscles', target_port: 'angles' }],
    };
    const dynamicScenario = { ...scenario, graph: dynamicGraph };
    const sceneRegistry = buildScenarioEntityRegistry({ scenario: dynamicScenario, graph: dynamicGraph });
    const missingProvider = buildResolvedScene({
      scenario: dynamicScenario, graph: dynamicGraph, registry: sceneRegistry,
      components: [...components, baseMuscle],
    });
    expect(missingProvider.validation).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'workspace_muscle_path_frame_provider_missing', severity: 'error' }),
    ]));

    const boundMuscle: ComponentDefinition = {
      ...baseMuscle,
      representation: {
        ...baseMuscle.representation!,
        elements: [{
          id: 'muscles', archetype: 'muscle_path', metadata: {},
          frame_provider: { kind: 'from_input_port', input_port: 'angles' },
        }],
      },
    };
    const missingPose = buildResolvedScene({
      scenario: dynamicScenario, graph: dynamicGraph, registry: sceneRegistry,
      components: [...components, boundMuscle], poseValues: {}, requirePoseValues: true,
    });
    expect(missingPose.validation).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'workspace_muscle_path_host_pose_missing', severity: 'error' }),
    ]));
    const unknownFrame = buildResolvedScene({
      scenario: dynamicScenario, graph: dynamicGraph, registry: sceneRegistry,
      components: [...components, boundMuscle],
      poseValues: { 'host::output:state': [0, 0] }, requirePoseValues: true,
    });
    expect(unknownFrame.validation).toEqual(expect.arrayContaining([
      expect.objectContaining({ type: 'workspace_muscle_path_unknown_frame', severity: 'error' }),
    ]));
    expect(unknownFrame.elements.some((element) => element.local_id === 'muscles:bad')).toBe(false);
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
