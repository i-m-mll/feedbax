import { describe, expect, it } from 'vitest';
import {
  addObjectiveTerm,
  createObjectiveTermFromAnchors,
  createObjectiveTerm,
  lossSpecFromObjectiveSpec,
  objectiveGraphPortTarget,
  objectiveSelectorSubpath,
  removeObjectiveTerm,
  selectorWithSubpath,
  setObjectiveTermEnabled,
  sourceSelectorForEntity,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import { graphNodeEntityId, graphPortEntityId, mechanicsEntityId } from '@/features/scenario/entities';
import type { ResolvedSceneAnchor } from '@/features/scenario/projections';
import type { StudioObjectiveSpec, StudioScenarioEntityRegistry } from '@/types/workspace';

const baseSpec: StudioObjectiveSpec = {
  schema_version: 'feedbax.studio.objective.v1',
  terms: [],
  legacy_loss_spec: null,
  metadata: {},
};

describe('scenario objective operations', () => {
  it('creates, updates, disables, removes, and lowers objective terms', () => {
    const term = createObjectiveTerm({
      spec: baseSpec,
      label: 'Endpoint',
      sourceSelector: {
        namespace: 'graph_port',
        compact: 'port:mechanics.effector',
        target_id: 'mechanics',
        path: 'effector',
        metadata: {},
      },
    });
    const withTerm = addObjectiveTerm(baseSpec, term);
    const updated = updateObjectiveTerm(withTerm, term.id, { weight: 2, penalty: 'l2' });

    expect(lossSpecFromObjectiveSpec(updated).children?.[term.id]).toMatchObject({
      label: 'Endpoint',
      selector: 'port:mechanics.effector',
      weight: 2,
      norm: 'l2',
    });

    const disabled = setObjectiveTermEnabled(updated, term.id, false);
    expect(lossSpecFromObjectiveSpec(disabled).children).toEqual({});

    const removed = removeObjectiveTerm(disabled, term.id);
    expect(removed.terms).toEqual([]);
  });

  it('keeps substate selectors related to their source graph port', () => {
    const portSelector = {
      namespace: 'graph_port' as const,
      compact: 'port:mechanics.effector',
      target_id: 'mechanics',
      path: 'effector',
      metadata: { direction: 'output' },
    };
    const substateSelector = selectorWithSubpath(portSelector, 'position');
    const term = createObjectiveTerm({
      spec: baseSpec,
      label: 'Effector position',
      sourceSelector: substateSelector,
    });
    const withTerm = addObjectiveTerm(baseSpec, term);

    expect(substateSelector).toMatchObject({
      namespace: 'state_path',
      compact: 'path:states.mechanics.effector.pos',
      metadata: {
        graph_port_node_id: 'mechanics',
        graph_port_name: 'effector',
        subpath: 'position',
      },
    });
    expect(objectiveSelectorSubpath(substateSelector)).toBe('position');
    expect(objectiveGraphPortTarget(substateSelector)).toEqual({
      nodeId: 'mechanics',
      direction: 'output',
      port: 'effector',
    });
    expect(lossSpecFromObjectiveSpec(withTerm).children?.[term.id]).toMatchObject({
      selector: 'path:states.mechanics.effector.pos',
    });
  });

  it('preserves target selectors, target values, and retention when lowering to training loss', () => {
    const sourceSelector = {
      namespace: 'graph_output' as const,
      compact: 'graph_output:effector',
      target_id: 'effector',
      path: 'effector',
      role: 'observed' as const,
      metadata: {
        graph_port_node_id: 'mechanics',
        graph_port_name: 'effector',
      },
    };
    const targetSelector = {
      namespace: 'task_data' as const,
      compact: 'task_data:targets.effector',
      target_id: 'scenario:train',
      path: 'targets.effector',
      role: 'observed' as const,
      metadata: {},
    };
    const term = createObjectiveTerm({
      spec: baseSpec,
      label: 'Reach target',
      sourceSelector,
      targetSelector,
    });
    const withTerm = updateObjectiveTerm(addObjectiveTerm(baseSpec, term), term.id, {
      target_value: [0, 0],
      retention: {
        mode: 'trajectory',
        reason: 'loss',
        metadata: {},
      },
      temporal_selector: { mode: 'sum' },
    });

    expect(lossSpecFromObjectiveSpec(withTerm).children?.[term.id]).toMatchObject({
      selector: 'graph_output:effector',
      target_selector: 'task_data:targets.effector',
      target_value: [0, 0],
      retention: {
        mode: 'trajectory',
        reason: 'loss',
      },
      time_agg: { mode: 'sum' },
    });
  });

  it('preserves matrix-quadratic payloads when lowering to training loss', () => {
    const sourceSelector = {
      namespace: 'graph_output' as const,
      compact: 'graph_output:effector',
      target_id: 'effector',
      path: 'effector',
      role: 'observed' as const,
      metadata: {},
    };
    const term = createObjectiveTerm({
      spec: baseSpec,
      label: 'Terminal quadratic',
      sourceSelector,
    });
    const withTerm = updateObjectiveTerm(addObjectiveTerm(baseSpec, term), term.id, {
      type_id: 'MatrixQuadraticLoss',
      matrix: [
        [2, 0.5],
        [0.5, 4],
      ],
      matrix_kind: 'dense',
      temporal_selector: { mode: 'final' },
    });

    expect(lossSpecFromObjectiveSpec(withTerm).children?.[term.id]).toMatchObject({
      type: 'MatrixQuadraticLoss',
      selector: 'graph_output:effector',
      matrix: [
        [2, 0.5],
        [0.5, 4],
      ],
      matrix_kind: 'dense',
      time_agg: { mode: 'final' },
    });
  });

  it('keeps schema-backed selector metadata on objective terms', () => {
    const sourceSelector = {
      namespace: 'state_path' as const,
      compact: 'path:states.decoder.readout',
      target_id: 'decoder',
      path: 'states.decoder.readout',
      expected_shape: ['time', 4],
      dtype: 'float32',
      units: 'a.u.',
      frame: 'decoder',
      role: 'observed' as const,
      metadata: {
        label: 'Decoder readout',
        value_schema: {
          id: 'value:path:states.decoder.readout',
          label: 'Decoder readout',
          kind: 'trajectory',
          dtype: 'float32',
          shape: ['time', 4],
          rank: 2,
          units: 'a.u.',
          frame: 'decoder',
          origin: 'declared',
          metadata: { temporal_support: 'trajectory' },
        },
      },
    };

    const term = createObjectiveTerm({
      spec: baseSpec,
      label: 'Decoder metric',
      sourceSelector,
    });
    const updated = updateObjectiveTerm(addObjectiveTerm(baseSpec, term), term.id, {
      role: 'metric',
    });

    expect(updated.terms[0]).toMatchObject({
      units: 'a.u.',
      metadata: {
        source_selector_compact: 'path:states.decoder.readout',
        source_value_schema_id: 'value:path:states.decoder.readout',
        source_dtype: 'float32',
        source_shape: ['time', 4],
        source_frame: 'decoder',
      },
    });
  });

  it('uses explicit graph port selectors instead of inferring from graph nodes', () => {
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
          metadata: {
            node_id: 'mechanics',
            output_ports: ['effector', 'state'],
          },
        },
        [graphPortEntityId('mechanics', 'output', 'effector')]: {
          id: graphPortEntityId('mechanics', 'output', 'effector'),
          kind: 'graph_port',
          label: 'mechanics.effector',
          summary: 'Output port',
          scenario_id: 'scenario:train',
          stage_id: 'stage:train',
          selector: {
            namespace: 'graph_port',
            compact: 'port:mechanics.effector',
            target_id: 'mechanics',
            path: 'effector',
            metadata: { direction: 'output' },
          },
          relations: [],
          metadata: {
            node_id: 'mechanics',
            port: 'effector',
            direction: 'output',
          },
        },
        [mechanicsEntityId('scenario:train', 'mechanics')]: {
          id: mechanicsEntityId('scenario:train', 'mechanics'),
          kind: 'mechanics_object',
          label: 'mechanics',
          summary: 'TwoLinkArm',
          scenario_id: 'scenario:train',
          stage_id: 'stage:train',
          selector: {
            namespace: 'mechanics_object',
            compact: 'mechanics:scenario:train.mechanics',
            target_id: 'scenario:train:mechanics',
            path: 'mechanics',
            metadata: {},
          },
          relations: [{ kind: 'binds', entity_id: graphNodeEntityId('mechanics'), label: 'graph node', metadata: {} }],
          metadata: {
            node_id: 'mechanics',
          },
        },
      },
    };

    expect(sourceSelectorForEntity(registry.entities[graphNodeEntityId('mechanics')], registry)).toBeNull();
    expect(sourceSelectorForEntity(registry.entities[mechanicsEntityId('scenario:train', 'mechanics')], registry)).toBeNull();
    expect(sourceSelectorForEntity(registry.entities[graphPortEntityId('mechanics', 'output', 'effector')], registry)).toMatchObject({
      compact: 'port:mechanics.effector',
      namespace: 'graph_port',
    });
  });

  it('authors objective terms from workspace anchors with canonical selectors and timing metadata', () => {
    const sourceAnchor: ResolvedSceneAnchor = {
      id: 'mechanics_object:scenario:train:mechanics::anchor:effector',
      entity_id: 'mechanics_object:scenario:train:mechanics',
      local_id: 'effector',
      label: 'Effector',
      semantic_role: 'endpoint',
      position: [0.7, 0],
      selectable: true,
      hoverable: true,
      interaction_roles: ['selectable', 'hoverable'],
      objective_roles: ['objective-source'],
      frame: 'world.xy',
      selector: {
        namespace: 'mechanics_object',
        compact: 'output:effector',
        target_id: 'effector',
        path: 'pos',
        metadata: {
          anchor_subpath: 'position',
          graph_port_node_id: 'mechanics',
          graph_port_name: 'effector',
          graph_port_direction: 'output',
        },
      },
      metadata: {},
    };
    const illustrativeTarget: ResolvedSceneAnchor = {
      id: 'task_object:scenario:train:task::anchor:sample-goal',
      entity_id: 'task_object:scenario:train:task',
      local_id: 'sample-goal',
      label: 'Sample goal',
      semantic_role: 'target',
      position: [0.5, 0.1],
      selectable: true,
      hoverable: true,
      interaction_roles: ['selectable', 'hoverable'],
      objective_roles: ['illustrative', 'canonical-for:goal'],
      frame: 'world.xy',
      selector: null,
      metadata: { illustrative: true },
    };
    const canonicalTarget: ResolvedSceneAnchor = {
      id: 'task_object:scenario:train:task::anchor:goal',
      entity_id: 'task_object:scenario:train:task',
      local_id: 'goal',
      label: 'Goal',
      semantic_role: 'target',
      position: [0.8, 0],
      selectable: true,
      hoverable: true,
      interaction_roles: ['selectable', 'hoverable', 'editable'],
      objective_roles: ['objective-target'],
      frame: 'world.xy',
      selector: {
        namespace: 'task_data',
        compact: 'task_data:targets.effector.pos',
        target_id: 'scenario:train',
        path: 'targets.effector.pos',
        role: 'observed',
        metadata: { time_mask: { epochs: ['move'] }, discount: 'power', discount_exp: 6 },
      },
      metadata: {},
    };

    const result = createObjectiveTermFromAnchors({
      spec: baseSpec,
      sourceAnchor,
      targetAnchor: illustrativeTarget,
      anchors: [sourceAnchor, illustrativeTarget, canonicalTarget],
    });

    expect(result?.target.canonicalized).toBe(true);
    expect(result?.target.message).toContain('using canonical Goal');
    expect(result?.term.source_selector).toMatchObject({
      namespace: 'state_path',
      compact: 'path:states.mechanics.effector.pos',
      metadata: {
        graph_port_node_id: 'mechanics',
        graph_port_name: 'effector',
      },
    });
    expect(result?.term.target_selector).toMatchObject({
      namespace: 'task_data',
      compact: 'task_data:targets.effector.pos',
    });
    expect(result?.term.metadata.target_timing).toEqual({
      time_mask: { epochs: ['move'] },
      discount: 'power',
      discount_exp: 6,
    });
  });
});
