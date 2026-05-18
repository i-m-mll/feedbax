import { describe, expect, it } from 'vitest';
import {
  addObjectiveTerm,
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
});
