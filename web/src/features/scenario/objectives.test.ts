import { describe, expect, it } from 'vitest';
import {
  addObjectiveTerm,
  createObjectiveTerm,
  lossSpecFromObjectiveSpec,
  removeObjectiveTerm,
  setObjectiveTermEnabled,
  sourceSelectorForEntity,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import { graphNodeEntityId } from '@/features/scenario/entities';
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

  it('infers a source selector from graph node entities', () => {
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
      },
    };

    expect(sourceSelectorForEntity(registry.entities[graphNodeEntityId('mechanics')], registry)).toMatchObject({
      compact: 'port:mechanics.effector',
      namespace: 'graph_port',
    });
  });
});
