import { describe, expect, it } from 'vitest';
import { graphNodeEntityId, objectiveEntityId } from '@/features/scenario/entities';
import {
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
  },
};

describe('scenario projection helpers', () => {
  it('builds workspace items from task, mechanics, and spatial objectives', () => {
    expect(workspaceProjectionItems(registry).map((item) => item.entity_id)).toEqual([
      'mechanics_object:scenario:train:mechanics',
      'task_object:scenario:train:task',
      objectiveEntityId('endpoint'),
    ]);
  });

  it('lists objectives and related entities', () => {
    expect(objectiveProjectionItems(registry).map((item) => item.label)).toEqual(['Endpoint']);
    expect(
      relatedProjectionItems(registry, 'mechanics_object:scenario:train:mechanics').map(
        (item) => item.entity_id
      )
    ).toEqual([graphNodeEntityId('mechanics'), objectiveEntityId('endpoint')]);
  });
});
