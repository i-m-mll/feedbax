import { expect, it } from 'vitest';
import type { Project, StudioProjectWindowMapRow, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { projectWindowPlan } from './projectWindowPlan';

it('counts a step\'s chosen runs only, so a step whose chosen runs all succeeded reads as done', () => {
  const row = (name: string, status: StudioProjectWindowMapRow['status']): StudioProjectWindowMapRow => ({
    key: { variant: 'base', step: 'evaluate', row: name }, label: name, status, coordinates: {},
  });
  const project: Project = { steps: { evaluate: { kind: 'evaluate', base: { inline: {} } } } };
  const saved: StudioSavedTrainRuns = {
    revision: '0'.repeat(64), rows: [],
    window_map: {
      variants: ['base'],
      steps: [{ variant: 'base', step_id: 'evaluate', kind: 'evaluate', group: null, document_ids: {}, inputs: [],
        status_counts: { succeeded: 2, not_chosen: 1 } }],
      rows: [row('a', 'succeeded'), row('b', 'not_chosen'), row('c', 'succeeded')],
      edges: [],
      problems: [],
    },
  };
  const [step] = projectWindowPlan(project, saved, 'base').steps;
  expect(step.runs.map(({ state }) => state)).toEqual(['succeeded', 'not_chosen', 'succeeded']);
  expect(step.statusCounts).toEqual({ succeeded: 2 });
});
