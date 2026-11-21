import { expect, it } from 'vitest';
import type { Project, StudioProjectWindowMapRow, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { projectWindowPlan } from '@/features/projectFolder/projectWindowPlan';
import { runColumns } from './StudioRunTable';

const project: Project = { steps: {
  train: { kind: 'train', base: { inline: {} }, sweep: { axes: { seed: { path: 'seed', values: [0, 1] } } } },
  evaluate: {
    kind: 'evaluate', base: { inline: {} }, sweep: { axes: { subject: { input: 'subject' } } },
    inputs: { subject: { step: 'train', checkpoint: 'last' } },
  },
} };

function evaluateColumns(subjects: readonly StudioProjectWindowMapRow['coordinates'][]) {
  const row = (step: string, name: string, coordinates: StudioProjectWindowMapRow['coordinates']): StudioProjectWindowMapRow => ({
    key: { variant: 'base', step, row: name }, label: name, status: 'not_run', coordinates,
  });
  const trains = [0, 1].map((seed) => row('train', `seed=${seed}`, { seed: { value_id: String(seed), label: null, value: seed } }));
  const saved: StudioSavedTrainRuns = {
    revision: '0'.repeat(64), rows: [],
    window_map: {
      variants: ['base'],
      steps: ['train', 'evaluate'].map((step_id) => ({
        variant: 'base', step_id, kind: step_id as 'train' | 'evaluate', group: null, document_ids: {}, status_counts: {},
        inputs: step_id === 'evaluate' ? [{ id: 'subject', source_step_id: 'train', subset: null }] : [],
      })),
      rows: [...trains, ...subjects.map((coordinates, index) => row('evaluate', `subject=${index}`, coordinates))],
      edges: [],
      problems: [],
    },
  };
  const evaluate = projectWindowPlan(project, saved, 'base').steps.find(({ id }) => id === 'evaluate')!;
  return runColumns(evaluate, evaluate.runs).map(({ id }) => id);
}

it('draws the seed an Evaluate run carries from its subject, not the subject that seed already names', () => {
  expect(evaluateColumns([0, 1].map((seed) => ({
    seed: { value_id: String(seed), label: null, value: seed },
    subject: { value_id: `seed=${seed}`, label: null, value: `seed=${seed}` },
  })))).toEqual(['seed']);
});

it('draws the rank of a Best subject, which no carried column spells', () => {
  expect(evaluateColumns([1, 2].map((rank) => ({
    subject: { value_id: `rank-${rank}`, label: `Best ${rank}`, value: rank },
  })))).toEqual(['subject']);
});
