import { expect, it } from 'vitest';
import { mapAxes, mapRunCells, orderedMapSweeps } from '@/components/pipeline/pipelineMapLayout';
import { runColumns } from '@/components/pipeline/StudioRunTable';
import type { Project, StudioProjectWindowMapRow, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { projectWindowPlan } from './projectWindowPlan';

type Row = StudioProjectWindowMapRow;
const seed = (value: number): Row['coordinates'] => ({ seed: { value_id: String(value), label: null, value } });
const row = (step: string, name: string, fields: Partial<Row> = {}): Row => ({
  key: { variant: 'base', step, row: name }, label: name, status: 'not_run', coordinates: {}, ...fields,
});
const trainRows = (step: string, seeds: readonly number[]) => seeds.map((value) => row(step, `seed=${value}`, { coordinates: seed(value) }));
/** An Evaluate row reading `train`'s run at `value`, carrying its seed as the planner sends it. */
const reading = (train: string, value: number, member = `seed=${value}`) => row('evaluate', `${train}-${value}`, {
  coordinates: seed(value), input_axes: { subject: { value_id: member, label: `seed=${value}`, carries: ['seed'] } },
});

function evaluateStep(
  trains: readonly string[], rows: readonly Row[], edges: NonNullable<StudioSavedTrainRuns['window_map']>['edges'] = [],
) {
  const project: Project = { steps: {
    ...Object.fromEntries(trains.map((id) => [id, { kind: 'train', base: { inline: {} } }])),
    evaluate: { kind: 'evaluate', base: { inline: {} }, sweep: { axes: { subject: { input: 'subject' } } } },
  } };
  const saved: StudioSavedTrainRuns = {
    revision: '0'.repeat(64), rows: [],
    window_map: {
      variants: ['base'],
      steps: [...trains, 'evaluate'].map((step_id) => ({
        variant: 'base', step_id, kind: step_id === 'evaluate' ? 'evaluate' : 'train', group: null, document_ids: {},
        status_counts: {},
        inputs: step_id === 'evaluate'
          ? trains.map((source, index) => ({ id: trains.length > 1 ? `subject.${index}` : 'subject', source_step_id: source, subset: null }))
          : [],
      })),
      rows: [...trains.flatMap((id) => trainRows(id, [0, 1, 2])), ...rows],
      edges,
      problems: [],
    },
  };
  return projectWindowPlan(project, saved, 'base').steps.find(({ id }) => id === 'evaluate')!;
}

it('draws an Evaluate step reading a seed sweep as one seed axis, its runs in one line', () => {
  const step = evaluateStep(['train'], [0, 1, 2].map((value) => reading('train', value)));
  const sweeps = orderedMapSweeps(step, {});
  expect(sweeps.map(({ id, values }) => [id, values.map(({ label }) => label)])).toEqual([['seed', ['0', '1', '2']]]);
  const { cells, rowCount, columnCount } = mapRunCells(step.runs, mapAxes(sweeps));
  expect([cells.length, rowCount, columnCount]).toEqual([3, 1, 3]);
  expect(runColumns(step, step.runs).map(({ id }) => id)).toEqual(['seed']);
});

it('draws the step and the seed for a subject listing two Train steps swept over seed', () => {
  const rows = ['train-a', 'train-b'].flatMap((train) => [0, 1, 2].map((value) => reading(train, value, `${train}/seed=${value}`)));
  const edges = rows.map((to) => ({
    from_key: { variant: 'base', step: to.key.row.split('-').slice(0, 2).join('-'), row: `seed=${to.coordinates.seed.value_id}` },
    to_key: to.key,
  }));
  const step = evaluateStep(['train-a', 'train-b'], rows, edges);
  expect(step.sweeps.map(({ id }) => id).sort()).toEqual(['model', 'seed']);
  const { cells, rowCount, columnCount } = mapRunCells(step.runs, mapAxes(orderedMapSweeps(step, {})));
  expect([cells.length, rowCount * columnCount]).toEqual([6, 6]);
  expect(runColumns(step, step.runs).map(({ id }) => id).sort()).toEqual(['model', 'seed']);
});

it('draws the rank of a Best subject, which nothing the members carry tells apart', () => {
  const step = evaluateStep(['train'], [1, 2].map((rank) => row('evaluate', `subject=rank-${rank}`, {
    input_axes: { subject: { value_id: `rank-${rank}`, label: `Best ${rank}`, carries: [] } },
  })));
  expect(step.sweeps.map(({ id, values }) => [id, values.map(({ label }) => label)])).toEqual([['subject', ['Best 1', 'Best 2']]]);
  expect(runColumns(step, step.runs).map(({ id }) => id)).toEqual(['subject']);
});
