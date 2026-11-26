import { expect, it } from 'vitest';
import { mapAxes, mapRunCells, orderedMapSweeps } from '@/components/pipeline/pipelineMapLayout';
import { runColumns } from '@/components/pipeline/StudioRunTable';
import type { Project, StudioProjectWindowMapRow, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { projectWindowPlan } from './projectWindowPlan';

type Row = StudioProjectWindowMapRow;
const seed = (value: number): Row['coordinates'] => ({ seed: { value_id: String(value), label: null, value, axis_value: String(value) } });
const row = (step: string, name: string, fields: Partial<Row> = {}): Row => ({
  key: { variant: 'base', step, row: name }, label: name, status: 'not_run', coordinates: {}, ...fields,
});
const trainRows = (step: string, seeds: readonly number[]) => seeds.map((value) => row(step, `seed=${value}`, { coordinates: seed(value) }));
/** An Evaluate row reading `train`'s run at `value`, carrying its seed as the planner sends it. */
const reading = (train: string, value: number, member = `seed=${value}`) => row('evaluate', `${train}-${value}`, {
  coordinates: seed(value), input_axes: { subject: { value_id: member, label: `seed=${value}`, carries: ['seed'], axis_value: member } },
});

/** A step's axes as the service lists them, for rows whose axis values are already their gathered values. */
function axes(rows: readonly Row[]) {
  const values = new Map<string, Map<string, string>>();
  for (const { coordinates, input_axes } of rows) {
    for (const [axis, { axis_value, label, value_id }] of Object.entries({ ...coordinates, ...input_axes })) {
      values.set(axis, (values.get(axis) ?? new Map()).set(axis_value, label ?? value_id));
    }
  }
  return [...values].map(([id, entries]) => ({
    id, label: id, values: [...entries].map(([value, label]) => ({ id: value, label, value_label: label, rule: value })),
  }));
}

function evaluateStep(
  trains: readonly string[], rows: readonly Row[], edges: NonNullable<StudioSavedTrainRuns['window_map']>['edges'] = [],
  evaluateAxes = axes(rows),
) {
  const mapRows = [...trains.flatMap((id) => trainRows(id, [0, 1, 2])), ...rows];
  const project: Project = { steps: {
    ...Object.fromEntries(trains.map((id) => [id, { kind: 'train', base: { inline: {} } }])),
    evaluate: { kind: 'evaluate', base: { inline: {} }, sweep: { axes: { subject: { input: 'subject' } } } },
  } };
  const saved: StudioSavedTrainRuns = {
    revision: '0'.repeat(64), rows: [],
    window_map: {
      document_names: {},
      variants: ['base'],
      steps: [...trains, 'evaluate'].map((step_id) => ({
        variant: 'base', step_id, label: step_id, kind: step_id === 'evaluate' ? 'evaluate' : 'train', group: null, document_ids: {},
        status_counts: {}, axes: step_id === 'evaluate' ? evaluateAxes : axes(mapRows.filter(({ key }) => key.step === step_id)),
        inputs: step_id === 'evaluate'
          ? trains.map((source, index) => ({ id: trains.length > 1 ? `subject.${index}` : 'subject', source_step_id: source, subset: null }))
          : [],
      })),
      rows: mapRows,
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
    input_axes: { subject: { value_id: `rank-${rank}`, label: `Best ${rank}`, carries: [], axis_value: `rank-${rank}` } },
  })));
  expect(step.sweeps.map(({ id, values }) => [id, values.map(({ label }) => label)])).toEqual([['subject', ['Best 1', 'Best 2']]]);
  expect(runColumns(step, step.runs).map(({ id }) => id)).toEqual(['subject']);
});

it('draws the seeds of an Evaluate reading two Train steps once each, in order, a seed\'s runs in its column', () => {
  /** The service's seed values: `value` names seed 0 alone; the other keys name different seeds in the two steps. */
  const gathered: Record<number, string> = { 0: 'value', 1: '=1', 2: '=2', 3: '=3' };
  const readings = ([['train-a', [0, 2, 3]], ['train-b', [0, 1, 2, 3]]] as const).flatMap(([train, seeds]) =>
    seeds.map((value, index) => row('evaluate', `${train}-${value}`, {
      coordinates: { seed: { value_id: index ? `value-${index}` : 'value', label: null, value, axis_value: gathered[value] } },
      input_axes: { subject: { value_id: `${train}/${index}`, label: null, carries: ['seed'], axis_value: `${train}/${index}` } },
    })));
  const edges = readings.map((to) => ({
    from_key: { variant: 'base', step: to.key.row.slice(0, 7), row: `seed=${to.coordinates.seed.value}` }, to_key: to.key,
  }));
  const step = evaluateStep(['train-a', 'train-b'], readings, edges, [
    { id: 'seed', label: 'Seed', values: [0, 1, 2, 3].map((value) => ({ id: gathered[value], label: `Seed ${value}`, value_label: String(value), rule: String(value) })) },
    { id: 'subject', label: 'Subject', values: readings.map(({ input_axes }) => ({ id: input_axes!.subject.axis_value, label: '', value_label: '', rule: '' })) },
  ]);
  const seed = step.sweeps.find(({ id }) => id === 'seed')!;
  expect(seed.values.map(({ valueLabel }) => valueLabel)).toEqual(['0', '1', '2', '3']);
  const { cells } = mapRunCells(step.runs, mapAxes(orderedMapSweeps(step, {})));
  const column = (run: string) => cells.find(({ mark }) => mark.kind === 'run' && mark.run.label.endsWith(run))?.column;
  expect([column('train-a-0'), column('train-b-0')]).toEqual([0, 0]);
  expect([column('train-a-2'), column('train-b-2')]).toEqual([2, 2]);
});

// Catches: the step drawer reordering while a revision's plan is read, because the plan and the project list steps differently.
it('lists steps in one order whether or not the plan has been read', () => {
  const project: Project = { steps: Object.fromEntries(['train-2', 'analyze', 'train-10'].map((id) =>
    [id, { kind: 'train', base: { inline: {} } }])) };
  const saved: StudioSavedTrainRuns = {
    revision: '0'.repeat(64), rows: [],
    window_map: {
      document_names: {},
      variants: ['base'], rows: [], edges: [], problems: [],
      steps: ['train-10', 'train-2', 'analyze'].map((step_id) => ({
        variant: 'base', step_id, label: step_id, kind: 'train', group: null, document_ids: {}, status_counts: {}, axes: [], inputs: [],
      })),
    },
  };
  const order = (read?: StudioSavedTrainRuns) => projectWindowPlan(project, read, 'base').steps.map(({ id }) => id);
  expect(order(saved)).toEqual(['analyze', 'train-2', 'train-10']);
  expect(order(undefined)).toEqual(order(saved));
});
