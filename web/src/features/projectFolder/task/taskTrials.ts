import { useQuery } from '@tanstack/react-query';
import { sampleTaskTrials } from '@/api/studioAuthoring';
import type { ModelDocument, TrialRecord, StudioTrialAuthoring, TaskDocument, TrialAggregate, TaskTrialSampleResponse } from '@/generated/studioContracts';
import { epochInstances } from './repeats';
import type { EpochBounds } from './taskModel';
import type { ShelfTask } from './shelfRows';
import type { Percentiles, SampledTrial, SignalSteps, TrialBatch } from './trialSample';

/** The task document a step's Trial authoring reads, when it is one. */
export const taskDocumentOf = (view: StudioTrialAuthoring): TaskDocument | undefined =>
  'epochs' in view.task ? view.task : undefined;

/** A step's task document as the shelf draws it, once the task is paired with the model and so has steps. */
export function shelfTask(view: StudioTrialAuthoring, document: string): ShelfTask | undefined {
  const task = taskDocumentOf(view);
  if (!task || view.n_steps === null) return undefined;
  const period = view.period_seconds;
  const ends = new Map(task.epochs.map((epoch) => [epoch.id, epoch.end]));
  const epochs: EpochBounds[] = [];
  for (const { id, name, steps } of view.epochs) {
    if (!steps) return undefined;
    const end = ends.get(id);
    epochs.push({
      id, name,
      ends: end?.kind === 'rest' ? 'rest' : steps.least < steps.greatest ? 'drawn' : 'length',
      least: steps.least * period, nominal: steps.nominal * period, greatest: steps.greatest * period,
    });
  }
  const order = task.epochs.map(({ id }) => id);
  return {
    document, duration: view.n_steps * period,
    epochs: epochInstances(epochs, task.repeats ?? [], task.variables).map(({ epoch }) => epoch),
    signals: task.signals.map(({ id, name, cells }) => ({
      id, name, covers: cells.flatMap(({ first_epoch, last_epoch }) => order.slice(order.indexOf(first_epoch), order.indexOf(last_epoch) + 1)),
    })),
  };
}

const numbers = (value: unknown): number[] => Array.isArray(value) ? value.flatMap(numbers)
  : typeof value === 'boolean' ? [Number(value)] : typeof value === 'number' ? [value] : [];

function drawnText(value: unknown): string {
  if (typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return String(Number(value.toPrecision(3)));
  if (Array.isArray(value)) return `[${value.map(drawnText).join(', ')}]`;
  return '';
}

/** A trial the route drew, as the shelf, the Variables table and the Inspector read it; values are keyed by id. */
export function sampledTrial(view: StudioTrialAuthoring, task: TaskDocument, trial: TrialRecord): SampledTrial {
  const period = view.period_seconds;
  return {
    intervals: trial.timeline.map(({ epoch, label, start, end }) => ({ id: epoch, name: label, start: start * period, end: end * period })),
    variables: Object.fromEntries(task.variables.flatMap(({ id }) =>
      id in trial.variables ? [[id, drawnText(trial.variables[id])]] : [])),
    signals: Object.fromEntries(task.signals.map(({ id }): [string, SignalSteps] => {
      const values = trial.inputs[id];
      const present = trial.present[id];
      const steps = Array.isArray(values) ? values : [];
      return [id, steps.map((value, step) => Array.isArray(present) && present[step] ? numbers(value) : null)];
    })),
  };
}

/** The range of a variable's drawn values, component by component, as text: one value where they all agree. */
function rangeText(least: unknown, greatest: unknown): string {
  if (Array.isArray(least) && Array.isArray(greatest)) return `[${least.map((low, index) => rangeText(low, greatest[index])).join(', ')}]`;
  if (least === greatest) return drawnText(least);
  return typeof least === 'boolean' ? `${drawnText(least)}, ${drawnText(greatest)}` : `${drawnText(least)}…${drawnText(greatest)}`;
}

/** A batch's aggregate, as the shelf, the Variables table and the Inspector read it; values are keyed by id. */
export function trialBatch(task: TaskDocument, count: number, aggregate: TrialAggregate): TrialBatch {
  return {
    count,
    epochs: aggregate.epochs.map(({ epoch, label, share }) => ({ id: epoch, name: label, share })),
    variables: Object.fromEntries(task.variables.flatMap(({ id }) => {
      const range = aggregate.variables[id];
      return range ? [[id, rangeText(range.least, range.greatest)]] : [];
    })),
    signals: Object.fromEntries(task.signals.flatMap(({ id }) => {
      const signal = aggregate.signals[id];
      return signal ? [[id, {
        presence: signal.presence,
        ...signal.percentiles && { percentiles: signal.percentiles as (Percentiles[] | null)[] },
      }]] : [];
    })),
  };
}

type Drawn = TaskTrialSampleResponse['drawn'];

function drawnAs<Kind extends Drawn['kind']>(drawn: Drawn, kind: Kind): Extract<Drawn, { kind: Kind }> {
  if (drawn.kind === 'problems') throw new Error(drawn.problems.map(({ reason }) => reason).join('\n'));
  if (drawn.kind !== kind) throw new Error(`The sampling route answered ${drawn.kind} where ${kind} was asked for.`);
  return drawn as Extract<Drawn, { kind: Kind }>;
}

function useDrawn<Shown>(view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  aggregate: boolean, enabled: boolean, read: (task: TaskDocument, drawn: Drawn) => Shown) {
  const task = view && taskDocumentOf(view);
  return useQuery({
    queryKey: ['task-trials', model, task, view?.input_bindings, seed, count, aggregate],
    queryFn: async ({ signal }) => read(task!, (await sampleTaskTrials({ model: model!, task: task!, input_bindings: view?.input_bindings, seed, count, aggregate }, signal)).drawn),
    enabled: enabled && !!task && !!model,
    staleTime: Infinity,
  });
}

/** The `count` trials drawn under `seed` from the step's task document, when `enabled`. */
export const useTaskSamples = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) => useDrawn(view, model, seed, count, false, enabled,
  (task, drawn) => drawnAs(drawn, 'trials').trials.map((trial) => sampledTrial(view!, task, trial)));

/** The aggregate of the `count` trials drawn under `seed` from the step's task document, when `enabled`. */
export const useTaskBatch = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) => useDrawn(view, model, seed, count, true, enabled,
  (task, drawn) => trialBatch(task, count, drawnAs(drawn, 'aggregate')));
