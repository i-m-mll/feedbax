import { useQuery } from '@tanstack/react-query';
import { sampleTaskTrials } from '@/api/studioAuthoring';
import type { ModelDocument, SampledTaskTrial, StudioTrialAuthoring, TaskDocument } from '@/generated/studioContracts';
import type { EpochBounds } from './taskModel';
import type { ShelfTask } from './shelfRows';
import type { SampledTrial, SignalSteps } from './trialSample';

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
  return {
    document, epochs, duration: view.n_steps * period,
    signals: task.signals.map(({ id, name, cells }) => ({
      id, name, cells: cells.map(({ first_epoch, last_epoch }) => ({ first: first_epoch, last: last_epoch })),
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

/** A trial the route drew, as the shelf, the Variables table and the Inspector read it; values are keyed by name. */
export function sampledTrial(view: StudioTrialAuthoring, task: TaskDocument, trial: SampledTaskTrial): SampledTrial {
  const period = view.period_seconds;
  return {
    intervals: trial.timeline.map(({ epoch, label, start, end }) => ({ id: epoch, name: label, start: start * period, end: end * period })),
    variables: Object.fromEntries(task.variables.flatMap(({ id, name }) =>
      name in trial.variables ? [[id, drawnText(trial.variables[name])]] : [])),
    signals: Object.fromEntries(task.signals.map(({ id, name }): [string, SignalSteps] => {
      const values = trial.inputs[name];
      const present = trial.present[name];
      const steps = Array.isArray(values) ? values : [];
      return [id, steps.map((value, step) => Array.isArray(present) && present[step] ? numbers(value) : null)];
    })),
  };
}

/** The `count` trials drawn under `seed` from the step's task document, when `enabled`. */
export function useTaskSamples(view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) {
  const task = view && taskDocumentOf(view);
  return useQuery({
    queryKey: ['task-trials', model, task, seed, count],
    queryFn: async ({ signal }) => {
      const response = await sampleTaskTrials({ model: model!, task: task!, seed, count }, signal);
      return response.trials.map((trial) => sampledTrial(view!, task!, trial));
    },
    enabled: enabled && !!task && !!model,
    staleTime: Infinity,
  });
}
