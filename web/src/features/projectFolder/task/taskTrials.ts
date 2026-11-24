import { useQuery } from '@tanstack/react-query';
import { sampleTaskTrials } from '@/api/studioAuthoring';
import type { ModelDocument, StudioTrialAuthoring, TaskDocument, TaskTrialSampleRequest, TaskTrialSampleResponse } from '@/generated/studioContracts';
import type { EpochBounds } from './taskModel';
import type { ShelfTask } from './shelfRows';
import type { TrialsUnavailable } from './trialSample';

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

type Drawn = TaskTrialSampleResponse['drawn'];

/** The route's answer as `kind`, or the Problems it answered with instead. */
function drawnAs<Kind extends 'trials' | 'aggregate'>(drawn: Drawn, kind: Kind): Extract<Drawn, { kind: Kind }> | TrialsUnavailable {
  if (drawn.kind === 'problems' || drawn.kind === kind) return drawn as Extract<Drawn, { kind: Kind }> | TrialsUnavailable;
  throw new Error(`The sampling route answered ${drawn.kind} where ${kind} was asked for.`);
}

/** The request that draws `count` trials under `seed` from the step's pairing: its Task, its Model, its bindings and its Objective. */
function sampleRequest(view: StudioTrialAuthoring, task: TaskDocument, model: ModelDocument, seed: number, count: number,
  aggregate: boolean): TaskTrialSampleRequest {
  return {
    model, task, seed, count, aggregate, input_overrides: view.input_overrides,
    ...view.objective && { objective: view.objective },
  };
}

function useDrawn<Shown>(view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  aggregate: boolean, enabled: boolean, read: (drawn: Drawn) => Shown) {
  const task = view && taskDocumentOf(view);
  const body = view && task && model ? sampleRequest(view, task, model, seed, count, aggregate) : undefined;
  return useQuery({
    queryKey: ['task-trials', body],
    queryFn: async ({ signal }) => read((await sampleTaskTrials(body!, signal)).drawn),
    enabled: enabled && !!body,
    staleTime: Infinity,
  });
}

/** The `count` trials drawn under `seed` from the step's pairing, when `enabled`. */
export const useTaskSamples = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) => useDrawn(view, model, seed, count, false, enabled, (drawn) => {
  const answer = drawnAs(drawn, 'trials');
  return answer.kind === 'problems' ? answer : answer.trials;
});

/** The aggregate of the `count` trials drawn under `seed` from the step's pairing, when `enabled`. */
export const useTaskBatch = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) => useDrawn(view, model, seed, count, true, enabled, (drawn) => drawnAs(drawn, 'aggregate'));
