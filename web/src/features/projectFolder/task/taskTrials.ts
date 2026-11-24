import { useQuery } from '@tanstack/react-query';
import { sampleTaskTrials } from '@/api/studioAuthoring';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import type { ModelDocument, StudioTrialAuthoring, TaskDocument, TaskTrialSampleRequest, TaskTrialSampleResponse } from '@/generated/studioContracts';
import type { EpochBounds } from './taskModel';
import type { ShelfTask } from './shelfRows';
import type { TrialsUnavailable } from './trialSample';

/** The task document a step's Trial authoring reads, when it is one. */
export const taskDocumentOf = (view: StudioTrialAuthoring): TaskDocument | undefined =>
  'epochs' in view.task ? view.task : undefined;

/**
 * A step's task document as the shelf draws it, once the task is paired with the model and so has steps, with the
 * terms of its Objective, held in `objectiveDocument`, that are on.
 */
export function shelfTask(view: StudioTrialAuthoring, document: string, objectiveDocument: string | undefined): ShelfTask | undefined {
  const task = taskDocumentOf(view);
  const period = view.period_seconds;
  if (!task || view.n_steps === null || period === null) return undefined;
  const ends = new Map(task.epochs.map((epoch) => [epoch.id, epoch.end]));
  const instances = view.epoch_instances ?? [];
  if (!instances.length) return undefined;
  const epochs = instances.map(({ id, name, steps }): EpochBounds => {
    const end = ends.get(id);
    return {
      id, name,
      ends: end?.kind === 'rest' ? 'rest' : steps.least < steps.greatest ? 'drawn' : 'length',
      least: steps.least * period, nominal: steps.nominal * period, greatest: steps.greatest * period,
    };
  });
  const order = task.epochs.map(({ id }) => id);
  const terms = (view.objective?.terms ?? []).flatMap(({ id, name, enabled }, index) =>
    enabled === false ? [] : [{ id, name: name ?? studioAccessibleName('term', index + 1) }]);
  return {
    document, duration: view.n_steps * period,
    epochs,
    ...objectiveDocument && { objective: { document: objectiveDocument, terms } },
    signals: task.signals.map(({ id, name, cells }) => ({
      id, name, covers: cells.flatMap(({ first_epoch, last_epoch }) => order.slice(order.indexOf(first_epoch), order.indexOf(last_epoch) + 1)),
    })),
  };
}

type Drawn = TaskTrialSampleResponse['drawn'];

/** The route's answer as `kind`, or the Problems it answered with instead. */
function drawnAs<Kind extends 'trials' | 'aggregate'>(drawn: Drawn, kind: Kind): Extract<Drawn, { kind: Kind }> | TrialsUnavailable {
  if (drawn.kind === 'problems' || drawn.kind === kind) return drawn as Extract<Drawn, { kind: Kind }> | TrialsUnavailable;
  throw new Error(`The sampling route answered ${drawn.kind} where ${kind} was asked for.`);
}

/** Which trials a request draws: the nominal trial, `count` samples under `seed`, or their aggregate. */
type Draw = { readonly nominal: true } | { readonly nominal: false; readonly seed: number; readonly count: number; readonly aggregate: boolean };

/** The request that draws `draw` from the step's pairing: its Task, its Model, its bindings and its Objective. */
function sampleRequest(view: StudioTrialAuthoring, task: TaskDocument, model: ModelDocument | undefined, draw: Draw): TaskTrialSampleRequest {
  return {
    task, input_bindings: view.input_bindings, ...draw,
    ...model && { model },
    ...view.objective && { objective: view.objective },
  };
}

function useDrawn<Shown>(view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, draw: Draw, enabled: boolean,
  read: (drawn: Drawn) => Shown) {
  const task = view && taskDocumentOf(view);
  const body = view && task ? sampleRequest(view, task, model, draw) : undefined;
  return useQuery({
    queryKey: ['task-trials', body],
    queryFn: async ({ signal }) => read((await sampleTaskTrials(body!, signal)).drawn),
    enabled: enabled && !!body,
    staleTime: Infinity,
  });
}

const records = (drawn: Drawn) => {
  const answer = drawnAs(drawn, 'trials');
  return answer.kind === 'problems' ? answer : answer.trials;
};

/** The step's nominal trial, as one record, when `enabled`. */
export const useNominalTrial = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, enabled: boolean) =>
  useDrawn(view, model, { nominal: true }, enabled, records);

/** The `count` trials drawn under `seed` from the step's pairing, when `enabled`. */
export const useTaskSamples = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) => useDrawn(view, model, { nominal: false, seed, count, aggregate: false }, enabled, records);

/** The aggregate of the `count` trials drawn under `seed` from the step's pairing, when `enabled`. */
export const useTaskBatch = (view: StudioTrialAuthoring | undefined, model: ModelDocument | undefined, seed: number, count: number,
  enabled: boolean) => useDrawn(view, model, { nominal: false, seed, count, aggregate: true }, enabled,
  (drawn) => drawnAs(drawn, 'aggregate'));
