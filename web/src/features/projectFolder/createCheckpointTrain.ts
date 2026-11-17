import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import type { PlanView } from '@/features/pipeline/planView';
import { selectedRuns } from '@/features/pipeline/selection';
import type { JsonValue, ProjectValueEdit, Step, StepReference, StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { planVariants, savedRunId, sharedRowTerms, stepSourceRows } from './projectWindowPlan';

const initializationPath = 'checkpoint_initialization';
const initializationLayer = 'initialize';

/**
 * `step` started from `source`'s checkpoint, initializing its model from it, or from its own Model when `source` is
 * null. `inherited` says whether the step's base already sets the initialization policy.
 */
export function withCheckpointStart(step: Step, inherited: boolean, source: StepReference | null): Step {
  const { checkpoint: _checkpoint, ...inputs } = step.inputs ?? {};
  const deltas = (step.deltas ?? []).filter(({ layer_id }) => layer_id !== initializationLayer);
  let base = step.base;
  if ('inline' in base) {
    const { [initializationPath]: _policy, ...payload } = base.inline;
    base = { inline: source ? { ...payload, [initializationPath]: 'initialize_from' } : payload };
  } else if (source || inherited) {
    deltas.push({
      layer_id: initializationLayer,
      patches: [source
        ? { path: initializationPath, op: inherited ? 'replace' : 'add', value: 'initialize_from' }
        : { path: initializationPath, op: 'remove' }],
      ...(inherited && { acknowledges_ancestor_paths: [initializationPath] }),
    });
  }
  const { inputs: _inputs, deltas: _deltas, ...rest } = step;
  const started = { ...inputs, ...(source && { checkpoint: source }) };
  return {
    ...rest, base,
    ...(Object.keys(started).length && { inputs: started }),
    ...(deltas.length && { deltas }),
  };
}

/** The edits that give an authored step `withCheckpointStart`'s start. */
export function checkpointStartEdits(step: Step, inherited: boolean, source: StepReference | null): ProjectValueEdit[] {
  const next = withCheckpointStart(step, inherited, source);
  const changed = (['base', 'inputs', 'deltas'] as const)
    .filter((key) => JSON.stringify(step[key]) !== JSON.stringify(next[key]));
  // A removal from an authored file takes its last key, so removals go last-first.
  const order = Object.keys(step);
  const removed = changed.filter((key) => next[key] === undefined)
    .sort((left, right) => order.indexOf(right) - order.indexOf(left));
  return [
    ...removed.map((key): ProjectValueEdit => ({ op: 'remove', path: '/' + key })),
    ...changed.filter((key) => next[key] !== undefined).map((key): ProjectValueEdit => ({
      op: step[key] === undefined ? 'add' : 'replace', path: '/' + key, value: next[key] as JsonValue,
    })),
  ];
}

export function trainSourceSubset(
  source: PipelineMapStepSource,
  view: PlanView,
  projection: StudioSavedTrainRuns,
  sharedProject: boolean,
) {
  const { stepId, selection } = source;
  const step = view.steps.find(({ id }) => id === stepId);
  if (step?.kind !== 'train' || !step.runs.length) throw new Error('Choose a runnable Train step.');
  const members = selection ? selectedRuns(step, selection) : step.runs;
  const variant = planVariants(projection)[0];
  if (!variant || !members.length || members.some((member) => member.variant !== variant)) throw new ProjectEditConflictError();
  let where: SubsetTerms | undefined;
  if (selection) {
    const selected = new Set(members.map(({ id }) => id));
    if (selection.kind === 'values') {
      where = { axes: Object.fromEntries(selection.terms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]])) };
    } else if (sharedProject) {
      where = sharedRowTerms(step, members);
      if (!where) throw new ProjectEditConflictError();
    } else {
      const rows = stepSourceRows(projection, stepId, variant).filter((row) =>
        selected.has(savedRunId({ ...row, step: stepId })));
      if (rows.length !== selected.size) throw new ProjectEditConflictError();
      where = { rows };
    }
  }
  return { step, members, where };
}

export async function createCheckpointTrain(
  editor: ProjectEditController,
  source: PipelineMapStepSource,
  view: PlanView,
  projection: StudioSavedTrainRuns,
) {
  const { step, where } = trainSourceSubset(source, view, projection, !!editor.getSnapshot().saved.project.sweep);
  const name = `${step.id}-continuation`;
  const stepIds = new Set(Object.keys(editor.getSnapshot().saved.project.steps ?? {}));
  let id = name;
  for (let number = 2; stepIds.has(id); number += 1) id = `${name}-${number}`;
  await editor.createFile({
    kind: 'step', id,
    step: withCheckpointStart({ kind: 'train', base: { step: step.id } }, step.inputs.some(({ id }) => id === 'checkpoint'),
      { step: step.id, checkpoint: 'last', ...(where && { where }) }),
  }, projection.revision);
  return id;
}
