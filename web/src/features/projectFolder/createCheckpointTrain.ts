import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import type { PlanView } from '@/features/pipeline/planView';
import { selectedRuns } from '@/features/pipeline/selection';
import type { Step, StepReference, StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { planVariants, selectedRowTerms, stepSourceRows } from './projectWindowPlan';
import { authoredSubset } from './selectionEdits';

const initializationPath = 'checkpoint_initialization';

/**
 * A new Train step, labelled `label`, continuing `source`'s step from its last checkpoint, initializing its model from it. `inherited` says
 * whether that step is itself a continuation, whose initialization policy the new step replaces.
 */
function continuation(source: StepReference, inherited: boolean, label: string): Step {
  return {
    kind: 'train', label, base: { step: source.step }, inputs: { checkpoint: source },
    deltas: [{
      layer_id: 'initialize',
      patches: [{ path: initializationPath, op: inherited ? 'replace' : 'add', value: 'initialize_from' }],
      ...(inherited && { acknowledges_ancestor_paths: [initializationPath] }),
    }],
  };
}

/** The runs of a model-producing step of one of `kinds` that `source` chooses, with the terms that choose them. */
export function modelSourceSubset(
  source: PipelineMapStepSource,
  view: PlanView,
  projection: StudioSavedTrainRuns,
  kinds: readonly ('train' | 'synthesize')[],
) {
  const { stepId, selection } = source;
  const found = view.steps.find(({ id }) => id === stepId);
  const step = found && (kinds as readonly string[]).includes(found.kind) ? found : undefined;
  if (!step?.runs.length) throw new Error('Choose a runnable model step.');
  const members = selection ? selectedRuns(step, selection) : step.runs;
  const variant = planVariants(projection)[0];
  if (!variant || !members.length || members.some((member) => member.variant !== variant)) throw new ProjectEditConflictError();
  let where: SubsetTerms | undefined;
  if (selection) {
    const terms = selectedRowTerms(stepSourceRows(projection, stepId, variant), new Set(members.map(({ id }) => id)));
    if (!terms) throw new ProjectEditConflictError();
    where = authoredSubset(terms);
  }
  return { step, members, where };
}

export async function createCheckpointTrain(
  editor: ProjectEditController,
  source: PipelineMapStepSource,
  view: PlanView,
  projection: StudioSavedTrainRuns,
) {
  const { step, where } = modelSourceSubset(source, view, projection, ['train']);
  const { id, label } = await editor.nameMember({ kind: 'step', step: 'train' }, projection.revision);
  await editor.createFile({
    kind: 'step', id,
    step: continuation({ step: step.id, checkpoint: 'last', ...(where && { where }) },
      step.inputs.some(({ id }) => id === 'checkpoint'), label),
  }, projection.revision);
  return id;
}
