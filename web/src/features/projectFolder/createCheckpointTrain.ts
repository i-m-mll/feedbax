import { valuesRule } from '@/components/values/runRule';
import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import type { PlanView } from '@/features/pipeline/planView';
import { selectedRuns } from '@/features/pipeline/selection';
import type { Step, StepReference, StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { planVariants, savedRunId, sharedRowTerms, stepSourceRows } from './projectWindowPlan';
import { generatedName, holds } from './generatedName';

const initializationPath = 'checkpoint_initialization';

/**
 * A new Train step continuing `source`'s step from its last checkpoint, initializing its model from it. `inherited` says
 * whether that step is itself a continuation, whose initialization policy the new step replaces.
 */
function continuation(source: StepReference, inherited: boolean): Step {
  return {
    kind: 'train', base: { step: source.step }, inputs: { checkpoint: source },
    deltas: [{
      layer_id: 'initialize',
      patches: [{ path: initializationPath, op: inherited ? 'replace' : 'add', value: 'initialize_from' }],
      ...(inherited && { acknowledges_ancestor_paths: [initializationPath] }),
    }],
  };
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
      where = { rule: valuesRule(step.ruleAxes ?? {},
        Object.fromEntries(selection.terms.map(({ sweepId, valueIds }) => [sweepId, valueIds]))) };
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
  const id = generatedName('train', holds(editor.getSnapshot().saved.project.steps));
  await editor.createFile({
    kind: 'step', id,
    step: continuation({ step: step.id, checkpoint: 'last', ...(where && { where }) },
      step.inputs.some(({ id }) => id === 'checkpoint')),
  }, projection.revision);
  return id;
}
