import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import type { PlanView } from '@/features/pipeline/planView';
import { selectedRuns } from '@/features/pipeline/selection';
import type { StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { planVariants, savedRunId, sharedRowTerms, stepSourceRows } from './projectWindowPlan';

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
  const stepId = step.id;
  const name = `${stepId}-continuation`;
  const stepIds = new Set(Object.keys(editor.getSnapshot().saved.project.steps ?? {}));
  const inheritsInitialization = step.inputs.some(({ id }) => id === 'checkpoint');
  let id = name;
  for (let number = 2; stepIds.has(id); number += 1) id = `${name}-${number}`;
  await editor.createFile({
    kind: 'step', id,
    step: {
      kind: 'train',
      base: { step: stepId },
      inputs: { checkpoint: { step: stepId, checkpoint: 'last', ...(where && { where }) } },
      deltas: inheritsInitialization ? [] : [{
        layer_id: 'initialize',
        patches: [{
          path: 'method_payload.payload.checkpoint_initialization', op: 'add', value: 'initialize_from',
        }],
      }],
    },
  }, projection.revision);
  return id;
}
