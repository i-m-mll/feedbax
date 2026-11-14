import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import type { PlanView } from '@/features/pipeline/planView';
import { selectedRuns } from '@/features/pipeline/selection';
import type { StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId, selectedProducerRow } from './projectWindowPlan';

export function trainSourceSubset(
  source: PipelineMapStepSource,
  view: PlanView,
  projection: StudioSavedTrainRuns,
) {
  const { stepId, selection } = source;
  const step = view.steps.find(({ id }) => id === stepId);
  if (step?.kind !== 'train' || !step.runs.length) throw new Error('Choose a runnable Train step.');
  const members = selection ? selectedRuns(step, selection) : step.runs;
  let where: SubsetTerms | undefined;
  if (selection) {
    const selected = new Set(members.map(({ id }) => id));
    if (!selected.size) throw new ProjectEditConflictError();
    if (selection.kind === 'values') {
      where = { axes: Object.fromEntries(selection.terms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]])) };
    } else {
      const rows = (projection.window_map?.rows ?? []).flatMap((row) => {
        if (row.key.step !== stepId || !selected.has(savedRunId(row.key))) return [];
        const member = selectedProducerRow(row);
        return member ? [member] : [];
      });
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
  const { step, where } = trainSourceSubset(source, view, projection);
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
