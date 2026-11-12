import { studioName, type StudioRunStatusId } from '@/components/activity/studioVocabulary';
import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import { selectedRuns } from '@/features/pipeline/selection';
import type { StudioSavedTrainRow, StudioSavedTrainRuns, StudioSavedTrainKey, SubsetTerms } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';

function runId(key: StudioSavedTrainKey) {
  return JSON.stringify([key.variant, key.step, key.row]);
}

export function rowId({ key }: StudioSavedTrainRow) {
  return runId(key);
}

export function savedTrainPlan(stepIds: readonly string[], data?: StudioSavedTrainRuns): PlanView {
  const rows = data?.rows ?? [];
  const byStep = new Map<string, StudioSavedTrainRow[]>();
  for (const row of rows) {
    const step = byStep.get(row.key.step);
    if (step) step.push(row);
    else byStep.set(row.key.step, [row]);
  }
  return {
    edges: rows.flatMap(({ key, checkpoint_producer: source }) => source ? [{
      from: { stepId: source.step, runId: runId(source) },
      to: { stepId: key.step, runId: runId(key) },
    }] : []),
    steps: stepIds.map((id): PlanStepView => {
      const sweeps = new Map<string, Map<string, string>>();
      const statusCounts: Partial<Record<StudioRunStatusId, number>> = {};
      const runs = (byStep.get(id) ?? []).map((row): PlanRunView => {
        const values = Object.entries(row.coordinates).map(([axis, { value_id, label }]) => {
          const choices = sweeps.get(axis) ?? new Map<string, string>();
          choices.set(value_id, label);
          sweeps.set(axis, choices);
          return [axis, value_id];
        });
        statusCounts[row.status] = (statusCounts[row.status] ?? 0) + 1;
        return {
          id: rowId(row), label: row.label, variant: row.key.variant, values: Object.fromEntries(values),
          status: row.status, reason: row.reason ?? undefined,
          progress: row.attempts.at(-1)?.progress ?? undefined,
        };
      });
      const subsets = data?.checkpoint_subsets.filter((subset) => subset.step === id) ?? [];
      const subset = subsets[0];
      const sourceRows = subset ? byStep.get(subset.source_step) ?? [] : [];
      return {
        id, label: id, kind: 'train', runs, statusCounts,
        sweeps: Array.from(sweeps, ([axis, choices]) => ({
          id: axis, label: axis, values: Array.from(choices, ([key, label]) => ({ id: key, label })),
        })),
        inputs: subset ? [{
          id: 'checkpoint', label: studioName('checkpoint'), sourceStepId: subset.source_step,
          subset: {
            terms: [
              ...Object.entries(subset.terms.axes ?? {}).map(([axis, ids]) => {
                const labels = ids.map((key) =>
                  sourceRows.find((row) => row.coordinates[axis]?.value_id === key)?.coordinates[axis].label ?? key);
                return `${axis} ${labels.join(', ')}`;
              }),
              ...(subset.terms.rows ?? []),
            ],
            count: subsets.reduce((count, entry) => count + entry.count, 0),
            total: subsets.reduce((count, entry) => count + entry.total, 0),
          },
        }] : [],
      };
    }),
  };
}

export async function createCheckpointTrain(
  editor: ProjectEditController,
  source: PipelineMapStepSource,
  view: PlanView,
  projection: StudioSavedTrainRuns,
) {
  const { stepId, selection } = source;
  const step = view.steps.find(({ id }) => id === stepId);
  if (!step || !step.runs.length) throw new Error('Choose a runnable Train step.');
  let where: SubsetTerms | undefined;
  if (selection) {
    const selected = new Set(selectedRuns(step, selection).map(({ id }) => id));
    if (!selected.size) throw new ProjectEditConflictError();
    if (selection.kind === 'values') {
      where = { axes: Object.fromEntries(selection.terms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]])) };
    } else {
      const sourceRows = projection.rows.filter(({ key }) => key.step === stepId);
      const rows = new Set(sourceRows.filter((row) => selected.has(rowId(row))).map(({ key }) => key.row));
      if (sourceRows.some((row) => rows.has(row.key.row) && !selected.has(rowId(row)))) {
        throw new Error('Select those row IDs in every variant to create one shared continuation.');
      }
      where = { rows: [...rows] };
    }
  }
  const name = `${stepId}-continuation`;
  const stepIds = new Set(Object.keys(editor.getSnapshot().saved.project.steps ?? {}));
  const inheritsInitialization = projection.rows.some(
    (row) => row.key.step === stepId && row.checkpoint_producer !== null,
  );
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
