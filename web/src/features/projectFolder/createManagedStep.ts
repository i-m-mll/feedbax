import type { PlanRunView, PlanStepView } from '@/features/pipeline/planView';
import type {
  CreateNamedDocument, CreateNamedStep, ReportDocument, StepReference, StudioProjectWindowMapElement, StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { containsCreatedFile, ProjectEditConflictError, type ProjectEditController } from './projectEditController';

import { planVariants, savedRunId, sharedRowTerms, stepSourceRows } from './projectWindowPlan';

type NamedStep = CreateNamedStep['step'];

async function createManagedStep(
  editor: ProjectEditController,
  revision: string,
  sourceStepId: string,
  document: CreateNamedDocument['document'],
  step: Pick<NamedStep, 'kind' | 'inputs'>,
) {
  const { saved } = editor.getSnapshot();
  const sourceKind = step.kind === 'analyze' ? 'evaluate' : 'analyze';
  if (saved.revision !== revision || saved.project.steps?.[sourceStepId]?.kind !== sourceKind) throw new ProjectEditConflictError();
  let id: string = step.kind;
  for (let number = 2; saved.project.steps?.[id] || saved.project.documents?.[id] &&
      (!containsCreatedFile(saved.project, { kind: 'document', id, document }) ||
       Object.values(saved.project.steps ?? {}).some(({ base }) => 'document' in base && base.document === id));
    number += 1) id = step.kind + '-' + number;
  const file: CreateNamedDocument = { kind: 'document', id, document };
  const acknowledged = containsCreatedFile(saved.project, file) ? saved
    : (await editor.createFile(file, revision)).snapshot;
  await editor.createFile({
    kind: 'step', id, step: { ...step, base: { document: id } },
  }, acknowledged.revision);
  return id;
}

export function analyzeInput(
  step: PlanStepView, runs: readonly PlanRunView[], projection: StudioSavedTrainRuns, sharedProject: boolean,
): StepReference | undefined {
  const variant = planVariants(projection)[0];
  if (!variant || !runs.length || runs.some((run) => run.variant !== variant)) return undefined;
  const completed = new Set(projection.window_map?.evaluations?.filter(({ result }) => result).map(({ key }) => savedRunId(key)));
  if (!runs.every(({ id }) => completed.has(id))) return undefined;
  const selected = new Set(runs.map(({ id }) => id));
  const rows = stepSourceRows(projection, step.id, variant)
    .filter((row) => selected.has(savedRunId({ ...row, step: step.id })));
  if (rows.length !== selected.size) return undefined;
  if (rows.length === 1) return { step: step.id, row: rows[0].row };
  const where = sharedProject ? sharedRowTerms(step, runs) : { rows };
  return where ? { step: step.id, where } : undefined;
}

export function createAnalyze(editor: ProjectEditController, revision: string, source: StepReference) {
  return createManagedStep(editor, revision, source.step, {
    schema: 'feedbax.analysis_document.v1',
    payload: {
      schema_version: 'feedbax.analysis_document.v1',
      inputs: { states: { kind: 'dataset' } },
      operations: {},
      views: {},
    },
  }, {
    kind: 'analyze', inputs: { states: source },
  });
}

export type ReportFigureSource = Omit<StudioProjectWindowMapElement, 'figure'> & {
  figure: NonNullable<StudioProjectWindowMapElement['figure']>;
};

export function isReportFigureSource(element: StudioProjectWindowMapElement): element is ReportFigureSource {
  return !!element.figure;
}

export function reportFigureRole(source: ReportFigureSource) {
  return 'figure-' + source.name;
}

export function createReport(editor: ProjectEditController, revision: string, source: ReportFigureSource, title: string) {
  const schema = 'feedbax.report_document.v1';
  const role = reportFigureRole(source);
  const payload: ReportDocument = {
    schema_version: schema,
    title,
    blocks: [{ figure: role }],
    completeness: 'complete',
  };
  return createManagedStep(editor, revision, source.key.step, {
    schema,
    payload,
  }, {
    kind: 'report', inputs: { [role]: source.figure },
  });
}
