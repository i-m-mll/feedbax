import type { PlanRunView, PlanStepView } from '@/features/pipeline/planView';
import type {
  CreateNamedDocument, CreateNamedStep, ReportDocument, StepReference, StudioProjectWindowMapElement, StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { containsCreatedFile, ProjectEditConflictError, type ProjectEditController } from './projectEditController';

import { blankDocuments, namedDocument, type UnnamedDocument } from './useProjectDocuments';
import { planVariants, selectedRowTerms, stepSourceRows } from './projectWindowPlan';
import { withSubsetTerms } from './selectionEdits';

type NamedStep = CreateNamedStep['step'];

async function createManagedStep(
  editor: ProjectEditController,
  revision: string,
  sourceStepId: string,
  document: UnnamedDocument,
  step: Pick<NamedStep, 'kind' | 'inputs'>,
) {
  const { saved } = editor.getSnapshot();
  const sourceKind = step.kind === 'analyze' ? 'evaluate' : 'analyze';
  if (saved.revision !== revision || saved.project.steps?.[sourceStepId]?.kind !== sourceKind) throw new ProjectEditConflictError();
  const steps = saved.project.steps ?? {};
  const documents = saved.project.documents ?? {};
  const read = (name: string) => Object.values(steps).some(({ base }) => 'document' in base && base.document === name);
  /** A document an earlier attempt created and no step reads, which this attempt takes instead of making another. */
  const left = Object.keys(documents).find((name) => !read(name) && containsCreatedFile(saved.project,
    { kind: 'document', id: name, document: { ...document, label: documents[name].label } as CreateNamedDocument['document'] }));
  const file: CreateNamedDocument = left === undefined
    ? { kind: 'document', ...await namedDocument(editor, document, revision) as Pick<CreateNamedDocument, 'id' | 'document'> }
    : { kind: 'document', id: left, document: documents[left] as CreateNamedDocument['document'] };
  const documentId = file.id;
  const acknowledged = left !== undefined ? saved : (await editor.createFile(file, revision)).snapshot;
  const { id, label } = await editor.nameMember({ kind: 'step', step: step.kind }, acknowledged.revision);
  await editor.createFile({
    kind: 'step', id, step: { ...step, label, base: { document: documentId } },
  }, acknowledged.revision);
  return id;
}

export function analyzeInput(
  step: PlanStepView, runs: readonly PlanRunView[], projection: StudioSavedTrainRuns,
): StepReference | undefined {
  const variant = planVariants(projection)[0];
  if (!variant || !runs.length || runs.some((run) => run.variant !== variant)) return undefined;
  const where = selectedRowTerms(stepSourceRows(projection, step.id, variant), new Set(runs.map(({ id }) => id)));
  return where && withSubsetTerms({ step: step.id }, where);
}

export function createAnalyze(editor: ProjectEditController, revision: string, source: StepReference) {
  return createManagedStep(editor, revision, source.step, blankDocuments.analysis, {
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
