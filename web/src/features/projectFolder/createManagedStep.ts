import type {
  CreateNamedDocument, CreateNamedStep, ReportDocument, StudioProjectWindowMapElement, StudioSavedTrainKey,
} from '@/generated/studioContracts';
import { containsCreatedFile, ProjectEditConflictError, type ProjectEditController } from './projectEditController';

type NamedStep = CreateNamedStep['step'];

async function createManagedStep(
  editor: ProjectEditController,
  revision: string,
  source: StudioSavedTrainKey,
  document: CreateNamedDocument['document'],
  step: Pick<NamedStep, 'kind' | 'inputs'>,
) {
  const { saved } = editor.getSnapshot();
  const sourceKind = step.kind === 'analyze' ? 'evaluate' : 'analyze';
  if (saved.revision !== revision || saved.project.sweep || source.variant !== 'default' ||
      saved.project.steps?.[source.step]?.kind !== sourceKind) throw new ProjectEditConflictError();
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

export function createAnalyze(editor: ProjectEditController, revision: string, source: StudioSavedTrainKey) {
  return createManagedStep(editor, revision, source, {
    schema: 'feedbax.analysis_document.v1',
    payload: { schema_version: 'feedbax.analysis_document.v1', operations: {}, views: {} },
  }, {
    kind: 'analyze', inputs: { states: { step: source.step, row: source.row } },
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
  return createManagedStep(editor, revision, source.key, {
    schema,
    payload,
  }, {
    kind: 'report', inputs: { [role]: source.figure },
  });
}
