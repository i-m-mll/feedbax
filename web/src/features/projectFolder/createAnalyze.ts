import type { CreateNamedDocument, StudioSavedTrainKey } from '@/generated/studioContracts';
import { containsCreatedFile, ProjectEditConflictError, type ProjectEditController } from './projectEditController';

export async function createAnalyze(editor: ProjectEditController, revision: string, source: StudioSavedTrainKey) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== revision || saved.project.sweep || source.variant !== 'default' ||
      saved.project.steps?.[source.step]?.kind !== 'evaluate') throw new ProjectEditConflictError();
  const document = {
    schema: 'feedbax.analysis_document.v1',
    payload: { schema_version: 'feedbax.analysis_document.v1', operations: {}, views: {} },
  };
  let id = 'analyze';
  for (let number = 2; saved.project.steps?.[id] || saved.project.documents?.[id] &&
      (!containsCreatedFile(saved.project, { kind: 'document', id, document }) ||
       Object.values(saved.project.steps ?? {}).some(({ base }) => 'document' in base && base.document === id));
    number += 1) id = 'analyze-' + number;
  const file: CreateNamedDocument = { kind: 'document', id, document };
  const acknowledged = containsCreatedFile(saved.project, file) ? saved
    : (await editor.createFile(file, revision)).snapshot;
  await editor.createFile({
    kind: 'step', id, step: {
      kind: 'analyze', base: { document: id }, inputs: { states: { step: source.step, row: source.row } },
    },
  }, acknowledged.revision);
  return id;
}
