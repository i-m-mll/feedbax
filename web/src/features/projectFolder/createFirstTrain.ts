import type { CreateNamedDocument, Document, StudioProjectSnapshot } from '@/generated/studioContracts';
import { containsCreatedFile, ProjectEditConflictError, type ProjectEditController } from './projectEditController';

const model: Document = {
  schema: 'feedbax.spec.model',
  payload: { step_seconds: 0.01, ports: {}, interior: {} },
};
const task: Document = {
  schema: 'feedbax.spec.python_task',
  payload: { type: 'feedbax.task.reaching.v1', params: {} },
};
const objective: Document = {
  schema: 'feedbax.spec.objective_document',
  payload: { id: 'objective', terms: [] },
};
const methodSchema = 'feedbax.spec.training_method.studio_graph';

export function canCreateFirstTrain(project: StudioProjectSnapshot['project']) {
  return !project.sweep && Object.keys(project.steps ?? {}).length === 0;
}

async function ensureDocument(
  editor: ProjectEditController, saved: StudioProjectSnapshot, name: string, document: Document,
) {
  let id = name;
  for (let number = 2; Object.prototype.hasOwnProperty.call(saved.project.documents, id); number += 1) {
    if (containsCreatedFile(saved.project, { kind: 'document', id, document })) return { id, saved };
    id = `${name}-${number}`;
  }
  const file: CreateNamedDocument = { kind: 'document', id, document };
  const result = await editor.createFile(file, saved.revision);
  return { id, saved: result.snapshot };
}

export async function createFirstTrain(editor: ProjectEditController, reviewedRevision: string) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== reviewedRevision || !canCreateFirstTrain(saved.project)) {
    throw new ProjectEditConflictError();
  }

  // Publish the step after the documents are durable so creation can be retried.
  const graph = await ensureDocument(editor, saved, 'model', model);
  const taskDocument = await ensureDocument(editor, graph.saved, 'task', task);
  const objectiveDocument = await ensureDocument(editor, taskDocument.saved, 'objective', objective);
  const id = 'train';
  const nBatches = 1000;
  const learningRate = 0.001;
  await editor.createFile({
    kind: 'step', id,
    step: {
      kind: 'train',
      base: { inline: {
        trial_documents: {
          model_name: graph.id,
          model: { document: graph.id },
          python_task: { document: taskDocument.id },
          objective: { document: objectiveDocument.id },
        },
        training_config: {
          n_batches: nBatches,
          batch_size: 64,
        },
        method_ref: { package: 'feedbax', name: 'studio_graph', version: 'v1' },
        method_payload: {
          schema_id: methodSchema,
          schema_version: `${methodSchema}.v1`,
          payload: {
            optimizer: { type: 'adam', params: { learning_rate: learningRate } },
            target_total: nBatches,
            key_schedule: 'independent_v1',
          },
        },
      } },
    },
  }, objectiveDocument.saved.revision);
  return id;
}
