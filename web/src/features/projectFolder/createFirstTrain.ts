import { studioName } from '@/components/activity/studioVocabulary';
import type { Document, StudioProjectCreateFile, StudioProjectSnapshot } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';

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
const taskDocumentSchema = 'feedbax.spec.task_document';
const methodSchema = 'feedbax.spec.training_method.studio_graph';

export function canCreateFirstTrain(project: StudioProjectSnapshot['project']) {
  return Object.keys(project.steps ?? {}).length === 0;
}

export async function createFirstTrain(editor: ProjectEditController, reviewedRevision: string) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== reviewedRevision || !canCreateFirstTrain(saved.project)) {
    throw new ProjectEditConflictError();
  }

  const files: StudioProjectCreateFile['file'][] = [];
  const documents = saved.project.documents ?? {};
  const documentNames = Object.keys(documents).sort();
  function bindDocument(name: string, document: Document, schemas = [document.schema]) {
    const existing = documentNames.find((id) => schemas.includes(documents[id].schema));
    if (existing !== undefined) return existing;

    let id = name;
    for (let number = 2; Object.prototype.hasOwnProperty.call(documents, id); number += 1) {
      id = `${name}-${number}`;
    }
    files.push({ kind: 'document', id, document });
    return id;
  }
  const graph = bindDocument('model', model);
  const taskDocument = bindDocument('task', task, [task.schema, taskDocumentSchema]);
  const taskField = documents[taskDocument]?.schema === taskDocumentSchema ? 'task' : 'python_task';
  const objectiveDocument = bindDocument('objective', objective);
  const id = 'train';
  const nBatches = 1000;
  const learningRate = 0.001;
  files.push({
    kind: 'step', id,
    step: {
      kind: 'train',
      base: { inline: {
        trial_documents: {
          model_name: graph,
          model: { document: graph },
          [taskField]: { document: taskDocument },
          objective: { document: objectiveDocument },
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
  });
  await editor.createFiles(studioName('newStep'), files, reviewedRevision);
  return id;
}
