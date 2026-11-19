import { studioName } from '@/components/activity/studioVocabulary';
import type { Document, StudioProjectCreateFile } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { blankDocuments } from './useProjectDocuments';

const { model, task, objective } = blankDocuments;
const taskDocumentSchema = 'feedbax.spec.task_document';
export const supervisedTrainMethod = 'feedbax.supervised_gradient';

/** Create a Train step on the project's Model, Task and Objective documents, creating those that are missing. */
export async function createTrain(editor: ProjectEditController, reviewedRevision: string) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== reviewedRevision) throw new ProjectEditConflictError();

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
  const steps = saved.project.steps ?? {};
  let id = 'train';
  for (let number = 2; Object.prototype.hasOwnProperty.call(steps, id); number += 1) id = `train-${number}`;
  files.push({
    kind: 'step', id,
    step: {
      kind: 'train',
      base: { inline: {
        trial_documents: {
          model: { document: graph },
          [taskField]: { document: taskDocument },
          objective: { document: objectiveDocument },
        },
        training_config: {
          n_batches: 1000,
          batch_size: 64,
        },
        method: supervisedTrainMethod,
        settings: {
          optimizer: { type: 'adam', learning_rate: { type: 'constant_schedule', value: 0.001 } },
        },
      } },
    },
  });
  await editor.createFiles(studioName('newStep'), files, reviewedRevision);
  return id;
}
