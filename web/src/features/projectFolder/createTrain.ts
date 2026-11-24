import { studioName } from '@/components/activity/studioVocabulary';
import type { Document, StudioProjectCreateFile } from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { blankDocuments } from './useProjectDocuments';
import { trainingDocumentKeys } from './trainingDocument';
import { generatedName, holds } from './generatedName';

const { model, task, objective, training } = blankDocuments;
const pythonTaskSchema = 'feedbax.spec.python_task';

/** Create a Train step on the project's Model, Task, Objective and training documents, creating those that are missing. */
export async function createTrain(editor: ProjectEditController, reviewedRevision: string) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== reviewedRevision) throw new ProjectEditConflictError();

  const files: StudioProjectCreateFile['file'][] = [];
  const documents = saved.project.documents ?? {};
  const documentNames = Object.keys(documents).sort();
  function bindDocument(name: string, document: Document, schemas = [document.schema]) {
    const existing = documentNames.find((id) => schemas.includes(documents[id].schema));
    if (existing !== undefined) return existing;

    const id = generatedName(name, holds(documents));
    files.push({ kind: 'document', id, document });
    return id;
  }
  const graph = bindDocument('model', model);
  const taskDocument = bindDocument('task', task, [task.schema, pythonTaskSchema]);
  const taskField = documents[taskDocument]?.schema === pythonTaskSchema ? 'python_task' : 'task';
  const objectiveDocument = bindDocument('objective', objective);
  const trainingDocument = bindDocument('training', training);
  const id = generatedName('train', holds(saved.project.steps));
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
        ...Object.fromEntries(trainingDocumentKeys.map((key) => [key, { document: trainingDocument, pointer: `/${key}` }])),
      } },
    },
  });
  await editor.createFiles(studioName('newStep'), files, reviewedRevision);
  return id;
}
