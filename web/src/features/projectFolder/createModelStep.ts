import { studioName } from '@/components/activity/studioVocabulary';
import {
  firstTrainTask, type Document, type JsonValue, type StudioAuthoringCatalog, type StudioProjectCreateFile,
  type StudioProjectSnapshot,
} from '@/generated/studioContracts';
import { catalogEntries } from '@/api/studioAuthoring';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { blankDocuments, namedDocument, type UnnamedDocument } from './useProjectDocuments';
import { trainingDocumentKeys } from './trainingDocument';

const { model, task, objective, training } = blankDocuments;
const pythonTaskSchema = 'feedbax.spec.python_task';

/** Feedbax's simple reaches without its links, which name the fields of the shipped Model that a new step does not read. */
const seededTask: UnnamedDocument = (() => {
  if (!('payload' in firstTrainTask)) throw new Error('The shipped simple reaches is not a root document');
  return { ...firstTrainTask, payload: { ...firstTrainTask.payload as Record<string, JsonValue>, links: [] } };
})();

/** Binds a new step to the project's first document of each schema, adding to `files` the documents that are missing. */
function documentBinder(editor: ProjectEditController, saved: StudioProjectSnapshot, files: StudioProjectCreateFile['file'][]) {
  const documents = saved.project.documents ?? {};
  const documentNames = Object.keys(documents).sort();
  return async (document: UnnamedDocument, schemas = [document.schema]) => {
    const existing = documentNames.find((id) => schemas.includes(documents[id].schema));
    if (existing !== undefined) return existing;
    const named = await namedDocument(editor, document, saved.revision);
    files.push({ kind: 'document', ...named });
    return named.id;
  };
}

/** The Model, Task and Objective a new model-producing step reads, with the field that holds its Task. */
async function trialDocuments(saved: StudioProjectSnapshot, bind: ReturnType<typeof documentBinder>) {
  const graph = await bind(model);
  const taskDocument = await bind(seededTask, [task.schema, pythonTaskSchema]);
  const taskField = saved.project.documents?.[taskDocument]?.schema === pythonTaskSchema ? 'python_task' : 'task';
  return { model: graph, task: taskDocument, taskField, objective: await bind(objective) } as const;
}

function reviewed(editor: ProjectEditController, reviewedRevision: string) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== reviewedRevision) throw new ProjectEditConflictError();
  return saved;
}

/** Create a Train step on the project's Model, Task, Objective and training documents, creating those that are missing;
 * a missing Task is feedbax's simple reaches, which starts nothing until a variable is chosen to initialize the Model. */
export async function createTrain(editor: ProjectEditController, reviewedRevision: string) {
  const saved = reviewed(editor, reviewedRevision);
  const files: StudioProjectCreateFile['file'][] = [];
  const bind = documentBinder(editor, saved, files);
  const trial = await trialDocuments(saved, bind);
  const trainingDocument = await bind(training);
  const { id, label } = await editor.nameMember({ kind: 'step', step: 'train' }, reviewedRevision);
  files.push({
    kind: 'step', id,
    step: {
      kind: 'train', label,
      base: { inline: {
        trial_documents: {
          model: { document: trial.model },
          [trial.taskField]: { document: trial.task },
          objective: { document: trial.objective },
        },
        ...Object.fromEntries(trainingDocumentKeys.map((key) => [key, { document: trainingDocument, pointer: `/${key}` }])),
      } },
    },
  });
  await editor.createFiles(studioName('newStep'), files, reviewedRevision);
  return id;
}

/**
 * Create a Synthesize step on the project's Model, Task and Objective, creating those that are missing, by the first
 * synthesis method the catalog registers.
 */
export async function createSynthesize(editor: ProjectEditController, reviewedRevision: string, catalog: StudioAuthoringCatalog) {
  const method = catalogEntries(catalog, 'synthesis_methods')[0]?.key;
  if (!method) throw new Error('No synthesis method is registered.');
  const saved = reviewed(editor, reviewedRevision);
  const files: StudioProjectCreateFile['file'][] = [];
  const trial = await trialDocuments(saved, documentBinder(editor, saved, files));
  const { id, label } = await editor.nameMember({ kind: 'step', step: 'synthesize' }, reviewedRevision);
  files.push({
    kind: 'step', id,
    step: {
      kind: 'synthesize', label,
      base: { inline: {
        model: { document: trial.model },
        task: { document: trial.task },
        objective: { document: trial.objective },
        method,
      } },
    },
  });
  await editor.createFiles(studioName('newStep'), files, reviewedRevision);
  return id;
}
