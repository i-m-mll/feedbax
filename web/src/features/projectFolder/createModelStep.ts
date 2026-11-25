import { studioName } from '@/components/activity/studioVocabulary';
import {
  firstTrainTask, type Document, type JsonValue, type StudioProjectCreateFile, type StudioProjectSnapshot,
} from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { blankDocuments, namedDocument } from './useProjectDocuments';
import { trainingDocumentKeys } from './trainingDocument';
import { generatedName, holds } from './generatedName';

const { model, task, objective, training } = blankDocuments;
const pythonTaskSchema = 'feedbax.spec.python_task';

/** Feedbax's simple reaches without its links, which name the fields of the shipped Model that a new step does not read. */
const seededTask: Document = (() => {
  if (!('payload' in firstTrainTask)) throw new Error('The shipped simple reaches is not a root document');
  return { ...firstTrainTask, payload: { ...firstTrainTask.payload as Record<string, JsonValue>, links: [] } };
})();

/** Binds a new step to the project's first document of each schema, adding to `files` the documents that are missing. */
function documentBinder(saved: StudioProjectSnapshot, files: StudioProjectCreateFile['file'][]) {
  const documents = saved.project.documents ?? {};
  const documentNames = Object.keys(documents).sort();
  return (document: Document, schemas = [document.schema]) => {
    const existing = documentNames.find((id) => schemas.includes(documents[id].schema));
    if (existing !== undefined) return existing;
    const named = namedDocument(documents, document);
    files.push({ kind: 'document', ...named });
    return named.id;
  };
}

/** The Model, Task and Objective a new model-producing step reads, with the field that holds its Task. */
function trialDocuments(saved: StudioProjectSnapshot, bind: ReturnType<typeof documentBinder>) {
  const graph = bind(model);
  const taskDocument = bind(seededTask, [task.schema, pythonTaskSchema]);
  const taskField = saved.project.documents?.[taskDocument]?.schema === pythonTaskSchema ? 'python_task' : 'task';
  return { model: graph, task: taskDocument, taskField, objective: bind(objective) } as const;
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
  const bind = documentBinder(saved, files);
  const trial = trialDocuments(saved, bind);
  const trainingDocument = bind(training);
  const id = generatedName('train', holds(saved.project.steps));
  files.push({
    kind: 'step', id,
    step: {
      kind: 'train',
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

/** Create a Synthesize step by `method` on the project's Model, Task and Objective, creating those that are missing. */
export async function createSynthesize(editor: ProjectEditController, reviewedRevision: string, method: string) {
  const saved = reviewed(editor, reviewedRevision);
  const files: StudioProjectCreateFile['file'][] = [];
  const trial = trialDocuments(saved, documentBinder(saved, files));
  const id = generatedName('synthesize', holds(saved.project.steps));
  files.push({
    kind: 'step', id,
    step: {
      kind: 'synthesize',
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
