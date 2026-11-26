import { expect, type APIRequestContext } from '@playwright/test';
import type {
  ModelFieldResponse, ModelFieldView, NamedFileCreation, StudioProjectSnapshot, StudioSavedTrainKey, StudioTrainAuthoring,
} from '../../../src/generated/studioContracts';
import { seedManagedProject } from './canonicalTrain';
import { readSaved } from './studioKeyboard';

// The Service's own reads of a project, for specs whose claim is a value rather than the form a document stores it in.

/**
 * A read posted at the project's current revision. An edit that lands between reading the revision and the read makes
 * the Service refuse the stale revision; the read is then posted at the revision it names.
 */
async function readAtCurrent(request: APIRequestContext, projectPath: string, path: string, body: object) {
  let { revision } = await readSaved(request, projectPath);
  for (;;) {
    const response = await request.post(projectPath + path, { data: { ...body, revision } });
    if (response.status() === 409) {
      const { detail } = await response.json() as { detail: { current?: string } };
      if (detail.current && detail.current !== revision) {
        revision = detail.current;
        continue;
      }
    }
    expect(response.ok(), await response.text()).toBe(true);
    return response;
  }
}

/** The settings a Train step's runs train with, the method's defaults filled in, as the Runs view draws them. */
export async function trainSettings(request: APIRequestContext, projectPath: string, step = 'train', run?: StudioSavedTrainKey) {
  const response = await readAtCurrent(request, projectPath, `/authoring/train/${step}`, run ? { run } : {});
  return (await response.json() as StudioTrainAuthoring).settings;
}

/** The fields the Inspector shows for the top-level component `component` of Model document `document`. */
export async function modelFields(
  request: APIRequestContext, projectPath: string, document: string, component: string, run?: StudioSavedTrainKey,
): Promise<ModelFieldView[]> {
  const response = await readAtCurrent(request, projectPath, '/model/fields', {
    document, owner: { composites: [], element: { kind: 'component', id: component } }, ...(run && { run }),
  });
  return (await response.json() as ModelFieldResponse).fields ?? [];
}

/** The Inspector's field labelled `label`; the shared helpers above give its value, address text and sweep. */
export async function modelField(
  request: APIRequestContext, projectPath: string, document: string, component: string, label: string, run?: StudioSavedTrainKey,
) {
  const field = (await modelFields(request, projectPath, document, component, run)).find((item) => item.label === label);
  if (!field) throw new Error(`The Model ${document} shows no field ${label} on ${component}`);
  return field;
}

/** The project the Service stores for `files`, created in a project of their own: what any project holding them stores. */
export async function storedProject(request: APIRequestContext, files: readonly NamedFileCreation[]): Promise<StudioProjectSnapshot['project']> {
  return (await seedManagedProject(request, files)).snapshot.project;
}
