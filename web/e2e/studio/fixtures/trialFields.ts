import { expect, type APIRequestContext } from '@playwright/test';
import type {
  StudioProjectSnapshot, StudioSavedTrainKey, StudioTrialAuthoring, StudioTrialField,
} from '../../../src/generated/studioContracts';

const defaultRun: StudioSavedTrainKey = { variant: 'default', step: 'train', row: 'default' };

/** The Service's Trial view of a run at the project's current revision: the read Studio's Task and Objective views use. */
export async function trialView(request: APIRequestContext, projectId: string, run = defaultRun): Promise<StudioTrialAuthoring> {
  const path = `/api/projects/${encodeURIComponent(projectId)}`;
  const { revision } = await (await request.get(path)).json() as StudioProjectSnapshot;
  const response = await request.post(`${path}/authoring/trials`, { data: { revision, run } });
  // A save between the two reads makes the revision stale; the view is read again at the next.
  if (response.status() === 409) return trialView(request, projectId, run);
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioTrialAuthoring;
}

/** The field the Service shows at `path` of the view (`['task', 'variables', key, 'value']`), as Studio shows it. */
export async function trialField(request: APIRequestContext, projectId: string, path: readonly string[], run = defaultRun):
  Promise<StudioTrialField | undefined> {
  const pointer = '/' + path.map((part) => part.replace(/~/g, '~0').replace(/\//g, '~1')).join('/');
  return (await trialView(request, projectId, run)).fields.find((field) => field.path === pointer);
}
