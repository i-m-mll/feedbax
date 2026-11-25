import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StudioProjectDocument, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { appendManagedFiles, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

async function open(page: Page, project: string) {
  const response = await page.request.get(`/api/projects/${project}/open`);
  expect(response.ok(), await response.text()).toBe(true);
  return (await response.json() as StudioProjectDocument).snapshot;
}

async function saved(page: Page, project: string, action: () => Promise<unknown>) {
  const saving = page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === `/api/projects/${project}`);
  await action();
  const response = await saving;
  expect(response.ok(), await response.text()).toBe(true);
  await expect(page.getByText('Edited', { exact: true })).toBeHidden();
  return open(page, project);
}

async function runStep(page: Page, project: string, step: string) {
  await mapStep(page, step).click();
  await tab(page, 'Runs').click();
  const admitted = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === `/api/projects/${project}/train-runs`);
  await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
  expect((await admitted).status()).toBe(202);
  await expect(page.getByRole('table', { name: 'Runs', exact: true }).getByText('Done', { exact: true })).toBeVisible({
    timeout: 18_000,
  });
}

test('one reference field sets an Evaluate subject by list and by crosshair, and an Analyze role by kind', async ({ page, studio }) => {
  test.setTimeout(60_000);
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'reference-field'));
  const initial = await open(page, project);
  const analysis = {
    schema: 'feedbax.analysis_document.v1',
    payload: {
      schema_version: 'feedbax.analysis_document.v1',
      inputs: { states: { kind: 'dataset' }, policy: { kind: 'model' }, protocol: { kind: 'task' } },
      operations: {}, views: {},
    },
  };
  await appendManagedFiles(page.request, initial, [
    { kind: 'document', id: 'analysis', document: analysis },
    { kind: 'step', id: 'analyze', step: { kind: 'analyze', base: { document: 'analysis' }, inputs: { states: { step: 'baseline' } } } },
  ]);
  // A document is offered by its label, the name the Model and Task views give it.
  const models = Object.entries(initial.project.documents ?? {}).filter(([, { schema }]) => schema === 'feedbax.spec.model')
    .map(([id, { label }]) => ({ id, name: label ?? id }));
  const tasks = Object.entries(initial.project.documents ?? {})
    .filter(([, { schema }]) => schema === 'feedbax.spec.python_task' || schema === 'feedbax.spec.task_document')
    .map(([id, { label }]) => label ?? id);
  expect(models.length).toBeGreaterThan(0);

  await page.goto('/?project=' + encodeURIComponent(project));
  await mapStep(page, 'baseline').click();
  await tab(page, 'Runs').click();
  const inputs = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const subject = inputs.getByRole('combobox', { name: 'Subject', exact: true });
  await expect(subject).toHaveText('train');

  const toDocument = await saved(page, project, () => pick(subject, models[0].name));
  expect(toDocument.project.steps!.baseline.inputs).toMatchObject({
    subject: { kind: 'initial_model', source: { document: models[0].id } },
    task: { kind: 'task', source: { document: expect.any(String) } },
  });
  await expect(subject).toHaveText(models[0].name);
  await expect(inputs.getByRole('combobox', { name: 'Task', exact: true })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);

  await inputs.getByRole('button', { name: 'Pick Subject', exact: true }).click();
  await expect(page.locator('[data-map-step="train"]')).toHaveAttribute('data-pick', 'allowed');
  await expect(page.locator('[data-map-step="baseline"]')).toHaveAttribute('data-pick', 'refused');
  const toCheckpoint = await saved(page, project, () => mapStep(page, 'train').click());
  expect(toCheckpoint.project.steps!.baseline.inputs).toEqual({ subject: { step: 'train', checkpoint: 'last' } });
  await expect(page.locator('[data-pick]')).toHaveCount(0);
  await expect(subject).toHaveText('train');
  await expect(inputs.getByRole('combobox', { name: 'Task', exact: true })).toHaveCount(0);

  await mapStep(page, 'analyze').click();
  await tab(page, 'Analysis').click();
  const canvas = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await canvas.getByRole('group', { name: 'Select policy', exact: true }).click();
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Show Inspector', exact: true }).click();
  const policy = inspector.getByRole('combobox', { name: 'Policy', exact: true });
  await policy.click();
  const listbox = page.getByRole('listbox');
  await expect(listbox.getByRole('option')).toHaveText([...models.map(({ name }) => name), 'train']);
  await page.keyboard.press('Escape');
  const bound = await saved(page, project, () => pick(policy, 'train'));
  expect(bound.project.steps!.analyze.inputs?.policy).toEqual({ step: 'train', checkpoint: 'last' });
  await canvas.getByRole('group', { name: 'Select protocol', exact: true }).click();
  const protocol = inspector.getByRole('combobox', { name: 'Protocol', exact: true });
  await protocol.click();
  await expect(listbox.getByRole('option')).toHaveText(tasks);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('Evaluate edits that remove runs ask first and keep the other results', { tag: '@round' }, async ({ page, studio }) => {
  test.setTimeout(90_000);
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'reference-removal'));
  const initial = await open(page, project);
  await appendManagedFiles(page.request, initial, [{ kind: 'step', id: 'trials', step: {
    ...initial.project.steps!.baseline, sweep: { axes: { subject: { input: 'subject' } } },
  } }]);
  await page.addInitScript(() => window.localStorage.setItem('feedbax.settings', JSON.stringify({ removalThreshold: 0 })));
  await page.goto('/?project=' + encodeURIComponent(project));
  await runStep(page, project, 'train');
  await runStep(page, project, 'trials');
  const plan = async () => {
    const { revision } = await open(page, project);
    const response = await page.request.get(`/api/projects/${project}/train-runs`, { params: { revision, window_map: 'true' } });
    return await response.json() as StudioSavedTrainRuns;
  };
  const requests = (runs: StudioSavedTrainRuns) => Object.fromEntries([
    ...runs.rows.map(({ key, request_id }) => [key.step + '/' + key.row, request_id]),
    ...(runs.window_map?.evaluations ?? []).map(({ key, request_id }) => [key.step + '/' + key.row, request_id]),
  ]);
  const before = requests(await plan());
  const trialSet = page.getByRole('region', { name: 'Trials', exact: true });
  const removal = page.getByRole('dialog', { name: /^Remove \d+ runs?\?$/ });
  const condition = trialSet.getByRole('checkbox').first();
  await pick(trialSet.getByRole('combobox'), 'Selected Conditions');
  await condition.check();
  await expect(removal).toContainText('Existing results remain on disk but disappear from the project.');
  await removal.getByRole('button', { name: 'Cancel', exact: true }).click();
  expect(requests(await plan())).toEqual(before);

  await condition.uncheck();
  const removed = await saved(page, project, async () => {
    await condition.check();
    await removal.getByRole('button', { name: 'Remove', exact: true }).click();
  });
  const after = requests(await plan());
  const changed = Object.keys(before).filter((key) => key.startsWith('trials/'));
  expect(changed.length).toBeGreaterThan(0);
  for (const key of changed) expect(after[key]).not.toBe(before[key]);
  const kept = Object.fromEntries(Object.entries(before).filter(([key]) => !changed.includes(key)));
  expect(Object.fromEntries(Object.keys(kept).map((key) => [key, after[key]]))).toEqual(kept);
  const done = (await plan()).rows.find(({ key }) => key.step === 'train');
  expect(done?.result).toBeTruthy();
  expect(removed.project.steps!.trials.base).toMatchObject({ inline: { params: { trial_set: { kind: 'conditions' } } } });
});
