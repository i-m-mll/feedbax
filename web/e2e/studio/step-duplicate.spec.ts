import { expect, test, jobTimeout } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type { StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const runs = (page: Page) => page.getByRole('table', { name: 'Runs', exact: true });

test('a duplicated Train step reads its source\'s results until it differs, and one Undo removes it', { tag: '@round' }, async ({ page, request, studio }) => {
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', studio.home, 'step-duplicate',
  ));
  const projectPath = `/api/projects/${projectId}`;
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function rows() {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    return Object.fromEntries((await response.json() as StudioSavedTrainRuns).rows.map((row) => [row.key.step, row]));
  }
  async function command(name: string) {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: new RegExp(`^${name}\\b`) }).click();
  }

  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name: 'Runs', exact: true }).click();
  const submitted = page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/train-runs`);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const accepted = await submitted;
  expect(accepted.status(), await accepted.text()).toBe(202);
  expect((await accepted.json() as StudioJobSubmission).items.map(({ disposition }) => disposition)).toEqual(['admitted']);
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible({ timeout: jobTimeout });
  const before = await saved();

  await command('Duplicate Step Train');
  await expect.poll(async () => Object.keys((await saved()).project.steps ?? {}).sort()).toEqual(['baseline', 'train', 'train-2']);
  const duplicated = await saved();
  expect(duplicated.project.steps!['train-2']).toEqual({ ...before.project.steps!.train, label: 'Train 2' });
  expect(Object.keys(duplicated.project.documents ?? {}).sort()).toEqual(Object.keys(before.project.documents ?? {}).sort());
  const read = await rows();
  expect(read['train-2'].request_id).toBe(read.train.request_id);
  expect(read['train-2'].status).toBe('succeeded');
  expect(read['train-2'].result).toEqual(read.train.result);
  await expect(mapStep(page, 'train-2')).toHaveAttribute('aria-pressed', 'true');
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible();

  await command('Undo');
  await expect.poll(async () => Object.keys((await saved()).project.steps ?? {}).sort()).toEqual(['baseline', 'train']);
  const restored = await saved();
  expect(restored.project).toEqual(before.project);
  expect(restored.presentation).toEqual(before.presentation);

  await mapStep(page, 'train').click();
  await command('Duplicate Step Train');
  await expect.poll(async () => Object.keys((await saved()).project.steps ?? {}).sort()).toEqual(['baseline', 'train', 'train-2']);
  const edited = await request.patch(projectPath, { data: {
    revision: (await saved()).revision, address: { kind: 'step', id: 'train-2' },
    patches: [{ op: 'replace', path: '/base/inline/training_config/n_batches', value: 3 }],
  } });
  expect(edited.ok(), await edited.text()).toBe(true);
  const changed = await rows();
  expect(changed.train.status).toBe('succeeded');
  expect(changed['train-2'].request_id).not.toBe(changed.train.request_id);
  expect(changed['train-2'].status).not.toBe('succeeded');
});
