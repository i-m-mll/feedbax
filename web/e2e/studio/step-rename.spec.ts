import { expect, test, type Page } from '@playwright/test';
import type { StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  channel: 'chrome',
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

const runs = (page: Page) => page.getByRole('table', { name: 'Runs', exact: true });

test('renaming a Train step keeps its runs and its Evaluate reader Done, and one Undo restores it', async ({ page, request }, info) => {
  test.setTimeout(90_000);
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', String(info.config.metadata.home), 'step-rename',
  ));
  const projectPath = `/api/projects/${projectId}`;
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function statuses() {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    const plan = (await response.json() as StudioSavedTrainRuns).window_map!;
    return Object.fromEntries(plan.rows.map(({ key, status }) => [key.step, status]));
  }
  async function run(step: string) {
    await mapStep(page, step).click();
    await page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name: 'Runs', exact: true }).click();
    const response = page.waitForResponse((response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname === `${projectPath}/train-runs`);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const accepted = await response;
    expect(accepted.status(), await accepted.text()).toBe(202);
    expect((await accepted.json() as StudioJobSubmission).items.map(({ disposition }) => disposition)).toEqual(['admitted']);
    await expect(runs(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 30_000 });
  }

  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  await run('train');
  await run('baseline');
  const before = await saved();
  expect(await statuses()).toEqual({ train: 'succeeded', baseline: 'succeeded' });

  await mapStep(page, 'train').click();
  await mapStep(page, 'train').press('Enter');
  const name = page.locator('[data-map-step="train"]').getByRole('textbox', { name: 'Rename train', exact: true });
  await expect(name).toBeFocused();
  await name.fill('baseline');
  await name.press('Enter');
  await expect(name).toBeFocused();
  await name.fill('fit');
  await name.press('Enter');

  await expect(mapStep(page, 'fit')).toBeVisible();
  await expect.poll(async () => Object.keys((await saved()).project.steps ?? {}).sort()).toEqual(['baseline', 'fit']);
  const renamed = await saved();
  expect(renamed.file_revisions.filter(({ address }) => address.kind === 'step').map(({ address }) => address.id).sort())
    .toEqual(['baseline', 'fit']);
  expect(renamed.project.steps!.fit.group).toBe('fitting');
  expect(renamed.project.steps!.baseline.inputs?.subject).toMatchObject({ step: 'fit' });
  const targets = Object.values(renamed.project.sweep!.axes!.mass as { values: Record<string, { patches: Record<string, unknown> }> })
    .flatMap((axis) => Object.values(axis).flatMap(({ patches }) => Object.keys(patches)));
  expect(targets.length).toBeGreaterThan(0);
  for (const target of targets) expect(target.startsWith('steps.fit.{')).toBe(true);
  expect(renamed.presentation?.steps).toEqual({ fit: before.presentation!.steps!.train });
  expect(await statuses()).toEqual({ fit: 'succeeded', baseline: 'succeeded' });
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible();
  await mapStep(page, 'baseline').click();
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: /^Undo\b/ }).click();
  await expect(mapStep(page, 'train')).toBeVisible();
  const restored = await saved();
  expect(restored.project).toEqual(before.project);
  expect(restored.presentation).toEqual(before.presentation);
  expect(await statuses()).toEqual({ train: 'succeeded', baseline: 'succeeded' });

  await mapStep(page, 'train').click();
  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: 'Rename Step train' }).click();
  const palette = page.locator('[data-map-step="train"]').getByRole('textbox', { name: 'Rename train', exact: true });
  await expect(palette).toBeFocused();
  await palette.press('Escape');
  await expect(mapStep(page, 'train')).toBeVisible();
});
