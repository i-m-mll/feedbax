import { expect, test } from '@playwright/test';
import { join } from 'node:path';
import type { StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('a real Model edit updates map, drawer, and Runs from the saved plan', async ({ page, request }) => {
  test.setTimeout(45_000);
  const files = await fixture('files');
  const { snapshot, projectPath } = await seedManagedProject(request, files.slice(0, 3));
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  const card = page.getByRole('article', { name: 'Train train' });
  const drawer = page.getByRole('navigation', { name: 'Steps' });
  const table = page.getByRole('table', { name: 'Runs' });
  const run = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === `${projectPath}/train-runs`);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  expect((await run).status()).toBe(202);
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 15_000 });

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await rail.getByRole('tab', { name: 'Model', exact: true }).click();
  await page.getByRole('complementary', { name: 'Details Drawer' }).getByRole('button', { name: 'plant', exact: true }).click();
  const mass = page.getByRole('complementary', { name: 'Inspector' }).getByRole('spinbutton', { name: 'Edit Mass' });
  await expect(mass).toHaveValue('1');
  const write = page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === projectPath);
  await mass.fill('2');
  await mass.press('Enter');
  expect((await write).ok()).toBe(true);
  await expect(table.getByText('Out of date', { exact: true })).toBeVisible();
  await expect(card.getByText('1 Out of date')).toBeVisible();
  await expect(drawer.getByText('1 Out of date')).toBeVisible();
  await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
  await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: 'Undo', exact: true }).click();
  await expect(table.getByText('Done', { exact: true })).toBeVisible();
  await expect(card.getByText('1 Done')).toBeVisible();
  await expect(drawer.getByText('1 Done')).toBeVisible();
});

test('the managed window shares selection and restores the folded slot without editing the folder', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('files'));
  const id = snapshot.project_id;
  const key = `feedbax:studio-window:v1:${id}`;
  const prefs = () => page.evaluate((key) => JSON.parse(localStorage.getItem(key)!), key);
  await page.goto(`/?project=${encodeURIComponent(id)}`);
  await mapStep(page, 'train').click();
  const primary = page.locator('[data-slot="primary"]');
  const secondary = page.locator('[data-slot="secondary"]');
  await expect(primary.getByRole('table', { name: 'Runs' })).toBeVisible();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await expect(secondary.getByRole('heading', { name: 'Scene', exact: true })).toBeVisible();

  await mapStep(page, 'successor').click();
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Forward', exact: true }).click();
  await expect(mapStep(page, 'successor')).toHaveAttribute('aria-pressed', 'true');

  const separator = page.getByRole('separator', { name: 'Slot Width', exact: true });
  await separator.press('ArrowLeft');
  const width = await separator.getAttribute('aria-valuenow');
  const before = await prefs();
  await page.setViewportSize({ width: 620, height: 850 });
  await expect(separator).toBeHidden();
  await expect(page.locator('.studio-window-slot__strip')).toBeVisible();
  expect(await prefs()).toEqual(before);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(separator).toHaveAttribute('aria-valuenow', width!);

  await page.getByRole('button', { name: 'Show Details', exact: true }).click();
  await expect(mapStep(page, 'successor')).toBeHidden();
  await expect(page.locator('.studio-window-map__compact').getByRole('button', { name: 'Select successor' })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Show Details', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(secondary.getByRole('heading', { name: 'Scene', exact: true })).toBeVisible();
  await expect(separator).toHaveAttribute('aria-valuenow', width!);

  const latest = await request.get(projectPath);
  expect(latest.ok()).toBe(true);
  const reopened = await latest.json() as StudioProjectSnapshot;
  expect(reopened.revision).toBe(snapshot.revision);
  expect(reopened.presentation_revision).toBe(snapshot.presentation_revision);
});

test('mounted Scene owns its drawer and inspector and Problems matches the saved Plan', async ({ page, request }, info) => {
  const home = String(info.config.metadata.home);
  const id = 'project-window-scene';
  const path = `/api/projects/${id}`;
  await runProjectFixture('tests.fixtures.scene_geometry', 'seed',
    join(home, 'state', 'web', 'projects', id));
  const initial = await request.get(path);
  expect(initial.ok()).toBe(true);
  let snapshot = await initial.json() as StudioProjectSnapshot;
  const [, task, template] = await fixture('files');
  for (const file of [
    task,
    { ...template, step: { ...template.step, base: { inline: { ...template.step.base.inline, graph: { document: 'chain' } } } } },
    { ...template, id: 'single', step: { ...template.step, base: { inline: { ...template.step.base.inline, graph: { document: 'single' } } } } },
    { kind: 'step', id: 'broken', step: { kind: 'train', base: { inline: {} } } },
  ]) {
    const saved = await request.post(`${path}/files`, { data: { revision: snapshot.revision, file } });
    expect(saved.status(), await saved.text()).toBe(201);
    ({ snapshot } = await saved.json() as StudioProjectDocument);
  }
  await page.goto(`/?project=${id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const scene = page.locator('[data-slot="secondary"]').getByRole('tabpanel', { name: 'Scene', exact: true });
  const upper = scene.getByRole('button', { name: 'Select upper', exact: true });
  await expect(upper).toBeVisible();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer' });
  await drawer.getByRole('region', { name: 'Bodies' }).getByRole('button', { name: 'upper', exact: true }).click();
  await expect(upper).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toContainText('upper.proximal');
  const primary = page.locator('[data-slot="primary"] > section');
  await primary.focus();
  await page.keyboard.press('Control+3');
  await expect(primary.getByRole('heading', { name: 'Task', exact: true })).toBeVisible();
  await expect(primary.locator('.studio-window-slot__controls')).toContainText('task');
  await expect(page.getByRole('complementary', { name: 'Inspector' })).not.toContainText('upper.proximal');
  await scene.focus();
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toContainText('upper.proximal');
  await mapStep(page, 'single').click();
  await expect(scene.locator('header')).toContainText('single');
  await expect(scene.getByRole('button', { name: 'Select forearm', exact: true })).toBeHidden();
  await expect(page.getByRole('complementary', { name: 'Inspector' })).not.toContainText('upper.proximal');
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(upper).toBeVisible();

  const read = await request.get(`${path}/train-runs?revision=${snapshot.revision}&window_map=true`);
  expect(read.ok()).toBe(true);
  const { window_map } = await read.json() as StudioSavedTrainRuns;
  const problems = window_map!.problems.filter(({ variant }) => variant === null || variant === 'default');
  expect(problems.some(({ step }) => step === 'broken')).toBe(true);
  const problemButton = page.getByRole('button', { name: 'Problems', exact: true });
  await expect(problemButton).toHaveText(`Problems ${problems.length}`);
  await expect(problemButton).toHaveAccessibleDescription(String(problems.length));
  await problemButton.click();
  const list = page.getByRole('dialog', { name: 'Problems' });
  await expect(list.getByRole('listitem')).toHaveCount(problems.length);
  const problem = problems.find(({ step }) => step === 'broken')!;
  await list.getByRole('button', { name: `${problem.path}: ${problem.detail}`, exact: true }).first().click();
  await expect(mapStep(page, 'broken')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('dialog', { name: 'Problems' })).toBeHidden();
});
