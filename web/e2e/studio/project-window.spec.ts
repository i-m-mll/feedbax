import { expect, test } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

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

  await page.getByRole('button', { name: 'Split View', exact: true }).click();
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await rail.getByRole('button', { name: 'Model', exact: true }).click();
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
  await page.getByRole('button', { name: 'Split View', exact: true }).click();
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

  await page.getByRole('button', { name: 'Details Only', exact: true }).click();
  await expect(mapStep(page, 'successor')).toBeHidden();
  await expect(page.locator('.studio-window-map__compact').getByRole('button', { name: 'Select successor' })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('button', { name: 'Details Only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(secondary.getByRole('heading', { name: 'Scene', exact: true })).toBeVisible();
  await expect(separator).toHaveAttribute('aria-valuenow', width!);

  const latest = await request.get(projectPath);
  expect(latest.ok()).toBe(true);
  const reopened = await latest.json() as StudioProjectSnapshot;
  expect(reopened.revision).toBe(snapshot.revision);
  expect(reopened.presentation_revision).toBe(snapshot.presentation_revision);
});
