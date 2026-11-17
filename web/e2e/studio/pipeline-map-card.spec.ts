import { expect, test } from './fixtures/studio';
import type { Locator } from '@playwright/test';
import { captureStudioWindow, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { palette } from './fixtures/studioKeyboard';

test.use({ viewport: { width: 1440, height: 900 } });

async function box(locator: Locator) {
  const bounds = await locator.boundingBox();
  expect(bounds).not.toBeNull();
  return bounds!;
}

test('the map card draws one sweep in a line, its head as glyph and name, and the add control outside its border', async ({ page, request }, info) => {
  test.setTimeout(55_000);
  const [model, task, objective, train, successor] = await fixture('managed-files');
  train.step.sweep = { axes: { 'batch-size': { values: Object.fromEntries([2, 4, 8, 16].map((size) => [
    `b${size}`, { label: String(size), patches: { 'training_config.batch_size': { value: size } } },
  ])) } } };
  const { snapshot } = await seedManagedProject(request, [model, task, objective, train, successor]);
  await page.goto(`/?project=${snapshot.project_id}`);
  expect(await page.evaluate(() => document.documentElement.dataset.card)).toBe('ring');

  const card = page.locator('[data-map-step="train"]');
  await mapStep(page, 'train').click();
  await expect(card).toHaveAttribute('data-map-density', 'full');
  const marks = card.locator('.studio-map-run-mark');
  await expect(marks).toHaveCount(4);
  const tops = await Promise.all((await marks.all()).map(async (mark) => Math.round((await box(mark)).y)));
  expect(new Set(tops).size).toBe(1);

  await expect(card.locator('.studio-card__head')).toHaveText('train');
  await card.getByRole('button', { name: 'Select batch-size 4', exact: true }).click();
  await expect(card.locator('[id$="-status"]')).toHaveCount(0);
  await expect(card.getByText(/\d+ of \d+/)).toHaveCount(0);

  const add = card.getByRole('button', { name: /^New Step/ });
  for (const zoom of ['Zoom In', 'Zoom In', 'Zoom Out', 'Zoom Out', 'Zoom Out']) {
    const [border, control] = [await box(card), await box(add)];
    expect(control.x).toBeGreaterThan(border.x + border.width);
    await page.getByRole('button', { name: zoom, exact: true }).first().click();
  }
  await mapStep(page, 'train').click();
  await page.getByRole('button', { name: 'Fit View', exact: true }).first().click();
  const map = page.locator('[data-studio-focus-region="map"]:visible');
  const shoot = async (name: string) => {
    const path = info.outputPath(`${name}.png`);
    await map.screenshot({ path, animations: 'disabled' });
    await info.attach(name, { path, contentType: 'image/png' });
  };
  await shoot('map-light');
  await captureStudioWindow(page, info, 'window-light');
  await page.evaluate(() => {
    localStorage.setItem('feedbax.appearance.theme', 'dark');
    window.dispatchEvent(new StorageEvent('storage'));
  });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await shoot('map-dark');

  await palette(page, 'cardTurbo', 'Turbo Cards');
  await expect(page.locator('html')).toHaveAttribute('data-card', 'turbo');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-card', 'turbo');
  await mapStep(page, 'train').click();
  await shoot('map-dark-turbo');
});

test('the map drawer keeps New Step in view, aligned with its rows, while its list scrolls', async ({ page, request }, info) => {
  test.setTimeout(55_000);
  const [model, task, objective, train, successor] = await fixture('managed-files');
  const extra = Array.from({ length: 12 }, (_, index) => ({ ...structuredClone(train), id: `other-${index}` }));
  const { snapshot } = await seedManagedProject(request, [model, task, objective, train, successor, ...extra]);
  await page.setViewportSize({ width: 1440, height: 640 });
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const drawer = page.getByRole('navigation', { name: 'Steps' });
  if (!await drawer.isVisible()) await palette(page, 'showSteps', 'Show Steps');
  const newStep = drawer.getByRole('button', { name: 'New Step', exact: true });
  const before = await box(newStep);
  const list = drawer.locator('.studio-scroll-fade');
  await list.evaluate((node) => { node.scrollTop = node.scrollHeight; });
  await expect.poll(() => list.evaluate((node) => node.scrollTop)).toBeGreaterThan(0);
  expect(await box(newStep)).toEqual(before);
  const row = drawer.getByRole('button', { name: /^Select train Train/ });
  await row.scrollIntoViewIfNeeded();
  const glyph = (rowFrame: Locator) => rowFrame.locator(':scope > span').nth(1);
  expect(Math.round((await box(glyph(row.locator('xpath=../..')))).x)).toBe(Math.round((await box(glyph(newStep))).x));
  await expect(page.locator('[data-map-density="compact"] [id$="-status"]').first()).toBeVisible();
  await captureStudioWindow(page, info, 'drawer-compact');
});
