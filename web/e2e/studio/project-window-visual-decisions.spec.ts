import { expect, test } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { captureStudioWindow, mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  viewport: { width: 1440, height: 900 },
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
});

test('Studio visual decisions use shell roles and keep the narrow window usable', async ({ page }, info) => {
  const id = 'visual-decisions';
  const folder = join(String(info.config.metadata.home), 'state', 'web', 'projects', id);
  await runProjectFixture('tests.fixtures.scene_geometry', 'seed', folder);
  const steps = join(folder, 'steps');
  await mkdir(steps, { recursive: true });
  const kinds = ['train', 'evaluate', 'analyze', 'report'] as const;
  await Promise.all(kinds.map((kind) => writeFile(join(steps, `${kind}.json`), JSON.stringify({
    kind, base: { inline: kind === 'train' ? { trial_documents: { model_name: 'chain', model: { document: 'chain' } } } : {} },
  }))));
  await page.goto(`/?project=${id}`);
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('button', { name: 'Show Map', exact: true }).click();
  await mapStep(page, 'train').press('Enter');
  const cards = page.locator('[data-map-card]');
  await expect(cards).toHaveCount(4);
  const colors = await cards.evaluateAll((elements) => {
    const probe = document.createElement('span');
    document.body.append(probe);
    const variables = (names: string[]) => names.map((name) => {
      probe.style.color = `var(--${name})`;
      return getComputedStyle(probe).color;
    });
    const kindTints = variables(elements.map((element) => `studio-kind-${(element as HTMLElement).dataset.mapStep}`));
    const reserved = variables(['studio-shell-accent', ...['warning', 'running', 'success', 'failure'].map((role) => `studio-status-${role}`)]);
    const borders = elements.map((element) => getComputedStyle(element).borderTopColor);
    probe.remove();
    return { kindTints, reserved, borders };
  });
  expect(new Set(colors.kindTints).size).toBe(4);
  for (const color of colors.kindTints) expect(colors.reserved).not.toContain(color);
  expect(colors.borders.filter((color) => colors.kindTints.includes(color))).toHaveLength(3);
  await captureStudioWindow(page, info, 'visual-map-kinds');
  const greyscale = await page.addStyleTag({ content: 'html { filter: grayscale(1) }' });
  await expect.poll(async () => (await cards.locator('[data-map-kind-label]').allTextContents()).sort())
    .toEqual(['Analyze', 'Evaluate', 'Report', 'Train']);
  await captureStudioWindow(page, info, 'visual-map-greyscale');
  await greyscale.evaluate((node) => node.remove());

  await page.getByRole('button', { name: 'Show Both', exact: true }).click();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const scene = page.getByRole('tabpanel', { name: 'Scene', exact: true });
  const upper = scene.getByRole('button', { name: 'Select upper', exact: true }).filter({ has: page.locator('polyline') });
  const body = upper.locator('polyline');
  await expect(body).toBeVisible();
  const before = await body.evaluate((node) => getComputedStyle(node).stroke);
  expect(colors.reserved).not.toContain(before);
  await upper.press('Enter');
  await expect(upper).toHaveAttribute('aria-pressed', 'true');
  await expect(body).not.toHaveCSS('stroke', before);
  await expect(body).toHaveCSS('stroke', await scene.evaluate((node) => getComputedStyle(node).color));
  await captureStudioWindow(page, info, 'visual-scene-selection');

  await mapStep(page, 'evaluate').press('Enter');
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  await scene.focus();
  await page.setViewportSize({ width: 620, height: 860 });
  await expect(page.getByRole('complementary', { name: 'Details Drawer', exact: true })).toBeHidden();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })).toBeHidden();
  await expect(page.locator('.studio-window-slot__strip')).toHaveCount(0);
  await expect(runs).toBeVisible();
  await expect(scene).toBeVisible();
  await captureStudioWindow(page, info, 'visual-empty-sides-released');
  await page.setViewportSize({ width: 500, height: 860 });
  await expect(page.locator('.studio-window-slot__strip')).toBeVisible();
  await expect(scene).toBeVisible();
});
