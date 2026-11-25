import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const menu = (page: Page) => page.getByRole('menu');

/** The step menu lists Rename Step, Change Key and Duplicate Step, with Rename's key at the right. */
async function expectStepMenu(page: Page) {
  await expect(menu(page).getByRole('menuitem')).toHaveText(['Rename StepF2', 'Change Key', 'Duplicate Step']);
}

test('a step\'s context menu opens on its card, its drawer row and its name in Runs, and acts on that step', async ({ page, request, studio }) => {
  const projectId: string = JSON.parse(await runProjectFixture(
    'web.e2e.studio.fixtures.canonical_evaluate', 'seed-step-rename', studio.home, 'step-menu',
  ));
  async function steps() {
    const response = await request.get(`/api/projects/${projectId}`);
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectSnapshot).project.steps ?? {};
  }
  /** Duplicate Step from the open menu copies `source`. */
  async function duplicates(source: string) {
    const before = await steps();
    await choose('Duplicate Step');
    await expect.poll(async () => Object.keys(await steps()).length).toBe(Object.keys(before).length + 1);
    const after = await steps();
    const copy = Object.keys(after).find((id) => !(id in before))!;
    expect(after[copy]).toEqual(before[source]);
  }
  async function choose(item: string) {
    await menu(page).getByRole('menuitem', { name: item }).click();
    await expect(menu(page)).toHaveCount(0);
  }
  async function renames(field: Locator, restore: Locator) {
    await expect(field).toBeFocused();
    await field.press('Escape');
    await expect(restore).toBeVisible();
  }

  await page.goto(`/?project=${encodeURIComponent(projectId)}`);
  const card = page.locator('[data-map-step="train"]');
  const cardRename = card.getByRole('textbox', { name: 'Rename train', exact: true });

  await mapStep(page, 'train').click({ button: 'right' });
  await expectStepMenu(page);
  await choose('Rename Step');
  await renames(cardRename, mapStep(page, 'train'));

  await card.locator('[data-result-run]').first().click({ button: 'right' });
  await expect(menu(page).getByRole('menuitem')).toHaveText(['Check', 'Accept', 'Duplicate Run', 'Export']);
  await page.keyboard.press('Escape');
  await expect(menu(page)).toHaveCount(0);

  await page.locator('[data-map-canvas]').click({ button: 'right', position: { x: 8, y: 8 } });
  await expect(menu(page).locator('[data-step-insert]').first()).toBeVisible();
  await page.keyboard.press('Escape');

  await mapStep(page, 'baseline').focus();
  await page.keyboard.press('Shift+F10');
  await expectStepMenu(page);
  await duplicates('baseline');

  const drawer = page.getByRole('navigation', { name: 'Steps', exact: true });
  const row = drawer.getByRole('treeitem', { name: /^train\b/ });
  if (!await row.isVisible()) {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name: /^Show Steps\b/ }).click();
  }
  await row.click({ button: 'right' });
  await expectStepMenu(page);
  await choose('Rename Step');
  await renames(drawer.getByRole('textbox', { name: 'Rename', exact: true }), row);
  await row.focus();
  await page.keyboard.press('ContextMenu');
  await expectStepMenu(page);
  await duplicates('train');

  await mapStep(page, 'train').click();
  await page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name: 'Runs', exact: true }).click();
  const title = page.getByRole('heading', { name: 'train', exact: true, level: 3 });
  await title.click({ button: 'right' });
  await expectStepMenu(page);
  await choose('Rename Step');
  await renames(cardRename, mapStep(page, 'train'));
  await title.locator('..').focus();
  await page.keyboard.press('Shift+F10');
  await duplicates('train');
});
