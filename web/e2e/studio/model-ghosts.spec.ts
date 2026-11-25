import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { chooseSweepValue, mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { modelEdit, pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

async function boxes(model: Locator, ids: readonly string[]) {
  return Object.fromEntries(await Promise.all(ids.map(async (id) => {
    const box = (await model.locator(`.react-flow__node[data-id="${id}"]`).boundingBox())!;
    return [id, [Math.round(box.x), Math.round(box.y)]] as const;
  })));
}

/**
 * A set entry that removes a component draws it where it was, dashed with the D mark, while the entry is selected;
 * nothing else moves, and Reset on the mark gives the entry the component back.
 */
test('a component a selected set entry removes is drawn in place as a ghost, and Reset restores it', async ({ page, request, studio }) => {
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.model_ghosts', 'seed', studio.home, 'ghosts')) as string;
  const projectPath = `/api/projects/${project}`;
  const card = page.locator('[data-map-step="train"]');
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const plant = model.locator('.react-flow__node[data-id="plant"]');
  const ghost = plant.locator('[data-ghost="true"]');
  const removedMark = plant.getByRole('button', { name: 'Removed', exact: true });
  const drive = model.locator('.react-flow__edge[data-id="drive"]');
  const placed = ['@inputs', 'cell', 'readout', 'plant', '@outputs'];

  async function open(page: Page) {
    await page.goto(`/?project=${project}`);
    await mapStep(page, 'train').click();
    await page.keyboard.press('Meta+2');
  }
  await open(page);
  await expect(plant).toBeVisible();
  await expect(ghost).toHaveCount(0);
  const before = await boxes(model, placed);

  await chooseSweepValue(card, 'Arms', 'Drop');
  await expect(removedMark).toBeVisible();
  await expect(ghost).toHaveCount(1);
  await expect(drive).toBeVisible();
  expect(await boxes(model, placed)).toEqual(before);

  await mapStep(page, 'train').click();
  await expect(removedMark).toHaveCount(0);
  await expect(ghost).toHaveCount(0);
  expect(await boxes(model, placed)).toEqual(before);

  await chooseSweepValue(card, 'Arms', 'Drop');
  await removedMark.click();
  const edited = await modelEdit(page, projectPath,
    () => page.getByRole('menuitem', { name: 'Reset', exact: true }).click(), 'Set Entry');
  expect(edited.request().postDataJSON()).toMatchObject({
    command: { operation: { op: 'reset_difference' } }, axes: { arm: ['drop'] },
  });
  expect((await edited.json()).result.status).toBe('applied');
  await expect(removedMark).toHaveCount(0);
  await expect(ghost).toHaveCount(0);
  await expect(plant).toBeVisible();
  const stored = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  const entries = (stored.project.documents!.arms as unknown as { payload: { entries: Record<string, { differences?: object }> } }).payload.entries;
  expect(entries.drop.differences ?? {}).toEqual({});
});

/** Outside the primary variant the Model read with the selected entry is not refused, and its removal is drawn the same way. */
test('a set entry selected in another variant draws its removal as a ghost', async ({ page, studio }) => {
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.model_ghosts', 'seed', studio.home, 'ghosts-variant')) as string;
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const plant = model.locator('.react-flow__node[data-id="plant"]');
  await page.goto(`/?project=${project}`);
  await pick(page.getByRole('combobox', { name: 'Variant', exact: true }), 'control');
  await mapStep(page, 'train').click();
  await page.keyboard.press('Meta+2');
  await expect(plant).toBeVisible();
  await chooseSweepValue(page.locator('[data-map-step="train"]'), 'Arms', 'Drop');
  await expect(plant.locator('[data-ghost="true"]')).toHaveCount(1);
  await expect(plant.getByRole('img', { name: 'Removed', exact: true })).toBeVisible();
});
