import { expect, test } from './fixtures/studio';
import type { Locator, Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { chooseSweepValue, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type File = { kind: string; id: string; document?: unknown; step?: { sweep?: unknown } };

/** The ordinary Train step, swept over a set whose `drop` entry removes the Point mass and the wire that drives it. */
async function files() {
  const ordinary = await fixture('ordinary-files', '2') as File[];
  const set: File = { kind: 'document', id: 'arms', document: { schema: 'feedbax.set_document.v1', payload: { entries: {
    keep: {},
    drop: { differences: { '/interior/components/plant': { op: 'remove' }, '/interior/wires/drive': { op: 'remove' } } },
  } } } };
  return [...ordinary.filter(({ kind }) => kind === 'document'), set,
    ...ordinary.filter(({ kind }) => kind === 'step').map((file) => ({ ...file, step: { ...file.step, sweep: { axes: { arm: { set: 'arms' } } } } }))];
}

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
test('a component a selected set entry removes is drawn in place as a ghost, and Reset restores it', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await files());
  const card = page.locator('[data-map-step="train"]');
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const plant = model.locator('.react-flow__node[data-id="plant"]');
  const ghost = plant.locator('[data-ghost="true"]');
  const removedMark = plant.getByRole('button', { name: 'Removed', exact: true });
  const drive = model.locator('.react-flow__edge[data-id="drive"]');
  const placed = ['@inputs', 'cell', 'readout', 'plant', '@outputs'];

  async function open(page: Page) {
    await page.goto(`/?project=${snapshot.project_id}`);
    await mapStep(page, 'train').click();
    await page.keyboard.press('Meta+2');
  }
  await open(page);
  await expect(plant).toBeVisible();
  await expect(ghost).toHaveCount(0);
  const before = await boxes(model, placed);

  await chooseSweepValue(card, 'arms', 'drop');
  await expect(removedMark).toBeVisible();
  await expect(ghost).toHaveCount(1);
  await expect(drive).toBeVisible();
  expect(await boxes(model, placed)).toEqual(before);

  await mapStep(page, 'train').click();
  await expect(removedMark).toHaveCount(0);
  await expect(ghost).toHaveCount(0);
  expect(await boxes(model, placed)).toEqual(before);

  await chooseSweepValue(card, 'arms', 'drop');
  await removedMark.click();
  const [edited] = await Promise.all([
    page.waitForResponse((response) => new URL(response.url()).pathname === `${projectPath}/model/edit`),
    page.getByRole('menuitem', { name: 'Reset', exact: true }).click(),
  ]);
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
