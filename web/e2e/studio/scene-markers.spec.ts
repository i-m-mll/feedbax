import { expect, test } from './fixtures/studio';
import type { StudioProjectDocument, StudioProjectExamples } from '../../src/generated/studioContracts';
import { mapStep } from './fixtures/canonicalTrain';
import { testProjectName } from './fixtures/studioKeyboard';

test.use({ viewport: { width: 1440, height: 900 } });

// Catches the shipped example's start and target drawn by one marker, a marker chosen on the list that the canvas does
// not draw, and one that is not stored on the Task, so that reopening the project draws the default again.
test('the example\'s start and target draw different markers, and a chosen marker survives reopening', async ({ page, request }) => {
  const { examples } = await (await request.get('/api/projects/examples')).json() as StudioProjectExamples;
  const created = await request.post('/api/projects', { data: { name: testProjectName(), example: examples[0].id } });
  expect(created.status()).toBe(201);
  const { snapshot } = await created.json() as StudioProjectDocument;
  const scene = page.getByRole('tabpanel', { name: 'Scene', exact: true });
  const marker = (name: string) => page.getByRole('tree', { name: 'Outline', exact: true })
    .getByRole('treeitem', { name: `${name} Marker`, exact: true }).getByRole('button', { name: 'Marker', exact: true });
  const drawn = (name: string) => scene.getByRole('button', { name: `Select ${name}`, exact: true }).locator('[data-scene-shown]');
  const open = async () => {
    await page.goto(`/?project=${snapshot.project_id}`);
    await mapStep(page, 'simple-reaches').click();
    await page.getByRole('tablist', { name: 'Details' }).getByRole('tab', { name: 'Scene', exact: true }).click();
    // The example's first trial is sampled once its Task is compiled, which a fresh Service takes longer than a step to do.
    await expect(drawn('target')).toHaveCount(1, { timeout: 30_000 });
  };
  await open();
  const start = await drawn('start').getAttribute('data-scene-shown');
  expect(start).not.toBe(await drawn('target').getAttribute('data-scene-shown'));

  await marker('target').click();
  const menu = page.getByRole('menu', { name: 'Marker target', exact: true });
  await menu.getByRole('menuitemradio', { name: 'Triangle' }).click();
  await expect(drawn('target')).toHaveAttribute('data-scene-shown', 'triangle open');
  await marker('target').click();
  await menu.getByRole('menuitemcheckbox', { name: 'Filled' }).click();
  await expect(drawn('target')).toHaveAttribute('data-scene-shown', 'triangle filled');

  await open();
  await expect(drawn('target')).toHaveAttribute('data-scene-shown', 'triangle filled');
  await expect(drawn('start')).toHaveAttribute('data-scene-shown', start!);
});
