import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, StudioTrialAuthoring } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

test('renaming a term two Trains read asks at its cell where the name applies, and Escape leaves the cell as it was', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('history-files'));
  /** The first term's name in the Objective each Train reads. */
  const names = async () => {
    const { revision } = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
    return Promise.all(['train', 'sibling'].map(async (step) => {
      const response = await request.post(projectPath + '/authoring/trials', {
        data: { revision, run: { variant: 'default', step, row: 'default' } },
      });
      expect(response.ok(), await response.text()).toBe(true);
      return (await response.json() as StudioTrialAuthoring).objective?.terms?.[0]?.name ?? null;
    }));
  };
  const primary = page.locator('[data-slot="primary"]').getByRole('tabpanel');
  const table = primary.getByRole('table', { name: 'Objective Terms', exact: true });
  const name = table.getByRole('textbox', { name: 'Name', exact: true }).first();
  const menu = page.getByRole('menu', { name: 'Apply To', exact: true });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  await expect(name).toBeEditable();
  await mapStep(page, 'sibling').click({ modifiers: ['Shift'] });
  await expect(primary.locator('.studio-window-slot__header').getByRole('button', { name: /^Objective Document / }))
    .toHaveText(/1 of 2$/);
  await expect(name).toBeEditable();
  const before = await name.inputValue();
  const stored = await names();

  await name.fill('Effort');
  await name.press('Enter');
  await expect(menu).toBeVisible();
  const cell = await name.locator('xpath=ancestor::td').boundingBox();
  const shown = await menu.boundingBox();
  expect(Math.abs(shown!.x - cell!.x)).toBeLessThan(8);
  expect(shown!.y).toBeGreaterThanOrEqual(cell!.y + cell!.height - 1);
  await expect(menu.getByRole('menuitem')).toHaveText(['All 2 Steps', 'Only This Step']);
  await page.keyboard.press('Escape');
  await expect(menu).toBeHidden();
  await expect(name).toHaveValue(before);
  expect(await names()).toEqual(stored);

  await name.fill('Effort');
  await name.press('Enter');
  await menu.getByRole('menuitem', { name: 'All 2 Steps', exact: true }).click();
  await expect.poll(names).toEqual(['Effort', 'Effort']);
  await expect(name).toHaveValue('Effort');
});
