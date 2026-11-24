import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

interface Term { id: string; name?: string }
interface File { kind: string; id: string; document?: { payload?: { terms?: Term[] } } }

/** The ordinary project, with its Objective's one term named Effort and a second term, Speed, after it. */
async function twoTerms() {
  const files: File[] = await fixture('ordinary-files', '2');
  const objective = files.find(({ id }) => id === 'simple-reaches-objective')!.document!.payload!;
  const [effort] = objective.terms!;
  objective.terms = [{ ...effort, name: 'Effort' }, { ...effort, id: 'speed', name: 'Speed' }];
  return files;
}

// Catches a Weight that refuses a sweep, a sweep stored at the term's position (it would follow Move Earlier to the
// other term), an Inspector that shows the swept Weight without its values, and an Add Sweep that cannot reach a Weight.
test("a Weight cell's sweep is an axis on the Train step, listed in the Inspector, that stays on its term", async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await twoTerms());
  const train = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps!.train;
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const table = page.locator('[data-slot="primary"]').getByRole('tabpanel').getByRole('table', { name: 'Objective Terms', exact: true });
  /** The Weight of the term at `index`, counting from the first. */
  const weight = (index: number) => table.getByRole('row').nth(index + 1).getByRole('textbox', { name: 'Weight', exact: true });
  const card = page.getByRole('article', { name: 'Train train' });

  await expect(weight(1)).toBeEditable();
  await enter(weight(1), '0.1, 1, 10');
  await expect(sweepLabel(card, 'speed-weight')).toHaveAccessibleName('Sweep speed-weight 3');
  await expect(card.locator('[data-status-mark="not_run"]')).toHaveCount(3);
  expect((await train()).sweep?.axes?.['speed-weight']).toMatchObject({ values: {
    'value-1': { patches: { 'trial_documents.objective.terms.speed.weight.value': { value: 0.1 } } },
  } });

  await weight(1).click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('button', { name: /^Select Weight / })).toHaveText(['0.1', '1', '10']);

  const outline = page.getByRole('complementary', { name: 'Details Drawer' }).getByRole('tree', { name: 'Objective Terms' });
  await outline.getByRole('treeitem', { name: 'Speed', exact: true }).click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Move Earlier', exact: true }).click();
  await expect(table.getByRole('row').nth(1).getByRole('textbox', { name: /^Rename / })).toHaveValue('Speed');
  await expect(card.locator('[data-status-mark="not_run"]')).toHaveCount(3);
  await expect(weight(1)).toHaveValue('1');
  await weight(0).click();
  await expect(inspector.getByRole('button', { name: /^Select Weight / })).toHaveText(['0.1', '1', '10']);

  // A chosen value selects its run, and its cell is marked as differing from the step's Weight.
  const cell = table.getByRole('row').nth(1).getByRole('group', { name: 'Weight field', exact: true });
  await expect(cell.getByRole('img', { name: 'Changed', exact: true })).toHaveCount(0);
  await inspector.getByRole('button', { name: 'Select Weight 10', exact: true }).click();
  await expect(inspector.getByRole('button', { name: 'Select Weight 10', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(cell.getByRole('img', { name: 'Changed', exact: true })).toBeVisible();

  // Add Sweep lists the other term's Weight under Objective and puts the keyboard on its cell, in Sweep kind.
  await card.getByRole('button', { name: 'Add Sweep train' }).click();
  const open = async (menu: string, item: string) => {
    await page.getByRole('menu', { name: menu }).getByRole('menuitem', { name: item, exact: true }).press('Enter');
    await expect(page.getByRole('menu', { name: item })).toBeVisible();
  };
  await open('Add Sweep', 'Objective');
  await open('Objective', 'Effort');
  await page.getByRole('menu', { name: 'Effort' }).getByRole('menuitem', { name: 'Weight', exact: true }).press('Enter');
  await expect(weight(1)).toBeFocused();
  await expect(table.getByRole('row').nth(2).getByRole('button', { name: 'Weight kind: Sweep', exact: true })).toBeVisible();
});
