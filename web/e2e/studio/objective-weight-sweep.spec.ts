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
// other term), and an Inspector that shows the swept Weight without its values.
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
  await expect(table.getByRole('row').nth(1).getByRole('textbox', { name: 'Name', exact: true })).toHaveValue('Speed');
  await expect(card.locator('[data-status-mark="not_run"]')).toHaveCount(3);
  await expect(weight(1)).toHaveValue('1');
  await weight(0).click();
  await expect(inspector.getByRole('button', { name: /^Select Weight / })).toHaveText(['0.1', '1', '10']);
});
