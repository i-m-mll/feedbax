import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

interface Term { id: string; measure: unknown; weight: unknown; name?: string }

/** The ordinary Train project open on its Objective view, whose one term penalises a quantity's squared size. */
async function open(page: Page, request: APIRequestContext) {
  const files = await fixture('ordinary-files', '2');
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const terms = async () => {
    const saved = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
    return (rootPayload(saved.project.documents?.['simple-reaches-objective']) as unknown as { terms: Term[] }).terms;
  };
  const table = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Objective Terms', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  return { table, inspector, terms };
}

// Catches a penalty swap that retypes or loses the operand, adds an argument, or touches another field of the term;
// and a Formula that cannot be reached again from a Draw chosen around it (the owner's annotation 4).
test("a Measure's outer penalty is chosen in its parts, and a kind is left as it was entered", async ({ page, request }) => {
  test.setTimeout(45_000);
  const { table, inspector, terms } = await open(page, request);
  const [before] = await terms();
  const measure = table.getByRole('textbox', { name: 'Measure', exact: true });
  await expect(measure).toHaveValue(/^sqnorm\(.+\)$/);
  const operand = (await measure.inputValue()).slice('sqnorm('.length, -1);
  await measure.click();
  await inspector.getByRole('combobox', { name: 'Function', exact: true }).click();
  await expect(page.getByRole('option')).not.toContainText(['sub']);
  await page.getByRole('option', { name: 'huber', exact: true }).click();
  await expect(measure).toHaveValue(`huber(${operand})`);
  await expect.poll(async () => (await terms())[0].measure).toMatchObject({ function: 'huber', args: [(before.measure as { args: unknown[] }).args[0]] });
  const [after] = await terms();
  expect({ ...after, measure: undefined }).toEqual({ ...before, measure: undefined });

  const weight = table.getByRole('textbox', { name: 'Weight', exact: true });
  const formula = '2 * 0.5';
  await enter(weight, formula);
  await expect.poll(async () => (await terms())[0].weight).toMatchObject({ kind: 'call', function: 'mul' });
  await table.getByRole('button', { name: 'Weight kind: Formula', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Draw', exact: true }).click();
  await expect(inspector.getByRole('textbox', { name: 'Mean', exact: true })).toHaveValue(formula);
  await table.getByRole('button', { name: 'Weight kind: Draw', exact: true }).click();
  await expect(page.getByRole('menuitemradio', { name: 'Draw', exact: true })).toBeChecked();
  await page.getByRole('menuitemradio', { name: 'Formula', exact: true }).click();
  await expect(weight).toHaveValue(formula);
  await expect(weight).toBeFocused();
  await expect(table.getByRole('button', { name: 'Weight kind: Formula', exact: true })).toBeVisible();
  await expect(inspector.getByRole('textbox', { name: 'Mean', exact: true })).toHaveCount(0);

  // A Shape made from the Weight holds it, and draws its curve over the term's steps in the trial the clock shows.
  await table.getByRole('button', { name: 'Weight kind: Formula', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Shape', exact: true }).click();
  await expect(weight).toHaveValue(`hold(${formula})`);
  await expect(inspector.getByRole('combobox', { name: 'Function', exact: true })).toHaveText('hold');
  await expect(inspector.getByRole('img', { name: 'Weight curve', exact: true })).toBeVisible();
});

// Catches an Add Term that is not where the new row appears, a new row that is not selected, and a Measure the user
// must find before typing it.
test('Add Term at the foot of the Term column adds a row with its Measure focused', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { table, terms } = await open(page, request);
  await expect(table.getByRole('row')).toHaveCount(3);
  const add = table.getByRole('button', { name: 'Add Term', exact: true });
  await expect(add).toBeVisible();
  await add.click();
  await page.getByRole('menuitem', { name: 'Goal', exact: true }).click();
  await expect.poll(async () => (await terms()).length).toBe(2);
  const row = table.getByRole('row').nth(2);
  await expect(row.getByRole('textbox', { name: 'Rename Goal', exact: true })).toHaveValue('Goal');
  await expect(row.getByRole('textbox', { name: 'Measure', exact: true })).toBeFocused();
  await expect(row.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(goal)');
});
