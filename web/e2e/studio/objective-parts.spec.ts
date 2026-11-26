import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

interface Term { measure: unknown; weight: unknown; label?: string }

/** The ordinary Train project open on its Objective view, whose one term penalises a quantity's squared size. */
async function open(page: Page, request: APIRequestContext) {
  const files = await fixture('ordinary-files', '2');
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const terms = async () => {
    const saved = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
    return Object.values((rootPayload(saved.project.documents?.['simple-reaches-objective']) as unknown as { terms: Record<string, Term> }).terms);
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
  await expect(inspector.getByRole('region', { name: 'Weight field', exact: true }).getByRole('combobox', { name: 'Function', exact: true })).toHaveText('hold');
  await expect(inspector.getByRole('img', { name: 'Weight curve', exact: true })).toBeVisible();
});

// Catches a menu whose view the service no longer holds left without its functions, instead of read again from a
// reloaded view.
test('a function menu of a view the service no longer holds is read again from the reloaded view', async ({ page, request }) => {
  const { table, inspector } = await open(page, request);
  const contexts: string[] = [];
  let expired = false;
  await page.route((url) => url.pathname.endsWith('/authoring/trials/expression-menu'), async (route) => {
    contexts.push((route.request().postDataJSON() as { expression_context: string }).expression_context);
    if (expired) return route.fallback();
    expired = true;
    await route.fulfill({ status: 409, json: { detail: { code: 'trial_expression_context_unavailable', message: 'Reload' } } });
  });
  await table.getByRole('textbox', { name: 'Measure', exact: true }).click();
  await inspector.getByRole('combobox', { name: 'Function', exact: true }).click();
  await expect(page.getByRole('option', { name: 'huber', exact: true })).toBeVisible();
  expect(contexts).toHaveLength(2);
  expect(contexts[1]).not.toBe(contexts[0]);
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

// Catches an Enabled check pushed off a narrow table by the Inspector, a shipped term whose Enabled cannot be turned
// off, an unnamed term whose Term cell is blank, and an Inspector whose fields move to put the clicked one first.
test('with the Inspector open, an unnamed term shows its name and its Enabled check turns it off', async ({ page, request }) => {
  test.setTimeout(45_000);
  await page.setViewportSize({ width: 1024, height: 760 });
  const { table, inspector, terms } = await open(page, request);
  const [before] = await terms();
  expect(before.label).toBeUndefined();
  await expect(table.getByRole('textbox', { name: 'Rename Term 1', exact: true })).toHaveAttribute('placeholder', 'Term 1');
  await table.getByRole('textbox', { name: 'Measure', exact: true }).click();
  const labels = () => inspector.locator('.value-field, [role="group"]').evaluateAll((fields) =>
    fields.map((field) => field.getAttribute('aria-label')).filter(Boolean));
  const order = await labels();
  expect(order.length).toBeGreaterThan(3);

  const check = table.getByRole('checkbox');
  const box = (await table.evaluate((element) => element.parentElement!.getBoundingClientRect().toJSON())) as DOMRect;
  const shown = (await check.boundingBox())!;
  expect(shown.x + shown.width).toBeLessThanOrEqual(box.x + box.width);
  await check.click();
  await expect.poll(async () => ((await terms())[0] as Term & { enabled?: boolean }).enabled).toBe(false);
  expect(await labels()).toEqual(order);
});

// Catches an inspector that keeps showing the previous step's selected term after another step is chosen on the map,
// here an Evaluate scored by that same Objective.
test("choosing another step shows that step, not the last step's selected term", async ({ page, request }) => {
  const { table, inspector } = await open(page, request);
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('menu').getByRole('menuitem', { name: 'Evaluate', exact: true }).click();
  await expect(mapStep(page, 'evaluate')).toBeVisible();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  await table.getByRole('textbox', { name: 'Measure', exact: true }).click();
  const term = inspector.getByRole('combobox', { name: 'Function', exact: true });
  await expect(term).toBeVisible();
  await mapStep(page, 'evaluate').click();
  await expect(page.getByRole('tab', { name: 'Objective', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(table.getByRole('row', { name: /^Term/ }).first()).toBeVisible();
  await expect(term).toHaveCount(0);
});
