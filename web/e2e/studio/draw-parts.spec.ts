import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const literal = (value: number) => ({ kind: 'literal', value });
const uniform = (low: number, high: number) => ({ kind: 'draw', family: 'uniform', args: [literal(low), literal(high)] });

/** The ordinary Train project with its Objective's first term weighted by `weight`, open on the Objective view. */
async function open(page: Page, request: APIRequestContext, weight: unknown) {
  const files = await fixture('ordinary-files', '2') as { kind: string; id: string; document?: { payload: { terms: object[] } } }[];
  const objective = files.find(({ id }) => id === 'simple-reaches-objective')!.document!;
  objective.payload.terms[0] = { ...objective.payload.terms[0], weight };
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const writes: unknown[] = [];
  page.on('request', (sent) => {
    if (sent.method() === 'POST' && sent.url().endsWith('/authoring/trials') && sent.postDataJSON()?.change) writes.push(sent.postDataJSON().change);
  });
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const weightOf = async () => {
    const response = await request.get(projectPath);
    const saved = await response.json() as StudioProjectSnapshot;
    return (rootPayload(saved.project.documents?.['simple-reaches-objective']) as { terms: { weight: unknown }[] }).terms[0].weight;
  };
  const table = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Objective Terms', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  return { table, inspector, writes, weightOf };
}

// Catches parts that do not round-trip to the text: a part read from the wrong argument, an edit that writes a draw
// other than the one shown, and a family change that writes a draw before its parameters are filled.
test('a draw in the Inspector shows its parts, and an edit of a part writes its text', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { inspector, writes, weightOf } = await open(page, request, uniform(0.3, 1));
  await page.getByRole('tree', { name: 'Objective Terms', exact: true }).getByRole('treeitem').first().click();
  await expect(inspector.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('U(0.3, 1)');
  await expect(inspector.getByRole('combobox', { name: 'Family', exact: true })).toHaveText('Uniform');
  await expect(inspector.getByRole('textbox', { name: 'Low', exact: true })).toHaveValue('0.3');
  await expect(inspector.getByRole('textbox', { name: 'High', exact: true })).toHaveValue('1');
  await expect(inspector.getByRole('img', { name: 'Weight density', exact: true })).toBeVisible();

  await enter(inspector.getByRole('textbox', { name: 'High', exact: true }), '2');
  await expect.poll(weightOf).toMatchObject(uniform(0.3, 2));
  await expect(inspector.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('U(0.3, 2)');

  const written = writes.length;
  await inspector.getByRole('combobox', { name: 'Family', exact: true }).click();
  await page.getByRole('option', { name: 'Normal', exact: true }).click();
  await expect(inspector.getByRole('textbox', { name: 'Mean', exact: true })).toHaveValue('');
  await expect(inspector.getByRole('textbox', { name: 'SD', exact: true })).toHaveValue('');
  await enter(inspector.getByRole('textbox', { name: 'Mean', exact: true }), '1');
  await expect(inspector.getByRole('textbox', { name: 'Mean', exact: true })).toHaveValue('1');
  expect(writes).toHaveLength(written);
  expect(await weightOf()).toMatchObject(uniform(0.3, 2));
  await enter(inspector.getByRole('textbox', { name: 'SD', exact: true }), '0.2');
  await expect.poll(weightOf).toMatchObject({ kind: 'draw', family: 'normal', args: [literal(1), literal(0.2)] });
  await expect(inspector.getByRole('textbox', { name: 'Weight', exact: true })).toHaveValue('N(1, 0.2)');
});

// Catches a table cell whose parts are not shown where the owner chose (the Inspector), a part edit that writes
// elsewhere than the cell's value, and a density strip drawn in the narrow cell.
test('selecting a Weight cell that holds a draw shows its parts in the Inspector, and an edit there writes the cell', async ({ page, request }) => {
  test.setTimeout(45_000);
  const { table, inspector, weightOf } = await open(page, request, uniform(0.5, 2));
  const cell = table.getByRole('textbox', { name: 'Weight', exact: true });
  await expect(cell).toHaveValue('U(0.5, 2)');
  await cell.click();
  await expect(inspector.getByRole('textbox', { name: 'Low', exact: true })).toHaveValue('0.5');
  await expect(inspector.getByRole('textbox', { name: 'High', exact: true })).toHaveValue('2');
  await expect(table.getByRole('button', { name: 'Weight kind: Draw', exact: true })).toBeVisible();
  await expect(table.getByRole('img', { name: 'Weight density', exact: true })).toHaveCount(0);

  await enter(inspector.getByRole('textbox', { name: 'Low', exact: true }), '1');
  await expect.poll(weightOf).toMatchObject(uniform(1, 2));
  await expect(cell).toHaveValue('U(1, 2)');
});
