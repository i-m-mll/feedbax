import { expect, test } from './fixtures/studio';
import { setDocumentSchemaId, type StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type File = { kind: string; id: string; step?: Record<string, unknown>; document?: { payload?: { terms?: Record<string, object> } } };

/** The ordinary project, with its Objective's one term a family named Effort over `k` from 0 to 5. */
async function termFamily() {
  const files: File[] = await fixture('ordinary-files', '2');
  const objective = files.find(({ id }) => id === 'simple-reaches-objective')!.document!.payload!;
  const [[key, term]] = Object.entries(objective.terms!);
  objective.terms = { [key]: { ...term, label: 'Effort', over: { k: '0..5' }, weight: 'k + 1' } };
  return files;
}

// Catches a family listed as the terms it generates or without its count, an index rule missing from the Inspector, a
// count that does not follow the rule, an edit of the rule or of the family's Weight that is not stored on the one
// member, and a reopened project that shows any of them differently.
test('an Objective term family is one term whose index rule and Weight are edited on it', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await termFamily());
  const table = page.locator('[data-slot="primary"]').getByRole('tabpanel').getByRole('table', { name: 'Objective Terms', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const weight = table.getByRole('row').nth(1).getByRole('textbox', { name: 'Weight', exact: true });
  const index = inspector.getByRole('region', { name: 'Over', exact: true }).getByRole('textbox', { name: 'k', exact: true });
  const open = async () => {
    await page.goto(`/?project=${snapshot.project_id}`);
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: 'Objective', exact: true }).click();
    await weight.click();
  };

  await open();
  const term = table.getByRole('row').nth(1).getByRole('rowheader');
  await expect(table.getByRole('row')).toHaveCount(2);
  await expect(term.getByRole('textbox', { name: /^Rename / })).toHaveValue('Effort');
  await expect(term).toContainText('× 6');
  await expect(weight).toHaveValue('k + 1');
  await expect(index).toHaveValue('0..5');

  await enter(index, '0..7');
  await expect(index).toHaveValue('0..7');
  await expect(term).toContainText('× 8');
  await enter(weight, '2 * k');
  await expect(weight).toHaveValue('2 * k');
  await expect(table.getByRole('row')).toHaveCount(2);

  await open();
  await expect(table.getByRole('row')).toHaveCount(2);
  await expect(term).toContainText('× 8');
  await expect(weight).toHaveValue('2 * k');
  await expect(index).toHaveValue('0..7');
});

/** The delayed-reaches files with a Set of a Light entry and a family of three loads, which the Train sweeps. */
async function entryFamily() {
  const all = await fixture('delayed-reaches-files') as File[];
  all.find(({ id }) => id === 'train')!.step!.sweep = { axes: { load: { set: 'loads' } } };
  return [...all, { kind: 'document', id: 'loads', document: {
    schema: setDocumentSchemaId, label: 'Loads',
    payload: { entries: { light: { label: 'Light' }, loaded: { label: 'Load {k}', over: { k: '1..3' } } } },
  } }];
}

// Catches a Set family listed as its generated entries or without its count, and a weight typed on the family that
// is not stored on the family's own key (where it gives the weight to each generated entry).
test('a Set entry family is one entry with its count, weighed as one', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await entryFamily());
  const card = page.locator('[data-map-step="train"]');
  const row = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Variables', exact: true }).getByRole('row')
    .filter({ has: page.getByRole('list', { name: 'Values Loads', exact: true }) });
  const step = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps!.train;

  await page.goto(`/?project=${snapshot.project_id}`);
  await sweepLabel(card, 'Loads').click();
  await page.getByRole('menuitem', { name: 'To Field', exact: true }).click();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const entries = row.getByRole('listitem');
  await expect(entries).toHaveCount(3);
  await expect(entries.nth(0)).not.toContainText('×');
  await expect(entries.nth(1)).toHaveText(/^Load \{k\}\s*× 3/);

  const weight = entries.nth(1).getByRole('textbox', { name: /^Weight / });
  await enter(weight, '3');
  await expect.poll(async () => (await step()).set_fields?.load?.sampling).toEqual({ kind: 'weighted', weights: { loaded: 3 } });
  await expect(weight).toHaveValue('3');
});
