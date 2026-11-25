import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject, sweepLabel } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type File = { kind: string; id: string; step?: Record<string, unknown> };

/** The delayed-reaches files with a Set of three entries, which the Train sweeps. */
async function files() {
  const all = await fixture('delayed-reaches-files') as File[];
  all.find(({ id }) => id === 'train')!.step!.sweep = { axes: { load: { set: 'loads' } } };
  return [...all, { kind: 'document', id: 'loads', document: {
    schema: 'feedbax.set_document.v1', label: 'Loads',
    payload: { entries: { light: { label: 'Light' }, mid: { label: 'Mid' }, heavy: { label: 'Heavy' } } },
  } }];
}

// Catches a To Field that leaves the axis's runs on the card or shows no variable for the Set in the Task view, a
// Value that drops an entry or None, a weight typed at its entry that is not stored at that entry, and a To Sweep that
// does not restore the axis's runs.
test('To Field moves a Set into the run as a variable of the Task, its weights are edited there, and To Sweep moves it back', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await files());
  const card = page.locator('[data-map-step="train"]');
  const marks = card.locator('.studio-map-run-mark');
  const primary = page.locator('[data-slot="primary"]');
  const row = primary.getByRole('table', { name: 'Variables', exact: true }).getByRole('row')
    .filter({ has: page.getByRole('list', { name: 'Values loads', exact: true }) });
  const step = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps!.train;

  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(marks).toHaveCount(3);
  await sweepLabel(card, 'loads').click();
  await page.getByRole('menuitem', { name: 'To Field', exact: true }).click();
  await expect(marks).toHaveCount(1);
  await expect.poll(async () => Object.keys((await step()).set_fields ?? {})).toEqual(['load']);

  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(row.getByRole('listitem')).toHaveText(['Light', 'Mid', 'Heavy', 'None']);
  const weights = () => row.getByRole('textbox', { name: /^Weight / }).evaluateAll((fields) => fields.map((field) => (field as HTMLInputElement).value));
  await expect.poll(weights).toEqual(['1', '1', '1', '1']);

  await row.getByRole('textbox', { name: 'Weight Heavy', exact: true }).fill('3');
  await row.getByRole('textbox', { name: 'Weight Heavy', exact: true }).press('Enter');
  await row.getByRole('textbox', { name: 'Weight None', exact: true }).fill('0');
  await row.getByRole('textbox', { name: 'Weight None', exact: true }).press('Enter');
  await expect.poll(async () => (await step()).set_fields?.load?.sampling)
    .toEqual({ kind: 'weighted', weights: { heavy: 3 }, none: 0 });
  await expect.poll(weights).toEqual(['1', '1', '3', '0']);

  await row.hover();
  await row.getByRole('button', { name: 'Variable loads', exact: true }).click();
  await page.getByRole('menuitem', { name: 'To Sweep', exact: true }).click();
  await expect(marks).toHaveCount(3);
  await expect(row).toHaveCount(0);
  expect((await step()).sweep?.axes?.load).toEqual({ set: 'loads' });
});

// Catches a Set placed in the trials with no field for its condition, a condition field that does not complete from
// the Task's variables, a condition that is not stored on the Set field, an unknown name saved without its Problem on
// the field, and a cleared field that leaves the condition stored.
test('a Set in the trials takes a condition on Task variables, refuses an unknown name on its field, and clears', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await files());
  const card = page.locator('[data-map-step="train"]');
  const condition = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Variables', exact: true })
    .getByLabel('Condition', { exact: true });
  const stored = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot)
    .project.steps!.train.set_fields?.load?.condition;
  const write = async (text: string) => {
    await condition.fill(text);
    await condition.press('Enter');
  };

  await page.goto(`/?project=${snapshot.project_id}`);
  await sweepLabel(card, 'loads').click();
  await page.getByRole('menuitem', { name: 'To Field', exact: true }).click();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(condition).toHaveValue('');

  await condition.fill('not ');
  await condition.pressSequentially('ca');
  await expect(page.getByRole('listbox', { name: 'Completions' }).getByRole('option')).toContainText(['catch']);
  await write('not catch');
  await expect.poll(stored).toMatchObject({ kind: 'call', function: 'not', args: [{ kind: 'name', reference: 'catch' }] });
  await expect(condition).not.toHaveAttribute('aria-invalid', 'true');

  await write('not cach');
  await expect(condition).toHaveAttribute('aria-invalid', 'true');
  await expect(condition).toHaveValue('not cach');

  await write('');
  await expect.poll(stored).toBeUndefined();
  await expect(condition).not.toHaveAttribute('aria-invalid', 'true');
});
