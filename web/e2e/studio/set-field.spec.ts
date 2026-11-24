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
// Value that drops an entry or None, and a To Sweep that does not restore the axis's runs.
test('To Field moves a Set into the run as a variable of the Task, and To Sweep moves it back', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await files());
  const card = page.locator('[data-map-step="train"]');
  const marks = card.locator('.studio-map-run-mark');
  const primary = page.locator('[data-slot="primary"]');
  const row = primary.getByRole('table', { name: 'Variables', exact: true }).getByRole('row')
    .filter({ has: page.getByRole('list', { name: 'Values load', exact: true }) });
  const step = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps!.train;

  await page.goto(`/?project=${snapshot.project_id}`);
  await expect(marks).toHaveCount(3);
  await sweepLabel(card, 'load').click();
  await page.getByRole('menuitem', { name: 'To Field', exact: true }).click();
  await expect(marks).toHaveCount(1);
  await expect.poll(async () => Object.keys((await step()).set_fields ?? {})).toEqual(['load']);

  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await expect(row.getByRole('listitem')).toHaveText(['Light1', 'Mid1', 'Heavy1', 'None1']);

  await row.hover();
  await row.getByRole('button', { name: 'Variable load', exact: true }).click();
  await page.getByRole('menuitem', { name: 'To Sweep', exact: true }).click();
  await expect(marks).toHaveCount(3);
  await expect(row).toHaveCount(0);
  expect((await step()).sweep?.axes?.load).toEqual({ set: 'loads' });
});
