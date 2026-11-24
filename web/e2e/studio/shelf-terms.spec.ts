import { expect, test } from './fixtures/studio';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const target = { kind: 'call', function: 'sqnorm', args: [{ kind: 'name', reference: 'target_pos' }] };
const squared = { base: 'real', dimensions: { length: 2 } };

// Catches a term's bar that disagrees with where training applies it (drawn over its whole window, ignoring that the
// target is absent before the cue), a term that applies nowhere losing its row, and a term row whose selection does not
// reach the Objective view or the Inspector.
test('each term that is on has a row on the shelf, barred where it applies, and selects the term', async ({ page, request }) => {
  const files = await fixture('delayed-reaches-files') as { document?: { payload: Record<string, any> } }[];
  for (const file of files) {
    const payload = file.document?.payload;
    if (payload?.terms) payload.terms.push(
      { id: 'reach', name: 'reach', measure: target, measure_type: squared, epochs: ['move', 'hold'] },
      { id: 'held', name: 'held', measure: target, measure_type: squared, epochs: ['hold'] },
      { id: 'off', name: 'off', measure: target, measure_type: squared, enabled: false },
    );
  }
  const { snapshot } = await seedManagedProject(request, files);
  const clock = page.locator('footer.studio-window-bottom').getByRole('group', { name: 'Clock', exact: true });
  const picker = clock.getByRole('button', { name: /^Trial / });
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  const row = (name: string) => shelf.getByRole('group', { name, exact: true });
  const starts = async (name: string) => (await row(name).locator('[data-start]').evaluateAll((marks) =>
    marks.map((mark) => Number(mark.getAttribute('data-start')))));

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await picker.click();
  await page.getByRole('radio', { name: 'nominal', exact: true }).click();
  await expect(row('reach').locator('[data-start]')).toHaveCount(1);

  const move = Number(await row('Epochs').getByRole('button', { name: 'move', exact: true }).getAttribute('data-start'));
  const [present] = await starts('target_pos');
  const [reach] = await starts('reach');
  expect(reach).toBeCloseTo(Math.max(move, present), 6);
  await expect(row('held')).toBeVisible();
  await expect(row('held').locator('[data-start]')).toHaveCount(0);
  await expect(row('off')).toHaveCount(0);

  await row('reach').getByRole('button', { name: 'reach', exact: true }).click();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('region', { name: 'reach', exact: true })).toBeVisible();
  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const terms = page.locator('[data-slot="primary"]').getByRole('table', { name: 'Objective Terms', exact: true });
  await expect(terms.locator('tbody tr[aria-selected="true"]')).toHaveCount(1);
  await expect(terms.locator('tbody tr[aria-selected="true"] th, tbody tr[aria-selected="true"] td').first()).toHaveText('reach');
});
