import { expect, test } from './fixtures/studio';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

/** The fixture's Train batch size, which bounds the count the clock samples. */
const batchSize = 4;
const hold = 0.05;
const limit = 0.24;

// Catches an Until that the kind menu does not offer, a draft that leaves the caret after `≤` (the condition typed
// into the limit), an Until End shown without its Condition and Limit parts, a recorded epoch whose end is not marked
// with how it ended, a condition's ending marked as the limit's (or the reverse), and an ending at another step than
// the condition held for its duration or the limit reached.
test('an End made Until in the Task view ends each trial at its condition or its limit, and the shelf says which', { tag: '@slow' }, async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('delayed-reaches-files'));
  const primary = page.locator('[data-slot="primary"]');
  const clock = page.locator('footer.studio-window-bottom').getByRole('group', { name: 'Clock', exact: true });
  const picker = clock.getByRole('button', { name: /^Trial / });
  const epochs = page.getByRole('region', { name: 'Shelf', exact: true }).getByRole('group', { name: 'Epochs', exact: true });
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();

  const cue = primary.getByRole('group', { name: 'End field', exact: true }).nth(1);
  await cue.getByRole('button', { name: /^End kind: / }).click();
  await page.getByRole('menuitemradio', { name: 'Until', exact: true }).click();
  const end = cue.getByRole('textbox', { name: 'End', exact: true });
  await expect(end).toBeFocused();
  await page.keyboard.type(`held(catch, ${hold} s)`);
  for (const _ of ' ≤ ') await page.keyboard.press('ArrowRight');
  await page.keyboard.type(`${limit} s`);
  await page.keyboard.press('Enter');
  await expect(end).toHaveValue(`held(catch, ${hold} s) ≤ ${limit} s`);
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('textbox', { name: 'Condition', exact: true })).toHaveValue(`held(catch, ${hold} s)`);
  await expect(inspector.getByRole('textbox', { name: 'Limit', exact: true })).toHaveValue(`${limit} s`);

  const caught = primary.getByRole('table', { name: 'Variables', exact: true }).getByRole('row')
    .filter({ has: page.getByRole('textbox', { name: 'Rename catch', exact: true }) }).getByRole('cell').nth(-2);
  const seen = new Set<string>();
  for (let index = 1; index <= batchSize; index += 1) {
    await picker.click();
    await page.getByRole('radio', { name: `sample ${index} of ${batchSize}` }).click();
    await expect(picker).toHaveText(`sample ${index} of ${batchSize}`);
    const condition = await caught.textContent() === 'true';
    const ended = epochs.getByRole('button', { name: condition ? 'cue, Condition' : 'cue, Limit', exact: true });
    await expect(ended).toHaveAttribute('data-ended', condition ? 'condition' : 'limit');
    const [start, stop] = await Promise.all([ended.getAttribute('data-start'), ended.getAttribute('data-end')]);
    expect(Number(stop) - Number(start)).toBeCloseTo(condition ? hold : limit, 9);
    seen.add(String(condition));
  }
  expect(seen.size, 'the samples should include a trial of each ending').toBe(2);
});
