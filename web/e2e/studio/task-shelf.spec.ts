import type { APIRequestContext } from '@playwright/test';
import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, TaskTrialSampleResponse } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

type Documents = NonNullable<StudioProjectSnapshot['project']['documents']>;

/** The fixture's Train batch size, which bounds the count the clock samples. */
const batchSize = 4;

/** The trials the sampling route draws from the project's task under `seed`: the runtime's own trials. */
async function routeTrials(request: APIRequestContext, documents: Documents, seed: number) {
  const payload = (name: string) => (documents[name] as unknown as { payload: unknown }).payload;
  const response = await request.post('/api/execution/task-trials/sample', {
    data: { model: payload('point-mass'), task: payload('delayed-reaches-task'), seed, count: batchSize },
  });
  expect(response.ok(), await response.text()).toBe(true);
  const { drawn } = await response.json() as TaskTrialSampleResponse;
  if (drawn.kind !== 'trials') throw new Error('The route answered an aggregate.');
  return drawn.trials;
}

// Catches a picker that does not start at the nominal trial, a nominal epochs row without the whisker of a drawn
// length, a shelf that does not shorten the slots, a sample drawn with other boundaries than the runtime's under the
// same seed, a signal bar before its first cell, a seed that does not reach the sampler or is lost on reload, a shelf
// row that does not select its signal, a count beyond the step's batch size, and a timeline left in the Task view. In
// All, catches a shelf that keeps one epochs row, an epoch drawn as present in every trial (or none) where only some
// trials are in it, and a Drawn column that shows one trial's value instead of the range.
test('the clock picks the nominal trial or a sample of a task document, and the shelf draws that trial', async ({ page, request }) => {
  test.setTimeout(60_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('delayed-reaches-files'));
  const { documents = {} } = ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project;
  const period = (documents['point-mass'] as unknown as { payload: { step_seconds: number } }).payload.step_seconds;
  const holdEnd = (trials: Awaited<ReturnType<typeof routeTrials>>, index: number) =>
    trials[index].timeline.find(({ epoch }) => epoch === 'hold')!.end * period;
  const seeded = await routeTrials(request, documents, 0);
  let seed = 1;
  let reseeded = await routeTrials(request, documents, seed);
  while (holdEnd(reseeded, 0) === holdEnd(seeded, 0)) reseeded = await routeTrials(request, documents, ++seed);

  const primary = page.locator('[data-slot="primary"]');
  const footer = page.locator('footer.studio-window-bottom');
  const clock = footer.getByRole('group', { name: 'Clock', exact: true });
  const picker = clock.getByRole('button', { name: /^Trial / });
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  const epochs = shelf.getByRole('group', { name: 'Epochs', exact: true });
  const hold = epochs.getByRole('button', { name: 'hold', exact: true });
  const open = async () => {
    await page.goto(`/?project=${snapshot.project_id}`);
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: 'Task', exact: true }).click();
  };
  const pick = async (entry: string) => {
    await picker.click();
    await page.getByRole('radio', { name: entry }).click();
    await expect(picker).toHaveText(entry);
  };

  await open();
  await expect(picker).toHaveText('nominal');
  await expect(primary.getByRole('table', { name: 'Variables', exact: true })).toBeVisible();
  await expect(primary.getByRole('group', { name: 'Timeline' })).toHaveCount(0);
  const slot = async () => (await primary.boundingBox())!.height;
  const shelfSwitch = clock.getByRole('button', { name: 'Shelf', exact: true });
  await expect(hold).toBeVisible();
  const short = await slot();
  await shelfSwitch.click();
  await expect(hold).toBeHidden();
  expect(await slot()).toBeGreaterThan(short);
  await shelfSwitch.click();
  await expect(hold).toBeVisible();
  const whisker = epochs.getByRole('img', { name: /^hold / });
  await expect(whisker).toBeVisible();
  expect(Number(await whisker.getAttribute('data-least'))).toBeCloseTo(5 * period, 9);
  expect(Number(await whisker.getAttribute('data-greatest'))).toBeCloseTo(14 * period, 9);

  await pick('sample 2 of 4');
  await expect(hold).toHaveAttribute('data-end', String(holdEnd(seeded, 1)));
  await expect(epochs.getByRole('img', { name: /^hold / })).toHaveCount(0);
  const target = shelf.getByRole('group', { name: 'target_pos', exact: true });
  await expect(target.locator('[data-start]').first()).toHaveAttribute('data-start', String(holdEnd(seeded, 1)));
  const drawn = seeded[1].variables.catch as boolean;
  const variables = primary.getByRole('table', { name: 'Variables', exact: true });
  await expect(variables.getByRole('columnheader', { name: 'Drawn', exact: true })).toBeVisible();
  await expect(variables.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Rename catch', exact: true }) })
    .getByRole('cell').nth(-2)).toHaveText(String(drawn));

  await target.getByRole('button', { name: 'target_pos', exact: true }).click();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await expect(drawer.getByRole('tree', { name: 'Signals', exact: true }).getByRole('treeitem', { name: 'target_pos', exact: true })).toHaveAttribute('aria-selected', 'true');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('img', { name: 'Plot target_pos', exact: true })).toBeVisible();

  await pick('sample 1 of 4');
  await expect(hold).toHaveAttribute('data-end', String(holdEnd(seeded, 0)));
  await picker.click();
  const seedField = page.getByRole('textbox', { name: 'Seed', exact: true });
  await seedField.fill(String(seed));
  await seedField.press('Enter');
  await page.keyboard.press('Escape');
  await expect(hold).toHaveAttribute('data-end', String(holdEnd(reseeded, 0)));

  await open();
  await expect(picker).toHaveText('sample 1 of 4');
  await expect(hold).toHaveAttribute('data-end', String(holdEnd(reseeded, 0)));

  await expect(clock.getByRole('button', { name: 'Mean', exact: true })).toBeDisabled();
  await clock.getByRole('button', { name: 'All', exact: true }).click();
  await expect(picker).toHaveText(`${batchSize} trials`);
  await expect(epochs).toHaveCount(0);
  for (const name of ['hold', 'cue', 'move']) await expect(shelf.getByRole('group', { name, exact: true })).toHaveCount(1);
  const ends = reseeded.map((_, index) => holdEnd(reseeded, index));
  const bands = shelf.getByRole('group', { name: 'hold', exact: true });
  await expect(bands.locator('[data-share="all"]')).toHaveAttribute('data-end', String(Math.min(...ends)));
  await expect(bands.locator('[data-share="some"]')).toHaveAttribute('data-end', String(Math.max(...ends)));
  const catches = [...new Set(reseeded.map(({ variables }) => variables.catch as boolean))].sort();
  await expect(variables.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Rename catch', exact: true }) })
    .getByRole('cell').nth(-2)).toHaveText(catches.map(String).join(', '));
});
