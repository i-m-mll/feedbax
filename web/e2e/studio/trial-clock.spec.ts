import type { Locator } from '@playwright/test';
import { expect, test } from './fixtures/studio';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const centre = async (locator: Locator) => {
  const box = (await locator.boundingBox())!;
  return box.x + box.width / 2;
};

// Catches a sample with no transport, a Play that moves the readout but not the shelf's playhead (or the reverse), a
// scrubber whose track is not the shelf's time axis, a shelf that does not open in the Task view, and a shelf that the
// Task view reopens after the user closed it there.
test('one clock plays every picked trial, its scrubber on the shelf\'s axis, and the shelf is the Task view\'s', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('delayed-reaches-files'));
  const footer = page.locator('footer.studio-window-bottom');
  const clock = footer.getByRole('group', { name: 'Clock', exact: true });
  const picker = clock.getByRole('button', { name: /^Trial / });
  const shelf = page.getByRole('region', { name: 'Shelf', exact: true });
  const shelfSwitch = clock.getByRole('button', { name: 'Shelf', exact: true });
  const readout = clock.locator('output');
  const tab = (name: string) => page.getByRole('tab', { name, exact: true });

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await tab('Task').click();
  await expect(shelf).toBeVisible();
  await expect(shelfSwitch).toHaveAttribute('aria-pressed', 'true');

  await picker.click();
  await page.getByRole('radio', { name: 'sample 2 of 4' }).click();
  await expect(picker).toHaveText('sample 2 of 4');
  await expect(readout).toHaveText(/^0\.00 s \/ \d+\.\d\d s$/);
  await clock.getByRole('button', { name: 'Play', exact: true }).click();
  await expect(readout).not.toHaveText(/^0\.00 s/);
  await clock.getByRole('button', { name: 'Pause', exact: true }).click();
  const [time, duration] = (await readout.textContent())!.match(/\d+\.\d\d/g)!.map(Number);
  const axis = shelf.getByRole('group', { name: 'Epochs', exact: true }).locator('.touch-none');
  const playhead = axis.locator('[data-shelf-playhead]');
  const span = (await axis.boundingBox())!;
  expect(((await centre(playhead)) - span.x) / span.width).toBeCloseTo(time / duration, 1);
  const thumb = clock.locator('[data-clock-track] [data-index]');
  expect(Math.abs(await centre(thumb) - await centre(playhead))).toBeLessThan(1.5);
  const track = (await clock.locator('[data-clock-track]').boundingBox())!;
  expect(Math.abs(track.x - span.x)).toBeLessThan(1);
  expect(Math.abs(track.width - span.width)).toBeLessThan(1);

  await shelfSwitch.click();
  await expect(shelf).toBeHidden();
  expect((await clock.locator('[data-clock-track]').boundingBox())!.x).toBeCloseTo(track.x, 0);
  await tab('Model').click();
  await expect(shelf).toBeHidden();
  await tab('Task').click();
  await expect(shelf).toBeHidden();
});
