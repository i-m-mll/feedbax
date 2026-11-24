import { expect, test } from '@playwright/test';

test.use({ trace: 'off' });

/** Catches a Task view part or the shelf throwing or logging an error on either fixture's shape (a repeat, an Until End, a batch). */
test('draws the Task view mockup and the shelf for both fixtures without a console error', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/gallery.html#task');

  for (const name of ['Delayed reaches', 'Three reaches']) {
    const states = page.getByRole('region', { name, exact: true });
    await expect(states.getByRole('region', { name: 'Shelf' })).toHaveCount(2);
    await expect(states.getByRole('img', { name: /^Plot / })).toHaveCount(2);
  }
  const window = page.locator('.studio-project-window');
  await window.getByRole('button', { name: 'Task Delayed reaches' }).click();
  await page.getByRole('menuitemradio', { name: 'Three reaches' }).click();
  await expect(window.getByRole('table', { name: 'Epochs' }).getByRole('img', { name: 'Absent' })).toHaveCount(2);
  expect(errors).toEqual([]);
});
