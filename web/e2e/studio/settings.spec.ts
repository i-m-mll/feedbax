import type { APIRequestContext, Page } from '@playwright/test';
import { expect, test } from './fixtures/studio';
import { seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

/** Opens Settings in a loaded project's window; the window drawn while the project loads is replaced. */
async function openSettings(page: Page, request: APIRequestContext) {
  const { snapshot, configuration } = await seedManagedProject(request, []);
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  await expect(page.getByRole('heading', { level: 1, name: configuration.project })).toBeVisible();
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  return page.getByRole('region', { name: 'Settings', exact: true });
}

/**
 * Catches a specimen drawn apart from the components it samples, or one that reads the setting only when Settings
 * opens: its field row must grow with Density while Settings stays open.
 */
test('changing Density in Settings changes the specimen row height at once', async ({ page, request }) => {
  const settings = await openSettings(page, request);
  const density = settings.getByRole('group', { name: 'Density', exact: true });
  // The specimen is inert, so it is outside the accessibility tree.
  const field = settings.locator('[data-settings-specimen] input').first();
  await density.getByRole('button', { name: 'Compact', exact: true }).click();
  const compact = (await field.boundingBox())!.height;
  await density.getByRole('button', { name: 'Regular', exact: true }).click();
  await expect.poll(async () => (await field.boundingBox())!.height).toBeGreaterThan(compact);
  await expect(settings).toBeVisible();
});

/** Catches a search that leaves unmatched rows, or their emptied sections, on the page. */
test('the search field hides the rows that do not match', async ({ page, request }) => {
  const settings = await openSettings(page, request);
  await settings.getByRole('searchbox', { name: 'Search Settings' }).fill('density');
  await expect(settings.getByRole('group', { name: 'Density', exact: true })).toBeVisible();
  await expect(settings.getByRole('group', { name: 'Theme', exact: true })).toBeHidden();
  await expect(settings.getByRole('spinbutton', { name: 'Removal Threshold' })).toBeHidden();
  await expect(settings.getByRole('heading', { name: 'Shortcuts' })).toBeHidden();
});
