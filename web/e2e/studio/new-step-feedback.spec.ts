import { expect, test } from './fixtures/studio';
import type { Route } from '@playwright/test';
import { seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

// Catches: a chosen step that appears only once the Service has saved it and the plan is read, so the click seems lost.
test('a chosen step is drawn on the map before the Service has saved it', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  await page.goto(`/?project=${snapshot.project_id}`);
  const held: Route[] = [];
  const files = (url: URL) => url.pathname === `${projectPath}/files`;
  await page.route(files, (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    held.push(route);
  });
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true }).click();
  const adding = page.locator('[data-map-adding]');
  await expect(adding).toBeVisible();
  await expect.poll(() => held.length).toBeGreaterThan(0);

  await page.unroute(files);
  for (const route of held) await route.fallback();
  await expect(page.getByRole('article', { name: 'Train train', exact: true })).toBeVisible();
  await expect(adding).toHaveCount(0);
});
