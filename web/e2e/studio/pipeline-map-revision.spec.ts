import { expect, test } from './fixtures/studio';
import type { Route } from '@playwright/test';
import { runProjectFixture } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

test('while a new revision\'s plan is read, the map draws the previous plan with its own project', async ({ page, studio }) => {
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.set_authoring', 'seed', studio.home, 'stale-plan'));
  const path = '/api/projects/' + project;
  const marks = page.locator('[data-map-step="pair"] .studio-map-run-mark');
  const labels = () => marks.evaluateAll((items) => items.map((item) => item.getAttribute('aria-label')));
  const saved = () => expect(page.getByText('Edited', { exact: true })).toBeHidden();

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded' });
  await expect(marks).toHaveCount(2);
  const original = await labels();
  await marks.first().click();
  await marks.first().click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Duplicate Run', exact: true }).click();
  await expect(marks).toHaveCount(3);
  await saved();
  const duplicated = await labels();

  const held: Route[] = [];
  let seen: () => void;
  const planRequested = new Promise<void>((resolve) => { seen = resolve; });
  const plan = (url: URL) => url.pathname === path + '/train-runs' && url.searchParams.has('window_map');
  await page.route(plan, (route) => {
    if (route.request().method() !== 'GET') return route.fallback();
    held.push(route);
    seen();
  });
  const undone = page.waitForResponse((response) =>
    response.request().method() === 'PATCH' && new URL(response.url()).pathname === path);
  await page.keyboard.press('ControlOrMeta+z');
  expect((await undone).ok()).toBe(true);
  await planRequested;
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  expect(await labels()).toEqual(duplicated);

  await page.unroute(plan);
  for (const route of held) await route.fallback();
  await expect.poll(labels).toEqual(original);
});
