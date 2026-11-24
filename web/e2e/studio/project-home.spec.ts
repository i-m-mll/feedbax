import { expect, test } from './fixtures/studio';
import type { StudioProjectList } from '../../src/generated/studioContracts';
import { runProjectFixture } from './fixtures/canonicalTrain';

test('the root URL lists the home\'s projects, opens one from its row, and makes none of its own', async ({ page, request, studio }) => {
  const project: string = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'listed'));
  const listed = async () => (await (await request.get('/api/projects')).json() as StudioProjectList).projects;
  const before = await listed();
  const { name } = before.find(({ project_id }) => project_id === project)!;
  await page.goto('/');
  const table = page.getByRole('table', { name: 'Projects', exact: true });
  await expect(table.getByRole('link', { name, exact: true })).toBeVisible();
  expect((await listed()).length).toBe(before.length);
  await table.getByRole('link', { name, exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`[?&]project=${project}\\b`));
  await expect(page.getByRole('tree', { name: 'Steps', exact: true })).toBeVisible();
});
