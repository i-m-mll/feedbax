import { expect, test } from './fixtures/studio';
import type { StudioProjectList } from '../../src/generated/studioContracts';
import { runProjectFixture } from './fixtures/canonicalTrain';

test('the start page stays until a project is opened from its row, and makes none of its own', async ({ page, request, studio }) => {
  const project: string = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'listed'));
  const listed = async () => (await (await request.get('/api/projects')).json() as StudioProjectList).projects;
  const before = await listed();
  const { name } = before.find(({ project_id }) => project_id === project)!;
  await page.goto('/');
  const start = page.getByRole('dialog', { name: 'Studio', exact: true });
  const table = start.getByRole('table', { name: 'Projects', exact: true });
  await expect(table.getByRole('link', { name, exact: true })).toBeVisible();
  await page.keyboard.press('Escape');
  await page.mouse.click(4, 4);
  await expect(start).toBeVisible();
  expect((await listed()).length).toBe(before.length);
  await table.getByRole('link', { name, exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`[?&]project=${project}\\b`));
  await expect(page.getByRole('tree', { name: 'Steps', exact: true })).toBeVisible();
});

test('an example makes a named copy that opens, and Delete removes a filtered row', async ({ page, request }) => {
  const listed = async () => (await (await request.get('/api/projects')).json() as StudioProjectList).projects;
  await page.goto('/');
  const start = page.getByRole('dialog', { name: 'Studio', exact: true });
  await start.getByRole('button', { name: 'Examples', exact: true }).click();
  await page.getByRole('menu', { name: 'Examples' }).getByRole('menuitem').first().click();
  const naming = page.getByRole('dialog', { name: 'New Project', exact: true });
  await naming.getByRole('textbox', { name: 'Name' }).fill('reach_copy');
  await naming.getByRole('button', { name: 'New Project', exact: true }).click();
  await expect(page.getByRole('tree', { name: 'Steps', exact: true })).toBeVisible();
  const copy = (await listed()).find(({ name }) => name === 'reach_copy')!;
  expect(page.url()).toContain(`project=${copy.project_id}`);
  expect(copy.step_count).toBeGreaterThan(0);

  await page.goto('/');
  const table = start.getByRole('table', { name: 'Projects', exact: true });
  await start.getByRole('textbox', { name: 'Filter' }).fill('REACH_C');
  await expect(table.getByRole('link')).toHaveText(['reach_copy']);
  const row = table.getByRole('row', { name: /reach_copy/ });
  await row.hover();
  await row.getByRole('button', { name: 'Project reach_copy', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Delete' }).click();
  await page.getByRole('dialog', { name: 'Delete reach_copy?' }).getByRole('button', { name: 'Delete', exact: true }).click();
  await expect(table.getByRole('link')).toHaveCount(0);
  expect((await listed()).some(({ name }) => name === 'reach_copy')).toBe(false);
});
