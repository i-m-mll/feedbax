import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import { runProjectFixture } from './fixtures/canonicalTrain';
import { chooseStepKind } from './fixtures/studioKeyboard';
import { hold } from './fixtures/heldRequest';

test.use({ viewport: { width: 1440, height: 900 } });

const steps = (page: Page) => page.getByRole('tree', { name: 'Steps', exact: true });
const stepRow = (page: Page, name: RegExp) => steps(page).getByRole('treeitem', { name });

test('the selected step is still selected after a reload', async ({ page, studio }) => {
  const project: string = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'reload'));
  await page.goto('/?project=' + encodeURIComponent(project));
  await expect(stepRow(page, /^baseline\b/)).toHaveAttribute('aria-selected', 'true');
  await stepRow(page, /^train\b/).click();
  await expect(stepRow(page, /^train\b/)).toHaveAttribute('aria-selected', 'true');
  await page.reload();
  await expect(stepRow(page, /^train\b/)).toHaveAttribute('aria-selected', 'true');
  await expect(stepRow(page, /^baseline\b/)).not.toHaveAttribute('aria-selected', 'true');
});

test('a new Analyze step is listed in the drawer as soon as it is made', async ({ page, request, studio }) => {
  const project: string = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'seed', studio.home, 'new-analyze'));
  await page.goto('/?project=' + encodeURIComponent(project));
  await stepRow(page, /^baseline\b/).click();
  await expect(stepRow(page, /^Analyze\b/)).toHaveCount(0);
  const { revision } = await (await request.get(`/api/projects/${project}`)).json() as { revision: string };
  // The new revision's plan is held: a slow plan read must not keep the new step out of the drawer.
  const plan = await hold(page, `/api/projects/${project}/train-runs`, (_body, sent) =>
    sent.method() === 'GET' && new URL(sent.url()).searchParams.get('revision') !== revision);
  await page.getByRole('complementary', { name: 'Map Drawer', exact: true }).getByRole('button', { name: 'New Step', exact: true }).click();
  await chooseStepKind(page, 'Analyze');
  await plan.sent;
  await expect(stepRow(page, /^Analyze\b/)).toHaveAttribute('aria-selected', 'true');
  plan.release();
});
