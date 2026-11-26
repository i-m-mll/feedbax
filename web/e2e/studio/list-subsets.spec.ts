import { expect, test } from './fixtures/studio';
import { type Page } from '@playwright/test';
import type { StudioJobSnapshot, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

test('freezes one measured selection of a listed Evaluate subject and leaves the others', { tag: '@round' }, async ({ page, request, studio }) => {
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.list_subsets', 'seed', studio.home, 'subsets'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  const mutation = (method: string, route: string) => page.waitForResponse((response) =>
    response.request().method() === method && new URL(response.url()).pathname === route);

  async function plan() {
    const snapshot = await (await request.get(path)).json() as StudioProjectSnapshot;
    const response = await request.get(runPath, { params: { revision: snapshot.revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!;
  }
  async function run(step: string, count: number) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = mutation('POST', runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    expect((await admitted).status()).toBe(202);
    await expect(runs.getByText('Done', { exact: true }).or(runs.getByText('Failed', { exact: true })))
      .toHaveCount(count, { timeout: 60_000 });
    const failed = (await (await request.get('/api/jobs', { params: { project } })).json() as StudioJobSnapshot).jobs
      .filter(({ status }) => status === 'failed').map(({ kind, reason, log_tail }) => ({ kind, reason, log_tail }));
    expect(failed).toEqual([]);
  }

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded' });
  await run('train', 3);
  await run('train_b', 3);
  await run('baseline', 5);

  const before = await plan();
  const inputs = before.steps.find(({ step_id }) => step_id === 'baseline')!.inputs;
  expect(inputs.map(({ id, subset }) => [id, subset?.count, subset?.total])).toEqual([
    ['subject.0', 2, 3], ['subject.1', 2, 3], ['subject.2', 1, 3],
  ]);
  const requests = Object.fromEntries(before.evaluations!.filter(({ key }) => key.step === 'baseline')
    .map(({ key, request_id, bound_subject }) => [key.row, { request_id, rank: bound_subject?.key }]));

  // Each measured selection has To Rows on its rule field's menu; the named subset has no rule field.
  const selections = inspector.getByRole('list', { name: 'Subject', exact: true }).locator(':scope > li');
  await expect(selections).toHaveCount(3);
  const rule = (index: number) => selections.nth(index).getByRole('combobox', { name: 'Runs', exact: true });
  const toRows = page.getByRole('menuitem', { name: 'To Rows', exact: true });
  async function menu(index: number) {
    await rule(index).click({ button: 'right' });
    return toRows;
  }
  async function close() {
    await page.keyboard.press('Escape');
    await expect(toRows).toHaveCount(0);
  }
  await expect(await menu(0)).toBeEnabled();
  await close();
  await expect(await menu(1)).toBeEnabled();
  await close();
  await expect(rule(2)).toHaveCount(0);

  const preparing = mutation('POST', path + '/authoring/subset-to-rows');
  const saving = mutation('PATCH', path);
  await (await menu(0)).click();
  expect((await preparing).request().postDataJSON()).toMatchObject({ step: 'baseline', input: 'subject.0' });
  const saved = await saving;
  expect(saved.ok(), await saved.text()).toBe(true);
  const subject = (await saved.json() as StudioProjectSnapshot).project.steps!.baseline.inputs!.subject as Array<{
    step: string; where: Record<string, unknown>;
  }>;
  expect(subject.map(({ step, where }) => [step, Object.keys(where)])).toEqual([
    ['train', ['rule', 'bound_ranks']], ['train_b', ['rule']], ['train', ['subset']],
  ]);
  await expect(await menu(0)).toHaveCount(0);
  await close();
  await expect(await menu(1)).toBeEnabled();

  // The frozen selection's rows keep their IDs; the other selections' rows keep their requests.
  const first = new Set(inputs[0].subset!.ranks);
  const after = Object.fromEntries((await plan()).evaluations!.filter(({ key }) => key.step === 'baseline')
    .map(({ key, request_id }) => [key.row, request_id]));
  expect(Object.keys(after).sort()).toEqual(Object.keys(requests).sort());
  for (const [row, { request_id, rank }] of Object.entries(requests)) {
    expect(after[row] === request_id).toBe(!rank || !first.has(rank));
  }
});
