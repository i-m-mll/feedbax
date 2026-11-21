import { expect, test } from './fixtures/studio';
import { type Locator, type Page } from '@playwright/test';
import type { StudioJobSnapshot, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { chooseSweepValue, enter, mapStep, runProjectFixture, sweepLabel } from './fixtures/canonicalTrain';
import { outlineRow, placeFromLibrary } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

/** Opens a mark's context menu and chooses a command from it. */
async function fromMenu(target: Locator, command: string) {
  await target.click({ button: 'right' });
  await target.page().getByRole('menuitem', { name: command, exact: true }).click();
}

test('authors a set across runs: Duplicate Run, Save Set, a set as a sweep, Inline Set', { tag: '@round' }, async ({ page, request, studio }) => {
  test.setTimeout(240_000);
  const project = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.set_authoring', 'seed', studio.home, 'sets'));
  const path = '/api/projects/' + project;
  const runPath = path + '/train-runs';
  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const model = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const card = (step: string) => page.locator(`[data-map-step="${step}"]`);
  const marks = (step: string) => card(step).locator('.studio-map-run-mark');
  const mutation = (method: string, route: string) => page.waitForResponse((response) =>
    response.request().method() === method && new URL(response.url()).pathname === route);

  async function snapshot() {
    const response = await request.get(path);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function plan() {
    const response = await request.get(runPath, { params: { revision: (await snapshot()).revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!;
  }
  async function evaluations(step: string) {
    return Object.fromEntries((await plan()).evaluations!.filter(({ key }) => key.step === step)
      .map(({ key, request_id, status }) => [key.row, { request_id, status }]));
  }
  async function rows(step: string) {
    return (await plan()).rows.filter(({ key }) => key.step === step).map(({ key }) => key.row).sort();
  }
  async function saved() {
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
  }
  async function run(step: string, count: number) {
    await mapStep(page, step).click();
    await tab(page, 'Runs').click();
    const admitted = mutation('POST', runPath);
    await page.getByRole('tabpanel', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
    expect((await admitted).status()).toBe(202);
    await expect(runs.getByText('Done', { exact: true })).toHaveCount(count, { timeout: 90_000 });
    const failed = (await (await request.get('/api/jobs', { params: { project } })).json() as StudioJobSnapshot).jobs
      .filter(({ status }) => status === 'failed').map(({ kind, reason }) => ({ kind, reason }));
    expect(failed).toEqual([]);
  }
  async function modelEdit(action: () => Promise<unknown>, selection: { axes: Record<string, string[]>; chosen_steps?: string[] }) {
    const [edited] = await Promise.all([mutation('POST', `${path}/model/edit`), action()]);
    expect(edited.ok(), await edited.text()).toBe(true);
    expect(edited.request().postDataJSON()).toMatchObject(selection);
    const { result } = await edited.json() as { result: { status: string; problem?: { detail: string } } };
    expect(result.status, result.problem?.detail).toBe('applied');
    await saved();
  }
  /** The runs an optional axis keeps beside their copies: its first mark, named by no value. */
  async function chooseOriginals(step: string, axis: string) {
    await sweepLabel(card(step), axis).click();
    await page.getByRole('menu', { name: `Sweep ${axis}` }).getByRole('menuitem').first().click();
  }

  await page.goto('/?project=' + encodeURIComponent(project), { waitUntil: 'domcontentloaded' });
  await run('train', 1);
  await run('train_b', 1);
  await run('pair', 2);
  const before = await evaluations('pair');
  expect(Object.keys(before)).toHaveLength(2);

  // One run selected gains one copy; Undo removes it.
  await marks('pair').first().click();
  await fromMenu(marks('pair').first(), 'Duplicate Run');
  await expect(marks('pair')).toHaveCount(3);
  await saved();
  await page.keyboard.press('ControlOrMeta+z');
  await expect(marks('pair')).toHaveCount(2);
  await saved();

  // The whole card gains a copy of each run, which is selected; an added component and its wire is stored on it.
  await mapStep(page, 'pair').click();
  await fromMenu(marks('pair').first(), 'Duplicate Run');
  await expect(marks('pair')).toHaveCount(4);
  await saved();
  await tab(page, 'Model').click();
  await expect(model.locator('.react-flow__node[data-id="plant"]')).toBeVisible();
  await modelEdit(() => placeFromLibrary(page, 'GRU'), { axes: { run: ['run-2'] } });
  const gru = model.locator('.react-flow__node').filter({ hasText: 'GRU' }).first();
  await expect(gru).toHaveClass(/\bdraggable\b/);
  const handle = (direction: 'input' | 'output') => gru.locator(`[data-handleid="${direction}:hidden"]`);
  await modelEdit(async () => {
    await handle('output').hover();
    await page.mouse.down();
    await handle('input').hover();
    await page.mouse.up();
  }, { axes: { run: ['run-2'] } });

  // The original runs gain another copy: an intervention on it.
  await chooseOriginals('pair', 'run');
  await fromMenu(marks('pair').first(), 'Duplicate Run');
  await expect(marks('pair')).toHaveCount(6);
  await saved();
  await outlineRow(page, 'Wrapper.Force → Plant.force').click();
  await inspector.getByRole('button', { name: 'Add Attachment', exact: true }).click();
  await modelEdit(() => page.getByRole('menuitem', { name: 'Add Intervention', exact: true }).click(), { axes: { run: ['run-3'] } });

  // And once more: a field.
  await chooseOriginals('pair', 'run');
  await fromMenu(marks('pair').first(), 'Duplicate Run');
  await expect(marks('pair')).toHaveCount(8);
  await saved();
  await outlineRow(page, 'Plant').click();
  await modelEdit(() => enter(inspector.getByRole('textbox', { name: 'Mass', exact: true }), '3'), { axes: { run: ['run-4'] } });

  const edited = await evaluations('pair');
  expect(Object.keys(edited)).toHaveLength(8);
  for (const [row, original] of Object.entries(before)) expect(edited[row]).toEqual(original);
  const copies = Object.entries(edited).filter(([row]) => !(row in before));
  expect(new Set(copies.map(([, { request_id }]) => request_id)).size).toBe(6);
  for (const [, { request_id, status }] of copies) {
    expect(Object.values(before).map((item) => item.request_id)).not.toContain(request_id);
    expect(status).not.toBe('succeeded');
  }

  // Save Set: the axis reads the new set `set`.
  await sweepLabel(card('pair'), 'run').click();
  await page.getByRole('menuitem', { name: 'Save Set', exact: true }).click();
  await expect(sweepLabel(card('pair'), 'set')).toBeVisible();
  await saved();
  const stored = (await snapshot()).project;
  expect(stored.steps!.pair.sweep!.axes!.run).toEqual({ set: 'set', optional: true });
  expect(Object.keys((stored.documents!.set as { payload: { entries: object } }).payload.entries).sort())
    .toEqual(['run-2', 'run-3', 'run-4']);
  expect(await evaluations('pair')).toEqual(edited);

  // A Train card's Add Sweep offers the set, and has one run per entry.
  await mapStep(page, 'other').click();
  await card('other').getByRole('button', { name: 'Add Sweep other', exact: true }).click();
  await page.getByRole('menuitem', { name: 'set', exact: true }).click();
  await expect(marks('other')).toHaveCount(3);
  await saved();
  expect(await rows('other')).toEqual(['set=run-2', 'set=run-3', 'set=run-4']);

  // An entry edited from the Train card with the Evaluate card taken out makes a set based on `set`.
  await chooseSweepValue(card('other'), 'set', 'run-4');
  await tab(page, 'Model').click();
  await mapStep(page, 'pair').click({ modifiers: ['Shift'] });
  await outlineRow(page, 'Plant').click();
  await modelEdit(() => enter(inspector.getByRole('textbox', { name: 'Mass', exact: true }), '4'), {
    chosen_steps: ['other'], axes: { set: ['run-4'] },
  });
  const narrowed = (await snapshot()).project;
  expect(narrowed.documents!['set-2']).toMatchObject({ base: { document: 'set' } });
  expect(narrowed.steps!.other.sweep!.axes!.set).toEqual({ set: 'set-2' });
  expect(narrowed.steps!.pair.sweep!.axes!.run).toEqual({ set: 'set', optional: true });
  await sweepLabel(card('other'), 'set-2').click();
  const listed = page.getByRole('menuitemradio');
  await expect(listed).toHaveText(['set', 'set-2']);
  await expect(page.getByRole('menuitemradio', { name: 'set-2', exact: true })).toHaveAttribute('aria-checked', 'true');
  await page.keyboard.press('Escape');

  // Inline Set returns the Evaluate card to an inline axis, and no run is out of date.
  await sweepLabel(card('pair'), 'set').click();
  await page.getByRole('menuitem', { name: 'Inline Set', exact: true }).click();
  await expect(sweepLabel(card('pair'), 'run')).toBeVisible();
  await saved();
  expect((await snapshot()).project.steps!.pair.sweep!.axes!.run).toMatchObject({ optional: true, values: expect.any(Object) });
  expect(await evaluations('pair')).toEqual(edited);
});
