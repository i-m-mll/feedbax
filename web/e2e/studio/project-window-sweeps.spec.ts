import { expect, test, jobTimeout } from './fixtures/studio';
import { type APIRequestContext } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import {
  authorLearningRateSweep, enter, fixture, mapStep, openResultsSlot, seedManagedProject, sweepLabel, mapCard } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

async function readPlan(request: APIRequestContext, path: string): Promise<{ snapshot: StudioProjectSnapshot; plan: StudioSavedTrainRuns }> {
  const response = await request.get(path);
  expect(response.ok(), await response.text()).toBe(true);
  const snapshot: StudioProjectSnapshot = await response.json();
  const result = await request.get(`${path}/train-runs?revision=${snapshot.revision}&window_map=true`);
  // A save that lands between the two reads moves the revision on; read the newer one.
  if (result.status() === 409 && (await result.json()).detail?.current) return await readPlan(request, path);
  expect(result.ok(), await result.text()).toBe(true);
  const plan: StudioSavedTrainRuns = await result.json();
  expect(plan.window_map?.problems).toEqual([]);
  return { snapshot, plan };
}

test('the saved Train field, map and Runs share sweep values and row identity', { tag: '@round' }, async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const card = mapCard(page, 'train');
  const table = page.getByRole('table', { name: 'Runs' });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  const rows = table.locator('tbody tr[data-index]');
  await expect(table).toHaveAttribute('aria-rowcount', '2');
  await expect(rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true })).toHaveValue('0.001');

  await authorLearningRateSweep(rate);
  await sweepLabel(card, 'Learning Rate').click();
  for (const name of ['0.001', '0.002', '0.003']) {
    await expect(page.getByRole('menuitem', { name, exact: true })).toBeVisible();
    await expect(table.getByRole('cell', { name, exact: true })).toBeVisible();
  }
  await expect(table.getByText('Not run', { exact: true })).toHaveCount(3);
  await page.getByRole('menuitem', { name: '0.002', exact: true }).click();
  await expect(rate.getByRole('button', { name: 'Select Learning Rate 0.002', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await mapStep(page, 'train').click();

  const rateText = rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true });
  const pending = (await readPlan(request, projectPath)).plan;
  const medium = pending.window_map!.rows.find(({ coordinates }) => coordinates['learning-rate']?.value === 0.002)!;
  await enter(rateText, '0.001, 0.0025, 0.003');
  await expect(table.getByRole('cell', { name: '0.0025', exact: true })).toBeVisible();
  const changedBeforeRun = (await readPlan(request, projectPath)).plan.rows;
  expect(changedBeforeRun.map(({ key }) => key)).toEqual(pending.rows.map(({ key }) => key));
  expect(changedBeforeRun.filter((row, i) => row.request_id !== pending.rows[i].request_id).map(({ key }) => key.row))
    .toEqual([medium.key.row]);

  const admission = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === `${projectPath}/train-runs`);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const accepted = await admission;
  expect(accepted.status(), await accepted.text()).toBe(202);
  await expect(table.getByText('Done', { exact: true })).toHaveCount(3, { timeout: jobTimeout });
  const done = (await readPlan(request, projectPath)).plan.rows;

  await enter(rateText, '0.001, 0.00275, 0.003');
  await expect(table.getByText('Out of date', { exact: true })).toHaveCount(1);
  await expect(card.locator('[data-status-mark="stale"]')).toHaveCount(1);
  await table.getByRole('row').filter({ hasText: 'Out of date' }).getByRole('button').click();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByText('0.0025 → 0.00275', { exact: false })).toBeVisible();
  await page.reload();
  await expect(rateText).toHaveValue('0.001, 0.00275, 0.003');
  await expect(table.getByText('Done', { exact: true })).toHaveCount(2);
  const { plan: reopened } = await readPlan(request, projectPath);
  expect(reopened.rows.map(({ key }) => key)).toEqual(done.map(({ key }) => key));
  for (const [i, row] of reopened.rows.entries()) {
    if (row.key.row === medium.key.row) {
      expect(row.request_id).not.toBe(done[i].request_id);
      expect(row.status).toBe('stale');
      expect(row.previous?.receipt).toEqual(done[i].result?.receipt);
      expect(row.change).toMatchObject({ before: 0.0025, after: 0.00275 });
    } else {
      expect(row).toEqual(done[i]);
    }
  }

  await card.getByRole('button', { name: 'Add Sweep Train' }).click();
  await page.getByRole('menu', { name: 'Add Sweep' }).getByRole('menuitem', { name: 'Seed', exact: true }).click();
  const seed = page.getByRole('region', { name: 'Seed field', exact: true });
  const seedText = seed.getByRole('textbox', { name: 'Seed', exact: true });
  await expect(seedText).toBeFocused();
  await expect(seedText).toHaveValue(/^\d+$/);
  await enter(seedText, `${await seedText.inputValue()}, 9`);
  await expect(table).toHaveAttribute('aria-rowcount', '7');
  await expect(card.getByRole('button', { name: 'Sweep Seed 2', exact: true })).toBeVisible();
  // Unarranged, the longer of the first two sweeps runs across, so seed comes first.
  await expect(card).toContainText('Seed 2 · Learning Rate 3');
  await expect(table.getByRole('rowheader').locator('[data-run-ordinal]')).toHaveText(['1', '2', '3', '4', '5', '6']);
  await expect(table.getByRole('columnheader')).toHaveText(['Learning Rate', 'Seed', 'Status', 'Progress']);
  await expect(page.locator('body')).not.toContainText(/value-\w|=value\b/);
  const runsSlot = page.locator('[data-slot="primary"]');
  const runsHeader = runsSlot.locator('.studio-window-slot__header');
  const run = runsHeader.getByRole('button', { name: 'Run', exact: true });
  const stop = runsHeader.getByRole('button', { name: 'Stop', exact: true });
  await expect(runsHeader.getByRole('combobox', { name: 'Machine' })).toBeVisible();
  await expect(run).toBeVisible();
  await expect(runsHeader.getByRole('button', { name: /^(Check|Accept|New Step)/ })).toHaveCount(0);
  await page.screenshot({ path: test.info().outputPath('runs-wide.png') });
  await page.setViewportSize({ width: 1180, height: 900 });
  await page.screenshot({ path: test.info().outputPath('runs-medium.png') });
  await page.setViewportSize({ width: 820, height: 900 });
  await expect(run).toBeVisible();
  expect(await runsHeader.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await page.screenshot({ path: test.info().outputPath('runs-narrow.png') });
  await page.setViewportSize({ width: 520, height: 900 });
  await expect.poll(() => runsSlot.evaluate((node) => node.getBoundingClientRect().width)).toBe(260);
  const more = runsHeader.getByRole('button', { name: 'More', exact: true });
  await expect(run).toBeVisible();
  await expect(stop).toBeVisible();
  await expect(more).toBeVisible();
  expect(await runsHeader.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await page.screenshot({ path: test.info().outputPath('runs-minimum.png') });
  await more.click();
  const menu = page.getByRole('menu', { name: 'More' });
  // The machine picker leaves the bar last, so at the minimum width only commands are rows of More.
  await expect(runsHeader.getByRole('combobox', { name: 'Machine' })).toBeVisible();
  await expect(menu.getByRole('menuitem').first()).toBeVisible();
  await expect(menu.getByRole('menuitem', { name: 'Run', exact: true })).toHaveCount(0);
  await page.screenshot({ path: test.info().outputPath('runs-minimum-menu.png') });
  await page.keyboard.press('Escape');
  await page.setViewportSize({ width: 1440, height: 900 });
  await rate.getByRole('button', { name: 'Select Learning Rate 0.00275', exact: true }).click();
  const selected = rows.and(table.locator('tr[data-selected=true]'));
  await expect(selected).toHaveCount(2);
  await expect(selected.getByRole('cell', { name: '0.00275', exact: true })).toHaveCount(2);
  await expect(seed.getByRole('button', { name: /^Select Seed /, pressed: true })).toHaveCount(2);
  await expect(card.locator('.studio-map-run-mark[aria-pressed="true"]')).toHaveCount(2);
  await expect(rate.getByRole('button', { name: 'Select Learning Rate 0.00275' })).toHaveAttribute('aria-pressed', 'true');
  await expect(rate.getByRole('button', { name: 'Select Learning Rate 0.001' })).toHaveAttribute('aria-pressed', 'false');

  const { plan: crossed } = await readPlan(request, projectPath);
  const keptBySeed = new Map(crossed.window_map!.rows.filter(({ coordinates }) =>
    coordinates['learning-rate']?.value_id === medium.coordinates['learning-rate'].value_id,
  ).map(({ key, coordinates }) => [coordinates.seed.value_id,
    crossed.rows.find((row) => row.key.row === key.row)!.request_id]));
  await mapStep(page, 'train').click();
  const removal = page.getByRole('dialog', { name: 'Remove 4 training runs?', exact: true });
  await enter(rateText, '0.00275');
  await removal.getByRole('button', { name: 'Cancel', exact: true }).click();
  await expect(rateText).toHaveValue('0.001, 0.00275, 0.003');
  await expect(table).toHaveAttribute('aria-rowcount', '7');
  await enter(rateText, '0.00275');
  await removal.getByRole('button', { name: 'Remove', exact: true }).click();
  await expect(rateText).toHaveValue('0.00275');
  await expect(rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true })).toBeVisible();
  await expect(table).toHaveAttribute('aria-rowcount', '3');
  await expect(sweepLabel(card, 'Learning Rate')).toHaveCount(0);
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await page.getByRole('button', { name: 'Forward', exact: true }).click();
  await expect(mapStep(page, 'train')).toHaveAttribute('aria-pressed', 'true');
  await expect(selected).toHaveCount(2);
  const { plan: reduced, snapshot: kept } = await readPlan(request, projectPath);
  expect(reduced.rows).toHaveLength(2);
  expect(reduced.window_map!.rows.map(({ coordinates }) => Object.keys(coordinates))).toEqual([['seed'], ['seed']]);
  expect(reduced.window_map!.rows.map(({ coordinates }) => coordinates.seed.value)).toEqual([0, 9]);
  for (const { key, coordinates } of reduced.window_map!.rows) {
    const row = reduced.rows.find((item) => item.key.row === key.row)!;
    expect(row.request_id).toBe(keptBySeed.get(coordinates.seed.value_id));
  }
  const base = kept.project.steps?.train.base;
  expect(base && 'inline' in base ? base.inline : {}).toMatchObject({
    settings: { optimizer: { learning_rate: { value: 0.00275 } } },
  });
});

// Catches a sweep written before its values are typed, and a nested field the command does not reach.
test('Add Sweep goes to a nested field and sweeps it only by the text it commits', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const sweep = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps?.train.sweep;
  await page.goto(`/?project=${snapshot.project_id}`);
  const card = mapCard(page, 'train');
  const b1 = page.getByRole('region', { name: 'B1 field', exact: true });
  const text = b1.getByRole('textbox', { name: 'B1', exact: true });
  const chooseB1 = async () => {
    await card.getByRole('button', { name: 'Add Sweep Train' }).click();
    await page.getByRole('menu', { name: 'Add Sweep' }).getByRole('menuitem', { name: 'Optimizer', exact: true }).click();
    await page.getByRole('menu', { name: 'Optimizer' }).getByRole('menuitem', { name: 'B1', exact: true }).click();
    await expect(text).toBeFocused();
    await expect(b1.getByRole('button', { name: 'B1 kind: Sweep', exact: true })).toBeVisible();
  };
  const before = await sweep();
  await chooseB1();
  expect(await sweep()).toEqual(before);
  await page.keyboard.press('Escape');
  await expect(b1.getByRole('button', { name: 'B1 kind: Value', exact: true })).toBeVisible();
  await expect(text).toHaveValue('0.9');
  expect(await sweep()).toEqual(before);

  await chooseB1();
  await enter(text, '1e-3, 1e-2');
  await expect(b1.getByRole('button', { name: /^Select B1 / })).toHaveText(['0.001', '0.01']);
  await expect.poll(async () => Object.keys((await sweep())?.axes ?? {}).length).toBe(Object.keys(before?.axes ?? {}).length + 1);
});

// Catches the kind menu's Sweep writing a sweep before any values are typed.
test("a field's kind menu puts it in Sweep kind and writes nothing until its text is committed", async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const train = async () => ((await (await request.get(projectPath)).json()) as StudioProjectSnapshot).project.steps?.train;
  await page.goto(`/?project=${snapshot.project_id}`);
  const b2 = page.getByRole('region', { name: 'B2 field', exact: true });
  const text = b2.getByRole('textbox', { name: 'B2', exact: true });
  await expect(text).toBeEditable();
  const value = await text.inputValue();
  const before = await train();
  await b2.getByRole('button', { name: 'B2 kind: Value', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Sweep', exact: true }).click();
  await expect(text).toBeFocused();
  await expect(b2.getByRole('button', { name: 'B2 kind: Sweep', exact: true })).toBeVisible();
  await expect(text).toHaveValue(value);
  expect(await train()).toEqual(before);
  await page.keyboard.press('Escape');
  await expect(b2.getByRole('button', { name: 'B2 kind: Value', exact: true })).toBeVisible();
  expect(await train()).toEqual(before);
});

test('an addressed Model numeric field authors a step sweep from the inspector', async ({ page, request }) => {
  const [model, task, objective, train] = await fixture('ordinary-files', '2');
  const { snapshot, projectPath } = await seedManagedProject(request, [model, task, objective, train]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('navigation', { name: 'Details', exact: true }).getByRole('tab', { name: 'Model', exact: true }).click();
  // The inspector first projects the authored fields, then the selected run's; only the latter can sweep.
  const runFields = page.waitForResponse((response) => response.url().endsWith('/model/fields')
    && 'run' in JSON.parse(response.request().postData() ?? '{}'));
  await page.getByRole('tabpanel', { name: 'Model', exact: true })
    .getByRole('article', { name: 'Point mass', exact: true }).locator('header').click();
  const field = page.getByRole('region', { name: 'Mass field', exact: true });
  await runFields;
  await field.getByRole('button', { name: 'Mass kind: Value', exact: true }).click();
  await page.getByRole('menuitemradio', { name: 'Sweep', exact: true }).click();
  const text = field.getByRole('textbox', { name: 'Mass', exact: true });
  await expect(text).toBeFocused();
  await enter(text, '1, 2');
  await expect.poll(async () => (await readPlan(request, projectPath)).plan.window_map!.rows
    .map(({ coordinates }) => coordinates.mass?.value)).toEqual([1, 2]);
  const { plan } = await readPlan(request, projectPath);
  expect(new Set(plan.rows.map(({ request_id }) => request_id)).size).toBe(2);
  await expect(mapCard(page, 'train').locator('[data-status-mark="not_run"]')).toHaveCount(2);
});

test('an authored component range reopens as its exact rule and To List keeps row IDs', async ({ page, request }) => {
  const [model, task, objective, train] = await fixture('ordinary-files', '2');
  train.step.sweep = { axes: { mass: {
    path: JSON.stringify({ composites: [],
      element: { kind: 'field', parent: { id: 'plant', kind: 'component' }, path: ['fields', 'mass'] },
    }),
    range: { start: 1.000123456789, stop: 2.000987654321, count: 3 },
  } } };
  const { snapshot, projectPath } = await seedManagedProject(request, [model, task, objective, train]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('navigation', { name: 'Details', exact: true }).getByRole('tab', { name: 'Model', exact: true }).click();
  await page.getByRole('tabpanel', { name: 'Model', exact: true })
    .getByRole('article', { name: 'Point mass', exact: true }).locator('header').click();
  const field = page.getByRole('region', { name: 'Mass field', exact: true });
  await expect(field.getByRole('textbox', { name: 'First', exact: true })).toHaveValue('1.000123456789');
  await expect(field.getByRole('textbox', { name: 'Last', exact: true })).toHaveValue('2.000987654321');
  await expect(field.getByRole('textbox', { name: 'Count', exact: true })).toHaveValue('3');
  const { plan: authored } = await readPlan(request, projectPath);
  await field.getByRole('button', { name: 'To List', exact: true }).click();
  await expect(field.getByRole('textbox', { name: 'Mass', exact: true })).toHaveValue(/^1\.000123456789, .+, 2\.000987654321$/);
  const { plan: listed } = await readPlan(request, projectPath);
  expect(listed.rows).toEqual(authored.rows);
});

test('a press on a run\'s row shows that run\'s Results, and a reload keeps it', { tag: '@round' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const table = page.getByRole('table', { name: 'Runs' });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  await authorLearningRateSweep(rate);
  await enter(rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true }), '0.001, 0.003');
  await expect(table).toHaveAttribute('aria-rowcount', '3');
  const admission = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === `${projectPath}/train-runs`);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const accepted = await admission;
  expect(accepted.status(), await accepted.text()).toBe(202);
  const jobs = ((await accepted.json()) as { items: { job_id: string }[] }).items.map(({ job_id }) => job_id);
  await expect(table.getByText('Done', { exact: true })).toHaveCount(2, { timeout: jobTimeout });

  const status = (index: number) => table.locator('tbody tr[data-index]').nth(index).getByRole('cell').filter({ hasText: 'Done' });
  await status(0).click();
  await openResultsSlot(page);
  const results = page.locator('[data-slot="secondary"]').getByRole('region', { name: 'Results', exact: true }).last();
  const shown = async () => {
    const history = results.getByRole('img', { name: 'Metric train_loss', exact: true }).locator('[data-result-attempt]');
    await expect(history.locator('polyline')).toBeAttached({ timeout: jobTimeout });
    return await history.getAttribute('data-result-attempt');
  };
  const first = await shown();
  await status(1).click();
  await expect.poll(shown).not.toBe(first);
  const second = await shown();
  expect(jobs).toEqual(expect.arrayContaining([first, second]));

  await page.reload();
  await expect(table.locator('tbody tr[data-index]').nth(1).getByRole('button').first()).toHaveAttribute('aria-current', 'true');
  await expect.poll(shown).toBe(second);
});
