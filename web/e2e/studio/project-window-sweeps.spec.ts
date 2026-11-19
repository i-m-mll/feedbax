import { expect, test } from './fixtures/studio';
import { type APIRequestContext } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { authorLearningRateSweep, enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

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
  test.setTimeout(55_000);
  const files = await fixture('ordinary-files', '2');
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const card = page.getByRole('article', { name: 'Train train' });
  const table = page.getByRole('table', { name: 'Runs' });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  const rows = table.locator('tbody tr[data-index]');
  await expect(table).toHaveAttribute('aria-rowcount', '2');
  await expect(rate.getByRole('textbox', { name: 'Learning Rate', exact: true })).toHaveValue('0.001');

  await authorLearningRateSweep(rate);
  await expect(card.getByRole('button', { name: 'Sweep learning-rate' })).toHaveText('learning-rate 3');
  await card.getByRole('button', { name: 'Sweep learning-rate' }).click();
  for (const name of ['0.001', '0.002', '0.003']) {
    await expect(card.getByRole('listitem').getByRole('button', { name: `Select learning-rate ${name}`, exact: true })).toBeVisible();
    await expect(table.getByRole('cell', { name, exact: true })).toBeVisible();
  }
  await expect(table.getByText('Not run', { exact: true })).toHaveCount(3);
  await card.getByRole('listitem').getByRole('button', { name: 'Select learning-rate 0.002', exact: true }).click();
  await expect(rate.getByRole('button', { name: 'Select Learning Rate 0.002', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await mapStep(page, 'train').click();

  const rateText = rate.getByRole('textbox', { name: 'Learning Rate', exact: true });
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
  await expect(table.getByText('Done', { exact: true })).toHaveCount(3, { timeout: 25_000 });
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

  await card.getByRole('button', { name: 'Add Sweep train' }).click();
  await page.getByRole('menu', { name: 'Add Sweep' }).getByRole('menuitem', { name: 'Seed', exact: true }).click();
  const seed = page.getByRole('region', { name: 'Seed field', exact: true });
  const seedText = seed.getByRole('textbox', { name: 'Seed', exact: true });
  await expect(seedText).toHaveValue(/^\d+,$/);
  await enter(seedText, `${(await seedText.inputValue()).replace(/,$/, '')}, 9`);
  await expect(table).toHaveAttribute('aria-rowcount', '7');
  await expect(card.getByRole('button', { name: 'Sweep seed' })).toHaveText('seed 2');
  // Unarranged, the longer of the first two sweeps runs across, so seed comes first.
  await expect(card).toContainText('seed 2 · learning-rate 3');
  await expect(table.getByRole('rowheader').locator('[data-run-ordinal]')).toHaveText(['1', '2', '3', '4', '5', '6']);
  await expect(table.getByRole('columnheader')).toHaveText(['learning-rate', 'seed', 'Status', 'Progress']);
  await expect(page.locator('body')).not.toContainText(/value-\w|=value\b/);
  const runsSlot = page.locator('[data-slot="primary"]');
  const runsHeader = runsSlot.locator('.studio-window-slot__header');
  const run = runsHeader.getByRole('button', { name: 'Run', exact: true });
  const stop = runsHeader.getByRole('button', { name: 'Stop', exact: true });
  await expect(runsHeader.getByRole('combobox', { name: 'Machine' })).toBeVisible();
  await expect(run).toHaveText('Run');
  await expect(runsHeader.getByRole('button', { name: /^(Check|Accept|New Step)/ })).toHaveCount(0);
  await page.screenshot({ path: test.info().outputPath('runs-wide.png') });
  await page.setViewportSize({ width: 1180, height: 900 });
  await page.screenshot({ path: test.info().outputPath('runs-medium.png') });
  await page.setViewportSize({ width: 820, height: 900 });
  const more = runsHeader.getByRole('button', { name: 'More', exact: true });
  await expect(more).toBeVisible();
  await expect(run).toBeVisible();
  expect(await runsHeader.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await page.screenshot({ path: test.info().outputPath('runs-narrow.png') });
  await more.click();
  const menu = page.getByRole('menu', { name: 'More' });
  await expect(menu.getByRole('menuitemradio').first()).toBeVisible();
  await expect(menu.getByRole('menuitem', { name: 'Run', exact: true })).toHaveCount(0);
  await page.screenshot({ path: test.info().outputPath('runs-narrow-menu.png') });
  await page.keyboard.press('Escape');
  await page.setViewportSize({ width: 700, height: 900 });
  await expect.poll(() => runsSlot.evaluate((node) => node.getBoundingClientRect().width)).toBe(260);
  await expect(run).toBeVisible();
  await expect(stop).toBeVisible();
  await expect(more).toBeVisible();
  expect(await runsHeader.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true);
  await page.screenshot({ path: test.info().outputPath('runs-minimum.png') });
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
  await expect(card.getByRole('button', { name: 'Sweep learning-rate' })).toHaveCount(0);
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
  await page.getByRole('menuitem', { name: 'Sweep', exact: true }).click();
  await enter(field.getByRole('textbox', { name: 'Mass', exact: true }), '1, 2');
  await expect.poll(async () => (await readPlan(request, projectPath)).plan.window_map!.rows
    .map(({ coordinates }) => coordinates.mass?.value)).toEqual([1, 2]);
  const { plan } = await readPlan(request, projectPath);
  expect(new Set(plan.rows.map(({ request_id }) => request_id)).size).toBe(2);
  await expect(page.getByRole('article', { name: 'Train train' }).locator('[data-status-mark="not_run"]')).toHaveCount(2);
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
