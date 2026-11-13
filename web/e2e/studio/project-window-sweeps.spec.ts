import { expect, test, type APIRequestContext } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { authorLearningRateSweep, enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

async function readPlan(request: APIRequestContext, path: string) {
  const response = await request.get(path);
  expect(response.ok()).toBe(true);
  const snapshot: StudioProjectSnapshot = await response.json();
  const result = await request.get(`${path}/train-runs?revision=${snapshot.revision}&window_map=true`);
  expect(result.ok()).toBe(true);
  const plan: StudioSavedTrainRuns = await result.json();
  expect(plan.window_map?.problems).toEqual([]);
  return { snapshot, plan };
}

test('the saved Train field, map and Runs share named sweep values and row identity', async ({ page, request }) => {
  test.setTimeout(55_000);
  const files = await fixture('files');
  const { snapshot, projectPath } = await seedManagedProject(request, files.slice(0, 3));
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
  for (const name of ['Slow · 0.001', 'Medium · 0.002', 'Fast · 0.003']) {
    await expect(card.getByRole('listitem').getByRole('button', { name: `Select learning-rate ${name}`, exact: true })).toBeVisible();
    await expect(table.getByRole('cell', { name, exact: true })).toBeVisible();
  }
  await expect(table.getByText('Not run', { exact: true })).toHaveCount(3);
  await card.getByRole('listitem').getByRole('button', { name: 'Select learning-rate Medium · 0.002', exact: true }).click();
  await expect(rate.getByRole('button', { name: 'Select Learning Rate Medium', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await mapStep(page, 'train').click();

  const pending = (await readPlan(request, projectPath)).plan;
  const medium = pending.window_map!.rows.find(({ coordinates }) => coordinates['learning-rate']?.label === 'Medium')!;
  await enter(rate.getByRole('textbox', { name: 'Medium', exact: true }), '0.0025');
  await expect(table.getByRole('cell', { name: 'Medium · 0.0025' })).toBeVisible();
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

  await enter(rate.getByRole('textbox', { name: 'Medium', exact: true }), '0.00275');
  await expect(table.getByText('Out of date', { exact: true })).toHaveCount(1);
  await expect(card.getByText('1 Out of date')).toBeVisible();
  await table.getByRole('row').filter({ hasText: 'Out of date' }).getByRole('button').click();
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true })
    .getByText('0.0025 → 0.00275', { exact: false })).toBeVisible();
  await page.reload();
  await expect(rate.getByRole('textbox', { name: 'Medium', exact: true })).toHaveValue('0.00275');
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
  await page.getByRole('dialog', { name: 'Add Sweep' }).getByRole('button', { name: 'Seed', exact: true }).click();
  const seed = page.getByRole('region', { name: 'Seed field', exact: true });
  await expect(seed.getByRole('textbox', { name: 'Seed value 1', exact: true })).toBeVisible();
  const firstSeed = await seed.getByRole('textbox', { name: 'Seed value 1', exact: true }).inputValue();
  await enter(seed.getByRole('textbox', { name: 'Seed', exact: true }), `${firstSeed}, 9`);
  await expect(table).toHaveAttribute('aria-rowcount', '7');
  await expect(card.getByRole('button', { name: 'Sweep seed' })).toHaveText('seed 2');
  await expect(card).toContainText(/learning-rate 3\s*×\s*seed 2/);
  await rate.getByRole('button', { name: 'Select Learning Rate Medium', exact: true }).click();
  const selected = rows.and(table.locator('tr[data-selected=true]'));
  await expect(selected).toHaveCount(2);
  await expect(selected.getByRole('cell', { name: 'Medium · 0.00275', exact: true })).toHaveCount(2);
  await expect(seed.getByRole('button', { name: /^Select Seed /, pressed: true })).toHaveCount(2);
  await expect(card.getByText('2 of 6', { exact: false })).toBeVisible();
  await expect(rate.getByRole('button', { name: 'Select Learning Rate Medium' })).toHaveAttribute('aria-pressed', 'true');
  await expect(rate.getByRole('button', { name: 'Select Learning Rate Slow' })).toHaveAttribute('aria-pressed', 'false');

  const { plan: crossed } = await readPlan(request, projectPath);
  const keptBySeed = new Map(crossed.window_map!.rows.filter(({ coordinates }) =>
    coordinates['learning-rate']?.value_id === medium.coordinates['learning-rate'].value_id,
  ).map(({ key, coordinates }) => [coordinates.seed.value_id,
    crossed.rows.find((row) => row.key.row === key.row)!.request_id]));
  await mapStep(page, 'train').click();
  await rate.getByRole('button', { name: 'Learning Rate kind: Sweep' }).click();
  await rate.getByRole('menuitem', { name: 'Value' }).click();
  await rate.getByRole('combobox', { name: 'Keep Value Learning Rate' }).selectOption({ label: '0.00275' });
  await rate.getByRole('button', { name: 'Keep Value' }).click();
  await expect(rate.getByRole('textbox', { name: 'Learning Rate', exact: true })).toHaveValue('0.00275');
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
    training_config: { learning_rate: 0.00275 },
    method_payload: { payload: { optimizer: { params: { learning_rate: 0.00275 } } } },
  });
});

test('an inline component numeric field authors a step sweep from the inspector', async ({ page, request }) => {
  const [model, task, train] = await fixture('files');
  train.step.base.inline.graph = model.document.payload;
  const { snapshot, projectPath } = await seedManagedProject(request, [task, train]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('navigation', { name: 'Details', exact: true }).getByRole('tab', { name: 'Model', exact: true }).click();
  await page.getByRole('complementary', { name: 'Details Drawer' }).getByRole('button', { name: 'plant', exact: true }).click();
  const field = page.getByRole('complementary', { name: 'Inspector' }).getByRole('group', { name: 'Mass', exact: true });
  await field.getByRole('button', { name: 'To Sweep Mass', exact: true }).click();
  await field.getByRole('button', { name: 'Add Value Mass' }).click();
  const edits = field.getByRole('button', { name: /^Edit Mass / });
  await expect(edits).toHaveCount(2);
  await edits.last().click();
  await enter(field.getByRole('spinbutton'), '2');
  await expect(field.getByRole('spinbutton', { name: 'Edit Mass 2', exact: true })).toBeEnabled();
  const { plan } = await readPlan(request, projectPath);
  expect(plan.window_map!.rows.map(({ coordinates }) => coordinates.mass.value)).toEqual([1, 2]);
  expect(new Set(plan.rows.map(({ request_id }) => request_id)).size).toBe(2);
  await expect(page.getByRole('article', { name: 'Train train' }).getByText('2 Not run')).toBeVisible();
});


test('an authored component range reopens as its exact rule and To List keeps row IDs', async ({ page, request }) => {
  const [model, task, train] = await fixture('files');
  train.step.base.inline.graph = model.document.payload;
  train.step.sweep = { axes: { mass: {
    path: 'graph.nodes.plant.params.mass',
    range: { start: 1.000123456789, stop: 2.000987654321, count: 3 },
  } } };
  const { snapshot, projectPath } = await seedManagedProject(request, [task, train]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await page.reload();
  const field = page.getByRole('group', { name: 'plant · Mass', exact: true });
  await expect(field.getByText('1.000123456789', { exact: true })).toBeVisible();
  await expect(field.getByText('2.000987654321', { exact: true })).toBeVisible();
  await expect(field.getByText('Count', { exact: true })).toBeVisible();
  const { plan: authored } = await readPlan(request, projectPath);
  await field.getByRole('button', { name: 'To List', exact: true }).click();
  const edits = field.getByRole('button', { name: /^Edit plant · Mass / });
  await expect(edits).toHaveCount(3);
  await edits.first().click();
  await expect(field.getByRole('spinbutton')).toHaveValue('1.000123456789');
  await edits.last().click();
  await expect(field.getByRole('spinbutton')).toHaveValue('2.000987654321');
  const { plan: listed } = await readPlan(request, projectPath);
  expect(listed.rows).toEqual(authored.rows);
});
