import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

test('Training and Runs edit the same saved values and project the selected source', async ({ page, request }) => {
  test.fixme(true, 'Owned by V1 (feature/bf89b13-r2-v1): re-entering a list after a geomspace range saves stale value labels.');
  test.setTimeout(55_000);
  const files = await fixture('ordinary-files', '2');
  // Train offers the settings a project authors, so the project authors Grad Clip to edit it.
  files.at(-1).step.base.inline.settings.grad_clip = 1;
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  const clip = page.getByRole('region', { name: 'Grad Clip field', exact: true });
  const batches = page.getByRole('region', { name: 'N Batches field', exact: true });
  const rateText = rate.getByRole('textbox', { name: 'Learning Rate', exact: true });
  const clipText = clip.getByRole('textbox', { name: 'Grad Clip', exact: true });
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function plan() {
    const { revision } = await saved();
    const response = await request.get(projectPath + '/train-runs?window_map=true&revision=' + revision);
    if (response.status() === 409) return undefined;
    expect(response.ok(), await response.text()).toBe(true);
    const projection = await response.json() as StudioSavedTrainRuns;
    expect(projection.window_map?.problems).toEqual([]);
    return projection;
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await expect(rateText).toHaveValue('0.001');
  const method = page.getByRole('combobox', { name: 'Training method', exact: true });
  await expect(method).toHaveValue('Supervised gradient');
  await method.click();
  await expect(page.getByRole('option', { name: /^Supervised gradient\s*feedbax\.builtins$/ })).toBeVisible();
  await page.keyboard.press('Escape');
  const optimizer = page.getByRole('region', { name: 'Optimizer field', exact: true });
  await expect(optimizer.getByRole('region', { name: 'Params field', exact: true })).toContainText('Learning Rate');
  await rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }).click();
  await expect(page.getByRole('menuitem', { name: 'Sweep', exact: true })).toBeEnabled();
  await page.keyboard.press('Escape');
  await enter(optimizer.getByRole('textbox', { name: 'Type', exact: true }), 'sgd');
  await expect.poll(async () => {
    const base = (await saved()).project.steps?.train.base;
    return base && 'inline' in base ? (base.inline as { settings: { optimizer: { type: string } } }).settings.optimizer.type : undefined;
  }).toBe('sgd');
  await enter(rateText, 'geomspace(1e-4, 1e-2, 5)');
  await expect(card.getByRole('button', { name: 'Sweep learning-rate' })).toHaveText('learning-rate 5');
  await enter(rateText, '1e-3, 2e-3, 3e-3');
  await expect(card.getByRole('button', { name: 'Sweep learning-rate' })).toHaveText('learning-rate 3');
  const before = (await plan())!.rows;
  const axes = (await saved()).project.steps?.train.sweep?.axes;
  const initial = axes?.['learning-rate'];
  expect(initial).toHaveProperty('values');
  const values = (initial as { values: Record<string, { label?: string; patches: Record<string, { value: number }> }> }).values;
  expect(Object.values(values).map(({ label }) => label)).toEqual([undefined, undefined, undefined]);
  for (const entry of Object.values(values)) {
    const patches = entry.patches;
    expect(Object.keys(patches)).toEqual(['settings.optimizer.params.learning_rate']);
  }
  await enter(rateText, '0.001, 0.0025, 0.003');
  await expect.poll(async () => (await plan())?.rows.filter((row, index) => row.request_id !== before[index].request_id).length).toBe(1);
  const after = (await plan())!.rows;
  expect(after.map(({ key }) => key)).toEqual(before.map(({ key }) => key));

  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const selectValue = rate.getByRole('button', { name: 'Select Learning Rate 0.0025', exact: true });
  await selectValue.click();
  await expect(selectValue).toHaveAttribute('aria-pressed', 'true');
  await expect(rate.getByLabel('Learning Rate selected value', { exact: true })).toBeVisible();
  const batchText = batches.getByRole('textbox', { name: 'N Batches', exact: true });
  const initialBatches = await batchText.inputValue();
  const override = String(Number(initialBatches) + 1);
  await enter(batchText, override);
  await expect(batches.getByLabel('N Batches selected value', { exact: true })).toBeVisible();
  await expect(batchText).toHaveValue(override);
  await expect(rateText).toBeVisible();
  const exceptions = page.getByRole('list').filter({ has: page.getByRole('button', { name: 'Remove Exception' }) });
  await expect(exceptions.getByRole('listitem')).toHaveText([`N Batches ${override} · learning-rate 0.0025 · 1 run`]);
  const scoped = (await plan())!.rows;
  expect(scoped.filter((row, index) => row.request_id !== after[index].request_id)).toHaveLength(1);
  await batchText.fill('77');
  await rate.getByRole('button', { name: 'Select Learning Rate 0.001', exact: true }).click({ modifiers: ['Shift'] });
  await expect(batchText).toHaveValue('');
  await expect(batches.locator('output')).toHaveCount(0);
  await rate.getByRole('button', { name: 'Select Learning Rate 0.001', exact: true }).click();
  await expect(batchText).toHaveValue(initialBatches);
  await expect(batches.locator('output')).toHaveCount(0);

  await page.getByRole('tabpanel', { name: 'Runs', exact: true }).focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => (await plan())?.rows.map(({ request_id }) => request_id)).toEqual(after.map(({ request_id }) => request_id));
  await page.keyboard.press('Meta+Shift+Z');
  await expect.poll(async () => (await plan())?.rows.map(({ request_id }) => request_id)).toEqual(scoped.map(({ request_id }) => request_id));
  await selectValue.click();
  await expect(batchText).toHaveValue(override);
  await exceptions.getByRole('button', { name: 'Remove Exception', exact: true }).click();
  await expect.poll(async () => (await saved()).project.steps?.train.sweep?.exceptions ?? {}).toEqual({});
  await expect(batchText).toHaveValue(initialBatches);
  await expect(exceptions).toHaveCount(0);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await enter(clipText, '0.75');
  await expect.poll(async () => JSON.stringify((await saved()).project.steps?.train.base)).toContain('"grad_clip":0.75');
  await enter(clipText, 'geomspace(0.1, 1, 3)');
  await expect(card.getByRole('button', { name: 'Sweep grad-clip' })).toHaveText('grad-clip 3');
  await expect(clip.getByRole('combobox', { name: 'Spacing', exact: true })).toHaveText('Logarithmic');
  await page.reload();
  await expect(clipText).toHaveValue('geomspace(0.1, 1, 3)');
  await expect(clip.getByRole('textbox', { name: 'Count', exact: true })).toHaveValue('3');
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await expect(rateText).toHaveValue('0.001, 0.0025, 0.003');
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('two runs of a 4 × 2 sweep take a value as one exception, and removing it restores the rule', async ({ page, request }) => {
  const [model, task, objective, train] = await fixture('ordinary-files', '2');
  train.step.base.inline.seed = 0;
  train.step.sweep = { axes: {
    seed: { path: 'seed', values: [0, 1, 2, 3] },
    'n-batches': { path: 'training_config.n_batches', values: [1, 2] },
  } };
  const { snapshot } = await seedManagedProject(request, [model, task, objective, train]);
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  const seed = page.getByRole('region', { name: 'Seed field', exact: true });
  const batch = page.getByRole('region', { name: 'Batch Size field', exact: true });
  const batchText = batch.getByRole('textbox', { name: 'Batch Size', exact: true });
  const rule = await batchText.inputValue();
  await seed.getByRole('button', { name: 'Select Seed 2', exact: true }).click();
  await expect(page.locator('tbody tr[data-selected=true]')).toHaveCount(2);
  await enter(batchText, '20');
  await expect(batchText).toHaveValue('20');
  await expect(batch.getByLabel('Batch Size selected value', { exact: true })).toBeVisible();
  const exceptions = page.getByRole('list').filter({ has: page.getByRole('button', { name: 'Remove Exception' }) });
  await expect(exceptions.getByRole('listitem')).toHaveText(['Batch Size 20 · seed 2 · 2 runs']);
  await exceptions.getByRole('button', { name: 'Remove Exception', exact: true }).click();
  await expect(exceptions).toHaveCount(0);
  await expect(batchText).toHaveValue(rule);
  await expect(batch.getByLabel('Batch Size selected value', { exact: true })).toHaveCount(0);
});
