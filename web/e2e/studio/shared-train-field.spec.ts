import { expect, test } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

test('Training and Runs edit the same saved values and project the selected source', async ({ page, request }) => {
  test.setTimeout(55_000);
  const { snapshot, projectPath } = await seedManagedProject(request, (await fixture('files')).slice(0, 3));
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
    expect(projection.window_map?.problems).toEqual(projection.rows.map(({ key }) => expect.objectContaining({
      code: 'cannot_run', path: '/trial_documents', row: key.row,
    })));
    return projection;
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await expect(rateText).toHaveValue('0.001');
  await enter(rateText, 'geomspace(1e-4, 1e-2, 5)');
  await expect(rate.getByRole('alert')).toContainText('writes more than one setting');
  await rateText.press('Escape');
  await enter(rateText, '1e-3, 2e-3, 3e-3');
  await expect(card.getByRole('button', { name: 'Sweep learning-rate' })).toHaveText('learning-rate 3');
  const before = (await plan())!.rows;
  const axes = (await saved()).project.steps?.train.sweep?.axes;
  const initial = axes?.['learning-rate'];
  expect(initial).toHaveProperty('values');
  const values = (initial as { values: Record<string, { patches: Record<string, { value: number }> }> }).values;
  for (const entry of Object.values(values)) {
    const patches = entry.patches;
    expect(patches['training_config.learning_rate'].value).toBe(patches['method_payload.payload.optimizer.params.learning_rate'].value);
  }
  const second = rate.getByRole('textbox', { name: 'Learning Rate value 2', exact: true });
  await enter(second, '0.0025');
  await expect.poll(async () => (await plan())?.rows.filter((row, index) => row.request_id !== before[index].request_id).length).toBe(1);
  const after = (await plan())!.rows;
  expect(after.map(({ key }) => key)).toEqual(before.map(({ key }) => key));

  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const selectValue = rate.getByRole('button', { name: 'Select Learning Rate 0.0025', exact: true });
  await selectValue.click();
  await expect(selectValue).toHaveAttribute('aria-pressed', 'true');
  await expect(rate.getByLabel('Learning Rate selected value 0.0025', { exact: true })).toBeVisible();
  const batchText = batches.getByRole('textbox', { name: 'N Batches', exact: true });
  const initialBatches = await batchText.inputValue();
  const override = String(Number(initialBatches) + 1);
  await enter(batchText, override);
  await expect(batches.getByLabel(`N Batches selected value ${override}`, { exact: true })).toBeVisible();
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
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => (await saved()).project.steps?.train.sweep?.exceptions ?? {}).toEqual({});
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await enter(clipText, '0.75');
  await expect.poll(async () => JSON.stringify((await saved()).project.steps?.train.base)).toContain('"grad_clip":0.75');
  await enter(clipText, 'geomspace(0.1, 1, 3)');
  await expect(card.getByRole('button', { name: 'Sweep grad-clip' })).toHaveText('grad-clip 3');
  await expect(clip.getByLabel('Spacing')).toHaveValue('log');
  await page.reload();
  await expect(clipText).toHaveValue('geomspace(0.1, 1, 3)');
  await expect(clip.getByLabel('Count')).toHaveValue('3');
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await expect(rateText).toHaveValue('0.001, 0.0025, 0.003');
  await expect(page.getByRole('dialog')).toHaveCount(0);
});
