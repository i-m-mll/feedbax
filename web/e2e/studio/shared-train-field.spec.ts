import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';

test.use({ viewport: { width: 1440, height: 900 } });

test('the inspector edits the saved values for the step and its selected runs, and projects the selected source', async ({ page, request }) => {
  test.setTimeout(55_000);
  const files = await fixture('ordinary-files', '2');
  // Train offers the settings a project authors, so the project authors Grad Clip to edit it.
  files.at(-1).step.base.inline.settings.grad_clip = 1;
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const card = page.getByRole('article', { name: 'Train train', exact: true });
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  const clip = page.getByRole('region', { name: 'Grad Clip field', exact: true });
  const batches = page.getByRole('region', { name: 'Batches field', exact: true });
  const rateText = rate.getByRole('textbox', { name: 'Learning Rate Value', exact: true });
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
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await expect(rateText).toHaveValue('0.001');
  const method = page.getByRole('combobox', { name: 'Method', exact: true });
  await expect(method).toHaveValue('Supervised gradient');
  await method.click();
  await expect(page.getByRole('option', { name: /^Supervised gradient\s*feedbax\.builtins$/ })).toBeVisible();
  await page.keyboard.press('Escape');
  const optimizer = page.getByRole('region', { name: 'Optimizer', exact: true });
  await expect(optimizer.getByRole('region', { name: 'Learning Rate field', exact: true })).toBeVisible();
  await rate.getByRole('button', { name: 'Learning Rate kind: Value', exact: true }).click();
  await expect(page.getByRole('menuitemradio', { name: 'Sweep', exact: true })).toBeEnabled();
  await page.keyboard.press('Escape');
  await pick(optimizer.getByRole('combobox', { name: 'Optimizer', exact: true }), 'SGD');
  await expect.poll(async () => {
    const base = (await saved()).project.steps?.train.base;
    return base && 'inline' in base ? (base.inline as { settings: { optimizer: { type: string } } }).settings.optimizer.type : undefined;
  }).toBe('sgd');
  await enter(rateText, 'geomspace(1e-4, 1e-2, 5)');
  await expect(card.getByRole('button', { name: 'Sweep learning-rate 5', exact: true })).toBeVisible();
  await enter(rateText, '1e-3, 2e-3, 3e-3');
  await expect(card.getByRole('button', { name: 'Sweep learning-rate 3', exact: true })).toBeVisible();
  const before = (await plan())!.rows;
  const axes = (await saved()).project.steps?.train.sweep?.axes;
  const initial = axes?.['learning-rate'];
  expect(initial).toHaveProperty('values');
  const values = (initial as { values: Record<string, { label?: string; patches: Record<string, { value: number }> }> }).values;
  expect(Object.values(values).map(({ label }) => label)).toEqual([undefined, undefined, undefined]);
  for (const entry of Object.values(values)) {
    const patches = entry.patches;
    expect(Object.keys(patches)).toEqual(['settings.optimizer.learning_rate.value']);
  }
  await enter(rateText, '0.001, 0.0025, 0.003');
  await expect.poll(async () => (await plan())?.rows.filter((row, index) => row.request_id !== before[index].request_id).length).toBe(1);
  const after = (await plan())!.rows;
  expect(after.map(({ key }) => key)).toEqual(before.map(({ key }) => key));

  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const selectValue = rate.getByRole('button', { name: 'Select Learning Rate 0.0025', exact: true });
  await selectValue.click();
  await expect(selectValue).toHaveAttribute('aria-pressed', 'true');
  await expect(rate.getByLabel('Changed', { exact: true })).toHaveAccessibleDescription('Selected Value · 0.0025');
  const batchText = batches.getByRole('textbox', { name: 'Batches', exact: true });
  const initialBatches = await batchText.inputValue();
  const override = String(Number(initialBatches) + 1);
  await enter(batchText, override);
  await expect(batches.getByLabel('Changed', { exact: true })).toHaveAccessibleDescription(`Changed from ${initialBatches}`);
  await expect(batchText).toHaveValue(override);
  await expect(rateText).toBeVisible();
  // An edit for exactly one sweep value's runs is stored on that value, not as an exception.
  async function storedOnValue() {
    const sweep = (await saved()).project.steps?.train.sweep;
    const axis = sweep?.axes?.['learning-rate'] as { values?: Record<string, { patches: Record<string, { value?: unknown }> }> } | undefined;
    const values = Object.values(axis?.values ?? {});
    return {
      exceptions: Object.keys(sweep?.exceptions ?? {}).length,
      batches: values.flatMap(({ patches }) => Object.entries(patches)
        .filter(([path]) => path.endsWith('n_batches')).map(([, patch]) => patch.value)),
    };
  }
  await expect.poll(storedOnValue).toEqual({ exceptions: 0, batches: [Number(override)] });
  const scoped = (await plan())!.rows;
  expect(scoped.filter((row, index) => row.request_id !== after[index].request_id)).toHaveLength(1);
  // A draft is saved for the runs it was typed for when the selection moves away from them.
  await batchText.fill('77');
  await rate.getByRole('button', { name: 'Select Learning Rate 0.001', exact: true }).click({ modifiers: ['Shift'] });
  await expect.poll(storedOnValue).toEqual({ exceptions: 0, batches: [77] });
  await expect(batchText).toHaveValue('');
  await expect(batches.locator('output')).toHaveCount(0);
  await rate.getByRole('button', { name: 'Select Learning Rate 0.001', exact: true }).click();
  await expect(batchText).toHaveValue(initialBatches);
  await expect(batches.locator('output')).toHaveCount(0);

  await page.getByRole('tabpanel', { name: 'Runs', exact: true }).focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(storedOnValue).toEqual({ exceptions: 0, batches: [Number(override)] });
  await page.keyboard.press('Meta+Z');
  await expect.poll(async () => (await plan())?.rows.map(({ request_id }) => request_id)).toEqual(after.map(({ request_id }) => request_id));
  await page.keyboard.press('Meta+Shift+Z');
  await expect.poll(async () => (await plan())?.rows.map(({ request_id }) => request_id)).toEqual(scoped.map(({ request_id }) => request_id));
  await selectValue.click();
  await expect(batchText).toHaveValue(override);
  await page.getByRole('tabpanel', { name: 'Runs', exact: true }).focus();
  await page.keyboard.press('Meta+Z');
  await expect.poll(storedOnValue).toEqual({ exceptions: 0, batches: [] });
  await expect(batchText).toHaveValue(initialBatches);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await enter(clipText, '0.75');
  await expect.poll(async () => JSON.stringify((await saved()).project.steps?.train.base)).toContain('"grad_clip":0.75');
  await enter(clipText, 'geomspace(0.1, 1, 3)');
  await expect(card.getByRole('button', { name: 'Sweep grad-clip 3', exact: true })).toBeVisible();
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
  await expect(batch.getByLabel('Changed', { exact: true })).toHaveAccessibleDescription(`Changed from ${rule}`);
  const exceptions = page.getByRole('list').filter({ has: page.getByRole('button', { name: 'Remove Exception' }) });
  const exception = exceptions.getByRole('listitem');
  await expect(exception).toHaveCount(1);
  await expect(exception).toContainText('Batch Size 20');
  await expect(exception.getByRole('combobox', { name: 'Runs', exact: true })).toHaveValue('seed=2');
  await expect(exception).toContainText('2/8');
  await exceptions.getByRole('button', { name: 'Remove Exception', exact: true }).click();
  await expect(exceptions).toHaveCount(0);
  await expect(batchText).toHaveValue(rule);
  await expect(batch.getByLabel('Changed', { exact: true })).toHaveCount(0);
});

test('choosing an optimizer keeps the inspector still, and reveals only the parameters it brings in', async ({ page, request }) => {
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  await page.setViewportSize({ width: 1440, height: 640 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const optimizer = page.getByRole('region', { name: 'Optimizer', exact: true });
  const choice = optimizer.getByRole('combobox', { name: 'Optimizer', exact: true });
  await expect(choice).toHaveValue('Adam');
  /** The Inspector's scroll, its clear area (inside the edge fades), and the Optimizer field's label and picker. */
  const view = () => choice.evaluate((control) => {
    const scroller = control.closest('.studio-scroll-fade')!;
    const { scrollPaddingTop, scrollPaddingBottom } = getComputedStyle(scroller);
    const { top } = scroller.getBoundingClientRect();
    return {
      scrollTop: scroller.scrollTop,
      max: scroller.scrollHeight - scroller.clientHeight,
      clearTop: top + parseFloat(scrollPaddingTop),
      clearBottom: top + scroller.clientHeight - parseFloat(scrollPaddingBottom),
      label: document.getElementById(control.getAttribute('aria-labelledby')!)!.getBoundingClientRect().top,
      picker: control.getBoundingClientRect().bottom,
    };
  });
  const scrollBy = (delta: number) => choice.evaluate((control, by) => { control.closest('.studio-scroll-fade')!.scrollTop += by; }, delta);
  const placePicker = async () => { const { picker, clearBottom } = await view(); await scrollBy(picker - clearBottom); };
  const bottom = (field: string) => page.getByRole('region', { name: `${field} field`, exact: true })
    .evaluate((element) => element.getBoundingClientRect().bottom);

  // SGD has fewer parameters than Adam, and they fit below the picker: the Inspector stays where it is.
  await scrollBy((await view()).label - (await view()).clearTop);
  const end = await view();
  await pick(choice, 'SGD');
  await expect(page.getByRole('region', { name: 'B1 field', exact: true })).toHaveCount(0);
  const removed = await view();
  expect(Math.abs(removed.scrollTop - Math.min(end.scrollTop, removed.max))).toBeLessThanOrEqual(1);
  expect(removed.scrollTop).toBeGreaterThan(0);

  // Fromage's Min Norm fits above the picker: it ends at the bottom of the clear area, and the picker stays in view.
  await placePicker();
  await pick(choice, 'Fromage');
  await expect.poll(async () => Math.abs(await bottom('Min Norm') - (await view()).clearBottom)).toBeLessThanOrEqual(1);
  expect((await view()).label).toBeGreaterThan((await view()).clearTop);

  // Adam's parameters are taller than the Inspector: the field's label ends at the top of the clear area.
  await placePicker();
  await pick(choice, 'Adam');
  await expect(page.getByRole('region', { name: 'Nesterov field', exact: true })).toBeAttached();
  await expect.poll(async () => { const { label, clearTop } = await view(); return Math.abs(label - clearTop); }).toBeLessThanOrEqual(1);
  const revealed = await view();
  expect(await bottom('Nesterov')).toBeGreaterThan(revealed.clearBottom);
  expect(revealed.scrollTop).toBeLessThan(revealed.max);
});
