import { expect, test } from './fixtures/studio';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { appendManagedFiles, enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { pick } from './fixtures/controls';
import { chooseStepKind } from './fixtures/studioKeyboard';

test.use({ viewport: { width: 1440, height: 900 } });

test('a continuation started from its Model keeps its definition and continues again, and a step based on a row shows its refusal', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '2');
  const train = files.at(-1).step;
  train.sweep = { rows: { slow: {}, fast: { patches: { 'settings.optimizer.learning_rate.value': { value: 0.003 } } } } };
  const model = files.find((file: { document?: { schema: string } }) => file.document?.schema === 'feedbax.spec.model').id;
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  async function saved() {
    const response = await request.get(projectPath);
    expect(response.ok()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  }
  async function requests(step: string) {
    const { revision } = await saved();
    const response = await request.get(`${projectPath}/train-runs?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    const { rows } = await response.json() as { rows: { key: { step: string }; request_id: string }[] };
    return rows.filter(({ key }) => key.step === step).map(({ request_id }) => request_id).sort();
  }

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await page.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await expect(mapStep(page, 'train-2')).toHaveAttribute('aria-pressed', 'true');
  const continuation = (await saved()).project.steps!['train-2'];
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const start = inspector.getByRole('combobox', { name: 'Start', exact: true });
  await expect(start).toHaveText('train');

  await pick(start, model);
  await expect.poll(async () => 'inline' in (await saved()).project.steps!['train-2'].base).toBe(true);
  expect(await requests('train-2')).toEqual(await requests('train'));
  await expect(start).toHaveText(model);
  await expect(inspector.getByRole('textbox', { name: 'Batches', exact: true })).toHaveValue('2');

  await pick(start, 'train');
  await expect.poll(async () => (await saved()).project.steps!['train-2'].base).toEqual({ step: 'train' });
  const restored = (await saved()).project.steps!['train-2'];
  expect(restored.inputs).toEqual(continuation.inputs);
  expect(restored.sweep).toBeUndefined();
  expect(restored.deltas?.flatMap(({ patches }) => patches)).toEqual(continuation.deltas?.flatMap(({ patches }) => patches));
  await expect(inspector.getByRole('textbox', { name: 'Batches', exact: true })).toHaveValue('2');

  const { snapshot: based } = await appendManagedFiles(request, await saved(), [
    { kind: 'step', id: 'row-based', step: { kind: 'train', base: { step: 'train', row: 'slow' } } },
  ]);
  await page.reload();
  await mapStep(page, 'row-based').click();
  const refusal = page.locator('[aria-description="Train step \'row-based\' takes its definition from row \'slow\' of step \'train\'"]');
  await expect(refusal.first()).toBeVisible();
  await expect(inspector.getByRole('combobox', { name: 'Start', exact: true })).toHaveCount(0);
  expect(based.project.steps!['row-based']).toBeDefined();
});

// Catches a Start that offers only another step's last checkpoint: Start is the one input editor, so At 500 is a rule
// it takes, and the plan then reads update 500 of the train step's run.
test('a Train step started at update 500 of another plans a continuation from that checkpoint', async ({ page, request }) => {
  const files = await fixture('ordinary-files', '600');
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  async function plan() {
    const { revision } = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
    const response = await request.get(`${projectPath}/train-runs?window_map=true&revision=${revision}`);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  }
  const requestOf = async () => (await plan()).rows.find(({ key }) => key.step === 'train-2')?.request_id;

  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await page.getByRole('button', { name: 'New Step train', exact: true }).click();
  await chooseStepKind(page, 'Train');
  await expect(mapStep(page, 'train-2')).toHaveAttribute('aria-pressed', 'true');
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(inspector.getByRole('combobox', { name: 'Start', exact: true })).toHaveText('train');
  const last = await requestOf();

  await pick(inspector.getByRole('combobox', { name: 'Checkpoint', exact: true }), 'At');
  const saving = page.waitForResponse((response) => response.request().method() === 'PATCH' && new URL(response.url()).pathname === projectPath);
  await enter(inspector.getByRole('textbox', { name: 'Update', exact: true }), '500');
  expect((await saving).ok()).toBe(true);
  const saved = await (await request.get(projectPath)).json() as StudioProjectSnapshot;
  expect(saved.project.steps!['train-2'].inputs!.checkpoint).toEqual({ step: 'train', checkpoint: { at: 500 } });
  const planned = await plan();
  expect(planned.window_map!.problems.filter(({ step }) => step === 'train-2')).toEqual([]);
  expect(planned.window_map!.steps.find(({ step_id }) => step_id === 'train-2')!.inputs)
    .toEqual([expect.objectContaining({ source_step_id: 'train', checkpoint: { at: 500 } })]);
  expect(await requestOf()).not.toEqual(last);
  await expect(inspector.getByRole('textbox', { name: 'Update', exact: true })).toHaveValue('500');
});
