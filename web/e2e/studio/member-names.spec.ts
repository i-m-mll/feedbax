import { expect, test, jobTimeout } from './fixtures/studio';
import type { Page } from '@playwright/test';
import { analysisDocumentSchemaId, type StudioJobSubmission, type StudioProjectSnapshot, type StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { activityJob, documentPicker, problemGroup } from './fixtures/members';

test.use({ viewport: { width: 1440, height: 900 } });

async function runStep(page: Page, step: string, runsPath: string) {
  await mapStep(page, step).click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  await runs.getByRole('table', { name: 'Runs' }).getByRole('button').first().click();
  const run = runs.getByRole('button', { name: 'Run', exact: true });
  await expect(run).toBeEnabled();
  const admission = page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === runsPath);
  await run.click();
  const response = await admission;
  expect(response.status(), await response.text()).toBe(202);
  const [item] = (await response.json() as StudioJobSubmission).items;
  expect(item?.disposition).toBe('admitted');
  expect(item?.job_id).toBeTruthy();
  await expect(runs.getByRole('table', { name: 'Runs' }).getByText('Done', { exact: true })).toBeVisible({ timeout: jobTimeout });
  return item.job_id!;
}

test('a member reads the same in the map, Inspector, picker, Problems and Activity', { tag: '@round' }, async ({ page, request }) => {
  const [model, task, objective, train, sibling] = await fixture('history-files');
  const siblingLabel = 'Reach sibling';
  sibling.step.label = siblingLabel;
  const brokenLabel = 'Broken reach';
  const broken = { kind: 'step', id: 'broken_reach', step: { kind: 'train', label: brokenLabel, base: { inline: {
    trial_documents: train.step.base.inline.trial_documents,
  } } } };
  const { snapshot, projectPath, configuration } = await seedManagedProject(request, [model, task, objective, train, sibling, broken]);
  const runsPath = `${projectPath}/train-runs`;
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('heading', { level: 3 });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  const labels = { train: train.step.label, sibling: siblingLabel, broken_reach: brokenLabel };
  const cardLabel = (step: string) => page.locator(`[data-map-step="${step}"] .studio-map-card__head`);
  async function stepNames(names: Record<string, string>) {
    for (const [id, label] of Object.entries(names)) {
      await expect(cardLabel(id)).toHaveText(label);
      await mapStep(page, id).click();
      await expect(inspector).toHaveText(label);
    }
  }
  await stepNames(labels);
  const jobs = new Map<string, string>();
  for (const id of ['train', 'sibling']) jobs.set(id, await runStep(page, id, runsPath));

  const header = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header');
  const problems = page.getByRole('dialog', { name: 'Problems' });
  const activity = page.getByRole('region', { name: 'Activity', exact: true })
    .getByRole('region', { name: configuration.project, exact: true });
  async function otherNames(names: Record<string, string>, modelLabel: string) {
    await mapStep(page, 'train').click();
    await page.getByRole('tab', { name: 'Model', exact: true }).click();
    await expect(documentPicker(header, model.id)).toHaveText(modelLabel);
    await page.getByRole('tab', { name: 'Task', exact: true }).click();
    await expect(documentPicker(header, task.id)).toHaveText(task.document.label);

    await page.getByRole('button', { name: /^Problems \d+$/ }).click();
    await expect(problemGroup(problems, 'step:broken_reach').locator('h2')).toHaveText(names.broken_reach);
    await page.keyboard.press('Escape');

    const activityToggle = page.getByRole('button', { name: /^Activity( \d+)?$/ });
    await activityToggle.click();
    for (const step of ['train', 'sibling']) {
      await expect(activityJob(activity, jobs.get(step)!).locator('button > span:last-child')).toHaveText(names[step]);
    }
    await activityToggle.click();
  }
  await otherNames(labels, model.document.label);

  const renamed = { ...labels, train: 'Trained reach', broken_reach: 'Incomplete reach' };
  for (const id of ['train', 'broken_reach'] as const) {
    const step = mapStep(page, id);
    await step.click();
    await step.press('F2');
    await enter(page.locator(`[data-map-step="${id}"]`).getByRole('textbox', { name: /^Rename / }), renamed[id]);
    await expect(cardLabel(id)).toHaveText(renamed[id]);
  }
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await documentPicker(header, model.id).click();
  await page.getByRole('menu', { name: 'Model Document' }).getByRole('menuitem', { name: 'Rename', exact: true }).click();
  const modelLabel = 'Point mass controller';
  await enter(header.getByRole('textbox'), modelLabel);
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  await stepNames(renamed);
  await otherNames(renamed, modelLabel);
});

// Catches a signal rename that leaves the Model input it fed unfed, an Outline that does not say what feeds it, and an
// Inputs card that keeps the old name or lists the renamed signal again as an unused port.
test('renaming a Task signal updates its readers and keeps the Train request', { tag: '@round' }, async ({ page, request }) => {
  const [model, task, objective, train] = await fixture('ordinary-files', '2');
  objective.document.payload.terms.push({
    id: 'target-size',
    name: 'Target size',
    measure: { kind: 'call', function: 'sqnorm', args: [{ kind: 'name', reference: 'target_pos' }] },
    measure_type: { base: 'real', dimensions: { length: 2 } },
  });
  const evaluate = { kind: 'step', id: 'evaluate', step: {
    kind: 'evaluate', label: 'Evaluation', base: { inline: {
      evaluation_type: 'feedbax.rollout',
      params: { seed: 731, trial_set: { kind: 'every_condition' }, trial_policy: 'paired', variation: [] },
    } },
    inputs: { subject: { step: train.id, checkpoint: 'last' } },
  } };
  const analysis = { kind: 'document', id: 'alignment-analysis', document: {
    schema: analysisDocumentSchemaId, label: 'Alignment', payload: {
      inputs: { states: { kind: 'dataset' } },
      operations: { align: {
        label: 'Align', operation: 'feedbax.aligned_velocity',
        inputs: { states: { input: 'states' } },
        params: { target: '/v1/task/output/target_pos' },
      } },
    },
  } };
  const analyze = { kind: 'step', id: 'analyze', step: {
    kind: 'analyze', label: 'Analysis', base: { document: analysis.id },
    inputs: { states: { step: evaluate.id } },
  } };
  const { snapshot, projectPath } = await seedManagedProject(
    request, [model, task, objective, analysis, train, evaluate, analyze],
  );
  async function requestId() {
    const saved = await request.get(projectPath);
    expect(saved.ok(), await saved.text()).toBe(true);
    const { revision } = await saved.json() as StudioProjectSnapshot;
    const plan = await request.get(`${projectPath}/train-runs?window_map=true&revision=${revision}`);
    expect(plan.ok(), await plan.text()).toBe(true);
    const { rows } = await plan.json() as StudioSavedTrainRuns;
    return rows.find(({ key }) => key.step === 'train')?.request_id;
  }
  const before = await requestId();
  expect(before).toBeTruthy();
  await page.goto(`/?project=${snapshot.project_id}`);
  await runStep(page, train.id, `${projectPath}/train-runs`);
  await runStep(page, evaluate.id, `${projectPath}/train-runs`);
  await mapStep(page, train.id).click();
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  const grid = page.locator('[data-slot="primary"]').getByRole('tabpanel').getByRole('table', { name: 'Epochs', exact: true });
  await enter(grid.getByRole('textbox', { name: 'Rename target position', exact: true }), 'aim');
  await expect(grid.getByRole('textbox', { name: 'Rename aim', exact: true })).toBeVisible();
  const input = page.getByRole('complementary', { name: 'Details Drawer', exact: true })
    .getByRole('tree', { name: 'Model Inputs', exact: true }).getByRole('treeitem').filter({ hasText: 'target_pos' });
  await expect(input).toContainText('aim');
  await expect(input.locator('[data-status-mark]')).toHaveCount(0);

  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const inputs = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Model', exact: true })
    .locator('[data-id="@inputs"] .studio-model-port > span:first-of-type');
  await expect(inputs.filter({ hasText: /^aim$/ })).toHaveCount(1);
  await expect(inputs.filter({ hasText: /^target_pos$/ })).toHaveCount(0);
  expect(await requestId()).toBe(before);

  await page.getByRole('tab', { name: 'Objective', exact: true }).click();
  const terms = page.getByRole('table', { name: 'Objective Terms', exact: true });
  const targetTerm = terms.getByRole('row').filter({ has: page.getByRole('textbox', { name: 'Rename Target size', exact: true }) });
  await expect(targetTerm.getByRole('textbox', { name: 'Measure', exact: true })).toHaveValue('sqnorm(aim)');

  await mapStep(page, analyze.id).click();
  await page.getByRole('tab', { name: 'Analysis', exact: true }).click();
  await page.getByRole('group', { name: 'Select Align', exact: true }).click();
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const target = inspector.getByRole('combobox', { name: 'Target', exact: true });
  await expect(target).toHaveText('aim');
  await target.click();
  await expect(page.getByRole('option', { name: 'aim', exact: true })).toBeVisible();
});
