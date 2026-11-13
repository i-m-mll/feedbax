import { expect, test } from '@playwright/test';
import type {
  StudioJobSubmission, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { enter, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ channel: 'chrome', trace: 'off', viewport: { width: 1440, height: 900 } });

test('saves and runs managed optimizer and attachment sweeps', async ({ page, request }, info) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, []);
  const id = snapshot.project_id;
  const url = '/?project=' + encodeURIComponent(id);
  const runsPath = projectPath + '/train-runs';
  await runProjectFixture('web.e2e.studio.fixtures.stored_model', info.config.metadata.home, id, 'managed-fields');
  const save = async () => {
    const response = await request.get(projectPath + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return ((await response.json()) as StudioProjectDocument).snapshot;
  };
  const plan = async (revision: string) => {
    const response = await request.get(runsPath + '?window_map=true&revision=' + revision);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  await page.goto(url);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  const training = page.getByRole('tabpanel', { name: 'Training', exact: true });
  const rate = training.getByRole('textbox', { name: 'Learning Rate', exact: true });
  await expect(rate).toHaveValue('0.001');
  await enter(rate, 'geomspace(0.001, 0.004, 2)');
  await expect.poll(async () => (await save()).project.steps?.train.sweep?.axes?.['learning-rate']).toMatchObject({
    path: 'method_payload.payload.optimizer.params.learning_rate',
    range: { start: 0.001, stop: 0.004, count: 2, scale: 'log' },
  });

  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await model.locator('[data-id="wrapper"] header').dblclick();
  await model.locator('[data-id="controller"] header').dblclick();
  await model.getByRole('button', { name: 'Attachment Scale', exact: true }).click();
  const factor = inspector.getByRole('textbox', { name: 'Factor', exact: true });
  await expect(factor).toHaveValue('1.0');
  await enter(factor, 'linspace(0.5, 1, 2)');
  const factorAxis = (saved: StudioProjectSnapshot) => saved.project.steps?.train.sweep?.axes?.factor;
  await expect.poll(async () => factorAxis(await save())).toMatchObject({
    path: expect.stringContaining('"id":"training-scale"'),
    range: { start: 0.5, stop: 1, count: 2 },
  });

  const saved = await save();
  const base = saved.project.steps?.train.base;
  if (!base || !('inline' in base)) throw new Error('Managed Train base is unavailable');
  expect(base.inline).toHaveProperty('trial_documents');
  expect(base.inline).not.toHaveProperty('training_config.learning_rate');
  const before = await plan(saved.revision);
  expect(before.window_map?.problems).toEqual([]);
  expect(before.rows).toHaveLength(4);
  expect(new Set(before.rows.map(({ request_id }) => request_id)).size).toBe(4);
  await page.reload();
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Training', exact: true }).click();
  await expect(rate).toHaveValue('geomspace(0.001, 0.004, 2)');
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await model.getByRole('button', { name: 'Attachment Scale', exact: true }).click();
  await expect(factor).toHaveValue('linspace(0.5, 1, 2)');
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const first = before.rows[0];
  await runs.getByRole('table', { name: 'Runs' }).getByRole('button').first().click();
  const response = page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === runsPath);
  await runs.getByRole('button', { name: 'Run', exact: true }).click();
  const admitted = await response;
  expect(admitted.status(), await admitted.text()).toBe(202);
  const { items } = await admitted.json() as StudioJobSubmission;
  expect(items).toEqual([expect.objectContaining({
    run_key: JSON.stringify([first.key.variant, first.key.step, first.key.row]),
    disposition: 'admitted', job_id: expect.any(String),
  })]);
  await expect.poll(async () => (await plan(saved.revision)).rows.find(({ request_id }) =>
    request_id === first.request_id)?.attempts[0]?.status, { timeout: 22_000 }).toBe('succeeded');
  const completed = (await plan(saved.revision)).rows.find(({ request_id }) => request_id === first.request_id)!;
  expect(completed.attempts).toEqual([expect.objectContaining({
    job_id: items[0].job_id, status: 'succeeded',
    progress: { completed_updates: 2, total_updates: 2 },
    successful_receipt: expect.objectContaining({ manifest_sha256: expect.any(String) }),
  })]);
  console.info('Managed Field execution', JSON.stringify({
    revision: saved.revision, request: completed.request_id, row: completed.key,
    rateAxis: saved.project.steps?.train.sweep?.axes?.['learning-rate'],
    factorAxis: factorAxis(saved), attempt: completed.attempts[0],
  }));
});
