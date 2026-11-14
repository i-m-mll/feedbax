import { expect, test, type Page } from '@playwright/test';
import type {
  StudioActivityJob, StudioJobEventHistory, StudioJobSnapshot, StudioJobSubmission,
  StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainKey, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import {
  appendManagedFiles, authorLearningRateSweep, fixture, mapStep, openResultsSlot,
  runProjectFixture, seedManagedProject,
} from './fixtures/canonicalTrain';

const primary = (page: Page) => page.locator('[data-studio-focus-region="primary"]');
const runs = (page: Page) => primary(page).getByRole('table', { name: 'Runs', exact: true });
const runButton = (page: Page) => primary(page).getByRole('button', { name: 'Run', exact: true });

function responseFor(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path,
  );
}

async function select(page: Page, step: string) {
  await mapStep(page, step).click();
  await primary(page).focus();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
}

async function admit(page: Page, projectPath: string) {
  const posted = responseFor(page, projectPath + '/train-runs');
  await runButton(page).click();
  const response = await posted;
  expect(response.status(), await response.text()).toBe(202);
  const { items } = await response.json() as StudioJobSubmission;
  expect(items).toHaveLength(1);
  expect(items[0].disposition).toBe('admitted');
  return items[0];
}

test('M1 starts an ordinary run, watches loss, and resumes the checkpoint from Activity',
  async ({ page, request }) => {
    test.setTimeout(50_000);
    let creates = 0;
    page.on('request', (request) => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/projects') creates += 1;
    });
    const creating = responseFor(page, '/api/projects');
    await page.goto('/');
    const created = await creating;
    expect(created.status()).toBe(201);
    const { snapshot } = await created.json() as StudioProjectDocument;
    const id = snapshot.project_id;
    await expect(page).toHaveURL(new RegExp('project=' + id + '$'));
    const { projectPath } = await appendManagedFiles(request, snapshot, await fixture('ordinary-files', '1200'));
    await page.reload();
    await select(page, 'train');
    expect(creates).toBe(1);
    const item = await admit(page, projectPath);
    const jobId = item.job_id!;
    await runs(page).getByRole('button').first().click();
    await openResultsSlot(page);
    const plot = page.getByRole('img', { name: 'Loss', exact: true });
    await expect(plot.locator('line title').filter({ hasText: /samples?/ }).first()).toBeAttached({ timeout: 14_000 });

    await page.getByRole('button', { name: 'Activity', exact: true }).click();
    const activity = page.getByRole('region', { name: 'Activity', exact: true });
    const attempt = activity.getByRole('list', { name: 'Runs' }).getByRole('listitem').filter({ hasText: id });
    const stopping = responseFor(page, '/api/jobs/stop');
    await attempt.getByRole('button', { name: 'Stop', exact: true }).click();
    expect((await stopping).status()).toBe(202);
    const resume = attempt.getByRole('button', { name: 'Resume', exact: true });
    await expect(resume).toBeEnabled({ timeout: 10_000 });
    const stopped = await request.get('/api/jobs', { params: { project: id } });
    expect(stopped.ok()).toBe(true);
    const snapshotOfJobs = await stopped.json() as StudioJobSnapshot;
    const parent = snapshotOfJobs.jobs.find((job) => job.job_id === jobId);
    expect(parent).toMatchObject({ status: 'stopped', can_resume: true });

    const resuming = responseFor(page, `/api/jobs/${jobId}/resume`);
    await resume.click();
    const resumed = await resuming;
    expect(resumed.status(), await resumed.text()).toBe(202);
    const continuation = await resumed.json() as StudioActivityJob;
    expect(continuation).toMatchObject({
      continues: jobId, request_id: parent!.request_id, run_key: item.run_key,
    });
    expect(continuation.job_id).not.toBe(jobId);

    const replayed = page.waitForResponse((response) => {
      const url = new URL(response.url());
      return url.pathname === '/api/events/history' && url.searchParams.get('project') === id;
    });
    await page.reload();
    await select(page, 'train');
    const replay = await replayed;
    expect(replay.ok(), await replay.text()).toBe(true);
    const history = await replay.json() as StudioJobEventHistory;
    expect(history.events.some((event) => event.job_id === jobId && event.kind === 'progress')).toBe(true);
    await expect(runs(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });
  },
);

test('Activity shows the original CLI and Studio clients for a saved Train request', async ({ page, request }, info) => {
  test.setTimeout(45_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const project = snapshot.project_id;
  const label = 'Batch review';
  await page.goto('/?project=' + project);
  await select(page, 'train');
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const entries = activity.getByRole('list', { name: 'Runs', exact: true }).getByRole('listitem').filter({ hasText: project });

  await runProjectFixture('web.e2e.studio.fixtures.canonical_train',
    'run-saved-cli', String(info.config.metadata.home), project, label);
  const cli = entries.filter({ has: page.getByText('CLI', { exact: true }) });
  await expect(cli.getByText(label, { exact: true })).toBeVisible();
  await expect(cli.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const studioJob = await admit(page, projectPath);
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const studio = entries.filter({ has: page.getByText('Studio', { exact: true }) });
  await expect(studio.getByText('Client', { exact: true })).toBeVisible();
  await expect(studio.getByText('Label', { exact: true })).toHaveCount(0);
  await expect(cli.getByText(label, { exact: true })).toBeVisible();
  await expect(studio.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });

  const response = await request.get('/api/jobs', { params: { project } });
  expect(response.ok(), await response.text()).toBe(true);
  const { jobs } = await response.json() as StudioJobSnapshot;
  expect(jobs).toHaveLength(2);
  expect(jobs.find(({ job_id }) => job_id === studioJob.job_id)?.actor).toEqual({ client: 'studio' });
  expect(jobs.find(({ actor }) => actor?.client === 'cli')?.actor).toEqual({ client: 'cli', label });
  expect(new Set(jobs.map(({ request_id }) => request_id)).size).toBe(1);
  await info.attach('activity-actors', {
    body: JSON.stringify(jobs.map(({ job_id, request_id, actor }) => ({ job_id, request_id, actor }))),
    contentType: 'application/json',
  });
});

test('M1 runs only the selected row of a sweep authored in Studio', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '30'));
  await page.goto('/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  await authorLearningRateSweep(rate);
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const picked = runs(page).getByRole('button', { name: /Medium/ });
  await picked.click();
  const submitted = await admit(page, projectPath);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(1);
  await expect(runs(page).getByText('Not run', { exact: true })).toHaveCount(2);
  const response = await request.get('/api/jobs', { params: { project: snapshot.project_id } });
  expect(response.ok()).toBe(true);
  const { jobs, cursor } = await response.json() as StudioJobSnapshot;
  expect(jobs).toHaveLength(1);
  expect(jobs[0]).toMatchObject({ run_key: submitted.run_key, job_id: submitted.job_id });
  const historyResponse = await request.get('/api/events/history', {
    params: { project: snapshot.project_id, through: cursor },
  });
  expect(historyResponse.ok()).toBe(true);
  const { events } = await historyResponse.json() as StudioJobEventHistory;
  const intervals = events
    .filter((event) => event.kind === 'progress' && event.job_id === jobs[0].job_id)
    .flatMap((event) => event.data.metric_interval ? [event.data.metric_interval] : []) as Array<{
      start_exclusive: number;
      end_inclusive: number;
      metrics: { train_loss: { count: number; min: number; mean: number; max: number } };
    }>;
  let end = 0;
  for (const interval of intervals) {
    const loss = interval.metrics.train_loss;
    expect(interval.start_exclusive).toBe(end);
    expect(loss.count).toBe(interval.end_inclusive - end);
    expect(loss.mean).toBeGreaterThanOrEqual(loss.min);
    expect(loss.mean).toBeLessThanOrEqual(loss.max);
    end = interval.end_inclusive;
  }
  expect(end).toBe(30);
});

test('M2 authors and widens a saved Evaluate subset without resubmitting completed rows', async ({ page, request }) => {
  test.setTimeout(58_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  const evaluation = 'web.e2e.studio.fixtures.canonical_evaluate';
  const files = JSON.parse(await runProjectFixture(evaluation, 'walkthrough-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const path = projectPath + '/train-runs';
  const mounted = '/?project=' + snapshot.project_id;
  const map = page.getByRole('region', { name: 'Pipeline Map', exact: true });
  const train = map.getByRole('article', { name: 'Train train', exact: true });
  const evaluated = map.getByRole('article', { name: 'Evaluate evaluate', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const steps = page.getByRole('complementary', { name: 'Map Drawer', exact: true });
  const readSnapshot = async () => {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioProjectSnapshot;
  };
  const readPlan = async () => {
    const { revision } = await readSnapshot();
    const response = await request.get(path, { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return await response.json() as StudioSavedTrainRuns;
  };
  const run = async (keys: StudioSavedTrainKey[]) => {
    const posted = responseFor(page, path);
    await runButton(page).click();
    const response = await posted;
    expect(response.status(), await response.text()).toBe(202);
    expect(response.request().postDataJSON().run_keys).toEqual(keys);
    const { items } = await response.json() as StudioJobSubmission;
    expect(items.map(({ disposition }) => disposition)).toEqual(keys.map(() => 'admitted'));
  };

  await page.goto(mounted);
  await select(page, 'train');
  await authorLearningRateSweep(drawer.getByRole('region', { name: 'Learning Rate field', exact: true }));
  const sweep = train.getByRole('button', { name: /^Sweep learning-rate / });
  await expect(sweep).toHaveText('learning-rate 3');
  const definition = await readPlan();
  const sourceRows = definition.window_map!.rows.filter(({ key }) => key.step === 'train');
  expect(sourceRows.map(({ coordinates }) => coordinates['learning-rate'].label)).toEqual(['Slow', 'Medium', 'Fast']);
  const ids = sourceRows.map(({ coordinates }) => coordinates['learning-rate'].value_id);
  const conditionIds = sourceRows[0].evaluation_conditions!.slice(0, 2);
  expect(conditionIds).toHaveLength(2);
  await run(sourceRows.map(({ key }) => key));
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(3, { timeout: 22_000 });

  await sweep.click();
  const value = (name: string) => train.getByRole('listitem').getByRole('button', {
    name: `Select learning-rate ${name}`, exact: true,
  });
  await value('Slow · 0.001').click();
  await value('Medium · 0.002').click({ modifiers: ['Shift'] });
  await expect(train).toContainText('2 of 3');
  const created = responseFor(page, projectPath + '/files');
  await train.getByRole('button', { name: 'New Step train', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step' }).getByRole('button', { name: 'Evaluate', exact: true }).click();
  const response = await created;
  expect(response.status(), await response.text()).toBe(201);
  const authored = (await response.json() as StudioProjectDocument).snapshot.project.steps.evaluate;
  expect(authored).toMatchObject({
    inputs: { subject: { step: 'train', checkpoint: 'last', where: { axes: { 'learning-rate': ids.slice(0, 2) } } } },
    sweep: { axes: { subject: { input: 'subject' } } },
  });
  const inputs = drawer.getByRole('region', { name: 'Inputs', exact: true });
  await expect(inputs).toContainText('learning-rate Slow · 0.001, Medium · 0.002 · 2 of 3');
  const trialSet = drawer.getByRole('group', { name: 'Trial Set', exact: true });
  await expect(trialSet.getByRole('combobox')).toBeFocused();
  await trialSet.getByRole('combobox').selectOption('conditions');
  for (const condition of conditionIds) await trialSet.getByRole('checkbox', { name: condition, exact: true }).check();
  const edited = (path: string) => page.waitForResponse((response) => {
    const request = response.request();
    return request.method() === 'PATCH' && new URL(response.url()).pathname === projectPath &&
      JSON.stringify(request.postDataJSON()).includes(path);
  });
  const trialEdit = edited('/base/inline/params/trial_set');
  await trialSet.getByRole('button', { name: 'Apply Trials', exact: true }).click();
  const savedTrials = await trialEdit;
  expect(savedTrials.ok(), await savedTrials.text()).toBe(true);
  const step = (await savedTrials.json() as StudioProjectSnapshot).project.steps.evaluate;
  expect(step.base).toMatchObject({ inline: { params: { trial_set: { kind: 'conditions', ids: conditionIds } } } });
  expect(step.base).not.toHaveProperty('inline.params.trials');

  const keys = (await readPlan()).window_map!.evaluations!.map(({ key }) => key);
  expect(keys).toHaveLength(2);
  await run(keys);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 16_000 });
  const before = (await readPlan()).window_map!.evaluations!;
  await expect(runButton(page)).toBeDisabled();

  await select(page, 'train');
  const fast = runs(page).getByRole('row').filter({ hasText: 'Fast' }).getByRole('button');
  await fast.click();
  await expect(primary(page).getByRole('region', { name: 'Used By' })).toHaveCount(0);
  await expect(steps.getByRole('button', { name: 'Select evaluate Evaluate 0 of 2', exact: true })).toBeVisible();
  const slow = runs(page).getByRole('row').filter({ hasText: 'Slow' }).getByRole('button');
  await slow.click();
  await expect(primary(page).getByRole('region', { name: 'Used By' })).toContainText('1 of 2');
  await expect(steps.getByRole('button', { name: 'Select evaluate Evaluate 1 of 2', exact: true })).toBeVisible();
  await expect(evaluated.locator('[data-map-focus="downstream"]')).toHaveCount(1);

  await select(page, 'evaluate');
  const subset = drawer.getByRole('group', { name: 'Subject Subset', exact: true });
  await subset.getByRole('checkbox', { name: 'Fast · 0.003', exact: true }).check();
  const subsetEdit = edited('/inputs/subject/where');
  await subset.getByRole('button', { name: 'Apply Subset', exact: true }).click();
  const savedSubset = await subsetEdit;
  expect(savedSubset.ok(), await savedSubset.text()).toBe(true);
  await expect(inputs).toContainText('3 of 3');
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(2);
  const widened = (await readPlan()).window_map!.evaluations!;
  expect(widened.slice(0, 2)).toEqual(before);
  expect(widened).toHaveLength(3);
  await run([widened[2].key]);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(3, { timeout: 16_000 });

  await page.reload();
  await select(page, 'evaluate');
  await expect(inputs).toContainText('3 of 3');
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(3);
  await runs(page).getByRole('row').filter({ hasText: 'Fast' }).getByRole('button').click();
  await expect(primary(page).getByRole('region', { name: 'Made From' })).toContainText('1 of 3');
  await expect(steps.getByRole('button', { name: 'Select train Train 1 of 3', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Show Map', exact: true }).click();
  await expect(train.locator('[data-map-focus="upstream"]')).toHaveCount(1);
  const final = await readPlan();
  const results = final.window_map!.evaluations!;
  expect(results.slice(0, 2)).toEqual(before);
  for (const row of results) {
    const edge = final.window_map!.edges.find(({ to_key }) => JSON.stringify(to_key) === JSON.stringify(row.key));
    const producer = final.rows.find(({ key }) => JSON.stringify(key) === JSON.stringify(edge?.from_key));
    expect(producer?.result).toBeDefined();
    expect(row.trials).toEqual(conditionIds);
    expect(row.result?.checkpoint_source).toEqual({
      job_id: producer!.result!.job_id, request_id: producer!.request_id, completed_updates: 2,
    });
    expect(row.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  }

  await page.getByRole('button', { name: 'Show Both', exact: true }).click();
  await select(page, 'train');
  await runs(page).getByRole('row').filter({ hasText: 'Fast' }).getByRole('button').click();
  const second = responseFor(page, projectPath + '/files');
  await primary(page).getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step' }).getByRole('button', { name: 'Evaluate', exact: true }).click();
  const secondCreated = await second;
  expect(secondCreated.status(), await secondCreated.text()).toBe(201);
  const { snapshot: withSecond } = await secondCreated.json() as StudioProjectDocument;
  expect(withSecond.project.steps['evaluate-2'].inputs?.subject).toMatchObject({ where: { rows: [sourceRows[2].key.row] } });
  await expect(trialSet.getByRole('combobox')).toHaveValue('every_condition');
  await select(page, 'evaluate');
  await expect(trialSet.getByRole('combobox')).toHaveValue('conditions');
  await select(page, 'evaluate-2');
  await expect(trialSet.getByRole('combobox')).toHaveValue('every_condition');
  await expect(trialSet.getByRole('button', { name: 'Apply Trials', exact: true })).toBeDisabled();
  console.info('Mounted Evaluate subset receipts', JSON.stringify({ project: snapshot.project_id, revision: final.revision, results }));
});

test('renders analysis and a report, then reruns only results affected by an edit', async ({ page }) => {
  test.fixme(true, 'M3 · 041aba9, 3c83a61, 9b6cf79, 4d1558c, 6b22472');
  await page.goto('/');
  await expect(page.getByRole('option', { name: 'Analyze', exact: true })).toBeVisible();
});

test('uses a variant and a subset across the outline, map, and inspector', async ({ page }) => {
  test.fixme(true, 'M3 · 01e4c26, 3e0ceba');
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Pipeline Map', exact: true })).toBeVisible();
});
