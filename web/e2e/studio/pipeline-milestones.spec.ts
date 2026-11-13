import { expect, test, type Page } from '@playwright/test';
import type {
  StudioActivityJob, StudioJobEventHistory, StudioJobSnapshot, StudioJobSubmission,
  StudioProjectDocument, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import {
  appendManagedFiles, authorLearningRateSweep, fixture, mapStep, openResultsSlot,
  runProjectFixture, seedManagedProject,
} from './fixtures/canonicalTrain';

const evaluationEnabled = process.env.FEEDBAX_STUDIO_EVALUATION_CANDIDATE === '1';
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

test('M2 reopens a project folder and evaluates two conditions on its training result', async ({ page, request }, info) => {
  test.fixme(!evaluationEnabled, 'M2 public gate · 736786f, dbbfb13, 339d687, 01e4c26, ed09a0a');
  test.setTimeout(50_000);
  const evaluation = 'web.e2e.studio.fixtures.canonical_evaluate';
  const authored = JSON.parse(await runProjectFixture(evaluation, 'walkthrough-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, authored.train);
  const mounted = '/?project=' + snapshot.project_id;
  await page.goto(mounted);
  await select(page, 'train');
  await admit(page, projectPath);
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 15_000 });

  await page.goto('about:blank');
  const revision = JSON.parse(await runProjectFixture(
    evaluation, 'save-walkthrough-conditions', String(info.config.metadata.home), snapshot.project_id,
  ));
  await page.goto(mounted);
  for (const step of authored.conditions.map(({ id }: { id: string }) => id)) {
    await select(page, step);
    await admit(page, projectPath);
    await expect(runs(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 15_000 });
  }
  const response = await request.get(projectPath + '/train-runs', {
    params: { revision, window_map: 'true' },
  });
  expect(response.ok(), await response.text()).toBe(true);
  const plan = await response.json() as StudioSavedTrainRuns;
  const train = plan.rows[0].result!;
  const conditions = plan.window_map!.evaluations!;
  expect(conditions).toHaveLength(2);
  expect(new Set(conditions.map((row) => row.request_id)).size).toBe(2);
  for (const condition of conditions) {
    expect(condition.result?.checkpoint_source?.job_id).toBe(train.job_id);
    expect(condition.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  }
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
