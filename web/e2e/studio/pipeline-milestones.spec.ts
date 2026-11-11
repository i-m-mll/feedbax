import { expect, test, type Page } from '@playwright/test';
import type {
  StudioActivityJob,
  StudioJobSnapshot,
  StudioJobSubmission,
} from '../../src/generated/studioContracts';
import { openSavedProject } from './projectControls';
import { reachingDocument, saveFixture } from './reachingProject';

function responseFor(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path,
  );
}

function lossMarks(page: Page) {
  return page.getByRole('region', { name: 'Run Inspector' })
    .getByRole('img', { name: 'Loss' })
    .locator('polyline, circle');
}

test('M1 starts an ordinary run, watches loss, and resumes the checkpoint from Activity',
  async ({ page, request }) => {
    const project = await saveFixture(request, await reachingDocument(1_200));
    await page.goto('/');
    await openSavedProject(page, project.metadata.name);
    await page.getByRole('button', { name: 'Train', exact: true }).click();

    const admitted = responseFor(page, '/api/submissions');
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const admission = await admitted;
    expect(admission.ok(), await admission.text()).toBe(true);
    const { items } = await admission.json() as StudioJobSubmission;
    expect(items).toHaveLength(1);
    expect(items[0].disposition).toBe('admitted');
    const stoppedJobId = items[0].job_id!;

    const activityButton = page.getByRole('button', { name: 'Activity', exact: true });
    const activity = page.getByRole('region', { name: 'Activity', exact: true });
    const activityRun = activity.getByRole('list', { name: 'Runs' }).getByRole('listitem').first();
    await activityButton.click();
    await activityRun.getByRole('button').first().click();
    await expect(lossMarks(page).first()).toBeVisible({ timeout: 12_000 });

    await activityButton.click();
    const stopping = responseFor(page, '/api/jobs/stop');
    await activityRun.getByRole('button', { name: 'Stop', exact: true }).click();
    expect((await stopping).ok()).toBe(true);
    const resume = activityRun.getByRole('button', { name: 'Resume', exact: true });
    await expect(resume).toBeEnabled({ timeout: 8_000 });
    const stopped = await request.get(`/api/jobs?project=${project.id}`);
    expect(stopped.ok()).toBe(true);
    const snapshot = await stopped.json() as StudioJobSnapshot;
    const parent = snapshot.jobs.find((job) => job.job_id === stoppedJobId);
    expect(parent).toMatchObject({ status: 'stopped', can_resume: true });

    const resumed = responseFor(page, `/api/jobs/${stoppedJobId}/resume`);
    await resume.click();
    const continuation = await resumed;
    expect(continuation.ok(), await continuation.text()).toBe(true);
    const attempt = await continuation.json() as StudioActivityJob;
    expect(attempt).toMatchObject({
      continues: stoppedJobId,
      request_id: parent?.request_id,
      run_key: items[0].run_key,
    });
    expect(attempt.job_id).not.toBe(stoppedJobId);

    await page.reload();
    const runs = page.getByRole('table', { name: 'Runs', exact: true });
    await runs.getByRole('button', { name: 'Train', exact: true }).click();
    await expect(lossMarks(page).first()).toBeVisible({ timeout: 10_000 });
    await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 15_000 });
  },
);

test('M1 runs only the selected row of a sweep authored in Studio', async ({ page, request }) => {
  const project = await saveFixture(request, await reachingDocument());
  await page.goto('/');
  await openSavedProject(page, project.metadata.name);
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  const axis = page.getByRole('form', { name: 'Axis' });
  await axis.getByLabel('Name', { exact: true }).fill('Learning rate');
  await axis.getByLabel('Path', { exact: true }).fill('training_spec.optimizer.params.learning_rate');
  await axis.getByLabel('Values', { exact: true }).fill('0.001, 0.0005');
  await axis.getByRole('button', { name: 'Add', exact: true }).click();

  const runs = page.getByRole('table', { name: 'Runs', exact: true });
  const selected = runs.getByRole('checkbox', { name: 'Select Learning rate=0.0005', exact: true });
  await expect(runs.getByRole('checkbox', { name: /^Select Learning rate=/ })).toHaveCount(2);
  await selected.check();
  const admitted = responseFor(page, '/api/submissions');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const admission = await admitted;
  expect(admission.ok(), await admission.text()).toBe(true);
  const submission = await admission.json() as StudioJobSubmission;
  expect(submission.items).toHaveLength(1);
  expect(submission.items[0].disposition).toBe('admitted');
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 12_000 });
  await expect(runs.getByText('Not run', { exact: true })).toBeVisible();
  const response = await request.get(`/api/jobs?project=${project.id}`);
  expect(response.ok()).toBe(true);
  const { jobs } = await response.json() as StudioJobSnapshot;
  expect(jobs).toHaveLength(1);
  expect(jobs[0]).toMatchObject({ run_key: submission.items[0].run_key, label: 'Learning rate=0.0005' });
});

test('reopens a project folder and evaluates two conditions on the training result', async ({ page }) => {
  test.fixme(true, 'M2 · 736786f, dbbfb13, 339d687, 01e4c26, ed09a0a');
  await page.goto('/');
  // Quit and reopen the project folder. Add an Evaluate step over the run with two conditions.
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await expect(page.getByRole('option', { name: 'Evaluate', exact: true })).toBeVisible();
});

test('renders analysis and a report, then reruns only results affected by an edit', async ({ page }) => {
  test.fixme(true, 'M3 · 041aba9, 3c83a61, 9b6cf79, 4d1558c, 6b22472');
  await page.goto('/');
  // Add two operations and one figure, render a report, and rerun the project.
  // Edit a parameter and check only its operation, figure and report are stale; undo restores Done.
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await expect(page.getByRole('option', { name: 'Analyze', exact: true })).toBeVisible();
});

test('uses a variant and a subset across the outline, map, and inspector', async ({ page }) => {
  test.fixme(true, 'M3 · 01e4c26, 3e0ceba');
  await page.goto('/');
  // Edit a variant without affecting base results. Sweep three rates and evaluate two selected runs.
  // Expand the subset to three and check one new run, reuse of the other two, and visible lineage.
  await expect(page.getByRole('button', { name: 'Pipeline Map', exact: true })).toBeVisible();
});
