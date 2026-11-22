import { expect, firstJobTimeout } from './fixtures/studio';
import { type Page } from '@playwright/test';
import type { StudioJobSnapshot, StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, openResultsSlot, runName, seedManagedProject } from './fixtures/canonicalTrain';
import { isolatedTest as test } from './fixtures/studio';
import { hostPid, processExists } from './fixtures/studioRecovery';

function lossCurve(page: Page) {
  return page.getByRole('img', { name: 'Metric train_loss', exact: true }).locator('polyline');
}

test.use({ viewport: { width: 1440, height: 900 } });

const unreachable = 'Service unreachable';

test('shows a lost service once in the bottom bar and reconnects by itself', async ({ page, studio, request }) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'), studio.origin);
  await page.goto(studio.origin + '/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  const runs = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Runs', exact: true });
  const table = runs.getByRole('table', { name: 'Runs', exact: true });
  await expect(table).toBeVisible();
  await runs.getByRole('button', { name: 'Split Details', exact: true }).click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.locator('[data-slot="secondary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  const modelNode = model.locator('.react-flow__node').first();
  await expect(modelNode).toBeVisible();
  const footer = page.locator('[data-studio-focus-region="bottom"]');
  const run = runs.getByRole('button', { name: 'Run', exact: true });
  await expect(run).toBeEnabled();

  await studio.stopBackend();
  await expect(footer.getByRole('alert')).toHaveText(new RegExp(unreachable));
  await expect(page.getByText(unreachable)).toHaveCount(1);
  await expect(footer.getByRole('button', { name: 'Retry', exact: true })).toBeEnabled();
  await expect(table).toBeVisible();
  await expect(modelNode).toBeVisible();
  await expect(mapStep(page, 'train')).toBeVisible();
  await expect(run).toBeDisabled();
  for (const region of [runs, model, page.locator('[data-studio-focus-region="map"]')]) {
    await expect(region.getByRole('alert')).toHaveCount(0);
  }

  const refetched = page.waitForResponse((response) =>
    new URL(response.url()).pathname === projectPath + '/train-runs' && response.request().method() === 'GET', { timeout: 20_000 });
  await studio.startBackend();
  await expect(footer.getByRole('alert')).toHaveCount(0, { timeout: 6_000 });
  expect((await refetched).ok()).toBe(true);
  await expect(run).toBeEnabled();
  await expect(table).toBeVisible();
  await expect(modelNode).toBeVisible();
});

test('a page loaded without the service shows its regions loading and the reason in the bottom bar', async ({
  page, studio, request,
}) => {
  test.setTimeout(50_000);
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'), studio.origin);
  await studio.stopBackend();
  await page.goto(studio.origin + '/?project=' + snapshot.project_id);
  const footer = page.locator('[data-studio-focus-region="bottom"]');
  await expect(footer.getByRole('alert')).toHaveText(new RegExp(unreachable));
  await expect(page.getByText(unreachable)).toHaveCount(1);
  await expect(page.locator('[data-studio-focus-region="map"] [aria-busy="true"]')).toBeVisible();

  await studio.startBackend();
  await expect(mapStep(page, 'train')).toBeVisible({ timeout: 8_000 });
  await expect(footer.getByRole('alert')).toHaveCount(0);
});

// Catches a test whose page carries the machine's live Mandible toolbar, or a test build stamped for a walkthrough.
test('a test server serves its pages without the notes toolbar or source stamps', async ({ page, studio, request }) => {
  expect(await (await request.get(studio.origin + '/')).text()).not.toContain('/embed/notes.js');
  await page.goto(studio.origin + '/');
  await expect(page.locator('#root > *').first()).toBeAttached();
  await expect(page.locator('[data-source]')).toHaveCount(0);
});

test('an endpoint answering with an error fails its own region only', async ({ page, studio, request }) => {
  test.setTimeout(50_000);
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'), studio.origin);
  await page.route((url) => url.pathname.endsWith('/model/projection'), (route) =>
    route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"projection failed"}' }));
  await page.goto(studio.origin + '/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  const runs = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Runs', exact: true });
  await expect(runs.getByRole('table', { name: 'Runs', exact: true })).toBeVisible();
  await runs.getByRole('button', { name: 'Split Details', exact: true }).click();
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.locator('[data-slot="secondary"]').getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.getByRole('alert')).toHaveText('Model unavailable');
  await expect(model.getByRole('button', { name: 'Retry', exact: true })).toBeVisible();
  await expect(page.locator('[data-studio-focus-region="bottom"]').getByRole('alert')).toHaveCount(0);
  await expect(runs.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('alert')).toHaveCount(1);
});

test('a mounted Model render failure reports to the service and retries only its slot', async ({
  page, studio, request,
}) => {
  test.setTimeout(50_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'), studio.origin);
  await hostPid(studio.home);
  const message = 'Model canvas observer failed';
  await page.addInitScript((detail) => {
    const NativeObserver = window.ResizeObserver;
    window.ResizeObserver = class extends NativeObserver {
      observe(target: Element, options?: ResizeObserverOptions) {
        if (document.documentElement.dataset.failModelObserve === 'true' &&
            target.matches('[data-slot="secondary"] .studio-model-surface .react-flow__renderer')) {
          throw new Error(detail);
        }
        super.observe(target, options);
      }
    };
  }, message);
  await page.goto(studio.origin + '/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const table = runs.getByRole('table', { name: 'Runs', exact: true });
  const row = table.getByRole('row').filter({ has: page.getByRole('button') });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  await expect(table).toBeVisible();
  await row.getByRole('button').click();
  await expect(inspector.getByRole('heading')).toBeVisible();
  await page.evaluate(() => { document.documentElement.dataset.failModelObserve = 'true'; });
  const report = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/studio/errors'
    && response.request().postDataJSON()?.message === message);
  void report.catch(() => {});
  // Scene has nothing to show for this Model, so the split opens Model, the first other view with content.
  await runs.getByRole('button', { name: 'Split Details', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.getByRole('alert')).toHaveText('Model failed to render');
  expect((await report).status()).toBe(204);
  const ref = await model.getByRole('alert').getAttribute('data-note-ref');
  expect(ref).toMatch(/^[0-9a-f]{8}$/);
  await expect.poll(studio.serviceLog).toMatch(new RegExp(`Studio error ${ref} .*"kind":"render".*"region":"model"`));
  await page.evaluate(() => { delete document.documentElement.dataset.failModelObserve; });

  await runs.focus();
  await expect(inspector.getByRole('heading')).toBeVisible();
  await expect(mapStep(page, 'train')).toBeVisible();
  await expect(table).toBeVisible();
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await expect(activity.getByRole('list', { name: 'Runs', exact: true })).toBeVisible();
  await activity.getByRole('button', { name: 'Close', exact: true }).click();

  const failedReports = () => studio.serviceLog().split(`"message":"${message}"`).length - 1;
  expect(failedReports()).toBe(1);
  await model.getByRole('button', { name: 'Retry', exact: true }).press('Enter');
  await expect(model).toBeFocused();
  await expect(model.getByRole('alert')).toHaveCount(0);
  await expect(model.locator('.react-flow__node').first()).toBeVisible();
  await expect(table).toBeVisible();
  await expect(mapStep(page, 'train')).toBeVisible();
  await runs.focus();
  await expect(inspector.getByRole('heading')).toBeVisible();
  const saved = await request.get(studio.origin + projectPath);
  expect(saved.ok()).toBe(true);
  const current = await saved.json() as StudioProjectSnapshot;
  expect(current).toMatchObject({ revision: snapshot.revision, presentation_revision: snapshot.presentation_revision });
  expect(current.project).toEqual(snapshot.project);
  expect(failedReports()).toBe(1);
});

test('reopens the same live run and restores loss after backend and host restart', { tag: '@round' }, async ({
  browser, page, request, studio,
}) => {
  test.setTimeout(50_000);
  const batches = 6000;
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', String(batches)), studio.origin);
  const mounted = '/?project=' + snapshot.project_id;
  const snapshots = new URL('/api/jobs', studio.origin);
  snapshots.searchParams.set('project', snapshot.project_id);
  const jobs = async () => {
    const response = await request.get(snapshots.href);
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioJobSnapshot).jobs;
  };

  await page.goto(studio.origin + mounted);
  await mapStep(page, 'train').click();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const submission = page.waitForResponse((response) =>
    new URL(response.url()).pathname === projectPath + '/train-runs' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const admission = await submission;
  expect(admission.status()).toBe(202);
  const { items } = await admission.json() as StudioJobSubmission;
  expect(items).toHaveLength(1);
  const jobId = items[0].job_id;
  if (!jobId) throw new Error('Studio did not admit the reaching run');
  const job = async () => (await jobs()).find((row) => row.job_id === jobId);

  await page.getByRole('table', { name: 'Runs', exact: true }).getByRole('button').first().click();
  await openResultsSlot(page);
  const liveLoss = lossCurve(page);
  await expect(liveLoss).toHaveAttribute('points', /\S+ \S+/, { timeout: firstJobTimeout });
  const retainedPoints = (await liveLoss.getAttribute('points'))!.split(' ').length;
  const live = await job();
  if (!live) throw new Error('The admitted reaching run is missing');
  expect(live.status).toBe('running');
  const originalHost = await hostPid(studio.home);

  await studio.stopBackend();
  await studio.startBackend();
  expect((await job())?.status).toBe('running');
  expect(await hostPid(studio.home)).toBe(originalHost);

  process.kill(originalHost, 'SIGKILL');
  await expect.poll(() => processExists(originalHost)).toBe(false);
  expect(await hostPid(studio.home)).not.toBe(originalHost);

  await page.close();
  const reopened = await browser.newPage({ baseURL: studio.origin, serviceWorkers: 'block' });
  try {
    await reopened.goto(mounted);
    await mapStep(reopened, 'train').click();
    await reopened.getByRole('tab', { name: 'Runs', exact: true }).click();
    const savedRuns = await request.get(studio.origin + projectPath + '/train-runs', {
      params: { revision: snapshot.revision },
    });
    expect(savedRuns.ok(), await savedRuns.text()).toBe(true);
    const plan = await savedRuns.json() as StudioSavedTrainRuns;
    const row = plan.rows.find(({ request_id, attempts }) =>
      request_id === live.request_id && attempts.some((attempt) => attempt.job_id === jobId),
    );
    if (!row) throw new Error('The reopened Plan does not contain the admitted Train attempt');
    await reopened.locator('[data-slot="primary"]').getByRole('table', { name: 'Runs', exact: true })
      .getByRole('button', { name: runName(row.key.step) }).click();
    await openResultsSlot(reopened);
    const restored = lossCurve(reopened);
    await expect(restored).toBeVisible();
    await expect.poll(async () =>
      (await restored.getAttribute('points'))?.split(' ').length ?? 0,
    ).toBeGreaterThanOrEqual(retainedPoints);
    await expect.poll(async () => (await job())?.status, { timeout: 25_000 }).toBe('succeeded');
    const finished = await jobs();
    expect(finished).toHaveLength(1);
    expect(finished[0]).toMatchObject({
      job_id: jobId,
      request_id: live.request_id,
      run_key: live.run_key,
      progress: { completed_updates: batches, total_updates: batches },
    });
  } finally {
    await reopened.close();
  }
});
