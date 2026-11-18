import { expect, type APIRequestContext, type Page, type TestInfo } from '@playwright/test';
import type { StudioActivityJob, StudioJobSnapshot, StudioJobSubmission } from '../../../src/generated/studioContracts';
import { mapStep, runName, runProjectFixture } from './canonicalTrain';
import { authorFirstTrain } from './firstTrainAuthoring';
import { activate, palette, post, tabTo } from './studioKeyboard';
import { hostPid, processExists } from './studioRecovery';
import type { Studio } from './studio';

export async function authorAndRecoverTrain(page: Page, request: APIRequestContext, info: TestInfo, studio: Studio) {
  const totalUpdates = 6000;
  const { home } = studio;
  // Softmax keeps the total force at one along the diagonal reach, regardless of training.
  const authored = await authorFirstTrain(page, request, totalUpdates, { targetY: 0.22, activation: 'softmax' });
  const { saved, projectPath, plan, savedProject, plantId } = authored;
  const project = saved.project_id;
  const witness = { module: 'web.e2e.studio.fixtures.managed_lifecycle', home };
  const observe = async (action: string, ...args: string[]) =>
    JSON.parse(await runProjectFixture(witness, action, home, project, ...args));
  const declared = await observe('declare-variants', saved.revision);
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Variant', exact: true })).toHaveText('base');
  const current = await savedProject();
  expect(current.revision).toBe(declared.revision);
  expect(current.project.documents).toEqual(saved.project.documents);
  expect(current.project.steps).toEqual(saved.project.steps);

  const settled = await plan(current.revision);
  const row = settled.rows.find(({ key }) => key.variant === 'base' && key.step === 'train')!;
  expect(row).toMatchObject({ key: { row: 'default' }, status: 'not_run', attempts: [] });
  expect(settled.rows.map(({ request_id }) => request_id)).toEqual([row.request_id, row.request_id]);
  const jobs = async () => {
    const response = await request.get('/api/jobs', { params: { project } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioJobSnapshot).jobs;
  };
  const job = async (id: string) => (await jobs()).find(({ job_id }) => job_id === id);
  const runs = page.getByRole('tabpanel', { name: 'Runs', exact: true });
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  const loss = results.getByRole('img', { name: 'Loss', exact: true });
  const line = (id: string) => loss.locator(`line[data-attempt="${id}"] title`);
  const selectRun = async () => {
    await activate(page, mapStep(page, 'train'));
    await page.keyboard.press('Meta+1');
    await activate(page, runs.getByRole('table', { name: 'Runs', exact: true }).getByRole('button', { name: runName(row.label) }));
  };
  const openRun = async () => {
    await expect(runs.getByRole('button', { name: 'Open Run', exact: true })).toBeEnabled();
    await palette(page, 'openRun', 'Open Run');
  };

  await selectRun();
  const submitting = post(page, projectPath + '/train-runs');
  await palette(page, 'run', 'Run');
  const admission = await submitting;
  expect(admission.status(), await admission.text()).toBe(202);
  expect(admission.request().postDataJSON()).toMatchObject({ revision: current.revision, run_keys: [row.key] });
  const { items } = await admission.json() as StudioJobSubmission;
  expect(items).toMatchObject([{ disposition: 'admitted', job_id: expect.any(String) }]);
  const initial = items[0].job_id!;
  await openRun();
  await expect(line(initial).first()).toBeAttached({ timeout: 16_000 });
  expect(await job(initial)).toMatchObject({ request_id: row.request_id, status: 'running' });
  const stopping = post(page, '/api/jobs/stop');
  await page.keyboard.press('Meta+Period');
  const stop = await stopping;
  expect(stop.status(), await stop.text()).toBe(202);
  expect(stop.request().postDataJSON()).toEqual({ job_ids: [initial], mode: 'checkpoint' });
  await expect.poll(async () => (await job(initial))?.status, { timeout: 12_000 }).toBe('stopped');
  const checkpoint = await observe('checkpoint', initial);
  expect(checkpoint.completed_updates).toBeGreaterThan(0);
  expect(checkpoint.completed_updates).toBeLessThan(totalUpdates);
  const previousLossCount = await line(initial).count();
  expect(previousLossCount).toBeGreaterThan(0);

  const activityButton = page.getByRole('button', { name: 'Activity', exact: true });
  await activate(page, activityButton);
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const resuming = post(page, `/api/jobs/${initial}/resume`);
  await activate(page, activity.getByRole('button', { name: 'Resume', exact: true }));
  const resume = await resuming;
  expect(resume.status(), await resume.text()).toBe(202);
  const continued = await resume.json() as StudioActivityJob;
  expect(continued).toMatchObject({ project_id: project, request_id: row.request_id, continues: initial, run_key: items[0].run_key });
  expect(continued.job_id).not.toBe(initial);
  const resumed = continued.job_id;
  const binding = await observe('resume', initial, resumed);
  expect(binding.checkpoint).toEqual(checkpoint);
  await activate(page, activity.getByRole('button', { name: 'Close', exact: true }));
  await expect.poll(async () => (await job(resumed))?.status, { timeout: 14_000 }).toBe('running');

  await studio.stopBackend();
  const footer = page.locator('[data-studio-focus-region="bottom"]');
  await expect(footer.getByRole('alert').filter({ hasText: 'Service unreachable' })).toBeVisible();
  await expect(page.getByRole('alert').filter({ hasText: 'Service unreachable' })).toHaveCount(1);
  const retry = footer.getByRole('alert').getByRole('button', { name: 'Retry', exact: true });
  await Promise.all([
    page.waitForEvent('requestfailed', (failed) => new URL(failed.url()).pathname === '/api/health'),
    activate(page, retry),
  ]);
  await tabTo(page, retry);
  await studio.startBackend();
  // A live event stream can reconnect and remove the Retry control as soon as the backend is up.
  const retriedAfterRestart = await retry.evaluateAll(([button]) => button === document.activeElement);
  if (retriedAfterRestart) {
    await page.keyboard.press('Enter');
  }
  await expect(footer.getByRole('alert').filter({ hasText: 'Service unreachable' })).toHaveCount(0);
  expect(await hostPid(home)).toBe(binding.host);
  expect(await job(resumed)).toMatchObject({ status: 'running', request_id: row.request_id });

  process.kill(binding.host, 'SIGKILL');
  await expect.poll(() => processExists(binding.host)).toBe(false);
  const replacementHost = await hostPid(home);
  expect(replacementHost).not.toBe(binding.host);
  await page.reload();
  await selectRun();
  await openRun();
  await expect(footer.getByRole('alert')).toHaveCount(0);
  await expect(line(initial).first()).toBeAttached();
  await expect.poll(() => line(initial).count()).toBeGreaterThanOrEqual(previousLossCount);
  await expect(line(resumed).first()).toBeAttached({ timeout: 16_000 });
  await expect.poll(async () => (await job(resumed))?.status, { timeout: 25_000 }).toBe('succeeded');
  const final = (await plan(current.revision)).rows.find(({ key }) => key.variant === 'base' && key.step === 'train')!;
  expect(final.attempts.map(({ job_id }) => job_id)).toEqual([initial, resumed]);
  expect(final.result).toMatchObject({
    job_id: resumed, request_id: row.request_id,
    progress: { completed_updates: totalUpdates, total_updates: totalUpdates },
    receipt: await observe('completed', initial, resumed),
  });
  expect(await savedProject()).toMatchObject({ project_id: project, revision: current.revision, project: current.project });
  await info.attach('first Train checkpoint and restart', {
    body: JSON.stringify({
      initial, resumed, checkpoint, oldHost: binding.host, replacementHost, retriedAfterRestart, result: final.result,
    }),
    contentType: 'application/json',
  });
  return { home, project, projectPath, plantId, stopped: initial, trained: final.result! };
}
