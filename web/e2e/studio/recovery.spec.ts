import { expect, test as base, type Page } from '@playwright/test';
import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { StudioJobSnapshot, StudioJobSubmission, StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, openResultsSlot, seedManagedProject } from './fixtures/canonicalTrain';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);

function processId(text: string, owner: string): number {
  const pid = Number(text.trim());
  if (!Number.isSafeInteger(pid) || pid <= 1) throw new Error(`Invalid isolated ${owner} PID`);
  return pid;
}

async function hostPid(home: string): Promise<number> {
  const { stdout } = await run('uv', ['run', '--no-sync', 'python', '-B', '-c', `
from pathlib import Path
import sys
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient

paths = AppPaths.resolve(instance_root=Path(sys.argv[1]))
print(HostClient(paths.host).ping()["pid"])
`, home], { cwd: repo, timeout: 10_000 });
  return processId(stdout, 'host');
}

function processExists(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ESRCH') return false;
    throw error;
  }
}

function lossCurve(page: Page) {
  return page.getByRole('img', { name: 'Loss', exact: true }).locator('polyline');
}

async function loopbackPorts() {
  const servers = [createServer(), createServer()];
  try {
    return await Promise.all(servers.map(async (server) => {
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const address = server.address();
      if (address === null || typeof address === 'string') throw new Error('Expected a loopback TCP port');
      return address.port;
    }));
  } finally {
    for (const server of servers) server.close();
  }
}

async function stop(child: ChildProcess) {
  if (child.exitCode === null && child.signalCode === null) {
    const exited = once(child, 'exit');
    child.kill('SIGTERM');
    await exited;
  }
  if (child.exitCode !== 0 && child.exitCode !== 143) {
    throw new Error(`Studio launcher exited with ${child.exitCode ?? child.signalCode}`);
  }
}

interface RecoveryStudio {
  origin: string;
  home: string;
  stopBackend: () => Promise<void>;
  startBackend: () => Promise<void>;
  serviceLog: () => string;
}

const test = base.extend<{ studio: RecoveryStudio }>({
  studio: async ({ request }, use, testInfo) => {
    const [webPort, backendPort] = await loopbackPorts();
    const origin = `http://127.0.0.1:${webPort}`;
    const home = join(tmpdir(), `feedbax-studio-recovery-${randomUUID()}`);
    const backendArgs = ['--home', home, '--backend-port', String(backendPort)];
    const log: string[] = [];
    const launchers: ChildProcess[] = [];
    const start = (...args: string[]) => {
      const child = spawn('./scripts/dev.sh', [...backendArgs, ...args], {
        cwd: repo,
        env: { ...process.env, FEEDBAX_STUDIO_TEST_OWNER_PID: String(process.pid) },
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      for (const stream of [child.stdout, child.stderr]) {
        stream.on('data', (chunk) => log.push(String(chunk)));
      }
      launchers.push(child);
    };
    const healthy = async () => {
      try {
        return (await request.get(`${origin}/api/health`, { timeout: 1000 })).ok();
      } catch {
        return false;
      }
    };
    const preview = process.env.FEEDBAX_STUDIO_E2E_PREVIEW === '1' ? ['--preview'] : [];
    start('--port', String(webPort), '--remove-test-home', ...preview);
    try {
      await expect.poll(healthy, { timeout: 35_000 }).toBe(true);
      await use({
        origin,
        home,
        stopBackend: async () => {
          const recorded = await readFile(join(home, 'cache', 'backend.pid'), 'utf8');
          process.kill(processId(recorded, 'backend'), 'SIGKILL');
          await expect.poll(healthy).toBe(false);
        },
        startBackend: async () => {
          start('--backend-only');
          await expect.poll(healthy, { timeout: 10_000 }).toBe(true);
        },
        serviceLog: () => log.join(''),
      });
    } finally {
      const stopped = await Promise.allSettled(launchers.map(stop));
      await testInfo.attach('Studio process log', { body: log.join(''), contentType: 'text/plain' });
      for (const result of stopped) {
        if (result.status === 'rejected') throw result.reason;
      }
    }
  },
});

test.use({
  channel: 'chrome',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('reports that the backend stopped and recovers in the same window', async ({ page, studio, request }) => {
  test.setTimeout(50_000);
  const { snapshot } = await seedManagedProject(request, await fixture('ordinary-files', '2'), studio.origin);
  await page.goto(studio.origin + '/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  const secondary = page.locator('[data-slot="secondary"]').getByRole('tabpanel');
  const runs = page.locator('[data-slot="primary"]').getByRole('tabpanel', { name: 'Runs', exact: true });
  await expect(runs.getByRole('table', { name: 'Runs', exact: true })).toBeVisible();
  await runs.getByRole('button', { name: 'Split Details', exact: true }).click();
  const tab = (name: string) => page.getByRole('tab', { name, exact: true });
  await tab('Task').click();
  const task = secondary.getByRole('combobox', { name: 'Task Document', exact: true });
  await expect(task).toBeVisible();
  const footer = page.locator('[data-studio-focus-region="bottom"]');
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const openActivity = footer.getByRole('button', { name: 'Activity', exact: true });
  await openActivity.click();
  await expect(activity.getByRole('list', { name: 'Runs', exact: true })).toBeVisible();
  await studio.stopBackend();
  const failure = footer.getByRole('alert');
  const unreachable = (region: ReturnType<typeof page.locator>) =>
    expect(region.getByRole('alert').filter({ hasText: 'Service unreachable' })).toBeVisible();
  await unreachable(activity);
  await activity.getByRole('button', { name: 'Close', exact: true }).click();
  await unreachable(footer);
  await unreachable(runs);
  await unreachable(secondary);
  await unreachable(page.locator('[data-studio-focus-region="map"]'));
  await unreachable(page.locator('[data-studio-focus-region="steps"]'));
  await expect(task).toBeHidden();
  await expect(runs.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
  await runs.focus();
  await unreachable(page.getByRole('complementary', { name: 'Details Drawer', exact: true }));
  await unreachable(page.getByRole('complementary', { name: 'Inspector', exact: true }));
  // Visit the remaining data-backed M2 views in the second slot during the same outage.
  await secondary.focus();
  for (const view of ['Model', 'Objective', 'Scene', 'Results', 'Training']) {
    await tab(view).click();
    await expect(secondary).toHaveAccessibleName(view);
    await unreachable(secondary);
  }
  const retry = footer.getByRole('button', { name: 'Retry', exact: true });
  await expect(retry).toBeEnabled();
  const batchCount = secondary.getByRole('textbox', { name: 'N Batches', exact: true });
  await studio.startBackend();
  await expect(batchCount).toBeEnabled();
  await tab('Task').click();
  await expect(task).toBeVisible();
  await expect(failure).toHaveCount(0);
  await expect(runs.getByRole('table', { name: 'Runs', exact: true })).toBeVisible();
  await expect(runs.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
  await expect(mapStep(page, 'train')).toBeVisible();
  await expect(page.locator('[data-studio-focus-region="steps"]').getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('complementary', { name: 'Details Drawer', exact: true }).getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('alert')).toHaveCount(0);
  await openActivity.click();
  await expect(activity.getByRole('list', { name: 'Runs', exact: true })).toBeVisible();
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
            target.matches('[data-slot="secondary"] .studio-model-canvas .react-flow__renderer')) {
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
  await runs.getByRole('button', { name: 'Split Details', exact: true }).click();
  await page.evaluate(() => { document.documentElement.dataset.failModelObserve = 'true'; });
  const report = page.waitForResponse((response) => new URL(response.url()).pathname === '/api/studio/render-failure'
    && response.request().postDataJSON()?.detail === message);
  void report.catch(() => {});
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model.getByRole('alert')).toHaveText('Model failed to render');
  expect((await report).status()).toBe(204);
  await expect.poll(studio.serviceLog).toContain(`Studio render failure region='model' type='Error' detail='${message}'`);
  await page.evaluate(() => { delete document.documentElement.dataset.failModelObserve; });

  await runs.focus();
  await expect(inspector.getByRole('heading')).toBeVisible();
  await expect(mapStep(page, 'train')).toBeVisible();
  await expect(table).toBeVisible();
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  await expect(activity.getByRole('list', { name: 'Runs', exact: true })).toBeVisible();
  await activity.getByRole('button', { name: 'Close', exact: true }).click();

  const failedReports = () => studio.serviceLog().split(`detail='${message}'`).length - 1;
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

test('reopens the same live run and restores loss after backend and host restart', async ({
  browser, page, request, studio,
}) => {
  test.setTimeout(50_000);
  const batches = 2400;
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
  await expect(liveLoss).toHaveAttribute('points', /\S+ \S+/);
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
      .getByRole('button', { name: row.label, exact: true }).click();
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
