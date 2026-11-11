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
import type { StudioJobSnapshot, StudioJobSubmission } from '../../src/generated/studioContracts';
import { reachingDocument, saveFixture } from './reachingProject';
import { openOverlay, openSavedProject } from './projectControls';

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
        return (await request.get(`${origin}/api/provider/health`, { timeout: 1000 })).ok();
      } catch {
        return false;
      }
    };
    start('--port', String(webPort), '--remove-test-home');
    try {
      await expect.poll(healthy, { timeout: 15_000 }).toBe(true);
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

test('reports that the backend stopped and recovers in the same window', async ({ page, studio }) => {
  await page.goto(studio.origin);
  const connection = page.getByRole('status', { name: 'Studio connection' });
  await expect(connection).toHaveText(/^connected$/i);
  await studio.stopBackend();
  await expect(connection).toHaveText(/service unreachable/i);

  await page.getByTitle('Open project', { exact: true }).click();
  const projects = openOverlay(page);
  const runs = page.getByRole('region', { name: 'Runs', exact: true });
  const projectFailure = projects.getByRole('alert').filter({ hasText: /service unreachable/i });
  const runFailure = runs.getByRole('alert').filter({ hasText: /service unreachable/i });
  await expect(projectFailure).toBeVisible();
  await expect(runFailure).toBeVisible();
  const retry = projects.getByRole('button', { name: 'Retry', exact: true });
  await expect(retry).toBeEnabled();

  await studio.startBackend();
  await retry.click();
  await expect(projectFailure).toHaveCount(0);
  await expect(runFailure).toHaveCount(0);
  await expect(connection).toHaveText(/^connected$/i);
});

test('reopens the same live run and restores loss after backend and host restart', async ({
  browser, page, request, studio,
}) => {
  test.setTimeout(50_000);
  const name = 'Reaching across restart';
  const batches = 2400;
  const document = await reachingDocument(batches, name);
  const project = await saveFixture(request, document, `${studio.origin}/api/graphs`);
  const snapshots = new URL('/api/jobs', studio.origin);
  snapshots.searchParams.set('project', project.id);
  const jobs = async () => {
    const response = await request.get(snapshots.href);
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioJobSnapshot).jobs;
  };

  await page.goto(studio.origin);
  await openSavedProject(page, name);
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  const submission = page.waitForResponse((response) =>
    new URL(response.url()).pathname === '/api/submissions' && response.request().method() === 'POST');
  await page.getByRole('region', { name: 'Runs', exact: true }).getByRole('button', { name: 'Run', exact: true }).click();
  const admission = await submission;
  expect(admission.status()).toBe(202);
  const { items } = await admission.json() as StudioJobSubmission;
  expect(items).toHaveLength(1);
  const jobId = items[0].job_id;
  if (!jobId) throw new Error('Studio did not admit the reaching run');
  const job = async () => (await jobs()).find((row) => row.job_id === jobId);

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
    await reopened.goto('/');
    await openSavedProject(reopened, name);
    const row = reopened.getByRole('table', { name: 'Runs', exact: true });
    await row.getByRole('button', { name: live.label, exact: true }).click();
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
